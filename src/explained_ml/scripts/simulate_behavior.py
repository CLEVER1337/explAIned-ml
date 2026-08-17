"""Generate behavioural events with a signal a model is supposed to recover.

    python -m explained_ml.scripts.simulate_behavior --events-per-user 200 --days 21

Why not `monitoring/load.sh`: that script is a Prometheus warm-up. One user, uniformly random
interactions, no preference structure. ALS trained on uniform-random interactions recovers
nothing, and NDCG over it is noise around chance — you cannot tell a working ranker from a
broken one. Here every user gets two hidden favourite topics and interacts mostly inside them,
so "did the model find the preference" is a question with an answer.

Two write paths:

  default            POST /articles/{id}/click|read|like|dislike|share with the user's token.
                     Correct end to end — Kafka, the consumer, ClickHouse, `rec:user_viewed`.
  --direct-clickhouse Insert straight into explained.user_events and write the viewed sets
                     ourselves. For when Kafka or the consumer is not running; events are
                     tagged `source='seed-script'` so they stay identifiable forever.

Only the direct path can backdate: the HTTP endpoints stamp `occurredAt` at emit time, so the
time-decay terms in trending and user embeddings have nothing to bite on without `--days`.
"""

import argparse
import asyncio
import json
import logging
import random
import sys
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx

from ..articles import Article, ArticleClient, ArticleServiceError
from ..clickhouse import ClickHouseClient, ClickHouseError
from ..config import get_settings
from ..identity import IdentityClient, IdentityError
from ..logs import EXIT_INFRA, EXIT_NOTHING_TO_DO, EXIT_OK, configure
from ..redis_store import VIEWED_PREFIX
from .corpus import TOPICS
from .seed_dev_corpus import DEFAULT_PASSWORD, USERS_FILE

logger = logging.getLogger("explained_ml.simulate_behavior")

PREFERRED_SHARE = 0.7
VIEWED_TTL_DAYS = 30

# The longest offset any follow-up event adds to its click, plus slack.
FUNNEL_SPAN = timedelta(seconds=120)

# Tag -> topic, inverted from the corpus spec so an article can be classified from what the
# article service actually returns.
TAG_TO_TOPIC = {tag.lower(): topic for topic, spec in TOPICS.items() for tag in spec["tags"]}


@dataclass(frozen=True, slots=True)
class Event:
    event_id: str
    event_type: str
    user_id: str
    article_id: str
    occurred_at: datetime
    metadata: dict[str, str]


def topic_of(article: Article) -> str | None:
    for tag in article.tag_list:
        if tag in TAG_TO_TOPIC:
            return TAG_TO_TOPIC[tag]

    return None


def plan_events(
    user_id: str,
    preferred: list[str],
    by_topic: dict[str, list[Article]],
    count: int,
    days: int,
    rng: random.Random,
    now: datetime,
) -> list[Event]:
    """One click always, then a funnel. Preference shows up as *which* articles get opened."""
    other_topics = [t for t in by_topic if t not in preferred]
    events: list[Event] = []

    while len(events) < count:
        prefer = rng.random() < PREFERRED_SHARE and preferred
        pool_topics = preferred if prefer else (other_topics or preferred)
        topic = rng.choice(pool_topics)
        pool = by_topic.get(topic) or []
        if not pool:
            continue

        article = rng.choice(pool)
        occurred = _timestamp(rng, days, now)

        events.append(_event("ArticleClicked", user_id, article.id, occurred))

        if rng.random() < 0.6:
            events.append(_event("ArticleRead", user_id, article.id, occurred + timedelta(seconds=40)))

            if prefer and rng.random() < 0.35:
                events.append(
                    _event("ArticleLiked", user_id, article.id, occurred + timedelta(seconds=70))
                )
            if prefer and rng.random() < 0.08:
                events.append(
                    _event(
                        "ArticleShared",
                        user_id,
                        article.id,
                        occurred + timedelta(seconds=90),
                        {"channel": rng.choice(["telegram", "link"])},
                    )
                )
        elif not prefer and rng.random() < 0.15:
            events.append(
                _event("ArticleDisliked", user_id, article.id, occurred + timedelta(seconds=20))
            )

    return events[:count]


def _event(
    event_type: str,
    user_id: str,
    article_id: str,
    occurred_at: datetime,
    metadata: dict[str, str] | None = None,
) -> Event:
    return Event(
        event_id=str(uuid.uuid4()),
        event_type=event_type,
        user_id=user_id,
        article_id=article_id,
        occurred_at=occurred_at,
        metadata=metadata or {},
    )


def _timestamp(rng: random.Random, days: int, now: datetime) -> datetime:
    # Recent days are weighted higher (squared uniform) so decay terms see a realistic slope
    # instead of a flat block of history.
    fraction = rng.random() ** 2
    hour = rng.choices(range(24), weights=[1 + 3 * (8 <= h <= 23) for h in range(24)])[0]

    stamp = now - timedelta(days=fraction * days)
    stamp = stamp.replace(hour=hour, minute=rng.randrange(60), second=rng.randrange(60))

    # Substituting the hour can push a timestamp past `now`. Every aggregate filters on
    # `occurred_at < as_of`, so a future event is simply invisible — roughly half of the most
    # recent day would silently vanish from training and trending. The margin covers the whole
    # funnel, whose later steps are offset from this click.
    return min(stamp, now - FUNNEL_SPAN)


async def simulate(
    events_per_user: int,
    days: int,
    seed_value: int,
    direct: bool,
    users_file: Path,
    settings,
) -> int:
    articles_client = ArticleClient(settings.articles_base_url)

    try:
        catalog = [a async for a in articles_client.iter_published()]
    except ArticleServiceError as exc:
        logger.error("could not read the catalog: %s", exc)
        return EXIT_INFRA
    finally:
        await articles_client.close()

    by_topic: dict[str, list[Article]] = {}
    for article in catalog:
        topic = topic_of(article)
        if topic:
            by_topic.setdefault(topic, []).append(article)

    if not by_topic:
        logger.error(
            "no published article carries a known topic tag (%d articles seen) — run "
            "seed_dev_corpus first",
            len(catalog),
        )
        return EXIT_NOTHING_TO_DO

    logger.info("catalog: %d articles across %d topics", len(catalog), len(by_topic))

    users = _load_users(users_file)
    if not users:
        logger.error("no %s — run seed_dev_corpus first", users_file)
        return EXIT_NOTHING_TO_DO

    rng = random.Random(seed_value)
    topics = sorted(by_topic)
    now = datetime.now(UTC)

    planned: list[Event] = []
    for index, user in enumerate(users):
        preferred = [topics[index % len(topics)], topics[(index + 3) % len(topics)]]
        planned.extend(
            plan_events(user["user_id"], preferred, by_topic, events_per_user, days, rng, now)
        )
        logger.info("user %s prefers %s", user["nickname"], ", ".join(preferred))

    logger.info("planned %d events", len(planned))

    if direct:
        return await _write_direct(planned, settings)

    return await _write_through_api(planned, users, settings)


def _load_users(users_file: Path) -> list[dict]:
    if not users_file.exists():
        return []

    return json.loads(users_file.read_text(encoding="utf-8"))


async def _write_direct(events: list[Event], settings) -> int:
    """Straight into ClickHouse, plus the viewed sets the .NET consumer would have written."""
    from redis.asyncio import Redis

    clickhouse = ClickHouseClient(
        settings.clickhouse_url,
        settings.clickhouse_database,
        settings.clickhouse_user,
        settings.clickhouse_password,
    )
    redis = Redis.from_url(settings.redis_url)

    try:
        rows = "\n".join(
            json.dumps(
                {
                    "event_id": e.event_id,
                    "event_type": e.event_type,
                    "user_id": e.user_id,
                    "article_id": e.article_id,
                    "occurred_at": e.occurred_at.astimezone(UTC).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                    "source": "seed-script",
                    "metadata": e.metadata,
                },
                ensure_ascii=False,
            )
            for e in events
        )

        table = f"{settings.clickhouse_database}.{settings.clickhouse_table}"
        await clickhouse.execute(f"INSERT INTO {table} FORMAT JSONEachRow\n{rows}")
        logger.info("inserted %d rows into %s", len(events), table)

        viewed = [e for e in events if e.event_type in ("ArticleRead", "ArticleClicked")]
        async with redis.pipeline(transaction=False) as pipe:
            for event in viewed:
                key = f"{VIEWED_PREFIX}{event.user_id}"
                pipe.sadd(key, event.article_id)
                pipe.expire(key, VIEWED_TTL_DAYS * 86400)

            await pipe.execute()

        logger.info("mirrored %d events into %s*", len(viewed), VIEWED_PREFIX)
        return EXIT_OK
    except ClickHouseError as exc:
        logger.error("insert failed: %s", exc)
        return EXIT_INFRA
    finally:
        await clickhouse.close()
        await redis.aclose()


async def _write_through_api(events: list[Event], users: list[dict], settings) -> int:
    """The honest path: the same endpoints a browser would hit.

    `occurred_at` is discarded here — the article service stamps it at emit time — so every
    event lands in the current minute. Use --direct-clickhouse when history matters.
    """
    identity = IdentityClient(settings.identity_base_url)
    http = httpx.AsyncClient(base_url=settings.articles_base_url.rstrip("/"), timeout=15.0)

    paths = {
        "ArticleClicked": "click",
        "ArticleRead": "read",
        "ArticleLiked": "like",
        "ArticleDisliked": "dislike",
        "ArticleShared": "share",
    }

    try:
        tokens: dict[str, str] = {}
        for user in users:
            try:
                tokens[user["user_id"]] = await identity.login(user["email"], DEFAULT_PASSWORD)
            except IdentityError as exc:
                logger.warning("cannot log in %s: %s", user["email"], exc)

        if not tokens:
            logger.error("no user could log in — is the identity service running?")
            return EXIT_INFRA

        sent = 0
        for event in events:
            token = tokens.get(event.user_id)
            if token is None:
                continue

            response = await http.post(
                f"/articles/{event.article_id}/{paths[event.event_type]}",
                headers={"Authorization": f"Bearer {token}"},
                json=event.metadata or None,
            )

            if response.status_code in (200, 202, 204):
                sent += 1

        logger.info("emitted %d of %d events through the API", sent, len(events))
        return EXIT_OK if sent else EXIT_INFRA
    except httpx.HTTPError as exc:
        logger.error("emitting failed: %s", exc)
        return EXIT_INFRA
    finally:
        await identity.close()
        await http.aclose()


def cli() -> int:
    settings = get_settings()

    parser = argparse.ArgumentParser(description="Generate behavioural events with a recoverable signal")
    parser.add_argument("--events-per-user", type=int, default=200)
    parser.add_argument("--days", type=int, default=21)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--users-file", type=Path, default=USERS_FILE)
    parser.add_argument(
        "--direct-clickhouse",
        action="store_true",
        help="bypass Kafka; insert into ClickHouse and write rec:user_viewed directly",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    configure(args.log_level)

    return asyncio.run(
        simulate(
            args.events_per_user,
            args.days,
            args.seed,
            args.direct_clickhouse,
            args.users_file,
            settings,
        )
    )


if __name__ == "__main__":
    sys.exit(cli())
