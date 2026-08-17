"""Is everything this repository needs actually there?

    python -m explained_ml.scripts.check_env

Every job exits 2 when infrastructure is down, which tells you *that* something is wrong but
not *what*. This prints the whole picture in one screen: which stores answer, how much data
they hold, and how stale each derived artifact is. It is the first thing to run after any
exit code 2, and the fastest way to see that a pipeline has silently stopped producing.
"""

import argparse
import asyncio
import logging
import sys
from datetime import UTC, datetime

from ..articles import ArticleClient, ArticleServiceError
from ..clickhouse import ClickHouseClient, ClickHouseError
from ..config import Settings, get_settings
from ..logs import EXIT_INFRA, EXIT_OK, configure
from ..redis_store import (
    ALS_CANDIDATES_PREFIX,
    ARTICLE_EMBEDDING_PREFIX,
    ARTICLE_FEATURES_PREFIX,
    TRENDING_KEY,
    USER_EMBEDDING_PREFIX,
    USER_FEATURES_PREFIX,
    VIEWED_PREFIX,
    RecommendationStore,
)

logger = logging.getLogger("explained_ml.check_env")

ARTIFACTS = (
    "article_embedding",
    "user_embedding",
    "trending",
    "features",
    "user_als_candidates",
)


class Report:
    def __init__(self) -> None:
        self.lines: list[tuple[bool, str, str]] = []

    def ok(self, subject: str, detail: str) -> None:
        self.lines.append((True, subject, detail))

    def bad(self, subject: str, detail: str) -> None:
        self.lines.append((False, subject, detail))

    @property
    def healthy(self) -> bool:
        return all(ok for ok, _, _ in self.lines)

    def render(self) -> str:
        width = max(len(subject) for _, subject, _ in self.lines)
        return "\n".join(
            f"  {'OK ' if ok else 'DOWN'}  {subject:<{width}}  {detail}"
            for ok, subject, detail in self.lines
        )


async def check(settings: Settings) -> Report:
    report = Report()

    await _check_redis(settings, report)
    await _check_clickhouse(settings, report)
    await _check_articles(settings, report)
    _check_mlflow(settings, report)

    return report


async def _check_redis(settings: Settings, report: Report) -> None:
    store = RecommendationStore(settings.redis_url, settings.embedding_dim)

    try:
        await store.redis.ping()
    except Exception as exc:  # noqa: BLE001
        report.bad("redis", f"{type(exc).__name__}: {exc}")
        await store.close()
        return

    report.ok("redis", settings.redis_url.split("@")[-1])

    try:
        counts = {
            "article embeddings": await _count(store, f"{ARTICLE_EMBEDDING_PREFIX}*"),
            "user embeddings": await _count(store, f"{USER_EMBEDDING_PREFIX}*"),
            "article feature rows": await _count(store, f"{ARTICLE_FEATURES_PREFIX}*"),
            "user feature rows": await _count(store, f"{USER_FEATURES_PREFIX}*"),
            "als candidate lists": await _count(store, f"{ALS_CANDIDATES_PREFIX}*"),
            "viewed sets": await _count(store, f"{VIEWED_PREFIX}*"),
        }

        for label, count in counts.items():
            (report.ok if count else report.bad)(label, str(count))

        trending = await store.trending_length()
        (report.ok if trending else report.bad)("trending list", f"{trending} ids in {TRENDING_KEY}")

        now = datetime.now(UTC)
        for name in ARTIFACTS:
            stamp = await store.get_freshness(name)
            if stamp is None:
                report.bad(f"freshness {name}", "never written")
            else:
                report.ok(f"freshness {name}", f"{(now - stamp).total_seconds() / 60:.0f} min ago")
    finally:
        await store.close()


async def _count(store: RecommendationStore, pattern: str) -> int:
    total = 0
    async for _key in store.redis.scan_iter(match=pattern, count=500):
        total += 1

    return total


async def _check_clickhouse(settings: Settings, report: Report) -> None:
    client = ClickHouseClient(
        settings.clickhouse_url,
        settings.clickhouse_database,
        settings.clickhouse_user,
        settings.clickhouse_password,
    )
    table = f"{settings.clickhouse_database}.{settings.clickhouse_table}"

    try:
        rows = await client.query_rows(
            f"SELECT count() AS rows, uniq(user_id) AS users, uniq(article_id) AS articles, "
            f"max(occurred_at) AS newest FROM {table} FINAL"
        )
        row = rows[0] if rows else {}
        count = int(row.get("rows", 0))

        (report.ok if count else report.bad)(
            table,
            f"{count} rows, {row.get('users', 0)} users, {row.get('articles', 0)} articles, "
            f"newest {row.get('newest', 'n/a')}",
        )

        impressions = await client.query_rows(
            "SELECT count() AS n FROM system.tables "
            f"WHERE database = '{settings.clickhouse_database}' AND name = 'feed_impressions'"
        )
        if impressions and int(impressions[0].get("n", 0)):
            served = await client.query_rows(
                f"SELECT count() AS rows FROM {settings.clickhouse_database}.feed_impressions"
            )
            report.ok("feed_impressions", f"{served[0].get('rows', 0)} rows")
        else:
            report.bad("feed_impressions", "table missing — train_lightgbm falls back to sampled negatives")
    except ClickHouseError as exc:
        report.bad("clickhouse", str(exc)[:160])
    finally:
        await client.close()


async def _check_articles(settings: Settings, report: Report) -> None:
    client = ArticleClient(settings.articles_base_url, settings.request_timeout_seconds)

    try:
        page = await client.recent(limit=100, offset=0)
        report.ok(
            "article service",
            f"{settings.articles_base_url}, {len(page)} published on the first page",
        ) if page else report.bad("article service", "no published articles — run seed_dev_corpus")
    except ArticleServiceError as exc:
        report.bad("article service", str(exc)[:160])
    finally:
        await client.close()


def _check_mlflow(settings: Settings, report: Report) -> None:
    try:
        from ..tracking import ModelUnavailable, resolve_alias

        try:
            _uri, version = resolve_alias(settings)
            report.ok(
                f"model @{settings.mlflow_model_alias}", f"{settings.mlflow_model_name} v{version}"
            )
        except ModelUnavailable:
            report.bad(
                f"model @{settings.mlflow_model_alias}",
                "not promoted yet — /rank answers 503 and the feed serves `unranked`",
            )
    except Exception as exc:  # noqa: BLE001
        report.bad("mlflow", f"{type(exc).__name__}: {str(exc)[:120]}")


def cli() -> int:
    parser = argparse.ArgumentParser(description="Probe everything explAIned-ml depends on")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    configure(args.log_level)

    report = asyncio.run(check(get_settings()))
    print(report.render())

    return EXIT_OK if report.healthy else EXIT_INFRA


if __name__ == "__main__":
    sys.exit(cli())
