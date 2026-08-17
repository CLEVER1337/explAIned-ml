"""Interaction history -> `rec:user_embedding:{uid}`, hourly, after `embeddings`.

    python -m explained_ml.jobs.build_user_embeddings

This closes the gap the original design left open: the FAISS service requires
`rec:user_embedding:{uid}` and nothing was ever going to write it. Without this job
`FaissCandidateSource` reports `Disabled` for every user forever, which the orchestrator
correctly refuses to call degradation — so the feed looks healthy while personalization is
simply absent.

The vector is a freshness-weighted centroid of the articles the user engaged with. Disliked
articles are dropped rather than subtracted: the opposite of an article's embedding is not a
direction that means anything, so pushing away needs a different mechanism than pulling toward.

A user with too little history is **skipped, not approximated**. A centroid built from one
click is noise wearing a personalization badge, and a missing key produces the honest `204`
that the whole cold-start path is designed around.
"""

import argparse
import asyncio
import logging
import sys
from collections import defaultdict
from datetime import UTC, datetime

import numpy as np

from ..clickhouse import ClickHouseClient, ClickHouseError
from ..codec import normalize
from ..config import Settings, get_settings
from ..logs import EXIT_INFRA, EXIT_NOTHING_TO_DO, EXIT_OK, configure
from ..redis_store import RecommendationStore
from ..sql import user_affinity

logger = logging.getLogger("explained_ml.build_user_embeddings")


async def build(
    settings: Settings,
    window_days: int,
    half_life_days: float,
    per_user: int,
    min_articles: int,
    only_users: list[str] | None = None,
    as_of: datetime | None = None,
    clickhouse: ClickHouseClient | None = None,
    store: RecommendationStore | None = None,
) -> int:
    moment = as_of or datetime.now(UTC)

    owns = clickhouse is None and store is None
    clickhouse = clickhouse or ClickHouseClient(
        settings.clickhouse_url,
        settings.clickhouse_database,
        settings.clickhouse_user,
        settings.clickhouse_password,
    )
    store = store or RecommendationStore(settings.redis_url, settings.embedding_dim)

    table = f"{settings.clickhouse_database}.{settings.clickhouse_table}"

    try:
        rows = await clickhouse.query_rows(
            user_affinity(table),
            {"as_of": moment, "window_days": window_days, "per_user": per_user},
        )
    except ClickHouseError as exc:
        logger.error("affinity query failed, existing embeddings untouched: %s", exc)
        return EXIT_INFRA
    finally:
        if owns:
            await clickhouse.close()

    wanted = set(only_users or [])
    per_user_rows: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        user_id = str(row.get("user_id") or "")
        if user_id and (not wanted or user_id in wanted):
            per_user_rows[user_id].append(row)

    if not per_user_rows:
        logger.error("no user has positive interactions in the last %d days", window_days)
        return EXIT_NOTHING_TO_DO

    try:
        article_ids = sorted({str(r["article_id"]) for rows_ in per_user_rows.values() for r in rows_})
        vectors = await store.get_article_embeddings(article_ids)
        logger.info(
            "%d users, %d distinct articles, %d of them embedded",
            len(per_user_rows),
            len(article_ids),
            len(vectors),
        )

        if not vectors:
            logger.error(
                "none of the interacted articles has an embedding — run the embeddings job first"
            )
            return EXIT_NOTHING_TO_DO

        centroids: dict[str, np.ndarray] = {}
        skipped = 0

        for user_id, user_rows in per_user_rows.items():
            centroid = _centroid(user_rows, vectors, moment, half_life_days, min_articles)
            if centroid is None:
                skipped += 1
                continue

            centroids[user_id] = centroid

        if not centroids:
            logger.error(
                "every one of the %d users fell below --min-articles=%d; nothing written",
                len(per_user_rows),
                min_articles,
            )
            return EXIT_NOTHING_TO_DO

        await store.set_user_embeddings(centroids)
        await store.mark_fresh("user_embedding")

        logger.info(
            "wrote %d user embeddings, skipped %d users with fewer than %d embedded articles",
            len(centroids),
            skipped,
            min_articles,
        )
        return EXIT_OK
    finally:
        if owns:
            await store.close()


def _centroid(
    rows: list[dict],
    vectors: dict[str, np.ndarray],
    as_of: datetime,
    half_life_days: float,
    min_articles: int,
) -> np.ndarray | None:
    weighted: list[np.ndarray] = []
    weights: list[float] = []

    for row in rows:
        vector = vectors.get(str(row["article_id"]))
        if vector is None:
            continue

        affinity = float(row.get("affinity", 0.0))
        if affinity <= 0:
            continue

        age_days = max((as_of - _parse(row.get("last_at"), as_of)).total_seconds() / 86400.0, 0.0)
        weight = affinity * (0.5 ** (age_days / half_life_days))
        if weight <= 0:
            continue

        weighted.append(vector)
        weights.append(weight)

    if len(weighted) < min_articles:
        return None

    stacked = np.vstack(weighted)
    centroid = np.average(stacked, axis=0, weights=np.asarray(weights, dtype=np.float64))

    normalized = normalize(centroid.astype(np.float32))
    if not np.any(normalized):
        # Degenerate history: everything cancelled out. A zero vector would make every cosine
        # exactly 0, which is worse than having no embedding at all — 204 is the honest answer.
        return None

    return normalized


def _parse(value, fallback: datetime) -> datetime:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)

    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace(" ", "T").replace("Z", "+00:00"))
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
        except ValueError:
            logger.warning("unparseable last_at %r, treating the interaction as current", value)

    return fallback


def cli() -> int:
    settings = get_settings()

    parser = argparse.ArgumentParser(description="Build user embeddings from interaction history")
    parser.add_argument("--window-days", type=int, default=settings.user_embedding_window_days)
    parser.add_argument("--half-life-days", type=float, default=7.0)
    parser.add_argument("--per-user", type=int, default=50)
    parser.add_argument(
        "--min-articles",
        type=int,
        default=2,
        help="below this the centroid is noise; the user stays cold and FAISS answers 204",
    )
    parser.add_argument("--only-user", action="append", default=None)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    configure(args.log_level)

    return asyncio.run(
        build(
            settings,
            args.window_days,
            args.half_life_days,
            args.per_user,
            args.min_articles,
            args.only_user,
        )
    )


if __name__ == "__main__":
    sys.exit(cli())
