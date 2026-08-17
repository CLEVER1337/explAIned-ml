"""ClickHouse -> `rec:trending:top100`, every 10 minutes.

    python -m explained_ml.jobs.update_trending

This is the highest-value job in the repository and the least clever one. The `trending` rung of
the degradation ladder is fully implemented on the C# side and has simply never had data: with
this key empty every user falls through to `recent`, which is a chronological list, not a
recommendation. One SQL query and one Redis list fix that for everybody, including users who
will never have an embedding.

Failure behaviour is the important part. An empty aggregate **does not** write an empty list —
it exits 1 and leaves the previous one serving. Blanking this key silently drops the entire
user base one rung, and the symptom (a feed that still returns 200) looks like success.
"""

import argparse
import asyncio
import logging
import sys
from datetime import UTC, datetime

from ..clickhouse import ClickHouseClient, ClickHouseError
from ..config import Settings, get_settings
from ..logs import EXIT_INFRA, EXIT_NOTHING_TO_DO, EXIT_OK, configure
from ..redis_store import RecommendationStore
from ..sql import trending

logger = logging.getLogger("explained_ml.update_trending")


async def build(
    settings: Settings,
    window_hours: int,
    half_life_hours: float,
    limit: int,
    min_reach: int,
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
            trending(table),
            {
                "as_of": moment,
                "window_hours": window_hours,
                "half_life_hours": half_life_hours,
                "min_reach": min_reach,
                "limit": limit,
            },
        )
    except ClickHouseError as exc:
        logger.error("aggregation failed, leaving the current list alone: %s", exc)
        return EXIT_INFRA
    finally:
        if owns:
            await clickhouse.close()

    article_ids = [str(row["article_id"]) for row in rows if row.get("article_id")]

    try:
        if not article_ids:
            existing = await store.trending_length()
            logger.error(
                "no article cleared reach >= %d in the last %dh; keeping the existing list "
                "(%d ids). Feeds stay on `trending` instead of dropping to `recent`.",
                min_reach,
                window_hours,
                existing,
            )
            return EXIT_NOTHING_TO_DO

        await store.replace_trending(article_ids, settings.trending_ttl_seconds)
        logger.info(
            "wrote %d trending ids (top score %.2f, window %dh, half-life %.1fh)",
            len(article_ids),
            float(rows[0].get("score", 0.0)),
            window_hours,
            half_life_hours,
        )
        return EXIT_OK
    finally:
        if owns:
            await store.close()


def cli() -> int:
    settings = get_settings()

    parser = argparse.ArgumentParser(description="Refresh rec:trending:top100 from ClickHouse")
    parser.add_argument("--window-hours", type=int, default=settings.trending_window_hours)
    parser.add_argument("--half-life-hours", type=float, default=12.0)
    parser.add_argument("--limit", type=int, default=settings.trending_size)
    parser.add_argument(
        "--min-reach",
        type=int,
        default=2,
        help="distinct users an article needs before it can trend; 1 for a tiny dev corpus",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    configure(args.log_level)

    return asyncio.run(
        build(
            settings,
            args.window_hours,
            args.half_life_hours,
            args.limit,
            args.min_reach,
        )
    )


if __name__ == "__main__":
    sys.exit(cli())
