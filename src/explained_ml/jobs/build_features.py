"""Materialize feature rows into Redis, hourly.

    python -m explained_ml.jobs.build_features
    python -m explained_ml.jobs.build_features --explain <user_id> <article_id>

`/rank` gets up to 200 candidates and the orchestrator gives it 120 ms including HTTP. A
ClickHouse aggregate per request is not merely slow, it is an order of magnitude outside the
budget — so the aggregates are computed here, once an hour, and the service does one Redis
round trip.

The rows written here come from the same `OfflineFeatureBuilder` that `train_lightgbm.py` uses,
which is what makes the online and offline paths the same code rather than two implementations
that agree by inspection.

Rows carry a TTL. If this job dies unnoticed the rows expire, candidates fall back to neutral
values, and the ranker degrades gradually instead of serving month-old popularity as if it were
current.
"""

import argparse
import asyncio
import logging
import sys
from datetime import UTC, datetime

from feature_store.compute import explain

from ..articles import ArticleClient, ArticleServiceError
from ..clickhouse import ClickHouseClient, ClickHouseError
from ..config import Settings, get_settings
from ..features import OfflineFeatureBuilder, contexts_from_rows
from ..logs import EXIT_INFRA, EXIT_NOTHING_TO_DO, EXIT_OK, configure
from ..redis_store import RecommendationStore

logger = logging.getLogger("explained_ml.build_features")


async def build(
    settings: Settings,
    article_window_hours: int = 24,
    user_window_days: int = 7,
    dry_run: bool = False,
    explain_pair: tuple[str, str] | None = None,
    as_of: datetime | None = None,
    clickhouse: ClickHouseClient | None = None,
    articles: ArticleClient | None = None,
    store: RecommendationStore | None = None,
) -> int:
    moment = as_of or datetime.now(UTC)

    owns = clickhouse is None and articles is None and store is None
    clickhouse = clickhouse or ClickHouseClient(
        settings.clickhouse_url,
        settings.clickhouse_database,
        settings.clickhouse_user,
        settings.clickhouse_password,
    )
    articles = articles or ArticleClient(settings.articles_base_url, settings.request_timeout_seconds)
    store = store or RecommendationStore(settings.redis_url, settings.embedding_dim)

    table = f"{settings.clickhouse_database}.{settings.clickhouse_table}"
    builder = OfflineFeatureBuilder(clickhouse, articles, table)

    try:
        rows = await builder.build(moment, article_window_hours, user_window_days)
    except ClickHouseError as exc:
        logger.error("aggregation failed, existing rows untouched: %s", exc)
        return EXIT_INFRA
    except ArticleServiceError as exc:
        logger.error("catalog walk failed, existing rows untouched: %s", exc)
        return EXIT_INFRA
    finally:
        if owns:
            await clickhouse.close()
            await articles.close()

    try:
        if not rows.articles:
            logger.error("no published article to describe; existing rows left to expire on TTL")
            return EXIT_NOTHING_TO_DO

        if explain_pair:
            await _explain(store, rows, moment, *explain_pair)

        if dry_run:
            logger.info(
                "dry run: would write %d article rows and %d user rows",
                len(rows.articles),
                len(rows.users),
            )
            return EXIT_OK

        await store.set_article_features(rows.articles, settings.features_ttl_seconds)
        await store.set_user_features(rows.users, settings.features_ttl_seconds)
        await store.mark_fresh("features")

        logger.info(
            "wrote %d article rows and %d user rows (ttl %ds)",
            len(rows.articles),
            len(rows.users),
            settings.features_ttl_seconds,
        )
        return EXIT_OK
    finally:
        if owns:
            await store.close()


async def _explain(
    store: RecommendationStore, rows, as_of: datetime, user_id: str, article_id: str
) -> None:
    """Print one candidate's named feature values. The affordance that makes feature bugs
    findable at all — a vector of 14 floats tells you nothing without its labels."""
    from feature_store.context import RequestContext

    user_embedding = await store.get_user_embedding(user_id)
    article_embeddings = await store.get_article_embeddings([article_id])

    user, articles = contexts_from_rows(
        user_id, [article_id], rows, user_embedding, article_embeddings
    )
    values = explain(user, articles[0], RequestContext(now=as_of))

    width = max(len(name) for name in values)
    logger.info("features for user=%s article=%s", user_id, article_id)
    for name, value in values.items():
        logger.info("  %-*s %12.4f", width, name, value)


def cli() -> int:
    settings = get_settings()

    parser = argparse.ArgumentParser(description="Materialize feature rows into Redis")
    parser.add_argument("--article-window-hours", type=int, default=24)
    parser.add_argument("--user-window-days", type=int, default=7)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--explain",
        nargs=2,
        metavar=("USER_ID", "ARTICLE_ID"),
        help="print the named feature values for one candidate",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    configure(args.log_level)

    return asyncio.run(
        build(
            settings,
            args.article_window_hours,
            args.user_window_days,
            dry_run=args.dry_run,
            explain_pair=tuple(args.explain) if args.explain else None,
        )
    )


if __name__ == "__main__":
    sys.exit(cli())
