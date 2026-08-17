"""Train the ranking model, weekly.

    python -m explained_ml.jobs.train_lightgbm --negatives impressions --promote

Two things here decide whether the model is worth anything, and neither is the hyperparameters.

**Where negatives come from.** `--negatives impressions` reads `feed_impressions`: what the feed
actually showed and the user did not click. That is the only negative set that matches what the
model will be asked to rank. `--negatives sampled` draws from popular-but-untouched articles
instead, and teaches the model that popularity predicts rejection — the opposite of true. The
sampled mode exists because `feed_impressions` does not exist until `ClickHouseImpressionLog`
is deployed; it is a stopgap and says so in the run's parameters.

**Impressions from a degraded feed are excluded.** A click on a `trending` page says the user
liked one of ten globally popular articles, not that the ranker chose well — that page had no
ranker. Learning from it teaches the model to reproduce the fallback it exists to avoid.

The split is temporal, never random: a random split over per-user feeds leaks trivially, since
the same article on the same day appears on both sides.
"""

import argparse
import asyncio
import logging
import random
import sys
import tempfile
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np

from feature_store.compute import build_matrix
from feature_store.context import RequestContext
from feature_store.schema import feature_names
from feature_store.signature import schema_version

from ..articles import ArticleClient, ArticleServiceError
from ..clickhouse import ClickHouseClient, ClickHouseError
from ..config import Settings, get_settings
from ..features import OfflineFeatureBuilder, contexts_from_rows
from ..logs import EXIT_INFRA, EXIT_NOTHING_TO_DO, EXIT_OK, configure
from ..ranking_metrics import grouped
from ..redis_store import RecommendationStore
from ..sql import labelled_events, popular_by_day

logger = logging.getLogger("explained_ml.train_lightgbm")

IMPRESSIONS_TABLE = "feed_impressions"
TRUSTED_LEVELS = ("personalized", "partial", "unranked")


class Dataset:
    """Rows grouped by (user, day) — the unit a ranker is judged on."""

    def __init__(self) -> None:
        self.rows: list[tuple[str, str, str, float]] = []  # user, day, article, label

    def add(self, user_id: str, day: str, article_id: str, label: float) -> None:
        self.rows.append((user_id, day, article_id, label))

    def groups(self) -> dict[tuple[str, str], list[tuple[str, float]]]:
        by_group: dict[tuple[str, str], list[tuple[str, float]]] = defaultdict(list)
        for user_id, day, article_id, label in self.rows:
            by_group[(user_id, day)].append((article_id, label))

        return by_group


async def build(
    settings: Settings,
    window_days: int,
    holdout_days: int,
    negatives_mode: str,
    neg_ratio: int,
    min_examples: int,
    min_groups: int,
    promote: bool,
    params: dict,
    as_of: datetime | None = None,
    clickhouse: ClickHouseClient | None = None,
    articles: ArticleClient | None = None,
    store: RecommendationStore | None = None,
    trainer=None,
    log_to_mlflow: bool = True,
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

    try:
        if negatives_mode == "impressions" and not await _has_impressions(clickhouse, settings):
            logger.error(
                "%s.%s does not exist yet — deploy ClickHouseImpressionLog, or rerun with "
                "--negatives sampled and accept a popularity-biased model",
                settings.clickhouse_database,
                IMPRESSIONS_TABLE,
            )
            return EXIT_NOTHING_TO_DO

        positives = await clickhouse.query_rows(
            labelled_events(table), {"as_of": moment, "window_days": window_days}
        )

        dataset = Dataset()
        for row in positives:
            dataset.add(
                str(row["user_id"]), str(row["day"]), str(row["article_id"]), float(row["label"])
            )

        if not dataset.rows:
            logger.error("no labelled interaction in the last %d days; nothing to learn", window_days)
            return EXIT_NOTHING_TO_DO

        await _add_negatives(
            dataset, clickhouse, settings, table, moment, window_days, negatives_mode, neg_ratio
        )

        rows = await OfflineFeatureBuilder(clickhouse, articles, table).build(moment)
    except ClickHouseError as exc:
        logger.error("training data query failed: %s", exc)
        return EXIT_INFRA
    except ArticleServiceError as exc:
        logger.error("catalog walk failed: %s", exc)
        return EXIT_INFRA
    finally:
        if owns:
            await clickhouse.close()
            await articles.close()

    try:
        matrices = await _featurize(dataset, rows, store, moment)
    finally:
        if owns:
            await store.close()

    if matrices is None:
        return EXIT_NOTHING_TO_DO

    features, labels, group_sizes, group_days = matrices

    if len(labels) < min_examples or len(group_sizes) < min_groups:
        logger.error(
            "%d examples in %d groups is below the floor (%d / %d); the champion keeps serving",
            len(labels),
            len(group_sizes),
            min_examples,
            min_groups,
        )
        return EXIT_NOTHING_TO_DO

    negatives = int((labels == 0).sum())
    if negatives == 0:
        # A positives-only set is not a hard ranking problem, it is a vacuous one: every item in
        # every group is relevant, so any ordering scores perfectly. MAP@20 comes out at exactly
        # 1.0 and looks like a triumph. This happens for real with --negatives impressions on a
        # small catalog, where everything the feed showed was also engaged with.
        logger.error(
            "every one of the %d examples is a positive — lambdarank has nothing to separate. "
            "With --negatives impressions this means the feed never showed anything the user "
            "ignored; widen the window or fall back to --negatives sampled",
            len(labels),
        )
        return EXIT_NOTHING_TO_DO

    logger.info(
        "%d examples: %d positive, %d negative (%.0f%%)",
        len(labels),
        len(labels) - negatives,
        negatives,
        100.0 * negatives / len(labels),
    )

    split = _temporal_split(group_days, moment, holdout_days)
    train, holdout = _partition(features, labels, group_sizes, split)

    if not holdout[2] or not train[2]:
        logger.error(
            "the %d-day holdout leaves one side empty; widen --window-days or shrink --holdout-days",
            holdout_days,
        )
        return EXIT_NOTHING_TO_DO

    logger.info(
        "training on %d examples / %d groups, holding out %d / %d",
        len(train[1]),
        len(train[2]),
        len(holdout[1]),
        len(holdout[2]),
    )

    booster = (trainer or _train_lightgbm)(train, holdout, params)
    scores = booster.predict(holdout[0])
    metrics = grouped(holdout[1], scores, holdout[2], k=20)
    metrics["train_ndcg_at_20"] = grouped(train[1], booster.predict(train[0]), train[2], k=20)[
        "ndcg_at_20"
    ]

    metrics.update(_baselines(holdout))

    logger.info(
        "holdout NDCG@20 %.4f, MAP@20 %.4f (train %.4f) vs baselines: random %.4f, popularity %.4f",
        metrics["ndcg_at_20"],
        metrics["map_at_20"],
        metrics["train_ndcg_at_20"],
        metrics["baseline_random_ndcg_at_20"],
        metrics["baseline_popularity_ndcg_at_20"],
    )

    if metrics["ndcg_at_20"] <= metrics["baseline_random_ndcg_at_20"]:
        logger.warning(
            "the model does not beat random ordering on the holdout — the features carry no "
            "signal for this data, and promoting it would be worse than serving `unranked`"
        )

    if log_to_mlflow:
        _log_run(settings, booster, holdout, metrics, params, negatives_mode, neg_ratio, promote)

    return EXIT_OK


def _baselines(holdout) -> dict[str, float]:
    """NDCG@20 for two orderings that required no model.

    An absolute NDCG is uninterpretable on its own — its value depends on group sizes and on
    how many positives the negative sampler left in each group. Against these two it says
    something: *random* is the floor a ranker must clear to be worth deploying at all, and
    *popularity* is what the `trending` rung already achieves for free. A model that only
    matches popularity has learned the one feature the fallback already has.
    """
    features, labels, groups = holdout
    if len(labels) == 0:
        return {}

    rng = np.random.default_rng(42)
    popularity_column = feature_names().index("a_clicks_24h_log")

    return {
        "baseline_random_ndcg_at_20": grouped(labels, rng.random(len(labels)), groups, k=20)[
            "ndcg_at_20"
        ],
        "baseline_popularity_ndcg_at_20": grouped(
            labels, features[:, popularity_column], groups, k=20
        )["ndcg_at_20"],
    }


async def _has_impressions(clickhouse: ClickHouseClient, settings: Settings) -> bool:
    rows = await clickhouse.query_rows(
        "SELECT count() AS n FROM system.tables "
        f"WHERE database = '{settings.clickhouse_database}' AND name = '{IMPRESSIONS_TABLE}'"
    )

    return bool(rows) and int(rows[0].get("n", 0)) > 0


async def _add_negatives(
    dataset: Dataset,
    clickhouse: ClickHouseClient,
    settings: Settings,
    table: str,
    as_of: datetime,
    window_days: int,
    mode: str,
    neg_ratio: int,
) -> None:
    engaged: dict[tuple[str, str], set[str]] = defaultdict(set)
    for user_id, day, article_id, _label in dataset.rows:
        engaged[(user_id, day)].add(article_id)

    if mode == "impressions":
        levels = "', '".join(TRUSTED_LEVELS)
        rows = await clickhouse.query_rows(
            f"""
SELECT user_id, toDate(served_at) AS day, arrayJoin(article_ids) AS article_id
FROM {settings.clickhouse_database}.{IMPRESSIONS_TABLE}
WHERE served_at <  {{as_of:DateTime64(3)}}
  AND served_at >= {{as_of:DateTime64(3)}} - INTERVAL {{window_days:UInt32}} DAY
  AND level IN ('{levels}')
""",
            {"as_of": as_of, "window_days": window_days},
        )

        added = 0
        for row in rows:
            key = (str(row["user_id"]), str(row["day"]))
            article_id = str(row["article_id"])
            if article_id not in engaged.get(key, set()):
                dataset.add(key[0], key[1], article_id, 0.0)
                added += 1

        logger.info("added %d shown-but-not-engaged negatives from %s", added, IMPRESSIONS_TABLE)
        return

    popular = await clickhouse.query_rows(
        popular_by_day(table), {"as_of": as_of, "window_days": window_days, "per_day": 500}
    )

    by_day: dict[str, list[str]] = defaultdict(list)
    for row in popular:
        by_day[str(row["day"])].append(str(row["article_id"]))

    rng = random.Random(42)
    added = 0
    for (user_id, day), positives in list(engaged.items()):
        pool = [a for a in by_day.get(day, []) if a not in positives]
        if not pool:
            continue

        for article_id in rng.sample(pool, k=min(len(pool), len(positives) * neg_ratio)):
            dataset.add(user_id, day, article_id, 0.0)
            added += 1

    logger.info("added %d popularity-sampled negatives (biased — see the module docstring)", added)


async def _featurize(dataset: Dataset, rows, store: RecommendationStore, as_of: datetime):
    by_group = dataset.groups()

    user_ids = sorted({user for user, _ in by_group})
    article_ids = sorted({a for items in by_group.values() for a, _ in items})

    article_embeddings = await store.get_article_embeddings(article_ids)
    user_embeddings = {u: await store.get_user_embedding(u) for u in user_ids}

    features: list[np.ndarray] = []
    labels: list[float] = []
    group_sizes: list[int] = []
    group_days: list[str] = []

    for (user_id, day), items in sorted(by_group.items()):
        ids = [article_id for article_id, _ in items]
        user, contexts = contexts_from_rows(
            user_id, ids, rows, user_embeddings.get(user_id), article_embeddings
        )

        matrix = build_matrix(user, contexts, RequestContext(now=as_of))
        features.append(matrix)
        labels.extend(label for _, label in items)
        group_sizes.append(len(items))
        group_days.append(day)

    if not features:
        logger.error("no group survived featurization")
        return None

    return np.vstack(features), np.asarray(labels, dtype=np.float64), group_sizes, group_days


def _temporal_split(group_days: list[str], as_of: datetime, holdout_days: int) -> list[bool]:
    """True = holdout. A random split leaks: the same article on the same day lands on both
    sides, and the model is scored on rows it effectively memorised."""
    cutoff = (as_of - timedelta(days=holdout_days)).date().isoformat()

    return [day >= cutoff for day in group_days]


def _partition(features: np.ndarray, labels: np.ndarray, group_sizes: list[int], holdout: list[bool]):
    train_rows: list[np.ndarray] = []
    train_labels: list[np.ndarray] = []
    train_groups: list[int] = []
    hold_rows: list[np.ndarray] = []
    hold_labels: list[np.ndarray] = []
    hold_groups: list[int] = []

    start = 0
    for size, is_holdout in zip(group_sizes, holdout, strict=True):
        end = start + size
        if is_holdout:
            hold_rows.append(features[start:end])
            hold_labels.append(labels[start:end])
            hold_groups.append(size)
        else:
            train_rows.append(features[start:end])
            train_labels.append(labels[start:end])
            train_groups.append(size)

        start = end

    def stack(rows, label_chunks, groups):
        width = features.shape[1]
        matrix = np.vstack(rows) if rows else np.zeros((0, width), dtype=np.float32)
        values = np.concatenate(label_chunks) if label_chunks else np.zeros(0)
        return matrix, values, groups

    return stack(train_rows, train_labels, train_groups), stack(hold_rows, hold_labels, hold_groups)


def _train_lightgbm(train, holdout, params: dict):
    import lightgbm as lgb

    train_set = lgb.Dataset(train[0], label=train[1], group=train[2], feature_name=feature_names())
    valid_set = lgb.Dataset(
        holdout[0], label=holdout[1], group=holdout[2], reference=train_set, feature_name=feature_names()
    )

    return lgb.train(
        {
            "objective": "lambdarank",
            "metric": "ndcg",
            "ndcg_eval_at": [20],
            "label_gain": [0, 1, 3, 7],
            "verbosity": -1,
            **params,
        },
        train_set,
        valid_sets=[valid_set],
        callbacks=[lgb.early_stopping(20, verbose=False)],
    )


def _log_run(settings, booster, holdout, metrics, params, negatives_mode, neg_ratio, promote) -> None:
    try:
        import mlflow
        import pandas as pd
        from mlflow.models import infer_signature

        from ..tracking import champion_metric, run
        from ..tracking import promote as move_alias

        names = feature_names()

        with run(settings, "train_lightgbm"):
            mlflow.log_params(
                {
                    **params,
                    "negatives_mode": negatives_mode,
                    "neg_ratio": neg_ratio,
                    "n_features": len(names),
                    "feature_schema_version": schema_version(),
                    "embedding_model": settings.embedding_model,
                }
            )
            mlflow.log_metrics(metrics)
            mlflow.set_tag("feature_schema_version", schema_version())

            frame = pd.DataFrame(holdout[0], columns=names)
            signature = infer_signature(frame, booster.predict(holdout[0]))

            info = mlflow.lightgbm.log_model(
                booster,
                name="model",
                signature=signature,
                registered_model_name=settings.mlflow_model_name,
            )

            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "feature_names.json"
                path.write_text("\n".join(names), encoding="utf-8")
                mlflow.log_artifact(str(path))

            version = getattr(info, "registered_model_version", None)
            if not promote:
                logger.info("registered version %s without moving @%s", version, settings.mlflow_model_alias)
                return

            current = champion_metric(settings, "ndcg_at_20")
            if current is not None and metrics["ndcg_at_20"] <= current:
                logger.info(
                    "not promoting: holdout NDCG@20 %.4f does not beat the champion's %.4f",
                    metrics["ndcg_at_20"],
                    current,
                )
                return

            if version is not None:
                move_alias(settings, str(version))
    except Exception as exc:  # noqa: BLE001 - a tracking failure must not lose the trained model
        logger.warning("could not log the run to MLflow: %s", exc)


def cli() -> int:
    settings = get_settings()

    parser = argparse.ArgumentParser(description="Train the LightGBM ranker")
    parser.add_argument("--window-days", type=int, default=30)
    parser.add_argument("--holdout-days", type=int, default=7)
    parser.add_argument("--negatives", choices=("sampled", "impressions"), default="sampled")
    parser.add_argument("--neg-ratio", type=int, default=4)
    parser.add_argument("--min-examples", type=int, default=500)
    parser.add_argument("--min-groups", type=int, default=50)
    parser.add_argument("--num-leaves", type=int, default=31)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--num-boost-round", type=int, default=300)
    parser.add_argument(
        "--promote",
        action="store_true",
        help=f"move the @{settings.mlflow_model_alias} alias if the holdout beats the champion",
    )
    parser.add_argument("--no-mlflow", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    configure(args.log_level)

    params = {
        "num_leaves": args.num_leaves,
        "learning_rate": args.learning_rate,
        "num_boost_round": args.num_boost_round,
    }

    try:
        return asyncio.run(
            build(
                settings,
                args.window_days,
                args.holdout_days,
                args.negatives,
                args.neg_ratio,
                args.min_examples,
                args.min_groups,
                args.promote,
                params,
                log_to_mlflow=not args.no_mlflow,
            )
        )
    except ImportError as exc:
        logger.error("training needs lightgbm: %s", exc)
        return EXIT_INFRA


if __name__ == "__main__":
    sys.exit(cli())
