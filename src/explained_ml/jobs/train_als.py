"""Implicit ALS -> `rec:user_als_candidates:{uid}`, daily.

    python -m explained_ml.jobs.train_als

This is the second personalized candidate source, and it answers a different question from
FAISS. FAISS asks "what looks like what this user has read"; ALS asks "what did users who
behaved like this one read". A user whose taste is not expressible as a centroid — someone with
two unrelated interests — gets nothing useful from the first and something useful from the
second.

Value shape is a Redis LIST, best first, exactly like `rec:trending:top100`, because
`RedisRecommendationStore` reads lists with `ListRangeAsync` and a matching shape makes
`AlsCandidateSource` a copy of `TrendingCandidateSource` rather than a new deserializer.
Scores are dropped: `TrendingCandidateSource` already synthesizes a score from rank, and two
sources disagreeing about score scale would corrupt the interleave.

A missing key is a cold-start user, not a failure — the C# source must report `Disabled`, the
same way `FaissCandidateSource` treats a 204.
"""

import os

# Before numpy: OpenBLAS reads this when its library loads, and `implicit` warns loudly that
# nesting its own parallelism inside a BLAS threadpool is a severe performance problem here.
# Setting it after the import is silently too late.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse  # noqa: E402
import asyncio  # noqa: E402
import logging  # noqa: E402
import sys  # noqa: E402
from collections import defaultdict  # noqa: E402
from datetime import UTC, datetime  # noqa: E402

import numpy as np  # noqa: E402

from ..clickhouse import ClickHouseClient, ClickHouseError  # noqa: E402
from ..config import Settings, get_settings  # noqa: E402
from ..logs import EXIT_INFRA, EXIT_NOTHING_TO_DO, EXIT_OK, configure  # noqa: E402
from ..redis_store import RecommendationStore  # noqa: E402
from ..sql import interactions  # noqa: E402

logger = logging.getLogger("explained_ml.train_als")


class Interactions:
    """The user-item matrix, plus the index maps needed to read results back out."""

    def __init__(self, rows: list[dict], min_user: int, min_item: int) -> None:
        by_user: dict[str, dict[str, float]] = defaultdict(dict)
        item_users: dict[str, set[str]] = defaultdict(set)

        for row in rows:
            user_id = str(row.get("user_id") or "")
            article_id = str(row.get("article_id") or "")
            affinity = float(row.get("affinity", 0.0))
            if not user_id or not article_id or affinity <= 0:
                continue

            by_user[user_id][article_id] = affinity
            item_users[article_id].add(user_id)

        # Filtering is not tidiness. An item touched by one user contributes a factor fitted to
        # that user alone, which ALS will happily recommend back to them and nobody else.
        keep_items = {item for item, users in item_users.items() if len(users) >= min_item}
        trimmed = {
            user: {i: v for i, v in items.items() if i in keep_items} for user, items in by_user.items()
        }
        trimmed = {user: items for user, items in trimmed.items() if len(items) >= min_user}

        self.user_ids = sorted(trimmed)
        self.article_ids = sorted({item for items in trimmed.values() for item in items})
        self._user_index = {user: i for i, user in enumerate(self.user_ids)}
        self._article_index = {item: i for i, item in enumerate(self.article_ids)}
        self._rows = trimmed

    def __len__(self) -> int:
        return len(self.user_ids)

    @property
    def item_count(self) -> int:
        return len(self.article_ids)

    @property
    def nnz(self) -> int:
        return sum(len(items) for items in self._rows.values())

    def matrix(self, alpha: float):
        """CSR of confidences. `1 + alpha * log1p(affinity)` is the Hu et al. weighting: the
        difference between one click and two matters far more than between forty and forty-one.
        """
        from scipy.sparse import csr_matrix

        data: list[float] = []
        rows: list[int] = []
        cols: list[int] = []

        for user, items in self._rows.items():
            for article, affinity in items.items():
                rows.append(self._user_index[user])
                cols.append(self._article_index[article])
                data.append(1.0 + alpha * float(np.log1p(affinity)))

        return csr_matrix(
            (data, (rows, cols)), shape=(len(self.user_ids), len(self.article_ids)), dtype=np.float32
        )


async def build(
    settings: Settings,
    window_days: int,
    factors: int,
    iterations: int,
    regularization: float,
    alpha: float,
    top_n: int,
    min_user_interactions: int,
    min_item_users: int,
    min_users: int,
    log_to_mlflow: bool = True,
    as_of: datetime | None = None,
    clickhouse: ClickHouseClient | None = None,
    store: RecommendationStore | None = None,
    factorizer=None,
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
            interactions(table), {"as_of": moment, "window_days": window_days}
        )
    except ClickHouseError as exc:
        logger.error("interaction query failed, yesterday's candidates keep serving: %s", exc)
        return EXIT_INFRA
    finally:
        if owns:
            await clickhouse.close()

    try:
        data = Interactions(rows, min_user_interactions, min_item_users)

        if len(data) < min_users:
            logger.error(
                "only %d users cleared >=%d interactions (need %d); no model trained and "
                "existing candidates left alone",
                len(data),
                min_user_interactions,
                min_users,
            )
            return EXIT_NOTHING_TO_DO

        logger.info(
            "fitting ALS on %d users x %d articles (%d interactions)",
            len(data),
            data.item_count,
            data.nnz,
        )

        matrix = data.matrix(alpha)
        model = factorizer or _implicit_model(factors, iterations, regularization)
        model.fit(matrix)

        candidates = _recommend(model, data, matrix, top_n)
        written = await store.set_als_candidates(candidates, settings.als_candidates_ttl_seconds)

        coverage = len({a for ids in candidates.values() for a in ids}) / max(data.item_count, 1)
        logger.info(
            "wrote candidates for %d users, catalog coverage %.2f", written, coverage
        )

        if log_to_mlflow:
            _log_run(settings, data, factors, iterations, regularization, alpha, coverage, model)

        return EXIT_OK
    finally:
        if owns:
            await store.close()


def _implicit_model(factors: int, iterations: int, regularization: float):
    from implicit.als import AlternatingLeastSquares

    return AlternatingLeastSquares(
        factors=factors,
        iterations=iterations,
        regularization=regularization,
        random_state=42,
    )


def _recommend(model, data: Interactions, matrix, top_n: int) -> dict[str, list[str]]:
    candidates: dict[str, list[str]] = {}

    for index, user_id in enumerate(data.user_ids):
        ids, _scores = model.recommend(
            index, matrix[index], N=top_n, filter_already_liked_items=True
        )
        picked = [data.article_ids[i] for i in np.asarray(ids).ravel() if 0 <= i < data.item_count]
        if picked:
            candidates[user_id] = picked

    return candidates


def _log_run(settings, data, factors, iterations, regularization, alpha, coverage, model) -> None:
    """Factors go to MLflow; candidate lists do not. The lists live in Redis, expire, and are
    regenerated daily — versioning them would be storing a cache."""
    try:
        import mlflow

        from ..tracking import run

        with run(settings, "train_als"):
            mlflow.log_params(
                {
                    "factors": factors,
                    "iterations": iterations,
                    "regularization": regularization,
                    "alpha": alpha,
                    "users": len(data),
                    "items": data.item_count,
                    "nnz": data.nnz,
                }
            )
            mlflow.log_metric("catalog_coverage", coverage)

            user_factors = getattr(model, "user_factors", None)
            item_factors = getattr(model, "item_factors", None)
            if user_factors is not None and item_factors is not None:
                import tempfile
                from pathlib import Path

                with tempfile.TemporaryDirectory() as tmp:
                    path = Path(tmp) / "als_factors.npz"
                    np.savez_compressed(
                        path,
                        user_factors=np.asarray(user_factors),
                        item_factors=np.asarray(item_factors),
                        user_ids=np.asarray(data.user_ids),
                        item_ids=np.asarray(data.article_ids),
                    )
                    mlflow.log_artifact(str(path))
    except Exception as exc:  # noqa: BLE001 - tracking must never fail the job
        logger.warning("could not log the ALS run to MLflow: %s", exc)


def cli() -> int:
    settings = get_settings()

    parser = argparse.ArgumentParser(description="Train implicit ALS and write candidate lists")
    parser.add_argument("--window-days", type=int, default=90)
    parser.add_argument("--factors", type=int, default=64)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--regularization", type=float, default=0.05)
    parser.add_argument("--alpha", type=float, default=40.0)
    parser.add_argument("--top-n", type=int, default=settings.als_candidates_per_user)
    parser.add_argument("--min-user-interactions", type=int, default=3)
    parser.add_argument("--min-item-users", type=int, default=2)
    parser.add_argument("--min-users", type=int, default=5)
    parser.add_argument("--no-mlflow", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    configure(args.log_level)

    try:
        return asyncio.run(
            build(
                settings,
                args.window_days,
                args.factors,
                args.iterations,
                args.regularization,
                args.alpha,
                args.top_n,
                args.min_user_interactions,
                args.min_item_users,
                args.min_users,
                log_to_mlflow=not args.no_mlflow,
            )
        )
    except ImportError as exc:
        logger.error("ALS needs the `jobs` dependency group: %s", exc)
        return EXIT_INFRA


if __name__ == "__main__":
    sys.exit(cli())
