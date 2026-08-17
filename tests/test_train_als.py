"""`implicit` itself is never exercised here — factorization is the library's job and it is not
installable in every environment. What is tested is everything around it, which is where the
bugs that reach production actually live: matrix construction, the filtering thresholds, index
round-tripping, and the Redis value shape the C# side depends on.
"""

from datetime import UTC, datetime

import numpy as np
import pytest

from explained_ml.config import Settings
from explained_ml.jobs.train_als import Interactions, build
from explained_ml.logs import EXIT_INFRA, EXIT_NOTHING_TO_DO, EXIT_OK
from explained_ml.redis_store import ALS_CANDIDATES_PREFIX, RecommendationStore, freshness_key

AS_OF = datetime(2026, 8, 17, 12, 0, tzinfo=UTC)


class FakeClickHouse:
    def __init__(self, rows=None, error: Exception | None = None) -> None:
        self.rows = rows or []
        self.error = error

    async def query_rows(self, sql: str, parameters: dict | None = None):
        if self.error:
            raise self.error
        return self.rows

    async def close(self):
        pass


class FakeALS:
    """Recommends items in a fixed order, so assertions are about plumbing, not about maths."""

    def __init__(self, per_user: dict[int, list[int]] | None = None) -> None:
        self.per_user = per_user or {}
        self.fitted_shape = None
        self.user_factors = np.zeros((2, 4), dtype=np.float32)
        self.item_factors = np.zeros((2, 4), dtype=np.float32)

    def fit(self, matrix):
        self.fitted_shape = matrix.shape

    def recommend(self, index, row, N, filter_already_liked_items=True):
        picked = self.per_user.get(index, list(range(N)))
        return np.array(picked[:N]), np.ones(len(picked[:N]))


def interaction(user: str, article: str, affinity: float = 3.0) -> dict:
    return {"user_id": user, "article_id": article, "affinity": affinity}


def dense_rows(users: int, articles: int) -> list[dict]:
    return [
        interaction(f"u{u}", f"a{a}") for u in range(users) for a in range(articles)
    ]


@pytest.fixture
def settings():
    return Settings(als_candidates_ttl_seconds=604800)


@pytest.fixture
def store(fake_redis):
    return RecommendationStore("redis://unused", 8, client=fake_redis)


async def run(settings, store, clickhouse, factorizer=None, **overrides):
    kwargs = {
        "window_days": 90,
        "factors": 4,
        "iterations": 2,
        "regularization": 0.05,
        "alpha": 40.0,
        "top_n": 5,
        "min_user_interactions": 2,
        "min_item_users": 2,
        "min_users": 2,
    }
    kwargs.update(overrides)

    return await build(
        settings,
        as_of=AS_OF,
        clickhouse=clickhouse,
        store=store,
        factorizer=factorizer or FakeALS(),
        log_to_mlflow=False,
        **kwargs,
    )


# --- matrix construction --------------------------------------------------------------


def test_items_touched_by_too_few_users_are_dropped():
    # A one-user item yields a factor fitted to that user alone; ALS then recommends it back
    # to them and to nobody else, which is memorisation dressed as collaboration.
    rows = [interaction("u1", "shared"), interaction("u2", "shared"), interaction("u1", "private")]

    data = Interactions(rows, min_user=1, min_item=2)

    assert data.article_ids == ["shared"]


def test_users_below_the_interaction_threshold_are_dropped():
    rows = [
        interaction("busy", "a1"),
        interaction("busy", "a2"),
        interaction("quiet", "a1"),
        interaction("other", "a1"),
        interaction("other", "a2"),
    ]

    data = Interactions(rows, min_user=2, min_item=2)

    assert "quiet" not in data.user_ids


def test_non_positive_affinity_is_ignored():
    rows = [interaction("u1", "a1", 3.0), interaction("u1", "a2", -4.0), interaction("u2", "a1", 1.0)]

    data = Interactions(rows, min_user=1, min_item=1)

    assert "a2" not in data.article_ids


def test_the_matrix_shape_matches_the_index_maps():
    data = Interactions(dense_rows(3, 4), min_user=1, min_item=1)

    matrix = data.matrix(alpha=40.0)

    assert matrix.shape == (len(data.user_ids), data.item_count) == (3, 4)
    assert matrix.nnz == data.nnz == 12


def test_confidence_grows_sublinearly_with_affinity():
    # log1p weighting: one click versus two matters far more than forty versus forty-one.
    data = Interactions(
        [interaction("u1", "a1", 1.0), interaction("u1", "a2", 40.0), interaction("u2", "a1", 1.0),
         interaction("u2", "a2", 1.0)],
        min_user=1,
        min_item=1,
    )

    dense = data.matrix(alpha=40.0).toarray()
    row = dense[data.user_ids.index("u1")]
    low, high = row[data.article_ids.index("a1")], row[data.article_ids.index("a2")]

    assert high > low
    assert high < low * 40


# --- the job --------------------------------------------------------------------------


async def test_candidates_are_written_as_a_list_best_first(settings, store, fake_redis):
    clickhouse = FakeClickHouse(dense_rows(3, 4))
    factorizer = FakeALS({0: [2, 0, 1]})

    assert await run(settings, store, clickhouse, factorizer) == EXIT_OK

    data = Interactions(dense_rows(3, 4), 2, 2)
    key = f"{ALS_CANDIDATES_PREFIX}{data.user_ids[0]}"
    expected = [data.article_ids[i].encode() for i in (2, 0, 1)]

    # Same shape as rec:trending:top100 — RedisRecommendationStore reads both with ListRangeAsync.
    assert fake_redis.lists[key] == expected


async def test_candidates_get_a_ttl(settings, store, fake_redis):
    await run(settings, store, FakeClickHouse(dense_rows(3, 4)))

    key = next(k for k in fake_redis.ttls if k.startswith(ALS_CANDIDATES_PREFIX))

    assert fake_redis.ttls[key] == 604800


async def test_freshness_is_stamped(settings, store, fake_redis):
    await run(settings, store, FakeClickHouse(dense_rows(3, 4)))

    assert freshness_key("user_als_candidates") in fake_redis.strings


async def test_the_model_is_fitted_on_the_filtered_matrix(settings, store):
    factorizer = FakeALS()

    await run(settings, store, FakeClickHouse(dense_rows(3, 4)), factorizer)

    assert factorizer.fitted_shape == (3, 4)


async def test_too_few_users_trains_nothing(settings, store, fake_redis):
    outcome = await run(settings, store, FakeClickHouse(dense_rows(2, 3)), min_users=5)

    assert outcome == EXIT_NOTHING_TO_DO
    assert not fake_redis.lists


async def test_a_failed_run_leaves_yesterdays_candidates_serving(settings, store, fake_redis):
    await run(settings, store, FakeClickHouse(dense_rows(3, 4)))
    before = dict(fake_redis.lists)

    outcome = await run(settings, store, FakeClickHouse([]), min_users=2)

    assert outcome == EXIT_NOTHING_TO_DO
    assert fake_redis.lists == before


async def test_a_clickhouse_outage_is_infra(settings, store):
    from explained_ml.clickhouse import ClickHouseError

    outcome = await run(settings, store, FakeClickHouse(error=ClickHouseError("down")))

    assert outcome == EXIT_INFRA


async def test_a_rerun_replaces_a_users_list(settings, store, fake_redis):
    data = Interactions(dense_rows(3, 4), 2, 2)
    key = f"{ALS_CANDIDATES_PREFIX}{data.user_ids[0]}"

    await run(settings, store, FakeClickHouse(dense_rows(3, 4)), FakeALS({0: [0, 1, 2]}))
    await run(settings, store, FakeClickHouse(dense_rows(3, 4)), FakeALS({0: [3]}))

    assert fake_redis.lists[key] == [data.article_ids[3].encode()]


async def test_out_of_range_indices_are_ignored(settings, store, fake_redis):
    # implicit returns padded arrays when a user has fewer than N recommendable items.
    factorizer = FakeALS({0: [0, 99, 1]})

    await run(settings, store, FakeClickHouse(dense_rows(3, 4)), factorizer)

    data = Interactions(dense_rows(3, 4), 2, 2)
    key = f"{ALS_CANDIDATES_PREFIX}{data.user_ids[0]}"

    assert fake_redis.lists[key] == [data.article_ids[0].encode(), data.article_ids[1].encode()]
