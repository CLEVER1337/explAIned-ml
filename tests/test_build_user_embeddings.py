from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from explained_ml.codec import encode_vector
from explained_ml.config import Settings
from explained_ml.jobs.build_user_embeddings import build
from explained_ml.logs import EXIT_INFRA, EXIT_NOTHING_TO_DO, EXIT_OK
from explained_ml.redis_store import (
    ARTICLE_EMBEDDING_PREFIX,
    USER_EMBEDDING_PREFIX,
    RecommendationStore,
    freshness_key,
)

DIM = 4
AS_OF = datetime(2026, 8, 17, 12, 0, tzinfo=UTC)


class FakeClickHouse:
    def __init__(self, rows=None, error: Exception | None = None) -> None:
        self.rows = rows or []
        self.error = error
        self.queries: list[tuple[str, dict]] = []

    async def query_rows(self, sql: str, parameters: dict | None = None):
        self.queries.append((sql, parameters or {}))
        if self.error:
            raise self.error
        return self.rows

    async def close(self):
        pass


def affinity(user: str, article: str, value: float, days_ago: float = 0.0) -> dict:
    return {
        "user_id": user,
        "article_id": article,
        "affinity": value,
        "last_at": (AS_OF - timedelta(days=days_ago)).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
    }


@pytest.fixture
def settings():
    return Settings(embedding_dim=DIM)


@pytest.fixture
def store(fake_redis):
    return RecommendationStore("redis://unused", DIM, client=fake_redis)


def seed_vectors(fake_redis, vectors: dict[str, list[float]]) -> None:
    for article_id, values in vectors.items():
        fake_redis.strings[f"{ARTICLE_EMBEDDING_PREFIX}{article_id}"] = encode_vector(
            np.array(values, dtype=np.float32), DIM
        )


async def run(settings, store, clickhouse, min_articles=2, half_life_days=7.0):
    return await build(
        settings,
        window_days=30,
        half_life_days=half_life_days,
        per_user=50,
        min_articles=min_articles,
        as_of=AS_OF,
        clickhouse=clickhouse,
        store=store,
    )


async def test_a_centroid_is_written_per_user(settings, store, fake_redis):
    seed_vectors(fake_redis, {"a1": [1, 0, 0, 0], "a2": [0, 1, 0, 0]})
    clickhouse = FakeClickHouse([affinity("u1", "a1", 3.0), affinity("u1", "a2", 3.0)])

    assert await run(settings, store, clickhouse) == EXIT_OK

    raw = fake_redis.strings[f"{USER_EMBEDDING_PREFIX}u1"]
    vector = np.frombuffer(raw, dtype="<f4")

    assert len(raw) == DIM * 4
    assert np.allclose(vector, np.array([0.5**0.5, 0.5**0.5, 0, 0]), atol=1e-6)


async def test_the_centroid_is_unit_length(settings, store, fake_redis):
    seed_vectors(fake_redis, {"a1": [3, 4, 0, 0], "a2": [0, 0, 5, 0]})
    clickhouse = FakeClickHouse([affinity("u1", "a1", 2.0), affinity("u1", "a2", 1.0)])

    await run(settings, store, clickhouse)
    vector = np.frombuffer(fake_redis.strings[f"{USER_EMBEDDING_PREFIX}u1"], dtype="<f4")

    assert np.isclose(np.linalg.norm(vector), 1.0, atol=1e-6)


async def test_recent_interactions_weigh_more(settings, store, fake_redis):
    seed_vectors(fake_redis, {"old": [1, 0, 0, 0], "new": [0, 1, 0, 0]})
    clickhouse = FakeClickHouse(
        [affinity("u1", "old", 3.0, days_ago=28.0), affinity("u1", "new", 3.0, days_ago=0.0)]
    )

    await run(settings, store, clickhouse, half_life_days=7.0)
    vector = np.frombuffer(fake_redis.strings[f"{USER_EMBEDDING_PREFIX}u1"], dtype="<f4")

    # Four half-lives of decay: the recent article should dominate.
    assert vector[1] > vector[0] * 5


async def test_stronger_affinity_weighs_more(settings, store, fake_redis):
    seed_vectors(fake_redis, {"liked": [1, 0, 0, 0], "clicked": [0, 1, 0, 0]})
    clickhouse = FakeClickHouse([affinity("u1", "liked", 8.0), affinity("u1", "clicked", 1.0)])

    await run(settings, store, clickhouse)
    vector = np.frombuffer(fake_redis.strings[f"{USER_EMBEDDING_PREFIX}u1"], dtype="<f4")

    assert vector[0] > vector[1]


async def test_a_user_below_the_threshold_stays_cold(settings, store, fake_redis):
    # A centroid from one click is noise. Leaving the key absent makes FAISS answer 204, which
    # the orchestrator records as Disabled — honest, and explicitly not degradation.
    seed_vectors(fake_redis, {"a1": [1, 0, 0, 0]})
    clickhouse = FakeClickHouse([affinity("u1", "a1", 3.0)])

    outcome = await run(settings, store, clickhouse, min_articles=2)

    assert outcome == EXIT_NOTHING_TO_DO
    assert f"{USER_EMBEDDING_PREFIX}u1" not in fake_redis.strings


async def test_articles_without_an_embedding_are_ignored(settings, store, fake_redis):
    seed_vectors(fake_redis, {"a1": [1, 0, 0, 0], "a2": [0, 1, 0, 0]})
    clickhouse = FakeClickHouse(
        [affinity("u1", "a1", 3.0), affinity("u1", "a2", 3.0), affinity("u1", "missing", 9.0)]
    )

    assert await run(settings, store, clickhouse) == EXIT_OK


async def test_no_interactions_writes_nothing(settings, store, fake_redis):
    assert await run(settings, store, FakeClickHouse([])) == EXIT_NOTHING_TO_DO
    assert not any(k.startswith(USER_EMBEDDING_PREFIX) for k in fake_redis.strings)


async def test_an_existing_embedding_survives_an_empty_run(settings, store, fake_redis):
    seed_vectors(fake_redis, {"a1": [1, 0, 0, 0], "a2": [0, 1, 0, 0]})
    await run(settings, store, FakeClickHouse([affinity("u1", "a1", 1.0), affinity("u1", "a2", 1.0)]))

    await run(settings, store, FakeClickHouse([]))

    assert f"{USER_EMBEDDING_PREFIX}u1" in fake_redis.strings


async def test_a_clickhouse_outage_is_infra(settings, store):
    from explained_ml.clickhouse import ClickHouseError

    outcome = await run(settings, store, FakeClickHouse(error=ClickHouseError("down")))

    assert outcome == EXIT_INFRA


async def test_articles_with_no_embeddings_at_all_stops_early(settings, store):
    clickhouse = FakeClickHouse([affinity("u1", "a1", 3.0), affinity("u1", "a2", 3.0)])

    assert await run(settings, store, clickhouse) == EXIT_NOTHING_TO_DO


async def test_a_cancelling_history_leaves_the_user_cold(settings, store, fake_redis):
    # Opposite vectors in equal measure produce a zero centroid, which would make every cosine
    # exactly 0 — strictly worse than no embedding, because it silences the 204 path.
    seed_vectors(fake_redis, {"a1": [1, 0, 0, 0], "a2": [-1, 0, 0, 0]})
    clickhouse = FakeClickHouse([affinity("u1", "a1", 3.0), affinity("u1", "a2", 3.0)])

    outcome = await run(settings, store, clickhouse)

    assert outcome == EXIT_NOTHING_TO_DO
    assert f"{USER_EMBEDDING_PREFIX}u1" not in fake_redis.strings


async def test_freshness_is_stamped(settings, store, fake_redis):
    seed_vectors(fake_redis, {"a1": [1, 0, 0, 0], "a2": [0, 1, 0, 0]})
    await run(settings, store, FakeClickHouse([affinity("u1", "a1", 1.0), affinity("u1", "a2", 1.0)]))

    assert freshness_key("user_embedding") in fake_redis.strings


async def test_only_user_narrows_the_run(settings, store, fake_redis):
    seed_vectors(fake_redis, {"a1": [1, 0, 0, 0], "a2": [0, 1, 0, 0]})
    rows = [
        affinity("u1", "a1", 1.0),
        affinity("u1", "a2", 1.0),
        affinity("u2", "a1", 1.0),
        affinity("u2", "a2", 1.0),
    ]

    await build(
        settings,
        window_days=30,
        half_life_days=7.0,
        per_user=50,
        min_articles=2,
        only_users=["u1"],
        as_of=AS_OF,
        clickhouse=FakeClickHouse(rows),
        store=store,
    )

    assert f"{USER_EMBEDDING_PREFIX}u1" in fake_redis.strings
    assert f"{USER_EMBEDDING_PREFIX}u2" not in fake_redis.strings


async def test_the_query_is_bounded_by_as_of(settings, store):
    clickhouse = FakeClickHouse([])
    await run(settings, store, clickhouse)

    sql, params = clickhouse.queries[0]

    assert params["as_of"] == AS_OF
    assert "FINAL" in sql
