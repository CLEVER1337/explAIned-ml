from datetime import UTC, datetime

import pytest

from explained_ml.config import Settings
from explained_ml.jobs.update_trending import build
from explained_ml.logs import EXIT_INFRA, EXIT_NOTHING_TO_DO, EXIT_OK
from explained_ml.redis_store import TRENDING_KEY, RecommendationStore, freshness_key

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


@pytest.fixture
def settings():
    return Settings(trending_ttl_seconds=3600)


@pytest.fixture
def store(fake_redis):
    return RecommendationStore("redis://unused", 8, client=fake_redis)


async def run(settings, store, clickhouse, min_reach=2):
    return await build(
        settings,
        window_hours=24,
        half_life_hours=12.0,
        limit=100,
        min_reach=min_reach,
        as_of=AS_OF,
        clickhouse=clickhouse,
        store=store,
    )


async def test_ranked_ids_are_written_in_query_order(settings, store, fake_redis):
    clickhouse = FakeClickHouse(
        [
            {"article_id": "a1", "score": 9.5, "reach": 4},
            {"article_id": "a2", "score": 3.0, "reach": 2},
        ]
    )

    assert await run(settings, store, clickhouse) == EXIT_OK
    assert fake_redis.lists[TRENDING_KEY] == [b"a1", b"a2"]


async def test_the_key_gets_the_configured_ttl(settings, store, fake_redis):
    await run(settings, store, FakeClickHouse([{"article_id": "a1", "score": 1.0, "reach": 2}]))

    assert fake_redis.ttls[TRENDING_KEY] == 3600


async def test_freshness_is_stamped(settings, store, fake_redis):
    await run(settings, store, FakeClickHouse([{"article_id": "a1", "score": 1.0, "reach": 2}]))

    assert freshness_key("trending") in fake_redis.strings


async def test_an_empty_aggregate_keeps_the_previous_list(settings, store, fake_redis):
    # The failure this guards against is invisible: the feed still answers 200, but every user
    # silently drops from the `trending` rung to `recent`.
    await run(settings, store, FakeClickHouse([{"article_id": "a1", "score": 1.0, "reach": 2}]))

    assert await run(settings, store, FakeClickHouse([])) == EXIT_NOTHING_TO_DO
    assert fake_redis.lists[TRENDING_KEY] == [b"a1"]


async def test_a_clickhouse_failure_keeps_the_previous_list(settings, store, fake_redis):
    from explained_ml.clickhouse import ClickHouseError

    await run(settings, store, FakeClickHouse([{"article_id": "a1", "score": 1.0, "reach": 2}]))

    outcome = await run(settings, store, FakeClickHouse(error=ClickHouseError("down")))

    assert outcome == EXIT_INFRA
    assert fake_redis.lists[TRENDING_KEY] == [b"a1"]


async def test_a_rerun_replaces_rather_than_appends(settings, store, fake_redis):
    await run(settings, store, FakeClickHouse([{"article_id": "a1", "score": 1.0, "reach": 2}]))
    await run(settings, store, FakeClickHouse([{"article_id": "a2", "score": 1.0, "reach": 2}]))

    assert fake_redis.lists[TRENDING_KEY] == [b"a2"]


async def test_rows_without_an_id_are_dropped(settings, store, fake_redis):
    clickhouse = FakeClickHouse(
        [{"article_id": "", "score": 9.0, "reach": 4}, {"article_id": "a2", "score": 1.0, "reach": 2}]
    )

    await run(settings, store, clickhouse)

    assert fake_redis.lists[TRENDING_KEY] == [b"a2"]


async def test_query_parameters_are_sent_server_side(settings, store):
    clickhouse = FakeClickHouse([{"article_id": "a1", "score": 1.0, "reach": 2}])

    await run(settings, store, clickhouse, min_reach=3)

    sql, params = clickhouse.queries[0]

    assert params["min_reach"] == 3
    assert params["as_of"] == AS_OF
    # The values come from CLI flags; interpolating them would let a flag rewrite the query.
    # Every one of them must still be a placeholder in the SQL that gets sent.
    for placeholder in ("{min_reach:UInt32}", "{limit:UInt32}", "{window_hours:UInt32}"):
        assert placeholder in sql


async def test_the_aggregate_uses_final(settings, store):
    clickhouse = FakeClickHouse([{"article_id": "a1", "score": 1.0, "reach": 2}])

    await run(settings, store, clickhouse)

    # ReplacingMergeTree only collapses retried inserts on merge; without FINAL popularity is
    # whatever the merge scheduler happened to have done.
    assert "FINAL" in clickhouse.queries[0][0]


async def test_the_window_is_bounded_on_both_sides(settings, store):
    clickhouse = FakeClickHouse([{"article_id": "a1", "score": 1.0, "reach": 2}])

    await run(settings, store, clickhouse)
    sql = clickhouse.queries[0][0]

    # The upper bound is what makes the same SQL reusable for historical training rows.
    assert "occurred_at <  {as_of:DateTime64(3)}" in sql
    assert "occurred_at >= {as_of:DateTime64(3)}" in sql
