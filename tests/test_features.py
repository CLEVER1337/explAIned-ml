from datetime import UTC, datetime

import numpy as np
import pytest

from explained_ml.articles import Article
from explained_ml.codec import encode_vector
from explained_ml.config import Settings
from explained_ml.features import (
    MAX_AUTHORS_PER_USER,
    OfflineFeatureBuilder,
    RedisFeatureLoader,
    contexts_from_rows,
)
from explained_ml.jobs.build_features import build
from explained_ml.logs import EXIT_NOTHING_TO_DO, EXIT_OK
from explained_ml.redis_store import (
    ARTICLE_EMBEDDING_PREFIX,
    USER_EMBEDDING_PREFIX,
    RecommendationStore,
    freshness_key,
)
from feature_store.compute import build_matrix
from feature_store.context import RequestContext
from feature_store.schema import (
    A_CLICKS_24H,
    A_PUBLISHED_TS,
    A_WORD_COUNT,
    U_AUTHOR_AFFINITY_PREFIX,
    U_EVENTS_7D,
)

DIM = 4
AS_OF = datetime(2026, 8, 17, 12, 0, tzinfo=UTC)
REQUEST = RequestContext(now=AS_OF)


class FakeClickHouse:
    """Answers by query shape — each SQL builder produces a recognisable SELECT list."""

    def __init__(self, article_rows=None, user_rows=None, author_rows=None) -> None:
        self.article_rows = article_rows or []
        self.user_rows = user_rows or []
        self.author_rows = author_rows or []
        self.queries: list[tuple[str, dict]] = []

    async def query_rows(self, sql: str, parameters: dict | None = None):
        self.queries.append((sql, parameters or {}))

        if "positives" in sql:
            return self.author_rows
        if "active_hours" in sql:
            return self.user_rows
        return self.article_rows

    async def close(self):
        pass


class FakeArticles:
    def __init__(self, articles: list[Article]) -> None:
        self.articles = articles

    async def iter_published(self, page_size: int = 100):
        for article in self.articles:
            yield article

    async def close(self):
        pass


def article(article_id: str, author: str = "author-1", words: int = 300) -> Article:
    return Article(
        id=article_id,
        title="T",
        content="C",
        description="D",
        tags="ml",
        author_id=author,
        published_at=AS_OF,
        updated_at=AS_OF,
        word_count=words,
    )


@pytest.fixture
def settings():
    return Settings(embedding_dim=DIM, features_ttl_seconds=3600)


@pytest.fixture
def store(fake_redis):
    return RecommendationStore("redis://unused", DIM, client=fake_redis)


# --- offline builder ------------------------------------------------------------------


async def test_article_rows_combine_catalog_facts_and_clickhouse_counts():
    clickhouse = FakeClickHouse(
        article_rows=[{"article_id": "a1", "clicks": 7, "reads": 3, "unique_users": 2}]
    )
    builder = OfflineFeatureBuilder(clickhouse, FakeArticles([article("a1", words=300)]), "t")

    rows = await builder.build(AS_OF)

    assert rows.articles["a1"][A_WORD_COUNT] == 300.0
    assert rows.articles["a1"][A_CLICKS_24H] == 7.0
    assert rows.articles["a1"][A_PUBLISHED_TS] == AS_OF.timestamp()


async def test_articles_with_no_events_get_explicit_zeros():
    # Absent is not the same as zero downstream: a missing key would make the feature fall back
    # to its default, which is the same number here but not by construction.
    builder = OfflineFeatureBuilder(FakeClickHouse(), FakeArticles([article("a1")]), "t")

    rows = await builder.build(AS_OF)

    assert rows.articles["a1"][A_CLICKS_24H] == 0.0


async def test_counts_for_unknown_articles_are_dropped():
    clickhouse = FakeClickHouse(article_rows=[{"article_id": "archived", "clicks": 99}])
    builder = OfflineFeatureBuilder(clickhouse, FakeArticles([article("a1")]), "t")

    rows = await builder.build(AS_OF)

    assert "archived" not in rows.articles


async def test_author_affinity_is_folded_from_article_interactions():
    clickhouse = FakeClickHouse(
        user_rows=[{"user_id": "u1", "events": 10}],
        author_rows=[
            {"user_id": "u1", "article_id": "a1", "positives": 3},
            {"user_id": "u1", "article_id": "a2", "positives": 4},
        ],
    )
    catalog = [article("a1", author="author-1"), article("a2", author="author-1")]
    builder = OfflineFeatureBuilder(clickhouse, FakeArticles(catalog), "t")

    rows = await builder.build(AS_OF)

    # Both articles share an author, so the counts add up rather than competing.
    assert rows.users["u1"][f"{U_AUTHOR_AFFINITY_PREFIX}author-1"] == 7.0


async def test_only_the_strongest_authors_are_kept():
    author_rows = [
        {"user_id": "u1", "article_id": f"a{i}", "positives": i} for i in range(MAX_AUTHORS_PER_USER + 10)
    ]
    catalog = [article(f"a{i}", author=f"author-{i}") for i in range(MAX_AUTHORS_PER_USER + 10)]
    clickhouse = FakeClickHouse(user_rows=[{"user_id": "u1"}], author_rows=author_rows)

    rows = await OfflineFeatureBuilder(clickhouse, FakeArticles(catalog), "t").build(AS_OF)
    kept = [k for k in rows.users["u1"] if k.startswith(U_AUTHOR_AFFINITY_PREFIX)]

    assert len(kept) == MAX_AUTHORS_PER_USER


async def test_every_query_is_bounded_by_as_of():
    clickhouse = FakeClickHouse()
    await OfflineFeatureBuilder(clickhouse, FakeArticles([article("a1")]), "t").build(AS_OF)

    for sql, params in clickhouse.queries:
        assert params["as_of"] == AS_OF
        assert "FINAL" in sql


# --- online loader --------------------------------------------------------------------


async def test_online_loader_returns_a_context_per_candidate(settings, store, fake_redis):
    await store.set_article_features({"a1": {A_CLICKS_24H: 5.0}}, 60)

    user, articles = await RedisFeatureLoader(store).load("u1", ["a1", "a2"])

    assert [a.article_id for a in articles] == ["a1", "a2"]
    assert articles[0].features[A_CLICKS_24H] == 5.0


async def test_a_candidate_without_a_row_is_scored_not_dropped(settings, store):
    # The orchestrator asked for these ids and expects all of them back, ranked.
    _, articles = await RedisFeatureLoader(store).load("u1", ["missing"])

    assert len(articles) == 1
    assert articles[0].features == {}


async def test_embeddings_are_attached_when_present(settings, store, fake_redis):
    vector = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    fake_redis.strings[f"{ARTICLE_EMBEDDING_PREFIX}a1"] = encode_vector(vector, DIM)
    fake_redis.strings[f"{USER_EMBEDDING_PREFIX}u1"] = encode_vector(vector, DIM)

    user, articles = await RedisFeatureLoader(store).load("u1", ["a1"])

    assert user.embedding is not None
    assert np.allclose(articles[0].embedding, vector)


async def test_a_cold_user_loads_without_an_embedding(settings, store):
    user, _ = await RedisFeatureLoader(store).load("nobody", ["a1"])

    assert user.embedding is None
    assert user.features == {}


# --- the two paths must agree ---------------------------------------------------------


async def test_the_online_row_equals_the_offline_row(settings, store, fake_redis):
    """The guarantee the whole split rests on: a row that goes through Redis is unchanged.

    Anything else means training saw numbers the service will never see, which is the failure
    mode that looks excellent offline and does nothing in production.
    """
    clickhouse = FakeClickHouse(
        article_rows=[{"article_id": "a1", "clicks": 7, "reads": 3, "likes": 2, "unique_users": 5}],
        user_rows=[{"user_id": "u1", "events": 40, "reads": 20, "likes": 4, "distinct_articles": 15}],
        author_rows=[{"user_id": "u1", "article_id": "a1", "positives": 6}],
    )
    catalog = [article("a1", author="author-9", words=420)]
    rows = await OfflineFeatureBuilder(clickhouse, FakeArticles(catalog), "t").build(AS_OF)

    vector = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
    fake_redis.strings[f"{ARTICLE_EMBEDDING_PREFIX}a1"] = encode_vector(vector, DIM)
    fake_redis.strings[f"{USER_EMBEDDING_PREFIX}u1"] = encode_vector(vector, DIM)

    offline_user, offline_articles = contexts_from_rows(
        "u1", ["a1"], rows, vector, {"a1": vector}
    )
    offline_matrix = build_matrix(offline_user, offline_articles, REQUEST)

    await store.set_article_features(rows.articles, 60)
    await store.set_user_features(rows.users, 60)
    online_user, online_articles = await RedisFeatureLoader(store).load(
        "u1", ["a1"], authors=rows.authors
    )
    online_matrix = build_matrix(online_user, online_articles, REQUEST)

    assert np.array_equal(offline_matrix, online_matrix)


async def test_author_affinity_survives_the_round_trip(settings, store):
    # The per-author keys live inside the user row; a serializer that flattened or sorted them
    # differently would silently zero this feature online only.
    rows_users = {"u1": {U_EVENTS_7D: 10.0, f"{U_AUTHOR_AFFINITY_PREFIX}author-9": 6.0}}
    await store.set_user_features(rows_users, 60)

    user, _ = await RedisFeatureLoader(store).load("u1", [])

    assert user.features[f"{U_AUTHOR_AFFINITY_PREFIX}author-9"] == 6.0


# --- the job --------------------------------------------------------------------------


async def test_the_job_writes_both_row_kinds(settings, store, fake_redis):
    clickhouse = FakeClickHouse(user_rows=[{"user_id": "u1", "events": 3}])
    articles = FakeArticles([article("a1")])

    outcome = await build(settings, as_of=AS_OF, clickhouse=clickhouse, articles=articles, store=store)

    assert outcome == EXIT_OK
    assert "rec:article_features:a1" in fake_redis.strings
    assert "rec:user_features:u1" in fake_redis.strings
    assert freshness_key("features") in fake_redis.strings


async def test_rows_carry_a_ttl(settings, store, fake_redis):
    # A dead job must let rows expire into neutral values rather than serve month-old
    # popularity as if it were current.
    await build(
        settings,
        as_of=AS_OF,
        clickhouse=FakeClickHouse(),
        articles=FakeArticles([article("a1")]),
        store=store,
    )

    assert fake_redis.ttls["rec:article_features:a1"] == 3600


async def test_an_empty_catalog_leaves_existing_rows_alone(settings, store, fake_redis):
    await build(
        settings,
        as_of=AS_OF,
        clickhouse=FakeClickHouse(),
        articles=FakeArticles([article("a1")]),
        store=store,
    )

    outcome = await build(
        settings, as_of=AS_OF, clickhouse=FakeClickHouse(), articles=FakeArticles([]), store=store
    )

    assert outcome == EXIT_NOTHING_TO_DO
    assert "rec:article_features:a1" in fake_redis.strings


async def test_dry_run_writes_nothing(settings, store, fake_redis):
    outcome = await build(
        settings,
        dry_run=True,
        as_of=AS_OF,
        clickhouse=FakeClickHouse(),
        articles=FakeArticles([article("a1")]),
        store=store,
    )

    assert outcome == EXIT_OK
    assert "rec:article_features:a1" not in fake_redis.strings
