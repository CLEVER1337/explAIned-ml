import numpy as np
import pytest

from explained_ml.codec import encode_vector
from explained_ml.redis_store import (
    ALS_CANDIDATES_PREFIX,
    ARTICLE_EMBEDDING_PREFIX,
    ARTICLE_FEATURES_PREFIX,
    FINGERPRINTS_KEY,
    META_PREFIX,
    TRENDING_KEY,
    USER_EMBEDDING_PREFIX,
    RecommendationStore,
    decode_row,
    encode_row,
    freshness_key,
)

DIM = 8


@pytest.fixture
def store(fake_redis):
    return RecommendationStore("redis://unused", DIM, client=fake_redis)


def vector(seed: int) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal(DIM).astype(np.float32)


def test_key_names_match_the_cross_boundary_contract():
    # These four strings are read by the C# orchestrator or by explAIned-faiss. Renaming one
    # is an outage that no compiler will catch, so it is pinned here.
    assert TRENDING_KEY == "rec:trending:top100"
    assert ARTICLE_EMBEDDING_PREFIX == "rec:article_embedding:"
    assert USER_EMBEDDING_PREFIX == "rec:user_embedding:"
    assert ALS_CANDIDATES_PREFIX == "rec:user_als_candidates:"


@pytest.mark.parametrize(
    "key",
    [
        FINGERPRINTS_KEY,
        freshness_key("article_embedding"),
        freshness_key("user_embedding"),
        freshness_key("trending"),
        freshness_key("features"),
        freshness_key("user_als_candidates"),
    ],
)
def test_no_bookkeeping_key_lands_in_a_scanned_namespace(key):
    # explAIned-faiss builds its index by SCANning `rec:article_embedding:*` and decoding
    # everything it finds as a 384-float vector. A freshness stamp parked in that namespace is
    # handed to decode_vector as if it were data.
    assert not key.startswith(ARTICLE_EMBEDDING_PREFIX)
    assert not key.startswith(USER_EMBEDDING_PREFIX)
    assert key.startswith(META_PREFIX)


async def test_article_embeddings_round_trip(store, fake_redis):
    written = await store.set_article_embeddings({"a1": vector(1), "a2": vector(2)})

    assert written == 2
    assert f"{ARTICLE_EMBEDDING_PREFIX}a1" in fake_redis.strings

    read_back = await store.get_article_embeddings(["a1", "a2"])

    assert np.allclose(read_back["a1"], vector(1))


async def test_embeddings_are_stored_in_the_faiss_wire_format(store, fake_redis):
    await store.set_article_embeddings({"a1": vector(1)})

    raw = fake_redis.strings[f"{ARTICLE_EMBEDDING_PREFIX}a1"]

    assert len(raw) == DIM * 4
    assert raw == encode_vector(vector(1), DIM)


async def test_missing_embeddings_are_skipped_not_faked(store):
    await store.set_article_embeddings({"a1": vector(1)})

    result = await store.get_article_embeddings(["a1", "absent"])

    assert set(result) == {"a1"}


async def test_a_malformed_embedding_is_treated_as_absent(store, fake_redis):
    fake_redis.strings[f"{ARTICLE_EMBEDDING_PREFIX}bad"] = b"\x00\x01\x02"

    assert await store.get_article_embeddings(["bad"]) == {}


async def test_iterating_ids_uses_scan_not_keys(store, fake_redis):
    await store.set_article_embeddings({"a1": vector(1), "a2": vector(2)})

    assert not hasattr(fake_redis, "keys"), "the fake deliberately has no KEYS to call"

    found = {article_id async for article_id in store.iter_embedded_article_ids()}

    assert found == {"a1", "a2"}


async def test_replace_trending_writes_the_list_in_order(store, fake_redis):
    await store.replace_trending(["a", "b", "c"], ttl_seconds=3600)

    assert fake_redis.lists[TRENDING_KEY] == [b"a", b"b", b"c"]
    assert fake_redis.ttls[TRENDING_KEY] == 3600


async def test_replace_trending_deletes_before_pushing(store, fake_redis):
    await store.replace_trending(["a", "b"], ttl_seconds=60)
    await store.replace_trending(["c"], ttl_seconds=60)

    # Without the DEL the second run would append and the list would grow without bound.
    assert fake_redis.lists[TRENDING_KEY] == [b"c"]


async def test_replace_trending_ignores_an_empty_result(store, fake_redis):
    await store.replace_trending(["a", "b"], ttl_seconds=60)
    await store.replace_trending([], ttl_seconds=60)

    # An empty aggregate must never blank the key: every user would drop to the `recent` rung.
    assert fake_redis.lists[TRENDING_KEY] == [b"a", b"b"]


async def test_replace_trending_stamps_freshness(store, fake_redis):
    await store.replace_trending(["a"], ttl_seconds=60)

    assert freshness_key("trending") in fake_redis.strings


async def test_als_candidates_are_a_list_best_first(store, fake_redis):
    written = await store.set_als_candidates({"u1": ["a", "b"], "u2": ["c"]}, ttl_seconds=600)

    assert written == 2
    # Same shape as trending, because RedisRecommendationStore reads both with ListRangeAsync.
    assert fake_redis.lists[f"{ALS_CANDIDATES_PREFIX}u1"] == [b"a", b"b"]
    assert fake_redis.ttls[f"{ALS_CANDIDATES_PREFIX}u1"] == 600


async def test_users_without_candidates_are_skipped(store, fake_redis):
    await store.set_als_candidates({"u1": [], "u2": ["a"]}, ttl_seconds=600)

    assert f"{ALS_CANDIDATES_PREFIX}u1" not in fake_redis.lists


def test_feature_rows_round_trip_through_one_serializer():
    encoded = encode_row({"clicks_24h": 3, "author_aff:x": 1.5})

    assert decode_row(encoded) == {"clicks_24h": 3.0, "author_aff:x": 1.5}


async def test_article_features_are_fetched_in_one_mget(store, fake_redis):
    await store.set_article_features({"a1": {"clicks_24h": 2}, "a2": {"clicks_24h": 9}}, 3600)

    calls: list[int] = []
    original = fake_redis.mget

    async def counting_mget(keys):
        calls.append(len(keys))
        return await original(keys)

    fake_redis.mget = counting_mget
    rows = await store.get_article_features(["a1", "a2"])

    # One MGET of N keys, not N round trips — this is what keeps /rank inside 120 ms at 200
    # candidates.
    assert calls == [2]
    assert rows["a2"]["clicks_24h"] == 9.0


async def test_a_malformed_feature_row_is_treated_as_absent(store, fake_redis):
    fake_redis.strings[f"{ARTICLE_FEATURES_PREFIX}bad"] = b"not json"

    assert await store.get_article_features(["bad"]) == {}


async def test_user_features_missing_is_an_empty_row_not_an_error(store):
    assert await store.get_user_features("nobody") == {}


async def test_freshness_round_trips(store):
    await store.mark_fresh("trending")

    assert await store.get_freshness("trending") is not None
    assert await store.get_freshness("never_run") is None
