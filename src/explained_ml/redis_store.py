"""Redis access for the ML side.

`rec:` is the recommendation-loop prefix (see explAInedRecommendationService/CONTRACT.md);
`explAIned_` belongs to the .NET `IDistributedCache` path and must not be used here.

This repository is the *writer* for every key below except `rec:user_viewed:{uid}`, which the
.NET consumer owns and we only read.
"""

import json
import logging
from collections.abc import AsyncIterator, Iterable, Mapping
from datetime import UTC, datetime

import numpy as np
from redis.asyncio import Redis

from .codec import VectorFormatError, decode_vector, encode_vector

ARTICLE_EMBEDDING_PREFIX = "rec:article_embedding:"
USER_EMBEDDING_PREFIX = "rec:user_embedding:"
ARTICLE_FEATURES_PREFIX = "rec:article_features:"
USER_FEATURES_PREFIX = "rec:user_features:"
ALS_CANDIDATES_PREFIX = "rec:user_als_candidates:"
VIEWED_PREFIX = "rec:user_viewed:"
TRENDING_KEY = "rec:trending:top100"

# All bookkeeping this repository owns lives under one prefix that no other component reads.
#
# This is not decoration. explAIned-faiss builds its index by SCANning `rec:article_embedding:*`
# and MGETing whatever it finds, so a freshness stamp written as `rec:article_embedding:updated_at`
# would be handed to `decode_vector` as if it were a 384-float vector. Metadata never lives
# inside a namespace that something else scans.
META_PREFIX = "rec:meta:"
FINGERPRINTS_KEY = f"{META_PREFIX}article_fingerprints"

logger = logging.getLogger(__name__)


def freshness_key(name: str) -> str:
    """`rec:meta:trending:updated_at` and friends.

    Written by us on purpose: a scheduler's own UI is invisible to Prometheus and to the
    C# orchestrator, so staleness has to live in the same store as the data.
    """
    return f"{META_PREFIX}{name}:updated_at"


class RecommendationStore:
    def __init__(self, redis_url: str, dim: int, client: Redis | None = None) -> None:
        self._dim = dim
        self._redis = client if client is not None else Redis.from_url(redis_url)

    @property
    def redis(self) -> Redis:
        return self._redis

    async def close(self) -> None:
        await self._redis.aclose()

    # --- embeddings -------------------------------------------------------------------

    async def set_article_embeddings(self, vectors: Mapping[str, np.ndarray]) -> int:
        return await self._set_vectors(ARTICLE_EMBEDDING_PREFIX, vectors)

    async def set_user_embeddings(self, vectors: Mapping[str, np.ndarray]) -> int:
        return await self._set_vectors(USER_EMBEDDING_PREFIX, vectors)

    async def _set_vectors(self, prefix: str, vectors: Mapping[str, np.ndarray]) -> int:
        if not vectors:
            return 0

        pipe = self._redis.pipeline(transaction=False)
        for entity_id, vector in vectors.items():
            pipe.set(f"{prefix}{entity_id}", encode_vector(vector, self._dim))

        await pipe.execute()
        return len(vectors)

    async def get_article_embeddings(self, article_ids: list[str]) -> dict[str, np.ndarray]:
        if not article_ids:
            return {}

        raw_values = await self._redis.mget([f"{ARTICLE_EMBEDDING_PREFIX}{a}" for a in article_ids])

        out: dict[str, np.ndarray] = {}
        for article_id, raw in zip(article_ids, raw_values, strict=True):
            if raw is None:
                continue
            try:
                out[article_id] = decode_vector(raw, self._dim)
            except VectorFormatError as exc:
                logger.warning("skipping embedding for %s: %s", article_id, exc)

        return out

    async def get_user_embedding(self, user_id: str) -> np.ndarray | None:
        """None means the user is cold — a normal state, and the reason FAISS answers 204."""
        raw = await self._redis.get(f"{USER_EMBEDDING_PREFIX}{user_id}")
        if raw is None:
            return None

        try:
            return decode_vector(raw, self._dim)
        except VectorFormatError:
            logger.warning("user %s has a malformed embedding, treating it as absent", user_id)
            return None

    async def iter_embedded_article_ids(self, batch_size: int = 500) -> AsyncIterator[str]:
        """SCAN over `rec:article_embedding:*`. Never KEYS — this runs against the live Redis."""
        async for key in self._redis.scan_iter(match=f"{ARTICLE_EMBEDDING_PREFIX}*", count=batch_size):
            yield _strip(key, ARTICLE_EMBEDDING_PREFIX)

    # --- content fingerprints ---------------------------------------------------------

    async def get_fingerprints(self, key: str) -> dict[str, str]:
        """One HGETALL rather than a key per article: the embeddings job needs the whole set
        to decide what changed, and 200 000 individual GETs would be the slow part of an
        otherwise cheap incremental run."""
        raw = await self._redis.hgetall(key)

        return {
            (k.decode("utf-8") if isinstance(k, bytes) else k): (
                v.decode("utf-8") if isinstance(v, bytes) else v
            )
            for k, v in raw.items()
        }

    async def set_fingerprints(self, key: str, fingerprints: Mapping[str, str]) -> None:
        if not fingerprints:
            return

        await self._redis.hset(key, mapping=dict(fingerprints))

    # --- trending ---------------------------------------------------------------------

    async def replace_trending(self, article_ids: list[str], ttl_seconds: int) -> None:
        """DEL + RPUSH in one transaction.

        Without MULTI the orchestrator can observe the empty window between the two commands
        and degrade to `recent` for no reason.
        """
        if not article_ids:
            return

        async with self._redis.pipeline(transaction=True) as pipe:
            pipe.delete(TRENDING_KEY)
            pipe.rpush(TRENDING_KEY, *article_ids)
            pipe.expire(TRENDING_KEY, ttl_seconds)
            await pipe.execute()

        await self.mark_fresh("trending")

    async def trending_length(self) -> int:
        """Only used to make "nothing to write" log lines say what survived."""
        return int(await self._redis.llen(TRENDING_KEY))

    # --- ALS candidates ---------------------------------------------------------------

    async def set_als_candidates(
        self, candidates: Mapping[str, list[str]], ttl_seconds: int
    ) -> int:
        """LIST of article ids, best first — the same shape as `rec:trending:top100`,
        because `RedisRecommendationStore` reads both with `ListRangeAsync`."""
        if not candidates:
            return 0

        written = 0
        async with self._redis.pipeline(transaction=False) as pipe:
            for user_id, article_ids in candidates.items():
                if not article_ids:
                    continue
                key = f"{ALS_CANDIDATES_PREFIX}{user_id}"
                pipe.delete(key)
                pipe.rpush(key, *article_ids)
                pipe.expire(key, ttl_seconds)
                written += 1

            await pipe.execute()

        await self.mark_fresh("user_als_candidates")
        return written

    # --- feature rows -----------------------------------------------------------------
    #
    # Stored as compact JSON STRINGs rather than HASHes so that 200 candidates come back in a
    # single MGET instead of 200 pipelined HGETALLs. /rank has ~120 ms for everything including
    # HTTP, and the per-command server cost is the part that scales with the candidate pool.

    async def set_article_features(
        self, features: Mapping[str, Mapping[str, float]], ttl_seconds: int
    ) -> int:
        return await self._set_rows(ARTICLE_FEATURES_PREFIX, features, ttl_seconds)

    async def set_user_features(
        self, features: Mapping[str, Mapping[str, float]], ttl_seconds: int
    ) -> int:
        return await self._set_rows(USER_FEATURES_PREFIX, features, ttl_seconds)

    async def _set_rows(
        self, prefix: str, features: Mapping[str, Mapping[str, float]], ttl_seconds: int
    ) -> int:
        if not features:
            return 0

        async with self._redis.pipeline(transaction=False) as pipe:
            for entity_id, values in features.items():
                pipe.set(f"{prefix}{entity_id}", encode_row(values), ex=ttl_seconds)

            await pipe.execute()

        return len(features)

    async def get_article_features(self, article_ids: list[str]) -> dict[str, dict[str, float]]:
        if not article_ids:
            return {}

        raw_values = await self._redis.mget([f"{ARTICLE_FEATURES_PREFIX}{a}" for a in article_ids])
        return _decode_rows(article_ids, raw_values)

    async def get_user_features(self, user_id: str) -> dict[str, float]:
        raw = await self._redis.get(f"{USER_FEATURES_PREFIX}{user_id}")
        if raw is None:
            return {}

        return _decode_rows([user_id], [raw]).get(user_id, {})

    # --- viewed set (written by the .NET consumer) ------------------------------------

    async def get_viewed(self, user_id: str) -> set[str]:
        members = await self._redis.smembers(f"{VIEWED_PREFIX}{user_id}")
        return {m.decode("utf-8") if isinstance(m, bytes) else m for m in members}

    # --- freshness --------------------------------------------------------------------

    async def mark_fresh(self, name: str, when: datetime | None = None) -> None:
        stamp = (when or datetime.now(UTC)).isoformat()
        await self._redis.set(freshness_key(name), stamp)

    async def get_freshness(self, name: str) -> datetime | None:
        raw = await self._redis.get(freshness_key(name))
        if raw is None:
            return None

        text = raw.decode("utf-8") if isinstance(raw, bytes) else raw
        try:
            return datetime.fromisoformat(text)
        except ValueError:
            logger.warning("unparseable freshness stamp for %s: %r", name, text)
            return None


def _strip(key: bytes | str, prefix: str) -> str:
    text = key.decode("utf-8") if isinstance(key, bytes) else key
    return text.removeprefix(prefix)


def encode_row(values: Mapping[str, float]) -> bytes:
    """The only serializer for a feature row.

    `build_features` writes with it and the ranking service reads with it, so the offline and
    online paths cannot disagree about encoding — only about which numbers they put in.
    """
    return json.dumps({k: float(v) for k, v in values.items()}, separators=(",", ":")).encode("utf-8")


def decode_row(raw: bytes | str) -> dict[str, float]:
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object, got {type(payload).__name__}")

    return {str(k): float(v) for k, v in payload.items()}


def _decode_rows(ids: list[str], raw_values: list[bytes | None]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}

    for entity_id, raw in zip(ids, raw_values, strict=True):
        if raw is None:
            continue
        try:
            out[entity_id] = decode_row(raw)
        except (ValueError, TypeError) as exc:
            # Same rule as a malformed vector in explAIned-faiss: treat it as absent rather
            # than failing the request. A neutral row scores; a 500 does not.
            logger.warning("malformed feature row for %s: %s", entity_id, exc)

    return out


def chunked(items: Iterable[str], size: int) -> Iterable[list[str]]:
    batch: list[str] = []
    for item in items:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []

    if batch:
        yield batch
