"""Fakes shared across the suite.

The rule inherited from explAIned-faiss: the unit suite touches no Redis, no ClickHouse, no
network and no model download. Everything below is hand-rolled for that reason — a fake that
implements only the handful of methods the code actually calls is easier to reason about than
a library that emulates all of Redis.
"""

import hashlib

import numpy as np
import pytest

DIM = 8


@pytest.fixture
def dim() -> int:
    return DIM


class FakeRedis:
    """Only the commands RecommendationStore issues, backed by plain dicts."""

    def __init__(self, data: dict[str, bytes] | None = None) -> None:
        self.strings: dict[str, bytes] = dict(data or {})
        self.lists: dict[str, list[bytes]] = {}
        self.sets: dict[str, set[bytes]] = {}
        self.hashes: dict[str, dict[bytes, bytes]] = {}
        self.ttls: dict[str, int] = {}
        self.closed = False

    # --- strings ---

    async def get(self, key):
        return self.strings.get(_s(key))

    async def set(self, key, value, ex: int | None = None):
        self.strings[_s(key)] = _b(value)
        if ex is not None:
            self.ttls[_s(key)] = ex
        return True

    async def mget(self, keys):
        return [self.strings.get(_s(k)) for k in keys]

    async def delete(self, *keys):
        removed = 0
        for key in keys:
            for bucket in (self.strings, self.lists, self.sets):
                if _s(key) in bucket:
                    del bucket[_s(key)]
                    removed += 1
        return removed

    async def expire(self, key, seconds):
        self.ttls[_s(key)] = seconds
        return True

    # --- lists ---

    async def rpush(self, key, *values):
        bucket = self.lists.setdefault(_s(key), [])
        bucket.extend(_b(v) for v in values)
        return len(bucket)

    async def llen(self, key):
        return len(self.lists.get(_s(key), []))

    async def lrange(self, key, start, stop):
        bucket = self.lists.get(_s(key), [])
        return bucket[start:] if stop == -1 else bucket[start : stop + 1]

    # --- hashes ---

    async def hgetall(self, key):
        return dict(self.hashes.get(_s(key), {}))

    async def hset(self, key, mapping=None, **kwargs):
        bucket = self.hashes.setdefault(_s(key), {})
        for field, value in (mapping or {}).items():
            bucket[_b(field)] = _b(value)
        return len(mapping or {})

    # --- sets ---

    async def sadd(self, key, *members):
        bucket = self.sets.setdefault(_s(key), set())
        bucket.update(_b(m) for m in members)
        return len(members)

    async def smembers(self, key):
        return set(self.sets.get(_s(key), set()))

    # --- scan ---

    async def scan_iter(self, match: str, count: int = 100):
        prefix = match.rstrip("*")
        for key in list(self.strings):
            if key.startswith(prefix):
                yield key.encode("utf-8")

    def pipeline(self, transaction: bool = False):
        return FakePipeline(self)

    async def aclose(self):
        self.closed = True


class FakePipeline:
    """Queues commands and replays them on execute(), which is all the store relies on."""

    def __init__(self, redis: FakeRedis) -> None:
        self._redis = redis
        self._queue: list[tuple[str, tuple, dict]] = []

    def __getattr__(self, name: str):
        def queue(*args, **kwargs):
            self._queue.append((name, args, kwargs))
            return self

        return queue

    async def execute(self):
        results = []
        for name, args, kwargs in self._queue:
            results.append(await getattr(self._redis, name)(*args, **kwargs))

        self._queue.clear()
        return results

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class HashingEncoder:
    """A deterministic stand-in for Sentence-BERT.

    Same trick as explAIned-faiss's dev fixture: seed a generator from the text hash. Identical
    text gives an identical vector, different text gives an unrelated one, and nothing is
    downloaded. It is not semantic, so tests assert plumbing, never relevance.
    """

    def __init__(self, dim: int = DIM) -> None:
        self.dim = dim
        self.calls: list[list[str]] = []

    def encode(self, texts: list[str]) -> np.ndarray:
        self.calls.append(list(texts))
        rows = [self._one(text) for text in texts]
        return np.vstack(rows) if rows else np.zeros((0, self.dim), dtype=np.float32)

    def _one(self, text: str) -> np.ndarray:
        seed = int.from_bytes(hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest(), "big")
        rng = np.random.default_rng(seed)
        return rng.standard_normal(self.dim).astype(np.float32)


@pytest.fixture
def fake_redis() -> FakeRedis:
    return FakeRedis()


@pytest.fixture
def encoder() -> HashingEncoder:
    return HashingEncoder()


def _s(key) -> str:
    return key.decode("utf-8") if isinstance(key, bytes) else str(key)


def _b(value) -> bytes:
    if isinstance(value, bytes):
        return value
    return str(value).encode("utf-8")
