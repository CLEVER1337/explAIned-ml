"""Text -> 384 floats.

`sentence-transformers` is imported inside `SentenceBertEncoder.__init__`, never at module
scope. Two reasons: the ranking image does not install the `jobs` dependency group, and the
unit suite must not pull ~2.5 GB of torch to test what is essentially a loop over Redis.

`HashingEncoder` is a **fixture, not a model**. It exists so the whole pipeline can be run
end to end on a machine that cannot download model weights — the vectors it produces are
deterministic and well-formed but carry no meaning, so anything that looks like relevance in
its output is coincidence. explAIned-faiss makes the same distinction with
`seed_dev_embeddings.py`.
"""

import hashlib
import logging
from typing import Protocol

import numpy as np

logger = logging.getLogger(__name__)


class TextEncoder(Protocol):
    dim: int

    def encode(self, texts: list[str]) -> np.ndarray:
        """`(len(texts), dim)` float32. Callers normalise; encoders do not have to."""
        ...


class SentenceBertEncoder:
    """The real one. `paraphrase-multilingual-MiniLM-L12-v2`, 384 dims, matches EMBEDDING_DIM."""

    def __init__(self, model_name: str, dim: int, batch_size: int = 32) -> None:
        from sentence_transformers import SentenceTransformer

        self.dim = dim
        self._batch_size = batch_size
        self._model = SentenceTransformer(model_name)

        actual = self._model.get_sentence_embedding_dimension()
        if actual != dim:
            # A dimension mismatch would be caught by explAIned-faiss much later, as a stream
            # of "malformed embedding" warnings with no indication of the cause.
            raise ValueError(
                f"{model_name} produces {actual}-dim vectors but EMBEDDING_DIM is {dim}; "
                "the faiss index and every stored vector assume the latter"
            )

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)

        return np.asarray(
            self._model.encode(texts, batch_size=self._batch_size, show_progress_bar=False),
            dtype=np.float32,
        )


class HashingEncoder:
    """Deterministic nonsense of the right shape. Dev only — see the module docstring."""

    def __init__(self, dim: int) -> None:
        self.dim = dim

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)

        return np.vstack([self._one(text) for text in texts])

    def _one(self, text: str) -> np.ndarray:
        digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
        rng = np.random.default_rng(int.from_bytes(digest, "big"))
        return rng.standard_normal(self.dim).astype(np.float32)


def build_encoder(kind: str, model_name: str, dim: int, batch_size: int = 32) -> TextEncoder:
    if kind == "hashing":
        logger.warning(
            "using the hashing encoder: vectors will be well-formed and meaningless. "
            "Never run this against an index that anyone reads for real recommendations."
        )
        return HashingEncoder(dim)

    return SentenceBertEncoder(model_name, dim, batch_size)


def fingerprint(text: str) -> str:
    """Content hash, used to skip unchanged articles.

    Derived from the text rather than from `UpdatedAt` on purpose: a no-op edit bumps the
    timestamp but should not cost a re-encode, and `PublishedAt`/`UpdatedAt` semantics in the
    article service are already muddy enough (see articles.py).
    """
    return hashlib.blake2b(text.encode("utf-8"), digest_size=16).hexdigest()
