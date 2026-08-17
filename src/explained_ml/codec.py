"""Redis vector format — a deliberate copy of `explAIned-faiss/src/explained_faiss/codec.py`.

`rec:article_embedding:{aid}` and `rec:user_embedding:{uid}` hold a vector as raw
little-endian float32 — `embedding_dim * 4` bytes, no header, no separators. This repository
writes both keys; explAIned-faiss reads them. The repository has no `Shared/` layer on purpose,
so the two copies are kept honest by a golden-bytes test rather than by a package dependency
(see tests/test_codec.py). If you change anything here, change it there in the same commit.
"""

import numpy as np

DTYPE = np.dtype("<f4")


class VectorFormatError(ValueError):
    """A Redis value does not hold `dim` little-endian float32."""


def expected_bytes(dim: int) -> int:
    return dim * DTYPE.itemsize


def encode_vector(vector: np.ndarray, dim: int) -> bytes:
    if vector.shape != (dim,):
        raise VectorFormatError(f"expected shape ({dim},), got {vector.shape}")

    return np.ascontiguousarray(vector, dtype=DTYPE).tobytes()


def decode_vector(raw: bytes, dim: int) -> np.ndarray:
    want = expected_bytes(dim)
    if len(raw) != want:
        raise VectorFormatError(f"expected {want} bytes ({dim} little-endian float32), got {len(raw)}")

    return np.frombuffer(raw, dtype=DTYPE).astype(np.float32, copy=True)


def normalize(vector: np.ndarray) -> np.ndarray:
    """L2-normalise so faiss's inner product is cosine similarity.

    Normalising here rather than in the index builder keeps the guarantee with the writer:
    explAIned-faiss trusts that every stored vector is already unit length.
    """
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        return np.zeros_like(vector, dtype=np.float32)

    return (vector / norm).astype(np.float32, copy=False)
