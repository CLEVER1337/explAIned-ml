"""The vector format, and proof that our copy agrees with explAIned-faiss's.

`codec.py` is vendored rather than imported (the repository has no shared layer on purpose),
so the guarantee has to be a test. Two layers: golden bytes pin the format unconditionally,
and a cross-repo comparison catches drift wherever both checkouts exist — which is every
machine here, since they are sibling directories.
"""

import importlib.util
import os
import struct
from pathlib import Path

import numpy as np
import pytest

from explained_ml.codec import (
    VectorFormatError,
    decode_vector,
    encode_vector,
    expected_bytes,
    normalize,
)

FAISS_CODEC = Path(os.environ.get("FAISS_REPO", Path(__file__).parents[2] / "explAIned-faiss"))
FAISS_CODEC = FAISS_CODEC / "src" / "explained_faiss" / "codec.py"


def test_round_trip_preserves_values():
    vector = np.arange(8, dtype=np.float32)

    assert np.array_equal(decode_vector(encode_vector(vector, 8), 8), vector)


def test_layout_is_little_endian_float32_without_a_header():
    raw = encode_vector(np.array([1.0, -2.5, 0.0, 0.0], dtype=np.float32), 4)

    assert len(raw) == expected_bytes(4) == 16
    assert raw[:8] == struct.pack("<ff", 1.0, -2.5)


def test_decoded_vector_is_writable():
    # frombuffer alone yields a read-only view; faiss and numpy in-place ops need an owned array.
    decoded = decode_vector(encode_vector(np.ones(4, dtype=np.float32), 4), 4)
    decoded[0] = 5.0

    assert decoded[0] == 5.0


@pytest.mark.parametrize("length", [0, 15, 17])
def test_wrong_byte_length_is_rejected(length):
    with pytest.raises(VectorFormatError):
        decode_vector(b"\x00" * length, 4)


def test_wrong_shape_is_rejected():
    with pytest.raises(VectorFormatError):
        encode_vector(np.ones(3, dtype=np.float32), 4)


def test_normalize_makes_a_unit_vector():
    normalized = normalize(np.array([3.0, 4.0], dtype=np.float32))

    assert np.isclose(np.linalg.norm(normalized), 1.0)
    assert np.allclose(normalized, [0.6, 0.8])


def test_normalize_of_a_zero_vector_stays_zero():
    # A user with no usable history produces a zero centroid; it must not become NaN and
    # poison every cosine feature downstream.
    assert np.array_equal(normalize(np.zeros(4, dtype=np.float32)), np.zeros(4, dtype=np.float32))


@pytest.mark.skipif(not FAISS_CODEC.exists(), reason="sibling explAIned-faiss checkout not present")
def test_encoding_matches_explained_faiss_byte_for_byte():
    spec = importlib.util.spec_from_file_location("_faiss_codec", FAISS_CODEC)
    other = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(other)

    rng = np.random.default_rng(0)
    for _ in range(32):
        vector = rng.standard_normal(384).astype(np.float32)

        assert encode_vector(vector, 384) == other.encode_vector(vector, 384)
        assert np.array_equal(decode_vector(other.encode_vector(vector, 384), 384), vector)
