"""Tests for tanglecode.codec module."""

import struct
import numpy as np
import pytest

from tanglecode.braid import rank_vector
from tanglecode.codec import (
    MAGIC,
    MODE_HYBRID,
    MODE_RANK_ONLY,
    _pack_generators,
    _pack_nibble,
    _rle_decode,
    _rle_encode,
    _unpack_generators,
    _unpack_nibble,
    decode,
    encode,
)
from tanglecode.data import generate_synthetic


def test_nibble_packing_unpacking():
    """Verify 4-bit nibble encoding with sign bit and 3-bit generator index."""
    test_generators = [1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 6, -6, 7, -7]
    for g in test_generators:
        nibble = _pack_nibble(g)
        # 4 bits (0 <= nibble <= 15)
        assert 0 <= nibble <= 15
        unpacked = _unpack_nibble(nibble)
        assert unpacked == g

    # Test full 2-per-byte packing with even and odd generator lengths
    packed_even = _pack_generators([1, -2, 3, -4])
    assert len(packed_even) == 2
    assert _unpack_generators(packed_even, 4) == [1, -2, 3, -4]

    packed_odd = _pack_generators([1, -2, 3])
    assert len(packed_odd) == 2
    assert _unpack_generators(packed_odd, 3) == [1, -2, 3]


def test_rle_roundtrip():
    """Verify run-length encoding and decoding of per-tick counts."""
    counts = [0, 0, 0, 0, 1, 1, 0, 2, 0, 0, 0, 3]
    encoded = _rle_encode(counts)
    decoded, offset = _rle_decode(encoded, 0)
    assert offset == len(encoded)
    assert decoded == counts


def test_header_structure():
    """Verify that encoded binary begins with valid 14-byte header."""
    data = generate_synthetic(n_channels=4, n_samples=50, seed=123)
    binary = encode(data, mode=MODE_RANK_ONLY)

    assert len(binary) >= 14
    magic, n, raw_count, reduced_count, mode = struct.unpack_from("<4sBIIB", binary, 0)
    assert magic == MAGIC
    assert n == 4
    assert raw_count >= reduced_count
    assert mode == MODE_RANK_ONLY


def test_roundtrip_hybrid_mode():
    """Verify encode and decode roundtrip in hybrid mode preserving approximate amplitudes."""
    data = generate_synthetic(n_channels=6, n_samples=300, noise_std=0.02, seed=99)
    step_size = 0.02

    encoded = encode(data, mode=MODE_HYBRID, step_size=step_size)
    reconstructed = decode(encoded)

    assert reconstructed.shape == data.shape
    # Anchor frame at t=0 must match float32 precision exactly
    np.testing.assert_allclose(reconstructed[0], data[0], atol=1e-6)

    # Reconstructed continuous amplitude must remain within quantization tolerance
    max_err = np.max(np.abs(reconstructed - data))
    assert max_err <= step_size * 2.0


def test_roundtrip_rank_only_mode():
    """Verify encode and decode in rank-only mode faithfully restores rank trajectories."""
    data = generate_synthetic(n_channels=4, n_samples=100, noise_std=0.05, seed=7)

    encoded = encode(data, mode=MODE_RANK_ONLY)
    reconstructed = decode(encoded)

    assert reconstructed.shape == data.shape

    # Rank-only values are channel ranks (0.0 to 3.0)
    assert np.all(reconstructed >= 0.0)
    assert np.all(reconstructed < 4.0)

    # Check that rank vector at frame 0 matches original anchor frame
    assert rank_vector(reconstructed[0]) == rank_vector(data[0])

    # Check rank vectors across frames
    for t in range(len(data)):
        recon_ranks = rank_vector(reconstructed[t])
        # Values in reconstructed are distinct ranks (0, 1, 2, 3)
        assert sorted(reconstructed[t]) == [0.0, 1.0, 2.0, 3.0]
        assert len(set(recon_ranks)) == 4


def test_unsupported_channel_count():
    """Verify that > 8 channels is rejected by .tngl codec."""
    large_data = np.zeros((10, 9), dtype=np.float32)
    with pytest.raises(ValueError, match="supports 1 to 8 channels"):
        encode(large_data)
