from __future__ import annotations

import io
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from tanglecode.braid import (
    order_from_frame,
    reduce_crossings,
    replay_orders,
    word_from_frames,
)
from tanglecode.codec import (
    decode,
    decode_leb128,
    encode,
    encode_leb128,
    pack_generators,
    unpack_generators,
)
from tanglecode.data import generate_signals, load_csv


def test_round_trip_clean():
    """Round trip on clean data:

    - returns the same generators and ticks
    - replayed final order equals the true final order
    - .tngl is smaller than the CSV
    """
    clean_data = generate_signals("clean", n=6, T=3000, seed=7)
    initial_order, raw = word_from_frames(clean_data)
    expected_reduced, cancelled_pairs = reduce_crossings(raw)

    tngl_bytes, stats = encode(clean_data)

    # Decode
    n, T, anchor_order, decoded_reduced = decode(tngl_bytes)

    assert n == 6
    assert T == 3000
    assert anchor_order == initial_order
    assert len(decoded_reduced) == len(expected_reduced)

    # Verify same generators and ticks
    for dec_c, exp_c in zip(decoded_reduced, expected_reduced):
        assert dec_c.tick == exp_c.tick
        assert dec_c.gen == exp_c.gen

    # Replay on decoded word produces the true final order
    positions = replay_orders(anchor_order, decoded_reduced, T)
    replayed_final_order = np.argsort(positions[-1]).tolist()
    true_final_order = order_from_frame(clean_data[-1])
    assert replayed_final_order == true_final_order

    # .tngl is smaller than the CSV
    clean_csv_path = Path("data/test_clean.csv")
    assert clean_csv_path.exists(), "data/test_clean.csv must exist"
    csv_size = clean_csv_path.stat().st_size
    assert len(tngl_bytes) < csv_size
    assert stats["tngl_bytes"] == len(tngl_bytes)
    assert stats["tngl_bytes"] < csv_size
    assert stats["raw_crossings"] == len(raw)
    assert stats["reduced_crossings"] == len(expected_reduced)
    assert stats["cancelled_pairs"] == cancelled_pairs
    assert stats["percent_eliminated"] > 0


def test_round_trip_drift():
    """Round trip on drift data:

    - returns the same generators and ticks
    - replayed final order equals the true final order
    - .tngl is smaller than the CSV
    """
    drift_data = generate_signals("drift", n=6, T=3000, seed=7)
    initial_order, raw = word_from_frames(drift_data)
    expected_reduced, cancelled_pairs = reduce_crossings(raw)

    tngl_bytes, stats = encode(drift_data)

    n, T, anchor_order, decoded_reduced = decode(tngl_bytes)

    assert n == 6
    assert T == 3000
    assert anchor_order == initial_order
    assert len(decoded_reduced) == len(expected_reduced)

    # Verify same generators and ticks
    for dec_c, exp_c in zip(decoded_reduced, expected_reduced):
        assert dec_c.tick == exp_c.tick
        assert dec_c.gen == exp_c.gen

    # Replay produces true final order
    positions = replay_orders(anchor_order, decoded_reduced, T)
    replayed_final_order = np.argsort(positions[-1]).tolist()
    true_final_order = order_from_frame(drift_data[-1])
    assert replayed_final_order == true_final_order

    # .tngl is smaller than the CSV
    fault_csv_path = Path("data/test_fault.csv")
    assert fault_csv_path.exists(), "data/test_fault.csv must exist"
    csv_size = fault_csv_path.stat().st_size
    assert len(tngl_bytes) < csv_size
    assert stats["tngl_bytes"] < csv_size


def test_round_trip_stuck():
    """Round trip on stuck data."""
    stuck_data = generate_signals("stuck", n=6, T=3000, seed=7)
    initial_order, raw = word_from_frames(stuck_data)
    expected_reduced, _ = reduce_crossings(raw)

    tngl_bytes, stats = encode(stuck_data)
    n, T, anchor_order, decoded_reduced = decode(tngl_bytes)

    assert n == 6
    assert T == 3000
    assert anchor_order == initial_order
    assert len(decoded_reduced) == len(expected_reduced)

    positions = replay_orders(anchor_order, decoded_reduced, T)
    assert np.argsort(positions[-1]).tolist() == order_from_frame(stuck_data[-1])


def test_nibble_packing_odd_and_even_counts():
    """Nibble packing must handle odd counts and both generator signs/indices."""
    test_cases = [
        [],
        [1],
        [-1],
        [1, -2],
        [1, -2, 3],
        [-1, 2, -3, 4],
        [1, -2, 3, -4, 5],
        [7, -7, 6, -6, 5, -5, 4],
        [-3, 2, -1, 4, -5, 6, -7, 1, -2],
    ]
    for gens in test_cases:
        packed = pack_generators(gens)
        expected_bytes = (len(gens) + 1) // 2
        assert len(packed) == expected_bytes
        unpacked = unpack_generators(packed, len(gens))
        assert unpacked == gens, f"Failed for {gens}: got {unpacked}"


def test_leb128_encoding_decoding():
    """Verify unsigned LEB128 varint encoding and decoding."""
    test_values = [0, 1, 2, 63, 127, 128, 255, 300, 16383, 16384, 65535, 1000000]
    buf = bytearray()
    for val in test_values:
        buf.extend(encode_leb128(val))

    offset = 0
    decoded_values = []
    for _ in test_values:
        val, offset = decode_leb128(bytes(buf), offset)
        decoded_values.append(val)

    assert decoded_values == test_values
    assert offset == len(buf)

    with pytest.raises(ValueError, match="non-negative"):
        encode_leb128(-5)


def test_load_csv_and_validation():
    """Test load_csv features:

    - drops non-numeric columns
    - drops NaN rows
    - uses at most 8 columns (and returns warning string)
    - raises ValueError with plain-English message if < 2 numeric cols or < 300 rows
    """
    # 1. Successful loading of generated files
    clean_arr, warn = load_csv("data/test_clean.csv")
    assert clean_arr.shape == (3000, 6)
    assert warn is None

    # 2. Non-numeric columns + NaN rows
    csv_content = io.StringIO(
        "timestamp,label,ch1,ch2,ch3\n"
        + "".join(f"2026-01-01,walk,{i * 0.1},{i * 0.2},{i * 0.3}\n" for i in range(350))
        + "2026-01-01,walk,NaN,0.5,0.6\n"
    )
    arr, warn = load_csv(csv_content)
    assert arr.shape == (350, 3)
    assert warn is None

    # 3. More than 8 numeric columns -> capped to 8 with warning
    cols = [f"c{i}" for i in range(10)]
    df_10 = pd.DataFrame(np.ones((320, 10)), columns=cols)
    buf_10 = io.StringIO()
    df_10.to_csv(buf_10, index=False)
    buf_10.seek(0)
    arr_8, warn_8 = load_csv(buf_10)
    assert arr_8.shape == (320, 8)
    assert warn_8 is not None
    assert "8 columns" in warn_8

    # 4. Fewer than 2 numeric columns -> ValueError
    df_1 = pd.DataFrame({"c1": np.ones(350), "text": ["abc"] * 350})
    buf_1 = io.StringIO()
    df_1.to_csv(buf_1, index=False)
    buf_1.seek(0)
    with pytest.raises(ValueError, match="at least 2 numeric columns"):
        load_csv(buf_1)

    # 5. Fewer than 300 rows -> ValueError
    df_short = pd.DataFrame({"c1": np.ones(250), "c2": np.ones(250)})
    buf_short = io.StringIO()
    df_short.to_csv(buf_short, index=False)
    buf_short.seek(0)
    with pytest.raises(ValueError, match="at least 300 rows"):
        load_csv(buf_short)
