from __future__ import annotations

import pytest
import numpy as np
import pandas as pd

from tanglecode.anomaly import detect
from tanglecode.braid import order_from_frame, reduce_crossings, replay_orders, word_from_frames
from tanglecode.data import generate_signals
from tanglecode.ranking import get_sensor_ranking, get_place_changes_caption


def test_sensor_ranking_clean_and_drift_samples():
    """Verify ranking table matches order_from_frame, has N rows, and places run 1..N without gaps."""
    chosen_ticks = [0, 500, 1500, 1600, 2000]

    for kind in ("clean", "drift"):
        data = generate_signals(kind, n=6, T=3000, seed=7)
        T, n = data.shape
        init_order, raw = word_from_frames(data)
        red, _ = reduce_crossings(raw)
        pos = replay_orders(init_order, raw, T)
        det = detect(red, pos, n=n, T=T)
        ranges = det["ranges"]

        for tick in chosen_ticks:
            df = get_sensor_ranking(data, tick, ranges)

            # Exactly N rows
            assert len(df) == n, f"Expected {n} rows, got {len(df)}"

            # Columns check
            assert list(df.columns) == ["Place", "Sensor", "Reading"]

            # Places run 1 to N without gaps
            assert list(df["Place"]) == list(range(1, n + 1))

            # Table order matches order_from_frame
            expected_order = order_from_frame(data[tick])
            for place_idx, ch_idx in enumerate(expected_order):
                expected_place = place_idx + 1
                row = df.iloc[place_idx]
                assert row["Place"] == expected_place

                ch_num = ch_idx + 1
                assert f"Sensor {ch_num}" in row["Sensor"]

                expected_val = round(float(data[tick, ch_idx]), 4)
                assert row["Reading"] == expected_val


def test_sensor_ranking_suspect_marking():
    """Verify suspect sensor is flagged only when tick is inside a flagged range."""
    drift_data = generate_signals("drift", n=6, T=3000, seed=7)
    T, n = drift_data.shape
    init_order, raw = word_from_frames(drift_data)
    red, _ = reduce_crossings(raw)
    pos = replay_orders(init_order, raw, T)
    det = detect(red, pos, n=n, T=T)
    ranges = det["ranges"]
    assert len(ranges) > 0
    fault_range = ranges[0]
    fault_start = fault_range["start_tick"]
    fault_end = fault_range["end_tick"]

    # Before fault range: no suspect tag
    df_before = get_sensor_ranking(drift_data, fault_start - 100, ranges)
    assert not any("(suspect)" in str(s) for s in df_before["Sensor"])

    # Inside fault range: suspect tag present on Sensor 4
    df_inside = get_sensor_ranking(drift_data, (fault_start + fault_end) // 2, ranges)
    suspect_rows = df_inside[df_inside["Sensor"].str.contains(r"\(suspect\)")]
    assert len(suspect_rows) == 1
    assert "Sensor 4 (suspect)" in suspect_rows.iloc[0]["Sensor"]


def test_place_changes_caption():
    """Verify place changes caption for ticks below 100 and ticks >= 100."""
    data = generate_signals("drift", n=6, T=3000, seed=7)

    # Ticks below 100 must return None (skipped)
    assert get_place_changes_caption(data, 50) is None

    # Ticks with no change
    caption_1400 = get_place_changes_caption(data, 1400)
    assert caption_1400 == "Since 100 ticks ago: no sensor changed place."

    # Ticks with changes
    caption_1700 = get_place_changes_caption(data, 1700)
    assert caption_1700 is not None
    assert "Since 100 ticks ago:" in caption_1700
    assert "Sensor 4 moved from place 5 to place 6" in caption_1700
