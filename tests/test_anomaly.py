from __future__ import annotations

import numpy as np
import pytest

from tanglecode.anomaly import detect
from tanglecode.braid import (
    order_from_frame,
    reduce_crossings,
    replay_orders,
    word_from_frames,
)
from tanglecode.data import generate_signals


def test_detect_clean_zero_ranges():
    """Clean data has zero flagged anomaly ranges."""
    clean_data = generate_signals("clean", n=6, T=3000, seed=7)
    T, n = clean_data.shape
    init_order, raw = word_from_frames(clean_data)
    red, cancelled = reduce_crossings(raw)
    positions = replay_orders(init_order, raw, T)

    res = detect(red, positions, n=n, T=T, window=200, step=100, k=4.0, min_consecutive=4)
    assert len(res["ranges"]) == 0
    assert res["ranges"] == []


def test_detect_drift_flags_sensor_4():
    """Drift data flags a range starting between tick 1300 and 1700 with faulty_channel == 4."""
    drift_data = generate_signals("drift", n=6, T=3000, seed=7)
    T, n = drift_data.shape
    init_order, raw = word_from_frames(drift_data)
    red, cancelled = reduce_crossings(raw)
    positions = replay_orders(init_order, raw, T)

    res = detect(red, positions, n=n, T=T, window=200, step=100, k=4.0, min_consecutive=4)
    assert len(res["ranges"]) > 0

    matching = [
        r for r in res["ranges"]
        if 1300 <= r["start_tick"] <= 1700 and r["faulty_channel"] == 4
    ]
    assert len(matching) > 0
    match = matching[0]
    assert match["faulty_channel"] == 4
    assert match["kind"] in ("drift", "erratic")
    assert isinstance(match["rank_before"], int)
    assert isinstance(match["rank_after"], int)
    assert match["severity"] in ("Mild", "Moderate", "High")


def test_detect_stuck_flags_sensor_4():
    """Stuck data flags a range starting between tick 1300 and 1700 with faulty_channel == 4."""
    stuck_data = generate_signals("stuck", n=6, T=3000, seed=7)
    T, n = stuck_data.shape
    init_order, raw = word_from_frames(stuck_data)
    red, cancelled = reduce_crossings(raw)
    positions = replay_orders(init_order, raw, T)

    res = detect(red, positions, n=n, T=T, window=200, step=100, k=4.0, min_consecutive=4)
    assert len(res["ranges"]) > 0

    matching = [
        r for r in res["ranges"]
        if 1300 <= r["start_tick"] <= 1700 and r["faulty_channel"] == 4
    ]
    assert len(matching) > 0
    match = matching[0]
    assert match["faulty_channel"] == 4
    assert match["kind"] in ("drift", "erratic")
    assert isinstance(match["rank_before"], int)
    assert isinstance(match["rank_after"], int)
    assert match["severity"] in ("Mild", "Moderate", "High")


def test_detect_erratic_classification():
    """Erratic behavior (e.g. seed 42 clean dataset noise flare) is classified as erratic."""
    seed_data = generate_signals("clean", n=6, T=3000, seed=42)
    T, n = seed_data.shape
    init_order, raw = word_from_frames(seed_data)
    red, cancelled = reduce_crossings(raw)
    positions = replay_orders(init_order, raw, T)

    res = detect(red, positions, n=n, T=T, window=200, step=100, k=4.0, min_consecutive=4)
    assert len(res["ranges"]) > 0
    match = res["ranges"][0]
    assert match["kind"] == "erratic"
    assert match["severity"] in ("Mild", "Moderate", "High")
    assert isinstance(match["rank_before"], int)
    assert isinstance(match["rank_after"], int)


def test_detect_empty_windows():
    """Edge case: T smaller than window."""
    res = detect([], np.zeros((50, 4)), n=4, T=50, window=100)
    assert res["ranges"] == []
    assert res["window_scores"] == []
