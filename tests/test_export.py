from __future__ import annotations

import io
import csv
import numpy as np
import pytest

from tanglecode.braid import (
    Crossing,
    order_from_frame,
    reduce_crossings,
    replay_orders,
    word_from_frames,
)
from tanglecode.data import generate_signals
from tanglecode.export import swap_log_csv


def test_swap_log_csv_clean_and_drift_counts():
    """Verify swap_log_csv row count equals reduced crossing count on clean and drift data."""
    for kind in ("clean", "drift"):
        data = generate_signals(kind, n=6, T=3000, seed=7)
        _, raw = word_from_frames(data)
        reduced, _ = reduce_crossings(raw)

        csv_str = swap_log_csv(reduced)
        reader = list(csv.DictReader(io.StringIO(csv_str)))

        assert len(reader) == len(reduced)


def test_swap_log_csv_sensor_ranges_and_positions():
    """Verify sensors are 1-based between 1 and n, and position equals abs(gen)."""
    n = 6
    drift_data = generate_signals("drift", n=n, T=3000, seed=7)
    _, raw = word_from_frames(drift_data)
    reduced, _ = reduce_crossings(raw)

    csv_str = swap_log_csv(reduced)
    reader = list(csv.DictReader(io.StringIO(csv_str)))

    for row, c in zip(reader, reduced):
        s_up = int(row["sensor_moved_up"])
        s_down = int(row["sensor_moved_down"])
        pos = int(row["position"])
        tick = int(row["tick"])

        assert 1 <= s_up <= n
        assert 1 <= s_down <= n
        assert s_up == c.x + 1
        assert s_down == c.y + 1
        assert pos == abs(c.gen)
        assert tick == c.tick


def test_swap_log_csv_with_sampling_rate():
    """Verify time_s column is included and formatted to one decimal when sampling_rate is provided."""
    data = generate_signals("clean", n=6, T=500, seed=7)
    _, raw = word_from_frames(data)
    reduced, _ = reduce_crossings(raw)

    csv_str = swap_log_csv(reduced, sampling_rate=10.0)
    reader = list(csv.DictReader(io.StringIO(csv_str)))

    assert "time_s" in reader[0]
    for row in reader:
        expected_time = f"{float(row['tick']) / 10.0:.1f}"
        assert row["time_s"] == expected_time


def test_swap_log_csv_empty_header_only():
    """Verify header only is returned when reduced crossings is empty."""
    csv_str_no_rate = swap_log_csv([])
    assert csv_str_no_rate == "tick,sensor_moved_up,sensor_moved_down,position\n"

    csv_str_rate = swap_log_csv([], sampling_rate=5.0)
    assert csv_str_rate == "tick,time_s,sensor_moved_up,sensor_moved_down,position\n"


def test_replayed_final_order_equals_true_final():
    """Verify replaying the reduced word gives the exact true final order."""
    for kind in ("clean", "drift", "stuck"):
        data = generate_signals(kind, n=6, T=3000, seed=7)
        initial_order, raw = word_from_frames(data)
        reduced, _ = reduce_crossings(raw)
        positions = replay_orders(initial_order, reduced, len(data))

        replayed_final = np.argsort(positions[-1]).tolist()
        true_final = order_from_frame(data[-1])
        assert replayed_final == true_final
