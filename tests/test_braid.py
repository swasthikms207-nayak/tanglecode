from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tanglecode.braid import (
    Crossing,
    crossings_between,
    order_from_frame,
    reduce_crossings,
    replay_orders,
    word_from_frames,
)


def test_order_from_frame_basic_and_ties():
    """Test order_from_frame sorting ascending and tie-breaking by channel index.
    Position 1 = lowest value.
    """
    frame = [30.0, 10.0, 20.0]
    # Channels 0: 30, 1: 10, 2: 20 -> sorted order: channel 1 (10), channel 2 (20), channel 0 (30)
    assert order_from_frame(frame) == [1, 2, 0]

    # Ties: channel 0 and channel 2 both have value 10.0 -> channel 0 comes before channel 2
    ties_frame = [10.0, 25.0, 10.0]
    assert order_from_frame(ties_frame) == [0, 2, 1]

    # Pandas Series input
    s = pd.Series([5.0, 2.0, 8.0])
    assert order_from_frame(s) == [1, 0, 2]

    # Numpy 1D array
    arr = np.array([100, -50, 0])
    assert order_from_frame(arr) == [1, 2, 0]


def test_crossings_between_deterministic_bubble_sort():
    """Test crossings_between bubble sorting logic and generator signs."""
    # Identical orders: no crossings
    assert crossings_between([0, 1, 2], [0, 1, 2], tick=1) == []

    # Single swap: [0, 1] -> [1, 0]
    # Pos 1: x=0 moving up, y=1 moving down. x < y -> gen = +1
    c1 = crossings_between([0, 1], [1, 0], tick=1)
    assert c1 == [Crossing(tick=1, gen=1, x=0, y=1)]

    # Reverse swap: [1, 0] -> [0, 1]
    # Pos 1: x=1 moving up, y=0 moving down. x < y is False -> gen = -1
    c2 = crossings_between([1, 0], [0, 1], tick=2)
    assert c2 == [Crossing(tick=2, gen=-1, x=1, y=0)]

    # Higher channel index initially lower in position
    # [2, 0] -> [0, 2]: x=2, y=0. x < y is False -> gen = -1
    c3 = crossings_between([2, 0], [0, 2], tick=3)
    assert c3 == [Crossing(tick=3, gen=-1, x=2, y=0)]


def test_overtake_then_drop_back_reduces_to_empty():
    """Overtake then drop back next tick reduces to an empty word."""
    # 2 channels: 0 and 1
    # Tick 0: 0=10, 1=20 (order: [0, 1])
    # Tick 1: 0=25, 1=20 (order: [1, 0]) -> overtake by channel 0
    # Tick 2: 0=10, 1=20 (order: [0, 1]) -> drop-back by channel 0
    data = np.array([
        [10.0, 20.0],
        [25.0, 20.0],
        [10.0, 20.0],
    ])
    initial_order, raw_crossings = word_from_frames(data)
    assert initial_order == [0, 1]
    assert len(raw_crossings) == 2

    # Swap 1: pos 1, x=0, y=1 (0 < 1 => gen = +1)
    assert raw_crossings[0] == Crossing(tick=1, gen=1, x=0, y=1)
    # Swap 2: pos 1, x=1, y=0 (1 < 0 is False => gen = -1)
    assert raw_crossings[1] == Crossing(tick=2, gen=-1, x=1, y=0)

    reduced, cancelled_pairs = reduce_crossings(raw_crossings)
    assert reduced == []
    assert cancelled_pairs == 1

    # Replay on reduced word maintains initial order at all ticks
    positions = replay_orders(initial_order, reduced, n_ticks=3)
    for t in range(3):
        assert np.argsort(positions[t]).tolist() == [0, 1]


def test_far_away_crossing_in_between_still_cancels():
    """Same overtake and drop-back, with a far-away crossing (|i-j| >= 2) in between: still cancels."""
    # 4 channels: 0, 1, 2, 3
    # Pos 1 swap (between strands at pos 1 & 2): gen = 1
    # Pos 3 swap (between strands at pos 3 & 4): gen = 3 (|3 - 1| = 2 >= 2, commutes)
    # Pos 1 drop-back: gen = -1
    raw_crossings = [
        Crossing(tick=1, gen=1, x=0, y=1),
        Crossing(tick=2, gen=3, x=2, y=3),
        Crossing(tick=3, gen=-1, x=1, y=0),
    ]

    reduced, cancelled_pairs = reduce_crossings(raw_crossings)

    assert cancelled_pairs == 1
    assert len(reduced) == 1
    assert reduced[0] == Crossing(tick=2, gen=3, x=2, y=3)

    # Also test via full frame data
    data = np.array([
        [10.0, 20.0, 30.0, 40.0],  # Tick 0: [0, 1, 2, 3]
        [25.0, 20.0, 30.0, 40.0],  # Tick 1: [1, 0, 2, 3] (swap pos 1, gen=1)
        [25.0, 20.0, 45.0, 40.0],  # Tick 2: [1, 0, 3, 2] (swap pos 3, gen=3)
        [10.0, 20.0, 45.0, 40.0],  # Tick 3: [0, 1, 3, 2] (swap pos 1, gen=-1)
    ])
    init_order, frames_raw = word_from_frames(data)
    frames_red, frames_cancelled = reduce_crossings(frames_raw)
    assert frames_cancelled == 1
    assert len(frames_red) == 1
    assert frames_red[0].gen == 3
    assert frames_red[0].tick == 2
    assert frames_red[0].x == 2 and frames_red[0].y == 3


def test_lasting_crossing_survives():
    """A real lasting crossing survives."""
    data = np.array([
        [10.0, 20.0],  # Tick 0: [0, 1]
        [25.0, 20.0],  # Tick 1: [1, 0] (overtake)
        [30.0, 20.0],  # Tick 2: [1, 0] (no crossing, lasting)
    ])
    initial_order, raw = word_from_frames(data)
    assert len(raw) == 1
    assert raw[0] == Crossing(tick=1, gen=1, x=0, y=1)

    reduced, cancelled_pairs = reduce_crossings(raw)
    assert len(reduced) == 1
    assert cancelled_pairs == 0
    assert reduced[0] == raw[0]

    positions = replay_orders(initial_order, reduced, n_ticks=3)
    assert np.argsort(positions[0]).tolist() == [0, 1]
    assert np.argsort(positions[1]).tolist() == [1, 0]
    assert np.argsort(positions[2]).tolist() == [1, 0]


def test_adjacent_crossings_do_not_commute():
    """Adjacent-position crossings (|i-j| = 1) do NOT commute and are not wrongly cancelled."""
    # Crossing 1 at pos 1: gen = 1
    # Crossing 2 at pos 2: gen = 2 (|2 - 1| = 1 < 2, does NOT commute)
    # Crossing 3 at pos 1: gen = -1
    # gen=-1 cannot commute past gen=2, so it cannot cancel gen=1!
    c1 = Crossing(tick=1, gen=1, x=0, y=1)
    c2 = Crossing(tick=2, gen=2, x=0, y=2)
    c3 = Crossing(tick=3, gen=-1, x=1, y=0)

    crossings = [c1, c2, c3]
    reduced, cancelled_pairs = reduce_crossings(crossings)

    assert cancelled_pairs == 0
    assert len(reduced) == 3
    assert reduced == crossings

    # Similarly for gen=2 followed by gen=1 followed by gen=-2
    c4 = Crossing(tick=1, gen=2, x=1, y=2)
    c5 = Crossing(tick=2, gen=1, x=0, y=1)
    c6 = Crossing(tick=3, gen=-2, x=2, y=1)

    reduced2, cancelled2 = reduce_crossings([c4, c5, c6])
    assert cancelled2 == 0
    assert len(reduced2) == 3
    assert reduced2 == [c4, c5, c6]


def test_random_data_200_seeds():
    """Random data, 200 seeds: replaying the reduced word gives exactly the true final order.

    Also tests len(reduced) <= len(raw) and raw - reduced == 2 * cancelled_pairs.
    """
    for seed in range(200):
        rng = np.random.default_rng(seed)
        n_channels = rng.integers(2, 8)
        n_ticks = rng.integers(3, 25)

        # Generate continuous random walk to produce natural overtakes
        steps = rng.normal(0, 1, size=(n_ticks, n_channels))
        data = np.cumsum(steps, axis=0)

        initial_order, raw = word_from_frames(data)
        reduced, cancelled_pairs = reduce_crossings(raw)

        # Invariants
        assert len(reduced) <= len(raw), f"seed={seed}: len(reduced) > len(raw)"
        assert len(raw) - len(reduced) == 2 * cancelled_pairs, (
            f"seed={seed}: raw({len(raw)}) - reduced({len(reduced)}) != 2 * cancelled({cancelled_pairs})"
        )

        # Replay reduced word
        positions = replay_orders(initial_order, reduced, n_ticks)
        replayed_final_order = np.argsort(positions[-1]).tolist()
        true_final_order = order_from_frame(data[-1])

        assert replayed_final_order == true_final_order, (
            f"seed={seed}: replayed {replayed_final_order} != true {true_final_order}"
        )

        # Raw word replaying must match true order at every single tick
        raw_positions = replay_orders(initial_order, raw, n_ticks)
        for t in range(n_ticks):
            assert np.argsort(raw_positions[t]).tolist() == order_from_frame(data[t]), (
                f"seed={seed}, tick={t}: raw replay mismatch"
            )


def test_reduced_length_and_cancelled_pairs_invariant():
    """len(reduced) <= len(raw) and raw - reduced == 2 * cancelled_pairs."""
    # Test multiple cancellation patterns in sequence
    crossings = [
        Crossing(tick=1, gen=1, x=0, y=1),
        Crossing(tick=2, gen=3, x=2, y=3),
        Crossing(tick=3, gen=5, x=4, y=5),
        Crossing(tick=4, gen=-3, x=3, y=2),  # cancels with tick 2 (gen 3)
        Crossing(tick=5, gen=-1, x=1, y=0),  # cancels with tick 1 (gen 1)
        Crossing(tick=6, gen=-5, x=5, y=4),  # cancels with tick 3 (gen 5)
    ]
    reduced, cancelled = reduce_crossings(crossings)
    assert len(reduced) == 0
    assert cancelled == 3
    assert len(crossings) - len(reduced) == 2 * cancelled


def test_dataframe_input_and_edge_cases():
    """Test pandas DataFrame input and edge cases (empty data, single tick, 0 ticks)."""
    df = pd.DataFrame({
        "ch0": [10.0, 25.0, 10.0],
        "ch1": [20.0, 20.0, 20.0],
    })
    initial_order, raw = word_from_frames(df)
    assert initial_order == [0, 1]
    assert len(raw) == 2

    # Empty data
    empty_init, empty_raw = word_from_frames([])
    assert empty_init == []
    assert empty_raw == []

    # Single row data
    single_init, single_raw = word_from_frames([[1.0, 2.0]])
    assert single_init == [0, 1]
    assert single_raw == []

    # replay_orders with n_ticks=0
    pos_zero = replay_orders([0, 1], [], n_ticks=0)
    assert pos_zero.shape == (0, 2)

    # replay_orders with n_ticks=1
    pos_one = replay_orders([0, 1], [], n_ticks=1)
    assert pos_one.shape == (1, 2)
    assert pos_one[0, 0] == 1
    assert pos_one[0, 1] == 2
