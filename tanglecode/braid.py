from __future__ import annotations

import collections
from typing import Any, NamedTuple, Sequence
import numpy as np


class Crossing(NamedTuple):
    """Represents a single braid crossing between adjacent strands.

    Attributes:
        tick: Time index at which the crossing occurs.
        gen: Braid generator index (signed integer, ±1, ±2, ...).
        x: Channel index of strand at position i moving up.
        y: Channel index of strand at position i+1 moving down.
    """
    tick: int
    gen: int
    x: int
    y: int


def order_from_frame(frame: Any) -> list[int]:
    """Return list of channel indices sorted by value ascending (ties broken by channel index).

    Position 1 corresponds to the lowest value (index 0 in the returned list).
    """
    if hasattr(frame, "to_numpy"):
        vals = frame.to_numpy()
    elif isinstance(frame, np.ndarray):
        vals = frame
    else:
        vals = list(frame)
    n = len(vals)
    return sorted(range(n), key=lambda idx: (vals[idx], idx))


def crossings_between(
    prev_order: Sequence[int],
    curr_order: Sequence[int],
    tick: int,
) -> list[Crossing]:
    """Convert the change from prev_order to curr_order into adjacent swaps.

    Bubble-sorts prev_order to match curr_order (sort key = index in curr_order;
    sweep left to right repeatedly until sorted, deterministic).

    For each swap at 1-based position i, with x = strand at position i moving up
    and y = strand at position i+1 moving down:
      generator = +i if x < y (channel index), else -i.
    RULE: the LOWER channel index is always the over-strand, so an overtake
    followed by a drop-back gives opposite signs and cancels.

    Returns a list of Crossing tuples (tick, gen, x, y).
    """
    arr = list(prev_order)
    target_pos = {ch: idx for idx, ch in enumerate(curr_order)}
    crossings: list[Crossing] = []
    n = len(arr)

    while True:
        swapped = False
        for k in range(n - 1):
            if target_pos[arr[k]] > target_pos[arr[k + 1]]:
                i = k + 1  # 1-based position
                x = arr[k]
                y = arr[k + 1]
                gen = i if x < y else -i
                crossings.append(Crossing(tick=tick, gen=gen, x=x, y=y))
                arr[k], arr[k + 1] = arr[k + 1], arr[k]
                swapped = True
        if not swapped:
            break

    return crossings


def reduce_crossings(
    crossings: Sequence[Crossing],
) -> tuple[list[Crossing], int]:
    """Single pass with a stack to cancel commuting inverse pairs.

    For each incoming crossing c, scan the stack from the top downward.
    Skip items whose |gen| differs from |c.gen| by 2 or more (they commute with c).
    If the first non-commuting item has gen == -c.gen, remove it and count
    one cancelled pair; otherwise push c.

    Returns (reduced_list, cancelled_pairs), preserving tick, x, y on survivors.
    """
    stack: list[Crossing] = []
    cancelled_pairs = 0

    for c in crossings:
        cancelled = False
        for idx in range(len(stack) - 1, -1, -1):
            top_item = stack[idx]
            if abs(abs(top_item.gen) - abs(c.gen)) >= 2:
                # Commutes with c, skip
                continue
            else:
                # First non-commuting item
                if top_item.gen == -c.gen:
                    stack.pop(idx)
                    cancelled_pairs += 1
                    cancelled = True
                break
        if not cancelled:
            stack.append(c)

    return stack, cancelled_pairs


def replay_orders(
    initial_order: Sequence[int],
    reduced_crossings: Sequence[Crossing],
    n_ticks: int,
) -> np.ndarray:
    """Apply the reduced word to initial order and return an (n_ticks, n) array of positions.

    Each row t gives the rank (1-based position) of each channel at tick t.
    """
    n = len(initial_order)
    positions = np.zeros((n_ticks, n), dtype=int)
    if n_ticks <= 0:
        return positions

    curr_order = list(initial_order)
    crossings_by_tick: dict[int, list[Crossing]] = collections.defaultdict(list)
    for c in reduced_crossings:
        if 0 <= c.tick < n_ticks:
            crossings_by_tick[c.tick].append(c)

    # Any crossings at tick 0
    for c in crossings_by_tick.get(0, []):
        idx = abs(c.gen) - 1
        curr_order[idx], curr_order[idx + 1] = curr_order[idx + 1], curr_order[idx]

    # Positions at tick 0
    for rank, ch in enumerate(curr_order, start=1):
        positions[0, ch] = rank

    # Advance tick by tick
    for t in range(1, n_ticks):
        for c in crossings_by_tick.get(t, []):
            idx = abs(c.gen) - 1
            curr_order[idx], curr_order[idx + 1] = curr_order[idx + 1], curr_order[idx]
        for rank, ch in enumerate(curr_order, start=1):
            positions[t, ch] = rank

    return positions


def word_from_frames(data: Any) -> tuple[list[int], list[Crossing]]:
    """Run order_from_frame on every row and return (initial_order, raw_crossings)."""
    if hasattr(data, "to_numpy"):
        frames = data.to_numpy()
    elif isinstance(data, np.ndarray):
        frames = data
    else:
        frames = list(data)

    n_ticks = len(frames)
    if n_ticks == 0:
        return [], []

    initial_order = order_from_frame(frames[0])
    raw_crossings: list[Crossing] = []

    prev_order = initial_order
    for tick in range(1, n_ticks):
        curr_order = order_from_frame(frames[tick])
        crossings = crossings_between(prev_order, curr_order, tick=tick)
        raw_crossings.extend(crossings)
        prev_order = curr_order

    return initial_order, raw_crossings
