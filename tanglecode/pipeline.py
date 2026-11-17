from __future__ import annotations

from typing import Any, Sequence
import numpy as np

from tanglecode.braid import (
    Crossing,
    reduce_crossings,
    word_from_frames,
)


def compute_stats(
    raw: Sequence[Crossing | Any],
    reduced: Sequence[Crossing | Any],
    cancelled_pairs: int,
) -> dict[str, Any]:
    """Compute summary statistics for raw and reduced braid crossings.

    Returns:
        raw_crossings: total number of raw crossings
        reduced_crossings: number of surviving crossings
        cancelled_pairs: number of cancelling crossing pairs removed
        percent_eliminated: percentage of raw crossings eliminated
    """
    raw_crossings = len(raw)
    reduced_crossings = len(reduced)
    percent_eliminated = (
        ((raw_crossings - reduced_crossings) / raw_crossings * 100.0)
        if raw_crossings > 0
        else 0.0
    )
    return {
        "raw_crossings": raw_crossings,
        "reduced_crossings": reduced_crossings,
        "cancelled_pairs": cancelled_pairs,
        "percent_eliminated": percent_eliminated,
    }


def analyze_swaps(data: np.ndarray) -> tuple[list[int], list[Crossing], list[Crossing], int, dict[str, Any]]:
    """Run braid extraction, stack reduction, and calculate statistics from raw multichannel frames."""
    initial_order, raw = word_from_frames(data)
    reduced, cancelled_pairs = reduce_crossings(raw)
    stats = compute_stats(raw, reduced, cancelled_pairs)
    return initial_order, raw, reduced, cancelled_pairs, stats
