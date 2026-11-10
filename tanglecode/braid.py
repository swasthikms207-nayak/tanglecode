"""Braid group representations and topological reductions for multichannel time series."""

from typing import Sequence


class ReductionResult(tuple):
    """Result of word reduction containing the reduced word and cancelled pair count.

    Behaves as a 2-tuple `(reduced_word, cancelled_pair_count)`.
    Also allows attribute access `.word` and `.cancelled_pair_count`,
    and direct comparison with either a tuple or a list of generators.
    """

    def __new__(cls, reduced_word: Sequence[int], cancelled_pair_count: int):
        return super().__new__(cls, (list(reduced_word), int(cancelled_pair_count)))

    @property
    def word(self) -> list[int]:
        return self[0]

    @property
    def cancelled_pair_count(self) -> int:
        return self[1]

    def __eq__(self, other) -> bool:
        if isinstance(other, list):
            return self[0] == other
        return super().__eq__(other)

    def __repr__(self) -> str:
        return f"({self[0]!r}, {self[1]!r})"


def rank_vector(frame: Sequence[float]) -> list[int]:
    """Returns the argsort ranking of a list of n floats (ties broken by channel index).

    Parameters
    ----------
    frame : Sequence[float]
        Float values across n channels [val_0, val_1, ..., val_{n-1}].

    Returns
    -------
    list[int]
        Permutation of channel indices (0 to n-1) ordered from lowest to highest value.
        Ties are broken deterministically by smaller channel index first.
    """
    return sorted(range(len(frame)), key=lambda i: (frame[i], i))


def extract_braid_word(prev_ranks: Sequence[int], curr_ranks: Sequence[int]) -> list[int]:
    """Converts the change in ordering into a list of signed Artin generators.

    Uses bubble sort with adjacent swaps to transform `prev_ranks` into `curr_ranks`.
    Generators are represented as nonzero ints:
    +i represents sigma_i, and -i represents sigma_i^-1 (i from 1 to n-1).

    Sign rule:
    When two adjacent strands swap (at positions j and j+1), the strand moving UP
    in rank (from index j to j+1) is checked. By deterministic convention, between
    the two strands, the one with the smaller channel index crosses over.
    If the strand moving UP in rank crosses over, emit +i (where i = j + 1);
    otherwise emit -i.

    Parameters
    ----------
    prev_ranks : Sequence[int]
        Ordering of strands at previous time step.
    curr_ranks : Sequence[int]
        Ordering of strands at current time step.

    Returns
    -------
    list[int]
        List of signed Artin generators.
    """
    if len(prev_ranks) != len(curr_ranks):
        raise ValueError("prev_ranks and curr_ranks must have equal length")

    n = len(prev_ranks)
    if n <= 1:
        return []

    target_pos = {strand: idx for idx, strand in enumerate(curr_ranks)}
    arr = list(prev_ranks)
    word: list[int] = []

    for pass_num in range(n - 1):
        for j in range(n - 1 - pass_num):
            if target_pos[arr[j]] > target_pos[arr[j + 1]]:
                strand_up = arr[j]        # Moves from j to j+1 (UP in rank)
                strand_down = arr[j + 1]  # Moves from j+1 to j (DOWN in rank)

                # Deterministic rule: strand with smaller channel index crosses over
                crosses_over = (strand_up < strand_down)
                gen_idx = j + 1
                generator = gen_idx if crosses_over else -gen_idx
                word.append(generator)

                # Execute swap
                arr[j], arr[j + 1] = arr[j + 1], arr[j]

    return word


def reduce_word(word: Sequence[int]) -> ReductionResult:
    """Online single-pass reduction using a stack with free cancellation.

    If top == -g, pop (Reidemeister II move: sigma_i * sigma_i^-1 = identity).

    Parameters
    ----------
    word : Sequence[int]
        List of signed Artin generators.

    Returns
    -------
    ReductionResult
        (reduced_word, cancelled_pair_count).
    """
    stack: list[int] = []
    cancelled = 0

    for g in word:
        if g == 0:
            continue
        if stack and stack[-1] == -g:
            stack.pop()
            cancelled += 1
        else:
            stack.append(g)

    return ReductionResult(stack, cancelled)


def _canonical_sort_commuting(w: Sequence[int]) -> tuple[list[int], bool]:
    """Sort adjacent commuting generators into canonical order.

    Generators g1, g2 commute when |abs(g1) - abs(g2)| >= 2.
    Canonical order places the generator with smaller absolute value first.
    """
    arr = list(w)
    n = len(arr)
    any_swap = False

    while True:
        pass_swapped = False
        for i in range(n - 1):
            if abs(abs(arr[i]) - abs(arr[i + 1])) >= 2:
                if abs(arr[i]) > abs(arr[i + 1]):
                    arr[i], arr[i + 1] = arr[i + 1], arr[i]
                    pass_swapped = True
                    any_swap = True
        if not pass_swapped:
            break

    return arr, any_swap


def _reduce_with_commutativity(word: Sequence[int]) -> tuple[list[int], int]:
    """Repeatedly run stack cancellation and canonical commuting sort until convergence."""
    curr = list(word)
    total_cancelled = 0

    while True:
        # Run free cancellation
        curr, cancelled = reduce_word(curr)
        total_cancelled += cancelled

        # Canonical sort commuting neighbors
        sorted_curr, commuted = _canonical_sort_commuting(curr)

        # Check if cancellation is possible after sort
        test_curr, cancelled2 = reduce_word(sorted_curr)
        if cancelled2 > 0:
            curr = list(test_curr)
            total_cancelled += cancelled2
            continue

        curr = sorted_curr
        if cancelled == 0 and not commuted:
            break

    return curr, total_cancelled


def reduce_word_full(word: Sequence[int]) -> ReductionResult:
    """Extends reduce_word with far-commutativity and conditional braid relations.

    1. Far-commutativity (sigma_i sigma_j = sigma_j sigma_i when |i-j| >= 2):
       Canonically sorts commuting neighbors, then re-runs cancellation until no change.
    2. Braid relation (sigma_i sigma_{i+1} sigma_i -> sigma_{i+1} sigma_i sigma_{i+1}):
       Applied only when it enables a further cancellation.

    Parameters
    ----------
    word : Sequence[int]
        List of signed Artin generators.

    Returns
    -------
    ReductionResult
        (reduced_word, cancelled_pair_count).
    """
    curr, total_cancelled = _reduce_with_commutativity(word)

    while True:
        braid_applied = False
        n = len(curr)

        # Look for braid relation triples: g1 == g3, |abs(g1) - abs(g2)| == 1, same signs
        for k in range(n - 2):
            g1, g2, g3 = curr[k], curr[k + 1], curr[k + 2]
            if g1 == g3 and abs(abs(g1) - abs(g2)) == 1 and ((g1 > 0 and g2 > 0) or (g1 < 0 and g2 < 0)):
                # Equivalent replacement triple: g2, g1, g2
                candidate = curr[:k] + [g2, g1, g2] + curr[k + 3:]
                cand_reduced, cand_cancelled = _reduce_with_commutativity(candidate)

                if len(cand_reduced) < len(curr):
                    # Braid relation enabled a further cancellation!
                    curr = cand_reduced
                    total_cancelled += cand_cancelled
                    braid_applied = True
                    break

        if not braid_applied:
            break

    return ReductionResult(curr, total_cancelled)


def extract_crossings_with_channels(
    prev_ranks: Sequence[int],
    curr_ranks: Sequence[int],
) -> list[tuple[int, int, int]]:
    """Converts the change in ordering into signed Artin generators with channel pairs.

    Same logic as extract_braid_word, but each crossing is returned as a tuple
    (generator, channel_a, channel_b) where channel_a and channel_b are the two
    sensor channel indices that swapped.
    """
    if len(prev_ranks) != len(curr_ranks):
        raise ValueError("prev_ranks and curr_ranks must have equal length")

    n = len(prev_ranks)
    if n <= 1:
        return []

    target_pos = {strand: idx for idx, strand in enumerate(curr_ranks)}
    arr = list(prev_ranks)
    crossings: list[tuple[int, int, int]] = []

    for pass_num in range(n - 1):
        for j in range(n - 1 - pass_num):
            if target_pos[arr[j]] > target_pos[arr[j + 1]]:
                strand_up = arr[j]        # Moves from j to j+1 (UP in rank)
                strand_down = arr[j + 1]  # Moves from j+1 to j (DOWN in rank)

                # Deterministic rule: strand with smaller channel index crosses over
                crosses_over = (strand_up < strand_down)
                gen_idx = j + 1
                generator = gen_idx if crosses_over else -gen_idx
                crossings.append((generator, strand_up, strand_down))

                # Execute swap
                arr[j], arr[j + 1] = arr[j + 1], arr[j]

    return crossings


def reduce_with_channels(
    crossings: Sequence[tuple[int, int, int]],
) -> tuple[list[tuple[int, int, int]], int]:
    """Stack-based cancellation preserving channel pairs for surviving crossings.

    Cancellation happens when the top generator equals the negative of the
    incoming generator (top[0] == -incoming[0]).

    Returns
    -------
    tuple[list[tuple[int, int, int]], int]
        (reduced_crossings, cancelled_pair_count).
    """
    stack: list[tuple[int, int, int]] = []
    cancelled = 0

    for item in crossings:
        g = item[0]
        if g == 0:
            continue
        if stack and stack[-1][0] == -g:
            stack.pop()
            cancelled += 1
        else:
            stack.append(item)

    return stack, cancelled


__all__ = [
    "ReductionResult",
    "rank_vector",
    "extract_braid_word",
    "reduce_word",
    "reduce_word_full",
    "extract_crossings_with_channels",
    "reduce_with_channels",
]


