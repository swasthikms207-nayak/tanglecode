"""Tests for tanglecode.braid module."""

import pytest
from tanglecode.braid import (
    ReductionResult,
    extract_braid_word,
    rank_vector,
    reduce_word,
    reduce_word_full,
)


def test_rank_vector_basic():
    """Verify rank_vector returns argsort permutation correctly."""
    frame = [30.0, 10.0, 20.0]
    ranks = rank_vector(frame)
    # Channel 1 (10.0) < Channel 2 (20.0) < Channel 0 (30.0)
    assert ranks == [1, 2, 0]


def test_rank_vector_ties():
    """Verify ties are broken deterministically by channel index."""
    # Channel 0 and 2 have identical value 5.0, Channel 1 has 1.0, Channel 3 has 5.0
    frame = [5.0, 1.0, 5.0, 5.0]
    ranks = rank_vector(frame)
    # 1.0 at index 1 comes first
    # 5.0 at indices 0, 2, 3 ordered by channel index
    assert ranks == [1, 0, 2, 3]


def test_identity_word():
    """Verify that identity words (no change in ranking or empty) reduce to empty."""
    # 1. No change in rank vector
    prev_ranks = [0, 1, 2, 3]
    curr_ranks = [0, 1, 2, 3]
    word = extract_braid_word(prev_ranks, curr_ranks)
    assert word == []

    # 2. Empty word reduction
    reduced, count = reduce_word([])
    assert reduced == []
    assert count == 0

    # 3. Full reduction of empty word
    reduced_full, count_full = reduce_word_full([])
    assert reduced_full == []
    assert count_full == 0


def test_overtake_and_return_jitter_loop_collapsing_to_empty():
    """Verify an overtake followed by return (jitter loop) collapses to empty."""
    # Two strands: 0 and 1
    # Frame 0: Channel 0 is below Channel 1
    frame0 = [10.0, 20.0]
    ranks0 = rank_vector(frame0)  # [0, 1]

    # Frame 1: Channel 0 overtakes Channel 1
    frame1 = [25.0, 20.0]
    ranks1 = rank_vector(frame1)  # [1, 0]
    w1 = extract_braid_word(ranks0, ranks1)
    # Strand 0 moved UP in rank and has lower channel index (0 < 1), so it crosses over: +1
    assert w1 == [1]

    # Frame 2: Channel 0 returns below Channel 1 (jitter returned to initial state)
    frame2 = [10.0, 20.0]
    ranks2 = rank_vector(frame2)  # [0, 1]
    w2 = extract_braid_word(ranks1, ranks2)
    # Strand 1 moved UP in rank, but Strand 0 crosses over (0 < 1), so Strand 1 crosses under: -1
    assert w2 == [-1]

    # Total jitter word
    jitter_word = w1 + w2  # [1, -1]

    # Test online single-pass reduce_word
    reduced, cancelled_count = reduce_word(jitter_word)
    assert reduced == []
    assert cancelled_count == 1

    # Test full reduction
    reduced_full, full_count = reduce_word_full(jitter_word)
    assert reduced_full == []
    assert full_count == 1


def test_commuting_generators_reducing():
    """Verify that far-commuting generators (|i-j| >= 2) allow cancellations."""
    # Generators 1 and 3 commute: sigma_1 and sigma_3 (|1 - 3| = 2 >= 2)
    # In [1, 3, -1], single-pass reduce_word cannot cancel because 3 separates 1 and -1:
    word1 = [1, 3, -1]
    res_simple, cnt_simple = reduce_word(word1)
    assert res_simple == [1, 3, -1]
    assert cnt_simple == 0

    # But reduce_word_full uses far-commutativity to canonically sort commuting neighbors,
    # moving -1 adjacent to 1, causing cancellation:
    res_full1, cnt_full1 = reduce_word_full(word1)
    assert res_full1 == [3]
    assert cnt_full1 == 1

    # Similarly: [3, 1, -3] -> 3 and 1 commute -> [1, 3, -3] -> cancels 3,-3 -> [1]
    word2 = [3, 1, -3]
    res_full2, cnt_full2 = reduce_word_full(word2)
    assert res_full2 == [1]
    assert cnt_full2 == 1

    # Double commuting cancellation: [1, 4, -4, -1] -> [1, -1] -> []
    word3 = [1, 4, -4, -1]
    res_full3, cnt_full3 = reduce_word_full(word3)
    assert res_full3 == []
    assert cnt_full3 == 2


def test_real_non_jitter_crossing_preserved():
    """Verify that persistent, real crossings across frames are preserved."""
    # 3 channels: 0, 1, 2
    # Frame 0: Channels in order 0, 1, 2
    frame0 = [10.0, 20.0, 30.0]
    ranks0 = rank_vector(frame0)  # [0, 1, 2]

    # Frame 1: Channel 0 overtakes Channel 1 (real crossing)
    frame1 = [25.0, 20.0, 30.0]
    ranks1 = rank_vector(frame1)  # [1, 0, 2]
    w1 = extract_braid_word(ranks0, ranks1)  # [1]

    # Frame 2: Channel 2 drops below Channel 0 (another real crossing at sigma_2)
    frame2 = [25.0, 20.0, 22.0]
    ranks2 = rank_vector(frame2)  # [1, 2, 0]
    w2 = extract_braid_word(ranks1, ranks2)

    total_word = w1 + w2
    # Both crossings are genuine topological events and do not cancel each other
    reduced, cancelled = reduce_word_full(total_word)
    assert reduced == total_word
    assert cancelled == 0
    assert len(reduced) > 0


def test_braid_relation_enables_cancellation():
    """Verify that the braid relation sigma_i sigma_{i+1} sigma_i -> sigma_{i+1} sigma_i sigma_{i+1}

    is applied when it enables further cancellation.
    """
    # [1, 2, 1, -2] -> applying braid relation (1, 2, 1) -> (2, 1, 2)
    # results in [2, 1, 2, -2] -> 2 and -2 cancel -> [2, 1]
    word = [1, 2, 1, -2]
    # Single-pass cannot reduce this
    res_simple, cnt_simple = reduce_word(word)
    assert res_simple == [1, 2, 1, -2]
    assert cnt_simple == 0

    # Full reduction applies braid relation to enable cancellation
    res_full, cnt_full = reduce_word_full(word)
    assert res_full == [2, 1]
    assert cnt_full == 1

    # Reverse direction: [2, 1, 2, -1] -> [1, 2, 1, -1] -> [1, 2]
    word2 = [2, 1, 2, -1]
    res_full2, cnt_full2 = reduce_word_full(word2)
    assert res_full2 == [1, 2]
    assert cnt_full2 == 1

    # Braid relation with negative generators:
    # [-1, -2, -1, 2] -> [-2, -1, -2, 2] -> [-2, -1]
    word3 = [-1, -2, -1, 2]
    res_full3, cnt_full3 = reduce_word_full(word3)
    assert res_full3 == [-2, -1]
    assert cnt_full3 == 1

    # If braid relation does NOT enable cancellation, it is not applied
    unrelated = [1, 2, 1, 3]
    res_no_cancel, cnt_no_cancel = reduce_word_full(unrelated)
    assert res_no_cancel == [1, 2, 1, 3]
    assert cnt_no_cancel == 0


def test_reduction_result_ergonomics():
    """Verify ReductionResult supports tuple unpacking, indexing, properties, and list equality."""
    res = ReductionResult([1, 2], 3)
    # Tuple unpacking
    w, c = res
    assert w == [1, 2]
    assert c == 3

    # Indexing
    assert res[0] == [1, 2]
    assert res[1] == 3

    # Properties
    assert res.word == [1, 2]
    assert res.cancelled_pair_count == 3

    # Direct list comparison
    assert res == [1, 2]

    # Direct tuple comparison
    assert res == ([1, 2], 3)


def test_extract_braid_word_validation():
    """Verify error handling when input lengths mismatch."""
    with pytest.raises(ValueError, match="equal length"):
        extract_braid_word([0, 1], [0, 1, 2])


def test_large_permutation_reversal():
    """Verify bubble sort and generator signs on 4-element full reversal."""
    prev = [0, 1, 2, 3]
    curr = [3, 2, 1, 0]
    word = extract_braid_word(prev, curr)
    # Number of inversions for reverse permutation of 4 elements is 4*3/2 = 6
    assert len(word) == 6
    # All generators should be in valid range [1, 3]
    assert all(1 <= abs(g) <= 3 for g in word)

    # Reversal back to initial
    word_back = extract_braid_word(curr, prev)
    assert len(word_back) == 6

    # Full round trip jitter loop collapses under reduce_word_full via commutativity
    round_trip = word + word_back
    # Single-pass cancels the adjacent pairs at the boundary
    reduced_sp, count_sp = reduce_word(round_trip)
    assert count_sp == 2
    assert len(reduced_sp) == 8

    # Full reduction uses far-commutativity to completely collapse all 6 pairs
    reduced_full, count_full = reduce_word_full(round_trip)
    assert reduced_full == []
    assert count_full == 6

