"""TangleCode: Braid representations and topological reductions for multichannel time series."""

from tanglecode.braid import (
    ReductionResult,
    extract_braid_word,
    rank_vector,
    reduce_word,
    reduce_word_full,
)

__all__ = [
    "rank_vector",
    "extract_braid_word",
    "reduce_word",
    "reduce_word_full",
    "ReductionResult",
]
