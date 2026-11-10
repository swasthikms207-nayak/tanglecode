"""TangleCode: Braid representations and topological reductions for multichannel time series."""

from tanglecode.anomaly import (
    AnomalyRecord,
    BraidAnomalyDetector,
    compute_window_features,
    detect_anomalies,
)
from tanglecode.braid import (
    ReductionResult,
    extract_braid_word,
    rank_vector,
    reduce_word,
    reduce_word_full,
)
from tanglecode.codec import (
    MODE_HYBRID,
    MODE_RANK_ONLY,
    decode,
    encode,
)
from tanglecode.data import (
    generate_synthetic,
    inject_fault,
    load_csv,
)

__all__ = [
    "rank_vector",
    "extract_braid_word",
    "reduce_word",
    "reduce_word_full",
    "ReductionResult",
    "generate_synthetic",
    "inject_fault",
    "load_csv",
    "encode",
    "decode",
    "MODE_RANK_ONLY",
    "MODE_HYBRID",
    "AnomalyRecord",
    "BraidAnomalyDetector",
    "compute_window_features",
    "detect_anomalies",
]
