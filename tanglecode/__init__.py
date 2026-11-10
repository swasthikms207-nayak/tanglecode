"""TangleCode: Braid representations and topological reductions for multichannel time series."""

from tanglecode.anomaly import (
    AnomalyRecord,
    BraidAnomalyDetector,
    ChannelFaultDiagnosis,
    compute_window_features,
    count_surviving_crossings_per_channel,
    detect_anomalies,
    diagnose_window_fault,
    learn_channel_baseline,
)
from tanglecode.braid import (
    ReductionResult,
    extract_braid_word,
    extract_crossings_with_channels,
    rank_vector,
    reduce_with_channels,
    reduce_word,
    reduce_word_full,
)
from tanglecode.codec import (
    MODE_RANK_ONLY,
    decode,
    encode,
)
from tanglecode.data import (
    generate_synthetic,
    inject_fault,
    load_csv,
)
from tanglecode.reports import (
    anomaly_report_csv,
    summary_json,
)

__all__ = [
    "rank_vector",
    "extract_braid_word",
    "extract_crossings_with_channels",
    "reduce_word",
    "reduce_with_channels",
    "reduce_word_full",
    "ReductionResult",
    "generate_synthetic",
    "inject_fault",
    "load_csv",
    "encode",
    "decode",
    "MODE_RANK_ONLY",
    "AnomalyRecord",
    "BraidAnomalyDetector",
    "compute_window_features",
    "detect_anomalies",
    "count_surviving_crossings_per_channel",
    "learn_channel_baseline",
    "diagnose_window_fault",
    "ChannelFaultDiagnosis",
    "anomaly_report_csv",
    "summary_json",
]
