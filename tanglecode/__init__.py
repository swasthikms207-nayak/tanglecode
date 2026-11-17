"""Tanglecode: braid analysis, data simulation, swap-log export, and anomaly detection."""

from typing import Any

from tanglecode.braid import (
    Crossing,
    crossings_between,
    order_from_frame,
    reduce_crossings,
    replay_orders,
    word_from_frames,
)

__all__ = [
    "Crossing",
    "order_from_frame",
    "crossings_between",
    "reduce_crossings",
    "replay_orders",
    "word_from_frames",
    "swap_log_csv",
    "compute_stats",
    "generate_signals",
    "load_csv",
    "detect",
    "make_report_pdf",
]


def __getattr__(name: str) -> Any:
    if name in ("generate_signals", "load_csv"):
        import tanglecode.data as _data

        return getattr(_data, name)
    if name == "swap_log_csv":
        import tanglecode.export as _export

        return getattr(_export, name)
    if name == "compute_stats":
        import tanglecode.pipeline as _pipeline

        return getattr(_pipeline, name)
    if name == "detect":
        import tanglecode.anomaly as _anomaly

        return getattr(_anomaly, name)
    if name == "make_report_pdf":
        import tanglecode.report as _report

        return getattr(_report, name)
    raise AttributeError(f"module 'tanglecode' has no attribute '{name}'")
