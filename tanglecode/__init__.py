"""Tanglecode: braid analysis, data simulation, and topological codec."""

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
    "encode",
    "decode",
    "generate_signals",
    "load_csv",
]


def __getattr__(name: str) -> Any:
    if name in ("generate_signals", "load_csv"):
        import tanglecode.data as _data

        return getattr(_data, name)
    if name in ("encode", "decode"):
        import tanglecode.codec as _codec

        return getattr(_codec, name)
    raise AttributeError(f"module 'tanglecode' has no attribute '{name}'")
