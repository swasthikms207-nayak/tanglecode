from __future__ import annotations

from typing import Any, Sequence
import numpy as np
import pandas as pd

from tanglecode.braid import order_from_frame


def get_sensor_ranking(
    data: np.ndarray,
    tick: int,
    ranges: Sequence[dict[str, Any]] | None = None,
) -> pd.DataFrame:
    """Build a sensor ranking dataframe at a given tick, sorted lowest to highest reading.

    Args:
        data: Multichannel numpy array of shape (T, n).
        tick: Time tick index (0 <= tick < T).
        ranges: Optional list of flagged fault range dictionaries.

    Returns:
        DataFrame with columns ['Place', 'Sensor', 'Reading'].
    """
    row_vals = data[tick]
    order = order_from_frame(row_vals)

    # Check if tick is inside any flagged fault range
    is_suspect_tick = False
    suspect_channel = None
    if ranges:
        for r in ranges:
            if r["start_tick"] <= tick <= r["end_tick"]:
                is_suspect_tick = True
                suspect_channel = r.get("faulty_channel")
                break

    rows: list[dict[str, Any]] = []
    for place, ch_idx in enumerate(order, start=1):
        ch_num = ch_idx + 1
        reading = round(float(row_vals[ch_idx]), 4)
        if is_suspect_tick and ch_num == suspect_channel:
            sensor_name = f"Sensor {ch_num} (suspect)"
        else:
            sensor_name = f"Sensor {ch_num}"

        rows.append({
            "Place": place,
            "Sensor": sensor_name,
            "Reading": reading,
        })

    return pd.DataFrame(rows)


def get_place_changes_caption(data: np.ndarray, tick: int) -> str | None:
    """Compute the caption describing sensor rank changes compared to 100 ticks earlier.

    Returns None for tick < 100.
    """
    if tick < 100:
        return None

    curr_order = order_from_frame(data[tick])
    prev_order = order_from_frame(data[tick - 100])

    curr_places = {ch + 1: place for place, ch in enumerate(curr_order, start=1)}
    prev_places = {ch + 1: place for place, ch in enumerate(prev_order, start=1)}

    diff = [
        (ch, prev_places[ch], curr_places[ch])
        for ch in sorted(curr_places.keys())
        if prev_places[ch] != curr_places[ch]
    ]

    if not diff:
        return "Since 100 ticks ago: no sensor changed place."

    parts = [f"Sensor {ch} moved from place {p0} to place {p1}" for ch, p0, p1 in diff]
    return f"Since 100 ticks ago: {', '.join(parts)}."
