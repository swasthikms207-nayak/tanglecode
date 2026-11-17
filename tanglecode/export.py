from __future__ import annotations

import csv
import io
from typing import Any, Sequence
from tanglecode.braid import Crossing


def swap_log_csv(
    reduced_crossings: Sequence[Crossing | Any],
    sampling_rate: float | None = None,
) -> str:
    """Export surviving crossings to CSV swap-log format.

    Columns:
      tick, (time_s only if sampling_rate is given, one decimal), sensor_moved_up, sensor_moved_down, position

    Sensor numbers are 1-based (c.x + 1, c.y + 1).
    Position is 1-based (abs(c.gen)).
    Returns header only if there are no crossings.
    """
    has_rate = sampling_rate is not None and float(sampling_rate) > 0

    if has_rate:
        headers = ["tick", "time_s", "sensor_moved_up", "sensor_moved_down", "position"]
    else:
        headers = ["tick", "sensor_moved_up", "sensor_moved_down", "position"]

    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow(headers)

    for c in reduced_crossings:
        tick = int(c.tick) if hasattr(c, "tick") else int(c[0])
        gen = int(c.gen) if hasattr(c, "gen") else int(c[1])
        x = int(c.x) if hasattr(c, "x") else int(c[2])
        y = int(c.y) if hasattr(c, "y") else int(c[3])

        sensor_up = x + 1
        sensor_down = y + 1
        position = abs(gen)

        if has_rate:
            time_s = f"{tick / float(sampling_rate):.1f}"
            writer.writerow([tick, time_s, sensor_up, sensor_down, position])
        else:
            writer.writerow([tick, sensor_up, sensor_down, position])

    return out.getvalue()
