"""Export utilities for TangleCode anomaly reports and summary benchmarks."""

import csv
import gzip
import io
import json
from typing import Any, Dict, List, Optional, Tuple, Union


def anomaly_report_csv(results: Dict[str, Any]) -> str:
    """Generate CSV report of flagged anomaly windows.

    Columns:
    window_start_tick, window_end_tick, score, threshold, most_likely_channel, top_crossing_pair.

    If no windows are flagged, returns a CSV containing only the header row.

    Parameters
    ----------
    results : dict
        Results dictionary containing flagged window diagnostic details.

    Returns
    -------
    str
        UTF-8 CSV formatted text string.
    """
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "window_start_tick",
        "window_end_tick",
        "score",
        "threshold",
        "most_likely_channel",
        "top_crossing_pair",
    ])

    flagged_diagnoses = results.get("flagged_diagnoses", [])
    for diag in flagged_diagnoses:
        w_start = diag.get("window_start_tick", 0)
        w_end = diag.get("window_end_tick", 0)
        score = diag.get("score", 0.0)
        threshold = diag.get("threshold", 0.0)
        ch = diag.get("most_likely_channel", None)
        pair = diag.get("top_crossing_pair", None)

        ch_str = f"Channel {ch + 1}" if isinstance(ch, int) else (str(ch) if ch is not None else "")
        if pair is not None and len(pair) == 2:
            pair_str = f"Ch {pair[0] + 1} - Ch {pair[1] + 1}"
        else:
            pair_str = ""

        writer.writerow([
            w_start,
            w_end,
            f"{score:.4f}" if isinstance(score, (int, float)) else score,
            f"{threshold:.4f}" if isinstance(threshold, (int, float)) else threshold,
            ch_str,
            pair_str,
        ])

    return output.getvalue()


def summary_json(results: Dict[str, Any]) -> str:
    """Generate comprehensive JSON summary of dataset compression and fault detection.

    Fields included:
    dataset name, channel count, sample count, raw crossings, reduced crossings,
    percent eliminated, CSV bytes, gzip bytes, .tngl bytes, list of anomaly tick
    ranges, and the faulty channel (or null).

    Parameters
    ----------
    results : dict
        Results dictionary containing metrics and anomaly details.

    Returns
    -------
    str
        Pretty-printed JSON string ready for download.
    """
    raw_crossings = int(results.get("raw_crossings", 0))
    reduced_crossings = int(results.get("reduced_crossings", 0))
    if "percent_eliminated" in results:
        pct_elim = float(results["percent_eliminated"])
    elif raw_crossings > 0:
        pct_elim = round((raw_crossings - reduced_crossings) / raw_crossings * 100.0, 2)
    else:
        pct_elim = 0.0

    csv_bytes = int(results.get("raw_csv_size", results.get("csv_bytes", 0)))
    gzip_bytes = results.get("gzip_bytes", None)
    if gzip_bytes is None:
        raw_csv_data = results.get("raw_csv_bytes", None)
        if raw_csv_data is not None:
            gzip_bytes = len(gzip.compress(raw_csv_data))
        else:
            gzip_bytes = int(csv_bytes * 0.4) if csv_bytes > 0 else 0
    else:
        gzip_bytes = int(gzip_bytes)

    tngl_bytes = int(results.get("tngl_size", results.get("tngl_bytes", 0)))

    # Anomaly tick ranges
    tick_ranges = results.get("anomaly_tick_ranges", None)
    if tick_ranges is None:
        flagged_intervals = results.get("flagged_intervals", [])
        tick_ranges = [list(interval) for interval in flagged_intervals]

    # Faulty channel: 1-indexed string or 0-indexed int or None
    faulty_channel = results.get("faulty_channel", None)
    if faulty_channel is not None and isinstance(faulty_channel, int):
        faulty_channel_name = f"Channel {faulty_channel + 1}"
    else:
        faulty_channel_name = faulty_channel

    summary_dict = {
        "dataset_name": results.get("dataset_name", "dataset"),
        "channel_count": int(results.get("n_channels", results.get("channel_count", 0))),
        "sample_count": int(results.get("n_samples", results.get("sample_count", 0))),
        "raw_crossings": raw_crossings,
        "reduced_crossings": reduced_crossings,
        "percent_eliminated": pct_elim,
        "csv_bytes": csv_bytes,
        "gzip_bytes": gzip_bytes,
        "tngl_bytes": tngl_bytes,
        "anomaly_tick_ranges": tick_ranges,
        "faulty_channel": faulty_channel_name,
    }

    return json.dumps(summary_dict, indent=2)
