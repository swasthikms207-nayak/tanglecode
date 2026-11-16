from __future__ import annotations

import gzip
import json
from pathlib import Path
import struct
from typing import Any
import numpy as np

from tanglecode.braid import Crossing, order_from_frame, reduce_crossings, word_from_frames
from tanglecode.codec import (
    HEADER_STRUCT,
    MAGIC,
    MODE_DEFAULT,
    encode,
    encode_leb128,
    pack_generators,
)
from tanglecode.data import generate_signals, load_csv


def run_hysteresis(
    data: np.ndarray,
    raw_crossings_count: int,
) -> tuple[int, bytes, float, float]:
    """Run hysteresis baseline without braid reduction.

    A swap is accepted only if the amplitude gap between the two adjacent
    channels exceeds h = 2 * estimated noise std
    (noise std = median over channels of std(diff(x))/sqrt(2)).
    Crossings are encoded using the same .tngl codec without braid reduction.

    Returns:
      (accepted_crossing_count, encoded_bytes, noise_std, h)
    """
    T, n = data.shape

    # 1. Noise std estimation: median over channels of std(diff(x)) / sqrt(2)
    diffs_std = [float(np.std(np.diff(data[:, c]), ddof=1) / np.sqrt(2.0)) for c in range(n)]
    noise_std = float(np.median(diffs_std))
    h = 2.0 * noise_std

    # 2. Hysteresis swap acceptance
    current_order = order_from_frame(data[0])
    accepted_crossings: list[Crossing] = []

    for t in range(1, T):
        vals = data[t]
        while True:
            swapped = False
            for k in range(n - 1):
                u = current_order[k]
                v = current_order[k + 1]
                # If channel u (at lower rank) has overtaken channel v by more than h
                if vals[u] - vals[v] > h:
                    i = k + 1
                    gen = i if u < v else -i
                    accepted_crossings.append(Crossing(tick=t, gen=gen, x=u, y=v))
                    current_order[k], current_order[k + 1] = current_order[k + 1], current_order[k]
                    swapped = True
            if not swapped:
                break

    # 3. Encode accepted crossings with the same codec and NO braid reduction
    header = struct.pack(
        HEADER_STRUCT,
        MAGIC,
        n,
        raw_crossings_count,
        len(accepted_crossings),
        T,
        MODE_DEFAULT,
    )
    anchor = struct.pack(f"<{n}f", *data[0])

    timing_bytes = bytearray()
    prev_tick = 0
    for c in accepted_crossings:
        delta = c.tick - prev_tick
        timing_bytes.extend(encode_leb128(delta))
        prev_tick = c.tick

    payload = pack_generators([c.gen for c in accepted_crossings])
    encoded_bytes = header + anchor + bytes(timing_bytes) + payload

    return len(accepted_crossings), encoded_bytes, noise_std, h


def benchmark_dataset(
    data: np.ndarray,
    raw_csv_bytes: bytes,
    dataset_name: str,
) -> dict[str, Any]:
    """Benchmark raw CSV, gzip, hysteresis baseline, and TangleCode on a dataset."""
    T, n = data.shape

    # 1. Measure raw and reduced crossings
    initial_order, raw_crossings = word_from_frames(data)
    reduced_crossings, cancelled_pairs = reduce_crossings(raw_crossings)
    raw_count = len(raw_crossings)
    reduced_count = len(reduced_crossings)

    # 2. Raw CSV
    raw_csv_size = len(raw_csv_bytes)

    # 3. gzip of CSV
    gzip_bytes = gzip.compress(raw_csv_bytes, compresslevel=6)
    gzip_size = len(gzip_bytes)

    # 4. Hysteresis baseline
    h_count, h_bytes, noise_std, h = run_hysteresis(data, raw_count)
    h_size = len(h_bytes)

    # 5. TangleCode (with reduction)
    tngl_bytes, stats = encode(data)
    tngl_size = len(tngl_bytes)

    methods = {
        "Raw CSV": {
            "bytes": raw_csv_size,
            "ratio_vs_raw_csv": 1.0,
            "ratio_vs_gzip": float(gzip_size / raw_csv_size),
            "crossings_before": raw_count,
            "crossings_after": raw_count,
        },
        "gzip (CSV)": {
            "bytes": gzip_size,
            "ratio_vs_raw_csv": float(raw_csv_size / gzip_size),
            "ratio_vs_gzip": 1.0,
            "crossings_before": raw_count,
            "crossings_after": raw_count,
        },
        "Hysteresis baseline": {
            "bytes": h_size,
            "ratio_vs_raw_csv": float(raw_csv_size / h_size),
            "ratio_vs_gzip": float(gzip_size / h_size),
            "crossings_before": raw_count,
            "crossings_after": h_count,
            "estimated_noise_std": noise_std,
            "h": h,
        },
        "TangleCode (with reduction)": {
            "bytes": tngl_size,
            "ratio_vs_raw_csv": float(raw_csv_size / tngl_size),
            "ratio_vs_gzip": float(gzip_size / tngl_size),
            "crossings_before": raw_count,
            "crossings_after": reduced_count,
        },
    }

    return {
        "dataset_name": dataset_name,
        "n_samples": T,
        "n_channels": n,
        "raw_csv_bytes": raw_csv_size,
        "gzip_bytes": gzip_size,
        "methods": methods,
    }


def print_table(results: dict[str, Any]):
    """Print an objective, neutral comparison table without claiming a winner."""
    name = results["dataset_name"]
    T = results["n_samples"]
    n = results["n_channels"]

    print("\n" + "=" * 88)
    print(f"  Dataset: {name} (T={T}, n={n})")
    print("-" * 88)
    print(
        f"{'Method':<30} | {'Bytes':>10} | {'Ratio vs CSV':>13} | {'Ratio vs gzip':>14} | {'Crossings (Before -> After)':>28}"
    )
    print("-" * 88)

    for method, metrics in results["methods"].items():
        b_str = f"{metrics['bytes']:,}"
        r_csv = f"{metrics['ratio_vs_raw_csv']:.2f}x"
        r_gz = f"{metrics['ratio_vs_gzip']:.2f}x"
        c_str = f"{metrics['crossings_before']} -> {metrics['crossings_after']}"
        print(
            f"{method:<30} | {b_str:>10} | {r_csv:>13} | {r_gz:>14} | {c_str:>28}"
        )
    print("=" * 88)


def run_all_benchmarks(
    output_json_path: str = "results.json",
) -> dict[str, Any]:
    """Run benchmark across clean, drift, and stuck data, print tables, and write results.json."""
    data_dir = Path(__file__).resolve().parent.parent / "data"

    datasets_spec = [
        ("Clean", "clean", data_dir / "test_clean.csv"),
        ("Drift", "drift", data_dir / "test_fault.csv"),
        ("Stuck", "stuck", data_dir / "test_stuck.csv"),
    ]

    benchmark_results: dict[str, Any] = {}

    for label, kind, csv_path in datasets_spec:
        if csv_path.exists():
            raw_csv_bytes = csv_path.read_bytes()
            data, _ = load_csv(csv_path)
        else:
            data = generate_signals(kind, n=6, T=3000, seed=7)
            cols = [f"sensor_{i}" for i in range(1, 7)]
            import pandas as pd
            df = pd.DataFrame(data, columns=cols).round(4)
            raw_csv_bytes = df.to_csv(index=False).encode("utf-8")

        res = benchmark_dataset(data, raw_csv_bytes, label)
        print_table(res)
        benchmark_results[kind] = res

    # Write results.json
    out_path = Path(output_json_path)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_results, f, indent=2)
    print(f"\nWritten benchmark results to {out_path.resolve()}")

    return benchmark_results


if __name__ == "__main__":
    run_all_benchmarks()
