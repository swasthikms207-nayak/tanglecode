"""Benchmark comparing TangleCode rank-only against raw CSV, gzip, and hysteresis."""

import gzip
import io
import json
import struct
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from tanglecode.braid import extract_braid_word, rank_vector
from tanglecode.codec import (
    HEADER_STRUCT,
    MAGIC,
    MODE_RANK_ONLY,
    _pack_generators,
    _rle_encode,
    encode,
)
from tanglecode.data import generate_synthetic, load_csv


def _format_raw_csv_bytes(data: np.ndarray) -> bytes:
    """Format numpy array into UTF-8 CSV text bytes."""
    n_samples, n_channels = data.shape
    output = io.StringIO()
    header = ",".join(f"Ch_{c}" for c in range(n_channels))
    output.write(header + "\n")
    for row in data:
        output.write(",".join(f"{val:.6f}" for val in row) + "\n")
    return output.getvalue().encode("utf-8")


def _run_hysteresis_baseline(
    data: np.ndarray,
    threshold: float = 0.05,
) -> Tuple[int, bytes]:
    """Simple hysteresis baseline:

    Only register a rank swap if the amplitude difference exceeds a threshold,
    then apply NO braid reduction.
    """
    n_samples, n_channels = data.shape
    current_order = rank_vector(data[0])

    hysteresis_generators: list[int] = []
    per_tick_counts: list[int] = []

    for t in range(n_samples - 1):
        vals = data[t + 1]
        swaps_this_tick = 0

        for pass_num in range(n_channels - 1):
            for j in range(n_channels - 1 - pass_num):
                ch_up = current_order[j]
                ch_down = current_order[j + 1]
                diff = vals[ch_up] - vals[ch_down]
                if diff > threshold:
                    current_order[j], current_order[j + 1] = current_order[j + 1], current_order[j]
                    crosses_over = ch_up < ch_down
                    gen_idx = j + 1
                    generator = gen_idx if crosses_over else -gen_idx
                    hysteresis_generators.append(generator)
                    swaps_this_tick += 1

        per_tick_counts.append(swaps_this_tick)

    header = struct.pack(
        HEADER_STRUCT,
        MAGIC,
        n_channels,
        len(hysteresis_generators),
        len(hysteresis_generators),
        MODE_RANK_ONLY,
    )
    anchor = struct.pack(f"<{n_channels}f", *data[0])
    timing = _rle_encode(per_tick_counts)
    payload = _pack_generators(hysteresis_generators)
    packaged_bytes = header + anchor + timing + payload

    return len(hysteresis_generators), packaged_bytes


def benchmark_dataset(data: np.ndarray, dataset_name: str) -> Dict[str, Any]:
    """Run benchmark comparing Raw CSV, gzip, hysteresis, and TangleCode rank-only."""
    n_samples, n_channels = data.shape

    # 1. Measure raw crossings
    total_raw_crossings = 0
    for t in range(n_samples - 1):
        w = extract_braid_word(rank_vector(data[t]), rank_vector(data[t + 1]))
        total_raw_crossings += len(w)

    # 2. Raw CSV
    csv_bytes = _format_raw_csv_bytes(data)
    raw_csv_size = len(csv_bytes)

    # 3. gzip of CSV
    gzip_bytes = gzip.compress(csv_bytes, compresslevel=6)
    gzip_size = len(gzip_bytes)

    # 4. Simple hysteresis baseline
    h_crossings, h_bytes = _run_hysteresis_baseline(data, threshold=0.05)
    h_size = len(h_bytes)

    # 5. TangleCode Rank-Only
    tngl_bytes = encode(data, mode=MODE_RANK_ONLY)
    tngl_size = len(tngl_bytes)
    _, _, _, reduced_crossings, _ = struct.unpack_from(HEADER_STRUCT, tngl_bytes, 0)

    methods = {
        "Raw CSV": {
            "compressed_size_bytes": raw_csv_size,
            "compression_ratio": 1.0,
            "crossings": f"{total_raw_crossings} -> {total_raw_crossings}",
            "raw_crossings": total_raw_crossings,
            "reduced_crossings": total_raw_crossings,
        },
        "gzip (CSV)": {
            "compressed_size_bytes": gzip_size,
            "compression_ratio": float(raw_csv_size / gzip_size),
            "crossings": f"{total_raw_crossings} -> {total_raw_crossings}",
            "raw_crossings": total_raw_crossings,
            "reduced_crossings": total_raw_crossings,
        },
        "Hysteresis Baseline": {
            "compressed_size_bytes": h_size,
            "compression_ratio": float(raw_csv_size / h_size),
            "crossings": f"{total_raw_crossings} -> {h_crossings}",
            "raw_crossings": total_raw_crossings,
            "reduced_crossings": h_crossings,
        },
        "TangleCode Rank-Only": {
            "compressed_size_bytes": tngl_size,
            "compression_ratio": float(raw_csv_size / tngl_size),
            "crossings": f"{total_raw_crossings} -> {reduced_crossings}",
            "raw_crossings": total_raw_crossings,
            "reduced_crossings": reduced_crossings,
        },
    }

    return {
        "dataset_name": dataset_name,
        "n_samples": n_samples,
        "n_channels": n_channels,
        "raw_csv_size_bytes": raw_csv_size,
        "methods": methods,
    }


def print_table(dataset_results: Dict[str, Any]):
    """Print a small, clean benchmark table."""
    name = dataset_results["dataset_name"]
    print("\n" + "=" * 70)
    print(f"  BENCHMARK: {name}")
    print("=" * 70)
    print(f"{'Method':<22} | {'Size (B)':>10} | {'Ratio':>8} | {'Raw -> Reduced Crossings':>24}")
    print("-" * 70)
    for method, metrics in dataset_results["methods"].items():
        size_str = f"{metrics['compressed_size_bytes']:,}"
        ratio_str = f"{metrics['compression_ratio']:.2f}x"
        cross_str = metrics["crossings"]
        print(f"{method:<22} | {size_str:>10} | {ratio_str:>8} | {cross_str:>24}")
    print("=" * 70)


def run_benchmarks(
    real_csv_path: Optional[str] = "data/real.csv",
    output_json_path: str = "results.json",
) -> Dict[str, Any]:
    """Run benchmarks on synthetic data and optional real CSV dataset."""
    all_results: Dict[str, Any] = {}

    print("Benchmarking synthetic dataset (8 channels, 5000 samples)...")
    synthetic_data = generate_synthetic(n_channels=8, n_samples=5000, noise_std=0.05, seed=0)
    synth_res = benchmark_dataset(synthetic_data, "Synthetic IMU (8 ch, 5000 samples)")
    print_table(synth_res)
    all_results["synthetic"] = synth_res

    real_path = Path(real_csv_path) if real_csv_path else None
    if real_path and real_path.exists():
        print(f"\nBenchmarking real dataset: {real_path}...")
        try:
            real_data = load_csv(real_path)
            if real_data.shape[1] > 8:
                real_data = real_data[:, :8]
            real_res = benchmark_dataset(real_data, f"Real CSV: {real_path.name}")
            print_table(real_res)
            all_results["real"] = real_res
        except Exception as e:
            print(f"Skipping {real_path}: {e}")
            all_results["real"] = None
    else:
        all_results["real"] = None

    out_file = Path(output_json_path)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved benchmark results to {out_file.resolve()}")

    return all_results


if __name__ == "__main__":
    run_benchmarks()
