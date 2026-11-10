"""Benchmark comparing TangleCode against standard compression & hysteresis baselines."""

import gzip
import io
import json
import struct
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from tanglecode.braid import extract_braid_word, rank_vector
from tanglecode.codec import (
    HEADER_STRUCT,
    MAGIC,
    MODE_HYBRID,
    MODE_RANK_ONLY,
    _pack_generators,
    _rle_encode,
    decode,
    encode,
)
from tanglecode.data import generate_synthetic, load_csv


def compute_kendall_tau(r1: np.ndarray, r2: np.ndarray) -> float:
    """Compute mean Kendall's tau correlation between reconstructed and ground-truth ranks.

    Values range from -1.0 (complete disagreement) to +1.0 (identical ranking).
    """
    n_samples, n_channels = r1.shape
    if n_channels <= 1:
        return 1.0

    total_tau = 0.0
    total_pairs = n_channels * (n_channels - 1) // 2

    for t in range(n_samples):
        # Convert values to ranks
        ranks1 = np.argsort(np.argsort(r1[t]))
        ranks2 = np.argsort(np.argsort(r2[t]))

        concordant = 0
        discordant = 0
        for i in range(n_channels):
            for j in range(i + 1, n_channels):
                diff1 = ranks1[i] - ranks1[j]
                diff2 = ranks2[i] - ranks2[j]
                prod = diff1 * diff2
                if prod > 0:
                    concordant += 1
                elif prod < 0:
                    discordant += 1

        tau = (concordant - discordant) / total_pairs if total_pairs > 0 else 1.0
        total_tau += tau

    return float(total_tau / n_samples)


def _compress_zstd_or_zlib(data_bytes: bytes) -> Tuple[bytes, str]:
    """Compress bytes using zstandard if installed, otherwise fallback to zlib."""
    try:
        import zstandard as zstd  # type: ignore

        cctx = zstd.ZstdCompressor(level=3)
        return cctx.compress(data_bytes), "zstandard"
    except ImportError:
        import zlib

        return zlib.compress(data_bytes, level=6), "zlib"


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
) -> Tuple[int, bytes, np.ndarray, float]:
    """Simple hysteresis baseline:

    Only register a rank swap if the amplitude difference exceeds a threshold,
    then apply NO braid reduction.
    """
    n_samples, n_channels = data.shape
    start_time = time.perf_counter()

    current_order = rank_vector(data[0])
    recon_ranks = np.zeros((n_samples, n_channels), dtype=np.float32)
    for c in range(n_channels):
        recon_ranks[0, c] = float(current_order.index(c))

    hysteresis_generators: list[int] = []
    per_tick_counts: list[int] = []

    for t in range(n_samples - 1):
        vals = data[t + 1]
        swaps_this_tick = 0

        # Bubble sort with amplitude hysteresis threshold
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
        for c in range(n_channels):
            recon_ranks[t + 1, c] = float(current_order.index(c))

    encode_time = time.perf_counter() - start_time

    # Binary packaging for fair size comparison
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

    return len(hysteresis_generators), packaged_bytes, recon_ranks, encode_time


def benchmark_dataset(data: np.ndarray, dataset_name: str) -> Dict[str, Any]:
    """Run benchmark across all methods on a given dataset."""
    n_samples, n_channels = data.shape

    # 1. Measure raw crossings
    total_raw_crossings = 0
    for t in range(n_samples - 1):
        w = extract_braid_word(rank_vector(data[t]), rank_vector(data[t + 1]))
        total_raw_crossings += len(w)

    # 2. Raw CSV
    csv_bytes = _format_raw_csv_bytes(data)
    raw_csv_size = len(csv_bytes)

    methods_res: Dict[str, Dict[str, Any]] = {}

    # Method 1: Raw CSV
    methods_res["Raw CSV"] = {
        "compressed_size_bytes": raw_csv_size,
        "compression_ratio": 1.0,
        "crossings_eliminated_pct": 0.0,
        "encode_time_per_frame_us": 0.0,
        "kendall_tau": 1.0,
        "rmse": 0.0,
    }

    # Method 2: gzip of the CSV
    t0 = time.perf_counter()
    gzip_bytes = gzip.compress(csv_bytes, compresslevel=6)
    t_gzip = time.perf_counter() - t0
    methods_res["gzip (CSV)"] = {
        "compressed_size_bytes": len(gzip_bytes),
        "compression_ratio": float(raw_csv_size / len(gzip_bytes)),
        "crossings_eliminated_pct": 0.0,
        "encode_time_per_frame_us": float((t_gzip / n_samples) * 1e6),
        "kendall_tau": 1.0,
        "rmse": 0.0,
    }

    # Method 3: zstandard / zlib of float32 binary
    raw_binary = data.astype(np.float32).tobytes()
    t0 = time.perf_counter()
    zstd_bytes, compressor_name = _compress_zstd_or_zlib(raw_binary)
    t_zstd = time.perf_counter() - t0
    method_key = f"{compressor_name} (float32)"
    methods_res[method_key] = {
        "compressed_size_bytes": len(zstd_bytes),
        "compression_ratio": float(raw_csv_size / len(zstd_bytes)),
        "crossings_eliminated_pct": 0.0,
        "encode_time_per_frame_us": float((t_zstd / n_samples) * 1e6),
        "kendall_tau": 1.0,
        "rmse": 0.0,
    }

    # Method 4: Simple hysteresis baseline
    h_crossings, h_bytes, h_recon_ranks, t_hyst = _run_hysteresis_baseline(data, threshold=0.05)
    h_eliminated = (
        float((total_raw_crossings - h_crossings) / total_raw_crossings * 100.0)
        if total_raw_crossings > 0
        else 0.0
    )
    h_tau = compute_kendall_tau(h_recon_ranks, data)
    methods_res["Hysteresis Baseline"] = {
        "compressed_size_bytes": len(h_bytes),
        "compression_ratio": float(raw_csv_size / len(h_bytes)),
        "crossings_eliminated_pct": h_eliminated,
        "encode_time_per_frame_us": float((t_hyst / n_samples) * 1e6),
        "kendall_tau": h_tau,
        "rmse": None,
    }

    # Method 5: TangleCode Rank-Only
    t0 = time.perf_counter()
    tngl_rank_bytes = encode(data, mode=MODE_RANK_ONLY)
    t_rank = time.perf_counter() - t0

    _, _, _, reduced_count_rank, _ = struct.unpack_from(HEADER_STRUCT, tngl_rank_bytes, 0)
    tc_rank_eliminated = (
        float((total_raw_crossings - reduced_count_rank) / total_raw_crossings * 100.0)
        if total_raw_crossings > 0
        else 0.0
    )
    recon_rank = decode(tngl_rank_bytes)
    tau_rank = compute_kendall_tau(recon_rank, data)

    methods_res["TangleCode Rank-Only"] = {
        "compressed_size_bytes": len(tngl_rank_bytes),
        "compression_ratio": float(raw_csv_size / len(tngl_rank_bytes)),
        "crossings_eliminated_pct": tc_rank_eliminated,
        "encode_time_per_frame_us": float((t_rank / n_samples) * 1e6),
        "kendall_tau": tau_rank,
        "rmse": None,
    }

    # Method 6: TangleCode Hybrid
    t0 = time.perf_counter()
    tngl_hybrid_bytes = encode(data, mode=MODE_HYBRID, step_size=0.02)
    t_hybrid = time.perf_counter() - t0

    _, _, _, reduced_count_hyb, _ = struct.unpack_from(HEADER_STRUCT, tngl_hybrid_bytes, 0)
    tc_hyb_eliminated = (
        float((total_raw_crossings - reduced_count_hyb) / total_raw_crossings * 100.0)
        if total_raw_crossings > 0
        else 0.0
    )
    recon_hybrid = decode(tngl_hybrid_bytes)
    tau_hyb = compute_kendall_tau(recon_hybrid, data)
    rmse_hyb = float(np.sqrt(np.mean((recon_hybrid - data) ** 2)))

    methods_res["TangleCode Hybrid"] = {
        "compressed_size_bytes": len(tngl_hybrid_bytes),
        "compression_ratio": float(raw_csv_size / len(tngl_hybrid_bytes)),
        "crossings_eliminated_pct": tc_hyb_eliminated,
        "encode_time_per_frame_us": float((t_hybrid / n_samples) * 1e6),
        "kendall_tau": tau_hyb,
        "rmse": rmse_hyb,
    }

    return {
        "dataset_name": dataset_name,
        "n_samples": n_samples,
        "n_channels": n_channels,
        "raw_csv_size_bytes": raw_csv_size,
        "total_raw_crossings": total_raw_crossings,
        "methods": methods_res,
    }


def print_table(dataset_results: Dict[str, Any]):
    """Print an aligned ASCII table of benchmark metrics for a dataset."""
    name = dataset_results["dataset_name"]
    n_samples = dataset_results["n_samples"]
    n_channels = dataset_results["n_channels"]
    raw_crossings = dataset_results["total_raw_crossings"]
    raw_bytes = dataset_results["raw_csv_size_bytes"]

    print("\n" + "=" * 96)
    print(
        f"  DATASET: {name} (Samples: {n_samples}, Channels: {n_channels}, Raw Crossings: {raw_crossings}, Raw CSV: {raw_bytes:,} B)"
    )
    print("=" * 96)

    header = (
        f"{'Method':<24} | {'Size (B)':>10} | {'Ratio':>8} | {'Eliminated':>11} | "
        f"{'Enc Time/Fr':>12} | {'Kendall Tau':>11} | {'RMSE':>8}"
    )
    print(header)
    print("-" * 96)

    for method, metrics in dataset_results["methods"].items():
        size_str = f"{metrics['compressed_size_bytes']:,}"
        ratio_str = f"{metrics['compression_ratio']:.2f}x"
        elim_str = f"{metrics['crossings_eliminated_pct']:.1f}%"
        time_str = f"{metrics['encode_time_per_frame_us']:.2f} us"
        tau_str = f"{metrics['kendall_tau']:.4f}"
        rmse_val = metrics["rmse"]
        rmse_str = f"{rmse_val:.4f}" if rmse_val is not None else "N/A"

        row = (
            f"{method:<24} | {size_str:>10} | {ratio_str:>8} | {elim_str:>11} | "
            f"{time_str:>12} | {tau_str:>11} | {rmse_str:>8}"
        )
        print(row)
    print("=" * 96)


def run_benchmarks(
    real_csv_path: Optional[str] = "data/real.csv",
    output_json_path: str = "results.json",
) -> Dict[str, Any]:
    """Run benchmarks on synthetic data and optional real CSV data, printing tables and saving JSON."""
    all_results: Dict[str, Any] = {}

    # 1. Synthetic dataset
    print("Generating and benchmarking synthetic dataset (8 channels, 5000 samples)...")
    synthetic_data = generate_synthetic(n_channels=8, n_samples=5000, noise_std=0.05, seed=0)
    synth_results = benchmark_dataset(synthetic_data, "Synthetic IMU (8 ch, 5000 samples)")
    print_table(synth_results)
    all_results["synthetic"] = synth_results

    # 2. Real CSV dataset (if present)
    real_path = Path(real_csv_path) if real_csv_path else None
    if real_path and real_path.exists():
        print(f"\nLoading and benchmarking real dataset from {real_path}...")
        try:
            real_data = load_csv(real_path)
            # Support up to 8 channels
            if real_data.shape[1] > 8:
                real_data = real_data[:, :8]
            real_results = benchmark_dataset(real_data, f"Real CSV: {real_path.name}")
            print_table(real_results)
            all_results["real"] = real_results
        except Exception as e:
            print(f"Warning: Failed to benchmark {real_path}: {e}")
            all_results["real"] = None
    else:
        print(f"\nReal dataset '{real_csv_path}' not found, skipping gracefully.")
        all_results["real"] = None

    # Save to results.json
    out_file = Path(output_json_path)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults successfully saved to {out_file.resolve()}")

    return all_results


if __name__ == "__main__":
    run_benchmarks()
