"""TangleCode: Multichannel Time Series to Reduced Braid Words.

Demonstrates rank extraction, signed Artin generator conversion,
and topological Reidemeister reductions on multichannel signals.
"""

import json
from pathlib import Path

import numpy as np

from tanglecode.braid import (
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
from tanglecode.data import generate_synthetic, inject_fault


def run_pipeline(frames: list[list[float]], channel_names: list[str] | None = None) -> dict:
    """Processes a multichannel time series through the TangleCode pipeline."""
    if not frames:
        return {"raw_word": [], "reduced_single_pass": [], "reduced_full": []}

    num_channels = len(frames[0])
    if channel_names is None:
        channel_names = [f"Ch_{i}" for i in range(num_channels)]

    rank_vectors = [rank_vector(f) for f in frames]

    raw_word: list[int] = []
    step_words: list[list[int]] = []

    for t in range(len(frames) - 1):
        step_word = extract_braid_word(rank_vectors[t], rank_vectors[t + 1])
        step_words.append(step_word)
        raw_word.extend(step_word)

    single_pass_res, single_pass_cancelled = reduce_word(raw_word)
    full_res, full_cancelled = reduce_word_full(raw_word)

    return {
        "channel_names": channel_names,
        "rank_vectors": rank_vectors,
        "step_words": step_words,
        "raw_word": raw_word,
        "single_pass_word": list(single_pass_res),
        "single_pass_cancelled": single_pass_cancelled,
        "full_word": list(full_res),
        "full_cancelled": full_cancelled,
    }


def main():
    print("=" * 65)
    print("  TANGLECODE: Multichannel Braid Word Topological Reduction & Codec")
    print("=" * 65)

    data_path = Path(__file__).parent / "data" / "sample_signals.json"
    if data_path.exists():
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        frames = data.get("frames", [])
        channels = data.get("channels", None)
        print(f"Loaded sample data from {data_path}")
    else:
        print("Using built-in sample frames...")
        channels = ["A", "B", "C", "D"]
        frames = [
            [1.0, 2.0, 3.0, 4.0],
            [2.5, 2.0, 3.0, 4.0],
            [1.0, 2.0, 3.0, 4.0],
            [1.0, 2.0, 4.5, 4.0],
            [3.2, 2.0, 4.5, 4.0],
        ]

    results = run_pipeline(frames, channels)

    print(f"\nChannels: {results['channel_names']}")
    print(f"Number of frames: {len(frames)}")

    print("\n--- Word Reduction Results ---")
    print(f"Raw Braid Word:             {results['raw_word']} (len: {len(results['raw_word'])})")
    print(f"Single-pass Reduced Word:   {results['single_pass_word']} (len: {len(results['single_pass_word'])})")
    print(f"  Cancelled pairs (Reidemeister II): {results['single_pass_cancelled']}")
    print(f"Fully Reduced Word (Full):  {results['full_word']} (len: {len(results['full_word'])})")
    print(f"  Total cancellations:               {results['full_cancelled']}")

    # Demonstrate synthetic data generation, fault injection, and .tngl codec
    print("\n" + "=" * 65)
    print("  DEMO: Synthetic IMU Data, Fault Injection & .tngl Codec")
    print("=" * 65)

    sim_data = generate_synthetic(n_channels=6, n_samples=500, noise_std=0.03, seed=42)
    print(f"Generated synthetic IMU data: shape {sim_data.shape} ({sim_data.nbytes} bytes raw float32)")

    # Inject sensor drift fault at sample 250 on Channel 0
    faulty_data = inject_fault(sim_data, start=250, kind="drift", channel=0, drift_slope=0.01)
    print("Injected 'drift' fault on Channel 0 starting at sample 250.")

    # Encode in Rank-only mode
    tngl_rank = encode(faulty_data, mode=MODE_RANK_ONLY)
    recon_rank = decode(tngl_rank)
    print(f"\n[Rank-Only Mode]")
    print(f"  Compressed size:   {len(tngl_rank)} bytes")
    print(f"  Compression ratio: {sim_data.nbytes / len(tngl_rank):.1f}x")
    print(f"  Reconstructed shape: {recon_rank.shape}")

    # Encode in Hybrid mode
    tngl_hybrid = encode(faulty_data, mode=MODE_HYBRID, step_size=0.02)
    recon_hybrid = decode(tngl_hybrid)
    max_err = float(np.max(np.abs(recon_hybrid - faulty_data)))
    print(f"\n[Hybrid Mode]")
    print(f"  Compressed size:   {len(tngl_hybrid)} bytes")
    print(f"  Compression ratio: {sim_data.nbytes / len(tngl_hybrid):.1f}x")
    print(f"  Max reconstruction error: {max_err:.5f}")
    print("=" * 65)


if __name__ == "__main__":
    main()
