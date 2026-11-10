"""TangleCode: Multichannel Time Series to Reduced Braid Words.

Demonstrates rank extraction, signed Artin generator conversion,
and topological Reidemeister reductions on multichannel signals.
"""

import json
from pathlib import Path

from tanglecode.braid import (
    extract_braid_word,
    rank_vector,
    reduce_word,
    reduce_word_full,
)


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
    print("=" * 60)
    print("  TANGLECODE: Multichannel Braid Word Topological Reduction")
    print("=" * 60)

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
            [2.5, 2.0, 3.0, 4.0],  # A overtakes B (jitter overtake)
            [1.0, 2.0, 3.0, 4.0],  # A returns below B (jitter loop)
            [1.0, 2.0, 4.5, 4.0],  # C overtakes D (persistent crossing)
            [3.2, 2.0, 4.5, 4.0],  # A overtakes B permanently
        ]

    results = run_pipeline(frames, channels)

    print(f"\nChannels: {results['channel_names']}")
    print(f"Number of frames: {len(frames)}")

    print("\nRank vectors per frame:")
    for idx, rv in enumerate(results["rank_vectors"]):
        print(f"  Frame {idx:02d}: {rv} (values: {frames[idx]})")

    print("\nGenerators extracted per transition:")
    for t, step_w in enumerate(results["step_words"]):
        print(f"  Frame {t} -> {t+1}: {step_w}")

    print("\n--- Word Reduction Results ---")
    print(f"Raw Braid Word:             {results['raw_word']} (len: {len(results['raw_word'])})")
    print(f"Single-pass Reduced Word:   {results['single_pass_word']} (len: {len(results['single_pass_word'])})")
    print(f"  Cancelled pairs (Reidemeister II): {results['single_pass_cancelled']}")
    print(f"Fully Reduced Word (Full):  {results['full_word']} (len: {len(results['full_word'])})")
    print(f"  Total cancellations:               {results['full_cancelled']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
