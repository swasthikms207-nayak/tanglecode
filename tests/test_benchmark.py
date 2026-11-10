"""Tests for tanglecode.benchmark module."""

import json
import tempfile
from pathlib import Path

import numpy as np

from tanglecode.benchmark import benchmark_dataset, run_benchmarks
from tanglecode.data import generate_synthetic


def test_benchmark_dataset():
    """Verify benchmark_dataset computes the 3 required metrics across all 4 methods."""
    data = generate_synthetic(n_channels=4, n_samples=100, seed=42)
    results = benchmark_dataset(data, "Test Dataset")

    assert results["n_samples"] == 100
    assert results["n_channels"] == 4

    expected_methods = [
        "Raw CSV",
        "gzip (CSV)",
        "Hysteresis Baseline",
        "TangleCode Rank-Only",
    ]
    for method in expected_methods:
        assert method in results["methods"], f"Method {method} missing from benchmark results"
        m = results["methods"][method]
        assert "compressed_size_bytes" in m
        assert "compression_ratio" in m
        assert "crossings" in m
        assert m["compressed_size_bytes"] > 0
        assert m["compression_ratio"] >= 1.0

    rank_metrics = results["methods"]["TangleCode Rank-Only"]
    assert rank_metrics["compression_ratio"] > 1.0


def test_run_benchmarks_graceful_missing_real():
    """Verify run_benchmarks gracefully handles non-existent real.csv."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        json_out = Path(tmp_dir) / "test_results.json"
        missing_csv = Path(tmp_dir) / "does_not_exist.csv"

        res = run_benchmarks(real_csv_path=str(missing_csv), output_json_path=str(json_out))
        assert res["synthetic"] is not None
        assert res["real"] is None
        assert json_out.exists()

        with open(json_out, "r", encoding="utf-8") as f:
            saved = json.load(f)
        assert "synthetic" in saved
        assert saved["real"] is None
