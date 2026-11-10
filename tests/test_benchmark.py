"""Tests for tanglecode.benchmark module."""

import json
import tempfile
from pathlib import Path

import numpy as np

from tanglecode.benchmark import benchmark_dataset, compute_kendall_tau, run_benchmarks
from tanglecode.data import generate_synthetic


def test_compute_kendall_tau():
    """Verify Kendall tau calculation on perfect agreement, inversion, and ties."""
    r1 = np.array([[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]], dtype=np.float32)
    r2 = np.array([[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]], dtype=np.float32)
    assert compute_kendall_tau(r1, r2) == 1.0

    # Inverted ranking
    r_inv = np.array([[2.0, 1.0, 0.0], [2.0, 1.0, 0.0]], dtype=np.float32)
    assert compute_kendall_tau(r1, r_inv) == -1.0


def test_benchmark_dataset():
    """Verify benchmark_dataset computes all metrics across all methods."""
    data = generate_synthetic(n_channels=4, n_samples=100, seed=42)
    results = benchmark_dataset(data, "Test Dataset")

    assert results["n_samples"] == 100
    assert results["n_channels"] == 4
    assert results["total_raw_crossings"] >= 0

    expected_methods = [
        "Raw CSV",
        "gzip (CSV)",
        "Hysteresis Baseline",
        "TangleCode Rank-Only",
        "TangleCode Hybrid",
    ]
    for method in expected_methods:
        found = any(method in k for k in results["methods"].keys())
        assert found, f"Method {method} missing from benchmark results"

    rank_metrics = results["methods"]["TangleCode Rank-Only"]
    assert rank_metrics["compression_ratio"] > 1.0
    assert 0.0 <= rank_metrics["crossings_eliminated_pct"] <= 100.0
    assert 0.0 <= rank_metrics["kendall_tau"] <= 1.0

    hybrid_metrics = results["methods"]["TangleCode Hybrid"]
    assert hybrid_metrics["rmse"] is not None
    assert hybrid_metrics["rmse"] < 0.1


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
