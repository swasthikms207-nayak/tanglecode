from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from tanglecode.benchmark import benchmark_dataset, run_all_benchmarks, run_hysteresis
from tanglecode.data import generate_signals


def test_hysteresis_baseline_logic():
    """Verify hysteresis swap acceptance thresholding and encoding."""
    data = generate_signals("clean", n=6, T=500, seed=7)
    h_count, h_bytes, noise_std, h = run_hysteresis(data, raw_crossings_count=300)

    assert noise_std > 0.0
    assert h == 2.0 * noise_std
    assert h_count >= 0
    assert len(h_bytes) > 0
    # Magic bytes check
    assert h_bytes[:4] == b"TNGL"


def test_benchmark_dataset():
    """Verify benchmark metrics calculation across methods."""
    data = generate_signals("clean", n=6, T=350, seed=7)
    csv_bytes = b"dummy,csv,bytes\n1,2,3\n" * 100
    res = benchmark_dataset(data, csv_bytes, "TestDataset")

    assert res["dataset_name"] == "TestDataset"
    methods = res["methods"]
    assert "Raw CSV" in methods
    assert "gzip (CSV)" in methods
    assert "Hysteresis baseline" in methods
    assert "TangleCode (with reduction)" in methods

    for m_name, m_dict in methods.items():
        assert m_dict["bytes"] > 0
        assert m_dict["ratio_vs_raw_csv"] > 0.0
        assert m_dict["ratio_vs_gzip"] > 0.0
        assert m_dict["crossings_before"] >= 0
        assert m_dict["crossings_after"] >= 0


def test_run_all_benchmarks(tmp_path):
    """Verify running benchmarks outputs valid results.json."""
    json_path = tmp_path / "results_test.json"
    results = run_all_benchmarks(output_json_path=str(json_path))

    assert json_path.exists()
    assert "clean" in results
    assert "drift" in results
    assert "stuck" in results

    loaded = json.loads(json_path.read_text(encoding="utf-8"))
    assert set(loaded.keys()) == {"clean", "drift", "stuck"}
