"""Tests for tanglecode.data module."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
from tanglecode.data import generate_synthetic, inject_fault, load_csv


def test_generate_synthetic():
    """Verify synthetic IMU data generation dimensions and reproducibility."""
    data1 = generate_synthetic(n_channels=6, n_samples=1000, seed=42)
    assert data1.shape == (1000, 6)
    assert data1.dtype == np.float32

    # Reproducibility with same seed
    data2 = generate_synthetic(n_channels=6, n_samples=1000, seed=42)
    np.testing.assert_array_equal(data1, data2)

    # Different seed yields different noise
    data3 = generate_synthetic(n_channels=6, n_samples=1000, seed=43)
    assert not np.array_equal(data1, data3)


def test_inject_fault_drift():
    """Verify drift fault injection adds linear ramp to specified channel."""
    base = generate_synthetic(n_channels=4, n_samples=200, seed=1)
    start_idx = 100
    channel = 1

    faulty = inject_fault(base, start=start_idx, kind="drift", channel=channel, drift_slope=0.01)

    # Pre-fault samples must be identical
    np.testing.assert_array_equal(faulty[:start_idx], base[:start_idx])

    # Unaffected channels must be identical throughout
    for c in [0, 2, 3]:
        np.testing.assert_array_equal(faulty[:, c], base[:, c])

    # Injected channel after start must have drift added
    expected_drift = np.arange(100, dtype=np.float32) * 0.01
    np.testing.assert_allclose(faulty[start_idx:, channel] - base[start_idx:, channel], expected_drift, atol=1e-5)


def test_inject_fault_stuck():
    """Verify stuck fault freezes channel at start sample value."""
    base = generate_synthetic(n_channels=4, n_samples=200, seed=1)
    start_idx = 80
    channel = 2

    faulty = inject_fault(base, start=start_idx, kind="stuck", channel=channel)

    # Pre-fault samples must be identical
    np.testing.assert_array_equal(faulty[:start_idx], base[:start_idx])

    # Post-fault samples for channel 2 must all equal base[start_idx, channel]
    stuck_val = base[start_idx, channel]
    assert np.all(faulty[start_idx:, channel] == stuck_val)


def test_inject_fault_validation():
    """Verify error checking for invalid kind or out of bound indices."""
    base = np.zeros((100, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="Unknown fault kind"):
        inject_fault(base, start=10, kind="invalid_kind")

    with pytest.raises(ValueError, match="start index"):
        inject_fault(base, start=150, kind="drift")

    with pytest.raises(ValueError, match="channel"):
        inject_fault(base, start=10, kind="drift", channel=5)


def test_load_csv():
    """Verify loading real/mock CSV data with headers and text columns."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        csv_file = Path(tmp_dir) / "sensor_data.csv"
        csv_content = """timestamp,ax,ay,az,label
2026-01-01T00:00:00,1.25,2.50,3.75,walk
2026-01-01T00:00:01,1.30,2.45,3.80,walk
2026-01-01T00:00:02,1.20,2.55,3.70,walk
"""
        csv_file.write_text(csv_content, encoding="utf-8")

        loaded = load_csv(csv_file)
        assert loaded.shape == (3, 3)
        np.testing.assert_allclose(
            loaded,
            np.array([[1.25, 2.50, 3.75], [1.30, 2.45, 3.80], [1.20, 2.55, 3.70]], dtype=np.float32),
        )
