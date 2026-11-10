"""Tests for tanglecode.anomaly streaming braid anomaly detection module."""

import numpy as np
import pytest

from tanglecode.anomaly import (
    AnomalyRecord,
    BraidAnomalyDetector,
    compute_window_features,
    detect_anomalies,
)
from tanglecode.data import generate_synthetic, inject_fault


def test_anomaly_record_unpacking():
    """Verify AnomalyRecord unpacks cleanly as (window_start, score, flagged)."""
    rec = AnomalyRecord(100, 3.45, True, reduced_length=12, raw_length=24, ratio=0.5, most_frequent_generator=2)

    # 3-tuple unpacking
    start, score, flagged = rec
    assert start == 100
    assert score == pytest.approx(3.45)
    assert flagged is True

    # Indexing
    assert rec[0] == 100
    assert rec[1] == pytest.approx(3.45)
    assert rec[2] is True

    # Property access
    assert rec.window_start == 100
    assert rec.score == pytest.approx(3.45)
    assert rec.flagged is True
    assert rec.reduced_length == 12
    assert rec.raw_length == 24
    assert rec.ratio == pytest.approx(0.5)
    assert rec.most_frequent_generator == 2


def test_compute_window_features():
    """Verify window feature computation for raw, reduced, ratio, and dominant generator."""
    # Constant data produces 0 generators
    flat_data = np.ones((50, 4), dtype=np.float32)
    feat = compute_window_features(flat_data, window_start=0)
    assert feat.window_start == 0
    assert feat.raw_length == 0
    assert feat.reduced_length == 0
    assert feat.ratio == 0.0
    assert feat.most_frequent_generator is None

    # Oscillating synthetic data produces valid braid features
    data = generate_synthetic(n_channels=4, n_samples=100, noise_std=0.03, seed=42)
    feat_active = compute_window_features(data, window_start=10)
    assert feat_active.window_start == 10
    assert feat_active.raw_length > 0
    assert feat_active.reduced_length > 0
    assert 0.0 <= feat_active.ratio <= 1.0


def test_drift_anomaly_detection(capsys):
    """Verify drift fault is flagged, clean windows are mostly not, and report metrics."""
    data_clean = generate_synthetic(n_channels=4, n_samples=3000, noise_std=0.02, seed=42)
    fault_start = 1500

    # Inject drift fault at tick 1500 on Channel 0
    faulty_data = inject_fault(data_clean, start=fault_start, kind="drift", channel=0, drift_slope=0.015)

    detector = BraidAnomalyDetector(
        window_size=200,
        n_baseline=40,
        k=2.2,
        step=10,
        two_sided=True,
    )
    eval_metrics = detector.evaluate(faulty_data, fault_start=fault_start)

    fpr = eval_metrics["false_positive_rate"]
    detection_delay = eval_metrics["detection_delay_ticks"]
    first_flagged = eval_metrics["first_flagged_window_start"]

    print("\n--- Drift Anomaly Detection Report ---")
    print(f"Clean windows evaluated:   {eval_metrics['clean_windows_count']}")
    print(f"False positives count:     {eval_metrics['false_positives_count']}")
    print(f"False Positive Rate (FPR): {fpr:.2%}")
    print(f"Fault windows flagged:     {eval_metrics['fault_windows_flagged']} / {eval_metrics['fault_windows_total']}")
    print(f"First flagged window:      {first_flagged}")
    print(f"Detection delay:           {detection_delay} ticks")

    # Assertions: clean windows mostly not flagged (< 5% false positive rate)
    assert fpr < 0.05, f"FPR too high: {fpr:.2%}"

    # Fault is flagged
    assert eval_metrics["fault_windows_flagged"] > 0, "No fault window was flagged"
    assert detection_delay is not None
    assert detection_delay >= 0


def test_stuck_anomaly_detection(capsys):
    """Verify stuck sensor fault is flagged, clean windows are mostly not, and report metrics."""
    data_clean = generate_synthetic(n_channels=4, n_samples=3000, noise_std=0.02, seed=42)
    fault_start = 1500

    # Inject stuck fault at tick 1500 on Channel 1
    faulty_data = inject_fault(data_clean, start=fault_start, kind="stuck", channel=1)

    detector = BraidAnomalyDetector(
        window_size=200,
        n_baseline=40,
        k=2.2,
        step=10,
        two_sided=True,
    )
    eval_metrics = detector.evaluate(faulty_data, fault_start=fault_start)

    fpr = eval_metrics["false_positive_rate"]
    detection_delay = eval_metrics["detection_delay_ticks"]
    first_flagged = eval_metrics["first_flagged_window_start"]

    print("\n--- Stuck Anomaly Detection Report ---")
    print(f"Clean windows evaluated:   {eval_metrics['clean_windows_count']}")
    print(f"False positives count:     {eval_metrics['false_positives_count']}")
    print(f"False Positive Rate (FPR): {fpr:.2%}")
    print(f"Fault windows flagged:     {eval_metrics['fault_windows_flagged']} / {eval_metrics['fault_windows_total']}")
    print(f"First flagged window:      {first_flagged}")
    print(f"Detection delay:           {detection_delay} ticks")

    # Assertions: clean windows mostly not flagged (< 5% false positive rate)
    assert fpr < 0.05, f"FPR too high: {fpr:.2%}"

    # Fault is flagged
    assert eval_metrics["fault_windows_flagged"] > 0, "No fault window was flagged"
    assert detection_delay is not None
    assert detection_delay >= 0


def test_detect_anomalies_convenience_function():
    """Verify detect_anomalies helper returns list of AnomalyRecord tuples."""
    data = generate_synthetic(n_channels=4, n_samples=600, noise_std=0.03, seed=1)
    results = detect_anomalies(data, window_size=200, n_baseline=20, k=3.0, step=20)

    assert len(results) > 0
    for rec in results:
        assert isinstance(rec, tuple)
        start, score, flagged = rec
        assert isinstance(start, int)
        assert isinstance(score, float)
        assert isinstance(flagged, bool)
