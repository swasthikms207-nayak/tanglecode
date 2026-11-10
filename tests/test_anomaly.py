"""Tests for tanglecode.anomaly streaming braid anomaly detection module."""

import numpy as np
import pytest

from tanglecode.anomaly import (
    AnomalyRecord,
    BraidAnomalyDetector,
    compute_window_features,
    count_surviving_crossings_per_channel,
    detect_anomalies,
    diagnose_window_fault,
    learn_channel_baseline,
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


def test_per_channel_fault_scoring_identifies_known_channel(capsys):
    """Verify that per-channel fault scoring correctly names channel 4 for both drift and stuck."""
    from collections import Counter

    known_channel = 4
    n_channels = 6
    clean_data = generate_synthetic(
        n_channels=n_channels,
        n_samples=3000,
        noise_std=0.03,
        seed=42,
    )
    fault_start = 1500
    window_size = 200
    n_baseline = 40
    step = 10

    # Learn baseline per-channel crossing counts from clean data
    ch_mean, ch_std = learn_channel_baseline(
        clean_data,
        window_size=window_size,
        n_baseline=n_baseline,
        step=step,
    )
    assert len(ch_mean) == n_channels
    assert len(ch_std) == n_channels

    for kind in ["drift", "stuck"]:
        slope = 0.015 if kind == "drift" else 0.005
        faulty_data = inject_fault(
            clean_data,
            start=fault_start,
            kind=kind,
            channel=known_channel,
            drift_slope=slope,
        )

        detector = BraidAnomalyDetector(
            window_size=window_size,
            n_baseline=n_baseline,
            k=2.0,
            step=step,
            two_sided=True,
        )
        records = detector.predict(faulty_data)
        flagged_records = [r for r in records if r.flagged and r.window_start >= fault_start]

        assert len(flagged_records) > 0, f"No anomaly windows flagged for {kind}"

        diagnoses = []
        for r in flagged_records:
            w = faulty_data[r.window_start : r.window_start + window_size]
            diag = diagnose_window_fault(
                w,
                baseline_channel_mean=ch_mean,
                baseline_channel_std=ch_std,
                window_start=r.window_start,
            )
            diagnoses.append(diag)

        # Count votes across all flagged windows
        voted_channels = [d.most_likely_channel for d in diagnoses]
        top_channel, vote_count = Counter(voted_channels).most_common(1)[0]

        # Top crossing pair in the first flagged window
        first_diag = diagnoses[0]
        assert first_diag.top_crossing_pair is not None

        print(f"\n[{kind.upper()}] Fault on Channel {known_channel}:")
        print(f"  Flagged windows: {len(flagged_records)}")
        print(f"  Named most likely faulty sensor: Channel {top_channel} ({vote_count}/{len(flagged_records)} votes)")
        print(f"  First flagged window top crossing pair: {first_diag.top_crossing_pair}")

        assert (
            top_channel == known_channel
        ), f"Expected channel {known_channel} for {kind}, got channel {top_channel}"

