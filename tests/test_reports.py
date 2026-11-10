"""Tests for tanglecode.reports module."""

import json
from tanglecode.reports import anomaly_report_csv, summary_json


def test_anomaly_report_csv_empty_when_no_anomalies():
    """Verify that when no anomalies are flagged, CSV contains only the header."""
    results = {
        "flagged_diagnoses": [],
    }
    csv_text = anomaly_report_csv(results)
    lines = [line.strip() for line in csv_text.strip().split("\n") if line.strip()]
    assert len(lines) == 1
    assert lines[0] == "window_start_tick,window_end_tick,score,threshold,most_likely_channel,top_crossing_pair"


def test_anomaly_report_csv_with_flagged_windows():
    """Verify that flagged windows are properly formatted in CSV."""
    results = {
        "flagged_diagnoses": [
            {
                "window_start_tick": 1500,
                "window_end_tick": 1700,
                "score": 3.8421,
                "threshold": 2.5000,
                "most_likely_channel": 4,
                "top_crossing_pair": (3, 4),
            },
            {
                "window_start_tick": 1510,
                "window_end_tick": 1710,
                "score": 4.1200,
                "threshold": 2.5000,
                "most_likely_channel": 4,
                "top_crossing_pair": (4, 5),
            },
        ],
    }
    csv_text = anomaly_report_csv(results)
    lines = [line.strip() for line in csv_text.strip().split("\n") if line.strip()]
    assert len(lines) == 3
    assert "1500,1700,3.8421,2.5000,Channel 5,Ch 4 - Ch 5" in lines[1]
    assert "1510,1710,4.1200,2.5000,Channel 5,Ch 5 - Ch 6" in lines[2]


def test_summary_json():
    """Verify that summary_json produces valid JSON with all required fields."""
    results = {
        "dataset_name": "Test IMU",
        "n_channels": 6,
        "n_samples": 3000,
        "raw_crossings": 2100,
        "reduced_crossings": 400,
        "raw_csv_size": 150000,
        "gzip_bytes": 45000,
        "tngl_size": 2200,
        "flagged_intervals": [(1500, 1720)],
        "faulty_channel": 4,
    }
    json_text = summary_json(results)
    parsed = json.loads(json_text)

    assert parsed["dataset_name"] == "Test IMU"
    assert parsed["channel_count"] == 6
    assert parsed["sample_count"] == 3000
    assert parsed["raw_crossings"] == 2100
    assert parsed["reduced_crossings"] == 400
    assert parsed["percent_eliminated"] > 80.0
    assert parsed["csv_bytes"] == 150000
    assert parsed["gzip_bytes"] == 45000
    assert parsed["tngl_bytes"] == 2200
    assert parsed["anomaly_tick_ranges"] == [[1500, 1720]]
    assert parsed["faulty_channel"] == "Channel 5"


def test_summary_json_no_fault():
    """Verify that summary_json handles null faulty channel correctly."""
    results = {
        "dataset_name": "Clean Drone IMU",
        "n_channels": 4,
        "n_samples": 1000,
        "raw_crossings": 500,
        "reduced_crossings": 200,
        "raw_csv_size": 50000,
        "gzip_bytes": 15000,
        "tngl_size": 1100,
        "flagged_intervals": [],
        "faulty_channel": None,
    }
    json_text = summary_json(results)
    parsed = json.loads(json_text)
    assert parsed["faulty_channel"] is None
    assert parsed["anomaly_tick_ranges"] == []
