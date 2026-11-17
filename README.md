# TangleCode

**TangleCode** shrinks multi-sensor telemetry logs by up to **350×** and provides early warnings when a sensor begins drifting, sticking, or behaving erratically.

🔗 **Live Web Application**: [https://tanglecode-zxmupxjpc47e7isiw2h38s.streamlit.app/](https://tanglecode-zxmupxjpc47e7isiw2h38s.streamlit.app/)

---

## Table of Contents

- [Why TangleCode?](#why-tanglecode)
- [How It Works: Topological Braid Reduction](#how-it-works-topological-braid-reduction)
- [Core Features](#core-features)
- [Interactive Web App Walkthrough](#interactive-web-app-walkthrough)
- [The Swap Log CSV Specification](#the-swap-log-csv-specification)
- [Benchmark Results](#benchmark-results)
- [Mathematical Invariants](#mathematical-invariants)
- [Repository Architecture](#repository-architecture)
- [Getting Started](#getting-started)
- [Testing & Verification](#testing--verification)
- [Limitations & Operational Scope](#limitations--operational-scope)

---

## Why TangleCode?

In industrial systems, vehicle telemetry, and sensor arrays (temperature grids, vibration accelerometers, strain gauges), sensors frequently monitor correlated physical quantities. Traditional logging stores continuous high-frequency numerical amplitudes:
- **Storage Waste**: Most data consists of Gaussian noise and minor micro-fluctuations.
- **Threshold Brittleness**: Fixed amplitude alarm thresholds require manual re-calibration for drift, ambient temperature changes, and environmental cycles.
- **False Alarms**: Individual spike filtering either dampens detection speed or causes spurious notifications.

**TangleCode takes a topological approach**: instead of storing continuous amplitudes, it tracks the **relative ordering (permutation)** of sensors over time. High-frequency noise that causes adjacent sensors to jitter back and forth is topologically cancelled out, leaving a tiny record of genuine physical rank transitions and sensor faults.

---

## How It Works: Topological Braid Reduction

```
Sensors Over Time            Raw Crossings               Topological Reduction           Surviving Fault
─────────────────            ─────────────               ─────────────────────           ───────────────
S1 ───╲╱───╲╱───             (1, 2) at t=10              (1, 2) followed by              Sensor 4 permanently
S2 ───╱╲───╱╲───             (2, 1) at t=12    ───────▶  (2, 1) cancels out!   ───────▶  drifts to last place.
S3 ─────────────                                         (Noise eliminated: ~98%)        Rank: 4th ──▶ 6th
S4 ───────╲─────                                                                         Severity: Moderate
S5 ────────╲────
S6 ─────────╲───
```

1. **Permutation Frames**: At each time sample $t$, sensor values are sorted ascending to determine their permutation order.
2. **Braid Word Generation**: Adjacent order swaps between consecutive frames are decomposed into canonical braid generators $\sigma_i$ (representing sensor $x$ moving up past sensor $y$ at position $i$).
3. **Stack-Based Annihilation**: Applying the Artin braid group relation:
   $$\sigma_i \sigma_i^{-1} \to 1$$
   Any swap that quickly reverses itself is discarded as transient noise.
4. **Non-Parametric Anomaly Detection**:
   Using a sliding window ($W=200, \text{step}=100$), robust medians and Median Absolute Deviations (MAD) track:
   - **Crossing Frequency**: Rate of surviving order crossings involving channel $c$.
   - **Rank Position**: Mean relative rank position of channel $c$.
   An alert is triggered when an anomaly score exceeds threshold $k=4.0$ for at least 4 consecutive windows.

---

## Core Features

### 1. Topological Noise Cancellation
Eliminates spurious jitter without amplitude low-pass filtering. On typical sensor streams, **$98\%+$ of raw crossings are eliminated** as noise.

### 2. Ultra-Lightweight Swap-Log CSV Export
Exports only persistent structural swaps as an open, standard CSV file:
- **100× to 350× smaller** than the original uncompressed CSV.
- **35× to 130× smaller** than standard gzip compression on the CSV.

### 3. Fault Diagnosis & Severity Classification
Detects sensor anomalies, identifies the faulty channel (e.g., *Sensor 4*), classifies the behavior (*drift* vs. *erratic*), and assigns a color-coded severity:
- 🟡 **Mild**: Peak anomaly score $< 1.5 \times k$
- 🟠 **Moderate**: Peak anomaly score $< 3.0 \times k$
- 🔴 **High**: Peak anomaly score $\ge 3.0 \times k$

### 4. Professional One-Page PDF Fault Report
Generates an A4 summary document via ReportLab and Matplotlib (`Agg` backend):
- Exactly **one printable page** guaranteed through dynamic layout scaling.
- Color-coded verdict banner, fault description, and severity level.
- Three summary metric cards (Swaps found, Noise removed, Storage saved).
- Full-width time-series plot with highlighted fault zones and prominent faulty sensor line.
- Standardized engineering findings disclaimer.

### 5. Interactive "Sensor Ranking at a Moment"
- Scrub through time using a tick/seconds slider.
- Real-time table showing sensor ranks from lowest (Place 1) to highest (Place $N$).
- When the slider is inside a fault interval, the faulty sensor row is highlighted with a **red background** and labelled `(suspect)`.
- Caption tracks movement relative to 100 ticks earlier (*"Since 100 ticks ago: Sensor 4 moved from place 5 to place 6"*).

---

## Interactive Web App Walkthrough

Try the live app at [https://tanglecode-zxmupxjpc47e7isiw2h38s.streamlit.app/](https://tanglecode-zxmupxjpc47e7isiw2h38s.streamlit.app/) or run it locally:

```bash
streamlit run app.py
```

### 60-Second Demo

1. **Normal Baseline**:
   - Choose **"Healthy sensors"** and click **Analyze**.
   - See the green banner: **"All sensors normal"**.
   - Observe that $98.3\%$ of order swaps were eliminated as noise.
   - The interactive score plot stays safely below the red dashed "Alert level".
2. **Drift Detection**:
   - Return to Page 1, choose **"One sensor drifting"**, and click **Analyze**.
   - Notice the red alert: **"Fault detected: Sensor 4, around ticks 1600 to 3000"**.
   - Read the diagnostic: *"Sensor 4 moved from 4th to 6th place among the sensors and stayed there. It looks like it is drifting away from the group. Severity: Moderate"*.
   - Use the **"Sensor ranking at a moment"** slider to see Sensor 4 shift from 4th to 6th place, marked `Sensor 4 (suspect)` in red.
3. **Export Results**:
   - Click **"Download swap log (CSV)"** to save the compressed sequence.
   - Click **"Download fault report (PDF)"** to export the ready-to-share diagnostic sheet.

---

## The Swap Log CSV Specification

Surviving crossings are stored as human-readable CSV records:

| Column | Type | Description |
| :--- | :--- | :--- |
| `tick` | `int` | Integer sample index where the swap took place ($0 \le \text{tick} < T$). |
| `time_s` | `float` | *(Optional)* Elapsed time in seconds to 1 decimal place (included if sampling rate is provided). |
| `sensor_moved_up` | `int` | 1-based index of the sensor channel that moved up in rank. |
| `sensor_moved_down` | `int` | 1-based index of the sensor channel that moved down in rank. |
| `position` | `int` | 1-based topological position where the swap occurred ($1 \le \text{position} < n$). |

*Note: If no crossings survive noise removal, the file contains the header row only.*

---

## Benchmark Results

All figures below are measured directly across $T = 3,000$ samples and $n = 6$ sensor channels from [`results.json`](results.json):

### Clean Dataset (`test_clean.csv`)
- **Raw CSV**: 135,484 bytes | Ratio: 1.00× | Raw crossings: 2,417
- **gzip (CSV)**: 49,467 bytes | Ratio: 2.74× vs raw | 2,417 crossings
- **Hysteresis Baseline** ($h = 2\sigma$): 1,244 bytes | Ratio: 108.91× vs raw | 111 crossings
- **TangleCode Swap Log CSV**: **560 bytes** | **Ratio: 241.94× vs raw** (88.33× vs gzip) | **47 crossings**

### Drift Fault Dataset (`test_fault.csv`)
- **Raw CSV**: 134,966 bytes | Ratio: 1.00× | Raw crossings: 2,027
- **gzip (CSV)**: 49,989 bytes | Ratio: 2.70× vs raw | 2,027 crossings
- **Hysteresis Baseline** ($h = 2\sigma$): 1,156 bytes | Ratio: 116.75× vs raw | 103 crossings
- **TangleCode Swap Log CSV**: **384 bytes** | **Ratio: 351.47× vs raw** (130.18× vs gzip) | **31 crossings**

### Stuck Sensor Dataset (`test_stuck.csv`)
- **Raw CSV**: 134,956 bytes | Ratio: 1.00× | Raw crossings: 2,182
- **gzip (CSV)**: 47,431 bytes | Ratio: 2.85× vs raw | 2,182 crossings
- **Hysteresis Baseline** ($h = 2\sigma$): 1,288 bytes | Ratio: 104.78× vs raw | 115 crossings
- **TangleCode Swap Log CSV**: **439 bytes** | **Ratio: 307.42× vs raw** (108.04× vs gzip) | **36 crossings**

---

## Mathematical Invariants

TangleCode is verified against strict formal invariants:

1. **Crossing Conservation**:
   $$2 \times \text{cancelled\_pairs} + \text{len}(\text{reduced}) \equiv \text{len}(\text{raw})$$
2. **Exact Replay Guarantee**:
   Replaying the initial permutation $\pi_0$ through all raw adjacent swaps deterministically yields the exact permutation at tick $T-1$:
   $$\pi_0 \circ \prod_{k} \sigma_k \equiv \text{order\_from\_frame}(\text{data}[T-1])$$
3. **Deterministic Braid Ordering**:
   Ties in sensor readings are broken deterministically by sensor channel index, preventing ambiguous generator assignments.

---

## Repository Architecture

```
tanglecode/
├── app.py                   # Two-page interactive Streamlit web application
├── requirements.txt         # Core dependencies (numpy, pandas, streamlit, plotly, reportlab, matplotlib, pypdf)
├── results.json             # Measured benchmark metrics across all methods
├── verify.py                # Ground-truth mathematical verification suite
├── tanglecode/
│   ├── __init__.py          # Public API exports
│   ├── anomaly.py           # Sliding-window MAD anomaly detection & classification
│   ├── benchmark.py         # Automated comparison: Raw CSV vs gzip vs Hysteresis vs TangleCode
│   ├── braid.py             # Permutations, crossings, word extraction, stack reduction
│   ├── data.py              # Synthetic signal generation & robust CSV loading
│   ├── export.py            # Swap-log CSV formatting
│   ├── pipeline.py          # Summary statistics & crossing analytics
│   ├── ranking.py           # Moment-in-time ranking table & 100-tick delta tracking
│   └── report.py            # ReportLab & Matplotlib single-page A4 PDF report generator
└── tests/
    ├── test_anomaly.py      # Tests for detection thresholds, persistence, and classification
    ├── test_benchmark.py    # Tests for hysteresis rules and benchmark outputs
    ├── test_braid.py        # Tests for braid relations, cancellation, and order invariants
    ├── test_export.py       # Tests for swap log CSV column formats and ordering
    ├── test_ranking.py      # Tests for sensor ranking tables, places 1..N, and suspect tagging
    └── test_report.py       # Tests for valid %PDF output and strict 1-page constraints
```
```

## Limitations & Operational Scope

- **Preserves Ordering, Not Amplitudes**: TangleCode logs sensor relative rankings. If absolute sensor values must be reconstructed, store periodic baseline snapshots.
- **Normal Baseline Assumption**: The initial fraction of the time series (the first 8 windows or $20\%$ of data) is assumed to represent normal operation to compute MAD baselines.
- **Minimum Sensor Count**: Requires at least $n \ge 2$ correlated sensor channels and $T \ge 300$ samples.
- **Batch Processing**: Anomaly scoring is computed over sliding windows ($W=200, \text{step}=100$) rather than single-tick microsecond streaming.

---

## License

This project is open-source under the [MIT License](LICENSE).
