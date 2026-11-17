# TangleCode

TangleCode shrinks multi-sensor logs and warns you when one sensor starts misbehaving.

When several sensors measure related signals (such as accelerometers or temperature probes), high-frequency noise causes their relative order to jitter back and forth. TangleCode translates these order swaps into a mathematical sequence. Any swap that quickly reverses itself cancels out as transient noise. Genuine physical changes and sensor faults survive.

---

## What It Does in Plain Words

1. **Noise Removal**: Eliminates spurious order swaps without needing manual amplitude filters.
2. **Order-Based Export**: Exports surviving swaps into a lightweight swap-log CSV format, drastically reducing storage size.
3. **Fault Warning**: Detects when a sensor drifts or gets stuck, identifies which sensor is misbehaving (e.g., Sensor 4), and highlights when the fault occurred.
4. **One-Page PDF Fault Report**: Generates an A4 summary report with verdict, metrics, downsampled multichannel readings, and highlighted fault zones.
5. **Sensor Ranking at a Moment**: Interactively inspects relative sensor ranks and values across time, highlighting suspect sensors and tracking order changes compared with 100 ticks earlier.

---

## Setup

Requires Python 3.11+.

Clone the repository and install dependencies:

```bash
git clone https://github.com/swasthikms207-nayak/tanglecode.git
cd tanglecode
pip install -r requirements.txt
```

---

## How to Run

### Run the Interactive App
Launch the two-page Streamlit application:
```bash
streamlit run app.py
```

### Run the Verification Pipeline
Generate synthetic datasets and run the full end-to-end detection and verification suite:
```bash
python verify.py
```

### Run the Compression Benchmark
Measure performance against raw CSV, gzip, and a hysteresis baseline:
```bash
python -m tanglecode.benchmark
```

### Run the Unit Tests
Execute the automated pytest suite:
```bash
python -m pytest -v
```

---

## The Swap Log CSV Format

Surviving sensor order swaps are exported as a clean, human-readable CSV swap log:

| Column | Description |
| :--- | :--- |
| `tick` | Integer time step when the swap occurred. |
| `time_s` | *(Optional)* Time in seconds formatted to one decimal place, included only when a sampling rate is specified. |
| `sensor_moved_up` | 1-based sensor index of the sensor that moved up in rank. |
| `sensor_moved_down` | 1-based sensor index of the sensor that moved down in rank. |
| `position` | 1-based position where the swap took place ($1 \le \text{position} < n$). |

---

## Limitations

- **Preserves order, not exact amplitudes**: The swap log keeps the order of the sensors, not their exact numerical values.
- **Normal baseline assumption**: The initial fraction of the time series (the first 8 or more windows) is assumed to be normal operation to learn baseline behavior.
- **Batch analysis**: Processing occurs over time windows rather than single-sample real-time streaming.
- **Baseline comparison**: Evaluated against a hysteresis threshold baseline ($h = 2\sigma$) that filters swaps without topological reduction.

---

## Benchmark Results

The following measurements are copied directly from `results.json` (3,000 time ticks across 6 sensor channels):

### Clean Data (`test_clean.csv`)
- **Raw CSV Size**: 135,484 bytes
- **gzip Size**: 49,467 bytes
- **Methods**:
  - **Raw CSV**: 135,484 bytes | Ratio vs CSV: 1.00x | Ratio vs gzip: 0.37x | Swaps: 2,417 -> 2,417
  - **gzip (CSV)**: 49,467 bytes | Ratio vs CSV: 2.74x | Ratio vs gzip: 1.00x | Swaps: 2,417 -> 2,417
  - **Hysteresis swap log CSV**: 1,244 bytes | Ratio vs CSV: 108.91x | Ratio vs gzip: 39.76x | Swaps: 2,417 -> 111 (noise std: 0.0597, $h$: 0.1194)
  - **TangleCode swap log CSV**: 560 bytes | Ratio vs CSV: 241.94x | Ratio vs gzip: 88.33x | Swaps: 2,417 -> 47

### Drift Fault Data (`test_fault.csv`)
- **Raw CSV Size**: 134,966 bytes
- **gzip Size**: 49,989 bytes
- **Methods**:
  - **Raw CSV**: 134,966 bytes | Ratio vs CSV: 1.00x | Ratio vs gzip: 0.37x | Swaps: 2,027 -> 2,027
  - **gzip (CSV)**: 49,989 bytes | Ratio vs CSV: 2.70x | Ratio vs gzip: 1.00x | Swaps: 2,027 -> 2,027
  - **Hysteresis swap log CSV**: 1,156 bytes | Ratio vs CSV: 116.75x | Ratio vs gzip: 43.24x | Swaps: 2,027 -> 103 (noise std: 0.0597, $h$: 0.1194)
  - **TangleCode swap log CSV**: 384 bytes | Ratio vs CSV: 351.47x | Ratio vs gzip: 130.18x | Swaps: 2,027 -> 31

### Stuck Sensor Data (`test_stuck.csv`)
- **Raw CSV Size**: 134,956 bytes
- **gzip Size**: 47,431 bytes
- **Methods**:
  - **Raw CSV**: 134,956 bytes | Ratio vs CSV: 1.00x | Ratio vs gzip: 0.35x | Swaps: 2,182 -> 2,182
  - **gzip (CSV)**: 47,431 bytes | Ratio vs CSV: 2.85x | Ratio vs gzip: 1.00x | Swaps: 2,182 -> 2,182
  - **Hysteresis swap log CSV**: 1,288 bytes | Ratio vs CSV: 104.78x | Ratio vs gzip: 36.83x | Swaps: 2,182 -> 115 (noise std: 0.0594, $h$: 0.1188)
  - **TangleCode swap log CSV**: 439 bytes | Ratio vs CSV: 307.42x | Ratio vs gzip: 108.04x | Swaps: 2,182 -> 36

---

## 60-Second Demo

1. **Step 1: Start App**
   Run `streamlit run app.py` in your terminal.
2. **Step 2: Normal Baseline**
   Select **"Healthy sensors"** and click **Analyze**.
   - Notice the green banner: **"All sensors normal"**.
   - Review metrics showing order swaps reduced from ~2,417 to ~47 (~98% noise removed).
   - The plot shows scores remaining below the alert level.
   - Click **Back**.
3. **Step 3: Fault Detection**
   Select **"One sensor drifting"** (or upload `data/test_fault.csv`) and click **Analyze**.
   - Notice the red alert banner: **"Fault detected: Sensor 4, around ticks 1600 to 3000"** (detecting the drift injected at tick 1500).
   - In the plot, the score climbs above the dashed "Alert level", with the faulty interval shaded red.
   - Use the **"Sensor ranking at a moment"** slider to watch Sensor 4 drift from 4th to 6th place and get highlighted in red as suspect.
   - Click **"Download swap log (CSV)"** to export the swap sequence, or **"Download fault report (PDF)"** to export a clean one-page diagnostic report.
