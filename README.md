# TangleCode

TangleCode shrinks multi-sensor logs and warns you when one sensor starts misbehaving.

When several sensors measure related signals (such as accelerometers or temperature probes), high-frequency noise causes their relative order to jitter back and forth. TangleCode translates these order swaps into a mathematical sequence. Any swap that quickly reverses itself cancels out as transient noise. Genuine physical changes and sensor faults survive.

---

## What It Does in Plain Words

1. **Noise Removal**: Eliminates spurious order swaps without needing manual amplitude filters.
2. **Order-Based Compression**: Packs surviving swaps into an efficient `.tngl` binary format, drastically reducing storage size.
3. **Fault Warning**: Detects when a sensor drifts or gets stuck, identifies which sensor is misbehaving (e.g., Sensor 4), and highlights when the fault occurred.

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
Generate synthetic datasets and run the full end-to-end detection and round-trip verification:
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

## The `.tngl` Format

The `.tngl` file is a little-endian binary stream designed to compactly store sensor order history:

| Section | Size | Description |
| :--- | :---: | :--- |
| **Header** | 18 bytes | Magic `b"TNGL"` (4B), sensor count $n \le 8$ (`uint8`), raw swap count (`uint32`), surviving swap count (`uint32`), total frames $T$ (`uint32`), mode byte `0` (`uint8`). |
| **Anchor** | $4n$ bytes | Continuous floating-point values of the initial frame ($n \times \text{float32}$). |
| **Timing** | Variable | Tick difference from the previous survivor for each surviving swap, encoded as unsigned LEB128 varints. |
| **Payload** | $\lceil M/2 \rceil$ bytes | Surviving swaps packed 2-per-byte as 4-bit nibbles (bit 3: swap direction/sign, bits 0–2: swap position index $1..7$). Supports odd swap counts. |

---

## Limitations

- **Preserves order, not exact amplitudes**: The compressed format retains which sensor was above or below another, but discards exact numerical values beyond the first frame.
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
  - **Hysteresis baseline**: 215 bytes | Ratio vs CSV: 630.16x | Ratio vs gzip: 230.08x | Swaps: 2,417 -> 111 (noise std: 0.0597, $h$: 0.1194)
  - **TangleCode (with reduction)**: 119 bytes | Ratio vs CSV: 1,138.52x | Ratio vs gzip: 415.69x | Swaps: 2,417 -> 47

### Drift Fault Data (`test_fault.csv`)
- **Raw CSV Size**: 134,966 bytes
- **gzip Size**: 49,989 bytes
- **Methods**:
  - **Raw CSV**: 134,966 bytes | Ratio vs CSV: 1.00x | Ratio vs gzip: 0.37x | Swaps: 2,027 -> 2,027
  - **gzip (CSV)**: 49,989 bytes | Ratio vs CSV: 2.70x | Ratio vs gzip: 1.00x | Swaps: 2,027 -> 2,027
  - **Hysteresis baseline**: 203 bytes | Ratio vs CSV: 664.86x | Ratio vs gzip: 246.25x | Swaps: 2,027 -> 103 (noise std: 0.0597, $h$: 0.1194)
  - **TangleCode (with reduction)**: 95 bytes | Ratio vs CSV: 1,420.69x | Ratio vs gzip: 526.20x | Swaps: 2,027 -> 31

### Stuck Sensor Data (`test_stuck.csv`)
- **Raw CSV Size**: 134,956 bytes
- **gzip Size**: 47,431 bytes
- **Methods**:
  - **Raw CSV**: 134,956 bytes | Ratio vs CSV: 1.00x | Ratio vs gzip: 0.35x | Swaps: 2,182 -> 2,182
  - **gzip (CSV)**: 47,431 bytes | Ratio vs CSV: 2.85x | Ratio vs gzip: 1.00x | Swaps: 2,182 -> 2,182
  - **Hysteresis baseline**: 220 bytes | Ratio vs CSV: 613.44x | Ratio vs gzip: 215.60x | Swaps: 2,182 -> 115 (noise std: 0.0594, $h$: 0.1188)
  - **TangleCode (with reduction)**: 101 bytes | Ratio vs CSV: 1,336.20x | Ratio vs gzip: 469.61x | Swaps: 2,182 -> 36

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
   - Click **"Download compressed file (.tngl)"** to export the compacted sequence.
