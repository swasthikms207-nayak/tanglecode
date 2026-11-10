# TangleCode 🧬

**Topological Braid Group Sensor Encoding, Lossless-Rank Compression & Streaming Fault Detection**

TangleCode translates multichannel sensor streams (such as IMUs, EEGs, and strain gauges) into signed generators of the **Artin Braid Group $B_n$**. Transient noise loops correspond directly to **Reidemeister II crossing annihilations** ($\sigma_i \sigma_i^{-1} \to 1$), while true structural crossings and sensor faults remain invariant.

> **Note:** Rank-only mode keeps sensor ordering and discards exact amplitudes. It is built for structure and fault monitoring, not full recovery.

This yields:
1. **Topological Noise Annihilation**: Eliminates up to 81% of spurious jitter crossings without heuristic amplitude filters.
2. **Extreme Compression**: Over **93x compression ratio** compared to raw CSV by packing reduced braid generators into 4-bit nibbles.
3. **Streaming Anomaly & Fault Localization**: Real-time identification of sensor drift and frozen/stuck failures, pinpointing the specific faulty channel and most frequent crossing partner.
4. **Rich Multi-Format Exports**: One-click downloads for compressed `.tngl` binaries, diagnostic anomaly CSV reports, and comprehensive summary JSON benchmarks.

---

## Quickstart & Setup

### 1. Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/swasthikms207-nayak/tanglecode.git
cd tanglecode
pip install -r requirements.txt
```

### 2. Launch the Streamlit App

```bash
streamlit run app.py
```
*(or `python -m streamlit run app.py`)*

Open [http://localhost:8501](http://localhost:8501) in your browser.

### 3. Run Automated Tests & Benchmarks

```bash
# Run complete test suite (33 tests)
python -m pytest -v

# Run compression benchmark
python -m tanglecode.benchmark
```

---

## ⏱️ 60-Second Demo Script

Present TangleCode in under 60 seconds:

- **0:00 – 0:15 | The Core Idea**  
  > *"Multichannel sensor streams are flooded with high-frequency jitter. Instead of filtering noisy continuous amplitudes, TangleCode maps sensor rank permutations into the Artin Braid Group $B_n$. Transient jitter loops correspond mathematically to Reidemeister II crossing cancellations ($\sigma_i \sigma_i^{-1} \to 1$) that collapse to identity."*

- **0:15 – 0:30 | Page 1: Input & Analysis**  
  > *"On Page 1, choose between uploading custom sensor CSVs or curated samples: 'Clean drone IMU', 'IMU with drifting sensor', or 'IMU with stuck sensor'. Click **'Analyze'** to stream through rank extraction, braid reduction, nibble packing, and per-channel fault scoring in real-time."*

- **0:30 – 0:45 | Page 2: Metrics, Detection & Sensor Pinpointing**  
  > *"On Page 2, view the 3 headline metric cards: 81% jitter eliminated, yielding a **93x compression ratio**. In the Plotly chart, anomaly windows are shaded red. Directly below, TangleCode pinpoints the exact faulty sensor: e.g. **'Most likely faulty sensor: Channel 4 (crossed most with Channel 3), around ticks 1440 to 1715'**."*

- **0:45 – 1:00 | Sensor Pair Heatmap & Multi-Format Exports**  
  > *"The interactive heatmap reveals surviving crossings by sensor pair across the entire stream. Finally, download the nibble-packed `.tngl` binary, the detailed **'Anomaly report (CSV)'**, and the comprehensive **'Summary (JSON)'** ready for edge deployment."*

---

## Two-Page UI (`app.py`) Flow

### Page 1: Input
1. **Header & Description**: Concept summary.
2. **Data Source Selector**:
   - `Upload CSV`: File uploader detecting numeric sensor columns (requires $\ge 2$ channels).
   - `Use a sample`: Select from `Clean drone IMU`, `IMU with drifting sensor`, and `IMU with stuck sensor`.
3. **Analyze Button**: Runs the complete pipeline and transitions to Page 2.

### Page 2: Results
1. **Dataset Metadata**: Dataset name, channel count, and sample count.
2. **Headline Metric Cards**: Live measured counts for **Raw Crossings**, **Reduced Crossings**, and **Compression Ratio**.
3. **Topological Anomaly Detection Chart**: Plotly trajectory of irreducible reduced-word length across sliding windows with learned baseline $\mu \pm k\sigma$ threshold and red-shaded anomaly spans.
4. **Fault Diagnosis Sentence**:
   - If anomalies exist: `"Most likely faulty sensor: Channel X (crossed most with Channel Y), around ticks A to B."`
   - If none: `"No faulty sensor identified."`
5. **Sensor Pair Crossings Heatmap**: Interactive $n \times n$ Plotly heatmap showing surviving crossings between each sensor pair across the whole stream (blank diagonal).
6. **Triple Export Buttons**:
   - `Download .tngl file` (nibble-packed binary).
   - `Anomaly report (CSV)` (window ticks, z-score, threshold, faulty channel, top pair).
   - `Summary (JSON)` (dataset metrics, tick ranges, faulty channel).
7. **Lossy-Mode Callout & Back Button**: Small note and a `"Back"` button to clear results and return to Page 1.

---

## Binary Format Specification: `.tngl`

| Section | Size | Description |
| :--- | :---: | :--- |
| **Header** | 14 bytes | Magic `b"TNGL"` (4B), $n$ channels (uint8, $n \le 8$), raw generator count (uint32), reduced generator count (uint32), mode byte (uint8: `0 = MODE_RANK_ONLY`). |
| **Anchor Snapshot** | $4n$ bytes | First frame stored as $n \times \text{float32}$ continuous values. |
| **Timing (RLE)** | Variable | Run-length encoded per-tick generator counts preserving sample timing. |
| **Payload** | $\lceil M/2 \rceil$ bytes | Reduced generators packed 2-per-byte as 4-bit nibbles (bit 3: sign, bits 0-2: index). |

---

## Benchmark Results

Measured with `python -m tanglecode.benchmark` on synthetic IMU data (8 channels, 5,000 samples) and real gait IMU data (`data/real.csv`, 6 channels, 500 samples):

```
======================================================================
  BENCHMARK: Synthetic IMU (8 ch, 5000 samples)
======================================================================
Method                 |   Size (B) |    Ratio | Raw -> Reduced Crossings
----------------------------------------------------------------------
Raw CSV                |    382,564 |    1.00x |             2113 -> 2113
gzip (CSV)             |    163,960 |    2.33x |             2113 -> 2113
Hysteresis Baseline    |      7,568 |   50.55x |              2113 -> 903
TangleCode Rank-Only   |      4,111 |   93.06x |              2113 -> 397
======================================================================

======================================================================
  BENCHMARK: Real CSV: real.csv
======================================================================
Method                 |   Size (B) |    Ratio | Raw -> Reduced Crossings
----------------------------------------------------------------------
Raw CSV                |     28,516 |    1.00x |               382 -> 382
gzip (CSV)             |     10,570 |    2.70x |               382 -> 382
Hysteresis Baseline    |      1,515 |   18.82x |               382 -> 166
TangleCode Rank-Only   |      1,493 |   19.10x |               382 -> 182
======================================================================
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.
