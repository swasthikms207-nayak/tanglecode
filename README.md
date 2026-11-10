# TangleCode 🧬

**Topological Braid Group Sensor Encoding, Ultra-High Compression & Streaming Fault Detection**

TangleCode translates multichannel sensor streams (such as IMUs, EEGs, and strain gauges) into signed generators of the **Artin Braid Group $B_n$**. Transient noise loops correspond directly to **Reidemeister II crossing annihilations** ($\sigma_i \sigma_i^{-1} \to 1$), while true structural crossings and sensor faults remain invariant.

This yields:
1. **Topological Noise Rejection**: Up to 81% of spurious jitter crossings are annihilated without heuristic amplitude thresholds.
2. **Extreme Compression**: Up to **93x compression ratio** in Rank-Only mode, and **8.7x in Hybrid mode** with continuous amplitude recovery (RMSE < 0.006).
3. **Streaming Anomaly Detection**: Real-time detection of sensor drift and frozen/stuck failures with prompt detection delay (< 60 ticks) and low false-positive rates (< 1.5%).

---

## Quickstart & Setup

### 1. Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/swasthikms207-nayak/tanglecode.git
cd tanglecode
pip install -r requirements.txt
```

### 2. Launch the Streamlit Hackathon Demo

```bash
streamlit run app.py
```
*(or `python -m streamlit run app.py`)*

Open [http://localhost:8501](http://localhost:8501) in your browser.

### 3. Run Automated Tests & Benchmarks

```bash
# Run complete test suite (29 tests)
pytest -v

# Run compression & fidelity benchmarks
python -m tanglecode.benchmark
```

---

## ⏱️ 60-Second Hackathon Demo Script

Here is an effective script to present TangleCode to judges or audiences in 60 seconds:

- **0:00 – 0:15 | The Problem & The Core Idea**  
  > *"Every edge device from drones to wearables is flooded with noisy multichannel sensor streams. Filtering in the continuous amplitude domain introduces lag and drift. Instead, TangleCode projects channel dynamics into algebraic topology — specifically the Artin Braid Group $B_n$. Transient jitter loops correspond mathematically to Reidemeister II crossing annihilations ($\sigma_i \sigma_i^{-1} \to 1$). When we simplify the braid, the noise cancels itself out!"*

- **0:15 – 0:30 | Tab 1: Live Braid & Reidemeister Jitter Annihilation**  
  > *"Look at Tab 1. Here are 8 sensor channels weaving over time. Notice the crossings marked along the strands. As we increase the jitter slider, dozens of spurious noise crossings appear. Now watch what happens when we toggle **'Apply Reidemeister reduction'**: instantly, the transient noise loops vanish! Over 80% of crossings collapse, leaving only the true topological weave."*

- **0:30 – 0:45 | Tab 2: 93x Compression & Tab 3: Streaming Fault Detection**  
  > *"In Tab 2, our `.tngl` binary format packs reduced generators into 4-bit nibbles, achieving **93x compression** over raw CSV in Rank-Only mode, and **8.7x in Hybrid mode** with an RMSE under 0.006 (beating gzip and zstandard by 3.5x). In Tab 3, watch our streaming anomaly detector: click **'Inject Fault'** to simulate sensor drift or a frozen channel. Within 60 ticks, the reduced braid length breaks the baseline threshold, flagging the anomaly with less than 1.5% false positives."*

- **0:45 – 1:00 | Tab 4: Upload Custom Data & Export .tngl Binaries**  
  > *"Finally, in Tab 4, upload any real sensor CSV. TangleCode automatically parses numeric channels, computes braid word reductions, and gives you a one-click download of the compressed `.tngl` binary. TangleCode turns noisy multichannel data into compact topological invariants ready for edge microcontrollers."*

---

## Interactive Streamlit Demo Overview

The demo app ([`app.py`](app.py)) is organized into 4 interactive tabs:

### 1. 🧬 Live Braid
- **Animated Weaving Strands**: 8 vibrant colored strands weaving across time with individual crossing markers.
- **Jitter Noise Slider**: Live simulation of high-frequency sensor noise.
- **Reidemeister Toggle**: Toggles live braid reduction on and off, with live metrics showing Raw Crossings, Reduced Crossings, and Annihilated Jitter.

### 2. 📊 Compression Benchmark
- **Live Benchmarking**: Live measurement comparing Raw CSV, gzip, zstandard, Hysteresis, TangleCode Rank-Only, and TangleCode Hybrid.
- **Interactive Plotly Visualizations**: Log-scale compressed size and compression ratio comparisons.
- **Fidelity Metrics**: Kendall's rank correlation $\tau$ and amplitude RMSE.

### 3. 🚨 Fault Detection
- **Fault Injection Controls**: Simulate linear sensor drift or stuck/frozen channels on any selected channel.
- **Streaming Sliding Window Analysis**: Real-time trajectory of reduced word length per window with baseline $\pm k\sigma$ confidence bands and highlighted anomaly diamonds.
- **Performance Metrics**: False positive rate (FPR) and detection delay reported live.

### 4. 📁 Upload & Export
- **Custom CSV Ingestion**: Drag and drop your own sensor CSV files.
- **Topological Compression**: Immediate conversion to both Rank-Only and Hybrid `.tngl` formats.
- **Binary Export**: Instant download buttons for `.tngl` binary packages.

---

## Binary Format Specification: `.tngl`

| Section | Size | Description |
| :--- | :---: | :--- |
| **Header** | 14 bytes | Magic `b"TNGL"` (4B), $n$ channels (uint8, $n \le 8$), raw generator count (uint32), reduced generator count (uint32), mode byte (uint8: 0=rank-only, 1=hybrid). |
| **Anchor Snapshot** | $4n$ bytes | First frame stored as $n \times \text{float32}$ continuous amplitudes. |
| **Timing (RLE)** | Variable | Run-length encoded per-tick generator counts preserving timing. |
| **Payload** | $\lceil M/2 \rceil$ bytes | Reduced generators packed 2-per-byte as 4-bit nibbles (bit 3: sign, bits 0-2: generator index). |
| **Hybrid Section** | $4 + (T-1)n$ bytes | *(Mode 1 only)* float32 delta step size followed by int8 closed-loop delta modulation bytes. |

---

## Benchmark Results Summary

Measured on synthetic IMU signals ($8$ channels, $5,000$ samples, $2,113$ raw crossings) and real gait IMU data ($6$ channels, $500$ samples, $382$ raw crossings):

| Method | Synthetic Size | Synthetic Ratio | Crossings Eliminated | Kendall $\tau$ | RMSE |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Raw CSV** | 382,564 B | 1.00x | 0.0% | 1.0000 | 0.0000 |
| **gzip (CSV)** | 163,960 B | 2.33x | 0.0% | 1.0000 | 0.0000 |
| **zstandard (float32)** | 147,949 B | 2.59x | 0.0% | 1.0000 | 0.0000 |
| **Hysteresis Baseline** | 7,568 B | 50.55x | 57.3% | 0.9851 | N/A |
| **TangleCode Rank-Only** | **4,111 B** | **93.06x** | **81.2%** | 0.9194 | N/A |
| **TangleCode Hybrid** | 44,107 B | 8.67x | 81.2% | **0.9976** | **0.0058** |

---

## License

MIT License. See [LICENSE](LICENSE) for details.
