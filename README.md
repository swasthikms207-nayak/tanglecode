# TangleCode 🧬

**Topological Braid Group Sensor Encoding, Lossless-Rank Compression & Streaming Fault Detection**

TangleCode translates multichannel sensor streams (such as IMUs, EEGs, and strain gauges) into signed generators of the **Artin Braid Group $B_n$**. Transient noise loops correspond directly to **Reidemeister II crossing annihilations** ($\sigma_i \sigma_i^{-1} \to 1$), while true structural crossings and sensor faults remain invariant.

> **Note:** Rank-only mode keeps sensor ordering and discards exact amplitudes. It is built for structure and fault monitoring, not full recovery.

This yields:
1. **Topological Noise Annihilation**: Eliminates up to 81% of spurious jitter crossings without heuristic amplitude filters.
2. **Extreme Compression**: Over **93x compression ratio** compared to raw CSV by packing reduced braid generators into 4-bit nibbles.
3. **Streaming Anomaly Detection**: Real-time identification of sensor drift and frozen/stuck failures with low latency and low false-positive rates.

---

## Quickstart & Setup

### 1. Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/swasthikms207-nayak/tanglecode.git
cd tanglecode
pip install -r requirements.txt
```

### 2. Launch the Streamlit Single-Page App

```bash
streamlit run app.py
```
*(or `python -m streamlit run app.py`)*

Open [http://localhost:8501](http://localhost:8501) in your browser.

### 3. Run Automated Tests & Benchmarks

```bash
# Run pytest suite
python -m pytest -v

# Run compression benchmark
python -m tanglecode.benchmark
```

---

## ⏱️ 60-Second Demo Script

Present TangleCode in under 60 seconds:

- **0:00 – 0:15 | The Core Idea**  
  > *"Multichannel sensor streams are flooded with high-frequency jitter. Instead of filtering noisy continuous amplitudes, TangleCode maps sensor rank permutations into the Artin Braid Group $B_n$. Transient jitter loops correspond mathematically to Reidemeister II crossing cancellations ($\sigma_i \sigma_i^{-1} \to 1$) that collapse to identity."*

- **0:15 – 0:30 | Controls & Headline Metrics**  
  > *"Notice the top controls: adjust the noise slider or upload any sensor CSV. Look at the headline metrics: out of 2,113 raw crossings, braid reduction eliminates over 81% of jitter crossings, yielding a **93x compression ratio** over raw CSV in our compact `.tngl` format."*

- **0:30 – 0:45 | Streaming Anomaly Detection**  
  > *"Select **'Inject fault: drift'** or **'stuck'**. In the Plotly chart below, our streaming detector tracks irreducible braid-word length per window. The moment the fault begins, the topological weave deviates from the learned baseline, instantly highlighting the anomalous windows in red."*

- **0:45 – 1:00 | Binary Export & Edge Deployment**  
  > *"Finally, click **'Download .tngl file'** to save the nibble-packed topological binary. TangleCode delivers ultra-lightweight sensor representations built for edge microcontrollers and real-time structural health monitoring."*

---

## Single-Page UI (`app.py`) Layout

The web application is streamlined into a single focused page:
1. **Explanation & Note**: Concept overview and the explicit rank-only monitoring role.
2. **Controls**: Jitter noise level slider, "Inject fault" selector (`none`, `drift`, `stuck`), and CSV upload option.
3. **Headline Metrics**: Live measured counts for **Raw Crossings**, **Reduced Crossings**, and **Compression Ratio**.
4. **Topological Anomaly Detection Chart**: Interactive Plotly trajectory of irreducible reduced-word length across sliding windows with learned baseline $\mu \pm k\sigma$ threshold and flagged anomaly markers.
5. **Binary Export**: Instant download button for the generated `.tngl` file.

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
