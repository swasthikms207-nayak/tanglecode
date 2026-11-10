# TangleCode

Topological braid group representations and reductions for multichannel time series.

TangleCode translates order changes across multichannel time-series signals into signed Artin braid generators and performs Reidemeister reductions, distinguishing transient noise/jitter from persistent topological crossings.

## Features

- **`rank_vector(frame)`**: Returns argsort ranking of $n$ floats with deterministic channel-index tie breaking.
- **`extract_braid_word(prev_ranks, curr_ranks)`**: Bubble-sort adjacent swaps converting ordering shifts into signed Artin generators ($\sigma_i$ as $+i$, $\sigma_i^{-1}$ as $-i$). Priority sign rule deterministically assigns crossing signs based on strand index.
- **`reduce_word(word)`**: Online single-pass stack reduction handling free cancellation ($\sigma_i \sigma_i^{-1} = 1$, Reidemeister II move).
- **`reduce_word_full(word)`**: Full topological reduction applying:
  - Far-commutativity ($\sigma_i \sigma_j = \sigma_j \sigma_i$ for $|i - j| \ge 2$) via canonical sorting of commuting neighbors.
  - Braid relation ($\sigma_i \sigma_{i+1} \sigma_i \leftrightarrow \sigma_{i+1} \sigma_i \sigma_{i+1}$) applied whenever it unlocks further reductions.
- **`generate_synthetic(...)`**: Generates multi-channel simulated IMU signals (sinusoids with channel frequencies + Gaussian noise).
- **`inject_fault(...)`**: Injects sensor faults (`"drift"` or `"stuck"`) into designated channels.
- **`load_csv(...)`**: Ingests sensor data from CSV files, automatically isolating numeric sensor channels.
- **`.tngl` Binary Codec (`encode` / `decode`)**:
  - Compact 14-byte header (`magic="TNGL"`, $n$, raw generator count, reduced generator count, mode).
  - First frame anchor snapshot as float32 per channel.
  - Run-length encoded per-tick generator counts preserving timing.
  - 4-bit nibble packing for signed Artin generators (2 per byte, supporting $n \le 8$).
  - **Rank-only Mode** (lossy amplitude, exact rank order trajectory).
  - **Hybrid Mode** (int8 delta-modulated amplitudes with configurable step size).
- **Streaming Anomaly Detection (`tanglecode.anomaly`)**:
  - Sliding window topological analysis extracting reduced word lengths, compression ratios, and dominant generators.
  - Baseline distribution learning with $k$-sigma thresholding.
  - Streaming detection of sensor drift and frozen/stuck faults with low false-positive rates and prompt detection delay.

## Project Structure

```text
tanglecode/
├── tanglecode/
│   ├── __init__.py
│   ├── anomaly.py
│   ├── braid.py
│   ├── codec.py
│   ├── data.py
│   └── benchmark.py
├── tests/
│   ├── __init__.py
│   ├── test_anomaly.py
│   ├── test_benchmark.py
│   ├── test_braid.py
│   ├── test_codec.py
│   └── test_data.py
├── data/
│   ├── real.csv
│   └── sample_signals.json
├── app.py
├── requirements.txt
├── results.json
├── LICENSE
└── README.md
```

## Quick Start

### Installation

```bash
git clone https://github.com/swasthikms207-nayak/tanglecode.git
cd tanglecode
pip install -r requirements.txt
```

### Running Tests

```bash
pytest -v
```

### Running the Demo Application

```bash
python app.py
```

## License

MIT License. See [LICENSE](LICENSE) for details.
