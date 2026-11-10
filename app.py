"""TangleCode: Single-page Streamlit application for topological braid sensor encoding."""

import io
import tempfile
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from tanglecode.anomaly import BraidAnomalyDetector
from tanglecode.braid import extract_braid_word, rank_vector
from tanglecode.codec import MODE_RANK_ONLY, encode
from tanglecode.data import generate_synthetic, inject_fault, load_csv

# Page configuration
st.set_page_config(
    page_title="TangleCode | Topological Braid Sensor Compression & Anomaly Detection",
    page_icon="🧬",
    layout="wide",
)

# -----------------------------------------------------------------------------
# 1. Explanation of the idea and visible note
# -----------------------------------------------------------------------------
st.title("🧬 TangleCode")
st.markdown(
    """
    High-frequency multichannel sensor streams (such as IMUs, EEGs, and strain gauges) are inundated with
    jitter and measurement noise. Instead of filtering signals in the noisy amplitude domain, **TangleCode**
    maps multichannel sensor rank ordering into the **Artin Braid Group \\(B_n\\)**. Transient noise loops
    collapse to identity via Reidemeister II topological cancellations ($\\sigma_i \\sigma_i^{-1} \\to 1$),
    while true physical dynamics and sensor faults generate persistent topological tangles. This unlocks extreme
    lossless-rank compression and sensitive streaming anomaly detection directly in the topological domain.
    """
)

st.info(
    "**Note:** Rank-only mode keeps sensor ordering and discards exact amplitudes. "
    "It is built for structure and fault monitoring, not full recovery."
)

# -----------------------------------------------------------------------------
# 2. Controls: noise slider, "Inject fault" selectbox, CSV upload option
# -----------------------------------------------------------------------------
st.subheader("Controls")
ctrl_col1, ctrl_col2, ctrl_col3 = st.columns(3)

with ctrl_col1:
    noise_level = st.slider(
        "Jitter Noise Level",
        min_value=0.00,
        max_value=0.20,
        value=0.05,
        step=0.01,
        help="Standard deviation of additive Gaussian noise.",
    )

with ctrl_col2:
    fault_type = st.selectbox(
        "Inject fault",
        options=["none", "drift", "stuck"],
        index=0,
        help="Simulate sensor failure: none, drift (linear offset), or stuck (frozen sensor).",
    )

with ctrl_col3:
    uploaded_file = st.file_uploader(
        "CSV upload option (optional)",
        type=["csv"],
        help="Upload custom sensor CSV (numeric columns used as channels, up to 8 channels).",
    )

# Load or generate dataset
if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = Path(tmp.name)
    try:
        raw_data = load_csv(tmp_path)
        if raw_data.shape[1] > 8:
            raw_data = raw_data[:, :8]
            st.warning("Data truncated to first 8 channels (maximum supported by .tngl format).")
        if noise_level > 0.0:
            rng = np.random.default_rng(42)
            raw_data = raw_data + rng.normal(0.0, noise_level, size=raw_data.shape).astype(np.float32)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
else:
    raw_data = generate_synthetic(
        n_channels=8,
        n_samples=3000,
        noise_std=noise_level,
        seed=42,
    )

# Apply fault injection if requested
if fault_type != "none":
    fault_start = len(raw_data) // 2
    active_data = inject_fault(raw_data, start=fault_start, kind=fault_type, channel=0)
else:
    active_data = raw_data

# -----------------------------------------------------------------------------
# 3. Three headline metrics: raw crossings, reduced crossings, compression ratio
# -----------------------------------------------------------------------------
# Encode with TangleCode rank-only
tngl_bytes = encode(active_data, mode=MODE_RANK_ONLY)

# Measure raw crossings
n_samples, n_channels = active_data.shape
total_raw_crossings = 0
for t in range(n_samples - 1):
    w = extract_braid_word(rank_vector(active_data[t]), rank_vector(active_data[t + 1]))
    total_raw_crossings += len(w)

# Extract reduced crossings from header (bytes 5..8 is raw, 9..12 is reduced)
import struct
_, _, _, total_reduced_crossings, _ = struct.unpack_from("<4sBIIB", tngl_bytes, 0)

# Format raw CSV bytes to measure true compression ratio
csv_buffer = io.StringIO()
csv_buffer.write(",".join(f"Ch_{c}" for c in range(n_channels)) + "\n")
for row in active_data:
    csv_buffer.write(",".join(f"{v:.6f}" for v in row) + "\n")
raw_csv_size = len(csv_buffer.getvalue().encode("utf-8"))
tngl_size = len(tngl_bytes)
compression_ratio = raw_csv_size / tngl_size if tngl_size > 0 else 1.0

st.subheader("Headline Metrics")
metric_col1, metric_col2, metric_col3 = st.columns(3)

with metric_col1:
    st.metric(
        label="Raw Crossings",
        value=f"{total_raw_crossings:,}",
        help="Total Artin braid generators emitted before topological reduction.",
    )

with metric_col2:
    eliminated_pct = (
        ((total_raw_crossings - total_reduced_crossings) / total_raw_crossings * 100.0)
        if total_raw_crossings > 0
        else 0.0
    )
    st.metric(
        label="Reduced Crossings",
        value=f"{total_reduced_crossings:,}",
        delta=f"-{eliminated_pct:.1f}% jitter eliminated" if total_raw_crossings > 0 else None,
        help="Irreducible topological crossings after Reidemeister II & commuting reductions.",
    )

with metric_col3:
    st.metric(
        label="Compression Ratio",
        value=f"{compression_ratio:.1f}x",
        delta=f"{raw_csv_size:,} B → {tngl_size:,} B",
        help="Ratio of raw CSV representation size to compressed .tngl binary size.",
    )

# -----------------------------------------------------------------------------
# 4. One Plotly chart: reduced braid-word length per window, with flagged anomaly windows
# -----------------------------------------------------------------------------
st.subheader("Topological Anomaly Detection")

# Slide window and compute anomaly records
detector = BraidAnomalyDetector(
    window_size=200,
    n_baseline=30,
    k=2.5,
    step=15,
)
records = detector.predict(active_data)

window_starts = [r.window_start for r in records]
reduced_lengths = [r.reduced_length for r in records]
flagged_starts = [r.window_start for r in records if r.flagged]
flagged_lengths = [r.reduced_length for r in records if r.flagged]

fig = go.Figure()

# Plot reduced word length trajectory
fig.add_trace(
    go.Scatter(
        x=window_starts,
        y=reduced_lengths,
        mode="lines",
        name="Reduced Braid-Word Length",
        line=dict(color="#00d2ff", width=2),
        hovertemplate="Tick %{x}<br>Reduced Length: %{y}<extra></extra>",
    )
)

# Baseline threshold if available
if detector.baseline_mean is not None and detector.baseline_std is not None:
    mean_val = detector.baseline_mean
    upper_thresh = mean_val + detector.k * detector.baseline_std
    fig.add_hline(
        y=upper_thresh,
        line=dict(color="#ff9900", dash="dash", width=1.5),
        annotation_text=f"Threshold (μ + {detector.k}σ = {upper_thresh:.1f})",
        annotation_position="top right",
    )
    fig.add_hline(
        y=mean_val,
        line=dict(color="#888888", dash="dot", width=1.0),
        annotation_text=f"Baseline Mean ({mean_val:.1f})",
        annotation_position="bottom right",
    )

# Highlight flagged anomaly windows
if flagged_starts:
    fig.add_trace(
        go.Scatter(
            x=flagged_starts,
            y=flagged_lengths,
            mode="markers",
            name="Flagged Anomaly",
            marker=dict(
                color="#ff0055",
                size=11,
                symbol="diamond",
                line=dict(width=1, color="#ffffff"),
            ),
            hovertemplate="<b>ANOMALY DETECTED</b><br>Tick %{x}<br>Length: %{y}<extra></extra>",
        )
    )

fig.update_layout(
    template="plotly_dark",
    title="Irreducible Braid-Word Length per Sliding Window",
    xaxis_title="Time Tick (Window Start)",
    yaxis_title="Irreducible Reduced-Word Length",
    height=420,
    margin=dict(l=40, r=40, t=50, b=40),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1.0,
    ),
)

st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------------------------------
# 5. Download button for the .tngl file
# -----------------------------------------------------------------------------
st.subheader("Export")
filename = "sensor_data.tngl"
st.download_button(
    label="⬇️ Download .tngl file",
    data=tngl_bytes,
    file_name=filename,
    mime="application/octet-stream",
    help="Download compressed topological .tngl binary file.",
)
