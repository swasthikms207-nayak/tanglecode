"""TangleCode: Topological Braid Group Sensor Encoding & Hackathon Demo.

Interactive Streamlit application showcasing:
1. Dataset / Upload: Ingest custom CSV or choose from curated sample datasets, preview signals, and export .tngl binaries.
2. Live Braid: Weave strands for the active dataset, adjust jitter noise, and toggle Reidemeister reduction.
3. Compression: Live benchmarks comparing raw CSV, gzip, zstandard, hysteresis, and TangleCode.
4. Fault Detection: Streaming braid anomaly detection on drift and stuck faults injected into the active dataset.
"""

from io import BytesIO
from pathlib import Path
import time

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from tanglecode.anomaly import BraidAnomalyDetector
from tanglecode.benchmark import benchmark_dataset
from tanglecode.braid import extract_braid_word, rank_vector, reduce_word_full
from tanglecode.codec import (
    MODE_HYBRID,
    MODE_RANK_ONLY,
    decode,
    encode,
)
from tanglecode.data import generate_synthetic, inject_fault, load_csv

# Page configuration
st.set_page_config(
    page_title="TangleCode | Topological Braid Group Sensor Encoding",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling for polished dark theme look
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 800;
        background: -webkit-linear-gradient(45deg, #00d2ff, #3a7bd5, #00f2fe);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        color: #94a3b8;
        font-size: 1.05rem;
        margin-bottom: 1.2rem;
        line-height: 1.5;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# App Header
st.markdown('<div class="main-title">🧬 TangleCode</div>', unsafe_allow_html=True)
st.markdown(
    """
    <div class="sub-title">
    High-frequency multichannel sensor streams (such as IMUs, EEGs, and strain gauges) are inundated with high-frequency jitter and sensor noise. Instead of filtering signals in the noisy amplitude domain, <b>TangleCode</b> maps multichannel ordering dynamics into the <b>Artin Braid Group B<sub>n</sub></b>. Transient noise loops correspond directly to Reidemeister II crossing annihilations (&sigma;<sub>i</sub> &sigma;<sub>i</sub><sup>-1</sup> &rarr; 1), while true structural crossings and sensor faults remain invariant. This enables extreme compression (up to 93x), zero-drift delta reconstruction, and streaming anomaly detection with low false-positive rates.
    </div>
    """,
    unsafe_allow_html=True,
)

# Color palette for up to 8 strands
CHANNEL_COLORS = [
    "#00d2ff",
    "#ff007f",
    "#00ffcc",
    "#ffaa00",
    "#b026ff",
    "#ff416c",
    "#ffe600",
    "#00e676",
]

# Tabs order: Start directly from Dataset / Upload!
tabs = st.tabs(["📁 Dataset & Upload", "🧬 Live Braid", "📊 Compression Benchmark", "🚨 Fault Detection"])


# ==============================================================================
# TAB 1: DATASET & UPLOAD (LANDING PAGE)
# ==============================================================================
with tabs[0]:
    st.subheader("Dataset Ingestion & Binary Export")
    st.write(
        "Start here by selecting a curated sample dataset or uploading your own CSV file. "
        "The active dataset selected in this tab will power the Live Braid weaver, Compression benchmark, and Fault detector tabs."
    )

    data_source = st.radio(
        "Data Source Selection",
        [
            "Upload Custom CSV File",
            "Sample: Real 6-Axis Motion Sensor (data/real.csv)",
            "Sample: Synthetic 8-Channel IMU Stream",
            "Sample: Synthetic 4-Channel Correlated Stream",
        ],
        index=1 if Path("data/real.csv").exists() else 2,
        horizontal=True,
    )

    loaded_data = None
    dataset_label = ""

    if data_source == "Upload Custom CSV File":
        uploaded_file = st.file_uploader(
            "Upload CSV file (numeric columns will be extracted as channels)",
            type=["csv"],
            help="Upload a CSV with 2 to 8 numeric channels",
        )
        if uploaded_file is not None:
            try:
                temp_path = Path("data/_temp_upload.csv")
                temp_path.write_bytes(uploaded_file.getvalue())
                loaded_data = load_csv(temp_path)
                dataset_label = f"Uploaded: {uploaded_file.name}"
                if temp_path.exists():
                    temp_path.unlink()
            except Exception as e:
                st.error(f"Error reading CSV: {e}")
        else:
            st.info("👆 Please upload a CSV file or choose one of the sample datasets above.")

    elif data_source == "Sample: Real 6-Axis Motion Sensor (data/real.csv)":
        real_csv_path = Path("data/real.csv")
        if real_csv_path.exists():
            loaded_data = load_csv(real_csv_path)
            dataset_label = "Real 6-Axis Sensor (data/real.csv)"
        else:
            st.warning("data/real.csv not found on disk. Falling back to synthetic 8-channel IMU.")
            loaded_data = generate_synthetic(n_channels=8, n_samples=1000, noise_std=0.03, seed=42)
            dataset_label = "Synthetic 8-Channel IMU"

    elif data_source == "Sample: Synthetic 8-Channel IMU Stream":
        loaded_data = generate_synthetic(n_channels=8, n_samples=1500, noise_std=0.03, seed=0)
        dataset_label = "Synthetic 8-Channel IMU (1500 samples)"

    elif data_source == "Sample: Synthetic 4-Channel Correlated Stream":
        loaded_data = generate_synthetic(n_channels=4, n_samples=1200, noise_std=0.02, seed=42)
        dataset_label = "Synthetic 4-Channel Correlated Stream (1200 samples)"

    # Process loaded data
    if loaded_data is not None:
        n_rows, n_cols = loaded_data.shape
        if n_cols > 8:
            st.warning(
                f"Dataset contains {n_cols} channels. The `.tngl` format supports up to 8 channels. Using the first 8."
            )
            loaded_data = loaded_data[:, :8]
            n_cols = 8

        # Save to session_state so all tabs use this exact dataset
        st.session_state["active_data"] = loaded_data
        st.session_state["dataset_name"] = dataset_label

        st.success(f"Active Dataset: **{dataset_label}** — {n_rows:,} samples, {n_cols} channels")

        # Encode dataset into both modes for immediate export & stats
        step_size_val = st.number_input(
            "Hybrid Delta Step Size",
            min_value=0.001,
            max_value=1.0,
            value=0.02,
            step=0.005,
            help="Step size for int8 delta modulation in Hybrid mode",
        )

        tngl_rank = encode(loaded_data, mode=MODE_RANK_ONLY)
        tngl_hybrid = encode(loaded_data, mode=MODE_HYBRID, step_size=step_size_val)
        raw_size = loaded_data.nbytes

        u1, u2, u3, u4 = st.columns(4)
        with u1:
            st.metric("Raw Float32 Size", f"{raw_size:,} B")
        with u2:
            st.metric(
                "Rank-Only .tngl Size",
                f"{len(tngl_rank):,} B",
                delta=f"{raw_size / len(tngl_rank):.1f}x",
            )
        with u3:
            st.metric(
                "Hybrid .tngl Size",
                f"{len(tngl_hybrid):,} B",
                delta=f"{raw_size / len(tngl_hybrid):.1f}x",
            )
        with u4:
            recon_hyb = decode(tngl_hybrid)
            rmse_calc = float(np.sqrt(np.mean((recon_hyb - loaded_data) ** 2)))
            st.metric("Hybrid RMSE Error", f"{rmse_calc:.5f}")

        # Download Buttons
        d_col1, d_col2 = st.columns(2)
        with d_col1:
            st.download_button(
                label="📥 Download Rank-Only .tngl Binary",
                data=tngl_rank,
                file_name="stream_rank_only.tngl",
                mime="application/octet-stream",
            )
        with d_col2:
            st.download_button(
                label="📥 Download Hybrid .tngl Binary",
                data=tngl_hybrid,
                file_name="stream_hybrid.tngl",
                mime="application/octet-stream",
            )

        # Plot Channel Preview
        fig_preview = go.Figure()
        preview_samples = min(600, n_rows)
        for c in range(n_cols):
            fig_preview.add_trace(
                go.Scatter(
                    y=loaded_data[:preview_samples, c],
                    name=f"Channel {c}",
                    line=dict(width=1.5, color=CHANNEL_COLORS[c % len(CHANNEL_COLORS)]),
                )
            )
        fig_preview.update_layout(
            title=f"Dataset Preview ({dataset_label} — First {preview_samples} samples)",
            template="plotly_dark",
            xaxis_title="Time (Ticks)",
            yaxis_title="Sensor Amplitude",
            height=340,
            margin=dict(l=40, r=40, t=40, b=30),
        )
        st.plotly_chart(fig_preview, use_container_width=True)

    else:
        # Fallback dataset if nothing is selected
        if "active_data" not in st.session_state:
            default_data = generate_synthetic(n_channels=6, n_samples=1000, noise_std=0.03, seed=42)
            st.session_state["active_data"] = default_data
            st.session_state["dataset_name"] = "Default 6-Channel Stream"


# Retrieve active data for remaining tabs
active_data = st.session_state.get("active_data")
dataset_name = st.session_state.get("dataset_name", "Current Dataset")


# ==============================================================================
# TAB 2: LIVE BRAID (WEAVER & JITTER ANNIHILATION ON ACTIVE DATASET)
# ==============================================================================
with tabs[1]:
    st.subheader(f"Live Braid Weaver & Reidemeister Jitter Annihilation")
    st.caption(f"Currently analyzing active dataset: **{dataset_name}** ({active_data.shape[0]:,} samples, {active_data.shape[1]} channels)")

    n_samples_total, n_channels = active_data.shape

    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([1.5, 1.5, 1.5])
    with col_ctrl1:
        n_ticks = st.slider("Window Length (ticks)", 20, min(150, n_samples_total), min(60, n_samples_total), 5)
    with col_ctrl2:
        max_start = max(0, n_samples_total - n_ticks)
        tick_offset = st.slider("Window Start Offset", 0, max_start, 0, 10 if max_start > 10 else 1)
    with col_ctrl3:
        extra_noise = st.slider("Add Extra Jitter Noise (std)", 0.00, 0.15, 0.00, 0.01)
        apply_reduction = st.toggle("Apply Reidemeister reduction", value=False)

    # Slice window from active dataset and add optional extra jitter
    sub_stream = active_data[tick_offset : tick_offset + n_ticks].copy()
    if extra_noise > 0:
        rng = np.random.default_rng(tick_offset)
        sub_stream += rng.normal(0.0, extra_noise, size=sub_stream.shape).astype(np.float32)

    # Compute raw rank trajectories and crossings
    raw_ranks = np.zeros((n_ticks, n_channels), dtype=np.int32)
    for t in range(n_ticks):
        current_rank = rank_vector(sub_stream[t])
        for c in range(n_channels):
            raw_ranks[t, c] = current_rank.index(c)

    raw_crossings = []
    total_raw_crossings = 0
    for t in range(n_ticks - 1):
        step_word = extract_braid_word(
            rank_vector(sub_stream[t]), rank_vector(sub_stream[t + 1])
        )
        total_raw_crossings += len(step_word)
        for g in step_word:
            raw_crossings.append((t + 0.5, abs(g) - 0.5, g))

    # Compute reduced trajectories using TangleCode
    encoded_rank = encode(sub_stream, mode=MODE_RANK_ONLY)
    recon_rank = decode(encoded_rank)

    # Count reduced crossings
    reduced_crossings = []
    total_reduced_crossings = 0
    for t in range(n_ticks - 1):
        r_prev = rank_vector(recon_rank[t])
        r_curr = rank_vector(recon_rank[t + 1])
        step_word = extract_braid_word(r_prev, r_curr)
        total_reduced_crossings += len(step_word)
        for g in step_word:
            reduced_crossings.append((t + 0.5, abs(g) - 0.5, g))

    eliminated_count = total_raw_crossings - total_reduced_crossings
    eliminated_pct = (
        (eliminated_count / total_raw_crossings * 100) if total_raw_crossings > 0 else 0.0
    )

    # Metrics
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Raw Crossings in Window", f"{total_raw_crossings}")
    with m2:
        st.metric(
            "Reduced Crossings",
            f"{total_reduced_crossings}",
            delta=f"-{eliminated_pct:.1f}%",
            delta_color="inverse",
        )
    with m3:
        st.metric("Jitter Annihilated", f"{eliminated_count}")
    with m4:
        st.metric("Reidemeister Reduction", "ACTIVE" if apply_reduction else "OFF (Raw Weave)")

    # Plot Weaving Strands
    fig_braid = go.Figure()
    plot_ranks = recon_rank if apply_reduction else raw_ranks
    active_crossings = reduced_crossings if apply_reduction else raw_crossings

    for c in range(n_channels):
        fig_braid.add_trace(
            go.Scatter(
                x=list(range(n_ticks)),
                y=plot_ranks[:, c],
                mode="lines+markers",
                name=f"Channel {c}",
                line=dict(color=CHANNEL_COLORS[c % len(CHANNEL_COLORS)], width=3, shape="spline"),
                marker=dict(size=4),
                hoverinfo="text",
                hovertext=[
                    f"Ch {c} | Tick {t + tick_offset} | Rank {plot_ranks[t, c]}" for t in range(n_ticks)
                ],
            )
        )

    # Mark Crossings
    if active_crossings:
        cross_x = [pt[0] for pt in active_crossings]
        cross_y = [pt[1] for pt in active_crossings]
        cross_text = [
            f"Artin σ_{abs(pt[2])}{'⁺¹' if pt[2] > 0 else '⁻¹'}" for pt in active_crossings
        ]
        fig_braid.add_trace(
            go.Scatter(
                x=cross_x,
                y=cross_y,
                mode="markers",
                name="Crossings",
                marker=dict(
                    symbol="x",
                    size=10,
                    color="#ffffff" if apply_reduction else "#ff0055",
                    line=dict(width=2, color="#000000"),
                ),
                hoverinfo="text",
                hovertext=cross_text,
            )
        )

    title_text = (
        f"Irreducible Topological Braid ({dataset_name} — Jitter Annihilated via Reidemeister Moves)"
        if apply_reduction
        else f"Raw Multi-Channel Braid Weave ({dataset_name} — Raw Crossings Marked)"
    )
    fig_braid.update_layout(
        title=title_text,
        template="plotly_dark",
        xaxis_title=f"Relative Window Ticks (Offset {tick_offset})",
        yaxis_title=f"Strand Rank Position (0 to {n_channels - 1})",
        yaxis=dict(autorange="reversed", dtick=1),
        height=520,
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_braid, use_container_width=True)


# ==============================================================================
# TAB 3: COMPRESSION BENCHMARK (ON ACTIVE DATASET)
# ==============================================================================
with tabs[2]:
    st.subheader("Compression & Topological Reduction Benchmark")
    st.caption(f"Benchmarking compression on active dataset: **{dataset_name}** ({active_data.shape[0]:,} samples, {active_data.shape[1]} channels)")

    st.info(
        "ℹ️ **Encoding Modes**: **Rank-Only Mode** reconstructs exact channel rank order per tick "
        "(lossy amplitude, ultra-high compression up to 93x). **Hybrid Mode** combines braid topology "
        "with int8 closed-loop delta modulation to reconstruct continuous amplitudes with minimal error (RMSE < 0.01)."
    )

    btn_benchmark = st.button("🚀 Run Live Benchmark on Active Dataset", type="primary")

    bench_key = f"bench_{dataset_name}_{active_data.shape}"
    if btn_benchmark or bench_key not in st.session_state:
        with st.spinner(f"Running live benchmark on {dataset_name}..."):
            st.session_state[bench_key] = benchmark_dataset(active_data, dataset_name)

    results = st.session_state[bench_key]
    methods = results["methods"]

    # Table View
    table_rows = []
    for method_name, metrics in methods.items():
        table_rows.append(
            {
                "Method": method_name,
                "Size (Bytes)": f"{metrics['compressed_size_bytes']:,}",
                "Ratio": f"{metrics['compression_ratio']:.2f}x",
                "Crossings Eliminated": f"{metrics['crossings_eliminated_pct']:.1f}%",
                "Encode Time / Frame": f"{metrics['encode_time_per_frame_us']:.2f} µs",
                "Kendall Tau (Fidelity)": f"{metrics['kendall_tau']:.4f}",
                "RMSE (Amplitude)": (
                    f"{metrics['rmse']:.4f}" if metrics["rmse"] is not None else "N/A"
                ),
            }
        )
    st.dataframe(table_rows, use_container_width=True)

    # Bar Charts
    col_chart1, col_chart2 = st.columns(2)
    method_names = list(methods.keys())
    sizes = [m["compressed_size_bytes"] for m in methods.values()]
    ratios = [m["compression_ratio"] for m in methods.values()]
    colors = ["#475569", "#3b82f6", "#06b6d4", "#f59e0b", "#10b981", "#8b5cf6"]

    with col_chart1:
        fig_bar = go.Figure()
        fig_bar.add_trace(
            go.Bar(
                x=method_names,
                y=sizes,
                marker=dict(color=colors[: len(method_names)]),
                text=[f"{s:,} B" for s in sizes],
                textposition="auto",
            )
        )
        fig_bar.update_layout(
            title="Compressed File Size (Bytes) — Lower is Better",
            template="plotly_dark",
            yaxis_title="Size (Bytes, Log Scale)",
            yaxis_type="log",
            height=420,
            xaxis_tickangle=-25,
            margin=dict(l=40, r=40, t=50, b=80),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_chart2:
        fig_ratio = go.Figure()
        fig_ratio.add_trace(
            go.Bar(
                x=method_names,
                y=ratios,
                marker=dict(color=colors[: len(method_names)]),
                text=[f"{r:.1f}x" for r in ratios],
                textposition="auto",
            )
        )
        fig_ratio.update_layout(
            title="Compression Ratio vs Raw CSV — Higher is Better",
            template="plotly_dark",
            yaxis_title="Ratio (Multiplier)",
            height=420,
            xaxis_tickangle=-25,
            margin=dict(l=40, r=40, t=50, b=80),
        )
        st.plotly_chart(fig_ratio, use_container_width=True)


# ==============================================================================
# TAB 4: FAULT DETECTION (ON ACTIVE DATASET)
# ==============================================================================
with tabs[3]:
    st.subheader("Streaming Topological Fault Detection")
    st.caption(f"Evaluating sensor faults on active dataset: **{dataset_name}** ({active_data.shape[0]:,} samples, {active_data.shape[1]} channels)")

    n_samples_total, n_channels = active_data.shape

    f_col1, f_col2, f_col3, f_col4 = st.columns(4)
    with f_col1:
        fault_kind = st.selectbox(
            "Fault Kind", ["drift", "stuck"], format_func=lambda x: x.capitalize()
        )
    with f_col2:
        fault_channel = st.selectbox("Target Channel", list(range(n_channels)), index=0)
    with f_col3:
        default_start = min(n_samples_total - 250, max(250, int(n_samples_total * 0.55)))
        fault_start = st.slider("Fault Onset (Tick)", 200, n_samples_total - 200, default_start, 20)
    with f_col4:
        sigma_k = st.slider("Threshold (k-sigma)", 1.5, 4.0, 2.2, 0.1)

    btn_fault = st.button("🚨 Inject Fault & Run Detection", type="primary")

    # Inject fault into active data
    faulty_stream = inject_fault(
        active_data,
        start=fault_start,
        kind=fault_kind,
        channel=fault_channel,
        drift_slope=0.015,
    )

    n_baseline_val = min(40, max(10, fault_start // 20))
    detector = BraidAnomalyDetector(
        window_size=min(200, fault_start),
        n_baseline=n_baseline_val,
        k=sigma_k,
        step=10,
        two_sided=True,
    )
    eval_res = detector.evaluate(faulty_stream, fault_start=fault_start)
    records = eval_res["records"]

    # Metrics Row
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric(
            "False Positive Rate",
            f"{eval_res['false_positive_rate']:.2%}",
            help="Fraction of clean windows flagged before fault onset",
        )
    with c2:
        delay = eval_res["detection_delay_ticks"]
        st.metric(
            "Detection Delay",
            f"{delay} ticks" if delay is not None else "Not Detected",
            help="Ticks from fault onset to window boundary where fault is recognized",
        )
    with c3:
        st.metric(
            "Fault Windows Flagged",
            f"{eval_res['fault_windows_flagged']} / {eval_res['fault_windows_total']}",
        )
    with c4:
        st.metric(
            "Baseline Reduced Len",
            f"{eval_res['baseline_mean']:.2f} ± {eval_res['baseline_std']:.2f}",
        )

    # Plot Signals
    fig_fault = go.Figure()
    for c in range(n_channels):
        is_target = c == fault_channel
        fig_fault.add_trace(
            go.Scatter(
                y=faulty_stream[:, c],
                name=f"Channel {c}" + (" (Faulty)" if is_target else ""),
                line=dict(
                    width=3 if is_target else 1.2,
                    color="#ff0055" if is_target else CHANNEL_COLORS[c % len(CHANNEL_COLORS)],
                    dash="solid" if is_target else "dot",
                ),
            )
        )
    fig_fault.add_vline(
        x=fault_start,
        line_width=2,
        line_dash="dash",
        line_color="#eab308",
        annotation_text="Fault Onset",
    )
    fig_fault.update_layout(
        title=f"Multichannel Signals ({fault_kind.capitalize()} Fault on Channel {fault_channel} at tick {fault_start})",
        template="plotly_dark",
        xaxis_title="Time (Ticks)",
        yaxis_title="Amplitude",
        height=340,
        margin=dict(l=40, r=40, t=40, b=30),
    )
    st.plotly_chart(fig_fault, use_container_width=True)

    # Plot Anomaly Scores
    win_starts = [r.window_start for r in records]
    red_lens = [r.reduced_length for r in records]
    flagged = [r.flagged for r in records]

    fig_scores = go.Figure()
    b_mean = eval_res["baseline_mean"]
    b_std = eval_res["baseline_std"]
    upper_bound = b_mean + sigma_k * b_std
    lower_bound = max(0, b_mean - sigma_k * b_std)

    fig_scores.add_hrect(
        y0=lower_bound,
        y1=upper_bound,
        fillcolor="#10b981",
        opacity=0.15,
        line_width=0,
        annotation_text=f"Normal Baseline (±{sigma_k:.1f}σ)",
    )
    fig_scores.add_hline(
        y=b_mean,
        line_dash="dot",
        line_color="#10b981",
        annotation_text="Baseline Mean",
    )

    fig_scores.add_trace(
        go.Scatter(
            x=win_starts,
            y=red_lens,
            mode="lines",
            name="Reduced Length",
            line=dict(color="#38bdf8", width=2),
        )
    )

    flagged_x = [win_starts[i] for i in range(len(win_starts)) if flagged[i]]
    flagged_y = [red_lens[i] for i in range(len(win_starts)) if flagged[i]]
    fig_scores.add_trace(
        go.Scatter(
            x=flagged_x,
            y=flagged_y,
            mode="markers",
            name="Flagged Anomaly",
            marker=dict(color="#ef4444", size=8, symbol="diamond"),
        )
    )

    fig_scores.add_vline(
        x=fault_start,
        line_width=2,
        line_dash="dash",
        line_color="#eab308",
        annotation_text="Fault Onset",
    )
    fig_scores.update_layout(
        title="Streaming Reduced Word Length per Window (Red Diamonds = Flagged Anomalies)",
        template="plotly_dark",
        xaxis_title="Window Start Tick",
        yaxis_title="Irreducible Reduced Word Length",
        height=340,
        margin=dict(l=40, r=40, t=40, b=30),
    )
    st.plotly_chart(fig_scores, use_container_width=True)
