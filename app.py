"""TangleCode: Topological Braid Group Sensor Encoding & Hackathon Demo.

Workflow:
1. Landing Page: File Uploading & Sample Dataset Selection.
2. Analysis Page: Interactive options for Live Braid, Compression Benchmark, and Fault Detection.
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
    page_title="TangleCode | Topological Braid Sensor Encoding",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed",
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
    .status-card {
        background-color: #1e293b;
        border-radius: 8px;
        padding: 14px 20px;
        border: 1px solid #334155;
        margin-bottom: 1.5rem;
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

# Strand Colors
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

# Initialize state
if "active_data" not in st.session_state:
    st.session_state["active_data"] = None
if "dataset_name" not in st.session_state:
    st.session_state["dataset_name"] = ""
if "page" not in st.session_state:
    st.session_state["page"] = "upload"


# ==============================================================================
# PAGE 1: FILE UPLOADING & SAMPLE CHOOSING (FIRST SCREEN)
# ==============================================================================
if st.session_state["page"] == "upload" or st.session_state["active_data"] is None:
    st.markdown("### 📁 Step 1: Upload Sensor Data or Choose a Sample Dataset")
    st.write(
        "Upload your own CSV time-series file or choose from curated sensor benchmarks. "
        "Once loaded, you can proceed to the **Live Braid**, **Compression Benchmark**, and **Fault Detection** options."
    )

    data_source_mode = st.radio(
        "Select Dataset Source",
        [
            "Upload Custom CSV File",
            "Choose a Curated Sample Dataset",
        ],
        horizontal=True,
    )

    loaded_array = None
    source_name = ""

    if data_source_mode == "Upload Custom CSV File":
        uploaded_file = st.file_uploader(
            "Upload CSV File (numeric columns will be extracted as channels)",
            type=["csv"],
            help="Select any CSV with 2 to 8 numeric sensor channels",
        )
        if uploaded_file is not None:
            try:
                temp_path = Path("data/_temp_upload.csv")
                temp_path.write_bytes(uploaded_file.getvalue())
                loaded_array = load_csv(temp_path)
                source_name = f"Uploaded: {uploaded_file.name}"
                if temp_path.exists():
                    temp_path.unlink()
            except Exception as e:
                st.error(f"Error parsing uploaded CSV: {e}")
        else:
            st.info("👆 Please upload a CSV file or switch to 'Choose a Curated Sample Dataset' above.")

    else:
        sample_choice = st.selectbox(
            "Select Sample Dataset",
            [
                "Real 6-Axis Gait Motion Sensor (data/real.csv)",
                "Synthetic 8-Channel IMU Stream (8 channels, 1500 samples)",
                "Synthetic 4-Channel Correlated Motion (4 channels, 1200 samples)",
            ],
        )

        if sample_choice == "Real 6-Axis Gait Motion Sensor (data/real.csv)":
            real_path = Path("data/real.csv")
            if real_path.exists():
                loaded_array = load_csv(real_path)
                source_name = "Real 6-Axis Motion Sensor (data/real.csv)"
            else:
                st.warning("data/real.csv not found on disk. Falling back to synthetic 8-channel IMU.")
                loaded_array = generate_synthetic(n_channels=8, n_samples=1000, noise_std=0.03, seed=42)
                source_name = "Synthetic 8-Channel IMU"
        elif sample_choice == "Synthetic 8-Channel IMU Stream (8 channels, 1500 samples)":
            loaded_array = generate_synthetic(n_channels=8, n_samples=1500, noise_std=0.03, seed=0)
            source_name = "Synthetic 8-Channel IMU Stream"
        else:
            loaded_array = generate_synthetic(n_channels=4, n_samples=1200, noise_std=0.02, seed=42)
            source_name = "Synthetic 4-Channel Correlated Motion"

    # Preview and Proceed Button
    if loaded_array is not None:
        n_rows, n_cols = loaded_array.shape
        if n_cols > 8:
            st.warning(f"Dataset has {n_cols} channels. The `.tngl` format supports up to 8 channels. Using first 8.")
            loaded_array = loaded_array[:, :8]
            n_cols = 8

        st.success(f"✅ Ready: **{source_name}** ({n_rows:,} samples, {n_cols} channels)")

        # Preview waveforms
        fig_prev = go.Figure()
        preview_samples = min(500, n_rows)
        for c in range(n_cols):
            fig_prev.add_trace(
                go.Scatter(
                    y=loaded_array[:preview_samples, c],
                    name=f"Channel {c}",
                    line=dict(width=1.5, color=CHANNEL_COLORS[c % len(CHANNEL_COLORS)]),
                )
            )
        fig_prev.update_layout(
            title=f"Dataset Preview ({source_name} — First {preview_samples} samples)",
            template="plotly_dark",
            xaxis_title="Time (Ticks)",
            yaxis_title="Sensor Amplitude",
            height=300,
            margin=dict(l=40, r=40, t=40, b=30),
        )
        st.plotly_chart(fig_prev, use_container_width=True)

        if st.button("🚀 Proceed to Analysis (Live Braid, Compression & Fault Detection) ➔", type="primary"):
            st.session_state["active_data"] = loaded_array
            st.session_state["dataset_name"] = source_name
            st.session_state["page"] = "analysis"
            st.rerun()


# ==============================================================================
# PAGE 2: ANALYSIS OPTIONS (LIVE BRAID, COMPRESSION BENCHMARK, FAULT DETECTION)
# ==============================================================================
else:
    active_data = st.session_state["active_data"]
    dataset_name = st.session_state["dataset_name"]
    n_samples_total, n_channels = active_data.shape

    # Status Bar with Change Dataset Button
    with st.container():
        st.markdown(
            f"""
            <div class="status-card">
                <span style="font-size: 1.15rem; font-weight: 700; color: #38bdf8;">📊 Active Dataset: {dataset_name}</span>
                <span style="color: #94a3b8; margin-left: 15px;">({n_samples_total:,} ticks, {n_channels} channels)</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Top Controls Bar: Change Dataset & Download Binaries
    top_c1, top_c2, top_c3 = st.columns([2, 1.5, 1.5])
    with top_c1:
        if st.button("📂 Upload / Select a Different Dataset"):
            st.session_state["page"] = "upload"
            st.rerun()
    with top_c2:
        tngl_rank_bytes = encode(active_data, mode=MODE_RANK_ONLY)
        st.download_button(
            label="📥 Export Rank-Only .tngl",
            data=tngl_rank_bytes,
            file_name="stream_rank_only.tngl",
            mime="application/octet-stream",
            help="Download compact .tngl binary preserving exact channel rank trajectories",
        )
    with top_c3:
        tngl_hyb_bytes = encode(active_data, mode=MODE_HYBRID, step_size=0.02)
        st.download_button(
            label="📥 Export Hybrid .tngl",
            data=tngl_hyb_bytes,
            file_name="stream_hybrid.tngl",
            mime="application/octet-stream",
            help="Download .tngl binary with int8 delta modulation preserving continuous amplitudes",
        )

    st.write("")

    # Main Analysis Options
    analysis_tabs = st.tabs([
        "🧬 Live Braid",
        "📊 Compression Benchmark",
        "🚨 Fault Detection",
    ])

    # --------------------------------------------------------------------------
    # OPTION 1: LIVE BRAID
    # --------------------------------------------------------------------------
    with analysis_tabs[0]:
        st.subheader("Live Braid Weaver & Reidemeister Jitter Annihilation")
        st.write(
            "Visualize how the channels of your active dataset weave across each other. "
            "Toggle **Reidemeister reduction** to watch transient jitter crossings cancel out, leaving invariant structural dynamics."
        )

        col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([1.5, 1.5, 1.5])
        with col_ctrl1:
            n_ticks = st.slider("Display Window (ticks)", 20, min(150, n_samples_total), min(60, n_samples_total), 5)
        with col_ctrl2:
            max_start = max(0, n_samples_total - n_ticks)
            tick_offset = st.slider("Window Start Offset", 0, max_start, 0, 10 if max_start > 10 else 1)
        with col_ctrl3:
            extra_noise = st.slider("Add Extra Jitter Noise (std)", 0.00, 0.15, 0.00, 0.01)
            apply_reduction = st.toggle("Apply Reidemeister reduction", value=False)

        sub_stream = active_data[tick_offset : tick_offset + n_ticks].copy()
        if extra_noise > 0:
            rng = np.random.default_rng(tick_offset)
            sub_stream += rng.normal(0.0, extra_noise, size=sub_stream.shape).astype(np.float32)

        # Raw ranks & crossings
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

        # Reduced ranks & crossings
        encoded_rank = encode(sub_stream, mode=MODE_RANK_ONLY)
        recon_rank = decode(encoded_rank)

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

        # Strand Plot
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
            f"Irreducible Topological Braid ({dataset_name} — Jitter Filtered via Reidemeister II Moves)"
            if apply_reduction
            else f"Raw Multi-Channel Braid Weave ({dataset_name} — Raw Crossings Marked)"
        )
        fig_braid.update_layout(
            title=title_text,
            template="plotly_dark",
            xaxis_title=f"Relative Window Ticks (Offset {tick_offset})",
            yaxis_title=f"Strand Rank Position (0 to {n_channels - 1})",
            yaxis=dict(autorange="reversed", dtick=1),
            height=500,
            margin=dict(l=40, r=40, t=50, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_braid, use_container_width=True)

    # --------------------------------------------------------------------------
    # OPTION 2: COMPRESSION BENCHMARK
    # --------------------------------------------------------------------------
    with analysis_tabs[1]:
        st.subheader("Compression & Topological Reduction Benchmark")
        st.write(
            f"Benchmark standard compressors against TangleCode on the active dataset: **{dataset_name}**."
        )

        st.info(
            "ℹ️ **Encoding Modes**: **Rank-Only Mode** reconstructs exact channel rank order per tick "
            "(lossy amplitude, ultra-high compression up to 93x). **Hybrid Mode** combines braid topology "
            "with int8 closed-loop delta modulation to reconstruct continuous amplitudes with minimal error (RMSE < 0.01)."
        )

        btn_run_bench = st.button("🚀 Re-Run Live Benchmark on Active Dataset", type="primary")

        bench_cache_key = f"bench_{dataset_name}_{active_data.shape}"
        if btn_run_bench or bench_cache_key not in st.session_state:
            with st.spinner(f"Benchmarking compression on {dataset_name}..."):
                st.session_state[bench_cache_key] = benchmark_dataset(active_data, dataset_name)

        bench_results = st.session_state[bench_cache_key]
        methods_dict = bench_results["methods"]

        # Table
        table_rows = []
        for method_name, metrics in methods_dict.items():
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

        # Charts
        col_c1, col_c2 = st.columns(2)
        names = list(methods_dict.keys())
        sizes = [m["compressed_size_bytes"] for m in methods_dict.values()]
        ratios = [m["compression_ratio"] for m in methods_dict.values()]
        colors = ["#475569", "#3b82f6", "#06b6d4", "#f59e0b", "#10b981", "#8b5cf6"]

        with col_c1:
            fig_sizes = go.Figure()
            fig_sizes.add_trace(
                go.Bar(
                    x=names,
                    y=sizes,
                    marker=dict(color=colors[: len(names)]),
                    text=[f"{s:,} B" for s in sizes],
                    textposition="auto",
                )
            )
            fig_sizes.update_layout(
                title="Compressed Size (Bytes) — Lower is Better",
                template="plotly_dark",
                yaxis_title="Size (Bytes, Log Scale)",
                yaxis_type="log",
                height=400,
                xaxis_tickangle=-25,
                margin=dict(l=40, r=40, t=50, b=80),
            )
            st.plotly_chart(fig_sizes, use_container_width=True)

        with col_c2:
            fig_rat = go.Figure()
            fig_rat.add_trace(
                go.Bar(
                    x=names,
                    y=ratios,
                    marker=dict(color=colors[: len(names)]),
                    text=[f"{r:.1f}x" for r in ratios],
                    textposition="auto",
                )
            )
            fig_rat.update_layout(
                title="Compression Ratio vs Raw CSV — Higher is Better",
                template="plotly_dark",
                yaxis_title="Ratio (Multiplier)",
                height=400,
                xaxis_tickangle=-25,
                margin=dict(l=40, r=40, t=50, b=80),
            )
            st.plotly_chart(fig_rat, use_container_width=True)

    # --------------------------------------------------------------------------
    # OPTION 3: FAULT DETECTION
    # --------------------------------------------------------------------------
    with analysis_tabs[2]:
        st.subheader("Streaming Topological Fault Detection")
        st.write(
            f"Inject simulated sensor drift or stuck-sensor faults into **{dataset_name}** and observe real-time "
            "topological detection via sliding window braid reductions."
        )

        fc1, fc2, fc3, fc4 = st.columns(4)
        with fc1:
            fault_type = st.selectbox(
                "Fault Type", ["drift", "stuck"], format_func=lambda x: x.capitalize()
            )
        with fc2:
            target_ch = st.selectbox("Target Channel", list(range(n_channels)), index=0)
        with fc3:
            default_onset = min(n_samples_total - 200, max(200, int(n_samples_total * 0.55)))
            onset_tick = st.slider("Fault Onset (Tick)", 100, n_samples_total - 100, default_onset, 20)
        with fc4:
            k_sigma = st.slider("Threshold (k-sigma)", 1.5, 4.0, 2.2, 0.1)

        btn_detect = st.button("🚨 Inject Fault & Run Anomaly Detection", type="primary")

        faulty_signal = inject_fault(
            active_data,
            start=onset_tick,
            kind=fault_type,
            channel=target_ch,
            drift_slope=0.015,
        )

        n_baseline_count = min(40, max(10, onset_tick // 20))
        fault_detector = BraidAnomalyDetector(
            window_size=min(200, onset_tick),
            n_baseline=n_baseline_count,
            k=k_sigma,
            step=10,
            two_sided=True,
        )
        detector_eval = fault_detector.evaluate(faulty_signal, fault_start=onset_tick)
        recs = detector_eval["records"]

        # Metrics
        f_m1, f_m2, f_m3, f_m4 = st.columns(4)
        with f_m1:
            st.metric("False Positive Rate", f"{detector_eval['false_positive_rate']:.2%}")
        with f_m2:
            del_ticks = detector_eval["detection_delay_ticks"]
            st.metric("Detection Delay", f"{del_ticks} ticks" if del_ticks is not None else "Not Detected")
        with f_m3:
            st.metric("Fault Windows Flagged", f"{detector_eval['fault_windows_flagged']} / {detector_eval['fault_windows_total']}")
        with f_m4:
            st.metric("Baseline Reduced Len", f"{detector_eval['baseline_mean']:.2f} ± {detector_eval['baseline_std']:.2f}")

        # Signal Plot
        fig_fault_sig = go.Figure()
        for c in range(n_channels):
            is_fault = c == target_ch
            fig_fault_sig.add_trace(
                go.Scatter(
                    y=faulty_signal[:, c],
                    name=f"Channel {c}" + (" (Faulty)" if is_fault else ""),
                    line=dict(
                        width=3 if is_fault else 1.2,
                        color="#ff0055" if is_fault else CHANNEL_COLORS[c % len(CHANNEL_COLORS)],
                        dash="solid" if is_fault else "dot",
                    ),
                )
            )
        fig_fault_sig.add_vline(
            x=onset_tick,
            line_width=2,
            line_dash="dash",
            line_color="#eab308",
            annotation_text="Fault Onset",
        )
        fig_fault_sig.update_layout(
            title=f"Multichannel Waveforms ({fault_type.capitalize()} Fault Injected on Channel {target_ch})",
            template="plotly_dark",
            xaxis_title="Time (Ticks)",
            yaxis_title="Amplitude",
            height=320,
            margin=dict(l=40, r=40, t=40, b=30),
        )
        st.plotly_chart(fig_fault_sig, use_container_width=True)

        # Score Plot
        w_starts = [r.window_start for r in recs]
        w_reds = [r.reduced_length for r in recs]
        w_flags = [r.flagged for r in recs]

        fig_det = go.Figure()
        mean_val = detector_eval["baseline_mean"]
        std_val = detector_eval["baseline_std"]
        u_bound = mean_val + k_sigma * std_val
        l_bound = max(0, mean_val - k_sigma * std_val)

        fig_det.add_hrect(
            y0=l_bound,
            y1=u_bound,
            fillcolor="#10b981",
            opacity=0.15,
            line_width=0,
            annotation_text=f"Normal Baseline (±{k_sigma:.1f}σ)",
        )
        fig_det.add_hline(y=mean_val, line_dash="dot", line_color="#10b981", annotation_text="Baseline Mean")

        fig_det.add_trace(
            go.Scatter(
                x=w_starts,
                y=w_reds,
                mode="lines",
                name="Reduced Length",
                line=dict(color="#38bdf8", width=2),
            )
        )

        fl_x = [w_starts[i] for i in range(len(w_starts)) if w_flags[i]]
        fl_y = [w_reds[i] for i in range(len(w_starts)) if w_flags[i]]
        fig_det.add_trace(
            go.Scatter(
                x=fl_x,
                y=fl_y,
                mode="markers",
                name="Flagged Anomaly",
                marker=dict(color="#ef4444", size=8, symbol="diamond"),
            )
        )

        fig_det.add_vline(
            x=onset_tick,
            line_width=2,
            line_dash="dash",
            line_color="#eab308",
            annotation_text="Fault Onset",
        )
        fig_det.update_layout(
            title="Sliding Window Reduced Word Length (Red Diamonds = Flagged Anomalies)",
            template="plotly_dark",
            xaxis_title="Window Start Tick",
            yaxis_title="Irreducible Reduced Word Length",
            height=320,
            margin=dict(l=40, r=40, t=40, b=30),
        )
        st.plotly_chart(fig_det, use_container_width=True)
