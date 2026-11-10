"""TangleCode: Two-page Streamlit application for topological braid sensor encoding."""

import io
import struct
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import plotly.graph_objects as go
import streamlit as st

# Ensure fresh tanglecode modules in persistent/watching Streamlit processes
for mod_name in list(sys.modules.keys()):
    if mod_name == "tanglecode" or mod_name.startswith("tanglecode."):
        del sys.modules[mod_name]

from tanglecode.anomaly import (
    BraidAnomalyDetector,
    diagnose_window_fault,
    learn_channel_baseline,
)
from tanglecode.braid import (
    extract_braid_word,
    extract_crossings_with_channels,
    rank_vector,
    reduce_with_channels,
)
from tanglecode.codec import MODE_RANK_ONLY, encode
from tanglecode.data import generate_synthetic, inject_fault, load_csv
from tanglecode.reports import anomaly_report_csv, summary_json

# Page configuration
st.set_page_config(
    page_title="TangleCode",
    page_icon="🧬",
    layout="wide",
)

# Initialize session state for navigation
if "page" not in st.session_state:
    st.session_state["page"] = "input"
if "results" not in st.session_state:
    st.session_state["results"] = None


def merge_flagged_intervals(flagged_starts: List[int], window_size: int) -> List[Tuple[int, int]]:
    """Merge overlapping or adjacent flagged window spans into contiguous intervals."""
    if not flagged_starts:
        return []
    sorted_starts = sorted(flagged_starts)
    intervals: List[Tuple[int, int]] = []
    curr_start = sorted_starts[0]
    curr_end = curr_start + window_size

    for start in sorted_starts[1:]:
        end = start + window_size
        if start <= curr_end:
            curr_end = max(curr_end, end)
        else:
            intervals.append((curr_start, curr_end))
            curr_start = start
            curr_end = end
    intervals.append((curr_start, curr_end))
    return intervals


# =============================================================================
# PAGE 1: INPUT
# =============================================================================
if st.session_state["page"] == "input":
    st.title("TangleCode")
    st.write("Topological braid group encoding and streaming anomaly detection for multichannel sensor streams.")

    source_option = st.radio(
        "Data source",
        options=["Upload CSV", "Use a sample"],
        horizontal=True,
    )

    uploaded_file = None
    sample_choice = None

    if source_option == "Upload CSV":
        uploaded_file = st.file_uploader(
            "Upload CSV (each numeric column is one sensor channel)",
            type=["csv"],
            help="Upload a CSV file containing at least 2 numeric columns.",
        )
    else:
        sample_choice = st.selectbox(
            "Choose a sample dataset:",
            options=[
                "Clean drone IMU",
                "IMU with drifting sensor",
                "IMU with stuck sensor",
            ],
        )

    if st.button("Analyze", type="primary"):
        data = None
        dataset_name = ""

        if source_option == "Upload CSV":
            if uploaded_file is None:
                st.error("Please upload a CSV file before analyzing.")
            else:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = Path(tmp.name)
                try:
                    loaded = load_csv(tmp_path)
                    if loaded.ndim != 2 or loaded.shape[1] < 2:
                        st.error("Uploaded CSV must have at least 2 numeric columns.")
                    elif loaded.shape[0] < 10:
                        st.error("Uploaded CSV must contain at least 10 sample rows.")
                    else:
                        if loaded.shape[1] > 8:
                            st.warning("Data has more than 8 channels. Using the first 8 channels supported by .tngl.")
                            loaded = loaded[:, :8]
                        data = loaded
                        dataset_name = uploaded_file.name
                except Exception as e:
                    st.error(f"Error loading CSV: {e}")
                finally:
                    if tmp_path.exists():
                        tmp_path.unlink()
        else:
            dataset_name = sample_choice
            n_samples = 3000
            n_channels = 6
            clean_stream = generate_synthetic(
                n_channels=n_channels,
                n_samples=n_samples,
                noise_std=0.03,
                seed=42,
            )
            if sample_choice == "Clean drone IMU":
                data = clean_stream
            elif sample_choice == "IMU with drifting sensor":
                data = inject_fault(clean_stream, start=1500, kind="drift", channel=3, drift_slope=0.02)
            elif sample_choice == "IMU with stuck sensor":
                data = inject_fault(clean_stream, start=1500, kind="stuck", channel=3)

        if data is not None:
            n_samples, n_channels = data.shape

            # 1. Rank extraction and raw braid crossings
            total_raw_crossings = 0
            for t in range(n_samples - 1):
                w = extract_braid_word(rank_vector(data[t]), rank_vector(data[t + 1]))
                total_raw_crossings += len(w)

            # 2. Encode to rank-only .tngl binary format
            tngl_bytes = encode(data, mode=MODE_RANK_ONLY)
            _, _, _, total_reduced_crossings, _ = struct.unpack_from("<4sBIIB", tngl_bytes, 0)

            # 3. Compute compression ratio vs raw CSV representation
            csv_buf = io.StringIO()
            csv_buf.write(",".join(f"Ch_{c}" for c in range(n_channels)) + "\n")
            for row in data:
                csv_buf.write(",".join(f"{v:.6f}" for v in row) + "\n")
            raw_csv_str = csv_buf.getvalue()
            raw_csv_bytes = raw_csv_str.encode("utf-8")
            raw_csv_size = len(raw_csv_bytes)
            tngl_size = len(tngl_bytes)
            compression_ratio = raw_csv_size / tngl_size if tngl_size > 0 else 1.0

            # 4. Anomaly detection via sliding window braid reductions
            window_size = min(200, max(20, n_samples // 6))
            step = max(5, window_size // 15)
            n_baseline = min(40, max(5, (n_samples // 2 - window_size) // step))

            detector = BraidAnomalyDetector(
                window_size=window_size,
                n_baseline=n_baseline,
                k=3.5,
                step=step,
                two_sided=True,
            )
            records = detector.predict(data)
            flagged_records = [r for r in records if r.flagged]

            # 5. Per-channel fault diagnosis
            ch_mean, ch_std = learn_channel_baseline(
                data,
                window_size=window_size,
                n_baseline=n_baseline,
                step=step,
            )

            threshold_val = float(
                detector.baseline_mean + detector.k * detector.baseline_std
                if detector.baseline_mean is not None and detector.baseline_std is not None
                else 0.0
            )

            flagged_diagnoses: List[Dict[str, Any]] = []
            for r in flagged_records:
                w = data[r.window_start : r.window_start + window_size]
                diag = diagnose_window_fault(
                    w,
                    baseline_channel_mean=ch_mean,
                    baseline_channel_std=ch_std,
                    window_start=r.window_start,
                )
                flagged_diagnoses.append({
                    "window_start_tick": r.window_start,
                    "window_end_tick": r.window_start + window_size,
                    "score": float(r.score),
                    "threshold": threshold_val,
                    "most_likely_channel": diag.most_likely_channel,
                    "top_crossing_pair": diag.top_crossing_pair,
                })

            # Identify overall most likely faulty sensor across flagged windows
            if flagged_diagnoses:
                faulty_channel = Counter(
                    [d["most_likely_channel"] for d in flagged_diagnoses]
                ).most_common(1)[0][0]

                # Identify the partner channel that crossed most with the faulty sensor
                partner_counts: Counter[int] = Counter()
                for d in flagged_diagnoses:
                    pair = d["top_crossing_pair"]
                    if pair is not None and faulty_channel in pair:
                        other = pair[0] if pair[1] == faulty_channel else pair[1]
                        partner_counts[other] += 1
                crossed_with_channel = (
                    partner_counts.most_common(1)[0][0]
                    if partner_counts
                    else ((faulty_channel + 1) % n_channels)
                )

                anomaly_tick_start = min(d["window_start_tick"] for d in flagged_diagnoses)
                anomaly_tick_end = max(d["window_end_tick"] for d in flagged_diagnoses)
            else:
                faulty_channel = None
                crossed_with_channel = None
                anomaly_tick_start = None
                anomaly_tick_end = None

            # 6. Surviving crossings by sensor pair across the whole dataset
            whole_raw_crossings: list[tuple[int, int, int]] = []
            for t in range(n_samples - 1):
                whole_raw_crossings.extend(
                    extract_crossings_with_channels(rank_vector(data[t]), rank_vector(data[t + 1]))
                )
            whole_reduced_crossings, _ = reduce_with_channels(whole_raw_crossings)

            crossing_matrix = np.zeros((n_channels, n_channels), dtype=np.float64)
            for _, a, b in whole_reduced_crossings:
                crossing_matrix[a, b] += 1
                crossing_matrix[b, a] += 1

            # Blank diagonal
            for i in range(n_channels):
                crossing_matrix[i, i] = np.nan

            # Store results in session_state and transition to Page 2
            st.session_state["results"] = {
                "dataset_name": dataset_name,
                "n_channels": n_channels,
                "n_samples": n_samples,
                "raw_crossings": total_raw_crossings,
                "reduced_crossings": total_reduced_crossings,
                "raw_csv_size": raw_csv_size,
                "raw_csv_bytes": raw_csv_bytes,
                "tngl_size": tngl_size,
                "compression_ratio": compression_ratio,
                "tngl_bytes": tngl_bytes,
                "window_size": window_size,
                "step": step,
                "baseline_mean": detector.baseline_mean,
                "baseline_std": detector.baseline_std,
                "k": detector.k,
                "records": records,
                "flagged_diagnoses": flagged_diagnoses,
                "faulty_channel": faulty_channel,
                "crossed_with_channel": crossed_with_channel,
                "anomaly_tick_start": anomaly_tick_start,
                "anomaly_tick_end": anomaly_tick_end,
                "crossing_matrix": crossing_matrix,
            }
            st.session_state["page"] = "results"
            st.rerun()


# =============================================================================
# PAGE 2: RESULTS
# =============================================================================
elif st.session_state["page"] == "results":
    res = st.session_state.get("results")
    if res is None:
        st.session_state["page"] = "input"
        st.rerun()

    # Dataset title and metadata
    st.title("TangleCode")
    st.subheader(f"Dataset: {res['dataset_name']}")
    st.caption(f"Channels: **{res['n_channels']}** | Samples: **{res['n_samples']:,}**")

    # Three headline metric cards
    col1, col2, col3 = st.columns(3)
    raw_cnt = res["raw_crossings"]
    red_cnt = res["reduced_crossings"]
    elim_pct = ((raw_cnt - red_cnt) / raw_cnt * 100.0) if raw_cnt > 0 else 0.0

    col1.metric("Raw Crossings", f"{raw_cnt:,}")
    col2.metric(
        "Reduced Crossings",
        f"{red_cnt:,}",
        delta=f"-{elim_pct:.1f}% jitter" if raw_cnt > 0 else None,
    )
    col3.metric(
        "Compression Ratio",
        f"{res['compression_ratio']:.1f}x",
        delta=f"{res['raw_csv_size']:,} B → {res['tngl_size']:,} B",
    )

    st.write("")

    # Plotly Chart: reduced braid-word length per window with flagged anomaly windows shaded red
    records = res["records"]
    window_starts = [r.window_start for r in records]
    reduced_lengths = [r.reduced_length for r in records]
    flagged_starts = [r.window_start for r in records if r.flagged]
    flagged_lengths = [r.reduced_length for r in records if r.flagged]

    fig = go.Figure()

    # Shade flagged anomaly window spans in red
    flagged_intervals = merge_flagged_intervals(flagged_starts, res["window_size"])
    for idx, (x0, x1) in enumerate(flagged_intervals):
        fig.add_vrect(
            x0=x0,
            x1=x1,
            fillcolor="rgba(239, 68, 68, 0.25)",
            line_width=1,
            line_color="rgba(239, 68, 68, 0.5)",
            layer="below",
            annotation_text="Anomaly" if idx == 0 else None,
            annotation_position="top left",
        )

    # Line trace for reduced braid-word length
    fig.add_trace(
        go.Scatter(
            x=window_starts,
            y=reduced_lengths,
            mode="lines",
            name="Reduced Braid-Word Length",
            line=dict(color="#0284c7", width=2),
            hovertemplate="Tick %{x}<br>Reduced Length: %{y}<extra></extra>",
        )
    )

    # Flagged anomaly points
    if flagged_starts:
        fig.add_trace(
            go.Scatter(
                x=flagged_starts,
                y=flagged_lengths,
                mode="markers",
                name="Flagged Anomaly Windows",
                marker=dict(color="#dc2626", size=8, symbol="diamond"),
                hovertemplate="Flagged Tick %{x}<br>Length: %{y}<extra></extra>",
            )
        )

    # Baseline threshold line if available
    if res["baseline_mean"] is not None and res["baseline_std"] is not None:
        thresh = res["baseline_mean"] + res["k"] * res["baseline_std"]
        fig.add_hline(
            y=thresh,
            line=dict(color="#f59e0b", dash="dash", width=1.5),
            annotation_text=f"Threshold ({thresh:.1f})",
            annotation_position="top right",
        )

    fig.update_layout(
        template="plotly_white",
        title="Reduced Braid-Word Length per Sliding Window",
        xaxis_title="Time Tick (Window Start)",
        yaxis_title="Irreducible Reduced-Word Length",
        height=380,
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
    )

    st.plotly_chart(fig, use_container_width=True)

    # Plain caption under chart stating whether anomalies were detected and in which time range
    if flagged_intervals:
        range_strs = [f"ticks {x0}–{x1}" for x0, x1 in flagged_intervals]
        caption_text = f"Anomalies detected: Yes, in time range {', '.join(range_strs)}."
    else:
        caption_text = "Anomalies detected: None. Sensor signals remained within normal baseline boundaries."
    st.caption(caption_text)

    # Plain sentence stating most likely faulty sensor
    flagged_diagnoses = res.get("flagged_diagnoses", [])
    if flagged_diagnoses and res.get("faulty_channel") is not None:
        faulty_label = f"Channel {res['faulty_channel'] + 1}"
        other_label = (
            f"Channel {res['crossed_with_channel'] + 1}"
            if res.get("crossed_with_channel") is not None
            else "neighbor"
        )
        t_a = res.get("anomaly_tick_start", 0)
        t_b = res.get("anomaly_tick_end", 0)
        st.write(
            f"Most likely faulty sensor: {faulty_label} (crossed most with {other_label}), "
            f"around ticks {t_a} to {t_b}."
        )
    else:
        st.write("No faulty sensor identified.")

    st.write("")

    # Plotly Heatmap: Surviving crossings by sensor pair across the whole dataset
    n_ch = res["n_channels"]
    channel_labels = [f"Channel {i + 1}" for i in range(n_ch)]
    fig_heat = go.Figure(
        data=go.Heatmap(
            z=res["crossing_matrix"],
            x=channel_labels,
            y=channel_labels,
            colorscale="Blues",
            hoverongaps=False,
            colorbar=dict(title="Crossings"),
        )
    )
    fig_heat.update_layout(
        title="Surviving crossings by sensor pair",
        template="plotly_white",
        height=400,
        margin=dict(l=40, r=40, t=50, b=40),
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    st.write("")

    # Export utilities for the 3 download buttons
    base_name = (
        Path(res["dataset_name"]).stem
        if res["dataset_name"].endswith(".csv")
        else res["dataset_name"].lower().replace(" ", "_")
    )
    tngl_filename = f"{base_name}.tngl"
    report_csv_filename = f"{base_name}_anomaly_report.csv"
    summary_json_filename = f"{base_name}_summary.json"

    report_results_payload = {
        "dataset_name": res["dataset_name"],
        "n_channels": res["n_channels"],
        "n_samples": res["n_samples"],
        "raw_crossings": res["raw_crossings"],
        "reduced_crossings": res["reduced_crossings"],
        "raw_csv_size": res["raw_csv_size"],
        "raw_csv_bytes": res.get("raw_csv_bytes", None),
        "tngl_size": res["tngl_size"],
        "flagged_diagnoses": flagged_diagnoses,
        "flagged_intervals": flagged_intervals,
        "faulty_channel": res.get("faulty_channel", None),
    }

    report_csv_text = anomaly_report_csv(report_results_payload)
    summary_json_text = summary_json(report_results_payload)

    # Three download buttons in columns
    btn_col1, btn_col2, btn_col3 = st.columns(3)

    with btn_col1:
        st.download_button(
            label="Download .tngl file",
            data=res["tngl_bytes"],
            file_name=tngl_filename,
            mime="application/octet-stream",
        )

    with btn_col2:
        st.download_button(
            label="Anomaly report (CSV)",
            data=report_csv_text,
            file_name=report_csv_filename,
            mime="text/csv",
        )

    with btn_col3:
        st.download_button(
            label="Summary (JSON)",
            data=summary_json_text,
            file_name=summary_json_filename,
            mime="application/json",
        )

    # Small note
    st.caption(
        "Rank-only mode keeps sensor ordering and discards exact amplitudes. "
        "Built for structure and fault monitoring, not full recovery."
    )

    st.write("")

    # Back button to clear results and return to Page 1
    if st.button("Back"):
        st.session_state["results"] = None
        st.session_state["page"] = "input"
        st.rerun()
