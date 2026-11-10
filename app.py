"""TangleCode: Two-page Streamlit application for topological braid sensor encoding."""

import io
import struct
import tempfile
from pathlib import Path
from typing import List, Tuple

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from tanglecode.anomaly import BraidAnomalyDetector
from tanglecode.braid import extract_braid_word, rank_vector
from tanglecode.codec import MODE_RANK_ONLY, encode
from tanglecode.data import generate_synthetic, inject_fault, load_csv

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
                data = inject_fault(clean_stream, start=1500, kind="drift", channel=0, drift_slope=0.015)
            elif sample_choice == "IMU with stuck sensor":
                data = inject_fault(clean_stream, start=1500, kind="stuck", channel=1)

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
            raw_csv_size = len(csv_buf.getvalue().encode("utf-8"))
            tngl_size = len(tngl_bytes)
            compression_ratio = raw_csv_size / tngl_size if tngl_size > 0 else 1.0

            # 4. Anomaly detection via sliding window braid reductions
            window_size = min(200, max(20, n_samples // 6))
            step = max(5, window_size // 15)
            n_baseline = min(35, max(5, (n_samples // 2 - window_size) // step))

            detector = BraidAnomalyDetector(
                window_size=window_size,
                n_baseline=n_baseline,
                k=2.5,
                step=step,
            )
            records = detector.predict(data)

            # Store results in session_state and transition to Page 2
            st.session_state["results"] = {
                "dataset_name": dataset_name,
                "n_channels": n_channels,
                "n_samples": n_samples,
                "raw_crossings": total_raw_crossings,
                "reduced_crossings": total_reduced_crossings,
                "raw_csv_size": raw_csv_size,
                "tngl_size": tngl_size,
                "compression_ratio": compression_ratio,
                "tngl_bytes": tngl_bytes,
                "window_size": window_size,
                "step": step,
                "baseline_mean": detector.baseline_mean,
                "baseline_std": detector.baseline_std,
                "k": detector.k,
                "records": records,
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

    st.write("")

    # Download button for the .tngl file
    download_filename = (
        Path(res["dataset_name"]).stem + ".tngl"
        if res["dataset_name"].endswith(".csv")
        else "sensor_data.tngl"
    )
    st.download_button(
        label="Download .tngl file",
        data=res["tngl_bytes"],
        file_name=download_filename,
        mime="application/octet-stream",
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
