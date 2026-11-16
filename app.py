from __future__ import annotations

import io
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from tanglecode.anomaly import detect
from tanglecode.braid import (
    reduce_crossings,
    replay_orders,
    word_from_frames,
)
from tanglecode.codec import encode
from tanglecode.data import generate_signals, load_csv

def format_ordinal(val: int) -> str:
    n = int(val)
    if 11 <= (n % 100) <= 13:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


# Page configuration
st.set_page_config(
    page_title="TangleCode",
    page_icon="🧬",
    layout="centered",
)

# Navigation state
if "page" not in st.session_state:
    st.session_state["page"] = "input"
if "results" not in st.session_state:
    st.session_state["results"] = None


# =============================================================================
# PAGE 1: Choose data
# =============================================================================
if st.session_state["page"] == "input":
    st.title("TangleCode")
    st.write("Shrinks multi-sensor logs and warns you when one sensor starts misbehaving.")
    st.write("")

    sample_option = st.selectbox(
        "Choose a sample",
        options=["Healthy sensors", "One sensor drifting", "One sensor stuck"],
    )

    st.write("")
    uploaded_file = st.file_uploader(
        "Or upload your own CSV",
        type=["csv"],
        help="Upload a CSV with at least 2 numeric sensor columns and 300 rows.",
    )

    st.write("")
    sampling_rate_input = st.number_input(
        "Sampling rate (Hz), optional",
        min_value=0.0,
        value=None,
        step=1.0,
        help="Optional sampling rate to convert ticks to seconds.",
    )

    st.write("")
    if st.button("Analyze", type="primary", use_container_width=True):
        data = None
        raw_csv_bytes = None
        dataset_name = ""
        warning_msg = None

        if uploaded_file is not None:
            dataset_name = uploaded_file.name
            raw_csv_bytes = uploaded_file.getvalue()
            try:
                data, warning_msg = load_csv(io.BytesIO(raw_csv_bytes))
            except Exception as e:
                st.error(f"Could not load CSV: {e}")
                data = None
        else:
            dataset_name = sample_option
            kind_map = {
                "Healthy sensors": "clean",
                "One sensor drifting": "drift",
                "One sensor stuck": "stuck",
            }
            kind = kind_map[sample_option]
            data = generate_signals(kind, n=6, T=3000, seed=7)
            cols = [f"sensor_{i}" for i in range(1, 7)]
            df = pd.DataFrame(data, columns=cols).round(4)
            raw_csv_bytes = df.to_csv(index=False).encode("utf-8")

        if data is not None:
            if warning_msg:
                st.warning(warning_msg)

            T, n = data.shape
            rate = float(sampling_rate_input) if (sampling_rate_input is not None and sampling_rate_input > 0) else None
            initial_order, raw = word_from_frames(data)
            reduced, cancelled = reduce_crossings(raw)
            positions = replay_orders(initial_order, raw, T)

            detection = detect(
                reduced_crossings=reduced,
                positions=positions,
                n=n,
                T=T,
                window=200,
                step=100,
                k=4.0,
                min_consecutive=4,
            )

            tngl_bytes, stats = encode(data)

            st.session_state["results"] = {
                "dataset_name": dataset_name,
                "n_sensors": n,
                "sampling_rate": rate,
                "raw_count": len(raw),
                "reduced_count": len(reduced),
                "raw_csv_bytes": raw_csv_bytes,
                "tngl_bytes": tngl_bytes,
                "detection": detection,
            }
            st.session_state["page"] = "results"
            st.rerun()


# =============================================================================
# PAGE 2: Results
# =============================================================================
elif st.session_state["page"] == "results":
    res = st.session_state["results"]
    detection = res["detection"]
    ranges = detection["ranges"]
    raw_count = res["raw_count"]
    reduced_count = res["reduced_count"]
    raw_csv_bytes = res["raw_csv_bytes"]
    tngl_bytes = res["tngl_bytes"]
    dataset_name = res["dataset_name"]
    rate = res.get("sampling_rate")
    n_sensors = res.get("n_sensors", 6)

    # 1. Verdict banner
    if len(ranges) == 0:
        st.success("All sensors normal")
    else:
        first_fault = ranges[0]
        if rate:
            start_s = first_fault["start_tick"] / rate
            end_s = first_fault["end_tick"] / rate
            st.error(
                f"Fault detected: Sensor {first_fault['faulty_channel']}, "
                f"around {start_s:.1f}s to {end_s:.1f}s"
            )
        else:
            st.error(
                f"Fault detected: Sensor {first_fault['faulty_channel']}, "
                f"around ticks {first_fault['start_tick']} to {first_fault['end_tick']}"
            )

    # Fault description directly under verdict banner
    pct_removed = ((raw_count - reduced_count) / raw_count * 100.0) if raw_count > 0 else 0.0
    if len(ranges) == 0:
        desc = f"All {n_sensors} sensors kept a steady order. {pct_removed:.1f}% of order swaps were noise and were removed."
    else:
        first_fault = ranges[0]
        ch = first_fault["faulty_channel"]
        kind = first_fault.get("kind", "drift")
        severity = first_fault.get("severity", "Moderate")
        if kind == "drift":
            nb = format_ordinal(first_fault.get("rank_before", 1))
            na = format_ordinal(first_fault.get("rank_after", 1))
            desc = (
                f"Sensor {ch} moved from {nb} to {na} place among the sensors and stayed there. "
                f"It looks like it is drifting away from the group. "
                f"Severity: {severity}"
            )
        else:
            desc = (
                f"Sensor {ch} started swapping places with the other sensors far more often than normal. "
                f"It looks erratic. "
                f"Severity: {severity}"
            )

    st.write(desc)
    st.write("")

    # 2. Three st.metric cards
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            label="Order swaps found",
            value=f"{raw_count:,}",
        )

    with col2:
        st.metric(
            label="After noise removal",
            value=f"{reduced_count:,}",
            delta=f"-{pct_removed:.1f}%",
        )

    with col3:
        csv_kb = len(raw_csv_bytes) / 1024.0
        tngl_kb = len(tngl_bytes) / 1024.0
        ratio = len(raw_csv_bytes) / len(tngl_bytes) if len(tngl_bytes) > 0 else 1.0
        st.metric(
            label="File size",
            value=f"{tngl_kb:.1f} KB",
        )
        st.caption(f"CSV was {csv_kb:.1f} KB ({ratio:.1f}x smaller)")

    st.write("")

    # 3. ONE Plotly chart titled "Sensor behavior over time"
    fig = go.Figure()

    # Score per window as a line
    x_vals = [t / rate for t in detection["window_start_ticks"]] if rate else detection["window_start_ticks"]
    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=detection["window_scores"],
            mode="lines+markers",
            name="Score",
            line=dict(color="#1f77b4", width=2.5),
            marker=dict(size=4),
        )
    )

    # Dashed threshold line labelled "Alert level"
    threshold = detection["threshold"]
    fig.add_hline(
        y=threshold,
        line_dash="dash",
        line_color="#d62728",
        line_width=2,
        annotation_text="Alert level",
        annotation_position="top left",
        annotation_font=dict(color="#d62728", size=12),
    )

    # Flagged ranges shaded red with the label "Fault"
    for i, r in enumerate(ranges):
        x0 = r["start_tick"] / rate if rate else r["start_tick"]
        x1 = r["end_tick"] / rate if rate else r["end_tick"]
        fig.add_vrect(
            x0=x0,
            x1=x1,
            fillcolor="#ff4b4b",
            opacity=0.2,
            line_width=0,
            annotation_text="Fault" if i == 0 else "",
            annotation_position="top left",
            annotation_font=dict(color="#b00", size=11, weight="bold"),
        )

    fig.update_layout(
        title="Sensor behavior over time",
        xaxis_title="Time (s)" if rate else "Time (ticks)",
        yaxis_title="Unusual behavior score",
        template="plotly_white",
        height=420,
        margin=dict(l=40, r=30, t=50, b=40),
        showlegend=False,
    )

    st.plotly_chart(fig, use_container_width=True)
    st.caption("Above the alert level for several windows in a row means a sensor is misbehaving.")

    st.write("")

    # 4. One download button: "Download compressed file (.tngl)"
    base_stem = Path(dataset_name).stem if dataset_name.endswith(".csv") else dataset_name.lower().replace(" ", "_")
    st.download_button(
        label="Download compressed file (.tngl)",
        data=tngl_bytes,
        file_name=f"{base_stem}.tngl",
        mime="application/octet-stream",
        use_container_width=True,
    )

    st.write("")

    # 5. A small grey note
    st.caption("This compressed file keeps the order of the sensors, not their exact values.")

    st.write("")

    # 6. A "Back" button that clears results and returns to Page 1
    if st.button("Back", use_container_width=True):
        st.session_state["results"] = None
        st.session_state["page"] = "input"
        st.rerun()
