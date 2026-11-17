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
from tanglecode.data import generate_signals, load_csv
from tanglecode.export import swap_log_csv
from tanglecode.pipeline import compute_stats
from tanglecode.ranking import get_place_changes_caption, get_sensor_ranking
from tanglecode.report import make_report_pdf

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

            stats = compute_stats(raw, reduced, cancelled)
            swap_log_str = swap_log_csv(reduced)

            st.session_state["results"] = {
                "dataset_name": dataset_name,
                "n_sensors": n,
                "sampling_rate": None,
                "raw_count": len(raw),
                "reduced_count": len(reduced),
                "raw_csv_bytes": raw_csv_bytes,
                "swap_log_str": swap_log_str,
                "stats": stats,
                "detection": detection,
                "data": data,
                "T": T,
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
    swap_log_str = res["swap_log_str"]
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

    # Severity pane with distinct color for each severity level
    if len(ranges) > 0:
        first_fault = ranges[0]
        severity = first_fault.get("severity", "Moderate")
        severity_styles = {
            "Mild": {
                "bg": "#fefce8",
                "border": "#eab308",
                "badge_bg": "#ca8a04",
                "text": "#713f12",
                "icon": "⚠️",
                "desc": "Mild deviation — score slightly above the alert level.",
            },
            "Moderate": {
                "bg": "#fff7ed",
                "border": "#f97316",
                "badge_bg": "#ea580c",
                "text": "#7c2d12",
                "icon": "⚡",
                "desc": "Moderate deviation — score significantly above the alert level.",
            },
            "High": {
                "bg": "#fef2f2",
                "border": "#ef4444",
                "badge_bg": "#dc2626",
                "text": "#7f1d1d",
                "icon": "🚨",
                "desc": "High deviation — score far above the alert level.",
            },
        }
        style = severity_styles.get(severity, severity_styles["Moderate"])
        st.markdown(
            f"""
            <div style="
                background-color: {style['bg']};
                border: 1px solid {style['border']};
                border-left: 6px solid {style['border']};
                border-radius: 8px;
                padding: 10px 14px;
                margin: 6px 0 16px 0;
            ">
                <div style="display: flex; align-items: center; justify-content: space-between;">
                    <div style="display: flex; align-items: center; gap: 8px;">
                        <span style="font-size: 18px;">{style['icon']}</span>
                        <span style="font-size: 14px; font-weight: 600; color: {style['text']};">
                            Severity: {severity}
                        </span>
                    </div>
                    <span style="
                        background-color: {style['badge_bg']};
                        color: #ffffff;
                        font-size: 11px;
                        font-weight: 700;
                        padding: 2px 8px;
                        border-radius: 10px;
                        letter-spacing: 0.5px;
                    ">{severity.upper()}</span>
                </div>
                <div style="font-size: 12px; color: {style['text']}; opacity: 0.85; margin-top: 4px;">
                    {style['desc']}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

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
        orig_kb = len(raw_csv_bytes) / 1024.0
        swap_log_bytes = swap_log_str.encode("utf-8")
        swap_kb = len(swap_log_bytes) / 1024.0
        ratio = len(raw_csv_bytes) / len(swap_log_bytes) if len(swap_log_bytes) > 0 else 1.0
        st.metric(
            label="File size",
            value=f"{swap_kb:.1f} KB",
        )
        st.caption(f"Original CSV was {orig_kb:.1f} KB ({ratio:.1f} times smaller)")

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

    # Generate and cache PDF fault report in session state
    if "pdf_bytes" not in res or res["pdf_bytes"] is None:
        report_stats = dict(res.get("stats", {}))
        report_stats["raw_csv_bytes"] = raw_csv_bytes
        report_stats["swap_log_str"] = swap_log_str
        report_result = dict(detection)
        report_result["desc"] = desc
        res["pdf_bytes"] = make_report_pdf(
            name=dataset_name,
            n=n_sensors,
            T=res.get("T", 3000),
            stats=report_stats,
            result=report_result,
            data=res.get("data"),
            sampling_rate=rate,
        )
    pdf_bytes = res["pdf_bytes"]

    # 4. Download buttons: swap log CSV and fault report PDF side-by-side
    base_stem = Path(dataset_name).stem if dataset_name.endswith(".csv") else dataset_name.lower().replace(" ", "_")
    dl_col1, dl_col2 = st.columns(2)
    with dl_col1:
        st.download_button(
            label="Download swap log (CSV)",
            data=swap_log_str,
            file_name=f"{base_stem}_swap_log.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with dl_col2:
        st.download_button(
            label="Download fault report (PDF)",
            data=pdf_bytes,
            file_name=f"{base_stem}_fault_report.pdf",
            mime="application/pdf",
            use_container_width=True,
        )

    st.write("")

    # 5. A small grey note
    st.caption("The swap log keeps the order of the sensors, not their exact values.")

    st.write("")

    # 6. Sensor ranking at a moment
    st.subheader("Sensor ranking at a moment")
    st.caption("Move the slider to see which sensor reads lowest and which reads highest at any point in the recording.")

    data = res.get("data")
    T = res.get("T", data.shape[0] if data is not None else 3000)
    default_tick = ranges[0]["start_tick"] if len(ranges) > 0 else (T // 2)

    if rate is not None and rate > 0:
        max_time_s = float(round((T - 1) / rate, 1))
        def_time_s = float(round(default_tick / rate, 1))
        slider_time_s = st.slider(
            "Time (s)",
            min_value=0.0,
            max_value=max_time_s,
            value=def_time_s,
            step=0.1,
            format="%.1fs",
            key="sensor_ranking_time_slider",
        )
        chosen_tick = int(round(slider_time_s * rate))
        chosen_tick = max(0, min(T - 1, chosen_tick))
    else:
        chosen_tick = st.slider(
            "Time (tick)",
            min_value=0,
            max_value=T - 1,
            value=default_tick,
            step=1,
            key="sensor_ranking_time_slider",
        )

    if data is not None and 0 <= chosen_tick < len(data):
        ranking_df = get_sensor_ranking(data, chosen_tick, ranges)

        def highlight_fault(row):
            if "(suspect)" in str(row["Sensor"]):
                return ["background-color: #fee2e2; color: #b91c1c; font-weight: bold;"] * len(row)
            return [""] * len(row)

        styled_df = ranking_df.style.apply(highlight_fault, axis=1).format({"Reading": "{:.4f}"})
        st.dataframe(styled_df, use_container_width=True, hide_index=True)

        caption_txt = get_place_changes_caption(data, chosen_tick)
        if caption_txt is not None:
            st.caption(caption_txt)

    st.write("")

    # 7. A "Back" button that clears results and returns to Page 1
    if st.button("Back", use_container_width=True):
        st.session_state["results"] = None
        st.session_state["page"] = "input"
        st.rerun()
