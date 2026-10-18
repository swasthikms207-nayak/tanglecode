
import streamlit as st
import pandas as pd
import streamlit.components.v1 as components

from tangle_core import TangleCore

st.set_page_config(
    page_title="TangleCode Engine",
    page_icon="🪢",
    layout="wide",
)

# Diagnostic: confirms that Streamlit is rendering the script.
st.title("🪢 TangleCode: Knot-Theoretic Telemetry Engine")
st.caption(
    "A prototype for visualizing sensor jitter and "
    "experimental braid-word reductions."
)

st.success("Streamlit is rendering the application.")

# Sidebar controls
with st.sidebar:
    st.header("Sensor Parameters")

    steps = st.slider(
        "Time Frames",
        min_value=50,
        max_value=400,
        value=120,
        step=10,
    )

    noise = st.slider(
        "Jitter Amplitude",
        min_value=0.02,
        max_value=0.40,
        value=0.18,
        step=0.02,
    )

    st.info("8 parallel sensor tracks")

# Run the engine and display actionable errors.
try:
    core = TangleCore(n_channels=8)

    frames = core.generate_mock_telemetry(
        steps=steps,
        jitter=noise,
    )

    raw_braid = core.frames_to_braid(frames)
    reduced_braid, cancelled = core.untangle(raw_braid)

except Exception as exc:
    st.error("The TangleCode engine could not run.")
    st.exception(exc)
    st.info(
        "Check tangle_core.py and confirm that all three "
        "TangleCore methods are implemented."
    )
    st.stop()

# Metrics
st.subheader("Topological Reduction Metrics")

m1, m2, m3, m4 = st.columns(4)

raw_count = len(raw_braid)
reduced_count = len(reduced_braid)

reduction_pct = (
    (raw_count - reduced_count) / raw_count * 100
    if raw_count
    else 0.0
)

m1.metric("Raw Crossings", raw_count)
m2.metric("Remaining Crossings", reduced_count)
m3.metric("Cancelled Pairs", cancelled)
m4.metric("Word Reduction", f"{reduction_pct:.1f}%")

st.caption(
    "Word reduction measures the percentage of braid generators "
    "removed by the prototype's adjacent inverse-cancellation rule. "
    "It is not a measured percentage of real-world sensor noise removed."
)

st.divider()


def render_braid_svg(braid, label, n_channels=8, max_crossings=50):
    """Render a simple SVG illustration of a braid word."""

    subset = braid[:max_crossings]
    width = max(len(subset) * 20 + 60, 240)
    height = max(n_channels * 22 + 45, 180)
    spacing = 22
    start_y = 30

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="100%" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#111827" rx="8"/>',
        f'<text x="12" y="19" fill="#d1d5db" '
        f'font-size="11">{label} '
        f'({len(subset)} displayed crossings)</text>',
    ]

    # Horizontal strands
    for i in range(n_channels):
        y = start_y + i * spacing
        svg.append(
            f'<line x1="10" y1="{y}" x2="{width - 10}" y2="{y}" '
            f'stroke="#64748b" stroke-width="1.5"/>'
        )

    # Crossing symbols
    for idx, generator in enumerate(subset):
        g = abs(int(generator))

        if g < 1 or g >= n_channels:
            continue

        x = 30 + idx * 20
        y1 = start_y + (g - 1) * spacing
        y2 = start_y + g * spacing

        color = "#FF4B4B" if generator > 0 else "#00D4B2"

        svg.append(
            f'<line x1="{x - 5}" y1="{y1}" '
            f'x2="{x + 5}" y2="{y2}" '
            f'stroke="{color}" stroke-width="2.5"/>'
        )

        svg.append(
            f'<line x1="{x - 5}" y1="{y2}" '
            f'x2="{x + 5}" y2="{y1}" '
            f'stroke="{color}" stroke-width="2.5"/>'
        )

    svg.append("</svg>")
    return "".join(svg)


# Braid visualizations
st.subheader("Braid Visualization")

left, right = st.columns(2)

with left:
    st.markdown("#### 1. Raw Braid")
    components.html(
        render_braid_svg(raw_braid, "Raw channel inversions"),
        height=230,
        scrolling=False,
    )

with right:
    st.markdown("#### 2. Reduced Braid Word")
    components.html(
        render_braid_svg(reduced_braid, "After inverse cancellation"),
        height=230,
        scrolling=False,
    )

st.caption(
    "Red and teal indicate opposite generator signs. "
    "These are schematic illustrations, not a full geometric "
    "simulation of strand trajectories."
)

st.divider()

# Sensor signal chart
st.subheader("Raw Oscillating Input Signals")

df = pd.DataFrame(
    frames,
    columns=[f"Ch_{i}" for i in range(8)],
)

st.line_chart(
    df[["Ch_0", "Ch_1", "Ch_2", "Ch_3"]],
    height=300,
)

with st.expander("View telemetry data"):
    st.dataframe(df, use_container_width=True)

st.caption("TangleCode prototype • Synthetic telemetry data")
