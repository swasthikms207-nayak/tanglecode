from __future__ import annotations

import datetime
import io
import re
from typing import Any, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import (
    HRFlowable,
    Image,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

try:
    import pypdf
except ImportError:
    pypdf = None


def format_ordinal(val: int) -> str:
    """Format an integer as an ordinal string (1st, 2nd, 3rd, etc.)."""
    n = int(val)
    if 11 <= (n % 100) <= 13:
        return f"{n}th"
    last = n % 10
    if last == 1:
        return f"{n}st"
    elif last == 2:
        return f"{n}nd"
    elif last == 3:
        return f"{n}rd"
    return f"{n}th"


def _count_pdf_pages(pdf_bytes: bytes) -> int:
    """Count pages in a PDF byte string."""
    if pypdf is not None:
        try:
            reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
            return len(reader.pages)
        except Exception:
            pass
    # Fallback regex counting of /Type /Page (excluding /Type /Pages)
    matches = re.findall(rb"/Type\s*/Page\b", pdf_bytes)
    return len(matches) if matches else 1


def _render_chart_png(
    data: np.ndarray,
    n: int,
    T: int,
    ranges: Sequence[dict[str, Any]],
    sampling_rate: float | None,
    width_inches: float = 7.25,
    height_inches: float = 2.8,
) -> io.BytesIO:
    """Render multichannel sensor streams and fault regions to a PNG image buffer."""
    fig, ax = plt.subplots(figsize=(width_inches, height_inches), dpi=200)

    # Time axis
    if sampling_rate and sampling_rate > 0:
        x_axis = np.arange(T) / sampling_rate
        x_label = "Time (s)"
    else:
        x_axis = np.arange(T, dtype=float)
        x_label = "Time (ticks)"

    # Downsample if T > 1500 points per line
    if T > 1500:
        step = int(np.ceil(T / 1500))
        indices = np.arange(0, T, step)
        t_plot = x_axis[indices]
        data_plot = data[indices, :]
    else:
        t_plot = x_axis
        data_plot = data

    faulty_channel = ranges[0]["faulty_channel"] if len(ranges) > 0 else None

    # Color palette
    colors_cycle = plt.cm.tab10.colors

    # Plot sensors
    for ch_idx in range(n):
        ch_num = ch_idx + 1
        sensor_color = colors_cycle[ch_idx % len(colors_cycle)]
        if ch_num == faulty_channel:
            ax.plot(
                t_plot,
                data_plot[:, ch_idx],
                color="#dc2626",
                linewidth=2.2,
                label=f"Sensor {ch_num} (Fault)",
                zorder=5,
            )
        else:
            ax.plot(
                t_plot,
                data_plot[:, ch_idx],
                color=sensor_color,
                linewidth=0.85,
                alpha=0.75,
                label=f"Sensor {ch_num}",
                zorder=3,
            )

    # Shaded red fault ranges
    for idx, r in enumerate(ranges):
        x0 = r["start_tick"] / sampling_rate if sampling_rate else float(r["start_tick"])
        x1 = r["end_tick"] / sampling_rate if sampling_rate else float(r["end_tick"])
        ax.axvspan(x0, x1, facecolor="#fee2e2", edgecolor="#ef4444", linestyle="--", alpha=0.6, zorder=2)
        mid_x = (x0 + x1) / 2.0
        # Text label inside shaded region
        y_lims = ax.get_ylim()
        y_pos = y_lims[1] - 0.08 * (y_lims[1] - y_lims[0])
        ax.text(
            mid_x,
            y_pos,
            "Fault",
            color="#b91c1c",
            fontsize=8,
            fontweight="bold",
            ha="center",
            va="top",
            zorder=6,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#ffffff", edgecolor="#ef4444", alpha=0.9),
        )

    ax.set_title("Sensor Readings Over Time", fontsize=10, fontweight="bold", pad=6)
    ax.set_xlabel(x_label, fontsize=8, labelpad=3)
    ax.set_ylabel("Reading", fontsize=8, labelpad=3)
    ax.tick_params(labelsize=7.5)
    ax.grid(True, linestyle="--", alpha=0.35, zorder=0)

    # Legend with up to 6 columns
    ncol = min(n, 6)
    ax.legend(
        loc="upper right",
        fontsize=7,
        ncol=ncol,
        framealpha=0.9,
        edgecolor="#cccccc",
        handlelength=1.5,
    )

    fig.tight_layout(pad=0.8)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200)
    plt.close(fig)
    buf.seek(0)
    return buf


def _build_pdf(
    name: str,
    n: int,
    T: int,
    stats: dict[str, Any],
    result: dict[str, Any],
    data: np.ndarray,
    sampling_rate: float | None = None,
    chart_height_pt: float = 195.0,
    spacer_factor: float = 1.0,
) -> bytes:
    """Internal helper to construct PDF bytes given a specific chart height."""
    buf = io.BytesIO()

    # Printable area: A4 is 595.28 x 841.89 pt
    margin = 36.0  # 0.5 inch
    printable_width = 595.28 - 2 * margin  # 523.28 pt

    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=margin,
        rightMargin=margin,
        topMargin=margin,
        bottomMargin=margin,
    )

    styles = getSampleStyleSheet()

    # Base typography (one font family: Helvetica)
    title_style = ParagraphStyle(
        "ReportTitle",
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#0f172a"),
    )
    meta_style = ParagraphStyle(
        "ReportMeta",
        fontName="Helvetica",
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#475569"),
    )
    desc_style = ParagraphStyle(
        "ReportDesc",
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#1e293b"),
    )
    footer_style = ParagraphStyle(
        "ReportFooter",
        fontName="Helvetica-Oblique",
        fontSize=7.5,
        leading=10.5,
        textColor=colors.HexColor("#64748b"),
        alignment=1,  # Centered
    )

    story: list[Any] = []

    # 1. Title & Metadata
    story.append(Paragraph("TangleCode Fault Report", title_style))
    story.append(Spacer(1, 3 * spacer_factor))

    today_str = datetime.date.today().isoformat()
    rate_info = f"  |  Rate: <b>{sampling_rate:.1f} Hz</b>" if sampling_rate else ""
    meta_line = (
        f"Dataset: <b>{name}</b>  |  Sensors: <b>{n}</b>  |  "
        f"Samples: <b>{T:,}</b> ticks{rate_info}  |  Date: <b>{today_str}</b>"
    )
    story.append(Paragraph(meta_line, meta_style))
    story.append(Spacer(1, 8 * spacer_factor))
    story.append(HRFlowable(width="100%", thickness=0.75, color=colors.HexColor("#cbd5e1"), spaceAfter=6 * spacer_factor))

    # 2. Verdict colored box
    ranges = result.get("ranges", [])
    if len(ranges) == 0:
        verdict_text = "All sensors normal"
        verdict_bg = colors.HexColor("#f0fdf4")
        verdict_border = colors.HexColor("#22c55e")
        verdict_color = colors.HexColor("#15803d")
    else:
        first_fault = ranges[0]
        ch = first_fault["faulty_channel"]
        if sampling_rate and sampling_rate > 0:
            start_s = first_fault["start_tick"] / sampling_rate
            end_s = first_fault["end_tick"] / sampling_rate
            verdict_text = f"Fault detected: Sensor {ch}, around {start_s:.1f}s to {end_s:.1f}s"
        else:
            verdict_text = f"Fault detected: Sensor {ch}, around ticks {first_fault['start_tick']} to {first_fault['end_tick']}"
        verdict_bg = colors.HexColor("#fef2f2")
        verdict_border = colors.HexColor("#ef4444")
        verdict_color = colors.HexColor("#b91c1c")

    verdict_para_style = ParagraphStyle(
        "VerdictPara",
        fontName="Helvetica-Bold",
        fontSize=11.5,
        leading=15,
        textColor=verdict_color,
        alignment=1,
    )
    verdict_table = Table(
        [[Paragraph(f"<b>{verdict_text}</b>", verdict_para_style)]],
        colWidths=[printable_width],
    )
    verdict_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), verdict_bg),
            ("BOX", (0, 0), (-1, -1), 1.2, verdict_border),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("LEFTPADDING", (0, 0), (-1, -1), 10),
            ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ])
    )
    story.append(verdict_table)
    story.append(Spacer(1, 6 * spacer_factor))

    # 3. Fault description / severity or healthy summary
    desc_text: str | None = None
    if len(ranges) > 0:
        first_fault = ranges[0]
        ch = first_fault.get("faulty_channel", 1)
        sev = result.get("severity") or first_fault.get("severity")
        if "description" in result:
            desc_text = result["description"]
        elif "desc" in result:
            desc_text = result["desc"]
        elif sev is not None and "kind" in first_fault:
            kind = first_fault.get("kind", "drift")
            if kind == "drift":
                nb = format_ordinal(first_fault.get("rank_before", 1))
                na = format_ordinal(first_fault.get("rank_after", 1))
                desc_text = (
                    f"Sensor {ch} moved from {nb} to {na} place among the sensors and stayed there. "
                    f"It looks like it is drifting away from the group. "
                    f"Severity: <b>{sev}</b>"
                )
            else:
                desc_text = (
                    f"Sensor {ch} started swapping places with the other sensors far more often than normal. "
                    f"It looks erratic. "
                    f"Severity: <b>{sev}</b>"
                )
        elif sev is not None:
            desc_text = f"Severity: <b>{sev}</b>"
    else:
        if "healthy_summary" in result:
            desc_text = result["healthy_summary"]
        elif "description" in result:
            desc_text = result["description"]
        elif "desc" in result:
            desc_text = result["desc"]
        elif "summary" in result:
            desc_text = result["summary"]

    if desc_text:
        story.append(Paragraph(desc_text, desc_style))
        story.append(Spacer(1, 6 * spacer_factor))

    # 4. Three numbers in a row
    raw_count = int(stats.get("raw_crossings", stats.get("raw_count", 0)))
    reduced_count = int(stats.get("reduced_crossings", stats.get("reduced_count", 0)))
    pct_removed = stats.get("percent_eliminated", stats.get("percent_removed"))
    if pct_removed is None:
        pct_removed = ((raw_count - reduced_count) / raw_count * 100.0) if raw_count > 0 else 0.0
    else:
        pct_removed = float(pct_removed)

    # File size metrics
    if "orig_kb" in stats:
        orig_kb = float(stats["orig_kb"])
    elif "raw_csv_bytes" in stats and stats["raw_csv_bytes"] is not None:
        orig_kb = len(stats["raw_csv_bytes"]) / 1024.0
    else:
        df_bytes = len(pd.DataFrame(data).round(4).to_csv(index=False).encode("utf-8"))
        orig_kb = df_bytes / 1024.0

    if "swap_kb" in stats:
        swap_kb = float(stats["swap_kb"])
    elif "swap_log_str" in stats and stats["swap_log_str"] is not None:
        swap_kb = len(stats["swap_log_str"].encode("utf-8")) / 1024.0
    elif "swap_log_bytes" in stats and stats["swap_log_bytes"] is not None:
        swap_kb = len(stats["swap_log_bytes"]) / 1024.0
    else:
        # Approximate: ~45 byte header + ~18 bytes per reduced crossing
        swap_kb = (45.0 + reduced_count * 18.0) / 1024.0

    ratio = (orig_kb / swap_kb) if swap_kb > 0 else 1.0

    card_col_w = printable_width / 3.0
    lbl_style = ParagraphStyle(
        "CardLbl",
        fontName="Helvetica-Bold",
        fontSize=8,
        leading=10,
        textColor=colors.HexColor("#475569"),
        alignment=1,
    )
    val_style = ParagraphStyle(
        "CardVal",
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=16,
        textColor=colors.HexColor("#0f172a"),
        alignment=1,
    )
    sub_style = ParagraphStyle(
        "CardSub",
        fontName="Helvetica",
        fontSize=7.5,
        leading=10,
        textColor=colors.HexColor("#64748b"),
        alignment=1,
    )

    card1 = [
        Paragraph("Order swaps found", lbl_style),
        Spacer(1, 2),
        Paragraph(f"{raw_count:,}", val_style),
        Paragraph("raw crossings", sub_style),
    ]
    card2 = [
        Paragraph("After noise removal", lbl_style),
        Spacer(1, 2),
        Paragraph(f"{reduced_count:,}", val_style),
        Paragraph(f"-{pct_removed:.1f}% noise eliminated", sub_style),
    ]
    card3 = [
        Paragraph("File size", lbl_style),
        Spacer(1, 2),
        Paragraph(f"{swap_kb:.1f} KB", val_style),
        Paragraph(f"Orig: {orig_kb:.1f} KB ({ratio:.1f}x smaller)", sub_style),
    ]

    metrics_table = Table([[card1, card2, card3]], colWidths=[card_col_w, card_col_w, card_col_w])
    metrics_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
            ("BOX", (0, 0), (-1, -1), 0.75, colors.HexColor("#e2e8f0")),
            ("INNERGRID", (0, 0), (-1, -1), 0.75, colors.HexColor("#e2e8f0")),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ])
    )
    story.append(metrics_table)
    story.append(Spacer(1, 8 * spacer_factor))

    # 5. Full width Chart
    chart_height_inches = chart_height_pt / 72.0
    chart_width_inches = printable_width / 72.0
    img_buf = _render_chart_png(
        data=data,
        n=n,
        T=T,
        ranges=ranges,
        sampling_rate=sampling_rate,
        width_inches=chart_width_inches,
        height_inches=chart_height_inches,
    )
    chart_img = Image(img_buf, width=printable_width, height=chart_height_pt)
    story.append(chart_img)
    story.append(Spacer(1, 8 * spacer_factor))

    # 6. Sensor health table (if exists in app or result, at most 8 rows)
    health_table_data = result.get("health_table") or result.get("sensor_health")
    if health_table_data and isinstance(health_table_data, (list, tuple)) and len(health_table_data) > 0:
        table_rows = [["Sensor", "Status", "Swaps kept"]]
        for item in health_table_data[:8]:
            if isinstance(item, dict):
                s_name = str(item.get("Sensor", item.get("sensor", "")))
                s_stat = str(item.get("Status", item.get("status", "")))
                s_swaps = str(item.get("Swaps kept", item.get("swaps", "")))
            elif isinstance(item, (list, tuple)) and len(item) >= 3:
                s_name, s_stat, s_swaps = str(item[0]), str(item[1]), str(item[2])
            else:
                continue
            table_rows.append([s_name, s_stat, s_swaps])

        if len(table_rows) > 1:
            th_style = ParagraphStyle("TH", fontName="Helvetica-Bold", fontSize=7.5, leading=9, alignment=1)
            td_style = ParagraphStyle("TD", fontName="Helvetica", fontSize=7.5, leading=9, alignment=1)
            formatted_table = []
            for r_idx, row in enumerate(table_rows):
                st_use = th_style if r_idx == 0 else td_style
                formatted_table.append([Paragraph(cell, st_use) for cell in row])

            cw = printable_width / 3.0
            ht = Table(formatted_table, colWidths=[cw, cw, cw])
            ht.setStyle(
                TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f1f5f9")),
                    ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                    ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
                    ("TOPPADDING", (0, 0), (-1, -1), 3),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ])
            )
            story.append(ht)
            story.append(Spacer(1, 6 * spacer_factor))

    # 7. Footer note
    footer_text = (
        "This report is based on the order of the sensors, not their exact values. "
        "&lsquo;Looks like&rsquo; findings are likely causes, not confirmed ones. "
        "The first part of the recording is assumed normal."
    )
    story.append(Paragraph(footer_text, footer_style))

    # Build document
    doc.build(story)
    return buf.getvalue()


def make_report_pdf(
    name: str,
    n: int,
    T: int,
    stats: dict[str, Any],
    result: dict[str, Any],
    data: np.ndarray,
    sampling_rate: float | None = None,
) -> bytes:
    """Generate a clean, professional one-page A4 PDF fault report.

    Args:
        name: Name of dataset or uploaded file.
        n: Number of sensor channels.
        T: Total number of temporal samples (ticks).
        stats: Summary statistics dict (raw crossings, reduced crossings, file size).
        result: Anomaly detection result dict (ranges, threshold, scores).
        data: Multichannel numpy array of shape (T, n).
        sampling_rate: Optional sampling frequency in Hz.

    Returns:
        Exact bytes of a 1-page A4 PDF document.
    """
    # Start with generous chart height, then shrink if it overflows 1 page
    candidate_chart_heights = [215.0, 185.0, 160.0, 135.0, 110.0]

    for idx, chart_h in enumerate(candidate_chart_heights):
        spacer_factor = 1.0 if idx == 0 else 0.8
        pdf_bytes = _build_pdf(
            name=name,
            n=n,
            T=T,
            stats=stats,
            result=result,
            data=data,
            sampling_rate=sampling_rate,
            chart_height_pt=chart_h,
            spacer_factor=spacer_factor,
        )
        pages = _count_pdf_pages(pdf_bytes)
        if pages == 1:
            return pdf_bytes

    # If still > 1 page, return the smallest candidate
    return pdf_bytes
