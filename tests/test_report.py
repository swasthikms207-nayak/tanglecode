from __future__ import annotations

import io
import re
import pytest
import numpy as np

from tanglecode.anomaly import detect
from tanglecode.braid import reduce_crossings, replay_orders, word_from_frames
from tanglecode.data import generate_signals
from tanglecode.pipeline import compute_stats
from tanglecode.report import make_report_pdf

try:
    import pypdf
except ImportError:
    pypdf = None


def count_pages(pdf_bytes: bytes) -> int:
    if pypdf is not None:
        reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
        return len(reader.pages)
    matches = re.findall(rb"/Type\s*/Page\b", pdf_bytes)
    return len(matches) if matches else 1


def test_make_report_pdf_clean():
    """Verify make_report_pdf for clean sample returns valid 1-page PDF bytes."""
    data = generate_signals("clean", n=6, T=3000, seed=7)
    T, n = data.shape
    init_order, raw = word_from_frames(data)
    red, cancelled = reduce_crossings(raw)
    pos = replay_orders(init_order, raw, T)
    res = detect(red, pos, n=n, T=T)
    stats = compute_stats(raw, red, cancelled)

    pdf_bytes = make_report_pdf("clean", n, T, stats, res, data)

    assert isinstance(pdf_bytes, bytes)
    assert pdf_bytes.startswith(b"%PDF")
    assert count_pages(pdf_bytes) == 1


def test_make_report_pdf_drift():
    """Verify make_report_pdf for drift sample returns valid 1-page PDF bytes."""
    data = generate_signals("drift", n=6, T=3000, seed=7)
    T, n = data.shape
    init_order, raw = word_from_frames(data)
    red, cancelled = reduce_crossings(raw)
    pos = replay_orders(init_order, raw, T)
    res = detect(red, pos, n=n, T=T)
    stats = compute_stats(raw, red, cancelled)

    pdf_bytes = make_report_pdf("drift", n, T, stats, res, data)

    assert isinstance(pdf_bytes, bytes)
    assert pdf_bytes.startswith(b"%PDF")
    assert count_pages(pdf_bytes) == 1


def test_make_report_pdf_with_sampling_rate():
    """Verify make_report_pdf supports sampling_rate parameter and stays 1 page."""
    data = generate_signals("drift", n=6, T=3000, seed=7)
    T, n = data.shape
    init_order, raw = word_from_frames(data)
    red, cancelled = reduce_crossings(raw)
    pos = replay_orders(init_order, raw, T)
    res = detect(red, pos, n=n, T=T)
    stats = compute_stats(raw, red, cancelled)

    pdf_bytes = make_report_pdf("drift", n, T, stats, res, data, sampling_rate=100.0)

    assert isinstance(pdf_bytes, bytes)
    assert pdf_bytes.startswith(b"%PDF")
    assert count_pages(pdf_bytes) == 1
