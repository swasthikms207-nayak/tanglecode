from __future__ import annotations

from typing import Any, Sequence
import numpy as np


def detect(
    reduced_crossings: Sequence[Any],
    positions: np.ndarray,
    n: int,
    T: int,
    window: int = 200,
    step: int = 100,
    k: float = 4.0,
    min_consecutive: int = 4,
    baseline_frac: float = 0.2,
    min_baseline_windows: int = 8,
) -> dict[str, Any]:
    """Detect anomalies in multichannel sensor streams using braid word topological reductions and ranks.

    Baseline definition:
      The baseline is computed from the first max(min_baseline_windows, int(baseline_frac * n_windows))
      windows, which are assumed to represent normal operation without sensor faults.

    Per window (start s, ticks s..s+window-1) compute, for each channel c:
      a) crossings_c = surviving crossings involving c in the window
      b) rank_c = mean of positions[window, c]

    Robust z-scores:
      z = (value - median) / max(1.4826 * MAD, floor)
      floor = 1.0 for crossing counts and 0.25 for mean rank.

    Scoring & Flagging:
      channel score = max(z_crossings_c, abs(z_rank_c))
      window score = max over channels
      A window is flagged if its score > k.
      A RANGE counts only if at least min_consecutive consecutive windows are flagged;
      adjacent flagged windows are merged.

    Returns:
      dict with:
        - "window_start_ticks": list of window start ticks
        - "window_scores": list of float window anomaly scores
        - "threshold": float threshold k
        - "ranges": list of dicts [{start_tick, end_tick, faulty_channel}],
          where faulty_channel is 1-based with the highest summed score over the range.
          If no range survives, ranges is [].
    """
    window_starts = list(range(0, T - window + 1, step))
    n_windows = len(window_starts)

    if n_windows == 0:
        return {
            "window_start_ticks": [],
            "window_starts": [],
            "window_scores": [],
            "threshold": float(k),
            "k": float(k),
            "ranges": [],
        }

    n_base = min(n_windows, max(min_baseline_windows, int(baseline_frac * n_windows)))

    crossings_mat = np.zeros((n_windows, n), dtype=float)
    rank_mat = np.zeros((n_windows, n), dtype=float)

    # Pre-extract channel pairs for each crossing
    crossing_records: list[tuple[int, int, int]] = []
    for c in reduced_crossings:
        t = int(c.tick) if hasattr(c, "tick") else int(c[0])
        if hasattr(c, "x") and hasattr(c, "y"):
            ch_x, ch_y = int(c.x), int(c.y)
        else:
            # Reconstruct channels involved from generator at tick t
            gen = int(c.gen) if hasattr(c, "gen") else int(c[1])
            pos_swap = abs(gen)
            # Find channels at rank pos_swap and pos_swap + 1
            tick_idx = min(max(0, t), T - 1)
            matching_x = np.where(positions[tick_idx] == pos_swap)[0]
            matching_y = np.where(positions[tick_idx] == pos_swap + 1)[0]
            ch_x = int(matching_x[0]) if len(matching_x) > 0 else 0
            ch_y = int(matching_y[0]) if len(matching_y) > 0 else 1
        crossing_records.append((t, ch_x, ch_y))

    # Compute window features
    for w_idx, s in enumerate(window_starts):
        e = s + window
        # a) surviving crossings involving channel c in the window
        for t, ch_x, ch_y in crossing_records:
            if s <= t < e:
                crossings_mat[w_idx, ch_x] += 1.0
                crossings_mat[w_idx, ch_y] += 1.0
        # b) rank_c = mean of positions[window, c]
        for c_idx in range(n):
            rank_mat[w_idx, c_idx] = float(np.mean(positions[s:e, c_idx]))

    # Robust baseline stats
    med_cross = np.median(crossings_mat[:n_base], axis=0)
    mad_cross = np.median(np.abs(crossings_mat[:n_base] - med_cross), axis=0)
    scale_cross = np.maximum(1.4826 * mad_cross, 1.0)

    med_rank = np.median(rank_mat[:n_base], axis=0)
    mad_rank = np.median(np.abs(rank_mat[:n_base] - med_rank), axis=0)
    scale_rank = np.maximum(1.4826 * mad_rank, 0.25)

    z_cross = (crossings_mat - med_cross) / scale_cross
    z_rank = (rank_mat - med_rank) / scale_rank

    ch_scores = np.maximum(z_cross, np.abs(z_rank))
    window_scores = np.max(ch_scores, axis=1)

    flagged = window_scores > k

    # Group contiguous flagged windows into ranges (runs >= min_consecutive)
    runs: list[tuple[int, int]] = []
    in_run = False
    run_start = 0

    for i, is_flagged in enumerate(flagged):
        if is_flagged and not in_run:
            in_run = True
            run_start = i
        elif not is_flagged and in_run:
            in_run = False
            if (i - run_start) >= min_consecutive:
                runs.append((run_start, i - 1))

    if in_run and (len(flagged) - run_start) >= min_consecutive:
        runs.append((run_start, len(flagged) - 1))

    # Format surviving ranges
    surviving_ranges: list[dict[str, Any]] = []
    for s_idx, e_idx in runs:
        start_tick = window_starts[s_idx]
        end_tick = window_starts[e_idx] + window
        ch_sums = np.sum(ch_scores[s_idx : e_idx + 1], axis=0)
        faulty_channel = int(np.argmax(ch_sums) + 1)  # 1-based
        ch_idx = faulty_channel - 1

        # kind: "drift" if faulty channel's rank-shift z-score is larger than
        # its crossing-count z-score over the range, else "erratic"
        z_rank_range = float(np.mean(np.abs(z_rank[s_idx : e_idx + 1, ch_idx])))
        z_cross_range = float(np.mean(z_cross[s_idx : e_idx + 1, ch_idx]))
        kind = "drift" if z_rank_range > z_cross_range else "erratic"

        # rank_before: the channel's median position in the baseline windows (1-based, rounded)
        rank_before = int(round(float(med_rank[ch_idx])))

        # rank_after: its median position over the last windows of the range (1-based, rounded)
        last_w = max(1, min(min_consecutive, e_idx - s_idx + 1))
        rank_after = int(round(float(np.median(rank_mat[e_idx - last_w + 1 : e_idx + 1, ch_idx]))))

        # severity: "Mild" if peak score < 1.5 * k, "Moderate" if < 3 * k, otherwise "High"
        peak_score = float(np.max(window_scores[s_idx : e_idx + 1]))
        if peak_score < 1.5 * k:
            severity = "Mild"
        elif peak_score < 3.0 * k:
            severity = "Moderate"
        else:
            severity = "High"

        surviving_ranges.append({
            "start_tick": start_tick,
            "end_tick": end_tick,
            "faulty_channel": faulty_channel,
            "kind": kind,
            "rank_before": rank_before,
            "rank_after": rank_after,
            "severity": severity,
        })

    return {
        "window_start_ticks": window_starts,
        "window_starts": window_starts,
        "window_scores": [float(x) for x in window_scores],
        "threshold": float(k),
        "k": float(k),
        "ranges": surviving_ranges,
    }
