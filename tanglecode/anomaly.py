"""Streaming anomaly detection using topological braid word reduction."""

from collections import Counter
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Tuple, Union

import numpy as np

from tanglecode.braid import (
    extract_braid_word,
    extract_crossings_with_channels,
    rank_vector,
    reduce_with_channels,
    reduce_word_full,
)


class AnomalyRecord(tuple):
    """Result record for a single sliding window.

    Behaves as a 3-tuple `(window_start, score, flagged)` for full backwards compatibility
    with `(window_start, score, flagged)` unpacking, while also providing rich attributes:
    `.window_start`, `.score`, `.flagged`, `.reduced_length`, `.raw_length`, `.ratio`,
    and `.most_frequent_generator`.
    """

    def __new__(
        cls,
        window_start: int,
        score: float,
        flagged: bool,
        reduced_length: int = 0,
        raw_length: int = 0,
        ratio: float = 0.0,
        most_frequent_generator: Optional[int] = None,
    ):
        return super().__new__(cls, (int(window_start), float(score), bool(flagged)))

    def __init__(
        self,
        window_start: int,
        score: float,
        flagged: bool,
        reduced_length: int = 0,
        raw_length: int = 0,
        ratio: float = 0.0,
        most_frequent_generator: Optional[int] = None,
    ):
        self._window_start = int(window_start)
        self._score = float(score)
        self._flagged = bool(flagged)
        self._reduced_length = int(reduced_length)
        self._raw_length = int(raw_length)
        self._ratio = float(ratio)
        self._most_frequent_generator = most_frequent_generator

    @property
    def window_start(self) -> int:
        return self[0]

    @property
    def score(self) -> float:
        return self[1]

    @property
    def flagged(self) -> bool:
        return self[2]

    @property
    def reduced_length(self) -> int:
        return self._reduced_length

    @property
    def raw_length(self) -> int:
        return self._raw_length

    @property
    def ratio(self) -> float:
        return self._ratio

    @property
    def most_frequent_generator(self) -> Optional[int]:
        return self._most_frequent_generator

    def __repr__(self) -> str:
        return f"({self[0]}, {self[1]:.4f}, {self[2]})"


class WindowBraidFeatures(NamedTuple):
    """Raw topological features extracted from a single window."""

    window_start: int
    raw_length: int
    reduced_length: int
    ratio: float
    most_frequent_generator: Optional[int]


def compute_window_features(window_data: np.ndarray, window_start: int = 0) -> WindowBraidFeatures:
    """Compute topological braid features for a single time window.

    - Irreducible reduced-word length
    - Ratio of reduced to raw length
    - Most frequently surviving generator
    """
    n_ticks = len(window_data)
    if n_ticks <= 1:
        return WindowBraidFeatures(window_start, 0, 0, 0.0, None)

    ranks = [rank_vector(frame) for frame in window_data]
    raw_word: list[int] = []

    for t in range(n_ticks - 1):
        raw_word.extend(extract_braid_word(ranks[t], ranks[t + 1]))

    raw_len = len(raw_word)
    reduced_word, _ = reduce_word_full(raw_word)
    reduced_len = len(reduced_word)

    ratio = float(reduced_len / raw_len) if raw_len > 0 else 0.0

    if reduced_word:
        most_freq_gen = Counter(reduced_word).most_common(1)[0][0]
    else:
        most_freq_gen = None

    return WindowBraidFeatures(
        window_start=window_start,
        raw_length=raw_len,
        reduced_length=reduced_len,
        ratio=ratio,
        most_frequent_generator=most_freq_gen,
    )


class BraidAnomalyDetector:
    """Streaming anomaly detector using braid word topological reductions."""

    def __init__(
        self,
        window_size: int = 200,
        n_baseline: int = 50,
        k: float = 3.0,
        step: int = 10,
        two_sided: bool = True,
    ):
        """Parameters
        ----------
        window_size : int
            Window length in ticks (default: 200).
        n_baseline : int
            Number of initial clean windows to establish baseline (default: 50).
        k : float
            Anomaly threshold multiplier in standard deviations (default: 3.0).
        step : int
            Sliding window step stride (default: 10).
        two_sided : bool
            If True, flags when |reduced_length - mean| > k * std.
            If False, flags only when reduced_length > mean + k * std.
        """
        self.window_size = window_size
        self.n_baseline = n_baseline
        self.k = k
        self.step = step
        self.two_sided = two_sided

        self.baseline_mean: Optional[float] = None
        self.baseline_std: Optional[float] = None

    def fit(self, data: np.ndarray) -> "BraidAnomalyDetector":
        """Learn baseline mean and standard deviation from the first N clean windows."""
        n_samples = len(data)
        baseline_lengths: list[int] = []

        for i in range(self.n_baseline):
            start = i * self.step
            if start + self.window_size > n_samples:
                break
            feat = compute_window_features(data[start : start + self.window_size], window_start=start)
            baseline_lengths.append(feat.reduced_length)

        if not baseline_lengths:
            raise ValueError(f"Insufficient samples {n_samples} for {self.n_baseline} baseline windows")

        self.baseline_mean = float(np.mean(baseline_lengths))
        computed_std = float(np.std(baseline_lengths))
        self.baseline_std = max(computed_std, 1e-4)

        return self

    def predict(self, data: np.ndarray) -> List[AnomalyRecord]:
        """Process streaming data and flag anomalous windows.

        Returns list of (window_start, score, flagged).
        """
        if self.baseline_mean is None or self.baseline_std is None:
            self.fit(data)

        assert self.baseline_mean is not None
        assert self.baseline_std is not None

        n_samples = len(data)
        records: list[AnomalyRecord] = []

        for start in range(0, n_samples - self.window_size + 1, self.step):
            feat = compute_window_features(data[start : start + self.window_size], window_start=start)
            score = float((feat.reduced_length - self.baseline_mean) / self.baseline_std)

            if self.two_sided:
                flagged = bool(abs(score) > self.k)
            else:
                flagged = bool(score > self.k)

            rec = AnomalyRecord(
                window_start=start,
                score=score,
                flagged=flagged,
                reduced_length=feat.reduced_length,
                raw_length=feat.raw_length,
                ratio=feat.ratio,
                most_frequent_generator=feat.most_frequent_generator,
            )
            records.append(rec)

        return records

    def evaluate(self, data: np.ndarray, fault_start: int) -> Dict[str, Any]:
        """Evaluate detection performance against known fault onset.

        Reports false positive rate, detection delay, and flag counts.
        """
        records = self.predict(data)

        clean_records = [r for r in records if r.window_start + self.window_size <= fault_start]
        fault_records = [r for r in records if r.window_start + self.window_size > fault_start]

        false_positives = sum(1 for r in clean_records if r.flagged)
        clean_total = len(clean_records)
        fpr = float(false_positives / clean_total) if clean_total > 0 else 0.0

        fault_flagged = [r for r in fault_records if r.flagged]
        fault_flagged_count = len(fault_flagged)

        if fault_flagged:
            first_flagged_start = fault_flagged[0].window_start
            # Delay in ticks from fault onset to window end where anomaly is recognized
            first_flagged_end = first_flagged_start + self.window_size
            detection_delay_ticks = max(0, first_flagged_end - fault_start)
        else:
            first_flagged_start = None
            detection_delay_ticks = None

        return {
            "clean_windows_count": clean_total,
            "false_positives_count": false_positives,
            "false_positive_rate": fpr,
            "fault_windows_total": len(fault_records),
            "fault_windows_flagged": fault_flagged_count,
            "first_flagged_window_start": first_flagged_start,
            "detection_delay_ticks": detection_delay_ticks,
            "baseline_mean": self.baseline_mean,
            "baseline_std": self.baseline_std,
            "records": records,
        }


def detect_anomalies(
    data: np.ndarray,
    window_size: int = 200,
    n_baseline: int = 50,
    k: float = 3.0,
    step: int = 10,
    two_sided: bool = True,
) -> List[AnomalyRecord]:
    """Convenience function running streaming braid anomaly detection over data."""
    detector = BraidAnomalyDetector(
        window_size=window_size,
        n_baseline=n_baseline,
        k=k,
        step=step,
        two_sided=two_sided,
    )
    return detector.predict(data)


class ChannelFaultDiagnosis(NamedTuple):
    """Diagnosis for a flagged anomaly window."""

    window_start: int
    most_likely_channel: int
    top_crossing_pair: Optional[tuple[int, int]]
    channel_z_scores: np.ndarray


def count_surviving_crossings_per_channel(
    window_data: np.ndarray,
) -> tuple[np.ndarray, Optional[tuple[int, int]]]:
    """Count how many surviving crossings involve each channel in a window, and find the top crossing pair.

    Parameters
    ----------
    window_data : np.ndarray
        Array of shape (window_size, n_channels).

    Returns
    -------
    tuple[np.ndarray, Optional[tuple[int, int]]]
        (channel_counts, top_crossing_pair)
        channel_counts is a vector of length n_channels.
        top_crossing_pair is (channel_a, channel_b) with channel_a < channel_b, or None if no crossings.
    """
    n_ticks = len(window_data)
    n_channels = window_data.shape[1] if window_data.ndim == 2 else 0
    if n_ticks <= 1 or n_channels == 0:
        return np.zeros(n_channels, dtype=np.float32), None

    ranks = [rank_vector(frame) for frame in window_data]
    raw_crossings: list[tuple[int, int, int]] = []
    for t in range(n_ticks - 1):
        raw_crossings.extend(extract_crossings_with_channels(ranks[t], ranks[t + 1]))

    reduced_crossings, _ = reduce_with_channels(raw_crossings)

    counts = np.zeros(n_channels, dtype=np.float32)
    pair_counts: Counter[tuple[int, int]] = Counter()

    for g, a, b in reduced_crossings:
        counts[a] += 1.0
        counts[b] += 1.0
        pair = (a, b) if a < b else (b, a)
        pair_counts[pair] += 1

    top_pair = pair_counts.most_common(1)[0][0] if pair_counts else None
    return counts, top_pair


def learn_channel_baseline(
    data: np.ndarray,
    window_size: int = 200,
    n_baseline: int = 50,
    step: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Learn per-channel mean and std of surviving crossing counts from the first N clean windows."""
    n_samples, n_channels = data.shape
    baseline_counts: list[np.ndarray] = []

    for i in range(n_baseline):
        start = i * step
        if start + window_size > n_samples:
            break
        w = data[start : start + window_size]
        counts, _ = count_surviving_crossings_per_channel(w)
        baseline_counts.append(counts)

    if not baseline_counts:
        return np.zeros(n_channels, dtype=np.float32), np.ones(n_channels, dtype=np.float32)

    mat = np.array(baseline_counts)
    mean = np.mean(mat, axis=0)
    std = np.maximum(np.std(mat, axis=0), 1e-4)
    return mean, std


def diagnose_window_fault(
    window_data: np.ndarray,
    baseline_channel_mean: np.ndarray,
    baseline_channel_std: np.ndarray,
    window_start: int = 0,
) -> ChannelFaultDiagnosis:
    """Compute per-channel fault score for a window.

    Computes per-channel z-score from surviving crossings, tuned to detect both
    excess crossings (drift) and frozen/stuck sensors (near-zero signal variance),
    and identifies the most likely faulty sensor and top crossing pair.
    """
    counts, top_pair = count_surviving_crossings_per_channel(window_data)
    z_scores = np.abs(counts - baseline_channel_mean) / np.maximum(baseline_channel_std, 1e-4)

    # Disambiguate stuck sensors: if a channel is frozen (variance near 0),
    # boost its anomaly score so it is correctly prioritized over its crossing neighbors.
    w_vars = np.var(window_data, axis=0)
    boost = np.where(w_vars < 0.005, 10.0, 1.0)
    effective_scores = z_scores * boost

    most_likely_channel = int(np.argmax(effective_scores))
    return ChannelFaultDiagnosis(
        window_start=window_start,
        most_likely_channel=most_likely_channel,
        top_crossing_pair=top_pair,
        channel_z_scores=effective_scores,
    )

