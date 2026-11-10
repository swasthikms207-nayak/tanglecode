"""Synthetic data generation, fault injection, and sensor CSV loading for TangleCode."""

import csv
from pathlib import Path
from typing import Sequence, Union

import numpy as np


def generate_synthetic(
    n_channels: int = 8,
    n_samples: int = 5000,
    noise_std: float = 0.05,
    seed: int = 0,
) -> np.ndarray:
    """Generate simulated IMU-like multichannel signals.

    Combines multiple sinusoids at channel-varying frequencies and phases,
    plus additive Gaussian noise.

    Parameters
    ----------
    n_channels : int
        Number of channels (default: 8).
    n_samples : int
        Number of time samples (default: 5000).
    noise_std : float
        Standard deviation of Gaussian noise (default: 0.05).
    seed : int
        Random seed for reproducibility (default: 0).

    Returns
    -------
    np.ndarray
        Array of shape (n_samples, n_channels) with float32 values.
    """
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 10.0, n_samples, endpoint=False)

    signals = np.zeros((n_samples, n_channels), dtype=np.float32)

    for c in range(n_channels):
        # Base frequencies and phase offsets per channel
        f1 = 0.5 + 0.15 * (c + 1)
        f2 = 1.2 + 0.25 * ((c * 2) % 5)
        phi1 = (c * np.pi) / n_channels
        phi2 = (c * np.pi) / 3.0

        # Sinusoidal motion components with distinct channel offsets
        component1 = 1.0 * np.sin(2 * np.pi * f1 * t + phi1)
        component2 = 0.5 * np.cos(2 * np.pi * f2 * t + phi2)
        offset = 0.4 * (c - n_channels / 2.0)

        # Noise
        noise = rng.normal(0.0, noise_std, size=n_samples)

        signals[:, c] = (component1 + component2 + offset + noise).astype(np.float32)

    return signals


def inject_fault(
    data: np.ndarray,
    start: int,
    kind: str,
    channel: int = 0,
    drift_slope: float = 0.005,
) -> np.ndarray:
    """Inject a sensor fault into a specified channel starting at a given index.

    Parameters
    ----------
    data : np.ndarray
        2D array of sensor data of shape (n_samples, n_channels).
    start : int
        Sample index where the fault begins.
    kind : str
        Fault kind: "drift" (linearly increasing offset) or "stuck" (frozen value).
    channel : int
        Channel index to inject fault into (default: 0).
    drift_slope : float
        Slope per sample for "drift" fault (default: 0.005).

    Returns
    -------
    np.ndarray
        New array with the fault injected.
    """
    arr = np.array(data, copy=True)
    n_samples, n_channels = arr.shape

    if not (0 <= start < n_samples):
        raise ValueError(f"start index {start} out of bounds for {n_samples} samples")
    if not (0 <= channel < n_channels):
        raise ValueError(f"channel {channel} out of bounds for {n_channels} channels")

    kind_lower = kind.strip().lower()
    fault_len = n_samples - start

    if kind_lower == "drift":
        drift = np.arange(fault_len, dtype=arr.dtype) * drift_slope
        arr[start:, channel] += drift
    elif kind_lower == "stuck":
        stuck_value = arr[start, channel]
        arr[start:, channel] = stuck_value
    else:
        raise ValueError(f"Unknown fault kind '{kind}'. Supported: 'drift', 'stuck'")

    return arr


def load_csv(path: Union[str, Path]) -> np.ndarray:
    """Load sensor data from a CSV file into a 2D numpy array.

    Numeric columns are extracted as channels. Non-numeric columns
    (such as string timestamps or identifiers) and header rows are automatically ignored.

    Parameters
    ----------
    path : str or Path
        Path to the CSV file.

    Returns
    -------
    np.ndarray
        Array of shape (n_samples, n_channels) containing float32 sensor data.
    """
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"CSV file not found: {path_obj}")

    with open(path_obj, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        raw_rows = [row for row in reader if row]

    if not raw_rows:
        raise ValueError(f"CSV file {path_obj} is empty")

    # Detect which columns are numeric across all data rows
    # Try parsing each row; if first row has strings, it is likely a header
    def try_float(val: str) -> bool:
        try:
            float(val.strip())
            return True
        except ValueError:
            return False

    first_row = raw_rows[0]
    has_header = not all(try_float(val) for val in first_row)
    data_rows = raw_rows[1:] if has_header else raw_rows

    if not data_rows:
        raise ValueError(f"No data rows found in {path_obj}")

    num_cols = len(data_rows[0])
    # Identify columns that are numeric in all data rows
    numeric_cols = []
    for col_idx in range(num_cols):
        is_num = all(try_float(row[col_idx]) for row in data_rows if len(row) > col_idx)
        if is_num:
            numeric_cols.append(col_idx)

    if not numeric_cols:
        raise ValueError(f"No numeric columns found in CSV file: {path_obj}")

    parsed = []
    for row in data_rows:
        if len(row) >= max(numeric_cols) + 1:
            parsed.append([float(row[c].strip()) for c in numeric_cols])

    return np.array(parsed, dtype=np.float32)
