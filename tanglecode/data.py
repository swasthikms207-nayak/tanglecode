from __future__ import annotations

from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd


class CSVResult(tuple):
    """2-tuple of (data, warning) that delegates attribute access to the data array."""

    def __new__(cls, data: np.ndarray, warning: str | None = None):
        return super().__new__(cls, (data, warning))

    @property
    def data(self) -> np.ndarray:
        return self[0]

    @property
    def warning(self) -> str | None:
        return self[1]

    def __getattr__(self, name: str) -> Any:
        return getattr(self[0], name)


def generate_signals(
    kind: str,
    n: int = 6,
    T: int = 3000,
    seed: int = 7,
) -> np.ndarray:
    """Generate multichannel synthetic signals according to the specified recipe.

    Recipe:
      t = arange(T)
      base = 0.8*sin(2*pi*t/700) + 0.4*sin(2*pi*t/230)
      channel i (0-based): gain = 1 + 0.05*(i-2.5);
        offset = (i-2.5)*0.12; value = gain*base + offset + normal(0, 0.06)
      using numpy default_rng(seed).

    kind "clean": as above.
    kind "drift": after tick T//2, add np.linspace(0, 2.0, T - T//2) to
      channel index 3 (shown as Sensor 4).
    kind "stuck": after tick T//2, channel index 3 is held at its value at
      tick T//2.
    """
    t = np.arange(T)
    base = 0.8 * np.sin(2 * np.pi * t / 700.0) + 0.4 * np.sin(2 * np.pi * t / 230.0)
    rng = np.random.default_rng(seed)

    signals = np.zeros((T, n), dtype=float)
    for i in range(n):
        gain = 1.0 + 0.05 * (i - 2.5)
        offset = (i - 2.5) * 0.12
        noise = rng.normal(0.0, 0.06, size=T)
        signals[:, i] = gain * base + offset + noise

    kind_norm = kind.strip().lower()
    mid = T // 2

    if kind_norm == "clean":
        pass
    elif kind_norm == "drift":
        if n <= 3:
            raise ValueError(f"Drift fault requires at least 4 channels (channel index 3), got {n}")
        signals[mid:, 3] += np.linspace(0.0, 2.0, T - mid)
    elif kind_norm == "stuck":
        if n <= 3:
            raise ValueError(f"Stuck fault requires at least 4 channels (channel index 3), got {n}")
        signals[mid:, 3] = signals[mid, 3]
    else:
        raise ValueError(f"Unknown signal kind '{kind}'. Expected 'clean', 'drift', or 'stuck'.")

    return signals


def load_csv(file: Any) -> tuple[np.ndarray, str | None]:
    """Load sensor data from CSV.

    Keeps numeric columns only, drops rows with NaN, uses at most the first 8 columns
    (returning a warning string if more were present), and raises ValueError with a
    plain-English message if fewer than 2 numeric columns or fewer than 300 rows are present.
    """
    df = pd.read_csv(file)

    # Convert columns to numeric if possible (handling string representations of numbers)
    numeric_cols = []
    for col in df.columns:
        s = pd.to_numeric(df[col], errors="coerce")
        if not s.isna().all():
            df[col] = s
            numeric_cols.append(col)

    num_df = df[numeric_cols].select_dtypes(include=[np.number])

    # Drop rows with NaN
    num_df = num_df.dropna()

    # Validate column count
    num_cols = num_df.shape[1]
    if num_cols < 2:
        raise ValueError(
            f"CSV file must have at least 2 numeric columns, but only found {num_cols}."
        )

    # Validate row count
    num_rows = num_df.shape[0]
    if num_rows < 300:
        raise ValueError(
            f"CSV file must contain at least 300 rows after dropping NaN values, but only found {num_rows}."
        )

    warning = None
    if num_cols > 8:
        warning = f"CSV file has {num_cols} numeric columns; using only the first 8 columns."
        num_df = num_df.iloc[:, :8]

    data = num_df.to_numpy(dtype=float)
    return CSVResult(data, warning)


if __name__ == "__main__":
    data_dir = Path(__file__).resolve().parent.parent / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    columns = [f"sensor_{i}" for i in range(1, 7)]

    # 1. Clean
    clean_data = generate_signals("clean", n=6, T=3000, seed=7)
    df_clean = pd.DataFrame(clean_data, columns=columns).round(4)
    clean_path = data_dir / "test_clean.csv"
    df_clean.to_csv(clean_path, index=False)
    print(f"Wrote {clean_path} (shape: {clean_data.shape})")

    # 2. Drift (test_fault.csv)
    drift_data = generate_signals("drift", n=6, T=3000, seed=7)
    df_drift = pd.DataFrame(drift_data, columns=columns).round(4)
    drift_path = data_dir / "test_fault.csv"
    df_drift.to_csv(drift_path, index=False)
    print(f"Wrote {drift_path} (shape: {drift_data.shape})")

    # 3. Stuck (test_stuck.csv)
    stuck_data = generate_signals("stuck", n=6, T=3000, seed=7)
    df_stuck = pd.DataFrame(stuck_data, columns=columns).round(4)
    stuck_path = data_dir / "test_stuck.csv"
    df_stuck.to_csv(stuck_path, index=False)
    print(f"Wrote {stuck_path} (shape: {stuck_data.shape})")
