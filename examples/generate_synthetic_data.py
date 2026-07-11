#!/usr/bin/env python3
"""Generate a small, fully synthetic monitoring table for pipeline smoke tests."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def generate(rows: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=rows, freq="2h")
    t = np.arange(rows, dtype=np.float64)
    water = 80.0 + 1.4 * np.sin(2.0 * np.pi * t / 84.0)
    seepage = 32.0 + 0.25 * np.roll(water, 6) + 0.15 * np.sin(2.0 * np.pi * t / 24.0)
    temperature = 23.0 + 4.0 * np.sin(2.0 * np.pi * t / 168.0)

    clean = pd.DataFrame({"date": dates.astype(str)})
    clean["H_001"] = water + rng.normal(0.0, 0.03, rows)
    clean["H_002"] = water + 0.2 + rng.normal(0.0, 0.04, rows)
    clean["seeP_001"] = seepage + rng.normal(0.0, 0.04, rows)
    clean["seeP_002"] = seepage - 0.1 + rng.normal(0.0, 0.05, rows)
    clean["temp_001"] = temperature + rng.normal(0.0, 0.08, rows)
    clean["temp_002"] = temperature + 0.3 + rng.normal(0.0, 0.08, rows)
    for index, phase in enumerate((0.0, 0.8, 1.6), start=1):
        response = (
            0.04 * np.roll(water, 12)
            + 0.025 * np.roll(seepage, 3)
            + 0.012 * np.roll(temperature, 24)
            + 0.25 * np.sin(2.0 * np.pi * t / 48.0 + phase)
        )
        clean[f"dx_{index:03d}"] = response + rng.normal(0.0, 0.025, rows)

    raw = clean.copy()
    value_columns = [column for column in raw.columns if column != "date"]
    missing = rng.random((rows, len(value_columns))) < 0.08
    missing[70:82, value_columns.index("H_001")] = True
    missing[130:146, value_columns.index("dx_002")] = True
    raw.loc[:, value_columns] = raw[value_columns].mask(missing)
    return raw, clean


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="artifacts/synthetic_input")
    parser.add_argument("--rows", type=int, default=256)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw, clean = generate(args.rows, args.seed)
    raw.to_csv(output_dir / "raw_monitoring.csv", index=False)
    clean.to_csv(output_dir / "saits_clean.csv", index=False)
    print(f"wrote {len(raw)} synthetic rows to {output_dir}")


if __name__ == "__main__":
    main()
