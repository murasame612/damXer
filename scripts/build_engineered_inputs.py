#!/usr/bin/env python3
"""Build the three input tables used by the DamXer paper experiments.

The output CSVs are model inputs only. Original target missingness is written to
a separate mask CSV for observed-only loss/metrics and is never appended as an
input feature.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


FAMILY_PATTERNS = {
    "H": re.compile(r"^(H|h)[_A-Za-z0-9-]"),
    "seep": re.compile(r"^(seeP|seep|SEEP|P|p)[_A-Za-z0-9-]"),
    "temp": re.compile(r"^(temp|TEMP|T|t)[_A-Za-z0-9-]"),
}


def classify_family(col: str, target_prefix: str) -> str | None:
    if col.startswith(target_prefix):
        return None
    for family, pattern in FAMILY_PATTERNS.items():
        if pattern.match(col):
            return family
    lower = col.lower()
    if "seep" in lower or "pore" in lower:
        return "seep"
    if "temp" in lower or "temperature" in lower:
        return "temp"
    if lower.startswith("h_") or lower.startswith("h-") or lower.startswith("h"):
        return "H"
    return None


def load_frame(path: Path, time_col: str, date_col: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if time_col in frame.columns:
        frame = frame.rename(columns={time_col: "date"})
    elif date_col in frame.columns:
        frame = frame.rename(columns={date_col: "date"})
    elif "date" not in frame.columns:
        raise ValueError(f"{path} is missing a date column")
    return frame


def numeric_fill(frame: pd.DataFrame, value_cols: list[str]) -> pd.DataFrame:
    values = frame[value_cols].apply(pd.to_numeric, errors="coerce")
    values = values.interpolate(method="linear", limit_direction="both")
    values = values.ffill().bfill()
    return values.fillna(0.0)


def slope(values: pd.Series, window: int) -> pd.Series:
    return (values - values.shift(window)) / float(window)


def aggregate_family(values: pd.DataFrame, cols: list[str], family: str, feature_profile: str = "base") -> pd.DataFrame:
    out = pd.DataFrame(index=values.index)
    if not cols:
        return out
    block = values[cols].astype(float)
    mean = block.mean(axis=1)
    out[f"{family}_mean"] = mean
    out[f"{family}_std"] = block.std(axis=1).fillna(0.0)
    out[f"{family}_min"] = block.min(axis=1)
    out[f"{family}_max"] = block.max(axis=1)
    out[f"{family}_range"] = out[f"{family}_max"] - out[f"{family}_min"]
    lags = (1, 3, 6, 12, 24, 48, 84)
    windows = (12, 84)
    if feature_profile == "rich":
        lags = (1, 3, 6, 12, 24, 48, 84, 168, 360)
        windows = (12, 24, 84, 168, 360, 720)
    elif feature_profile != "base":
        raise ValueError(f"unknown feature_profile={feature_profile!r}; expected 'base' or 'rich'")
    for lag in lags:
        out[f"{family}_mean_lag{lag}"] = mean.shift(lag)
        out[f"{family}_mean_delta_lag{lag}"] = mean - mean.shift(lag)
    for window in windows:
        out[f"{family}_mean_rollmean{window}"] = mean.rolling(window, min_periods=1).mean()
        out[f"{family}_mean_rollstd{window}"] = mean.rolling(window, min_periods=2).std()
        out[f"{family}_mean_slope{window}"] = slope(mean, window)
        if feature_profile == "rich":
            roll_min = mean.rolling(window, min_periods=1).min()
            roll_max = mean.rolling(window, min_periods=1).max()
            out[f"{family}_mean_rollrange{window}"] = roll_max - roll_min
            delta = mean.diff().fillna(0.0)
            out[f"{family}_mean_pos_delta_sum{window}"] = delta.clip(lower=0.0).rolling(window, min_periods=1).sum()
            out[f"{family}_mean_neg_delta_sum{window}"] = delta.clip(upper=0.0).rolling(window, min_periods=1).sum()
    return out


def build_time_features(dates: pd.Series) -> pd.DataFrame:
    dt = pd.to_datetime(dates)
    hour_angle = 2.0 * np.pi * dt.dt.hour.to_numpy(dtype=float) / 24.0
    doy_angle = 2.0 * np.pi * dt.dt.dayofyear.to_numpy(dtype=float) / 366.0
    return pd.DataFrame(
        {
            "hour_sin": np.sin(hour_angle),
            "hour_cos": np.cos(hour_angle),
            "doy_sin": np.sin(doy_angle),
            "doy_cos": np.cos(doy_angle),
        }
    )


def build_engineered_env(
    clean_values: pd.DataFrame,
    value_cols: list[str],
    target_prefix: str,
    feature_profile: str = "base",
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    family_cols: dict[str, list[str]] = {"H": [], "seep": [], "temp": []}
    ignored: list[str] = []
    for col in value_cols:
        family = classify_family(col, target_prefix)
        if family is None:
            ignored.append(col)
        else:
            family_cols[family].append(col)
    blocks = [aggregate_family(clean_values, family_cols[name], name, feature_profile) for name in ("H", "seep", "temp")]
    engineered = pd.concat([block for block in blocks if not block.empty], axis=1)
    engineered = engineered.replace([np.inf, -np.inf], np.nan)
    if feature_profile == "rich":
        engineered = engineered.ffill().fillna(0.0)
    else:
        engineered = engineered.ffill().bfill().fillna(0.0)
    return engineered, {**family_cols, "ignored_or_target": ignored}


def write_csv(path: Path, dates: pd.Series, values: pd.DataFrame) -> dict:
    out = pd.concat([pd.DataFrame({"date": dates.astype(str)}), values.reset_index(drop=True)], axis=1)
    out.to_csv(path, index=False)
    return {
        "path": str(path.resolve()),
        "rows": int(len(out)),
        "columns": int(len(out.columns)),
        "feature_columns": int(len(out.columns) - 1),
        "nan_count": int(out.drop(columns=["date"]).isna().sum().sum()),
        "first_value_columns": out.columns[1:8].tolist(),
        "last_columns": out.columns[-8:].tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-csv", required=True)
    parser.add_argument("--saits-clean-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--raw-time-col", default="采集时间")
    parser.add_argument("--date-col", default="date")
    parser.add_argument("--target-prefix", default="dx")
    parser.add_argument(
        "--date-start",
        default="",
        help="Optional inclusive timestamp bound applied after raw/clean alignment.",
    )
    parser.add_argument(
        "--date-end",
        default="",
        help="Optional exclusive timestamp bound applied after raw/clean alignment.",
    )
    parser.add_argument(
        "--feature-profile",
        choices=["base", "rich"],
        default="base",
        help="base preserves the original engineered ENV feature set; rich adds longer lags and multi-scale rolling statistics.",
    )
    args = parser.parse_args()

    raw_path = Path(args.raw_csv)
    clean_path = Path(args.saits_clean_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw = load_frame(raw_path, args.raw_time_col, args.date_col)
    clean = load_frame(clean_path, args.raw_time_col, args.date_col)
    if raw["date"].astype(str).tolist() != clean["date"].astype(str).tolist():
        raise ValueError("raw and SAITS clean date columns differ")

    input_rows = len(raw)
    dates = pd.to_datetime(raw["date"], errors="raise")
    keep = pd.Series(True, index=raw.index)
    if args.date_start:
        keep &= dates >= pd.Timestamp(args.date_start)
    if args.date_end:
        keep &= dates < pd.Timestamp(args.date_end)
    raw = raw.loc[keep].reset_index(drop=True)
    clean = clean.loc[keep].reset_index(drop=True)
    if raw.empty:
        raise ValueError(
            f"date window [{args.date_start!r}, {args.date_end!r}) selected no rows"
        )

    raw_value_cols = [col for col in raw.columns if col != "date"]
    clean_value_cols = [col for col in clean.columns if col != "date"]
    missing = [col for col in raw_value_cols if col not in clean_value_cols]
    if missing:
        raise ValueError(f"SAITS clean CSV is missing value columns: {missing[:8]}")

    target_cols = [col for col in raw_value_cols if col.startswith(args.target_prefix)]
    if not target_cols:
        raise ValueError(f"no target columns found for prefix {args.target_prefix!r}")

    raw_numeric = raw[raw_value_cols].apply(pd.to_numeric, errors="coerce")
    original_missing = raw_numeric.isna()
    clean_values = numeric_fill(clean, raw_value_cols)
    target_values = clean_values[target_cols]
    engineered_env, family_cols = build_engineered_env(clean_values, raw_value_cols, args.target_prefix, args.feature_profile)
    raw_env_cols = [*family_cols["H"], *family_cols["seep"], *family_cols["temp"]]
    raw_env_values = clean_values[raw_env_cols]
    time_features = build_time_features(raw["date"])

    mask_csv = output_dir / "dam_2h_target_observed_mask.csv"
    mask_out = pd.DataFrame({"date": raw["date"].astype(str)})
    for col in target_cols:
        mask_out[f"{col}_masked"] = original_missing[col].astype("int8")
    mask_out.to_csv(mask_csv, index=False)

    outputs = {
        "dx_only_nomask": write_csv(
            output_dir / "dam_2h_saits_dx_only_nomask.csv",
            raw["date"],
            target_values,
        ),
        "dx_engineered_env_nomask": write_csv(
            output_dir / "dam_2h_saits_dx_engineered_env_nomask.csv",
            raw["date"],
            pd.concat([target_values, engineered_env, time_features], axis=1),
        ),
        "dx_raw_env_nomask": write_csv(
            output_dir / "dam_2h_saits_dx_raw_env_nomask.csv",
            raw["date"],
            pd.concat([target_values, raw_env_values], axis=1),
        ),
    }
    manifest = {
        "raw_csv": str(raw_path.resolve()),
        "saits_clean_csv": str(clean_path.resolve()),
        "output_dir": str(output_dir.resolve()),
        "date_column": "date",
        "date_window": {
            "start_inclusive": args.date_start or None,
            "end_exclusive": args.date_end or None,
            "first_date": str(raw["date"].iloc[0]),
            "last_date": str(raw["date"].iloc[-1]),
            "input_rows": int(input_rows),
            "selected_rows": int(len(raw)),
        },
        "target_prefix": args.target_prefix,
        "target_columns": target_cols,
        "target_dim": len(target_cols),
        "mask_csv": str(mask_csv.resolve()),
        "mask_policy": "target missingness is evaluation metadata only; no mask/input_missing columns are model features",
        "feature_policy": "causal lagged and rolling environment aggregates plus time seasonality",
        "feature_profile": args.feature_profile,
        "family_columns": family_cols,
        "raw_environment_columns": raw_env_cols,
        "raw_environment_policy": (
            "completed original H/seeP/temp sensor channels; no lag, rolling, slope, "
            "family aggregation, or explicit calendar features"
        ),
        "engineered_env_columns": engineered_env.columns.tolist(),
        "time_feature_columns": time_features.columns.tolist(),
        "rows": int(len(raw)),
        "outputs": outputs,
        "missing_summary": {
            "target_missing_cells": int(original_missing[target_cols].sum().sum()),
            "target_total_cells": int(original_missing[target_cols].size),
            "target_missing_ratio": float(original_missing[target_cols].sum().sum() / original_missing[target_cols].size),
            "all_value_missing_cells": int(original_missing.sum().sum()),
            "all_value_total_cells": int(original_missing.size),
            "all_value_missing_ratio": float(original_missing.sum().sum() / original_missing.size),
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
