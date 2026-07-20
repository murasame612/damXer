#!/usr/bin/env python3
"""Build a mask-aware filtered-response target dataset for 2h dx forecasting.

This dataset turns the earlier diagnostic filters into a trainable downstream
target protocol. The filtered target is computed from the continuous SAITS-clean
series, while the downstream loss/metric mask is still defined by raw observed
cells. In other words, filled regions may provide continuity as model input, but
they are not treated as supervised truth.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, medfilt, savgol_filter

try:
    from aexp_events import metric as aexp_metric
    from aexp_events import note as aexp_note
    from aexp_events import param as aexp_param
    from aexp_events import progress as aexp_progress
except Exception:

    def aexp_metric(*args: Any, **kwargs: Any) -> None:
        return None

    def aexp_note(*args: Any, **kwargs: Any) -> None:
        return None

    def aexp_param(*args: Any, **kwargs: Any) -> None:
        return None

    def aexp_progress(*args: Any, **kwargs: Any) -> None:
        return None


def load_frame(path: Path, raw_time_col: str, date_col: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if raw_time_col in frame.columns:
        frame = frame.rename(columns={raw_time_col: "date"})
    elif date_col in frame.columns:
        frame = frame.rename(columns={date_col: "date"})
    elif "date" not in frame.columns:
        raise ValueError(f"{path} is missing a date column")
    return frame


def write_csv(path: Path, dates: pd.Series, values: pd.DataFrame) -> dict[str, Any]:
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


def odd_window(value: int, *, min_value: int = 3) -> int:
    value = max(value, min_value)
    return value if value % 2 == 1 else value + 1


def butter_lowpass(x: np.ndarray, cutoff_hours: float, step_hours: float, order: int) -> np.ndarray:
    fs = 1.0 / step_hours
    cutoff = 1.0 / cutoff_hours
    normal_cutoff = min(0.99, cutoff / (0.5 * fs))
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    padlen = min(3 * max(len(a), len(b)), max(1, len(x) - 2))
    return filtfilt(b, a, x, padlen=padlen)


def filter_one(
    x: np.ndarray,
    *,
    method: str,
    step_hours: float,
    sg_window: int,
    sg_polyorder: int,
    median_kernel: int,
    butter_cutoff_hours: float,
    butter_order: int,
) -> np.ndarray:
    if method == "none":
        return x.copy()
    if method == "sg":
        return savgol_filter(x, window_length=odd_window(sg_window), polyorder=sg_polyorder, mode="interp")
    if method == "butter":
        return butter_lowpass(x, cutoff_hours=butter_cutoff_hours, step_hours=step_hours, order=butter_order)
    if method == "median_sg":
        medianed = medfilt(x, kernel_size=odd_window(median_kernel))
        return savgol_filter(medianed, window_length=odd_window(sg_window), polyorder=sg_polyorder, mode="interp")
    raise ValueError(f"unknown filter method: {method}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-csv", required=True)
    parser.add_argument("--saits-clean-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--engineered-env-csv", default="")
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
    parser.add_argument("--method", choices=["sg", "butter", "median_sg", "none"], default="median_sg")
    parser.add_argument("--step-hours", type=float, default=2.0)
    parser.add_argument("--sg-window", type=int, default=9, help="Samples. 9 at 2h equals 18h.")
    parser.add_argument("--sg-polyorder", type=int, default=3)
    parser.add_argument("--median-kernel", type=int, default=5)
    parser.add_argument("--butter-cutoff-hours", type=float, default=16.0)
    parser.add_argument("--butter-order", type=int, default=3)
    args = parser.parse_args()

    raw_path = Path(args.raw_csv)
    clean_path = Path(args.saits_clean_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw = load_frame(raw_path, args.raw_time_col, args.date_col)
    clean = load_frame(clean_path, args.raw_time_col, args.date_col)
    if raw["date"].astype(str).tolist() != clean["date"].astype(str).tolist():
        raise ValueError("raw and SAITS-clean date columns differ")
    aligned_dates = raw["date"].astype(str).tolist()

    input_rows = len(raw)
    dates = pd.to_datetime(raw["date"], errors="raise")
    keep = pd.Series(True, index=raw.index)
    if args.date_start:
        keep &= dates >= pd.Timestamp(args.date_start)
    if args.date_end:
        keep &= dates < pd.Timestamp(args.date_end)
    if not keep.any():
        raise ValueError(
            f"date window [{args.date_start!r}, {args.date_end!r}) selected no rows"
        )

    raw_value_cols = [col for col in raw.columns if col != "date"]
    clean_value_cols = [col for col in clean.columns if col != "date"]
    target_cols = [col for col in raw_value_cols if col.startswith(args.target_prefix) and col in clean_value_cols]
    if not target_cols:
        raise ValueError(f"no target columns found for prefix {args.target_prefix!r}")

    aexp_note("build_filtered_response_dataset started")
    for name, value in {
        "raw_csv": str(raw_path.resolve()),
        "saits_clean_csv": str(clean_path.resolve()),
        "target_prefix": args.target_prefix,
        "target_dim": len(target_cols),
        "method": args.method,
        "sg_window": args.sg_window,
        "sg_polyorder": args.sg_polyorder,
        "median_kernel": args.median_kernel,
        "butter_cutoff_hours": args.butter_cutoff_hours,
        "butter_order": args.butter_order,
        "mask_policy": "raw_observed_only",
    }.items():
        aexp_param(name, value)

    target_raw = raw[target_cols].apply(pd.to_numeric, errors="coerce")
    target_clean = clean[target_cols].apply(pd.to_numeric, errors="coerce")
    filtered = pd.DataFrame(index=raw.index)
    residual = pd.DataFrame(index=raw.index)
    observed_mask = pd.DataFrame({"date": raw["date"].astype(str)})

    for idx, col in enumerate(target_cols, start=1):
        clean_values = target_clean[col].to_numpy(dtype=np.float64)
        if not np.isfinite(clean_values).all():
            clean_values = pd.Series(clean_values).interpolate(method="linear", limit_direction="both").ffill().bfill().to_numpy(
                dtype=np.float64
            )
        fitted = filter_one(
            clean_values,
            method=args.method,
            step_hours=args.step_hours,
            sg_window=args.sg_window,
            sg_polyorder=args.sg_polyorder,
            median_kernel=args.median_kernel,
            butter_cutoff_hours=args.butter_cutoff_hours,
            butter_order=args.butter_order,
        )
        raw_values = target_raw[col].to_numpy(dtype=np.float64)
        observed = np.isfinite(raw_values)
        filtered[col] = fitted
        residual[col] = np.where(observed, raw_values - fitted, np.nan)
        observed_mask[f"{col}_masked"] = (~observed).astype("int8")

        aexp_progress("channel", idx, total=len(target_cols), column=col)

    # Filtering is intentionally applied before the paper-date slice. The
    # bilateral target constructor therefore retains the same surrounding
    # context as the source experiment, including the final six target rows.
    raw = raw.loc[keep].reset_index(drop=True)
    clean = clean.loc[keep].reset_index(drop=True)
    target_raw = target_raw.loc[keep].reset_index(drop=True)
    target_clean = target_clean.loc[keep].reset_index(drop=True)
    filtered = filtered.loc[keep].reset_index(drop=True)
    residual = residual.loc[keep].reset_index(drop=True)
    observed_mask = observed_mask.loc[keep].reset_index(drop=True)

    channel_summary: list[dict[str, Any]] = []
    for col in target_cols:
        raw_values = target_raw[col].to_numpy(dtype=np.float64)
        observed = np.isfinite(raw_values)
        obs_res = residual[col].dropna().to_numpy(dtype=np.float64)
        raw_obs = raw_values[observed]
        channel_summary.append(
            {
                "column": col,
                "observed_count": int(observed.sum()),
                "observed_ratio": float(observed.mean()),
                "residual_mae_observed": float(np.mean(np.abs(obs_res))) if obs_res.size else None,
                "residual_rmse_observed": float(np.sqrt(np.mean(obs_res * obs_res))) if obs_res.size else None,
                "raw_std_observed": float(np.nanstd(raw_obs)) if raw_obs.size else None,
            }
        )

    mask_values = observed_mask.drop(columns=["date"]).to_numpy(dtype=np.float32)
    observed_values = 1.0 - mask_values
    residual_abs = residual.abs().to_numpy(dtype=np.float64)
    raw_missing = target_raw.isna()

    method_tag = args.method
    filtered_path = output_dir / "filtered_response.csv"
    residual_path = output_dir / "filtered_response_residual_observed.csv"
    mask_path = output_dir / "dam_2h_filtered_response_observed_mask.csv"
    dx_only_path = output_dir / f"dam_2h_filtered_response_{method_tag}_dx_only_nomask.csv"
    engineered_path = output_dir / f"dam_2h_filtered_response_{method_tag}_dx_engineered_env_nomask.csv"

    outputs: dict[str, Any] = {
        "filtered_response": write_csv(filtered_path, raw["date"], filtered),
        "filtered_response_residual_observed": write_csv(residual_path, raw["date"], residual),
        "filtered_response_dx_only_nomask": write_csv(dx_only_path, raw["date"], filtered),
    }
    observed_mask.to_csv(mask_path, index=False)

    engineered_env_columns: list[str] = []
    if args.engineered_env_csv:
        engineered_source = Path(args.engineered_env_csv)
        engineered = load_frame(engineered_source, args.raw_time_col, args.date_col)
        if aligned_dates != engineered["date"].astype(str).tolist():
            raise ValueError("raw and engineered-env date columns differ")
        engineered = engineered.loc[keep].reset_index(drop=True)
        engineered_value_cols = [col for col in engineered.columns if col != "date"]
        if len(engineered_value_cols) <= len(target_cols):
            raise ValueError("engineered-env CSV has no auxiliary columns after target_dim")
        engineered_env_columns = engineered_value_cols[len(target_cols) :]
        outputs["filtered_response_dx_engineered_env_nomask"] = write_csv(
            engineered_path,
            raw["date"],
            pd.concat(
                [
                    filtered.reset_index(drop=True),
                    engineered[engineered_env_columns].reset_index(drop=True),
                ],
                axis=1,
            ),
        )

    observed_ratio = float(observed_values.mean())
    missing_ratio = float(raw_missing.sum().sum() / raw_missing.size)
    summary = {
        "raw_csv": str(raw_path.resolve()),
        "saits_clean_csv": str(clean_path.resolve()),
        "engineered_env_csv": str(Path(args.engineered_env_csv).resolve()) if args.engineered_env_csv else None,
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
        "method": f"mask_aware_filtered_response_{args.method}",
        "filter_params": {
            "step_hours": args.step_hours,
            "sg_window": odd_window(args.sg_window),
            "sg_polyorder": args.sg_polyorder,
            "median_kernel": odd_window(args.median_kernel),
            "butter_cutoff_hours": args.butter_cutoff_hours,
            "butter_order": args.butter_order,
        },
        "mask_csv": str(mask_path.resolve()),
        "mask_policy": "1 means raw target missing; do not compute supervised loss or metric there",
        "engineered_env_columns": engineered_env_columns,
        "rows": int(len(raw)),
        "outputs": outputs,
        "missing_summary": {
            "raw_target_missing_cells": int(raw_missing.sum().sum()),
            "raw_target_total_cells": int(raw_missing.size),
            "raw_target_missing_ratio": missing_ratio,
            "observed_supervised_cells": int(observed_values.sum()),
            "observed_total_cells": int(observed_values.size),
            "observed_supervised_ratio": observed_ratio,
        },
        "residual_summary": {
            "observed_residual_rmse": float(np.sqrt(np.nanmean(residual_abs ** 2))),
            "observed_residual_mae": float(np.nanmean(residual_abs)),
        },
        "channel_summary": channel_summary,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    aexp_metric("filtered_response/raw_missing_ratio", missing_ratio)
    aexp_metric("filtered_response/observed_supervised_ratio", observed_ratio)
    aexp_metric("filtered_response/residual_rmse_observed", summary["residual_summary"]["observed_residual_rmse"])
    aexp_note(f"filtered response dataset written to {output_dir.resolve()}")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
