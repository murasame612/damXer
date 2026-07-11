#!/usr/bin/env python3
"""Run the frozen DamXer paper configuration over multiple random seeds."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
from pathlib import Path

import pandas as pd

from train_damxer import build_token_specs


PAPER_SEEDS = (2021, 2022, 2023, 2024, 2025)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--target-csv", required=True)
    parser.add_argument("--mask-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dx-variant", default="dam_2h_saits_dx_only_nomask.csv")
    parser.add_argument("--engineered-variant", default="dam_2h_saits_dx_engineered_env_nomask.csv")
    parser.add_argument("--variant", choices=["full", "no_lag_env", "no_env"], default="full")
    parser.add_argument("--seeds", default=",".join(map(str, PAPER_SEEDS)))
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--date-start", default="")
    parser.add_argument("--date-end", default="")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--strict-paper-shape", action="store_true")
    return parser.parse_args()


def load_and_validate(args: argparse.Namespace) -> dict:
    data_root = Path(args.data_root)
    paths = {
        "dx": data_root / args.dx_variant,
        "engineered": data_root / args.engineered_variant,
        "target": Path(args.target_csv),
        "mask": Path(args.mask_csv),
    }
    missing_files = [str(path) for path in paths.values() if not path.is_file()]
    if missing_files:
        raise FileNotFoundError("missing required input files: " + ", ".join(missing_files))

    frames = {name: pd.read_csv(path) for name, path in paths.items()}
    for name, frame in frames.items():
        if "date" not in frame.columns:
            raise ValueError(f"{name} input is missing the required date column")
    dates = frames["dx"]["date"].astype(str).tolist()
    for name in ("engineered", "target", "mask"):
        if frames[name]["date"].astype(str).tolist() != dates:
            raise ValueError(f"{name} dates do not match the dx input")

    dx_columns = [column for column in frames["dx"].columns if column != "date"]
    engineered_columns = [column for column in frames["engineered"].columns if column != "date"]
    if engineered_columns[: len(dx_columns)] != dx_columns:
        raise ValueError("the engineered input must begin with the dx columns in identical order")
    missing_target = [column for column in dx_columns if column not in frames["target"].columns]
    if missing_target:
        raise ValueError(f"target input is missing dx columns: {missing_target[:5]}")
    missing_mask = [f"{column}_masked" for column in dx_columns if f"{column}_masked" not in frames["mask"].columns]
    if missing_mask:
        raise ValueError(f"mask input is missing columns: {missing_mask[:5]}")

    n_rows = len(frames["dx"])
    n_train = int(n_rows * 0.7)
    n_test = int(n_rows * 0.2)
    n_val = n_rows - n_train - n_test
    split_windows = {
        "train": max(0, n_train - 720 - 96 + 1),
        "val": max(0, n_val - 96 + 1),
        "test": max(0, n_test - 96 + 1),
    }
    env_columns = engineered_columns[len(dx_columns) :]
    token_count = len(build_token_specs(env_columns, "lag"))
    report = {
        "rows": n_rows,
        "first_date": dates[0] if dates else None,
        "last_date": dates[-1] if dates else None,
        "target_dim": len(dx_columns),
        "engineered_feature_count": len(env_columns),
        "lag_token_count": token_count,
        "split_rows": {"train": n_train, "val": n_val, "test": n_test},
        "split_windows": split_windows,
    }
    if args.strict_paper_shape:
        expected = {
            "rows": 8400,
            "target_dim": 89,
            "lag_token_count": 60,
            "split_windows": {"train": 5065, "val": 745, "test": 1585},
        }
        mismatches = {key: (report[key], value) for key, value in expected.items() if report[key] != value}
        if mismatches:
            raise ValueError(f"inputs do not match the paper protocol: {mismatches}")
    return report


def summarize(values: list[float]) -> dict:
    return {
        "mean": statistics.mean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "n": len(values),
    }


def main() -> None:
    args = parse_args()
    protocol = load_and_validate(args)
    if args.check_only:
        print(json.dumps(protocol, indent=2))
        return

    seeds = [int(item) for item in args.seeds.replace(" ", ",").split(",") if item]
    if not seeds:
        raise ValueError("--seeds must contain at least one integer")
    output_dir = Path(args.output_dir)
    seed_dir = output_dir / "seeds"
    log_dir = output_dir / "logs"
    seed_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    env_mode = "none" if args.variant == "no_env" else "full"
    env_token_mode = "no_lag" if args.variant == "no_lag_env" else "lag"
    train_script = Path(__file__).with_name("train_damxer.py")
    rows = []
    for seed in seeds:
        result_path = seed_dir / f"seed_{seed}.json"
        log_path = log_dir / f"seed_{seed}.log"
        command = [
            sys.executable,
            str(train_script),
            "--data-root", args.data_root,
            "--dx-variant", args.dx_variant,
            "--engineered-variant", args.engineered_variant,
            "--target-csv", args.target_csv,
            "--mask-csv", args.mask_csv,
            "--output", str(result_path),
            "--dx-seq-len", "192",
            "--env-seq-len", "720",
            "--pred-len", "96",
            "--target-dim", "89",
            "--patch-len", "16",
            "--patch-stride", "16",
            "--hidden", "160",
            "--n-heads", "8",
            "--e-layers", "2",
            "--dropout", "0.2",
            "--epochs", str(args.epochs),
            "--patience", str(args.patience),
            "--min-delta", "1e-5",
            "--batch-size", "8",
            "--num-workers", str(args.num_workers),
            "--lr", "0.0001327691239087578",
            "--weight-decay", "0.00014163979315531797",
            "--loss-type", "huber",
            "--huber-delta", "0.2",
            "--predict-mode", "direct",
            "--env-mode", env_mode,
            "--env-token-mode", env_token_mode,
            "--zero-head-init",
            "--revin",
            "--revin-eps", "1e-5",
            "--gpu", str(args.gpu),
            "--seed", str(seed),
            "--trial-name", f"paper_{args.variant}_seed{seed}",
        ]
        if args.date_start:
            command.extend(["--date-start", args.date_start])
        if args.date_end:
            command.extend(["--date-end", args.date_end])
        print("+ " + " ".join(command), flush=True)
        process = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        log_path.write_text(process.stdout, encoding="utf-8")
        print(process.stdout, end="", flush=True)
        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, command)
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "seed": seed,
                "best_epoch": payload["best_epoch"],
                "val_mse": payload["val"]["mse"],
                "val_mae": payload["val"]["mae"],
                "val_rmse": payload["val"]["rmse"],
                "test_mse": payload["test"]["mse"],
                "test_mae": payload["test"]["mae"],
                "test_rmse": payload["test"]["rmse"],
                "result_json": str(result_path),
            }
        )

    summary_csv = output_dir / "summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "experiment": f"DamXer paper configuration: {args.variant}",
        "selection": "validation observed MSE; test evaluated after restoring the validation-best model state",
        "protocol": protocol,
        "seeds": seeds,
        "aggregates": {
            key: summarize([float(row[key]) for row in rows])
            for key in ("val_mse", "val_mae", "val_rmse", "test_mse", "test_mae", "test_rmse")
        },
        "rows": rows,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
