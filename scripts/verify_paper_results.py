#!/usr/bin/env python3
"""Verify reproduced forecasting aggregates against the frozen paper metrics."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

from paper_protocol import DEFAULT_CONFIG, load_config


DEFAULT_SEED_LEDGER = Path(__file__).resolve().parents[1] / "results" / "paper_seed_metrics.csv"


DAMXER_NAMES = {
    "full": "DamXer",
    "reduced": "DamXer reduced-lag ENV",
    "no_env": "DamXer displacement only",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--damxer-full", required=True)
    parser.add_argument("--damxer-reduced", required=True)
    parser.add_argument("--damxer-no-env", required=True)
    parser.add_argument("--dx-baselines", required=True)
    parser.add_argument("--raw-env", required=True)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--seed-ledger", default=str(DEFAULT_SEED_LEDGER))
    parser.add_argument("--atol", type=float, default=1e-9)
    parser.add_argument("--output", default="")
    return parser.parse_args()


def load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def normalize_stat(stat: dict) -> dict:
    return {"mean": float(stat["mean"]), "std": float(stat["std"]), "n": int(stat.get("n", 5))}


def damxer_stats(path: str | Path, expected_seeds: tuple[int, ...] = (2021, 2022, 2023, 2024, 2025)) -> dict:
    source = Path(path)
    if source.is_dir():
        seed_files = sorted(source.glob("seed_*.json"))
        if not seed_files:
            raise ValueError(f"DamXer result directory has no seed_*.json files: {source}")
        values = {"val": [], "test": []}
        selected: dict[int, Path] = {}
        for seed_file in seed_files:
            payload = load_json(seed_file)
            seed = int(payload.get("seed", seed_file.stem.removeprefix("seed_")))
            if seed not in expected_seeds:
                continue
            if seed in selected:
                raise ValueError(f"duplicate DamXer result for seed {seed}: {seed_file}")
            selected[seed] = seed_file
            values["val"].append(float(payload["val"]["mse"]))
            values["test"].append(float(payload["test"]["mse"]))
        missing = sorted(set(expected_seeds) - set(selected))
        if missing:
            raise ValueError(f"DamXer result directory is missing frozen seeds: {missing}")
        return {
            split: {
                "mean": statistics.mean(split_values),
                "std": statistics.stdev(split_values) if len(split_values) > 1 else 0.0,
                "n": len(split_values),
            }
            for split, split_values in values.items()
        }

    payload = load_json(source)
    aggregate = payload["aggregates"]
    val = aggregate.get("val_mse") or aggregate.get("val_observed_mse")
    test = aggregate.get("test_mse") or aggregate.get("test_observed_mse")
    if val is None or test is None:
        raise ValueError(f"DamXer summary lacks val/test MSE aggregates: {path}")
    return {"val": normalize_stat(val), "test": normalize_stat(test)}


def baseline_stats(path: str | Path) -> dict[str, dict]:
    payload = load_json(path)
    out = {}
    for model, metrics in payload.items():
        out[model] = {
            "val": normalize_stat(metrics["val_observed_mse"]),
            "test": normalize_stat(metrics["test_observed_mse"]),
        }
    return out


def percent_reduction(proposed: float, baseline: float) -> float:
    return 100.0 * (baseline - proposed) / baseline


def percent_degradation(variant: float, full: float) -> float:
    return 100.0 * (variant - full) / full


def seed_ledger_stats(path: str | Path) -> dict[str, dict]:
    grouped: dict[str, dict[str, list]] = {}
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            bucket = grouped.setdefault(row["variant"], {"val": [], "test": [], "seeds": []})
            bucket["val"].append(float(row["val_mse"]))
            bucket["test"].append(float(row["test_mse"]))
            bucket["seeds"].append(int(row["seed"]))
    return {
        name: {
            **{
                split: {
                    "mean": statistics.mean(values),
                    "std": statistics.stdev(values) if len(values) > 1 else 0.0,
                    "n": len(values),
                }
                for split, values in splits.items()
                if split in ("val", "test")
            },
            "seeds": sorted(splits["seeds"]),
        }
        for name, splits in grouped.items()
    }


def main() -> None:
    args = parse_args()
    _, config = load_config(args.config)
    actual = {
        DAMXER_NAMES["full"]: damxer_stats(args.damxer_full),
        DAMXER_NAMES["reduced"]: damxer_stats(args.damxer_reduced),
        DAMXER_NAMES["no_env"]: damxer_stats(args.damxer_no_env),
    }
    dx = baseline_stats(args.dx_baselines)
    actual.update(
        {
            "PatchTST": dx["PatchTST"],
            "iTransformer (dx only)": dx["iTransformer"],
            "DLinear": dx["DLinear"],
            "TimesNet": dx["TimesNet"],
            "FEDformer": dx["FEDformer"],
        }
    )
    raw = baseline_stats(args.raw_env)
    actual["iTransformer+raw ENV (720)"] = raw["iTransformer"]

    checks = []
    failed = []
    ledger = seed_ledger_stats(args.seed_ledger)
    for name, expected in config["expected_metrics"].items():
        row = actual[name]
        for split in ("val", "test"):
            for field in ("mean", "std"):
                expected_value = float(expected[f"{split}_mse_{field}"])
                actual_value = float(row[split][field])
                error = abs(actual_value - expected_value)
                check = {
                    "name": name,
                    "metric": f"{split}_mse_{field}",
                    "actual": actual_value,
                    "expected": expected_value,
                    "abs_error": error,
                    "ok": error <= args.atol,
                }
                checks.append(check)
                if not check["ok"]:
                    failed.append(check)
                ledger_value = float(ledger[name][split][field])
                ledger_error = abs(ledger_value - expected_value)
                ledger_check = {
                    "name": name,
                    "metric": f"seed_ledger_{split}_mse_{field}",
                    "actual": ledger_value,
                    "expected": expected_value,
                    "abs_error": ledger_error,
                    "ok": ledger_error <= args.atol,
                }
                checks.append(ledger_check)
                if not ledger_check["ok"]:
                    failed.append(ledger_check)
        for split in ("val", "test"):
            if row[split]["n"] != 5:
                failed.append(
                    {"name": name, "metric": f"{split}_seed_count", "actual": row[split]["n"], "expected": 5}
                )
            if ledger[name][split]["n"] != 5:
                failed.append(
                    {
                        "name": name,
                        "metric": f"seed_ledger_{split}_count",
                        "actual": ledger[name][split]["n"],
                        "expected": 5,
                    }
                )
        expected_seeds = [2021, 2022, 2023, 2024, 2025]
        if ledger[name]["seeds"] != expected_seeds:
            failed.append(
                {
                    "name": name,
                    "metric": "seed_ledger_ids",
                    "actual": ledger[name]["seeds"],
                    "expected": expected_seeds,
                }
            )

    full = actual["DamXer"]["test"]["mean"]
    claims = {
        "derived_rmse": math.sqrt(full),
        "reduction_vs_patchtst_pct": percent_reduction(full, actual["PatchTST"]["test"]["mean"]),
        "reduction_vs_itransformer_dx_pct": percent_reduction(full, actual["iTransformer (dx only)"]["test"]["mean"]),
        "reduction_vs_itransformer_raw_env_pct": percent_reduction(full, actual["iTransformer+raw ENV (720)"]["test"]["mean"]),
        "reduction_vs_dlinear_pct": percent_reduction(full, actual["DLinear"]["test"]["mean"]),
        "reduced_lag_degradation_pct": percent_degradation(actual["DamXer reduced-lag ENV"]["test"]["mean"], full),
        "no_env_degradation_pct": percent_degradation(actual["DamXer displacement only"]["test"]["mean"], full),
    }
    report = {
        "status": "pass" if not failed else "fail",
        "absolute_tolerance": args.atol,
        "actual": actual,
        "seed_ledger": {"path": str(Path(args.seed_ledger).resolve()), "aggregates": ledger},
        "checks": checks,
        "failed": failed,
        "paper_claims": claims,
    }
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
