#!/usr/bin/env python3
"""One entry point for checking or rerunning the frozen DamXer paper protocol."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import subprocess
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs" / "paper_clean_window.json"
MANIFEST = REPO / "release" / "benchmark_manifest.json"
DATA = REPO / "data" / "paper"
INPUTS = DATA / "inputs"
TARGETS = DATA / "targets"
TARGET = TARGETS / "filtered_response.csv"
MASK = TARGETS / "dam_2h_filtered_response_observed_mask.csv"
LEDGER = REPO / "results" / "paper_seed_metrics.csv"
ARTIFACTS = REPO / "artifacts" / "reproduction"
FROZEN_SEEDS = [2021, 2022, 2023, 2024, 2025]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=["check", "damxer", "paper"],
        default="check",
        help=(
            "check validates released data and recomputes the published table from the seed ledger; "
            "damxer retrains the three DamXer variants; paper retrains all nine settings"
        ),
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seeds", default="2021,2022,2023,2024,2025")
    parser.add_argument("--tslib-root", default=str(REPO / "external" / "Time-Series-Library"))
    parser.add_argument("--atol", type=float, default=0.005, help="aggregate MSE tolerance for a fresh rerun")
    parser.add_argument("--allow-dirty-code", action="store_true")
    parser.add_argument("--json", action="store_true", help="also print the complete integrity report")
    return parser.parse_args()


def run(command: list[str]) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=REPO, check=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def csv_shape(path: Path) -> tuple[int, int, str, str]:
    rows = 0
    first = ""
    last = ""
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        for row in reader:
            if not row:
                continue
            rows += 1
            if not first:
                first = row[0]
            last = row[0]
    return rows, len(header), first, last


def validate_data() -> list[dict]:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    records = []
    for item in manifest["files"]:
        path = DATA / item["archive_path"]
        if not path.is_file():
            raise FileNotFoundError(f"released paper file is missing: {path}")
        rows, columns, first, last = csv_shape(path)
        actual = {
            "role": item["role"],
            "path": str(path.relative_to(REPO)),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
            "rows": rows,
            "columns": columns,
            "first_timestamp": first,
            "last_timestamp": last,
        }
        for key in ("bytes", "sha256", "rows", "columns"):
            if actual[key] != item[key]:
                raise ValueError(
                    f"data manifest mismatch for {item['role']} {key}: "
                    f"{actual[key]!r} != {item[key]!r}"
                )
        records.append(actual)
    return records


def recompute_ledger() -> tuple[dict, list[dict]]:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    grouped: dict[str, dict[str, list[float] | list[int]]] = {}
    with LEDGER.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            bucket = grouped.setdefault(row["variant"], {"seeds": [], "val": [], "test": []})
            bucket["seeds"].append(int(row["seed"]))
            bucket["val"].append(float(row["val_mse"]))
            bucket["test"].append(float(row["test_mse"]))
    aggregates = {}
    checks = []
    for name, expected in config["expected_metrics"].items():
        if name not in grouped:
            raise ValueError(f"seed ledger is missing setting: {name}")
        bucket = grouped[name]
        seeds = sorted(int(value) for value in bucket["seeds"])
        if seeds != FROZEN_SEEDS:
            raise ValueError(f"{name} seeds are {seeds}, expected {FROZEN_SEEDS}")
        aggregates[name] = {"seeds": seeds}
        for split in ("val", "test"):
            values = [float(value) for value in bucket[split]]
            stat = {"mean": statistics.mean(values), "std": statistics.stdev(values), "n": len(values)}
            aggregates[name][split] = stat
            for field in ("mean", "std"):
                target = float(expected[f"{split}_mse_{field}"])
                error = abs(stat[field] - target)
                checks.append(
                    {
                        "setting": name,
                        "metric": f"{split}_mse_{field}",
                        "actual": stat[field],
                        "expected": target,
                        "abs_error": error,
                        "ok": error <= 1e-12,
                    }
                )
    failed = [item for item in checks if not item["ok"]]
    if failed:
        raise ValueError(f"published seed ledger does not match the frozen config: {failed[:3]}")
    return aggregates, checks


def write_check_report(
    data_records: list[dict], aggregates: dict, checks: list[dict], *, print_json: bool
) -> Path:
    full = aggregates["DamXer"]["test"]["mean"]
    strongest = aggregates["PatchTST"]["test"]["mean"]
    report = {
        "status": "pass",
        "mode": "integrity_and_published_ledger_reaggregation",
        "training_performed": False,
        "claim_boundary": (
            "This check validates the released CSV bytes and recomputes manuscript aggregates from "
            "the published seed ledger. It is not a fresh model-training result."
        ),
        "data": data_records,
        "aggregates": aggregates,
        "checks": checks,
        "derived_claims": {
            "damxer_test_mse": full,
            "damxer_test_mse_std": aggregates["DamXer"]["test"]["std"],
            "reduction_vs_patchtst_pct": 100.0 * (strongest - full) / strongest,
        },
    }
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    output = ARTIFACTS / "check_report.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if print_json:
        print(json.dumps(report, indent=2))
    else:
        print(f"PASS: {len(data_records)} released data files match the frozen manifest.")
        print("Published five-seed test MSE (mean +/- sample SD):")
        for name, stats in aggregates.items():
            print(f"  {name:35s} {stats['test']['mean']:.6f} +/- {stats['test']['std']:.6f}")
        print("NOTE: this fast check reaggregates the published seed ledger; it does not retrain models.")
    return output


def ensure_training_imports() -> None:
    try:
        import numpy  # noqa: F401
        import pandas  # noqa: F401
        import torch  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "Training dependencies are missing. Run ./reproduce.sh damxer or ./reproduce.sh paper; "
            "the wrapper creates a project-local .venv and installs requirements-forecasting.txt."
        ) from exc


def common_damxer(args: argparse.Namespace, variant: str) -> list[str]:
    command = [
        sys.executable,
        "scripts/run_paper_multiseed.py",
        "--data-root",
        str(INPUTS),
        "--target-csv",
        str(TARGET),
        "--mask-csv",
        str(MASK),
        "--output-dir",
        str(ARTIFACTS / variant),
        "--strict-paper-shape",
        "--variant",
        variant,
        "--device",
        args.device,
        "--gpu",
        str(args.gpu),
        "--seeds",
        args.seeds,
    ]
    if args.allow_dirty_code:
        command.append("--allow-dirty-code")
    return command


def ensure_tslib(path: Path) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    required = config["tslib"]["required_commit"]
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        run(["git", "clone", config["tslib"]["repository"], str(path)])
    run(["git", "-C", str(path), "checkout", required])
    actual = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    if actual != required:
        raise RuntimeError(f"Time-Series-Library is at {actual}, expected {required}")


def run_training(args: argparse.Namespace) -> None:
    ensure_training_imports()
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    for variant in ("full", "no_lag_env", "no_env"):
        run(common_damxer(args, variant))
    if args.mode == "damxer":
        print("DamXer retraining finished. Baselines were not rerun in --mode damxer.")
        return

    tslib = Path(args.tslib_root).resolve()
    ensure_tslib(tslib)
    for profile, output in (("dx_only", "baselines-dx"), ("raw_env", "baseline-raw-env")):
        command = [
            sys.executable,
            "scripts/run_paper_baselines.py",
            "--profile",
            profile,
            "--data-root",
            str(INPUTS),
            "--target-csv",
            str(TARGET),
            "--mask-csv",
            str(MASK),
            "--output-dir",
            str(ARTIFACTS / output),
            "--tslib-root",
            str(tslib),
            "--device",
            args.device,
            "--gpu",
            str(args.gpu),
            "--seeds",
            args.seeds,
        ]
        if args.allow_dirty_code:
            command.append("--allow-dirty-code")
        run(command)

    seeds = sorted(int(value) for value in args.seeds.replace(" ", ",").split(",") if value)
    if seeds != FROZEN_SEEDS:
        print("Skipping aggregate acceptance check because this was not the frozen five-seed set.")
        return
    run(
        [
            sys.executable,
            "scripts/verify_paper_results.py",
            "--damxer-full",
            str(ARTIFACTS / "full" / "summary.json"),
            "--damxer-reduced",
            str(ARTIFACTS / "no_lag_env" / "summary.json"),
            "--damxer-no-env",
            str(ARTIFACTS / "no_env" / "summary.json"),
            "--dx-baselines",
            str(ARTIFACTS / "baselines-dx" / "aggregate.json"),
            "--raw-env",
            str(ARTIFACTS / "baseline-raw-env" / "aggregate.json"),
            "--atol",
            str(args.atol),
            "--output",
            str(ARTIFACTS / "fresh_training_verification.json"),
        ]
    )


def main() -> None:
    args = parse_args()
    data_records = validate_data()
    aggregates, checks = recompute_ledger()
    output = write_check_report(data_records, aggregates, checks, print_json=args.json)
    print(f"Integrity report: {output}")
    if args.mode != "check":
        run_training(args)


if __name__ == "__main__":
    main()
