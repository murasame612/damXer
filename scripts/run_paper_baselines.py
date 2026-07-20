#!/usr/bin/env python3
"""Run a frozen Time-Series-Library baseline profile for the DamXer paper."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
from pathlib import Path

from paper_protocol import (
    DEFAULT_CONFIG,
    git_provenance,
    load_config,
    require_clean_git,
    runtime_versions,
    sha256_file,
    validate_baseline_inputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=["dx_only", "raw_env"], required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--target-csv", required=True)
    parser.add_argument("--mask-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tslib-root", required=True)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--seeds", default="")
    parser.add_argument("--models", default="")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--allow-dirty-code", action="store_true")
    parser.add_argument("--keep-predictions", action="store_true")
    return parser.parse_args()


def summarize(values: list[float]) -> dict:
    return {
        "mean": statistics.mean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "n": len(values),
    }


def nested_value(payload: dict, *path: str) -> float:
    cursor = payload
    for key in path:
        cursor = cursor[key]
    return float(cursor)


def main() -> None:
    args = parse_args()
    config_path, config = load_config(args.config)
    profile = config["tslib"]["profiles"][args.profile]
    common = config["tslib"]["common"]
    input_csv = Path(args.data_root) / profile["input_file"]
    validation = validate_baseline_inputs(
        input_csv=input_csv,
        target_csv=args.target_csv,
        mask_csv=args.mask_csv,
        profile_name=args.profile,
        config=config,
    )
    tslib_root = Path(args.tslib_root).resolve()
    if not (tslib_root / "models").is_dir():
        raise FileNotFoundError(f"Time-Series-Library models directory not found: {tslib_root / 'models'}")
    adapter_provenance = git_provenance(Path(__file__).resolve().parents[1])
    tslib_provenance = git_provenance(tslib_root)
    if not args.allow_dirty_code:
        require_clean_git(adapter_provenance, "DamXer repository")
        require_clean_git(tslib_provenance, "Time-Series-Library checkout")
    tslib_commit = tslib_provenance["commit"]
    required_commit = str(config["tslib"].get("required_commit", "")).strip()
    if required_commit and tslib_commit != required_commit:
        raise RuntimeError(
            f"Time-Series-Library commit is {tslib_commit}, expected frozen commit {required_commit}"
        )

    models = [item for item in args.models.replace(",", " ").split() if item] or list(profile["models"])
    unknown_models = sorted(set(models) - set(profile["models"]))
    if unknown_models:
        raise ValueError(f"models are outside frozen {args.profile} profile: {unknown_models}")
    seeds = [int(item) for item in args.seeds.replace(" ", ",").split(",") if item] or [
        int(value) for value in config["optimization"]["seeds"]
    ]
    protocol_manifest = {
        "protocol": f"paper_tslib_{args.profile}",
        "config": {"path": str(config_path), "sha256": sha256_file(config_path)},
        "adapter_provenance": adapter_provenance,
        "runtime": runtime_versions(),
        "validation": validation,
        "models": models,
        "seeds": seeds,
        "tslib": {
            "path": str(tslib_root),
            "commit": tslib_commit,
            "required_commit": required_commit or None,
            "pin_status": "matched" if required_commit else "unresolved",
            "commit_provenance": config["tslib"].get("commit_provenance"),
            "git_provenance": tslib_provenance,
        },
        "model_configs": {
            model: {
                **common,
                **config["tslib"].get("model_overrides", {}).get(model, {}),
                "seq_len": profile["seq_len"],
                "input_channels": profile["input_channels"],
                "label_len": profile["label_len"],
            }
            for model in models
        },
        "selection": "validation observed MSE; test metrics never enter selection",
    }
    if args.check_only:
        print(json.dumps(protocol_manifest, indent=2))
        return

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "protocol_manifest.json").write_text(
        json.dumps(protocol_manifest, indent=2), encoding="utf-8"
    )
    runner = Path(__file__).with_name("run_tslib_baseline.py")
    rows: list[dict] = []
    for model in models:
        run_config = {**common, **config["tslib"].get("model_overrides", {}).get(model, {})}
        evaluation_schedule = config["source_compatibility"].get(
            "baseline_evaluation_schedule_overrides", {}
        ).get(model, config["source_compatibility"]["baseline_evaluation_schedule"])
        label_len = int(profile["label_len"].get(model, profile["label_len"].get("default", 0)))
        for seed in seeds:
            run_dir = output_dir / model / f"seed_{seed}"
            summary_path = run_dir / "summary.json"
            log_path = run_dir / "run.log"
            run_dir.mkdir(parents=True, exist_ok=True)
            command = [
                sys.executable,
                str(runner),
                "--data_root", str(Path(args.data_root).resolve()),
                "--output", str(summary_path),
                "--model", model,
                "--tslib_root", str(tslib_root),
                "--mask_csv", str(Path(args.mask_csv).resolve()),
                "--target_csv", str(Path(args.target_csv).resolve()),
                "--variants", profile["input_file"],
                "--seq_len", str(profile["seq_len"]),
                "--label_len", str(label_len),
                "--pred_len", str(run_config["pred_len"]),
                "--target_dim", str(run_config["target_dim"]),
                "--epochs", str(run_config["epochs"]),
                "--patience", str(run_config["patience"]),
                "--batch_size", str(run_config["batch_size"]),
                "--num_workers", str(args.num_workers),
                "--lr", str(run_config["learning_rate"]),
                "--d_model", str(run_config["d_model"]),
                "--d_ff", str(run_config["d_ff"]),
                "--n_heads", str(run_config["n_heads"]),
                "--e_layers", str(run_config["e_layers"]),
                "--dropout", str(run_config["dropout"]),
                "--patch_len", str(run_config["patch_len"]),
                "--patch_stride", str(run_config["patch_stride"]),
                "--channel_independence", str(run_config.get("channel_independence", 1)),
                "--gpu", str(args.gpu),
                "--device", args.device,
                "--seed", str(seed),
                "--train_loss_mode", run_config["train_loss_mode"],
                "--selection_metric", run_config["selection_metric"],
                "--evaluation_schedule", evaluation_schedule,
                "--drop_unobserved_windows",
            ]
            if args.keep_predictions:
                command.extend(
                    [
                        "--prediction_npz_dir", str(run_dir / "predictions"),
                        "--prediction_splits", "val", "test",
                    ]
                )
            print("+ " + " ".join(command), flush=True)
            process = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            log_path.write_text(process.stdout, encoding="utf-8")
            print(process.stdout, end="", flush=True)
            if process.returncode != 0:
                raise subprocess.CalledProcessError(process.returncode, command)
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            result = payload[0] if isinstance(payload, list) else payload["results"][0]
            rows.append(
                {
                    "model": model,
                    "seed": seed,
                    "best_epoch": result["best_epoch"],
                    "val_observed_mse": nested_value(result, "best_val", "observed", "mse"),
                    "test_observed_mse": nested_value(result, "best_test", "observed", "mse"),
                    "test_observed_mae": nested_value(result, "best_test", "observed", "mae"),
                    "test_observed_rmse": nested_value(result, "best_test", "observed", "rmse"),
                    "summary_json": str(summary_path),
                }
            )

    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    aggregate = {}
    for model in models:
        model_rows = [row for row in rows if row["model"] == model]
        aggregate[model] = {
            metric: summarize([float(row[metric]) for row in model_rows])
            for metric in (
                "val_observed_mse",
                "test_observed_mse",
                "test_observed_mae",
                "test_observed_rmse",
            )
        }
    (output_dir / "aggregate.json").write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
