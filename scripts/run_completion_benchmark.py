#!/usr/bin/env python3
"""Run the fixed 360-sample dx completion benchmark from the raw table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from completion_protocol import (
    aggregate_seed_metrics,
    apply_pseudo_mask,
    build_fixed_block_mask,
    completion_metrics,
    load_completion_config,
    read_monitoring_csv,
    run_group_knn,
    run_pypots_completion,
    sha256_file,
    write_completed_csv,
    write_pseudo_mask,
)


def parse_list(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-csv", required=True)
    parser.add_argument("--config", default="configs/paper_completion.json")
    parser.add_argument("--output-dir", default="artifacts/completion/benchmark")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--models", default="saits,group_knn,imputeformer")
    parser.add_argument("--seeds", help="Comma-separated override")
    parser.add_argument("--epochs", type=int, help="Smoke-only override for neural models")
    parser.add_argument("--block-length", type=int, help="Smoke-only mask-length override")
    parser.add_argument("--strict-paper-shape", action="store_true")
    args = parser.parse_args()

    config = load_completion_config(args.config)
    dates, values, original_missing = read_monitoring_csv(args.raw_csv)
    expected = config["data_protocol"]
    target_columns = [column for column in values.columns if column.startswith("dx_")]
    if args.strict_paper_shape:
        actual = (len(values), values.shape[1], len(target_columns))
        frozen = (int(expected["rows"]), int(expected["channels"]), int(expected["dx_channels"]))
        if actual != frozen:
            raise ValueError(f"paper completion shape must be {frozen}, got {actual}")
    protocol = config["evaluation"]
    block_length = int(protocol["block_length_samples"] if args.block_length is None else args.block_length)
    pseudo_mask = build_fixed_block_mask(
        original_missing,
        target_columns,
        block_length,
        int(protocol["blocks_per_channel"]),
        int(protocol["mask_seed"]),
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    mask_path = output_dir / "pseudo_mask_positions.csv"
    write_pseudo_mask(mask_path, dates, values, pseudo_mask)
    eval_missing = apply_pseudo_mask(original_missing, pseudo_mask)
    models = parse_list(args.models)
    unknown = set(models) - {"saits", "group_knn", "imputeformer"}
    if unknown:
        raise ValueError(f"unknown completion models: {sorted(unknown)}")
    seeds = (
        [int(item) for item in parse_list(args.seeds)]
        if args.seeds
        else [int(protocol["reference_model_seed"])]
    )
    records = []
    if "group_knn" in models:
        prediction, run = run_group_knn(values, eval_missing, config["models"]["group_knn"])
        run_dir = output_dir / "group_knn"
        run_dir.mkdir(exist_ok=True)
        write_completed_csv(run_dir / "prediction.csv", dates, prediction)
        metrics = completion_metrics(values, prediction, original_missing, pseudo_mask)
        record = {"model": "group_knn", "seed": int(protocol["mask_seed"]), **metrics, "run": run}
        (run_dir / "metrics.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        records.append(record)
    for model_name in ("saits", "imputeformer"):
        if model_name not in models:
            continue
        for seed in seeds:
            model_config = dict(config["models"][model_name])
            if args.epochs is not None:
                model_config["epochs"] = args.epochs
            prediction, run = run_pypots_completion(
                values=values,
                missing=eval_missing,
                model_name=model_name,
                model_config=model_config,
                seed=seed,
                device=args.device,
                legacy_features_config=config["source_compatibility"],
            )
            run_dir = output_dir / model_name / f"seed_{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)
            write_completed_csv(run_dir / "prediction.csv", dates, prediction)
            metrics = completion_metrics(values, prediction, original_missing, pseudo_mask)
            record = {"model": model_name, "seed": seed, **metrics, "run": run}
            (run_dir / "metrics.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
            records.append(record)
    summary = {
        "status": "complete",
        "kind": "smoke" if args.epochs is not None or args.block_length is not None else "formal",
        "raw_csv": str(Path(args.raw_csv).resolve()),
        "raw_sha256": sha256_file(args.raw_csv),
        "pseudo_mask_csv": str(mask_path.resolve()),
        "pseudo_mask_sha256": sha256_file(mask_path),
        "pseudo_mask_count": sum(len(indices) for indices in pseudo_mask.values()),
        "protocol": {
            "mask_seed": int(protocol["mask_seed"]),
            "block_length_samples": block_length,
            "sampling_interval_hours": int(expected["sampling_interval_hours"]),
            "target_channels": len(target_columns),
        },
        "records": records,
        "aggregates": aggregate_seed_metrics(records),
        "input_fairness_note": config["source_compatibility"]["comparison_note"],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
