#!/usr/bin/env python3
"""Train the frozen SAITS completion stage and write a continuous table."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

from completion_protocol import (
    load_completion_config,
    read_monitoring_csv,
    run_pypots_completion,
    sha256_file,
    write_completed_csv,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-csv", required=True)
    parser.add_argument("--config", default="configs/paper_completion.json")
    parser.add_argument("--output-dir", default="artifacts/completion/canonical")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--epochs", type=int, help="Smoke-only override; omit for the paper run")
    parser.add_argument("--strict-paper-shape", action="store_true")
    args = parser.parse_args()

    config = load_completion_config(args.config)
    dates, values, missing = read_monitoring_csv(args.raw_csv)
    expected = config["data_protocol"]
    if args.strict_paper_shape and (
        len(values) != int(expected["rows"]) or values.shape[1] != int(expected["channels"])
    ):
        raise ValueError(
            f"paper completion shape must be {expected['rows']}x{expected['channels']}, "
            f"got {len(values)}x{values.shape[1]}"
        )
    model_config = dict(config["models"]["saits"])
    if args.epochs is not None:
        model_config["epochs"] = args.epochs
    seed = int(config["completion_seed"] if args.seed is None else args.seed)
    completed, run = run_pypots_completion(
        values=values,
        missing=missing,
        model_name="saits",
        model_config=model_config,
        seed=seed,
        device=args.device,
        legacy_features_config=config["source_compatibility"],
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    clean_path = output_dir / "saits_clean.csv"
    write_completed_csv(clean_path, dates, completed)
    summary = {
        "status": "complete",
        "kind": "smoke" if args.epochs is not None else "formal",
        "input": {
            "path": str(Path(args.raw_csv).resolve()),
            "bytes": Path(args.raw_csv).stat().st_size,
            "sha256": sha256_file(args.raw_csv),
        },
        "output": {
            "path": str(clean_path.resolve()),
            "bytes": clean_path.stat().st_size,
            "sha256": sha256_file(clean_path),
        },
        "run": run,
        "runtime": {"python": platform.python_version()},
        "source_compatibility_note": config["source_compatibility"]["note"],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
