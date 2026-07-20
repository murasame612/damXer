#!/usr/bin/env python3
"""Verify a formal completion summary against the frozen paper protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from completion_protocol import load_completion_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--config", default="configs/paper_completion.json")
    parser.add_argument("--metric-atol", type=float, default=5e-6)
    parser.add_argument("--allow-noncanonical-raw", action="store_true")
    args = parser.parse_args()

    config = load_completion_config(args.config)
    summary = json.loads(Path(args.summary).read_text(encoding="utf-8"))
    errors: list[str] = []
    protocol = config["evaluation"]
    data = config["data_protocol"]
    if summary.get("kind") != "formal":
        errors.append("summary is not marked as a formal run")
    if int(summary.get("pseudo_mask_count", -1)) != int(protocol["pseudo_mask_count"]):
        errors.append(
            f"pseudo-mask count is {summary.get('pseudo_mask_count')}, "
            f"expected {protocol['pseudo_mask_count']}"
        )
    if not args.allow_noncanonical_raw and summary.get("raw_sha256") != data["canonical_raw_sha256"]:
        errors.append("raw input SHA-256 differs from the frozen anonymized monitoring table")
    records = summary.get("records", [])
    reference_seed = int(protocol["reference_model_seed"])
    references = protocol["paper_reference_metrics"]
    for model_name, expected in references.items():
        candidates = [
            row
            for row in records
            if row.get("model") == model_name
            and (
                model_name == "group_knn"
                or int(row.get("seed", -1)) == reference_seed
            )
        ]
        if len(candidates) != 1:
            errors.append(f"expected one {model_name} reference record, found {len(candidates)}")
            continue
        actual = candidates[0]["macro_overall"]
        for metric, expected_value in expected.items():
            delta = abs(float(actual[metric]) - float(expected_value))
            if delta > args.metric_atol:
                errors.append(
                    f"{model_name} {metric} differs by {delta:.6g}: "
                    f"actual={actual[metric]} expected={expected_value}"
                )
    report = {
        "status": "pass" if not errors else "fail",
        "summary": str(Path(args.summary).resolve()),
        "errors": errors,
    }
    print(json.dumps(report, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
