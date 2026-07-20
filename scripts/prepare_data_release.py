#!/usr/bin/env python3
"""Assemble and verify a local Mendeley Data staging directory; never upload it."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

from paper_protocol import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-raw", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--target-root", required=True)
    parser.add_argument("--manifest", default="release/benchmark_manifest.json")
    parser.add_argument("--output-dir", default="release/staging")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    roots = {
        "source_raw": Path(args.source_raw).resolve(),
        "data": Path(args.data_root).resolve(),
        "target": Path(args.target_root).resolve(),
    }
    output = Path(args.output_dir).resolve()
    if not args.check_only and output.exists() and any(output.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty staging directory: {output}")
    if not args.check_only:
        output.mkdir(parents=True, exist_ok=True)
    records = []
    for item in manifest["files"]:
        role = item["role"]
        if role == "source_raw":
            source = roots["source_raw"]
        elif role in {"filtered_target", "observed_mask"}:
            source = roots["target"] / item["source_name"]
        else:
            source = roots["data"] / item["source_name"]
        if not source.is_file():
            raise FileNotFoundError(source)
        actual_hash = sha256_file(source)
        if source.stat().st_size != int(item["bytes"]) or actual_hash != item["sha256"]:
            raise ValueError(f"manifest mismatch for {role}: {source}")
        frame = pd.read_csv(source, nrows=1)
        if len(frame.columns) != int(item["columns"]):
            raise ValueError(f"column-count mismatch for {role}: {len(frame.columns)}")
        row_count = sum(1 for _ in source.open("rb")) - 1
        if row_count != int(item["rows"]):
            raise ValueError(f"row-count mismatch for {role}: {row_count}")
        destination = output / item["archive_path"]
        if not args.check_only:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        records.append(
            {
                "role": role,
                "archive_path": item["archive_path"],
                "bytes": source.stat().st_size if args.check_only else destination.stat().st_size,
                "sha256": actual_hash if args.check_only else sha256_file(destination),
            }
        )
    receipt = {
        "status": "verified",
        "check_only": args.check_only,
        "upload_performed": False,
        "platform": "Mendeley Data",
        "release_status": manifest["release_status"],
        "source_manifest": str(manifest_path),
        "files": records,
    }
    if not args.check_only:
        (output / "staging_receipt.json").write_text(
            json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
