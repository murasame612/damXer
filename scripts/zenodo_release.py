#!/usr/bin/env python3
"""Validate, stage, and upload the separate DamXer data release.

The script is fail-closed: real data cannot be staged or uploaded unless the
caller explicitly confirms that written data-owner approval exists. Uploading
creates a Zenodo draft only; publication remains a separate manual action.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import zipfile

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST_PATH = REPO_ROOT / "release" / "benchmark_manifest.json"
DATA_CARD = REPO_ROOT / "release" / "DATA_CARD.md"
ZENODO_BASE = "https://zenodo.org"
ZENODO_SANDBOX_BASE = "https://sandbox.zenodo.org"


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def csv_shape(path: Path) -> tuple[int, int, list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError(f"empty CSV: {path}") from exc
        rows = sum(1 for _ in reader)
    return rows, len(header), header


def resolve_files(
    input_root: Path, target_root: Path, manifest: dict
) -> list[tuple[dict, Path]]:
    resolved: list[tuple[dict, Path]] = []
    for item in manifest["files"]:
        root = (
            input_root
            if item["role"] in {"dx_input", "engineered_input"}
            else target_root
        )
        resolved.append((item, root / item["source_name"]))
    return resolved


def validate_files(
    input_root: Path, target_root: Path, manifest_path: Path
) -> tuple[dict, list[tuple[dict, Path]]]:
    manifest = load_json(manifest_path)
    resolved = resolve_files(input_root, target_root, manifest)
    headers: dict[str, list[str]] = {}
    for item, path in resolved:
        if not path.is_file():
            raise FileNotFoundError(path)
        rows, columns, header = csv_shape(path)
        actual_hash = sha256(path)
        if rows != manifest["rows"]:
            raise ValueError(
                f"{path.name}: expected {manifest['rows']} rows, found {rows}"
            )
        if columns != item["columns"]:
            raise ValueError(
                f"{path.name}: expected {item['columns']} columns, found {columns}"
            )
        if not item.get("sha256"):
            raise ValueError(
                f"{manifest_path}: private release manifest must contain SHA-256 values"
            )
        if actual_hash != item["sha256"]:
            raise ValueError(f"{path.name}: SHA-256 mismatch")
        if not header or header[0] != "date":
            raise ValueError(f"{path.name}: first column must be date")
        headers[item["role"]] = header

    dx = headers["dx_input"]
    engineered = headers["engineered_input"]
    target = headers["filtered_target"]
    mask = headers["observed_mask"]
    if engineered[: len(dx)] != dx:
        raise ValueError(
            "engineered input does not begin with the ordered dx input columns"
        )
    if target != dx:
        raise ValueError("filtered target columns do not match ordered dx input columns")
    expected_mask = ["date", *[f"{name}_masked" for name in dx[1:]]]
    if mask != expected_mask:
        raise ValueError("observed-mask columns do not match ordered target columns")
    return manifest, resolved


def require_approval(args: argparse.Namespace) -> None:
    approval = Path(args.approval_file).expanduser().resolve()
    if not args.confirm_owner_approval:
        raise PermissionError(
            "refusing real-data operation without --confirm-owner-approval"
        )
    if not approval.is_file():
        raise FileNotFoundError(f"written approval file not found: {approval}")


def command_validate(args: argparse.Namespace) -> None:
    _, resolved = validate_files(
        Path(args.input_root), Path(args.target_root), Path(args.manifest)
    )
    for item, path in resolved:
        print(f"OK {item['role']}: {path.name} ({item['sha256']})")
    print("All four paper benchmark files match the frozen manifest.")


def command_stage(args: argparse.Namespace) -> None:
    require_approval(args)
    input_root = Path(args.input_root).expanduser().resolve()
    target_root = Path(args.target_root).expanduser().resolve()
    manifest_path = Path(args.manifest).expanduser().resolve()
    manifest, resolved = validate_files(input_root, target_root, manifest_path)
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not args.overwrite:
        raise FileExistsError(
            f"output exists; pass --overwrite to replace it: {output}"
        )

    staged_manifest = dict(manifest)
    staged_manifest["release_status"] = "owner_approval_confirmed_locally"
    with tempfile.TemporaryDirectory(prefix="damxer-release-") as tmp:
        root = Path(tmp) / "damxer_processed_forecasting_benchmark_v1"
        (root / "inputs").mkdir(parents=True)
        (root / "targets").mkdir(parents=True)
        for item, source in resolved:
            destination = root / item["archive_path"]
            shutil.copyfile(source, destination)
        (root / "DATA_CARD.md").write_text(
            DATA_CARD.read_text(encoding="utf-8"), encoding="utf-8"
        )
        (root / "manifest.json").write_text(
            json.dumps(staged_manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        checksum_lines = [
            f"{item['sha256']}  {item['archive_path']}" for item, _ in resolved
        ]
        (root / "checksums.sha256").write_text(
            "\n".join(checksum_lines) + "\n", encoding="utf-8"
        )
        with zipfile.ZipFile(
            output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
        ) as archive:
            for path in sorted(root.rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(root.parent))
    print(f"Staged verified release archive: {output}")


def validate_metadata(metadata: dict) -> None:
    payload = metadata.get("metadata")
    if not isinstance(payload, dict):
        raise ValueError("metadata JSON must contain a top-level metadata object")
    rendered = json.dumps(payload, ensure_ascii=False)
    if "TODO" in rendered:
        raise ValueError("replace all TODO creator/affiliation fields before upload")
    if payload.get("access_right") != "restricted":
        raise ValueError("initial paper-review record must use restricted access")


def command_upload_draft(args: argparse.Namespace) -> None:
    try:
        import requests
    except ImportError as exc:
        raise RuntimeError(
            "upload-draft requires the project dependencies: "
            "python -m pip install -r requirements.txt"
        ) from exc

    require_approval(args)
    archive = Path(args.archive).expanduser().resolve()
    metadata_path = Path(args.metadata).expanduser().resolve()
    if not archive.is_file():
        raise FileNotFoundError(archive)
    metadata = load_json(metadata_path)
    validate_metadata(metadata)
    token = os.environ.get("ZENODO_ACCESS_TOKEN")
    if not token:
        raise RuntimeError("ZENODO_ACCESS_TOKEN is not set")

    base = ZENODO_SANDBOX_BASE if args.sandbox else ZENODO_BASE
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.post(
        f"{base}/api/deposit/depositions", json={}, headers=headers, timeout=60
    )
    response.raise_for_status()
    deposition = response.json()
    deposition_id = deposition["id"]
    try:
        update = requests.put(
            f"{base}/api/deposit/depositions/{deposition_id}",
            json=metadata,
            headers=headers,
            timeout=60,
        )
        update.raise_for_status()
        updated = update.json()
        bucket_url = updated["links"]["bucket"]
        with archive.open("rb") as handle:
            upload = requests.put(
                f"{bucket_url}/{archive.name}",
                data=handle,
                headers=headers,
                timeout=1800,
            )
        upload.raise_for_status()
    except Exception:
        print(
            f"Zenodo draft {deposition_id} was created but not completed.",
            file=sys.stderr,
        )
        raise

    reserved = updated.get("metadata", {}).get("prereserve_doi", {}).get("doi")
    print(f"Created restricted Zenodo draft: {updated['links']['html']}")
    if reserved:
        print(f"Reserved DOI: {reserved}")
    print("The record is still a draft and has NOT been published.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    validate = sub.add_parser(
        "validate", help="validate local canonical CSVs without copying or uploading"
    )
    validate.add_argument("--input-root", required=True)
    validate.add_argument("--target-root", required=True)
    validate.add_argument("--manifest", default=str(DEFAULT_MANIFEST_PATH))
    validate.set_defaults(func=command_validate)

    stage = sub.add_parser(
        "stage", help="build the verified release ZIP after owner approval"
    )
    stage.add_argument("--input-root", required=True)
    stage.add_argument("--target-root", required=True)
    stage.add_argument("--manifest", default=str(DEFAULT_MANIFEST_PATH))
    stage.add_argument("--approval-file", required=True)
    stage.add_argument("--confirm-owner-approval", action="store_true")
    stage.add_argument("--output", required=True)
    stage.add_argument("--overwrite", action="store_true")
    stage.set_defaults(func=command_stage)

    upload = sub.add_parser(
        "upload-draft", help="create and fill a restricted Zenodo draft"
    )
    upload.add_argument("--archive", required=True)
    upload.add_argument("--metadata", required=True)
    upload.add_argument("--approval-file", required=True)
    upload.add_argument("--confirm-owner-approval", action="store_true")
    upload.add_argument("--sandbox", action="store_true")
    upload.set_defaults(func=command_upload_draft)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
