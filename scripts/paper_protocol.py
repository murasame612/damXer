"""Shared configuration and integrity helpers for the paper reproduction scripts."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from pathlib import Path

import pandas as pd


DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "paper_clean_window.json"


def load_config(path: str | Path = DEFAULT_CONFIG) -> tuple[Path, dict]:
    config_path = Path(path).resolve()
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 2:
        raise ValueError(f"unsupported paper config schema: {payload.get('schema_version')!r}")
    return config_path, payload


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: str | Path) -> dict:
    file_path = Path(path).resolve()
    return {
        "path": str(file_path),
        "bytes": file_path.stat().st_size,
        "sha256": sha256_file(file_path),
    }


def git_commit(path: str | Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(Path(path).resolve()), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def git_provenance(path: str | Path) -> dict:
    """Record both HEAD and worktree state so a dirty tree cannot masquerade as HEAD."""
    repo = Path(path).resolve()
    commit = git_commit(repo)
    if commit is None:
        return {"path": str(repo), "commit": None, "dirty": None, "status_sha256": None}
    status = subprocess.check_output(
        ["git", "-C", str(repo), "status", "--porcelain=v1", "--untracked-files=all"],
        text=True,
    )
    return {
        "path": str(repo),
        "commit": commit,
        "dirty": bool(status),
        "status_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest() if status else None,
    }


def require_clean_git(provenance: dict, label: str) -> None:
    if provenance.get("commit") is None:
        raise RuntimeError(f"{label} is not a Git checkout; code provenance cannot be frozen")
    if provenance.get("dirty"):
        raise RuntimeError(
            f"{label} has uncommitted changes. Commit them before a paper run, or pass "
            "--allow-dirty-code for a diagnostic run whose manifest will be marked dirty."
        )


def runtime_versions() -> dict[str, str]:
    import numpy as np
    import torch

    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "torch": torch.__version__,
        "platform": platform.platform(),
    }


def _value_columns(frame: pd.DataFrame) -> list[str]:
    if "date" not in frame.columns:
        raise ValueError("CSV is missing the required date column")
    return [column for column in frame.columns if column != "date"]


def observed_window_count(mask: pd.DataFrame, target_columns: list[str], start: int, end: int, pred_len: int) -> int:
    mask_columns = [f"{column}_masked" for column in target_columns]
    missing = [column for column in mask_columns if column not in mask.columns]
    if missing:
        raise ValueError(f"mask CSV is missing columns: {missing[:5]}")
    observed = 1 - mask[mask_columns].astype("int8").to_numpy()
    return int(
        sum(observed[origin : origin + pred_len].sum() for origin in range(start, end - pred_len + 1))
    )


def validate_baseline_inputs(
    *,
    input_csv: str | Path,
    target_csv: str | Path,
    mask_csv: str | Path,
    profile_name: str,
    config: dict,
) -> dict:
    protocol = config["data_protocol"]
    profile = config["tslib"]["profiles"][profile_name]
    paths = {
        "input": Path(input_csv).resolve(),
        "target": Path(target_csv).resolve(),
        "mask": Path(mask_csv).resolve(),
    }
    missing_files = [str(path) for path in paths.values() if not path.is_file()]
    if missing_files:
        raise FileNotFoundError("missing required input files: " + ", ".join(missing_files))
    frames = {name: pd.read_csv(path) for name, path in paths.items()}
    dates = frames["input"]["date"].astype(str).tolist()
    for name in ("target", "mask"):
        if frames[name]["date"].astype(str).tolist() != dates:
            raise ValueError(f"{name} dates do not match the input CSV")

    input_columns = _value_columns(frames["input"])
    target_columns = _value_columns(frames["target"])
    target_dim = int(protocol["target_channels"])
    if input_columns[:target_dim] != target_columns[:target_dim]:
        raise ValueError("the first target channels in input and target CSVs differ")
    if len(target_columns) != target_dim:
        raise ValueError(f"target channel count is {len(target_columns)}, expected {target_dim}")
    if len(input_columns) != int(profile["input_channels"]):
        raise ValueError(
            f"{profile_name} input channel count is {len(input_columns)}, expected {profile['input_channels']}"
        )
    if frames["input"][input_columns].isna().any().any():
        raise ValueError("model input contains NaN values")

    rows = len(dates)
    if rows != int(protocol["timestamps"]):
        raise ValueError(f"row count is {rows}, expected {protocol['timestamps']}")
    if dates[0] != str(protocol["date_start"]) or dates[-1] != str(protocol["date_end"]):
        raise ValueError(
            f"date range is {dates[0]} through {dates[-1]}, expected "
            f"{protocol['date_start']} through {protocol['date_end']}"
        )

    split_rows = [int(value) for value in protocol["split_rows"]]
    pred_len = int(config["tslib"]["common"]["pred_len"])
    seq_len = int(profile["seq_len"])
    n_train, n_val, n_test = split_rows
    split_windows = [
        n_train - seq_len - pred_len + 1,
        n_val - pred_len + 1,
        n_test - pred_len + 1,
    ]
    expected_windows = (
        protocol["dx_only_split_windows"]
        if profile_name == "dx_only"
        else protocol["damxer_split_windows"]
    )
    if split_windows != expected_windows:
        raise ValueError(f"split windows are {split_windows}, expected {expected_windows}")

    val_start = n_train
    test_start = n_train + n_val
    observed_counts = {
        "validation": observed_window_count(frames["mask"], target_columns, val_start, test_start, pred_len),
        "test": observed_window_count(frames["mask"], target_columns, test_start, rows, pred_len),
    }
    report = {
        "profile": profile_name,
        "rows": rows,
        "first_date": dates[0],
        "last_date": dates[-1],
        "target_channels": target_dim,
        "input_channels": len(input_columns),
        "auxiliary_channels": len(input_columns) - target_dim,
        "split_windows": {
            "train": split_windows[0],
            "validation": split_windows[1],
            "test": split_windows[2],
        },
        "observed_window_cells": observed_counts,
        "files": {name: file_record(path) for name, path in paths.items()},
    }
    hashes = protocol["canonical_sha256"]
    assert_canonical_hashes(
        report["files"],
        {
            "input": hashes["dx_input" if profile_name == "dx_only" else "raw_env_input"],
            "target": hashes["target"],
            "mask": hashes["mask"],
        },
    )
    return report


def assert_canonical_hashes(records: dict[str, dict], expected: dict[str, str]) -> None:
    """Fail when a frozen paper input differs from its canonical byte hash."""
    mismatches = {
        name: {"actual": records.get(name, {}).get("sha256"), "expected": digest}
        for name, digest in expected.items()
        if name not in records or records[name]["sha256"] != digest
    }
    if mismatches:
        raise ValueError(f"canonical paper input hashes differ: {mismatches}")
