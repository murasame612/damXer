"""Reproducible completion utilities used by the DamXer data pipeline.

The implementation is intentionally self-contained so the public package does
not depend on the internal experiment repository.  The default configuration
preserves the historical PyPOTS runs byte-for-protocol, including the seven
deterministic index features appended only to SAITS.  Their two sinusoidal
periods are recorded as sample-index periods, not described as physical daily
or annual cycles.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev

import numpy as np
import pandas as pd


EPS = 1e-12


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_completion_config(path: str | Path) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported completion config schema")
    return payload


def read_monitoring_csv(path: str | Path) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    frame = pd.read_csv(path)
    date_column = "date" if "date" in frame.columns else "采集时间"
    if date_column not in frame.columns:
        raise ValueError("monitoring CSV must contain 'date' or '采集时间'")
    value_columns = [column for column in frame.columns if column != date_column]
    values = frame[value_columns].apply(pd.to_numeric, errors="coerce")
    missing = values.isna()
    if values.shape[1] == 0:
        raise ValueError("monitoring CSV has no value columns")
    if any((~missing[column]).sum() == 0 for column in value_columns):
        empty = [column for column in value_columns if (~missing[column]).sum() == 0]
        raise ValueError(f"columns without observations are unsupported: {empty[:5]}")
    return frame[date_column].astype(str), values, missing


def write_completed_csv(path: str | Path, dates: pd.Series, values: pd.DataFrame) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    completed = pd.concat(
        [pd.DataFrame({"date": dates.astype(str).to_numpy()}), values.reset_index(drop=True)],
        axis=1,
    )
    completed.to_csv(output, index=False, float_format="%.12g")


def make_windows(row_count: int, sequence_length: int, stride: int) -> list[tuple[int, int]]:
    if sequence_length <= 1 or stride <= 0:
        raise ValueError("sequence_length must exceed 1 and stride must be positive")
    if row_count <= sequence_length:
        return [(0, row_count)]
    starts = list(range(0, row_count - sequence_length + 1, stride))
    last = row_count - sequence_length
    if starts[-1] != last:
        starts.append(last)
    return [(start, start + sequence_length) for start in starts]


def legacy_index_features(row_count: int, config: dict) -> np.ndarray:
    """Return the seven deterministic features used by the historical SAITS run."""
    index = np.arange(row_count, dtype=np.float64)
    denominator = max(row_count - 1, 1)
    linear = index / denominator
    log_index = np.log1p(index) / math.log1p(denominator)
    periods = config["legacy_index_feature_periods_samples"]
    short_period = float(periods["short"])
    long_period = float(periods["long"])
    return np.stack(
        [
            linear,
            linear**2,
            log_index,
            np.sin(2.0 * math.pi * index / short_period),
            np.cos(2.0 * math.pi * index / short_period),
            np.sin(2.0 * math.pi * index / long_period),
            np.cos(2.0 * math.pi * index / long_period),
        ],
        axis=1,
    ).astype(np.float32)


def _normalized(values: pd.DataFrame, missing: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    observed_values = values.mask(missing)
    means = observed_values.mean(axis=0).to_numpy(dtype=np.float32)
    stds = observed_values.std(axis=0, ddof=0).to_numpy(dtype=np.float32)
    stds = np.where(np.isfinite(stds) & (stds > EPS), stds, 1.0).astype(np.float32)
    array = values.to_numpy(dtype=np.float32)
    normalized = (array - means) / stds
    normalized[missing.to_numpy(dtype=bool)] = np.nan
    return normalized.astype(np.float32), means, stds


def _window_stack(values: np.ndarray, windows: list[tuple[int, int]]) -> np.ndarray:
    return np.stack([values[start:end] for start, end in windows], axis=0).astype(np.float32)


def _combine_windows(
    predictions: np.ndarray,
    windows: list[tuple[int, int]],
    row_count: int,
) -> np.ndarray:
    output_dim = predictions.shape[-1]
    total = np.zeros((row_count, output_dim), dtype=np.float32)
    count = np.zeros((row_count, 1), dtype=np.float32)
    for window_index, (start, end) in enumerate(windows):
        total[start:end] += predictions[window_index, : end - start]
        count[start:end] += 1.0
    return total / np.maximum(count, 1.0)


def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def run_pypots_completion(
    *,
    values: pd.DataFrame,
    missing: pd.DataFrame,
    model_name: str,
    model_config: dict,
    seed: int,
    device: str,
    legacy_features_config: dict,
) -> tuple[pd.DataFrame, dict]:
    try:
        from pypots.imputation import ImputeFormer, SAITS
        from pypots.utils.random import set_random_seed
    except ImportError as exc:
        raise RuntimeError("PyPOTS 1.5 is required for SAITS/ImputeFormer completion") from exc

    set_random_seed(seed)
    normalized, means, stds = _normalized(values, missing)
    windows = make_windows(
        len(values), int(model_config["sequence_length"]), int(model_config["stride"])
    )
    value_dim = values.shape[1]
    if model_name == "saits" and bool(model_config.get("include_legacy_index_features", False)):
        features = legacy_index_features(len(values), legacy_features_config)
        model_values = np.concatenate([normalized, features], axis=1)
    else:
        model_values = normalized
    windowed = _window_stack(model_values, windows)
    common = {
        "n_steps": int(model_config["sequence_length"]),
        "n_features": int(model_values.shape[1]),
        "batch_size": int(model_config["batch_size"]),
        "epochs": int(model_config["epochs"]),
        "patience": model_config.get("patience"),
        "num_workers": 0,
        "device": resolve_device(device),
        "saving_path": None,
        "model_saving_strategy": None,
        "verbose": bool(model_config.get("verbose", False)),
    }
    if model_name == "saits":
        model = SAITS(
            **common,
            n_layers=int(model_config["n_layers"]),
            d_model=int(model_config["d_model"]),
            n_heads=int(model_config["n_heads"]),
            d_k=int(model_config["d_k"]),
            d_v=int(model_config["d_v"]),
            d_ffn=int(model_config["d_ffn"]),
            dropout=float(model_config["dropout"]),
            attn_dropout=float(model_config["attn_dropout"]),
            diagonal_attention_mask=bool(model_config["diagonal_attention_mask"]),
            ORT_weight=int(model_config["ORT_weight"]),
            MIT_weight=int(model_config["MIT_weight"]),
        )
    elif model_name == "imputeformer":
        model = ImputeFormer(
            **common,
            n_layers=int(model_config["n_layers"]),
            d_input_embed=int(model_config["d_input_embed"]),
            d_learnable_embed=int(model_config["d_learnable_embed"]),
            d_proj=int(model_config["d_proj"]),
            d_ffn=int(model_config["d_ffn"]),
            n_temporal_heads=int(model_config["n_temporal_heads"]),
            dropout=float(model_config["dropout"]),
            input_dim=1,
            output_dim=1,
            ORT_weight=float(model_config["ORT_weight"]),
            MIT_weight=float(model_config["MIT_weight"]),
        )
    else:
        raise ValueError(f"unsupported PyPOTS model: {model_name}")
    model.fit({"X": windowed})
    if model_name == "saits":
        predicted_windows = model.impute(
            {"X": windowed},
            diagonal_attention_mask=bool(model_config["diagonal_attention_mask"]),
        )
    else:
        predicted_windows = model.impute({"X": windowed})
    prediction_norm = _combine_windows(predicted_windows, windows, len(values))[:, :value_dim]
    predictions = prediction_norm * stds + means
    observed = values.to_numpy(dtype=np.float32)
    completed = np.where(missing.to_numpy(dtype=bool), predictions, observed)
    return pd.DataFrame(completed, columns=values.columns), {
        "model": model_name,
        "seed": seed,
        "device": resolve_device(device),
        "rows": len(values),
        "monitoring_channels": value_dim,
        "model_features": int(model_values.shape[1]),
        "window_count": len(windows),
        "missing_cells_filled": int(missing.to_numpy().sum()),
        "config": model_config,
    }


def _family(name: str) -> str:
    return name.split("_", 1)[0] if "_" in name else "other"


@dataclass(frozen=True)
class ColumnInfo:
    family: str
    zone: str | None
    group: str | None
    point: str | None
    depth: float | None


def _column_info(name: str) -> ColumnInfo:
    if "_" not in name:
        return ColumnInfo("other", None, None, None, None)
    family, rest = name.split("_", 1)
    tokens = rest.split("-")
    zone = tokens[0] if tokens else None
    group = "-".join(tokens[:2]) if len(tokens) >= 2 else zone
    point = "-".join(tokens[:-1]) if len(tokens) >= 2 else zone
    token = tokens[-1] if tokens else ""
    depth = None
    if token.upper().endswith("M"):
        try:
            depth = float(token[:-1])
        except ValueError:
            depth = None
    return ColumnInfo(family, zone, group, point, depth)


def _neighbor_score(target: ColumnInfo, candidate: ColumnInfo) -> tuple[float, float]:
    score = 100.0 if target.family == candidate.family else 0.0
    score += 30.0 if target.zone is not None and target.zone == candidate.zone else 0.0
    score += 40.0 if target.group is not None and target.group == candidate.group else 0.0
    score += 25.0 if target.point is not None and target.point == candidate.point else 0.0
    distance = 1_000_000.0
    if target.depth is not None and candidate.depth is not None:
        distance = abs(target.depth - candidate.depth)
        score += max(0.0, 25.0 - distance)
    return score, distance


def _linear_prefill(values: pd.DataFrame, missing: pd.DataFrame) -> pd.DataFrame:
    output = values.copy()
    for column in values.columns:
        output[column] = values[column].interpolate(method="linear", limit_direction="both")
    return output


def run_group_knn(
    values: pd.DataFrame,
    missing: pd.DataFrame,
    config: dict,
) -> tuple[pd.DataFrame, dict]:
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.preprocessing import StandardScaler

    columns = list(values.columns)
    infos = {column: _column_info(column) for column in columns}
    order = {column: index for index, column in enumerate(columns)}
    prefill = _linear_prefill(values, missing)
    completed = prefill.copy()
    selected_neighbors: dict[str, list[str]] = {}
    for target in columns:
        masked_indices = np.flatnonzero(missing[target].to_numpy())
        if len(masked_indices) == 0:
            continue
        trusted_indices = np.flatnonzero(~missing[target].to_numpy())
        candidates = []
        for candidate in columns:
            if candidate == target or (
                bool(config["same_family_only"])
                and infos[candidate].family != infos[target].family
            ):
                continue
            score, distance = _neighbor_score(infos[target], infos[candidate])
            candidates.append((-score, distance, order[candidate], candidate))
        candidates.sort()
        neighbors = [item[-1] for item in candidates[: int(config["max_feature_columns"])]]
        selected_neighbors[target] = neighbors
        if not neighbors or len(trusted_indices) < int(config["min_training_rows"]):
            continue
        matrix = prefill[neighbors].to_numpy(dtype=float)
        scaler = StandardScaler()
        train_x = scaler.fit_transform(matrix[trusted_indices])
        predict_x = scaler.transform(matrix[masked_indices])
        model = KNeighborsRegressor(
            n_neighbors=min(int(config["n_neighbors"]), len(trusted_indices)),
            weights=str(config["weights"]),
            n_jobs=int(config.get("n_jobs", 1)),
        )
        model.fit(train_x, values[target].to_numpy(dtype=float)[trusted_indices])
        completed.loc[masked_indices, target] = model.predict(predict_x)
    return completed, {
        "model": "group_knn",
        "rows": len(values),
        "monitoring_channels": len(columns),
        "missing_cells_filled": int(missing.to_numpy().sum()),
        "config": config,
        "neighbor_preview": dict(list(selected_neighbors.items())[:5]),
    }


def _trusted_runs(indices: set[int]) -> list[tuple[int, int]]:
    ordered = sorted(indices)
    if not ordered:
        return []
    runs: list[tuple[int, int]] = []
    start = previous = ordered[0]
    for index in ordered[1:]:
        if index == previous + 1:
            previous = index
        else:
            runs.append((start, previous))
            start = previous = index
    runs.append((start, previous))
    return runs


def build_fixed_block_mask(
    missing: pd.DataFrame,
    columns: list[str],
    block_length: int,
    blocks_per_column: int,
    seed: int,
) -> dict[str, set[int]]:
    rng = random.Random(seed)
    output: dict[str, set[int]] = {}
    for column in columns:
        trusted = set(np.flatnonzero(~missing[column].to_numpy()).tolist())
        starts: list[int] = []
        for run_start, run_end in _trusted_runs(trusted):
            if run_end - run_start + 1 >= block_length:
                starts.extend(range(run_start, run_end - block_length + 2))
        rng.shuffle(starts)
        chosen: list[int] = []
        used: set[int] = set()
        for start in starts:
            indices = set(range(start, start + block_length))
            if indices & used:
                continue
            chosen.append(start)
            used.update(indices)
            if len(chosen) >= blocks_per_column:
                break
        output[column] = {
            index for start in sorted(chosen) for index in range(start, start + block_length)
        }
    return output


def apply_pseudo_mask(missing: pd.DataFrame, pseudo_mask: dict[str, set[int]]) -> pd.DataFrame:
    output = missing.copy()
    for column, indices in pseudo_mask.items():
        if indices:
            output.loc[sorted(indices), column] = True
    return output


def write_pseudo_mask(
    path: str | Path,
    dates: pd.Series,
    values: pd.DataFrame,
    pseudo_mask: dict[str, set[int]],
) -> None:
    rows = []
    for column, indices in pseudo_mask.items():
        for index in sorted(indices):
            rows.append(
                {
                    "row_index": index,
                    "date": dates.iloc[index],
                    "column": column,
                    "true_value": values.iloc[index][column],
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)


def completion_metrics(
    truth: pd.DataFrame,
    prediction: pd.DataFrame,
    original_missing: pd.DataFrame,
    pseudo_mask: dict[str, set[int]],
) -> dict:
    by_column: dict[str, dict[str, float | int]] = {}
    for column, indices_set in pseudo_mask.items():
        indices = sorted(indices_set)
        if not indices:
            continue
        true_values = truth.iloc[indices][column].to_numpy(dtype=float)
        pred_values = prediction.iloc[indices][column].to_numpy(dtype=float)
        errors = pred_values - true_values
        trusted = truth.loc[~original_missing[column], column].to_numpy(dtype=float)
        scale = float(np.std(trusted)) if len(trusted) else 1.0
        by_column[column] = {
            "count": len(indices),
            "MAE": float(np.mean(np.abs(errors))),
            "RMSE": float(np.sqrt(np.mean(errors**2))),
            "nRMSE_std": float(np.sqrt(np.mean((errors / max(scale, EPS)) ** 2))),
        }
    keys = ("MAE", "RMSE", "nRMSE_std")
    macro = {
        "count": int(sum(int(row["count"]) for row in by_column.values())),
        "column_count": len(by_column),
        **{key: mean(float(row[key]) for row in by_column.values()) for key in keys},
    }
    return {"macro_overall": macro, "by_column": by_column}


def aggregate_seed_metrics(records: list[dict]) -> dict:
    output: dict[str, dict] = {}
    for model_name in sorted({str(record["model"]) for record in records}):
        rows = [record for record in records if record["model"] == model_name]
        output[model_name] = {"seeds": [row["seed"] for row in rows]}
        for metric in ("MAE", "RMSE", "nRMSE_std"):
            values = [float(row["macro_overall"][metric]) for row in rows]
            output[model_name][metric] = {
                "mean": mean(values),
                "std": stdev(values) if len(values) > 1 else 0.0,
                "n": len(values),
            }
    return output
