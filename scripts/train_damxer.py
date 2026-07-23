#!/usr/bin/env python3
"""End-to-end DamXer implementation for 2 h displacement forecasting.

This is a small TimeXer-inspired model, not a direct TimeXer port. The target
dx history is encoded as patch tokens. Engineered H/SEEP/temp lag features are
encoded as exogenous lag tokens. A target global token cross-attends the ENV
lag tokens, then the model directly predicts dx future values end to end.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset

try:
    from aexp_events import metric as aexp_metric
    from aexp_events import note as aexp_note
    from aexp_events import param as aexp_param
    from aexp_events import progress as aexp_progress
except Exception:
    def aexp_metric(*args, **kwargs):
        return None

    def aexp_note(*args, **kwargs):
        return None

    def aexp_param(*args, **kwargs):
        return None

    def aexp_progress(*args, **kwargs):
        return None


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str, gpu: int) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is unavailable")
        return torch.device(f"cuda:{gpu}")
    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("--device mps requested but MPS is unavailable")
        return torch.device("mps")
    if requested == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device(f"cuda:{gpu}")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _parse_date_bound(value: str, *, is_end: bool) -> pd.Timestamp | None:
    value = value.strip()
    if not value:
        return None
    if re.fullmatch(r"\d{4}-\d{1,2}", value):
        start = pd.Timestamp(value + "-01")
        return start + pd.DateOffset(months=1) if is_end else start
    ts = pd.Timestamp(value)
    if is_end and re.fullmatch(r"\d{4}-\d{1,2}-\d{1,2}", value):
        return ts + pd.DateOffset(days=1)
    return ts


def filter_by_date_window(
    dx_df: pd.DataFrame,
    eng_df: pd.DataFrame,
    mask_df: pd.DataFrame,
    target_df: pd.DataFrame | None,
    date_start: str,
    date_end: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame | None, dict]:
    if not date_start and not date_end:
        return dx_df, eng_df, mask_df, target_df, {
            "enabled": False,
            "date_start": "",
            "date_end": "",
            "rows_before": int(len(dx_df)),
            "rows_after": int(len(dx_df)),
        }
    if "date" not in dx_df.columns:
        raise ValueError("--date-start/--date-end require a date column in dx dataframe")
    for name, frame in (("engineered", eng_df), ("mask", mask_df), ("target", target_df)):
        if frame is not None and "date" not in frame.columns:
            raise ValueError(f"--date-start/--date-end require a date column in {name} dataframe")
    dates = pd.to_datetime(dx_df["date"])
    start = _parse_date_bound(date_start, is_end=False)
    end = _parse_date_bound(date_end, is_end=True)
    keep = pd.Series(True, index=dx_df.index)
    if start is not None:
        keep &= dates >= start
    if end is not None:
        keep &= dates < end
    if not bool(keep.any()):
        raise ValueError(f"date filter kept zero rows: start={date_start!r}, end={date_end!r}")
    keep_idx = keep.to_numpy()
    rows_before = int(len(dx_df))

    def take(frame: pd.DataFrame | None) -> pd.DataFrame | None:
        if frame is None:
            return None
        if len(frame) != rows_before:
            raise ValueError("date-filtered frames must have matching row counts before filtering")
        out = frame.loc[keep_idx].reset_index(drop=True)
        return out

    dx_out = take(dx_df)
    eng_out = take(eng_df)
    mask_out = take(mask_df)
    target_out = take(target_df)
    assert dx_out is not None and eng_out is not None and mask_out is not None
    info = {
        "enabled": True,
        "date_start": date_start,
        "date_end": date_end,
        "inclusive_start": str(start) if start is not None else "",
        "exclusive_end": str(end) if end is not None else "",
        "rows_before": rows_before,
        "rows_after": int(len(dx_out)),
        "first_date": str(dx_out["date"].iloc[0]),
        "last_date": str(dx_out["date"].iloc[-1]),
    }
    return dx_out, eng_out, mask_out, target_out, info


def observed_metric(pred: np.ndarray, true: np.ndarray, observed: np.ndarray) -> dict:
    err = pred - true
    denom = float(np.sum(observed))
    if denom <= 0:
        return {
            "mse": None,
            "mae": None,
            "rmse": None,
            "observed_count": 0,
            "observed_ratio": 0.0,
            "diff_mse": None,
            "diff_mae": None,
            "diff_rmse": None,
            "diff_abs_ratio": None,
            "diff_energy_ratio": None,
            "peak_diff_mse": None,
            "peak_diff_abs_ratio": None,
            "curvature_mse": None,
            "curvature_energy_ratio": None,
        }
    mse = float(np.sum(err * err * observed) / denom)
    mae = float(np.sum(np.abs(err) * observed) / denom)
    metrics = {
        "mse": mse,
        "mae": mae,
        "rmse": math.sqrt(mse),
        "observed_count": int(np.sum(observed)),
        "observed_ratio": float(np.mean(observed)),
        "diff_mse": None,
        "diff_mae": None,
        "diff_rmse": None,
        "diff_abs_ratio": None,
        "diff_energy_ratio": None,
        "peak_diff_mse": None,
        "peak_diff_abs_ratio": None,
        "curvature_mse": None,
        "curvature_energy_ratio": None,
    }
    if pred.shape[1] >= 2:
        pred_diff = pred[:, 1:, :] - pred[:, :-1, :]
        true_diff = true[:, 1:, :] - true[:, :-1, :]
        diff_observed = observed[:, 1:, :] * observed[:, :-1, :]
        diff_denom = float(np.sum(diff_observed))
        if diff_denom > 0:
            diff_err = pred_diff - true_diff
            diff_mse = float(np.sum(diff_err * diff_err * diff_observed) / diff_denom)
            diff_mae = float(np.sum(np.abs(diff_err) * diff_observed) / diff_denom)
            pred_abs = float(np.sum(np.abs(pred_diff) * diff_observed) / diff_denom)
            true_abs = float(np.sum(np.abs(true_diff) * diff_observed) / diff_denom)
            pred_energy = float(np.sum(pred_diff * pred_diff * diff_observed) / diff_denom)
            true_energy = float(np.sum(true_diff * true_diff * diff_observed) / diff_denom)
            metrics.update(
                {
                    "diff_mse": diff_mse,
                    "diff_mae": diff_mae,
                    "diff_rmse": math.sqrt(diff_mse),
                    "diff_abs_ratio": pred_abs / true_abs if true_abs > 1e-12 else None,
                    "diff_energy_ratio": math.sqrt(pred_energy / true_energy) if true_energy > 1e-12 else None,
                }
            )
            true_abs_values = np.abs(true_diff)[diff_observed > 0]
            if true_abs_values.size:
                threshold = float(np.quantile(true_abs_values, 0.9))
                peak_observed = diff_observed * (np.abs(true_diff) >= threshold)
                peak_denom = float(np.sum(peak_observed))
                if peak_denom > 0:
                    peak_mse = float(np.sum(diff_err * diff_err * peak_observed) / peak_denom)
                    peak_pred_abs = float(np.sum(np.abs(pred_diff) * peak_observed) / peak_denom)
                    peak_true_abs = float(np.sum(np.abs(true_diff) * peak_observed) / peak_denom)
                    metrics["peak_diff_mse"] = peak_mse
                    metrics["peak_diff_abs_ratio"] = peak_pred_abs / peak_true_abs if peak_true_abs > 1e-12 else None
    if pred.shape[1] >= 3:
        pred_curv = pred[:, 2:, :] - 2.0 * pred[:, 1:-1, :] + pred[:, :-2, :]
        true_curv = true[:, 2:, :] - 2.0 * true[:, 1:-1, :] + true[:, :-2, :]
        curv_observed = observed[:, 2:, :] * observed[:, 1:-1, :] * observed[:, :-2, :]
        curv_denom = float(np.sum(curv_observed))
        if curv_denom > 0:
            curv_err = pred_curv - true_curv
            curv_mse = float(np.sum(curv_err * curv_err * curv_observed) / curv_denom)
            pred_energy = float(np.sum(pred_curv * pred_curv * curv_observed) / curv_denom)
            true_energy = float(np.sum(true_curv * true_curv * curv_observed) / curv_denom)
            metrics["curvature_mse"] = curv_mse
            metrics["curvature_energy_ratio"] = math.sqrt(pred_energy / true_energy) if true_energy > 1e-12 else None
    return metrics


def masked_point_loss(
    pred: torch.Tensor,
    true: torch.Tensor,
    observed: torch.Tensor,
    loss_type: str,
    huber_delta: float,
) -> torch.Tensor:
    if loss_type == "huber":
        err = torch.nn.functional.huber_loss(pred, true, reduction="none", delta=huber_delta)
    else:
        err = (pred - true) ** 2
    return (err * observed).sum() / observed.sum().clamp_min(1.0)


def masked_mse_loss(pred: torch.Tensor, true: torch.Tensor, observed: torch.Tensor) -> torch.Tensor:
    err = (pred - true) ** 2
    return (err * observed).sum() / observed.sum().clamp_min(1.0)


def masked_temporal_diff_loss(
    pred: torch.Tensor,
    true: torch.Tensor,
    observed: torch.Tensor,
    loss_type: str,
    huber_delta: float,
) -> torch.Tensor:
    if pred.shape[1] < 2:
        return pred.new_tensor(0.0)
    pred_diff = pred[:, 1:, :] - pred[:, :-1, :]
    true_diff = true[:, 1:, :] - true[:, :-1, :]
    diff_observed = observed[:, 1:, :] * observed[:, :-1, :]
    if float(diff_observed.sum().detach().cpu()) <= 0.0:
        return pred.new_tensor(0.0)
    if loss_type == "huber":
        err = torch.nn.functional.huber_loss(pred_diff, true_diff, reduction="none", delta=huber_delta)
    else:
        err = (pred_diff - true_diff) ** 2
    return (err * diff_observed).sum() / diff_observed.sum().clamp_min(1.0)


def masked_diff_amplitude_loss(pred: torch.Tensor, true: torch.Tensor, observed: torch.Tensor) -> torch.Tensor:
    if pred.shape[1] < 2:
        return pred.new_tensor(0.0)
    pred_diff = pred[:, 1:, :] - pred[:, :-1, :]
    true_diff = true[:, 1:, :] - true[:, :-1, :]
    diff_observed = observed[:, 1:, :] * observed[:, :-1, :]
    if float(diff_observed.sum().detach().cpu()) <= 0.0:
        return pred.new_tensor(0.0)
    pred_amp = (pred_diff.abs() * diff_observed).sum() / diff_observed.sum().clamp_min(1.0)
    true_amp = (true_diff.abs() * diff_observed).sum() / diff_observed.sum().clamp_min(1.0)
    return ((pred_amp - true_amp) / true_amp.clamp_min(1e-6)) ** 2


def masked_second_diff_loss(
    pred: torch.Tensor,
    true: torch.Tensor,
    observed: torch.Tensor,
    loss_type: str,
    huber_delta: float,
) -> torch.Tensor:
    if pred.shape[1] < 3:
        return pred.new_tensor(0.0)
    pred_curv = pred[:, 2:, :] - 2.0 * pred[:, 1:-1, :] + pred[:, :-2, :]
    true_curv = true[:, 2:, :] - 2.0 * true[:, 1:-1, :] + true[:, :-2, :]
    curv_observed = observed[:, 2:, :] * observed[:, 1:-1, :] * observed[:, :-2, :]
    if float(curv_observed.sum().detach().cpu()) <= 0.0:
        return pred.new_tensor(0.0)
    if loss_type == "huber":
        err = torch.nn.functional.huber_loss(pred_curv, true_curv, reduction="none", delta=huber_delta)
    else:
        err = (pred_curv - true_curv) ** 2
    return (err * curv_observed).sum() / curv_observed.sum().clamp_min(1.0)


def validation_selection_score(metrics: dict, args) -> float:
    score = float(metrics["mse"])
    if args.selection_diff_lambda > 0.0 and metrics.get("diff_mse") is not None:
        score += args.selection_diff_lambda * float(metrics["diff_mse"])
    if args.selection_peak_lambda > 0.0 and metrics.get("peak_diff_mse") is not None:
        score += args.selection_peak_lambda * float(metrics["peak_diff_mse"])
    if args.selection_amp_lambda > 0.0 and metrics.get("diff_abs_ratio") is not None:
        amp_err = 1.0 - float(metrics["diff_abs_ratio"])
        score += args.selection_amp_lambda * amp_err * amp_err
    return score


def masked_moving_average(values: torch.Tensor, observed: torch.Tensor, window: int) -> torch.Tensor:
    if window <= 1:
        return values
    if window % 2 == 0:
        raise ValueError("decomposition moving-average window must be odd")
    pad = window // 2
    values_t = (values * observed).transpose(1, 2)
    observed_t = observed.transpose(1, 2)
    numerator = F.avg_pool1d(F.pad(values_t, (pad, pad), mode="replicate"), kernel_size=window, stride=1) * window
    denominator = F.avg_pool1d(F.pad(observed_t, (pad, pad), mode="replicate"), kernel_size=window, stride=1) * window
    return (numerator / denominator.clamp_min(1e-6)).transpose(1, 2)


def masked_decomposition_losses(
    pred: torch.Tensor,
    true: torch.Tensor,
    observed: torch.Tensor,
    window: int,
    loss_type: str,
    huber_delta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    pred_trend = masked_moving_average(pred, observed, window)
    true_trend = masked_moving_average(true, observed, window)
    pred_residual = pred - pred_trend
    true_residual = true - true_trend
    trend_loss = masked_point_loss(pred_trend, true_trend, observed, loss_type, huber_delta)
    residual_loss = masked_point_loss(pred_residual, true_residual, observed, loss_type, huber_delta)
    return trend_loss, residual_loss


class DamLagXerDataset(Dataset):
    def __init__(
        self,
        dx_df: pd.DataFrame,
        eng_df: pd.DataFrame,
        mask_df: pd.DataFrame,
        split: str,
        args,
        target_df: pd.DataFrame | None = None,
        dx_scaler: StandardScaler | None = None,
        env_scaler: StandardScaler | None = None,
    ):
        self.args = args
        self.split = split
        self.target_cols = [c for c in dx_df.columns if c != "date"][: args.target_dim]
        self.dates = dx_df["date"].astype(str).to_numpy() if "date" in dx_df.columns else np.arange(len(dx_df)).astype(str)
        self.eng_cols = [c for c in eng_df.columns if c != "date"]
        self.env_cols = self.eng_cols[args.target_dim :]
        if not self.env_cols:
            raise ValueError("engineered dataframe has no env columns after target_dim")

        self.dx_values = dx_df[self.target_cols].to_numpy(dtype=np.float32)
        if target_df is not None:
            if dx_df["date"].astype(str).tolist() != target_df["date"].astype(str).tolist():
                raise ValueError("dx input and target date columns differ")
            missing_targets = [col for col in self.target_cols if col not in target_df.columns]
            if missing_targets:
                raise ValueError(f"target CSV is missing target columns: {missing_targets[:5]}")
            self.target_values = target_df[self.target_cols].to_numpy(dtype=np.float32)
        else:
            self.target_values = self.dx_values
        self.env_values = eng_df[self.env_cols].to_numpy(dtype=np.float32)
        n = len(dx_df)
        n_train = int(n * 0.7)
        n_test = int(n * 0.2)
        n_val = n - n_train - n_test
        self.split_bounds = {
            "train": (0, n_train),
            "val": (n_train, n_train + n_val),
            "test": (n - n_test, n),
        }
        if dx_scaler is None:
            dx_scaler = StandardScaler().fit(self.dx_values[:n_train])
        if env_scaler is None:
            env_scaler = StandardScaler().fit(self.env_values[:n_train])
        self.dx_scaler = dx_scaler
        self.env_scaler = env_scaler
        self.dx_scaled = dx_scaler.transform(self.dx_values).astype(np.float32)
        self.target_scaled = dx_scaler.transform(self.target_values).astype(np.float32)
        self.env_scaled = env_scaler.transform(self.env_values).astype(np.float32)

        mask_cols = [f"{c}_masked" for c in self.target_cols]
        missing = [c for c in mask_cols if c not in mask_df.columns]
        if missing:
            raise ValueError(f"mask_csv is missing target mask columns: {missing[:5]}")
        masked = mask_df[mask_cols].to_numpy(dtype=np.float32)
        self.observed = (1.0 - masked).astype(np.float32)

        split_start, split_end = self.split_bounds[split]
        start = max(args.dx_seq_len, args.env_seq_len)
        if split != "train":
            start = max(start, split_start)
        stop = split_end - args.pred_len + 1
        candidates = list(range(start, max(start, stop)))
        if split == "train":
            candidates = [idx for idx in candidates if idx + args.pred_len <= split_end]
        if args.drop_unobserved_windows:
            candidates = [
                idx
                for idx in candidates
                if float(self.observed[idx : idx + args.pred_len].sum()) > 0.0
            ]
        self.sample_starts = candidates

    def __len__(self):
        return len(self.sample_starts)

    def __getitem__(self, index):
        pred_start = self.sample_starts[index]
        dx_start = pred_start - self.args.dx_seq_len
        env_start = pred_start - self.args.env_seq_len
        pred_end = pred_start + self.args.pred_len
        return (
            self.dx_scaled[dx_start:pred_start],
            self.env_scaled[env_start:pred_start],
            self.target_scaled[pred_start:pred_end],
            self.observed[pred_start:pred_end],
        )


def build_token_specs(env_cols: list[str], token_mode: str = "lag", token_filter_regex: str = "") -> list[dict]:
    specs = []
    for idx, col in enumerate(env_cols):
        match = re.match(
            r"^(H|seep|temp)_mean_(lag|delta_lag|rollmean|rollstd|slope|rollrange|pos_delta_sum|neg_delta_sum)(\d+)$",
            col,
        )
        if not match:
            continue
        family, kind, lag = match.group(1), match.group(2), int(match.group(3))
        kind_name = {
            "lag": "level",
            "delta_lag": "delta",
            "rollmean": "rollmean",
            "rollstd": "rollstd",
            "slope": "slope",
            "rollrange": "rollrange",
            "pos_delta_sum": "pos_delta_sum",
            "neg_delta_sum": "neg_delta_sum",
        }[kind]
        specs.append(
            {
                "index": idx,
                "column": col,
                "family": family,
                "kind": kind_name,
                "lag": lag,
                "token": f"{family}:{kind_name}:{lag}",
            }
        )
    if not specs:
        raise ValueError("no H/SEEP/temp lag-token columns found")
    if token_mode == "no_lag":
        min_lag_by_group: dict[tuple[str, str], int] = {}
        for spec in specs:
            group = (spec["family"], spec["kind"])
            min_lag_by_group[group] = min(int(spec["lag"]), min_lag_by_group.get(group, int(spec["lag"])))
        specs = [spec for spec in specs if int(spec["lag"]) == min_lag_by_group[(spec["family"], spec["kind"])]]
    elif token_mode != "lag":
        raise ValueError(f"unknown env token mode {token_mode!r}; expected 'lag' or 'no_lag'")
    if token_filter_regex:
        pattern = re.compile(token_filter_regex)
        specs = [
            spec
            for spec in specs
            if pattern.search(spec["token"]) or pattern.search(spec["column"])
        ]
        if not specs:
            raise ValueError(f"env token filter kept no tokens: {token_filter_regex!r}")
    return sorted(specs, key=lambda item: (item["family"], item["kind"], item["lag"]))


class DamLagXer(nn.Module):
    def __init__(
        self,
        target_dim: int,
        env_cols: list[str],
        dx_seq_len: int,
        pred_len: int,
        patch_len: int,
        patch_stride: int,
        hidden: int,
        n_heads: int,
        e_layers: int,
        dropout: float,
        env_mode: str = "full",
        env_token_mode: str = "lag",
        env_token_filter_regex: str = "",
        hf_branch: bool = False,
        hf_hidden: int = 32,
        hf_gated: bool = False,
        hf_gmax: float = 0.3,
        decomp_branch: bool = False,
        channel_attn: bool = False,
        channel_layers: int = 1,
    ):
        super().__init__()
        self.target_dim = target_dim
        self.pred_len = pred_len
        self.patch_len = patch_len
        self.patch_stride = patch_stride
        self.env_mode = env_mode
        self.env_token_mode = env_token_mode
        self.env_token_filter_regex = env_token_filter_regex
        self.hf_branch = hf_branch
        self.hf_gated = hf_gated
        self.hf_gmax = hf_gmax
        self.decomp_branch = decomp_branch
        self.channel_attn = channel_attn
        if env_mode == "none":
            self.token_specs = []
            self.token_indices = torch.tensor([], dtype=torch.long)
        else:
            self.token_specs = build_token_specs(env_cols, env_token_mode, env_token_filter_regex)
            self.token_indices = torch.tensor([spec["index"] for spec in self.token_specs], dtype=torch.long)
        n_patches = 1 + max(0, (dx_seq_len - patch_len) // patch_stride)
        self.patch_proj = nn.Linear(patch_len, hidden)
        self.patch_pos = nn.Parameter(torch.zeros(1, n_patches, hidden))
        self.channel_embed = nn.Embedding(target_dim, hidden)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=n_heads,
            dim_feedforward=hidden * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.target_encoder = nn.TransformerEncoder(encoder_layer, num_layers=e_layers)
        if channel_attn:
            channel_layer = nn.TransformerEncoderLayer(
                d_model=hidden,
                nhead=n_heads,
                dim_feedforward=hidden * 4,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.channel_encoder = nn.TransformerEncoder(channel_layer, num_layers=channel_layers)
        self.env_value_proj = nn.Linear(3, hidden)
        self.env_token_embed = nn.Embedding(max(1, len(self.token_specs)), hidden)
        self.cross_attn = nn.MultiheadAttention(hidden, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(hidden)
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, pred_len),
        )
        if decomp_branch:
            self.residual_head = nn.Sequential(
                nn.Linear(hidden * 2, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, pred_len),
            )
        if hf_branch:
            self.hf_conv = nn.Sequential(
                nn.Conv1d(1, hf_hidden, kernel_size=5, padding=2),
                nn.GELU(),
                nn.Conv1d(hf_hidden, hf_hidden, kernel_size=3, padding=2, dilation=2),
                nn.GELU(),
                nn.AdaptiveAvgPool1d(16),
                nn.Flatten(),
            )
            self.hf_head = nn.Linear(hf_hidden * 16, pred_len)
            nn.init.zeros_(self.hf_head.weight)
            nn.init.zeros_(self.hf_head.bias)
            if hf_gated:
                self.hf_gate = nn.Linear(hf_hidden * 16, pred_len)
                nn.init.zeros_(self.hf_gate.weight)
                nn.init.zeros_(self.hf_gate.bias)
            self.hf_scale = nn.Parameter(torch.tensor(0.1))
        self.last_attn_mean: torch.Tensor | None = None

    def forward(self, x_dx: torch.Tensor, x_env: torch.Tensor) -> torch.Tensor:
        batch, _, channels = x_dx.shape
        patches = x_dx.transpose(1, 2).unfold(dimension=-1, size=self.patch_len, step=self.patch_stride)
        # [B, C, N, P] -> [B*C, N, P]
        n_patches = patches.shape[2]
        patch_tokens = self.patch_proj(patches.reshape(batch * channels, n_patches, self.patch_len))
        patch_tokens = patch_tokens + self.patch_pos[:, :n_patches]
        patch_tokens = self.target_encoder(patch_tokens)
        channel_state = patch_tokens.mean(dim=1).reshape(batch, channels, -1)
        channel_ids = torch.arange(channels, device=x_dx.device)
        channel_state = channel_state + self.channel_embed(channel_ids).unsqueeze(0)
        if self.channel_attn:
            channel_state = self.channel_encoder(channel_state)
        target_query = channel_state.mean(dim=1, keepdim=True)

        if self.env_mode == "none":
            self.last_attn_mean = None
            context = torch.zeros_like(channel_state)
        else:
            token_indices = self.token_indices.to(x_env.device)
            env_series = x_env.index_select(dim=2, index=token_indices)
            env_stats = torch.stack(
                [
                    env_series.mean(dim=1),
                    env_series[:, -1, :],
                    env_series[:, -1, :] - env_series[:, 0, :],
                ],
                dim=-1,
            )
            token_ids = torch.arange(len(self.token_specs), device=x_env.device)
            env_tokens = self.env_value_proj(env_stats) + self.env_token_embed(token_ids).unsqueeze(0)
            context, attn = self.cross_attn(target_query, env_tokens, env_tokens, need_weights=True, average_attn_weights=False)
            self.last_attn_mean = attn.detach().mean(dim=(0, 1, 2)).cpu()
            context = self.norm(context + target_query).repeat(1, channels, 1)
        fused = torch.cat([channel_state, context], dim=-1)
        pred = self.head(fused).transpose(1, 2)
        if self.decomp_branch:
            pred = pred + self.residual_head(fused).transpose(1, 2)
        if self.hf_branch:
            dx_diff = torch.zeros_like(x_dx)
            dx_diff[:, 1:, :] = x_dx[:, 1:, :] - x_dx[:, :-1, :]
            hf_input = dx_diff.transpose(1, 2).reshape(batch * channels, 1, -1)
            hf_features = self.hf_conv(hf_input)
            hf_pred = self.hf_head(hf_features)
            if self.hf_gated:
                gate = torch.sigmoid(self.hf_gate(hf_features))
                hf_pred = self.hf_gmax * gate * torch.tanh(hf_pred)
            else:
                hf_pred = self.hf_scale * hf_pred
            hf_pred = hf_pred.reshape(batch, channels, self.pred_len).transpose(1, 2)
            pred = pred + hf_pred
        return pred

    def attention_summary(self) -> dict:
        if not self.token_specs:
            return {"tokens": [], "top_tokens": [], "grouped": {}}
        if self.last_attn_mean is None:
            weights = np.zeros(len(self.token_specs), dtype=np.float32)
        else:
            weights = self.last_attn_mean.numpy()
        rows = []
        grouped: dict[str, float] = {}
        for spec, weight in zip(self.token_specs, weights):
            value = float(weight)
            row = {**spec, "weight": value}
            rows.append(row)
            grouped[spec["family"]] = grouped.get(spec["family"], 0.0) + value
            grouped[f"{spec['family']}:{spec['kind']}"] = grouped.get(f"{spec['family']}:{spec['kind']}", 0.0) + value
        return {
            "tokens": rows,
            "top_tokens": sorted(rows, key=lambda item: item["weight"], reverse=True)[:12],
            "grouped": grouped,
        }


def make_loaders(dx_df, eng_df, mask_df, args, target_df=None):
    train_ds = DamLagXerDataset(dx_df, eng_df, mask_df, "train", args, target_df)
    val_ds = DamLagXerDataset(dx_df, eng_df, mask_df, "val", args, target_df, train_ds.dx_scaler, train_ds.env_scaler)
    test_ds = DamLagXerDataset(dx_df, eng_df, mask_df, "test", args, target_df, train_ds.dx_scaler, train_ds.env_scaler)
    loaders = {
        "train": DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers),
        "val": DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers),
        "test": DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers),
    }
    return train_ds, val_ds, test_ds, loaders


def make_prediction(
    model: nn.Module,
    x_dx: torch.Tensor,
    x_env: torch.Tensor,
    predict_mode: str,
    use_revin: bool,
    revin_eps: float,
) -> torch.Tensor:
    if use_revin:
        mean = x_dx.mean(dim=1, keepdim=True).detach()
        var = torch.var(x_dx, dim=1, keepdim=True, unbiased=False)
        stdev = torch.sqrt(var + revin_eps).detach()
        model_input = (x_dx - mean) / stdev
        raw_pred = model(model_input, x_env)
        if predict_mode == "residual_last":
            pred_norm = raw_pred + model_input[:, -1:, :]
        else:
            pred_norm = raw_pred
        return pred_norm * stdev + mean
    raw_pred = model(x_dx, x_env)
    if predict_mode == "residual_last":
        return raw_pred + x_dx[:, -1:, :]
    return raw_pred


def run_eval(model, loader, device, predict_mode: str, use_revin: bool, revin_eps: float):
    model.eval()
    preds, trues, observed = [], [], []
    with torch.no_grad():
        for x_dx, x_env, y, y_observed in loader:
            x_dx = x_dx.float().to(device)
            x_env = x_env.float().to(device)
            pred = make_prediction(model, x_dx, x_env, predict_mode, use_revin, revin_eps)
            preds.append(pred.cpu().numpy())
            trues.append(y.numpy())
            observed.append(y_observed.numpy())
    pred_np = np.concatenate(preds, axis=0)
    true_np = np.concatenate(trues, axis=0)
    observed_np = np.concatenate(observed, axis=0)
    return observed_metric(pred_np, true_np, observed_np)


def inverse_dx(dataset: DamLagXerDataset, values: np.ndarray) -> np.ndarray:
    shape = values.shape
    flat = values.reshape(-1, shape[-1])
    restored = dataset.dx_scaler.inverse_transform(flat)
    return restored.reshape(shape)


def collect_prediction_windows(
    model,
    loader,
    device,
    dataset: DamLagXerDataset,
    predict_mode: str,
    use_revin: bool,
    revin_eps: float,
) -> dict:
    model.eval()
    preds, trues, observed = [], [], []
    with torch.no_grad():
        for x_dx, x_env, y, y_observed in loader:
            x_dx = x_dx.float().to(device)
            x_env = x_env.float().to(device)
            pred = make_prediction(model, x_dx, x_env, predict_mode, use_revin, revin_eps)
            preds.append(pred.cpu().numpy())
            trues.append(y.numpy())
            observed.append(y_observed.numpy())
    pred_scaled = np.concatenate(preds, axis=0)
    true_scaled = np.concatenate(trues, axis=0)
    observed_np = np.concatenate(observed, axis=0)
    output_start_abs = np.asarray(dataset.sample_starts, dtype=np.int64)
    return {
        "pred": inverse_dx(dataset, pred_scaled),
        "true": inverse_dx(dataset, true_scaled),
        "pred_scaled": pred_scaled,
        "true_scaled": true_scaled,
        "observed": observed_np,
        "output_start_abs": output_start_abs,
        "feature_cols": np.asarray(dataset.target_cols, dtype=object),
        "date": np.asarray(dataset.dates, dtype=object),
        "split": np.asarray([dataset.split], dtype=object),
    }


def parse_sensitivity_specs(spec_text: str) -> list[dict]:
    specs = []
    for item in [part.strip() for part in spec_text.split(",") if part.strip()]:
        parts = item.split(":")
        kind = parts[0]
        if kind == "family" and len(parts) == 2:
            specs.append({"name": item, "type": kind, "family": parts[1]})
        elif kind == "lag" and len(parts) == 2:
            specs.append({"name": item, "type": kind, "lag": int(parts[1])})
        elif kind == "family_lag" and len(parts) == 3:
            specs.append({"name": item, "type": kind, "family": parts[1], "lag": int(parts[2])})
        elif kind == "token" and len(parts) == 4:
            specs.append({"name": item, "type": kind, "token": f"{parts[1]}:{parts[2]}:{parts[3]}"})
        else:
            raise ValueError(
                "invalid sensitivity spec "
                f"{item!r}; use family:H, lag:84, family_lag:H:84, or token:H:level:84"
            )
    return specs


def token_matches_spec(token: dict, spec: dict) -> bool:
    if spec["type"] == "family":
        return token["family"] == spec["family"]
    if spec["type"] == "lag":
        return int(token["lag"]) == int(spec["lag"])
    if spec["type"] == "family_lag":
        return token["family"] == spec["family"] and int(token["lag"]) == int(spec["lag"])
    if spec["type"] == "token":
        return token["token"] == spec["token"]
    return False


def mask_env_by_token_indices(x_env: torch.Tensor, token_indices: list[int]) -> torch.Tensor:
    if not token_indices:
        return x_env
    masked = x_env.clone()
    # Inputs are standardized, so zeroing corresponds to replacing the selected
    # environmental token series with its training-set mean.
    masked[:, :, token_indices] = 0.0
    return masked


def run_eval_with_env_token_mask(
    model,
    loader,
    device,
    predict_mode: str,
    use_revin: bool,
    revin_eps: float,
    token_indices: list[int],
):
    model.eval()
    preds, trues, observed = [], [], []
    with torch.no_grad():
        for x_dx, x_env, y, y_observed in loader:
            x_dx = x_dx.float().to(device)
            x_env = x_env.float().to(device)
            x_env = mask_env_by_token_indices(x_env, token_indices)
            pred = make_prediction(model, x_dx, x_env, predict_mode, use_revin, revin_eps)
            preds.append(pred.cpu().numpy())
            trues.append(y.numpy())
            observed.append(y_observed.numpy())
    pred_np = np.concatenate(preds, axis=0)
    true_np = np.concatenate(trues, axis=0)
    observed_np = np.concatenate(observed, axis=0)
    return observed_metric(pred_np, true_np, observed_np)


def collect_lag_sensitivity(model, loaders, device, args, spec_text: str) -> dict:
    specs = parse_sensitivity_specs(spec_text)
    results = []
    for spec in specs:
        matched = [token for token in model.token_specs if token_matches_spec(token, spec)]
        token_indices = [int(token["index"]) for token in matched]
        val_metrics = run_eval_with_env_token_mask(
            model, loaders["val"], device, args.predict_mode, args.revin, args.revin_eps, token_indices
        )
        test_metrics = run_eval_with_env_token_mask(
            model, loaders["test"], device, args.predict_mode, args.revin, args.revin_eps, token_indices
        )
        results.append(
            {
                "name": spec["name"],
                "type": spec["type"],
                "matched_token_count": len(matched),
                "matched_tokens": [token["token"] for token in matched],
                "val": val_metrics,
                "test": test_metrics,
            }
        )
    return {"mask_value": "standardized_zero_train_mean", "results": results}


def collect_attention_summary(model, loader, device, use_revin: bool, revin_eps: float):
    model.eval()
    acc = None
    total = 0
    with torch.no_grad():
        for x_dx, x_env, *_ in loader:
            x_dx = x_dx.float().to(device)
            x_env = x_env.float().to(device)
            if use_revin:
                mean = x_dx.mean(dim=1, keepdim=True).detach()
                var = torch.var(x_dx, dim=1, keepdim=True, unbiased=False)
                stdev = torch.sqrt(var + revin_eps).detach()
                model_input = (x_dx - mean) / stdev
            else:
                model_input = x_dx
            _ = model(model_input, x_env)
            weights = model.last_attn_mean
            if weights is None:
                continue
            batch = int(x_dx.shape[0])
            acc = weights.clone().float() * batch if acc is None else acc + weights.clone().float() * batch
            total += batch
    if acc is not None and total > 0:
        model.last_attn_mean = acc / float(total)
    return model.attention_summary()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--dx-variant", default="dam_2h_saits_dx_only_nomask.csv")
    parser.add_argument("--engineered-variant", default="dam_2h_saits_dx_engineered_env_nomask.csv")
    parser.add_argument(
        "--target-csv",
        default="",
        help=(
            "Optional separate filtered-response target CSV. History inputs are "
            "read from dx/engineered variants; future y is read from this target CSV "
            "and standardized with the input dx scaler."
        ),
    )
    parser.add_argument("--mask-csv", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dx-seq-len", type=int, default=192)
    parser.add_argument("--env-seq-len", type=int, default=720)
    parser.add_argument("--pred-len", type=int, default=96)
    parser.add_argument("--target-dim", type=int, default=89)
    parser.add_argument("--patch-len", type=int, default=16)
    parser.add_argument("--patch-stride", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--e-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--min-delta", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--loss-type", choices=["mse", "huber"], default="mse")
    parser.add_argument("--huber-delta", type=float, default=0.5)
    parser.add_argument("--smoothness-lambda", type=float, default=0.0)
    parser.add_argument("--curvature-lambda", type=float, default=0.0)
    parser.add_argument("--amplitude-lambda", type=float, default=0.0)
    parser.add_argument("--predict-mode", choices=["direct", "residual_last"], default="direct")
    parser.add_argument("--env-mode", choices=["full", "none"], default="full")
    parser.add_argument(
        "--env-token-mode",
        choices=["lag", "no_lag"],
        default="lag",
        help=(
            "Environment token ablation. 'lag' keeps all engineered lag tokens; "
            "'no_lag' keeps only the nearest available lag per family/kind."
        ),
    )
    parser.add_argument(
        "--env-token-filter-regex",
        default="",
        help=(
            "Optional regex over token labels or column names after env-token-mode filtering. "
            "Example token labels: H:level:84, seep:slope:168, temp:rollmean:360."
        ),
    )
    parser.add_argument("--hf-branch", action="store_true")
    parser.add_argument("--hf-hidden", type=int, default=32)
    parser.add_argument("--hf-gated", action="store_true")
    parser.add_argument("--hf-gmax", type=float, default=0.3)
    parser.add_argument("--decomp-branch", action="store_true")
    parser.add_argument(
        "--channel-attn",
        action="store_true",
        help="Enable iTransformer-style cross-channel self-attention over dx channel tokens.",
    )
    parser.add_argument("--channel-layers", type=int, default=1)
    parser.add_argument("--decomp-window", type=int, default=25)
    parser.add_argument("--trend-lambda", type=float, default=0.0)
    parser.add_argument("--residual-lambda", type=float, default=0.0)
    parser.add_argument("--selection-diff-lambda", type=float, default=0.0)
    parser.add_argument("--selection-peak-lambda", type=float, default=0.0)
    parser.add_argument("--selection-amp-lambda", type=float, default=0.0)
    parser.add_argument("--zero-head-init", action="store_true")
    parser.add_argument("--revin", action="store_true")
    parser.add_argument("--revin-eps", type=float, default=1e-5)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument(
        "--evaluation-schedule",
        choices=["per_epoch_test", "final_only"],
        default="final_only",
    )
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--trial-name", default="damxer")
    parser.add_argument("--event-trial", type=int, default=-1)
    parser.add_argument(
        "--date-start",
        default="",
        help="Optional inclusive date lower bound before chronological split, e.g. 2024-01.",
    )
    parser.add_argument(
        "--date-end",
        default="",
        help=(
            "Optional upper date bound before chronological split. If written as YYYY-MM, "
            "the whole month is included by using next month as the exclusive bound."
        ),
    )
    parser.add_argument("--drop-unobserved-windows", action="store_true")
    parser.add_argument("--prediction-npz", default="")
    parser.add_argument("--prediction-npz-dir", default="")
    parser.add_argument("--prediction-splits", nargs="+", choices=["val", "test"], default=["test"])
    parser.add_argument(
        "--sensitivity-specs",
        default="",
        help=(
            "Comma-separated env-token masking specs evaluated after best-checkpoint selection. "
            "Examples: family:H,lag:84,family_lag:H:84,token:H:level:84"
        ),
    )
    args = parser.parse_args()

    set_seed(args.seed)
    for name in (
        "dx_seq_len",
        "env_seq_len",
        "pred_len",
        "target_dim",
        "patch_len",
        "patch_stride",
        "hidden",
        "n_heads",
        "e_layers",
        "dropout",
        "epochs",
        "batch_size",
        "lr",
        "weight_decay",
        "loss_type",
        "huber_delta",
        "smoothness_lambda",
        "curvature_lambda",
        "amplitude_lambda",
        "predict_mode",
        "env_mode",
        "env_token_mode",
        "env_token_filter_regex",
        "hf_branch",
        "hf_hidden",
        "hf_gated",
        "hf_gmax",
        "decomp_branch",
        "channel_attn",
        "channel_layers",
        "decomp_window",
        "trend_lambda",
        "residual_lambda",
        "selection_diff_lambda",
        "selection_peak_lambda",
        "selection_amp_lambda",
        "zero_head_init",
        "revin",
        "revin_eps",
        "evaluation_schedule",
        "seed",
        "trial_name",
        "date_start",
        "date_end",
    ):
        aexp_param(name, getattr(args, name))
    aexp_note("dam-lagxer end-to-end start")

    data_root = Path(args.data_root)
    dx_df = pd.read_csv(data_root / args.dx_variant)
    eng_df = pd.read_csv(data_root / args.engineered_variant)
    target_df = pd.read_csv(args.target_csv) if args.target_csv else None
    mask_df = pd.read_csv(args.mask_csv)
    if "date" not in dx_df.columns:
        raise ValueError("dx input is missing the required date column")
    reference_dates = dx_df["date"].astype(str).tolist()
    for frame_name, frame in (("engineered input", eng_df), ("mask", mask_df), ("target", target_df)):
        if frame is None:
            continue
        if "date" not in frame.columns:
            raise ValueError(f"{frame_name} is missing the required date column")
        if frame["date"].astype(str).tolist() != reference_dates:
            raise ValueError(f"{frame_name} dates do not match the dx input")
    dx_df, eng_df, mask_df, target_df, date_filter = filter_by_date_window(
        dx_df,
        eng_df,
        mask_df,
        target_df,
        args.date_start,
        args.date_end,
    )
    if date_filter["enabled"]:
        aexp_note(
            "dam-lagxer date filter "
            f"start={date_filter['date_start']} end={date_filter['date_end']} "
            f"rows={date_filter['rows_after']}/{date_filter['rows_before']} "
            f"first={date_filter['first_date']} last={date_filter['last_date']}"
        )
    train_ds, val_ds, test_ds, loaders = make_loaders(dx_df, eng_df, mask_df, args, target_df)
    device = resolve_device(args.device, args.gpu)
    model = DamLagXer(
        target_dim=args.target_dim,
        env_cols=train_ds.env_cols,
        dx_seq_len=args.dx_seq_len,
        pred_len=args.pred_len,
        patch_len=args.patch_len,
        patch_stride=args.patch_stride,
        hidden=args.hidden,
        n_heads=args.n_heads,
        e_layers=args.e_layers,
        dropout=args.dropout,
        env_mode=args.env_mode,
        env_token_mode=args.env_token_mode,
        env_token_filter_regex=args.env_token_filter_regex,
        hf_branch=args.hf_branch,
        hf_hidden=args.hf_hidden,
        hf_gated=args.hf_gated,
        hf_gmax=args.hf_gmax,
        decomp_branch=args.decomp_branch,
        channel_attn=args.channel_attn,
        channel_layers=args.channel_layers,
    ).to(device)
    if args.zero_head_init:
        final_layer = model.head[-1]
        nn.init.zeros_(final_layer.weight)
        nn.init.zeros_(final_layer.bias)
        if args.decomp_branch:
            residual_layer = model.residual_head[-1]
            nn.init.zeros_(residual_layer.weight)
            nn.init.zeros_(residual_layer.bias)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best = {
        "score": float("inf"),
        "epoch": None,
        "state": None,
        "val_metrics": None,
        "test_metrics": None,
    }
    bad = 0
    fields = {
        "series": "damxer",
        "variant": f"dx{args.dx_seq_len}_env{args.env_seq_len}_pl{args.pred_len}",
        "trial_name": args.trial_name,
        "seed": args.seed,
        "device": str(device),
    }
    if args.event_trial >= 0:
        fields["trial"] = args.event_trial
    else:
        fields["trial"] = args.trial_name
    aexp_note(
        f"dam-lagxer windows train={len(train_ds)} val={len(val_ds)} test={len(test_ds)} "
        f"env_dim={len(train_ds.env_cols)} tokens={len(model.token_specs)}"
    )
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        point_losses = []
        smooth_losses = []
        curvature_losses = []
        amplitude_losses = []
        trend_losses = []
        residual_losses = []
        for x_dx, x_env, y, y_observed in loaders["train"]:
            x_dx = x_dx.float().to(device)
            x_env = x_env.float().to(device)
            y = y.float().to(device)
            y_observed = y_observed.float().to(device)
            pred = make_prediction(model, x_dx, x_env, args.predict_mode, args.revin, args.revin_eps)
            point_loss = masked_point_loss(pred, y, y_observed, args.loss_type, args.huber_delta)
            smooth_loss = masked_temporal_diff_loss(pred, y, y_observed, args.loss_type, args.huber_delta)
            curvature_loss = masked_second_diff_loss(pred, y, y_observed, args.loss_type, args.huber_delta)
            amplitude_loss = masked_diff_amplitude_loss(pred, y, y_observed)
            trend_loss, residual_loss = masked_decomposition_losses(
                pred,
                y,
                y_observed,
                args.decomp_window,
                args.loss_type,
                args.huber_delta,
            )
            loss = (
                point_loss
                + args.smoothness_lambda * smooth_loss
                + args.curvature_lambda * curvature_loss
                + args.amplitude_lambda * amplitude_loss
                + args.trend_lambda * trend_loss
                + args.residual_lambda * residual_loss
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            losses.append(float(loss.item()))
            point_losses.append(float(point_loss.item()))
            smooth_losses.append(float(smooth_loss.item()))
            curvature_losses.append(float(curvature_loss.item()))
            amplitude_losses.append(float(amplitude_loss.item()))
            trend_losses.append(float(trend_loss.item()))
            residual_losses.append(float(residual_loss.item()))
        train_loss = float(np.mean(losses))
        train_point_loss = float(np.mean(point_losses))
        train_smooth_loss = float(np.mean(smooth_losses))
        train_curvature_loss = float(np.mean(curvature_losses))
        train_amplitude_loss = float(np.mean(amplitude_losses))
        train_trend_loss = float(np.mean(trend_losses))
        train_residual_loss = float(np.mean(residual_losses))
        val_metrics = run_eval(model, loaders["val"], device, args.predict_mode, args.revin, args.revin_eps)
        test_metrics = (
            run_eval(model, loaders["test"], device, args.predict_mode, args.revin, args.revin_eps)
            if args.evaluation_schedule == "per_epoch_test"
            else None
        )
        selection_score = validation_selection_score(val_metrics, args)
        aexp_progress("epoch", epoch, total=args.epochs, stage="train", **fields)
        aexp_metric("train/loss", train_loss, step=epoch, epoch=epoch, stage="train", **fields)
        aexp_metric("train/point_loss", train_point_loss, step=epoch, epoch=epoch, stage="train", **fields)
        if args.smoothness_lambda > 0:
            aexp_metric("train/smoothness_loss", train_smooth_loss, step=epoch, epoch=epoch, stage="train", **fields)
        if args.curvature_lambda > 0:
            aexp_metric("train/curvature_loss", train_curvature_loss, step=epoch, epoch=epoch, stage="train", **fields)
        if args.amplitude_lambda > 0:
            aexp_metric("train/amplitude_loss", train_amplitude_loss, step=epoch, epoch=epoch, stage="train", **fields)
        if args.trend_lambda > 0:
            aexp_metric("train/trend_loss", train_trend_loss, step=epoch, epoch=epoch, stage="train", **fields)
        if args.residual_lambda > 0:
            aexp_metric("train/residual_loss", train_residual_loss, step=epoch, epoch=epoch, stage="train", **fields)
        aexp_metric("val/observed_mse", val_metrics["mse"], step=epoch, epoch=epoch, split="val", stage="eval", **fields)
        if test_metrics is not None:
            aexp_metric("test/observed_mse", test_metrics["mse"], step=epoch, epoch=epoch, split="test", stage="eval", **fields)
        for key in ("diff_mse", "diff_abs_ratio", "diff_energy_ratio", "peak_diff_mse", "peak_diff_abs_ratio", "curvature_mse", "curvature_energy_ratio"):
            if val_metrics.get(key) is not None:
                aexp_metric(f"val/{key}", val_metrics[key], step=epoch, epoch=epoch, split="val", stage="eval", **fields)
            if test_metrics is not None and test_metrics.get(key) is not None:
                aexp_metric(f"test/{key}", test_metrics[key], step=epoch, epoch=epoch, split="test", stage="eval", **fields)
        aexp_metric("val/selection_score", selection_score, step=epoch, epoch=epoch, split="val", stage="eval", **fields)
        line = (
            f"epoch={epoch} train={train_loss:.6f} val={val_metrics['mse']:.6f} "
        )
        if test_metrics is not None:
            line += f"test={test_metrics['mse']:.6f} "
        line += f"val_diff_ratio={val_metrics.get('diff_abs_ratio')} selection={selection_score:.6f}"
        print(
            line,
            flush=True,
        )
        if selection_score + args.min_delta < best["score"]:
            best = {
                "score": selection_score,
                "epoch": epoch,
                "state": copy.deepcopy(model.state_dict()),
                "val_metrics": val_metrics,
                "test_metrics": test_metrics,
            }
            bad = 0
        else:
            bad += 1
            if bad >= args.patience:
                break

    if best["state"] is None:
        raise RuntimeError("training finished without a validation-selected model state")
    model.load_state_dict(best["state"])
    if args.evaluation_schedule == "final_only":
        best["val_metrics"] = run_eval(model, loaders["val"], device, args.predict_mode, args.revin, args.revin_eps)
        best["test_metrics"] = run_eval(model, loaders["test"], device, args.predict_mode, args.revin, args.revin_eps)
    attention_summary = collect_attention_summary(model, loaders["val"], device, args.revin, args.revin_eps)
    sensitivity = collect_lag_sensitivity(model, loaders, device, args, args.sensitivity_specs) if args.sensitivity_specs else None
    prediction_npz = None
    prediction_npzs = {}
    if args.prediction_npz:
        prediction_npz = Path(args.prediction_npz)
        prediction_npz.parent.mkdir(parents=True, exist_ok=True)
        arrays = collect_prediction_windows(
            model,
            loaders["test"],
            device,
            test_ds,
            args.predict_mode,
            args.revin,
            args.revin_eps,
        )
        np.savez_compressed(prediction_npz, **arrays)
        prediction_npzs["test"] = str(prediction_npz)
        aexp_note(f"dam-lagxer saved test-window predictions to {prediction_npz}")
    if args.prediction_npz_dir:
        prediction_dir = Path(args.prediction_npz_dir)
        prediction_dir.mkdir(parents=True, exist_ok=True)
        split_loaders = {"val": (loaders["val"], val_ds), "test": (loaders["test"], test_ds)}
        for split in args.prediction_splits:
            if split not in split_loaders:
                raise ValueError(f"unknown prediction split={split!r}; expected one of {sorted(split_loaders)}")
            loader, dataset = split_loaders[split]
            arrays = collect_prediction_windows(
                model,
                loader,
                device,
                dataset,
                args.predict_mode,
                args.revin,
                args.revin_eps,
            )
            split_npz = prediction_dir / f"{args.trial_name}_{split}_windows.npz"
            np.savez_compressed(split_npz, **arrays)
            prediction_npzs[split] = str(split_npz)
            aexp_note(f"dam-lagxer saved {split}-window predictions to {split_npz}")
    summary = {
        "module": "damxer",
        "data_root": str(data_root),
        "dx_variant": args.dx_variant,
        "engineered_variant": args.engineered_variant,
        "target_csv": args.target_csv,
        "dx_seq_len": args.dx_seq_len,
        "env_seq_len": args.env_seq_len,
        "pred_len": args.pred_len,
        "target_dim": args.target_dim,
        "patch_len": args.patch_len,
        "patch_stride": args.patch_stride,
        "hidden": args.hidden,
        "n_heads": args.n_heads,
        "e_layers": args.e_layers,
        "dropout": args.dropout,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "loss_type": args.loss_type,
        "huber_delta": args.huber_delta,
        "smoothness_lambda": args.smoothness_lambda,
        "curvature_lambda": args.curvature_lambda,
        "amplitude_lambda": args.amplitude_lambda,
        "predict_mode": args.predict_mode,
        "env_mode": args.env_mode,
        "env_token_mode": args.env_token_mode,
        "env_token_filter_regex": args.env_token_filter_regex,
        "hf_branch": args.hf_branch,
        "hf_hidden": args.hf_hidden,
        "hf_gated": args.hf_gated,
        "hf_gmax": args.hf_gmax,
        "decomp_branch": args.decomp_branch,
        "channel_attn": args.channel_attn,
        "channel_layers": args.channel_layers,
        "decomp_window": args.decomp_window,
        "trend_lambda": args.trend_lambda,
        "residual_lambda": args.residual_lambda,
        "selection_diff_lambda": args.selection_diff_lambda,
        "selection_peak_lambda": args.selection_peak_lambda,
        "selection_amp_lambda": args.selection_amp_lambda,
        "zero_head_init": args.zero_head_init,
        "revin": args.revin,
        "revin_eps": args.revin_eps,
        "seed": args.seed,
        "device": str(device),
        "evaluation_schedule": args.evaluation_schedule,
        "trial_name": args.trial_name,
        "date_filter": date_filter,
        "best_epoch": best["epoch"],
        "selection": (
            "validation observed MSE"
            f" + {args.selection_diff_lambda}*val_diff_mse"
            f" + {args.selection_peak_lambda}*val_peak_diff_mse"
            f" + {args.selection_amp_lambda}*(1-val_diff_abs_ratio)^2"
        ),
        "selection_score": best["score"],
        "val": best["val_metrics"],
        "test": best["test_metrics"],
        "split_windows": {"train": len(train_ds), "val": len(val_ds), "test": len(test_ds)},
        "env_columns": train_ds.env_cols,
        "lag_tokens": model.token_specs,
        "attention_summary": attention_summary,
        "prediction_npz": str(prediction_npz) if prediction_npz else prediction_npzs.get("test"),
        "prediction_npzs": prediction_npzs,
    }
    if sensitivity is not None:
        baseline_val = best["val_metrics"]["mse"]
        baseline_test = best["test_metrics"]["mse"]
        for item in sensitivity["results"]:
            item["val_mse_delta"] = item["val"]["mse"] - baseline_val
            item["test_mse_delta"] = item["test"]["mse"] - baseline_test
            item["val_mse_relative_increase"] = item["val_mse_delta"] / baseline_val if baseline_val else None
            item["test_mse_relative_increase"] = item["test_mse_delta"] / baseline_test if baseline_test else None
        summary["lag_mask_sensitivity"] = sensitivity
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    aexp_metric("final/val_observed_mse", best["val_metrics"]["mse"], step=0, split="val", series="damxer")
    aexp_metric("final/test_observed_mse", best["test_metrics"]["mse"], step=1, split="test", series="damxer")
    for key in ("diff_mse", "diff_abs_ratio", "diff_energy_ratio", "peak_diff_mse", "peak_diff_abs_ratio", "curvature_mse", "curvature_energy_ratio"):
        if best["val_metrics"].get(key) is not None:
            aexp_metric(f"final/val_{key}", best["val_metrics"][key], step=0, split="val", series="damxer")
        if best["test_metrics"].get(key) is not None:
            aexp_metric(f"final/test_{key}", best["test_metrics"][key], step=1, split="test", series="damxer")
    aexp_note(f"dam-lagxer finished output={output}")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
