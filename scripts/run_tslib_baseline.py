#!/usr/bin/env python3
import argparse
import copy
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
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


class WindowDataset(Dataset):
    def __init__(
        self,
        df,
        seq_len,
        pred_len,
        split,
        scaler=None,
        mask_df=None,
        target_df=None,
        target_dim=82,
        drop_unobserved_windows=False,
    ):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.target_dim = target_dim
        n = len(df)
        n_train = int(n * 0.7)
        n_test = int(n * 0.2)
        n_val = n - n_train - n_test
        border1s = [0, n_train - seq_len, n - n_test - seq_len]
        border2s = [n_train, n_train + n_val, n]
        split_id = {"train": 0, "val": 1, "test": 2}[split]
        self.border1 = border1s[split_id]
        self.border2 = border2s[split_id]

        self.feature_cols = [c for c in df.columns if c != "date"]
        values = df[self.feature_cols].to_numpy(dtype=np.float32)
        if scaler is None:
            scaler = StandardScaler()
            scaler.fit(values[border1s[0] : border2s[0]])
        self.scaler = scaler
        self.data = scaler.transform(values).astype(np.float32)[self.border1 : self.border2]
        self.target_data = self.data
        if target_df is not None:
            if df["date"].astype(str).tolist() != target_df["date"].astype(str).tolist():
                raise ValueError("input CSV and target CSV date columns differ")
            target_cols = self.feature_cols[:target_dim]
            missing_targets = [col for col in target_cols if col not in target_df.columns]
            if missing_targets:
                preview = ", ".join(missing_targets[:5])
                raise ValueError(f"target_csv is missing target columns: {preview}")
            target_values = values.copy()
            target_values[:, :target_dim] = target_df[target_cols].to_numpy(dtype=np.float32)
            self.target_data = scaler.transform(target_values).astype(np.float32)[self.border1 : self.border2]
        if mask_df is None:
            self.observed = np.ones((len(self.data), target_dim), dtype=np.float32)
        else:
            mask_cols = [f"{c}_masked" for c in self.feature_cols[:target_dim]]
            missing = [c for c in mask_cols if c not in mask_df.columns]
            if missing:
                preview = ", ".join(missing[:5])
                raise ValueError(f"mask_csv is missing target mask columns: {preview}")
            masked = mask_df[mask_cols].to_numpy(dtype=np.float32)[self.border1 : self.border2]
            self.observed = (1.0 - masked).astype(np.float32)

        dates = pd.to_datetime(df["date"].iloc[self.border1 : self.border2])
        stamp = np.stack(
            [
                dates.dt.month.to_numpy(),
                dates.dt.day.to_numpy(),
                dates.dt.weekday.to_numpy(),
                dates.dt.hour.to_numpy(),
            ],
            axis=1,
        ).astype(np.float32)
        self.stamp = stamp
        base_len = len(self.data) - self.seq_len - self.pred_len + 1
        if base_len < 0:
            base_len = 0
        if drop_unobserved_windows:
            self.sample_indices = [
                index
                for index in range(base_len)
                if float(self.observed[index + self.seq_len : index + self.seq_len + self.pred_len].sum()) > 0.0
            ]
        else:
            self.sample_indices = list(range(base_len))

    def __len__(self):
        return len(self.sample_indices)

    def __getitem__(self, index):
        s_begin = self.sample_indices[index]
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len
        x = self.data[s_begin:s_end]
        y = self.target_data[r_begin:r_end]
        x_mark = self.stamp[s_begin:s_end]
        y_mark = self.stamp[r_begin:r_end]
        y_observed = self.observed[r_begin:r_end]
        return x, y, x_mark, y_mark, y_observed


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def metric(pred, true):
    err = pred - true
    mse = float(np.mean(err * err))
    mae = float(np.mean(np.abs(err)))
    rmse = math.sqrt(mse)
    return {"mse": mse, "mae": mae, "rmse": rmse}


def observed_metric(pred, true, observed):
    err = pred - true
    denom = float(np.sum(observed))
    if denom <= 0:
        return {"mse": None, "mae": None, "rmse": None, "observed_count": 0, "observed_ratio": 0.0}
    mse = float(np.sum(err * err * observed) / denom)
    mae = float(np.sum(np.abs(err) * observed) / denom)
    rmse = math.sqrt(mse)
    return {
        "mse": mse,
        "mae": mae,
        "rmse": rmse,
        "observed_count": int(np.sum(observed)),
        "observed_ratio": float(np.mean(observed)),
    }


def observed_metric_per_channel(pred, true, observed):
    err = pred - true
    denom = np.sum(observed, axis=(0, 1))
    safe_denom = np.maximum(denom, 1.0)
    mse = np.sum(err * err * observed, axis=(0, 1)) / safe_denom
    mae = np.sum(np.abs(err) * observed, axis=(0, 1)) / safe_denom
    return {
        "mse": mse.astype(float).tolist(),
        "mae": mae.astype(float).tolist(),
        "observed_count": denom.astype(int).tolist(),
    }


def parse_date_bound(text: str, *, is_end: bool) -> pd.Timestamp | None:
    if not text:
        return None
    stamp = pd.Timestamp(text)
    if is_end and len(text) == 7:
        stamp = stamp + pd.offsets.MonthBegin(1)
    return stamp


def filter_by_date_window(df, target_df=None, mask_df=None, date_start="", date_end=""):
    start = parse_date_bound(date_start, is_end=False)
    end = parse_date_bound(date_end, is_end=True)
    if start is None and end is None:
        return df, target_df, mask_df, None
    dates = pd.to_datetime(df["date"])
    keep = pd.Series(True, index=df.index)
    if start is not None:
        keep &= dates >= start
    if end is not None:
        keep &= dates < end
    if int(keep.sum()) == 0:
        raise ValueError(f"date window keeps zero rows: start={date_start!r} end={date_end!r}")
    filtered_df = df.loc[keep].reset_index(drop=True)
    filtered_target = target_df.loc[keep].reset_index(drop=True) if target_df is not None else None
    filtered_mask = mask_df.loc[keep].reset_index(drop=True) if mask_df is not None else None
    info = {
        "date_start": str(start) if start is not None else None,
        "date_end_exclusive": str(end) if end is not None else None,
        "rows_before": int(len(df)),
        "rows_after": int(len(filtered_df)),
        "first_date": str(filtered_df["date"].iloc[0]),
        "last_date": str(filtered_df["date"].iloc[-1]),
    }
    return filtered_df, filtered_target, filtered_mask, info


def masked_mse_loss(pred, true, observed):
    err = (pred - true) ** 2
    denom = observed.sum().clamp_min(1.0)
    return (err * observed).sum() / denom


def build_model(model_name, channels, seq_len, pred_len, args):
    if args.tslib_root:
        root = str(Path(args.tslib_root).resolve())
        if root not in sys.path:
            sys.path.insert(0, root)
    cfg = SimpleNamespace(
        task_name="long_term_forecast",
        seq_len=seq_len,
        label_len=args.label_len,
        pred_len=pred_len,
        enc_in=channels,
        dec_in=channels,
        c_out=channels,
        d_model=args.d_model,
        n_heads=args.n_heads,
        e_layers=args.e_layers,
        d_layers=1,
        d_ff=args.d_ff,
        moving_avg=25,
        factor=3,
        distil=True,
        dropout=args.dropout,
        embed="timeF",
        freq="h",
        activation="gelu",
        top_k=5,
        num_kernels=6,
        patch_len=args.patch_len,
        stride=args.patch_stride,
        channel_independence=args.channel_independence,
        decomp_method="moving_avg",
        use_norm=1,
        down_sampling_layers=args.down_sampling_layers,
        down_sampling_window=args.down_sampling_window,
        down_sampling_method=args.down_sampling_method,
        seg_len=96,
    )
    module = __import__(f"models.{model_name}", fromlist=["Model"])
    return module.Model(cfg).float()


def decoder_inputs(x, x_mark, y, y_mark, label_len):
    if label_len <= 0:
        dec_inp = torch.zeros((y.shape[0], 0, y.shape[-1]), dtype=y.dtype, device=y.device)
        dec_mark = y_mark
        return dec_inp, dec_mark
    if label_len > x.shape[1]:
        raise ValueError(f"label_len={label_len} exceeds seq_len={x.shape[1]}")
    label_values = x[:, -label_len:, :]
    label_marks = x_mark[:, -label_len:, :]
    zeros = torch.zeros((y.shape[0], y.shape[1], y.shape[-1]), dtype=y.dtype, device=y.device)
    dec_inp = torch.cat([label_values, zeros], dim=1)
    dec_mark = torch.cat([label_marks, y_mark], dim=1)
    return dec_inp, dec_mark


def run_eval(model, loader, device, target_dim, label_len):
    model.eval()
    losses = []
    preds = []
    trues = []
    observed_masks = []
    criterion = nn.MSELoss()
    with torch.no_grad():
        for x, y, x_mark, y_mark, y_observed in loader:
            x = x.float().to(device)
            y = y.float().to(device)
            x_mark = x_mark.float().to(device)
            y_mark = y_mark.float().to(device)
            y_observed = y_observed.float().to(device)
            dec_inp, dec_mark = decoder_inputs(x, x_mark, y, y_mark, label_len)
            out = model(x, x_mark, dec_inp, dec_mark)
            out_dx = out[:, -y.shape[1] :, :target_dim]
            y_dx = y[:, :, :target_dim]
            loss = criterion(out_dx, y_dx)
            losses.append(float(loss.item()))
            preds.append(out_dx.detach().cpu().numpy())
            trues.append(y_dx.detach().cpu().numpy())
            observed_masks.append(y_observed.detach().cpu().numpy())
    pred = np.concatenate(preds, axis=0)
    true = np.concatenate(trues, axis=0)
    observed = np.concatenate(observed_masks, axis=0)
    all_out = metric(pred, true)
    obs_out = observed_metric(pred, true, observed)
    out = dict(all_out)
    out["loss"] = float(np.mean(losses))
    out["all"] = all_out
    out["observed"] = obs_out
    out["observed_per_channel"] = observed_metric_per_channel(pred, true, observed)
    return out


def collect_predictions(model, loader, device, target_dim, dataset, label_len):
    model.eval()
    preds = []
    trues = []
    observed_masks = []
    with torch.no_grad():
        for x, y, x_mark, y_mark, y_observed in loader:
            x = x.float().to(device)
            y = y.float().to(device)
            x_mark = x_mark.float().to(device)
            y_mark = y_mark.float().to(device)
            y_observed = y_observed.float().to(device)
            dec_inp, dec_mark = decoder_inputs(x, x_mark, y, y_mark, label_len)
            out = model(x, x_mark, dec_inp, dec_mark)
            preds.append(out[:, -y.shape[1] :, :target_dim].detach().cpu().numpy())
            trues.append(y[:, :, :target_dim].detach().cpu().numpy())
            observed_masks.append(y_observed.detach().cpu().numpy())
    pred = np.concatenate(preds, axis=0)
    true = np.concatenate(trues, axis=0)
    observed = np.concatenate(observed_masks, axis=0)
    mean = dataset.scaler.mean_[:target_dim].reshape(1, 1, target_dim)
    scale = dataset.scaler.scale_[:target_dim].reshape(1, 1, target_dim)
    sample_indices = np.asarray(dataset.sample_indices, dtype=np.int64)
    output_start_abs = dataset.border1 + sample_indices + dataset.seq_len
    return {
        "pred": pred * scale + mean,
        "true": true * scale + mean,
        "observed": observed,
        "sample_indices": sample_indices,
        "output_start_abs": output_start_abs.astype(np.int64),
        "feature_cols": np.asarray(dataset.feature_cols[:target_dim], dtype=object),
    }


def train_one(model_name, csv_path, run_name, args):
    df = pd.read_csv(csv_path)
    target_df = pd.read_csv(args.target_csv) if args.target_csv else None
    mask_df = pd.read_csv(args.mask_csv) if args.mask_csv else None
    df, target_df, mask_df, date_filter = filter_by_date_window(
        df,
        target_df=target_df,
        mask_df=mask_df,
        date_start=args.date_start,
        date_end=args.date_end,
    )
    channels = len(df.columns) - 1
    target_dim = args.target_dim
    train_ds = WindowDataset(
        df,
        args.seq_len,
        args.pred_len,
        "train",
        mask_df=mask_df,
        target_df=target_df,
        target_dim=target_dim,
        drop_unobserved_windows=args.drop_unobserved_windows,
    )
    val_ds = WindowDataset(
        df,
        args.seq_len,
        args.pred_len,
        "val",
        scaler=train_ds.scaler,
        mask_df=mask_df,
        target_df=target_df,
        target_dim=target_dim,
        drop_unobserved_windows=args.drop_unobserved_windows,
    )
    test_ds = WindowDataset(
        df,
        args.seq_len,
        args.pred_len,
        "test",
        scaler=train_ds.scaler,
        mask_df=mask_df,
        target_df=target_df,
        target_dim=target_dim,
        drop_unobserved_windows=args.drop_unobserved_windows,
    )
    if len(train_ds) == 0 or len(val_ds) == 0 or len(test_ds) == 0:
        raise ValueError(
            f"{run_name}: empty split after windowing/filtering "
            f"(train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)})"
        )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    model = build_model(model_name, channels, args.seq_len, args.pred_len, args).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    best = {"val": float("inf"), "test": None, "epoch": None}
    best_state = None
    bad = 0
    log_lines = []
    aexp_note(
        f"{run_name} start train={len(train_ds)} val={len(val_ds)} test={len(test_ds)} "
        f"channels={channels} target_dim={target_dim} date_filter={date_filter}"
    )
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        t0 = time.time()
        for x, y, x_mark, y_mark, y_observed in train_loader:
            x = x.float().to(device)
            y = y.float().to(device)
            x_mark = x_mark.float().to(device)
            y_mark = y_mark.float().to(device)
            y_observed = y_observed.float().to(device)
            dec_inp, dec_mark = decoder_inputs(x, x_mark, y, y_mark, args.label_len)
            optim.zero_grad(set_to_none=True)
            out = model(x, x_mark, dec_inp, dec_mark)
            out_dx = out[:, -args.pred_len :, :target_dim]
            y_dx = y[:, :, :target_dim]
            if args.train_loss_mode == "observed":
                loss = masked_mse_loss(out_dx, y_dx, y_observed)
            else:
                loss = criterion(out_dx, y_dx)
            loss.backward()
            optim.step()
            losses.append(float(loss.item()))
        val = run_eval(model, val_loader, device, target_dim, args.label_len)
        test = run_eval(model, test_loader, device, target_dim, args.label_len)
        train_loss = float(np.mean(losses))
        val_select = val[args.selection_metric]
        test_select = test[args.selection_metric]
        line = (
            f"{run_name} epoch={epoch} train={train_loss:.6f} "
            f"val_{args.selection_metric}_mse={val_select['mse']:.6f} "
            f"val_{args.selection_metric}_mae={val_select['mae']:.6f} "
            f"test_{args.selection_metric}_mse={test_select['mse']:.6f} "
            f"test_{args.selection_metric}_mae={test_select['mae']:.6f} "
            f"time={time.time()-t0:.1f}s"
        )
        print(line, flush=True)
        log_lines.append(line)
        step = args.step_offset + epoch
        event_fields = {
            "series": "itransformer_downstream",
            "variant": run_name,
            "model": model_name,
            "seq_len": args.seq_len,
            "pred_len": args.pred_len,
            "seed": args.seed,
        }
        if args.event_trial >= 0:
            event_fields["trial"] = args.event_trial
            event_fields["trial_name"] = run_name
        aexp_progress("epoch", epoch, total=args.epochs, **event_fields)
        aexp_metric("train/loss", train_loss, step=step, epoch=epoch, **event_fields)
        aexp_metric("val/observed_mse", val["observed"]["mse"], step=step, epoch=epoch, split="val", **event_fields)
        aexp_metric("val/observed_mae", val["observed"]["mae"], step=step, epoch=epoch, split="val", **event_fields)
        aexp_metric("test/observed_mse", test["observed"]["mse"], step=step, epoch=epoch, split="test", **event_fields)
        aexp_metric("test/observed_mae", test["observed"]["mae"], step=step, epoch=epoch, split="test", **event_fields)
        if val_select["mse"] + args.min_delta < best["val"]:
            best = {"val": val_select["mse"], "val_metrics": val, "test": test, "epoch": epoch}
            best_state = copy.deepcopy(model.state_dict())
            bad = 0
        else:
            bad += 1
            if bad >= args.patience:
                break

    checkpoint_path = None
    prediction_npzs = {}
    if best_state is not None and (args.checkpoint_dir or args.prediction_npz_dir):
        model.load_state_dict(best_state)

    if args.checkpoint_dir and best_state is not None:
        checkpoint_dir = Path(args.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / f"{run_name}_best.pt"
        torch.save(
            {
                "model_state_dict": best_state,
                "model_name": model_name,
                "run_name": run_name,
                "channels": channels,
                "target_dim": target_dim,
                "seq_len": args.seq_len,
                "pred_len": args.pred_len,
                "best_epoch": best["epoch"],
                "best_val": best["val_metrics"],
                "best_test": best["test"],
                "feature_cols": train_ds.feature_cols,
                "scaler_mean": train_ds.scaler.mean_.astype(float).tolist(),
                "scaler_scale": train_ds.scaler.scale_.astype(float).tolist(),
                "config": {
                    "d_model": args.d_model,
                    "d_ff": args.d_ff,
                    "n_heads": args.n_heads,
                    "e_layers": args.e_layers,
                    "dropout": args.dropout,
                    "channel_independence": args.channel_independence,
                    "down_sampling_method": args.down_sampling_method,
                    "down_sampling_layers": args.down_sampling_layers,
                    "down_sampling_window": args.down_sampling_window,
                },
            },
            checkpoint_path,
        )
        aexp_note(f"{run_name} saved best checkpoint to {checkpoint_path}")

    if args.prediction_npz_dir and best_state is not None:
        split_loaders = {"val": (val_loader, val_ds), "test": (test_loader, test_ds)}
        prediction_dir = Path(args.prediction_npz_dir)
        prediction_dir.mkdir(parents=True, exist_ok=True)
        for split in args.prediction_splits:
            if split not in split_loaders:
                raise ValueError(f"unknown prediction split={split!r}; expected one of {sorted(split_loaders)}")
            loader, dataset = split_loaders[split]
            arrays = collect_predictions(model, loader, device, target_dim, dataset, args.label_len)
            arrays["split"] = np.asarray([split], dtype=object)
            prediction_npz = prediction_dir / f"{run_name}_{split}_windows.npz"
            np.savez_compressed(prediction_npz, **arrays)
            prediction_npzs[split] = str(prediction_npz)
            aexp_note(f"{run_name} saved {split}-window predictions to {prediction_npz}")

    return {
        "run_name": run_name,
        "model": model_name,
        "csv": str(csv_path),
        "target_csv": args.target_csv,
        "channels": channels,
        "target_dim": target_dim,
        "seq_len": args.seq_len,
        "pred_len": args.pred_len,
        "mask_csv": args.mask_csv,
        "date_filter": date_filter,
        "train_loss_mode": args.train_loss_mode,
        "selection_metric": args.selection_metric,
        "drop_unobserved_windows": args.drop_unobserved_windows,
        "seed": args.seed,
        "split_windows": {
            "train": len(train_ds),
            "val": len(val_ds),
            "test": len(test_ds),
        },
        "best_epoch": best["epoch"],
        "best_val": best["val_metrics"],
        "best_test": best["test"],
        "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
        "prediction_npz": prediction_npzs.get("test"),
        "prediction_npzs": prediction_npzs,
        "logs": log_lines,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="iTransformer")
    parser.add_argument("--mask_csv", default=None)
    parser.add_argument(
        "--target_csv",
        default="",
        help=(
            "Optional separate target CSV. Inputs are read from each variant CSV; "
            "the first target_dim future channels are supervised from this target CSV "
            "and standardized with the input scaler."
        ),
    )
    parser.add_argument("--variants", nargs="+", default=["dx82", "dx_temp171", "dx_H92", "dx_seeP93", "dx_env192"])
    parser.add_argument("--seq_len", type=int, default=92)
    parser.add_argument("--label_len", type=int, default=0)
    parser.add_argument("--pred_len", type=int, default=48)
    parser.add_argument("--target_dim", type=int, default=82)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--min_delta", type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--d_ff", type=int, default=256)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--e_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--patch_len", type=int, default=16)
    parser.add_argument("--patch_stride", type=int, default=8)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--tslib_root", default=None)
    parser.add_argument("--step_offset", type=int, default=0)
    parser.add_argument("--event_trial", type=int, default=-1)
    parser.add_argument("--train_loss_mode", choices=["all", "observed"], default="all")
    parser.add_argument("--selection_metric", choices=["all", "observed"], default="all")
    parser.add_argument("--drop_unobserved_windows", action="store_true")
    parser.add_argument("--date_start", default="")
    parser.add_argument(
        "--date_end",
        default="",
        help="Exclusive date upper bound. YYYY-MM is interpreted as the first day of the next month.",
    )
    parser.add_argument("--checkpoint_dir", default=None)
    parser.add_argument("--prediction_npz_dir", default=None)
    parser.add_argument("--prediction_splits", nargs="+", choices=["val", "test"], default=["test"])
    parser.add_argument("--channel_independence", type=int, default=1)
    parser.add_argument("--down_sampling_method", default=None)
    parser.add_argument("--down_sampling_layers", type=int, default=0)
    parser.add_argument("--down_sampling_window", type=int, default=1)
    args = parser.parse_args()
    set_seed(args.seed)
    for name in (
        "model",
        "seq_len",
        "pred_len",
        "target_dim",
        "epochs",
        "batch_size",
        "train_loss_mode",
        "selection_metric",
        "drop_unobserved_windows",
        "seed",
    ):
        aexp_param(name, getattr(args, name))
    aexp_note("tslib selective downstream matrix start")

    name_to_file = {
        "dx82": "dam_dx82.csv",
        "dxdy169": "dam_dxdy169.csv",
        "dx_temp171": "dam_dx_temp171.csv",
        "dx_H92": "dam_dx_H92.csv",
        "dx_seeP93": "dam_dx_seeP93.csv",
        "dx_env192": "dam_dx_env192.csv",
    }
    results = []
    data_root = Path(args.data_root)
    for variant in args.variants:
        if variant in name_to_file:
            csv_path = data_root / name_to_file[variant]
            variant_name = variant
        else:
            csv_path = Path(variant)
            if not csv_path.is_absolute():
                csv_path = data_root / csv_path
            variant_name = csv_path.stem
        run_name = f"{args.model}_{variant_name}_sl{args.seq_len}_pl{args.pred_len}"
        results.append(train_one(args.model, csv_path, run_name, args))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"wrote {output}", flush=True)


if __name__ == "__main__":
    main()
