#!/usr/bin/env bash
set -euo pipefail

# Paper-facing filtered-response TSLib baselines.
# This wrapper keeps the aexp recipe short and makes the protocol explicit:
# same SAITS-clean dx input, same filtered-response target, same observed-only mask,
# same 192 -> 96 window, same seed list, and val/test prediction dumps.

PY="${PY:-python}"
TSLIB_ROOT="${TSLIB_ROOT:-}"
MODELS="${MODELS:-iTransformer PatchTST}"
SEEDS="${SEEDS:-2021,2022,2023,2024,2025}"
EPOCHS="${EPOCHS:-30}"
PATIENCE="${PATIENCE:-6}"
GPU="${GPU:-0}"
SEQ_LEN="${SEQ_LEN:-192}"
PRED_LEN="${PRED_LEN:-96}"
TARGET_DIM="${TARGET_DIM:-89}"
DATE_START="${DATE_START:-}"
DATE_END="${DATE_END:-}"

INPUT_DIR="${INPUT_DIR:-$PWD/dataset/downstream_itransformer_2h_dx_engineered_env_nomask_v1}"
TARGET_DIR="${TARGET_DIR:-$PWD/dataset/filtered_response_2h_dx_median_sg_v1}"
VARIANT="${VARIANT:-dam_2h_saits_dx_only_nomask.csv}"
OUT_ROOT="${OUT_ROOT:-$PWD/outputs/paper_filtered_response_tslib_baselines_seed5_v1}"

if [ -z "$TSLIB_ROOT" ]; then
  echo "TSLIB_ROOT must point to a local Time-Series-Library checkout" >&2
  exit 2
fi

MASK_CSV="${TARGET_DIR}/dam_2h_filtered_response_observed_mask.csv"
TARGET_CSV="${TARGET_DIR}/filtered_response.csv"

mkdir -p "$OUT_ROOT"

cat > "$OUT_ROOT/protocol_manifest.json" <<JSON
{
  "protocol": "paper_filtered_response_tslib_baselines",
  "input_dir": "$INPUT_DIR",
  "variant": "$VARIANT",
  "target_csv": "$TARGET_CSV",
  "mask_csv": "$MASK_CSV",
  "models": "$MODELS",
  "seeds": "$SEEDS",
  "seq_len": $SEQ_LEN,
  "pred_len": $PRED_LEN,
  "target_dim": $TARGET_DIM,
  "epochs": $EPOCHS,
  "patience": $PATIENCE,
  "date_start": "$DATE_START",
  "date_end": "$DATE_END",
  "selection_metric": "observed",
  "train_loss_mode": "observed",
  "prediction_splits": ["val", "test"],
  "checkpoint_policy": "no checkpoints unless KEEP_CHECKPOINT=1"
}
JSON

IFS="," read -ra SEED_ARR <<< "$SEEDS"

for MODEL in $MODELS; do
  for SEED in "${SEED_ARR[@]}"; do
    OUT="$OUT_ROOT/$MODEL/seed_$SEED"
    mkdir -p "$OUT"

    LABEL_LEN=0
    if [ "$MODEL" = "FEDformer" ]; then
      LABEL_LEN=96
    fi

    # iTransformer uses the validation-selected trial025 config from the dx-only
    # baseline search. Other TSLib baselines use the documented standard config.
    BATCH_SIZE="${BASE_BATCH_SIZE:-16}"
    LR="${BASE_LR:-0.0005}"
    D_MODEL="${BASE_D_MODEL:-64}"
    D_FF="${BASE_D_FF:-128}"
    N_HEADS="${BASE_N_HEADS:-2}"
    E_LAYERS="${BASE_E_LAYERS:-1}"
    DROPOUT="${BASE_DROPOUT:-0.1}"
    if [ "$MODEL" = "iTransformer" ]; then
      BATCH_SIZE="${ITR_BATCH_SIZE:-8}"
      LR="${ITR_LR:-0.0017089863074183553}"
      D_MODEL="${ITR_D_MODEL:-64}"
      D_FF="${ITR_D_FF:-128}"
      N_HEADS="${ITR_N_HEADS:-2}"
      E_LAYERS="${ITR_E_LAYERS:-1}"
      DROPOUT="${ITR_DROPOUT:-0.1}"
    fi

    cmd=(
      "$PY" scripts/run_tslib_baseline.py
      --data_root "$INPUT_DIR"
      --output "$OUT/summary.json"
      --model "$MODEL"
      --tslib_root "$TSLIB_ROOT"
      --mask_csv "$MASK_CSV"
      --target_csv "$TARGET_CSV"
      --variants "$VARIANT"
      --seq_len "$SEQ_LEN"
      --label_len "$LABEL_LEN"
      --pred_len "$PRED_LEN"
      --target_dim "$TARGET_DIM"
      --epochs "$EPOCHS"
      --patience "$PATIENCE"
      --batch_size "$BATCH_SIZE"
      --lr "$LR"
      --d_model "$D_MODEL"
      --d_ff "$D_FF"
      --n_heads "$N_HEADS"
      --e_layers "$E_LAYERS"
      --dropout "$DROPOUT"
      --gpu "$GPU"
      --seed "$SEED"
      --train_loss_mode observed
      --selection_metric observed
      --drop_unobserved_windows
      --prediction_npz_dir "$OUT/predictions"
      --prediction_splits val test
    )

    if [ -n "$DATE_START" ]; then
      cmd+=(--date_start "$DATE_START")
    fi
    if [ -n "$DATE_END" ]; then
      cmd+=(--date_end "$DATE_END")
    fi

    if [ "${KEEP_CHECKPOINT:-0}" = "1" ]; then
      cmd+=(--checkpoint_dir "$OUT/checkpoints")
    fi

    pred_stem="${MODEL}_dam_2h_saits_dx_only_nomask_sl${SEQ_LEN}_pl${PRED_LEN}"
    if [ "${FORCE_RERUN:-0}" != "1" ] \
      && [ -f "$OUT/summary.json" ] \
      && [ -f "$OUT/predictions/${pred_stem}_val_windows.npz" ] \
      && [ -f "$OUT/predictions/${pred_stem}_test_windows.npz" ]; then
      echo "skip existing complete run: $MODEL seed=$SEED"
      continue
    fi

    printf '+ %q ' "${cmd[@]}"
    printf '\n'
    "${cmd[@]}"
  done
done

"$PY" - "$OUT_ROOT" <<'PY'
import csv
import json
import statistics
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = []
for summary_path in sorted(root.glob("*/seed_*/summary.json")):
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    row = {
        "model": summary_path.parents[1].name,
        "seed": summary_path.parent.name.replace("seed_", ""),
        "summary_json": str(summary_path),
    }
    if isinstance(payload, list):
        result = payload[0] if payload else {}
    else:
        result = payload.get("results", [{}])[0]
    nested = {
        "val_observed_mse": ("best_val", "observed", "mse"),
        "test_observed_mse": ("best_test", "observed", "mse"),
        "test_observed_mae": ("best_test", "observed", "mae"),
        "test_observed_rmse": ("best_test", "observed", "rmse"),
    }
    for key, path in nested.items():
        value = result.get(key)
        if value is None:
            cursor = result
            for part in path:
                cursor = cursor.get(part, {}) if isinstance(cursor, dict) else {}
            value = cursor if isinstance(cursor, (int, float)) else None
        row[key] = value
    rows.append(row)

if rows:
    csv_path = root / "summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    by_model = {}
    for row in rows:
        by_model.setdefault(row["model"], []).append(row)
    aggregate = {}
    for model, model_rows in sorted(by_model.items()):
        aggregate[model] = {}
        for key in ("val_observed_mse", "test_observed_mse", "test_observed_mae", "test_observed_rmse"):
            vals = [float(r[key]) for r in model_rows if r.get(key) not in (None, "")]
            if vals:
                aggregate[model][key] = {
                    "mean": statistics.mean(vals),
                    "std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
                    "n": len(vals),
                }
    (root / "aggregate.json").write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
PY
