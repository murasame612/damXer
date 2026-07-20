#!/usr/bin/env bash
set -euo pipefail

# Compatibility wrapper around the immutable Python paper-baseline runner.
# Select PROFILE=dx_only for the five generic baselines or PROFILE=raw_env for
# the 720-step iTransformer control.

PY="${PY:-python}"
PROFILE="${PROFILE:-dx_only}"
TSLIB_ROOT="${TSLIB_ROOT:-}"
DATA_ROOT="${DATA_ROOT:-$PWD/data/paper-inputs}"
TARGET_CSV="${TARGET_CSV:-$PWD/data/paper-targets/filtered_response.csv}"
MASK_CSV="${MASK_CSV:-$PWD/data/paper-targets/dam_2h_filtered_response_observed_mask.csv}"
OUT_ROOT="${OUT_ROOT:-$PWD/artifacts/paper/baselines/$PROFILE}"
GPU="${GPU:-0}"
DEVICE="${DEVICE:-auto}"
SEEDS="${SEEDS:-}"
MODELS="${MODELS:-}"

if [ -z "$TSLIB_ROOT" ]; then
  echo "TSLIB_ROOT must point to the frozen Time-Series-Library checkout" >&2
  exit 2
fi

cmd=(
  "$PY" scripts/run_paper_baselines.py
  --profile "$PROFILE"
  --data-root "$DATA_ROOT"
  --target-csv "$TARGET_CSV"
  --mask-csv "$MASK_CSV"
  --output-dir "$OUT_ROOT"
  --tslib-root "$TSLIB_ROOT"
  --gpu "$GPU"
  --device "$DEVICE"
)

if [ -n "$SEEDS" ]; then
  cmd+=(--seeds "$SEEDS")
fi
if [ -n "$MODELS" ]; then
  cmd+=(--models "$MODELS")
fi
if [ "${CHECK_ONLY:-0}" = "1" ]; then
  cmd+=(--check-only)
fi
if [ "${KEEP_PREDICTIONS:-0}" = "1" ]; then
  cmd+=(--keep-predictions)
fi

printf '+ %q ' "${cmd[@]}"
printf '\n'
exec "${cmd[@]}"
