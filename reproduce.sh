#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

MODE="${1:-check}"
if [[ "$MODE" != --* ]]; then
  shift || true
else
  MODE="check"
fi

if [[ "$MODE" == "check" ]]; then
  exec python3 scripts/reproduce.py --mode check "$@"
fi

if [[ ! -x .venv/bin/python ]]; then
  PYTHON="$(command -v python3.12 || command -v python3.11 || command -v python3.10 || command -v python3)"
  "$PYTHON" -m venv .venv
  .venv/bin/python -m pip install --upgrade pip
  .venv/bin/python -m pip install -r requirements-forecasting.txt
fi

exec .venv/bin/python scripts/reproduce.py --mode "$MODE" "$@"
