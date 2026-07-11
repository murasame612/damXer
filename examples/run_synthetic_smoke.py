#!/usr/bin/env python3
"""Exercise feature construction, target construction, and one DamXer epoch."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(command: list[str]) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="artifacts/synthetic_smoke")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    output = Path(args.output_dir).resolve()
    source = output / "source"
    engineered = output / "engineered"
    target = output / "target"
    result = output / "smoke_result.json"
    python = sys.executable

    run([python, str(root / "examples/generate_synthetic_data.py"), "--output-dir", str(source)])
    run(
        [
            python,
            str(root / "scripts/build_engineered_inputs.py"),
            "--raw-csv", str(source / "raw_monitoring.csv"),
            "--saits-clean-csv", str(source / "saits_clean.csv"),
            "--output-dir", str(engineered),
        ]
    )
    run(
        [
            python,
            str(root / "scripts/build_filtered_response.py"),
            "--raw-csv", str(source / "raw_monitoring.csv"),
            "--saits-clean-csv", str(source / "saits_clean.csv"),
            "--engineered-env-csv", str(engineered / "dam_2h_saits_dx_engineered_env_nomask.csv"),
            "--output-dir", str(target),
        ]
    )
    run(
        [
            python,
            str(root / "scripts/train_damxer.py"),
            "--data-root", str(engineered),
            "--mask-csv", str(target / "dam_2h_filtered_response_observed_mask.csv"),
            "--target-csv", str(target / "filtered_response.csv"),
            "--output", str(result),
            "--dx-seq-len", "32",
            "--env-seq-len", "48",
            "--pred-len", "8",
            "--target-dim", "3",
            "--patch-len", "8",
            "--patch-stride", "8",
            "--hidden", "16",
            "--n-heads", "4",
            "--e-layers", "1",
            "--epochs", "1",
            "--patience", "1",
            "--batch-size", "8",
            "--lr", "0.001",
            "--loss-type", "huber",
            "--huber-delta", "0.2",
            "--predict-mode", "direct",
            "--zero-head-init",
            "--revin",
            "--trial-name", "synthetic_smoke",
        ]
    )
    print(f"synthetic smoke test completed: {result}")


if __name__ == "__main__":
    main()
