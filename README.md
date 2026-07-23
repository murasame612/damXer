# DamXer

DamXer is a response-guided environmental lag-attention model for direct,
multi-step, multi-point earth-rockfill dam displacement forecasting under
incomplete monitoring.

This repository contains the paper-facing model implementation, frozen
configurations, anonymized paper data, and executable experiment pipeline.

## Release status

The data owner has authorized publication of the anonymized monitoring dataset
and processed forecasting benchmark. The canonical files are under
[`data/paper/`](data/paper/) and are protected by byte-size, shape, and SHA-256
checks in [`release/benchmark_manifest.json`](release/benchmark_manifest.json).

## Quick start

From a fresh clone, this command validates all six released CSV files and
recomputes every paper-table mean and sample standard deviation from the
published 45-row seed ledger:

```bash
./reproduce.sh
```

It uses only the Python standard library and writes
`artifacts/reproduction/check_report.json`. This fast check does not retrain a
model. For a genuine CUDA rerun of the three DamXer variants (15 training jobs),
use:

```bash
./reproduce.sh damxer --device cuda
```

For all nine paper settings (45 jobs), including automatic checkout of the
frozen Time-Series-Library adapter dependency, use:

```bash
./reproduce.sh paper --device cuda
```

The first training command creates a project-local `.venv` and installs
`requirements-forecasting.txt`; it never installs into the system Python. The
full `requirements.txt` additionally includes PyPOTS for rerunning SAITS. A guided version
of the same workflow is available in
[`notebooks/DamXer_reproduction.ipynb`](notebooks/DamXer_reproduction.ipynb).

## CUDA reproduction check

The public-repository entry point was executed on an NVIDIA RTX PRO 6000
Blackwell workstation on 2026-07-23 using PyTorch 2.11.0+cu128. All 15 DamXer
jobs completed successfully, and the three five-seed test-MSE aggregates
matched the frozen numerical ledger at stored precision:

| Variant | Active ENV tokens | Test MSE (mean +/- sample SD) |
|---|---:|---:|
| DamXer | 60 | 0.179124 +/- 0.006253 |
| Reduced-lag ENV | 15 | 0.192780 +/- 0.006759 |
| Displacement only | 0 | 0.203902 +/- 0.010496 |

The machine-readable execution receipt is
[`release/reproduction_pro6000_20260723.json`](release/reproduction_pro6000_20260723.json).
It records a release-candidate run; immutable release provenance begins with
the eventual repository commit and tag.

## What is included

```text
configs/paper_clean_window.json       frozen five-seed paper configuration
configs/paper_completion.json         frozen SAITS/completion configuration
scripts/run_saits_completion.py       incomplete table -> SAITS clean table
scripts/run_completion_benchmark.py   fixed-block SAITS/KNN/ImputeFormer comparison
scripts/verify_completion_results.py  completion protocol and metric checks
scripts/build_engineered_inputs.py    hydraulic/seepage/thermal lag features
scripts/build_filtered_response.py    Median(5) + Savitzky--Golay(9, 3) target
scripts/train_damxer.py                model, observed-only loss, training, evaluation
scripts/run_paper_multiseed.py         paper configuration and ablation runner
scripts/run_paper_baselines.py         frozen generic/raw-ENV baseline runner
scripts/run_tslib_baseline.py          optional Time-Series-Library adapter
scripts/run_tslib_baselines.sh         optional baseline batch wrapper
scripts/verify_paper_results.py        seed aggregation and paper-number checks
scripts/reproduce.py                   unified data-check and retraining entry point
scripts/prepare_data_release.py        canonical GitHub data assembly and SHA checks
reproduce.sh                           one-command wrapper and local environment setup
notebooks/DamXer_reproduction.ipynb    guided check and optional retraining workflow
data/paper/                            canonical anonymized paper dataset
data/README.md                         required input schema and mask semantics
examples/                              synthetic data and end-to-end smoke test
tests/                                 unit tests for masking and lag-token construction
results/paper_reported_metrics.csv     read-only values reported in the manuscript
results/paper_seed_metrics.csv         sanitized 45-seed-row numerical ledger
```

The core DamXer implementation does not vendor or directly depend on TimeXer.
The optional generic forecasting baselines require a separate checkout of
[Time-Series-Library](https://github.com/thuml/Time-Series-Library).
The frozen config pins the reconstructed checkout used by this release. The
source experiments did not record their TSLib commit, so this pin is not
presented as recovered provenance; the result verifier remains the numerical
acceptance check.

## Environment

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

GPU execution uses the CUDA support provided by the installed PyTorch build.
With `--device auto`, the scripts select CUDA first, then Apple MPS, and then
CPU. Use `--device cuda` for a paper-grade rerun on the original accelerator
class; CPU/MPS runs are useful execution checks but are not numerically
interchangeable with the reported CUDA runs.

## Synthetic smoke test

The synthetic example contains generic channel names and no project data. It
trains a one-epoch CPU SAITS smoke model, then checks input construction,
filtered-response target construction, the original mask convention, the
60-token environment bank, and one DamXer training epoch. It is an execution
test, not a paper-grade result.

```bash
python examples/run_synthetic_smoke.py
python -m unittest discover -s tests -v
```

Outputs are written under `artifacts/`, which is ignored by Git.

## Paper data contract

The frozen experiment expects five timestamp-aligned CSV files:

```text
<data-root>/dam_2h_saits_dx_only_nomask.csv
<data-root>/dam_2h_saits_dx_engineered_env_nomask.csv
<data-root>/dam_2h_saits_dx_raw_env_nomask.csv
<target-root>/filtered_response.csv
<target-root>/dam_2h_filtered_response_observed_mask.csv
```

The raw-ENV control contains the same 89 displacement channels followed by 111
completed original hydraulic, seepage, and temperature channels. The first
column is `date`. The target mask uses `1` for a missing cell that must be
excluded from supervision and metrics, and `0` for an originally observed
cell. See [data/README.md](data/README.md) for the complete schema.

Validate a local copy without starting training:

```bash
python scripts/run_paper_multiseed.py \
  --data-root /path/to/engineered-inputs \
  --target-csv /path/to/filtered-response/filtered_response.csv \
  --mask-csv /path/to/filtered-response/dam_2h_filtered_response_observed_mask.csv \
  --output-dir artifacts/paper/full \
  --strict-paper-shape \
  --check-only
```

For the paper slice, the check must report 8400 rows, 89 response channels, 60
environmental lag tokens, and 5065/745/1585 train/validation/test prediction
origins.

## Reproduce the frozen five-seed run

The one-command wrapper uses the included canonical paths. The equivalent
manual variables are:

```bash
DATA_ROOT=data/paper/inputs
TARGET_CSV=data/paper/targets/filtered_response.csv
MASK_CSV=data/paper/targets/dam_2h_filtered_response_observed_mask.csv
```

```bash
python scripts/run_paper_multiseed.py \
  --data-root "$DATA_ROOT" \
  --target-csv "$TARGET_CSV" \
  --mask-csv "$MASK_CSV" \
  --output-dir artifacts/paper/full \
  --strict-paper-shape \
  --variant full \
  --device cuda
```

Controlled ablations use the same entry point:

```bash
# Retain ENV inputs but reduce the explicit multi-lag token organization.
python scripts/run_paper_multiseed.py \
  --data-root "$DATA_ROOT" \
  --target-csv "$TARGET_CSV" \
  --mask-csv "$MASK_CSV" \
  --output-dir artifacts/paper/no_lag_env \
  --strict-paper-shape \
  --variant no_lag_env \
  --device cuda

# Remove the environmental branch.
python scripts/run_paper_multiseed.py \
  --data-root "$DATA_ROOT" \
  --target-csv "$TARGET_CSV" \
  --mask-csv "$MASK_CSV" \
  --output-dir artifacts/paper/no_env \
  --strict-paper-shape \
  --variant no_env \
  --device cuda
```

Run the five generic displacement-only baselines and the raw-ENV control from
the frozen Time-Series-Library checkout:

```bash
git clone https://github.com/thuml/Time-Series-Library.git /path/to/Time-Series-Library
git -C /path/to/Time-Series-Library checkout 4e938a1767106324dd753b2a44832bf870a0252e
TSLIB_ROOT=/path/to/Time-Series-Library

python scripts/run_paper_baselines.py \
  --profile dx_only \
  --data-root "$DATA_ROOT" \
  --target-csv "$TARGET_CSV" \
  --mask-csv "$MASK_CSV" \
  --output-dir artifacts/paper/baselines-dx \
  --tslib-root "$TSLIB_ROOT" \
  --device cuda

python scripts/run_paper_baselines.py \
  --profile raw_env \
  --data-root "$DATA_ROOT" \
  --target-csv "$TARGET_CSV" \
  --mask-csv "$MASK_CSV" \
  --output-dir artifacts/paper/baseline-raw-env \
  --tslib-root "$TSLIB_ROOT" \
  --device cuda
```

The frozen DamXer Trial 26 configuration uses seeds 2021--2025,
displacement/environmental histories of 192/720 steps, a 96-step horizon,
16/16 patch length and stride, hidden size 160, 8 heads, 2 layers, dropout 0.25,
observed-only Huber loss plus a 0.01 increment-loss weight, RevIN, and direct
prediction. Model-state selection uses validation observed MSE and the test
split is evaluated once after loading the selected checkpoint. PatchTST Trial
14 follows the same final-only rule. The older configured baselines retain
their source-run per-epoch test-loader schedule only for numerical compatibility;
test metrics never enter selection in either schedule.

## Reproduce completion and prepare forecasting inputs

Start from the anonymized incomplete monitoring table. The formal command uses
the frozen 200-epoch PyPOTS SAITS configuration and writes a continuous table:

```bash
python scripts/run_saits_completion.py \
  --raw-csv /path/to/anonymized_monitoring_2h_incomplete.csv \
  --config configs/paper_completion.json \
  --output-dir artifacts/completion/canonical \
  --device cuda \
  --strict-paper-shape
```

The historical source run appended seven deterministic sample-index features
to SAITS. Their periods (24 and 8766 samples) are preserved exactly for source
compatibility; at 2 h sampling they are not claimed to be literal daily or
annual cycles. ImputeFormer and Group KNN do not receive those features.

Reproduce the paper-facing continuous-block comparison on the same monitoring
table, dx channels, pseudo-mask positions, and metrics:

```bash
python scripts/run_completion_benchmark.py \
  --raw-csv /path/to/anonymized_monitoring_2h_incomplete.csv \
  --config configs/paper_completion.json \
  --output-dir artifacts/completion/benchmark \
  --models saits,group_knn,imputeformer \
  --seeds 42 \
  --device cuda \
  --strict-paper-shape

python scripts/verify_completion_results.py \
  --summary artifacts/completion/benchmark/summary.json
```

Seed 42 reproduces the paper table. For the five training runs used to draw
the figure variability bands, pass `--seeds 2021,2022,2023,2024,2025`.

Then construct the frozen forecasting inputs and target:

```bash
SAITS_CLEAN=artifacts/completion/canonical/saits_clean.csv

python scripts/build_engineered_inputs.py \
  --raw-csv /path/to/raw_monitoring.csv \
  --saits-clean-csv "$SAITS_CLEAN" \
  --output-dir artifacts/engineered \
  --date-start '2024-01-01 00:00:00' \
  --date-end '2025-12-01 00:00:00'

python scripts/build_filtered_response.py \
  --raw-csv /path/to/raw_monitoring.csv \
  --saits-clean-csv "$SAITS_CLEAN" \
  --output-dir artifacts/filtered-response \
  --date-start '2024-01-01 00:00:00' \
  --date-end '2025-12-01 00:00:00'
```

The target builder filters the aligned continuous series before applying the
date slice. This preserves the two-sided context used by the retrospective
paper protocol at the slice boundary.

## Verify reproduced metrics

After the three DamXer variants and two baseline profiles finish, verify all
nine table rows and the derived percentage claims:

```bash
python scripts/verify_paper_results.py \
  --damxer-full artifacts/paper/full/summary.json \
  --damxer-reduced artifacts/paper/no_lag_env/summary.json \
  --damxer-no-env artifacts/paper/no_env/summary.json \
  --dx-baselines artifacts/paper/baselines-dx/aggregate.json \
  --raw-env artifacts/paper/baseline-raw-env/aggregate.json
```

The verifier accepts either a DamXer aggregate summary or a directory
containing five `seed_*.json` files. It requires the frozen seeds 2021--2025
and uses sample standard deviation, matching the manuscript table.

SAITS is an adopted completion component rather than the paper's proposed
forecasting model. The GitHub data release includes the anonymized incomplete
monitoring table needed to rerun completion and the five processed forecasting
tables needed to rerun DamXer.

## Scope and claims

- The synthetic smoke test demonstrates software executability, not paper-grade
  accuracy.
- The fast check reaggregates published seed-level values; only `damxer` or
  `paper` mode performs fresh training on the released data.
- The exact source package build was not preserved. A fresh five-seed GPU run
  is therefore checked with the declared aggregate MSE tolerance (default
  `0.005`), not by requiring bitwise identity across CUDA hardware and releases.
- Attention weights indicate model relevance and are not causal estimates of
  physical lag.

## License and citation

The source code, configurations, documentation, and synthetic examples are
released under the [MIT License](LICENSE).

The authorized anonymized monitoring data and processed benchmark files under
`data/paper/` are released under the
[Creative Commons Attribution 4.0 International License](data/paper/LICENSE).
CC BY 4.0 permits sharing and adaptation, including commercial use, provided
that appropriate attribution is given, the license is linked, and changes are
indicated. Citation metadata are provided in
[CITATION.cff](CITATION.cff).
