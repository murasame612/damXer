# DamXer

DamXer is a response-guided environmental lag-attention model for direct,
multi-step, multi-point earth-rockfill dam displacement forecasting under
incomplete monitoring.

This repository contains the paper-facing model implementation and executable
experiment pipeline. It is intentionally separate from the internal research
workspace so that datasets, operational identifiers, model artifacts, server
paths, and exploratory experiments are not published accidentally.

## Release status

This is currently a **private code-only staging repository**. The real
dam-monitoring data are not committed here. Without the paper dataset, third
parties can inspect the method and execute the synthetic smoke test, but cannot
reproduce the paper's numerical results. The final data-access statement will
be documented in the associated paper before this repository is made public.

## What is included

```text
configs/paper_clean_window.json       frozen five-seed paper configuration
scripts/build_engineered_inputs.py    hydraulic/seepage/thermal lag features
scripts/build_filtered_response.py    Median(5) + Savitzky--Golay(9, 3) target
scripts/train_damxer.py                model, observed-only loss, training, evaluation
scripts/run_paper_multiseed.py         paper configuration and ablation runner
scripts/run_tslib_baseline.py          optional Time-Series-Library adapter
scripts/run_tslib_baselines.sh         optional baseline batch wrapper
data/README.md                         required input schema and mask semantics
examples/                              synthetic data and end-to-end smoke test
tests/                                 unit tests for masking and lag-token construction
results/paper_reported_metrics.csv     read-only values reported in the manuscript
```

The core DamXer implementation does not vendor or directly depend on TimeXer.
The optional generic forecasting baselines require a separate checkout of
[Time-Series-Library](https://github.com/thuml/Time-Series-Library).

## Environment

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

GPU execution uses the CUDA support provided by the installed PyTorch build.
The scripts fall back to CPU when CUDA is unavailable.

## Synthetic smoke test

The synthetic example contains generic channel names and no project data. It
checks input construction, filtered-response target construction, the original
mask convention, the 60-token environment bank, and one DamXer training epoch.

```bash
python examples/run_synthetic_smoke.py
python -m unittest discover -s tests -v
```

Outputs are written under `artifacts/`, which is ignored by Git.

## Paper data contract

The frozen experiment expects four timestamp-aligned CSV files:

```text
<data-root>/dam_2h_saits_dx_only_nomask.csv
<data-root>/dam_2h_saits_dx_engineered_env_nomask.csv
<target-root>/filtered_response.csv
<target-root>/dam_2h_filtered_response_observed_mask.csv
```

The first column is `date`. The target mask uses `1` for a missing or repaired
cell that must be excluded from supervision and metrics, and `0` for an
originally observed, trusted cell. See [data/README.md](data/README.md) for the
complete schema.

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

```bash
python scripts/run_paper_multiseed.py \
  --data-root /path/to/engineered-inputs \
  --target-csv /path/to/filtered-response/filtered_response.csv \
  --mask-csv /path/to/filtered-response/dam_2h_filtered_response_observed_mask.csv \
  --output-dir artifacts/paper/full \
  --strict-paper-shape \
  --variant full
```

Controlled ablations use the same entry point:

```bash
# Retain ENV inputs but reduce the explicit multi-lag token organization.
python scripts/run_paper_multiseed.py ... --variant no_lag_env

# Remove the environmental branch.
python scripts/run_paper_multiseed.py ... --variant no_env
```

The frozen configuration uses seeds 2021--2025, displacement/environmental
histories of 192/720 steps, a 96-step horizon, 16/16 patch length and stride,
hidden size 160, 8 heads, 2 layers, observed-only Huber loss, RevIN, and direct
prediction. Model-state selection uses validation observed MSE. The test split
is evaluated only after the validation-best state has been restored.

## Prepare compatible inputs

Given an incomplete raw table and an already completed continuous table:

```bash
python scripts/build_engineered_inputs.py \
  --raw-csv /path/to/raw_monitoring.csv \
  --saits-clean-csv /path/to/saits_clean.csv \
  --output-dir artifacts/engineered

python scripts/build_filtered_response.py \
  --raw-csv /path/to/raw_monitoring.csv \
  --saits-clean-csv /path/to/saits_clean.csv \
  --engineered-env-csv artifacts/engineered/dam_2h_saits_dx_engineered_env_nomask.csv \
  --output-dir artifacts/filtered-response
```

SAITS is an adopted completion component rather than the paper's proposed
forecasting model. The current release accepts a SAITS-completed table as an
input and does not redistribute the project-specific completion dataset.

## Scope and claims

- The synthetic smoke test demonstrates software executability, not paper-grade
  accuracy.
- The reported metrics in `results/` are manuscript values, not results
  regenerated from public data in this repository.
- Attention weights indicate model relevance and are not causal estimates of
  physical lag.

## License and citation

A source-code license and final citation metadata will be added after the
authors confirm institutional intellectual-property requirements and the final
paper author list, before the repository is made public.
