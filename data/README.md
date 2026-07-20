# Data interface

The Mendeley release has two layers: the anonymized incomplete monitoring table
used by the completion stage and the five frozen processed forecasting tables.
See [`release/README.md`](../release/README.md) for staging and hash checks.

The real dam-monitoring dataset is not included in this code repository. Data
availability is documented separately in the associated manuscript. The
synthetic example verifies that the software pipeline executes; it cannot
reproduce the numerical results reported in the paper.

## Paper input files

The frozen clean-window experiment expects five aligned CSV files:

```text
<data-root>/dam_2h_saits_dx_only_nomask.csv
<data-root>/dam_2h_saits_dx_engineered_env_nomask.csv
<data-root>/dam_2h_saits_dx_raw_env_nomask.csv
<target-root>/filtered_response.csv
<target-root>/dam_2h_filtered_response_observed_mask.csv
```

All files must contain the same ordered timestamps in a first column named
`date`.

- `dam_2h_saits_dx_only_nomask.csv`: `date` followed by 89 continuous,
  SAITS-completed horizontal-displacement channels.
- `dam_2h_saits_dx_engineered_env_nomask.csv`: the same 89 displacement
  columns first and in identical order, followed by engineered hydraulic,
  seepage, and thermal features.
- `dam_2h_saits_dx_raw_env_nomask.csv`: the same 89 displacement columns first,
  followed by 111 completed original channels in hydraulic, seepage, and
  temperature family order. It contains no lag, rolling, slope, family
  aggregation, or calendar features.
- `filtered_response.csv`: `date` followed by the same 89 channel names. The
  target is Median(5) followed by Savitzky--Golay(9, 3).
- `dam_2h_filtered_response_observed_mask.csv`: `date` followed by
  `<channel>_masked` columns. A value of `1` means that the corresponding
  future target cell was not originally observed and must be excluded from
  loss and metrics; `0` means an originally observed cell.

For the reported experiment, each file contains 8400 timestamps sampled every
2 h from 2024-01-01 00:00 through 2025-11-30 22:00. The chronological
70/10/20 split yields 5065/745/1585 train/validation/test prediction origins
for DamXer and the 720-step raw-ENV control. The 192-step displacement-only
generic baselines yield 5593/745/1585 origins.

See `schema.json` for the machine-readable contract and
`examples/run_synthetic_smoke.py` for an executable synthetic example.
