# Data interface

The real dam-monitoring dataset is not included in this code repository. Data
availability is documented separately in the associated manuscript. The
synthetic example verifies that the software pipeline executes; it cannot
reproduce the numerical results reported in the paper.

## Paper input files

The frozen clean-window experiment expects four aligned CSV files:

```text
<data-root>/dam_2h_saits_dx_only_nomask.csv
<data-root>/dam_2h_saits_dx_engineered_env_nomask.csv
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
- `filtered_response.csv`: `date` followed by the same 89 channel names. The
  target is Median(5) followed by Savitzky--Golay(9, 3).
- `dam_2h_filtered_response_observed_mask.csv`: `date` followed by
  `<channel>_masked` columns. A value of `1` means that the corresponding
  future target cell was not originally observed and must be excluded from
  loss and metrics; `0` means trusted observation.

For the reported experiment, each file contains 8400 timestamps sampled every
2 h from 2024-01-01 00:00 through 2025-11-30 22:00. The chronological
70/10/20 split yields 5065/745/1585 prediction origins after windowing.

See `schema.json` for the machine-readable contract and
`examples/run_synthetic_smoke.py` for an executable synthetic example.
