# DamXer processed forecasting benchmark

## Scope

This dataset contains (i) an anonymized incomplete 289-channel monitoring table
used to train and evaluate the completion stage and (ii) the processed input,
target, and missingness-mask tables used by the frozen DamXer forecasting
experiment. It does not contain coordinates, project names, acquisition-system
identifiers, or an original-to-anonymized point mapping.

## Temporal coverage

- Sampling interval: 2 h
- Number of timestamps: 8400
- Window: 2024-01-01 00:00 through 2025-11-30 22:00
- Split: chronological 70/10/20

Calendar timestamps and point-label policy must match the data-owner approval
for the released version. If the approved release uses relative time or
deterministic point aliases, the manifest and code configuration must be
versioned with that transformation.

## Files

- `source/anonymized_monitoring_2h_incomplete.csv`: 10448 timestamps and 289
  anonymized H, seeP, dx, dy, and temperature channels. NaN cells encode the
  original missingness used by the completion stage.
- `inputs/dam_2h_saits_dx_only_nomask.csv`: continuous completed histories for
  89 horizontal-displacement channels.
- `inputs/dam_2h_saits_dx_engineered_env_nomask.csv`: the same 89 channels,
  followed by 79 engineered hydraulic, seepage, thermal, and time features.
- `inputs/dam_2h_saits_dx_raw_env_nomask.csv`: the same 89 channels, followed
  by 111 completed original hydraulic, seepage, and temperature channels.
- `targets/filtered_response.csv`: Median(5) and Savitzky--Golay(9, 3)
  structural-response target for the same 89 channels.
- `targets/dam_2h_filtered_response_observed_mask.csv`: target mask where `0`
  denotes an original observation and `1` denotes an originally missing value
  excluded from supervised loss and metrics.

## Intended use

The package supports reproduction and extension of the paper's processed
forecasting protocol. It must not be interpreted as a raw sensing archive, a
real-time causal target, or a general benchmark covering multiple dams.

## Limitations and responsible use

The data originate from one earth-rockfill-dam case. Filtered response is an
offline evaluation target. Engineered environmental features are derived from
long monitoring histories and do not establish causal physical mechanisms.
Users must not attempt to re-identify the project, infer protected operational
details, or combine the trajectories with external sources for site discovery.

## Code

The model, frozen configuration, schema, synthetic verification example, and
evaluation pipeline are maintained separately at
`https://github.com/murasame612/damXer`.

## License and citation

The applicable data license and citation will be defined by the final Mendeley
Data record. The source-code license does not automatically apply to these
monitoring data.
