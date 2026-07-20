# DamXer data release

The monitoring data are not stored in this Git repository. The data owner has
authorized publication of the anonymized monitoring dataset and processed
forecasting benchmark. A Mendeley Data draft will be prepared after manuscript
acceptance and made publicly available upon publication. No upload is performed
by the reproduction scripts.

The release object contains two layers. The source layer provides the
anonymized incomplete 10448-by-289 monitoring table required to train SAITS and
rerun the completion comparison. The benchmark layer provides the five tables
consumed by the frozen forecasting protocol:

1. continuous SAITS-completed displacement histories;
2. displacement histories plus the 79 engineered environmental features;
3. displacement histories plus 111 completed original environmental channels;
4. filtered structural-response targets; and
5. the original missingness mask used for observed-only supervision.

It excludes coordinates, internal paths, original-to-anonymized identifier
mappings, owner documents, acquisition-system metadata, and model checkpoints.

## Before deposit

1. Re-run the strict input checks and five-seed metric verifier in the code
   repository.
2. Run `scripts/prepare_data_release.py` to assemble a local ignored staging
   directory. The script performs no upload.
3. Verify the committed byte sizes and SHA-256 values in
   `benchmark_manifest.json` against the private staging copy; the same
   canonical hashes are frozen in `configs/paper_clean_window.json`.
4. Confirm that the five benchmark CSV headers follow `data/schema.json` and that the date
   range is `[2024-01-01, 2025-12-01)`.
5. Prepare the Mendeley Data record as a draft only after acceptance, then
   inspect its title, authors, license, description, and anonymization before
   publication.
6. Replace the manuscript's temporary availability statement with the final
   Mendeley Data DOI only after the public record exists.

Example local assembly:

```bash
python scripts/prepare_data_release.py \
  --source-raw /private/path/Dam_2h_final_values_dx_dy.csv \
  --data-root /private/path/engineered-inputs \
  --target-root /private/path/filtered-response \
  --output-dir release/staging
```

Add `--check-only` to validate the private source files without copying them.
