# DamXer data release

The monitoring data are not stored in this Git repository. Under the available
data-owner authorization, the processed forecasting benchmark is planned for a
Mendeley Data deposit after manuscript acceptance. No upload is performed by
the reproduction scripts.

The release object contains only the five tables consumed by the frozen paper
protocol:

1. continuous SAITS-completed displacement histories;
2. displacement histories plus the 79 engineered environmental features;
3. displacement histories plus 111 completed original environmental channels;
4. filtered structural-response targets; and
5. the original missingness mask used for observed-only supervision.

It excludes the complete source archive, coordinates, internal paths,
original-to-anonymized identifier mappings, owner documents, and model
checkpoints.

## Before deposit

1. Re-run the strict input checks and five-seed metric verifier in the code
   repository.
2. Verify the committed byte sizes and SHA-256 values in
   `benchmark_manifest.json` against the private staging copy; the same
   canonical hashes are frozen in `configs/paper_clean_window.json`.
3. Confirm that the five CSV headers follow `data/schema.json` and that the date
   range is `[2024-01-01, 2025-12-01)`.
4. Prepare the Mendeley Data record as a draft only after acceptance, then
   inspect its title, authors, license, description, and anonymization before
   publication.
5. Replace the manuscript's temporary availability statement with the final
   Mendeley Data DOI only after the public record exists.
