# DamXer data release

The paper dataset is released separately from the source-code repository.
This directory contains only the release contract and safe upload tooling; it
does not contain monitoring values.

## Current status

Public redistribution of the processed monitoring tables and original point
identifiers requires written approval from the data owner. Until that approval
is documented, do not stage or upload the real CSV files, including to a
restricted third-party draft.

The intended release object is a **processed forecasting benchmark**, not the
complete 289-variable source archive. It contains the four tables consumed by
the frozen DamXer experiment:

1. continuous SAITS-clean displacement histories;
2. the same displacement histories plus engineered environmental features;
3. filtered structural-response targets; and
4. the original observation mask used for observed-only supervision.

## Safe workflow

1. Keep the GitHub repository code-only and run the synthetic smoke test.
2. Obtain written owner approval for the exact release artifact and identifier
   policy.
3. Fill every `TODO` field in `zenodo_metadata.template.json` and save it as a
   private local metadata file, for example `artifacts/zenodo_metadata.json`.
4. Validate the four canonical CSV files locally:

   ```bash
   python scripts/zenodo_release.py validate \
     --input-root /path/to/downstream_inputs \
     --target-root /path/to/filtered_response \
     --manifest artifacts/private_benchmark_manifest.json
   ```

5. Stage a verified ZIP after approval:

   ```bash
   python scripts/zenodo_release.py stage \
     --input-root /path/to/downstream_inputs \
     --target-root /path/to/filtered_response \
     --manifest artifacts/private_benchmark_manifest.json \
     --approval-file /path/to/written_owner_approval.pdf \
     --confirm-owner-approval \
     --output artifacts/damxer_processed_forecasting_benchmark_v1.zip
   ```

6. Create a restricted Zenodo draft through the REST API:

   ```bash
   export ZENODO_ACCESS_TOKEN='stored-locally-not-in-git'
   python scripts/zenodo_release.py upload-draft \
     --archive artifacts/damxer_processed_forecasting_benchmark_v1.zip \
     --metadata artifacts/zenodo_metadata.json \
     --approval-file /path/to/written_owner_approval.pdf \
     --confirm-owner-approval
   ```

The upload command creates and fills a draft only. It deliberately does not
publish the record. Review the draft in Zenodo, publish it with restricted file
visibility, and then create a secret `Can view` link for editors/reviewers.
After acceptance, change the file visibility to public.

Use `--sandbox` with `upload-draft` to test against
`https://sandbox.zenodo.org` before touching the production service.

## Security notes

- Never paste a Zenodo token into source files, command history shared in a
  paper supplement, issue, or chat transcript.
- Zenodo record metadata is public after publication even when files are
  restricted. Keep the title and description site-anonymized.
- A private or restricted upload is still disclosure to a third party; owner
  approval must precede upload.
- Original-to-anonymized point mappings, coordinates, internal paths, raw
  acquisition exports, and owner documents are excluded from the ZIP.
