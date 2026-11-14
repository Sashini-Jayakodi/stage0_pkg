# POC-3 (R9) Stage 1: BelgiumTSC data audit

Definitions: `LOG_r9_stage1_entry.md` Part A and Amendment 1. Stage 1 reads file names, the GT csv files, the readmes
and (for shared-unit check 2) image bytes. It loads no model and makes no design/pool split.

## Inputs (must already exist)
- `data/btsc_raw/zips/BelgiumTSC_Training.zip`, `data/btsc_raw/zips/BelgiumTSC_Testing.zip`
- `runs/btsc/stage1/downloads_manifest.json` (URL, times, size, SHA-256 of each zip)
- `data/btsc_raw/extracted/BelgiumTSC_Training/`, `data/btsc_raw/extracted/BelgiumTSC_Testing/`
  (made by `py -3.12 -m src.r9_stage1_integrity b`, Amendment 1 condition (b))

## Steps already run
- `py -3.12 -m src.r9_stage1_integrity a`: SHA-256 against the manifest, then every zip member's CRC.
- `py -3.12 -m src.r9_stage1_integrity b`: extraction into new empty folders and the `.ppm` image count.

## Audit run
```
py -3.12 -m src.r9_stage1_audit --check1-applies {yes|no} --dits-retry
```
- `--check1-applies`: the author's reading of the readme (Amendment rule in Part A (e)): `yes` only if the readme
  states that pole numbers are global across the training and testing releases. It is required, so the choice is made
  before the run and is written into the summary.
- `--dits-retry`: the one F-A3 retry of the DITS page (HTTPS, then HTTP), with time and error recorded. Nothing is
  downloaded from DITS.

## Input checks (refuses and stops)
- A zip whose SHA-256 differs from the manifest.
- An extracted tree that differs from its zip: a missing file, an extra file, or a file whose CRC differs from the zip
  member.
- An existing `runs/btsc/stage1/stage1_summary.json` (a recorded run is never overwritten).

## What it prints and writes
`runs/btsc/stage1/stage1_summary.json`, printed without the list of removed units, then the summary's SHA-256:
- `inputs_sha256`, `gt_sha256` (every GT csv), `readme_sha256`, `code_sha256`.
- `counts.training` / `counts.testing` (Part A (c)): images, class folders, distinct class IDs, pole numbers, sign
  units (pole, class), views per sign unit (1, 2, 3, 4+), sign units per class, poles with more than one class and the
  extra sign units they add, median views per pole, ROI shorter-side quantiles, consistency failures with examples.
- `shared` (Part A (e)): whether check 1 applied; flagged by check 1 only, check 2 only, both; check 1 and check 2
  totals; number removed.
- `gates` (Part A (f)): testing poles and sign units after removal; F-A1 (poles >= 600); F-A2.
- `projection` (Part A (g)): expected Cert size (0.6 x testing poles), expected n_c at a 70% commit fraction, and the
  largest tolerated number of errors at alpha 0.5, 1, 2, 3, 5% (exact binomial, delta 0.1; -1 = not certifiable).
- `dits`: both retry attempts and F-A3.

## Tests
`py -3.12 -m pytest tests/test_r9_stage1_audit.py tests/test_r9_stage1_integrity.py -q -p no:cacheprovider`
(synthetic trees and zips only).
