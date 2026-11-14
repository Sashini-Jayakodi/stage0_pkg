# POC-3 (R9) Stage 4: BelgiumTSC model and cache

Definitions: `LOG_r9_stage4_entry.md` Part A. Outputs: `runs/btsc/stage4/`.

## What it does
1. `split`: hash-checks the Stage 1 summary, manifest, zips, extracted trees, GT files and readmes; keeps one view per
   image (class folder, file name); drops the 65 removed units listed in the Stage 1 summary; splits the testing poles
   133 design / 530 pool (rarest-class strata, largest remainder, seed 0) and the training poles 136 validation / rest
   fit. Writes `train_views.csv`, `test_views.csv`, `split_poles.json` (with the SHA-256 of each pole list).
2. `cache`: 112 x 112 whole-image caches (no ROI crop), `cache/train_112.npy` and `cache/test_112.npy`.
3. `train`: fine-tunes a 62-class ResNet-18 from local ImageNet weights with the POC-1 recipe (checked against
   `configs/poc.yaml`), early stopping on the training-release validation poles only.
4. `embed`: embeddings for the training fit part, training validation part, design poles and pool poles. The pool file
   holds no labels.
5. `head`: tied-Gaussian head fitted on the training fit part; scores for every set; design-pole accuracy (no
   threshold). Pool: row count only.
6. `vchecks`: V4.1 to V4.6 and `stage4_summary.json`.

Every step checks the frozen files first, hash-checks what earlier steps wrote, refuses to overwrite a recorded output
and appends to `stage4_printout.txt`. Testing labels are read only through `src/r9_btsc_guard.py`, which refuses any
pool-pole label until `configs/frozen_r9_btsc.yaml` is frozen (F3).

## Commands
```
py -3.12 -m src.r9_stage4 split
py -3.12 -m src.r9_stage4 cache
py -3.12 -m src.r9_stage4 train
py -3.12 -m src.r9_stage4 embed
py -3.12 -m src.r9_stage4 head
py -3.12 -m src.r9_stage4 vchecks
```
Tests: `py -3.12 -m pytest tests/test_r9_stage4.py tests/test_r9_btsc_guard.py -q`.
