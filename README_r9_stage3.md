# POC-3 (R9) Stage 3: GTSRB run of the layer on four scores (exploratory, post-hoc)

Definitions: `LOG_r9_stage3_entry.md` Part A. Every result of this stage is exploratory (post-hoc).

## Step 3: design phase (design tracks only)
```
py -3.12 -m src.r9_stage3_design --run runs/20261003_stage1
```
- Checks the frozen files by blob ID (`src/r9_frozen.py`), then reads the 261 design tracks through the POC-2 Stage 2
  loader (hash-checked; the pool table is refused).
- Writes `runs/20261003_stage1/poc3/stage3/stage3_design.json`: S2 halves and calibration (Sum, Min; K = 5), grids,
  design commit fractions and errors, starts per (score, alpha, rule) and the uncertified reference.
- Its SHA-256 goes into Part A before any Cert number is computed.

## Step 5: Cert/Test run
```
py -3.12 -m src.r9_stage3 --run runs/20261003_stage1 --config configs/poc.yaml
```
- Checks the frozen files, then the SHA-256 of `stage3_design.json`, `stage4_summary.json`, `src/r8_stage4.py`,
  `baselines_design.json` and `train_meta.csv`; loads the pool with the POC-2 loader and its reproduction checks.
- Regenerates the five Stage 4 re-splits and stops unless every certified Stage 4 Test coverage reproduces exactly.
- Certifies S1, S2-sum, S2-min, S3 and S4 at every alpha with the exact binomial and HB (three starts), and reports
  the plan 5.4 metrics, the uncertified reference and the smallest certifiable alpha.
- Writes `runs/20261003_stage1/poc3/stage3/stage3_summary.json`.

## Tests
```
py -3.12 -m pytest tests/test_r9_frozen.py tests/test_r9_stage3_design.py tests/test_r9_stage3.py -q -p no:cacheprovider
```
