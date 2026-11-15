# POC-3 Stage 10: prospective check of the effective-delta formula

Declared addition to plan R9; synthetic data plus committed summaries. Definitions in `LOG_r9_stage10_entry.md` Part A.

## Step 2: population curves and boundary subset (no Cert sample)
```
py -3.12 -m src.r9_stage10_popcurves
```
Writes `runs/20261003_stage1/poc3/stage10/population_curves_stage10.json` (SHA-256 recorded in Part A (h) values; the
runner refuses any other bytes).

## Step 4: the run (once)
```
py -3.12 -m src.r9_stage10
```
- Checks the frozen files (`check_frozen()`), the Part A (c) predictions, the curve file and the three retrospective
  summaries (Stage 2, POC-2 Stage 3b, Stage 6) by SHA-256, then recomputes the curves and stops if they differ.
- 9 (rho, m) pairs x 2,000 runs, Cert 784 / design 261 signs, alpha 1% and 2%, every rule of the frozen engine.
- Prints predicted delta_eff against the observed frame-level rate (95% CP) per setting with the boundary flag, the
  summary over all 18 settings and the boundary subset, the sign-level rates, and the retrospective table.
- Writes `runs/20261003_stage1/poc3/stage10/stage10_summary.json`. Keep the whole printout.

## Tests
```
py -3.12 -m pytest tests/test_r9_stage10_popcurves.py tests/test_r9_stage10.py -q -p no:cacheprovider
```
