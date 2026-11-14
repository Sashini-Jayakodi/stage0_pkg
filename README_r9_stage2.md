# POC-3 Stage 2: certification engine v2 and V-checks

Definitions: `LOG_r9_stage2_entry.md` Part A. Synthetic data only; no dataset file is read.

## Files
- `src/r9_certify.py`: the engine (exact binomial test, Hoeffding-Bentkus comparator, power-aware start,
  fixed-sequence walk). Frozen after V1 to V4 pass.
- `src/r9_stage2_popcurves.py`: synthetic model and population curves (design values, Part A (f)).
- `src/r9_stage2_vchecks.py`: V1 to V4 and the reported B1 / Hoeffding-Bentkus rates.
- Tests: `tests/test_r9_certify.py`, `tests/test_r9_stage2_popcurves.py`, `tests/test_r9_stage2_vchecks.py`.

## Run
```
py -3.12 -m pytest tests -q -p no:cacheprovider
py -3.12 -m src.r9_stage2_vchecks
```
The runner checks the SHA-256 of `runs/20261003_stage1/poc3/stage2/population_curves_L0.json` and `_L1.json`
against the values recorded in Part A and stops if either differs. It writes
`runs/20261003_stage1/poc3/stage2/stage2_summary.json` and refuses to write it if `src/r9_certify.py` changed during
the run. Keep the whole printout for Part B.

Population curves (already committed; re-running gives the same values, only the `seconds` field differs):
```
py -3.12 -m src.r9_stage2_popcurves                            # level 0, P_FRAME 0.074
py -3.12 -m src.r9_stage2_popcurves --p-frame 0.05 --level 1   # added level
```
