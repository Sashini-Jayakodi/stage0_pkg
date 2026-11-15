# POC-3 (R9) Stage 6: BelgiumTSC confirmation

Definitions: `LOG_r9_stage6_entry.md` Part A. Every value comes from `configs/frozen_r9_btsc.yaml` (F3). Outputs:
`runs/btsc/stage6/`.

## What it does
1. Checks the 4 frozen files by git blob ID (`check_frozen()`), before anything is read.
2. Hash-checks the Stage 4 inputs (design and pool) against `stage4_summary.json` and `stage5_design.json` against its
   recorded SHA-256 or blob ID.
3. Recomputes the design phase from the hash-checked inputs; it must write the frozen yaml again byte for byte (LF).
4. Builds the units-by-lambdas tables on the 626 pool units (and the views-by-lambdas table for B1); pool labels come
   through `src/r9_btsc_guard.py` after F3. Recomputed pool head outputs must equal the stored Stage 4 scores.
5. RQ1: 2,000 draws of 397 pool poles with replacement; sign-level certificate per score and view-level B1, every
   alpha, exact binomial and HB primary start; violation counts against the whole-pool true risk.
6. V-check: the 5 score variants at the primary alpha 3%, exact binomial, upper 95% Clopper-Pearson bound <= 0.12.
7. RQ2: the 5 frozen re-splits, every score, alpha and rule (starts must equal the frozen ones), the uncertified
   reference, B2, B5, B5c and B6; Test metrics of plan 5.4.
8. The plan section 8 comparison table at 3%, the S2 against B6 table, and every Test cell above alpha.

The runner refuses to overwrite a recorded `stage6_summary.json` and writes the whole printout to
`stage6_printout.txt`.

## Command
```
py -3.12 -m src.r9_stage6
```
Tests: `py -3.12 -m pytest tests/test_r9_stage6.py -q`.
