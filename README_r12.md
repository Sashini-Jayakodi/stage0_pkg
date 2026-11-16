# r12: the r11 forecast on the GTSRB frame-level audit

Exploratory addition after r11; GTSRB is development data for the forecast. Definitions, observed rates, bootstrap and
the HB criteria are in `LOG_r12_entry.md` Part A (6491dc8), committed before any r12 code or number existed.

Status: run once on 16 Nov 2026 (code 53f3da6). Overall FAIL (all four HB judgments). Results in `LOG_r12_entry.md`
Part B. A post hoc diagnostic of this result (no criteria) is r12b: `LOG_r12b_entry.md`, `src/r12b_diagnostic.py`.

## What it does
The r11 forecast (`src/r11_forecast_ucb.py`, imported unchanged) with GTSRB inputs: the single pilot is the 261 design
tracks (30 frames each) on the POC-2 B1 grid; the walk starts at position 19 (the r11 start rule gives the POC-2 B1
order start; checked at run time). The audit is the POC-2 Stage 3b resampling: 2,000 draws of 784 of the 1,046 pool
tracks. Observed: the Stage 3b B1 HB rates (reproduced exactly) and a new exact-binomial B1 rate on the same draws. A
1,000-resample bootstrap of the 261 tracks gives a 10-90% interval for the forecast.

## The run (once)
```
py -3.12 -m src.r12_gtsrb_forecast --run runs/20261003_stage1
```
- Calls `load_frozen()` (R8) and `check_frozen()` (R9); hash-checks `stage3b_summary.json` and the POC-2 inputs through
  `src/r8_stage3_resample.py`; stops unless the design `work_c`/`work_e`, the start position 19 and the Stage 3b B1 HB
  rows reproduce.
- Prints the per-alpha table, the bootstrap detail and the Part A (f) criteria.
- Writes `runs/20261003_stage1/r12/r12_summary.json` and `r12_report.md`.

## Tests
```
py -3.12 -m pytest tests/test_r12_gtsrb_forecast.py -q -p no:cacheprovider
```
The forecast equals r11's on the same inputs; brute-force references for the binomial walk, per-track counts and the
bootstrap; criteria hand cases; tamper test for `stage3b_summary.json`.
