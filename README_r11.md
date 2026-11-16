# r11: forecast of the frame-level false-certificate rate from a pilot upper bound

Exploratory addition after POC-3 Stage 10; synthetic data only. Definitions, data sets, seeds and the held-out pass
criteria are in `LOG_r11_entry.md` Part A (867db93), committed before any r11 code or number existed, and Amendment 1
(109ec7b: pilots with rho_I >= 1 use the rho = 1 limit of the beta-binomial instead of stopping the run).

Status: run once on 16 Nov 2026 (code 109ec7b). Held-out check FAIL ((a) MAE 3.18 > 3.00; (b) and (c) pass). Results
and the post-run label correction of the two output files are in `LOG_r11_entry.md` Part B.

## What it does
For each run of each synthetic setting, the run's 261-sign pilot (design) sample gives, per B1 threshold, a one-sided
upper 80% Clopper-Pearson bound on the accepted-frame risk. The first threshold in the engine's walk order whose bound
exceeds alpha is j*. A beta-binomial model with the pilot's within-sign correlation at j*, and the risk set to the
bound, gives the probability that the frame-level test certifies at j*. The mean over runs is the setting forecast; it
is compared with the observed frame-level false-certificate rate (95% CP interval) and with the design-effect heuristic
Phi(Phi^-1(0.1) / sqrt(1 + (m - 1) rho)).

Three sets: development (the 18 Stage 10 settings, regenerated and checked against `stage10_summary.json`),
replication (the same 18 settings on new seeds, reported only) and held-out (rho 0.3/0.5 x m 8/12 x alpha 1/2%, judged
by Part A (e), exact binomial only).

## The run (once)
```
py -3.12 -m src.r11_forecast_ucb
```
- Calls `check_frozen()`, hash-checks `population_curves_stage10.json` and `stage10_summary.json`, recomputes the
  held-out heuristic table, and stops if the development regeneration does not reproduce the Stage 10 B1 violation
  counts exactly.
- 2,000 runs per setting, Cert 784 / design 261 signs, delta 0.1, the frozen engine `src/r9_certify.py`.
- Prints per setting and test the observed rate with its CP interval, the forecast (mean, median, 10-90% range), the
  heuristic, the gaps, the clipped mass, the share of runs with j* at or before the true first risky threshold, and the
  held-out criteria (a), (b), (c).
- Writes `runs/20261003_stage1/r11/r11_summary.json` and `r11_report.md`. Keep the whole printout.

## Tests
```
py -3.12 -m pytest tests/test_r11_forecast_ucb.py -q -p no:cacheprovider
```
Brute-force references for the CP bound, the ICC, the exact convolution (against sampling and enumeration), the k_max
search and the j* walk; tamper tests for the Stage 10 input files.
