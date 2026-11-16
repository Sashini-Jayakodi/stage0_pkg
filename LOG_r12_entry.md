## r12: the r11 forecast applied to the GTSRB frame-level resampling audit (exploratory, development data)

### Part A: fixed definitions (committed before any r12 code exists or any r12 number is computed)

Plan: `docs/poc3-implementation-plan-r9.md` (SHA-256 c0639973c70f35ac5cda1f11c49bf50a1639c632c48ccec8889d60a5ded03db4).
Exploratory addition after r11 (Part A 867db93, Amendment 1 109ec7b, Part B 9e8ca56). It changes no frozen file
(`src/r9_certify.py`, `tests/test_r9_certify.py`, plan R9, `configs/frozen_r9_btsc.yaml`, `configs/frozen_r8.yaml`, plan
R8), no POC-1 or POC-2 file and no earlier output. delta = 0.1 throughout. Every GTSRB track has been seen before, so
everything here is exploratory (post-hoc).

#### (a) Status of this check, stated before any run
- GTSRB is development data for the forecast. `src/r10_prelim_micro.py` Part 3 ran an earlier forecast on these same
  261 design tracks and these same 2,000 draws (16.6% against observed 17.0% at alpha 0.5%; 47.3% against 20.0% at
  1%), and the upper-bound rule of r11 (b) was written after that result (r11 Part A (a)). The observed rates in (d)
  are known. The criteria in (f) are therefore a consistency check on development data, not an out-of-sample test.
- There is one pilot (the 261 design tracks), not one per run as in r11. A single pilot's forecast varies widely (r11
  reported 10th-90th percentile ranges over pilots of several to tens of points); the bootstrap interval in (e)
  shows this spread for this pilot.

#### (b) The forecast: r11 Part A (b) and Amendment 1, unchanged
Every step of r11 (b) 1-6 and Amendment 1 applies as written, with these GTSRB inputs (substitutions, not changes):
- Pilot = the 261 seed-0 Fit tracks (`track_split == 0` in `splits/seed0.npz`), m = 30 frames each (7,830 design frames
  in `b1_design.json`), on the POC-2 B1 grid (20 working thresholds of `b1_design.json`). A_j, W_j and the per-track
  accepted-and-wrong counts are computed as in r10 Part 3 and checked against `b1_design.json` `work_c` and `work_e`.
- Cert size n = 784 tracks, N_c = 784 x 30 = 23,520 frames in the model.
- Walk start: the r11 rule, `start_for(t, work_c, 23520, alpha, 7830)`, gives position 19 for both tests at alpha 0.5,
  1 and 2% (computed from `work_c`, a design quantity), which is the POC-2 B1 order (19 down to 0). The runner checks
  this and stops if it differs.
- HB p-value: `C.hb_pvalue_counts(k, n_c, 23520, alpha)` = `hb_pvalue(alpha (1 - n_c / n) + k / n, n)`, the same
  mean-loss form as the POC-2 B1 test (`r8_stage3_engine.frame_mu`).
- The only change forced by the data: r11 (b) 7 averaged per-run forecasts over 2,000 pilots; here the forecast is the
  single value from the one pilot, with the bootstrap interval in (e) beside it.
- Known model difference: the pool contains one 29-frame track; the model gives every Cert track 30 frames. The
  observed rates use the real frame counts.

#### (c) Data
- Audit = the POC-2 Stage 3b draws: 2,000 draws of 784 of the 1,046 pool tracks, with replacement,
  `RS.draw_weights(1046, 784, 2000, 20261115)` (`src/r8_stage3_resample.py`). Inputs loaded and hash-checked by
  `RS.load_pool` / `RS.pool_tables` (POC-2 Stage 1 outputs, head, splits, `b1_design.json`); `stage3b_summary.json`
  checked against its recorded SHA-256 7591a438b569e7231f3a40d8da6420a81b6f65791e707df4c0101338a0bf7367.
  `load_frozen()` (R8) and `check_frozen()` (R9) run first.
- Pool true risk per threshold = pool micro risk W / A over the 1,046 pool tracks; a false certificate = a certified
  threshold whose pool mean loss is above alpha (`r8_stage3_engine.violations`, as Stage 3b).
- alpha in {0.5, 1, 2}%. Draw count 2,000 (the existing audit).

#### (d) Observed rates
- HB (primary): the Stage 3b B1 rows, recomputed on the same draws; the run stops unless violations and certified share
  equal `stage3b_summary.json` (0.5%: 340 of 2,000, 17.00%; 1%: 400, 20.00%; 2%: 0, 0.00%).
- Exact binomial (secondary, new): on the same 2,000 draws, B1 certified with `C.binom_pvalue(Wd W, Wd A, alpha)` and
  the same walk (position 19 down to 0); violation as above. Reported with its 95% Clopper-Pearson interval.

#### (e) Pilot bootstrap interval
- 1,000 resamples of the 261 design tracks, with replacement, by track; generator `default_rng([20261119, 12])`
  (base 20261119 unused by any earlier script), one (1000, 261) index array drawn once and used for every alpha and
  both tests. In each resample every step of (b) is recomputed (A_j, W_j, U_j, j*, f, rho_I, N_g, forecast).
- Reported: 10th and 90th percentiles of the resampled forecasts (numpy default linear interpolation), beside the
  pilot's own forecast. Descriptive only; it does not enter the criteria.

#### (f) Criteria (fixed now; HB only; consistency check on development data, see (a))
At alpha 0.5% and at alpha 1%, with "forecast" the pilot's own HB forecast and "observed" the Stage 3b HB B1
false-certificate rate, in percentage points:
- (a) |forecast - observed| <= 3 points
- (b) forecast >= observed - 2 points
Four judgments (two criteria x two alphas), each reported as PASS or FAIL. The check passes only if all four pass.
At alpha 2% (observed 0.00%): reported only. Exact binomial: reported only.

#### (g) Reported for each alpha and test
Observed rate [95% CP] and certified share; pilot forecast; bootstrap 10-90%; heuristic delta_eff with the recorded
POC-1 design effect 13.9 (36.55%, as Stage 10); j*, U_j*, f, rho_I (and which case: beta-binomial, rho_I <= 0, zero
errors, rho_I >= 1), N_g, k_max, clipped mass; j_true (first position in the order with pool risk above alpha) and
whether j* is at or before it; the share of bootstrap resamples with forecast 0 and with each rho_I case.

#### (h) Checks (all must hold, else the run stops) and tests
- Hash checks as in (c); `b1_design.json` `work_c` and `work_e` reproduced from the design tracks; start position 19 for
  every alpha and test; Stage 3b B1 HB rows reproduced exactly.
- Tests (`tests/test_r12_gtsrb_forecast.py`, synthetic): the r12 forecast equals r11's `run_forecast` on the same
  inputs (imported, not copied); the binomial walk against a brute-force loop; the bootstrap recomputation against a
  direct loop on one resample; tamper tests (a changed `stage3b_summary.json` is refused).

#### (i) Order of work and deviations
1. This Part A committed before any r12 code exists. 2. Code and tests; full suite; commit. 3. One run after the
author approves the tests; the whole printout kept; then stop. Part B only after review.
- Deviation: file names `src/r12_gtsrb_forecast.py`, `tests/test_r12_gtsrb_forecast.py`, `LOG_r12_entry.md`,
  `README_r12.md` and output folder `runs/20261003_stage1/r12/` are outside the POC-3 `r9_` naming rule, by the
  author's decision, as for r11.

### Part B: results (filled from the printout after review)

#### Run
- 16 Nov 2026, one run: `py -3.12 -m src.r12_gtsrb_forecast --run runs/20261003_stage1` on commit 53f3da6
  (`src/r12_gtsrb_forecast.py` SHA-256 615b6f537c78a4d3b229c34a02d38685ce7a0f65cc4668af61b9df235850d22d), 2.8 s,
  exit code 0, no crash, no fix. Not rerun.
- Every Part A (h) stop-check passed (printout line: "Checks (Part A (h)): frozen_grid_matches_stage1_grid PASS;
  gate_flags_reproduce PASS; pool_commit_table_reproduces PASS; b1_top_equals_stored_top1 PASS; design_tracks_261 PASS;
  design_frames_30_each PASS; design_work_c_reproduces PASS; design_work_e_reproduces PASS;
  start_19_every_alpha_and_test PASS; stage3b_b1_hb_rows_reproduce PASS").
- Outputs (`runs/20261003_stage1/r12/`): `r12_summary.json` SHA-256
  3d559b2fa5ec9de5d25b2d4e01079a083003ff45bee0c2f0b4c5f4a06d18488a; `r12_report.md` SHA-256
  ee0ed73455e79d434fe3a938233909a58fa8cddd3968b5a2ef95aa46aac47f07.

#### Result
**Overall: FAIL** (all four HB judgments failed: (a) and (b) at alpha 0.5% and at alpha 1%).

Criteria block, verbatim:
```
CRITERIA (Part A (f); HB only; consistency check on development data; 2% and exact binomial reported only):
  alpha 0.5%: forecast 6.61%, observed 17.00%, gap -10.39 pp
    (a) |forecast - observed| <= 3 points: FAIL
    (b) forecast >= observed - 2 points: FAIL
  alpha 1.0%: forecast 16.94%, observed 20.00%, gap -3.06 pp
    (a) |forecast - observed| <= 3 points: FAIL
    (b) forecast >= observed - 2 points: FAIL
  Overall (all four must pass): FAIL
```

Table, verbatim:
```
  alpha  test | observed               CP95   cert | forecast       boot p10-p90 boot med |   heur. |    gap | j*    U_j*      f   rho_I        case N_g k_max     clip | jt j*<=jt pool R(j*)
   0.5%    hb |  17.00% [15.38, 18.72]%  60.0% |   6.61% [  1.64,  16.29]%   9.05% |  36.55% | -10.39 | 13  0.559%  76.3%  0.1156   betabinom   23    70 6.24e-07 | 12    yes    0.495%
   0.5% binom |  26.55% [24.63, 28.54]%  73.1% |  13.33% [  4.75,  27.22]%  17.77% |  36.55% | -13.22 | 13  0.559%  76.3%  0.1156   betabinom   23    77 6.24e-07 | 12    yes    0.495%
   1.0%    hb |  20.00% [18.27, 21.82]%  99.9% |  16.94% [  0.00,  19.44]%   5.59% |  36.55% |  -3.06 |  0  1.035%  83.2%  0.1945   betabinom   25   167 1.53e-05 |  1     no    1.072%
   1.0% binom |  29.30% [27.31, 31.35]% 100.0% |  25.48% [  0.00,  29.39]%  10.63% |  36.55% |  -3.82 |  0  1.035%  83.2%  0.1945   betabinom   25   177 1.53e-05 |  1     no    1.072%
   2.0%    hb |   0.00% [ 0.00,  0.18]% 100.0% |   0.00% [  0.00,   0.00]%   0.00% |  36.55% |  +0.00 | -1     n/a    n/a     n/a        None  n/a   n/a 0.00e+00 | -1    n/a       n/a
   2.0% binom |   0.00% [ 0.00,  0.18]% 100.0% |   0.00% [  0.00,   0.00]%   0.00% |  36.55% |  +0.00 | -1     n/a    n/a     n/a        None  n/a   n/a 0.00e+00 | -1    n/a       n/a

Bootstrap detail (counts of 1,000): forecast zero share; rho_I case counts; resamples whose start is not 19
   0.5%    hb: zero   2.4%; betabinom 976, rho_I<=0 0, zero errors 0, rho_I>=1 0, no j* 24; start != 19: 0
   0.5% binom: zero   2.4%; betabinom 976, rho_I<=0 0, zero errors 0, rho_I>=1 0, no j* 24; start != 19: 0
   1.0%    hb: zero  48.6%; betabinom 514, rho_I<=0 0, zero errors 0, rho_I>=1 0, no j* 486; start != 19: 0
   1.0% binom: zero  48.6%; betabinom 514, rho_I<=0 0, zero errors 0, rho_I>=1 0, no j* 486; start != 19: 0
   2.0%    hb: zero  99.8%; betabinom 2, rho_I<=0 0, zero errors 0, rho_I>=1 0, no j* 998; start != 19: 0
   2.0% binom: zero  99.8%; betabinom 2, rho_I<=0 0, zero errors 0, rho_I>=1 0, no j* 998; start != 19: 0
```

#### Deviations
- None in the run. `README_r12.md`, named in Part A (i), was not written before the run.
