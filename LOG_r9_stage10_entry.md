## POC-3 (R9) Stage 10: prospective check of the effective-delta formula (declared addition, synthetic)

### Part A: fixed definitions (committed and pushed before any Stage 10 code exists or any Stage 10 simulation is run)

Plan: `docs/poc3-implementation-plan-r9.md` (SHA-256 c0639973c70f35ac5cda1f11c49bf50a1639c632c48ccec8889d60a5ded03db4).
Plan R9 has no Stage 10; this stage is a declared addition. It adds one analysis to RQ1 and changes no frozen file
(`src/r9_certify.py`, `tests/test_r9_certify.py`, plan R9, `configs/frozen_r9_btsc.yaml`) and no earlier result.
Synthetic data only for the prospective part; the retrospective part reads committed summaries only. No GTSRB or
BelgiumTSC data file is opened. delta = 0.1 throughout.

#### (a) The formula (fixed now)
- A frame-level (view-level) Learn-then-Test certificate at nominal delta treats every frame as an independent unit.
  If the wrong indicator has within-sign intraclass correlation rho and each sign has m frames, the design effect is
  DE = 1 + (m - 1) * rho. The predicted worst-case violation probability is
  delta_eff = Phi(Phi^-1(delta) / sqrt(DE)), with Phi the standard normal CDF and Phi^-1(0.1) = -1.281552.
- Plain reading: the frame-level test believes its error count is DE times less variable than it is, so the cut-off it
  uses at the 10% level is crossed by chance more often, by the amount the formula gives.
- What the formula is and is not, stated before any run:
  - It is a normal approximation for a threshold whose true risk sits just above alpha. It is not a proven bound.
  - It does not contain alpha, the grid or the test. Where no start-eligible B1 threshold has true risk just above
    alpha, the observed rate can be far below delta_eff (as in BelgiumTSC Stage 6 at alpha 2% and above, where it is
    0). Agreement in those settings says little about the formula; (e) reports them separately.
  - rho and m are the population values of the simulator (rho = wrong-indicator ICC over all frames, as solved in
    Stage 2 (b)). The frame-level certificate works on accepted frames only, whose ICC and frames per sign at the
    certified threshold differ from rho and m. This is a known simplification of the formula, not corrected here.
- History: the formula was written after the Stage 2, POC-2 Stage 3b and Stage 6 frame-level results were known. Those
  results are therefore retrospective (d) and do not count as a test of the formula.

#### (b) Prospective settings (fixed now; none was used in Stage 2)
- rho in {0.1, 0.2, 0.6} x m in {5, 10, 15} x alpha in {1%, 2%}: 18 settings, 9 (rho, m) pairs.
- Error level: P_FRAME = 0.074 (Stage 2 level 0), m_latent = -Phi^-1(0.074); r solved so the wrong-indicator ICC equals
  rho (`src.r9_stage2_popcurves.solve_r`, imported).
- Cert units n = 784 per run and design units n_design = 261 per run (the GTSRB sizes), the same for every m, so only
  rho and m change between settings. B1 uses n = 784 x m Cert frames and 261 x m design frames.
- Model, commit rule, grids, B1 definition, power-aware start, HB start rules, walk and violation definition: exactly
  Stage 2 Part A (b), (c), (e), (h) and Amendments 1 and 2. The sign-level grid and the B1 grid are built per (rho, m)
  from a 20,000-unit pilot sample by the Stage 2 rule (30 points, commit or accept fractions 100% to 30%,
  lambda_0 = 0).
- Engine: the frozen `src/r9_certify.py`, called through `certify` and `start_for` as in Stage 2; `check_frozen()`
  runs first.
- Simulator: `src.r9_stage2_popcurves.draw` only accepts the three Stage 2 cluster names, so Stage 10 adds a
  fixed-T draw in a new file with the same draw order. A test checks that for T = 3 and T = 30 it returns exactly what
  `draw(..., "fixed-3" / "fixed-30", ...)` returns for the same generator state. No Stage 2 file is edited.
- Runs: 2,000 per (rho, m). Population: 1,000,000 units per (rho, m) for the true risk curves. As in Stage 2, the
  same draws serve both alphas, both tests and both units.
- Seeds (new; Stage 2 used [20261114, 9, level, pair, stream]): numpy `default_rng([20261115, 10, pair, stream])`,
  pair = index of (rho, m) in the order rho 0.1, 0.2, 0.6 x m 5, 10, 15 (pair 0..8); stream 0 pilot grid,
  1 population, 2 runs.

#### (c) Predicted delta_eff (fixed now, before any Stage 10 run)
delta = 0.1. The prediction does not depend on alpha, so each value applies to both alphas.

| pair | rho | m | DE | predicted delta_eff |
|---|---|---|---|---|
| 0 | 0.1 | 5 | 1.40 | 13.94% |
| 1 | 0.1 | 10 | 1.90 | 17.63% |
| 2 | 0.1 | 15 | 2.40 | 20.41% |
| 3 | 0.2 | 5 | 1.80 | 16.97% |
| 4 | 0.2 | 10 | 2.80 | 22.19% |
| 5 | 0.2 | 15 | 3.80 | 25.55% |
| 6 | 0.6 | 5 | 3.40 | 24.35% |
| 7 | 0.6 | 10 | 6.40 | 30.62% |
| 8 | 0.6 | 15 | 9.40 | 33.80% |

Computed with scipy.stats.norm (cdf, ppf); the runner recomputes them and stops if any differs from this table by more
than 0.005 percentage points.

#### (d) Retrospective table (labelled retrospective: the formula was written after these results)
Read from committed summaries, each SHA-256-checked against the value recorded in `LOG.md` before reading.
| Source | Unit and test | rho, m | DE | predicted delta_eff | Observed |
|---|---|---|---|---|---|
| Stage 2, all 36 settings (`stage2_summary.json`, 8c152579...729f) | frame, exact binomial (B1_binom); B1_hb (primary start) beside it | rho 0: any m | 1 | 10.00% | from the summary |
| | | 0.3, 3 | 1.60 | 15.55% | from the summary |
| | | 0.3, view-mix (m* = 2.998) | 1.60 | 15.55% | from the summary |
| | | 0.3, 30 | 9.70 | 34.04% | from the summary |
| | | 0.45, 3 | 1.90 | 17.63% | from the summary |
| | | 0.45, view-mix (m* = 2.998) | 1.90 | 17.62% | from the summary |
| | | 0.45, 30 | 14.05 | 36.62% | from the summary |
| GTSRB POC-2 Stage 3b, alphas 0.5 to 5% (1% primary) (`stage3b_summary.json`, 7591a438...7367) | frame, HB (POC-2 engine) | 0.446, 30 (POC-1 measurement) | 13.9 | 36.55% | from the summary (1%: 20.0% [18.3, 21.8]%) |
| BelgiumTSC Stage 6, alphas 0.5 to 5% (`stage6_summary.json`, 71be7957...185f) | view, exact binomial and HB primary start | 0.631 [0.126, 0.887], 2.96 views | 2.24 [1.25, 2.74] | 19.59% [12.58, 21.94]% | from the summary |
- View-mix: variable cluster sizes use m* = sum(T^2) / sum(T) over the Stage 2 mix probabilities (14 / 8 / 816 / 10 out
  of 848 for T = 1 / 2 / 3 / 4), m* = 2.998.
- GTSRB DE: the recorded POC-1 value 13.9 is used (1 + 29 x 0.446 = 13.93 gives 36.57%).
- BelgiumTSC: the interval on delta_eff is delta_eff at the ends of the recorded DE interval.
- All 36 Stage 2 rows are shown, not only the worst cases, so the table cannot be read as selected.
- The GTSRB row used HB, which is conservative even for independent frames; the formula ignores the test.

#### (e) Outcome (reported, not scored)
Per prospective setting (18 rows): predicted delta_eff; observed frame-level violation rate (B1, exact binomial) with
two-sided 95% Clopper-Pearson interval (`src.r8_stage3_engine.rate_row`); the same for B1 with HB (primary start);
sign-level violation rate (exact binomial and HB) with CP interval; certification rates; and the boundary flag from
(h).
- Stated check: a setting agrees with the formula when the observed frame-level rate does not exceed delta_eff beyond
  its CP interval, that is CP lower bound <= delta_eff.
- Summary: the number of the 18 settings that agree; mean absolute gap |observed - predicted| and mean signed gap
  (observed - predicted), in percentage points, over all 18 and separately over the boundary subset of (h).
- A disagreement is reported as it comes out; nothing in (a) to (c) changes after the run.
- The sign-level rate is reported, not scored. A sign-level CP upper bound above 0.12 would be a result at odds with
  Stage 2 V2 and is reported as such; the frozen engine is not changed.

#### (f) Code, tests and outputs
- `src/r9_stage10_popcurves.py`: fixed-T draw, pilot grids, population curves and the boundary diagnostic (h); no
  Cert sample is drawn in this file.
- `src/r9_stage10.py`: runner (runs, rates, agreement summary, retrospective table); imports `src.r9_certify`,
  `src.r9_frozen`, `src.r9_stage10_popcurves`, `src.r9_stage2_popcurves`, `src.r9_stage2_vchecks`,
  `src.r8_stage3_synth`, `src.r8_stage3_engine`. No earlier file is edited.
- `tests/test_r9_stage10_popcurves.py`, `tests/test_r9_stage10.py`: delta_eff against a brute-force normal-CDF
  reference and against table (c); fixed-T draw equals the Stage 2 draw for T = 3 and 30; the boundary diagnostic
  against a loop; a small run; the agreement rule and gaps against a loop; tamper tests (a changed curve file,
  `stage2_summary.json`, `stage3b_summary.json` or `stage6_summary.json` is refused; the runner refuses to write if
  `src/r9_certify.py` changed during the run); a guard test that no path under `data/` appears in either module.
- Outputs in `runs/20261003_stage1/poc3/stage10/`: `population_curves_stage10.json` (step 2), `stage10_summary.json`
  and the printout (step 4). `README_r9_stage10.md` gives both commands.

#### (g) Order
1. This Part A with table (c) committed and pushed.
2. Population-curve script and tests; full suite; commit; run it; append the boundary subset (h) with the curve file
   hash to this entry; commit and push, before any Cert sample is drawn.
3. Runner and tests (including the reproduction test); full suite; commit and push.
4. One run; the whole printout kept; then stop. Part B only after the printout has been reviewed.

#### (h) Boundary diagnostic (rule fixed now; values computed from population curves only, before any Cert run)
- Population curves per (rho, m): the B1 grid from the pilot sample (stream 0); B1 accept fraction and true risk
  R(j) = accepted and wrong frames / accepted frames on the 1,000,000-unit population (stream 1). The same for the
  sign-level grid.
- Per (rho, m, alpha): the binomial power-aware start j_s from the population accept fractions with n = 784 x m Cert
  frames (`src.r9_certify.power_start`). Walk order j_s, j_s - 1, ..., 0. R_b = R(j) at the first j in that order with
  R(j) > alpha (R(j_s) itself if it is above alpha).
- A setting is in the boundary subset when R_b <= 2 x alpha: a threshold only slightly above alpha is reachable by the
  walk, so the formula's worst case can occur there. Settings outside the subset need a threshold well above alpha to
  pass the test, so a low observed rate there says little about the formula.
- The population accept fractions stand in for the per-run design fractions; the runner reports the start actually
  used per run.
- The runner recomputes the curves with the same seeds and stops if they differ from the committed curve file
  (hash-checked first).
