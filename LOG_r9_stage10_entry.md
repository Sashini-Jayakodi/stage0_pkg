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

#### (h) values: boundary subset (step 2, 15 Nov 2026; population curves only, no Cert sample drawn)
- Script `src/r9_stage10_popcurves.py` committed in a6ed1b3 (SHA-256
  797aae0712ef69f2cbd1eb62968b8162c178423547d7ef5eb675fafa311bc72d) before it was run; full suite 547 passed. Run
  command `py -3.12 -m src.r9_stage10_popcurves`, 18.9 s.
- Output `runs/20261003_stage1/poc3/stage10/population_curves_stage10.json` SHA-256
  ffa785eafc545a45b4a2d364b072111f5f8dd9440d4b0f308753e2dcce0b119f (git blob 69ad7832cc91ad62a8f60ad6ec623074f611743a);
  printout `population_curves_printout.txt` SHA-256 b619b25756a328f82d279af18ee503e9c03e7927314baa358b01174e9d915365.
- Predictions recomputed: all 9 within 0.005 percentage points of table (c).
- Calibration: population view error 7.38% to 7.42%; wrong-indicator ICC 0.0997 to 0.1001 (rho 0.1), 0.1993 to 0.2007
  (rho 0.2), 0.5994 to 0.6023 (rho 0.6).
- Boundary diagnostic: in every setting the start is the strictest grid point (j_s = 29); the first point above alpha
  in walk order is j_b = 13 at alpha 1% (R_b 1.079% to 1.132%) and j_b = 8 at alpha 2% (R_b 2.230% to 2.315%).
  **Boundary subset: all 18 of 18 settings.**
| pair | rho | m | R_b at 1% | R_b at 2% |
|---|---|---|---|---|
| 0 | 0.1 | 5 | 1.096% | 2.236% |
| 1 | 0.1 | 10 | 1.112% | 2.271% |
| 2 | 0.1 | 15 | 1.101% | 2.252% |
| 3 | 0.2 | 5 | 1.090% | 2.237% |
| 4 | 0.2 | 10 | 1.091% | 2.247% |
| 5 | 0.2 | 15 | 1.113% | 2.277% |
| 6 | 0.6 | 5 | 1.106% | 2.252% |
| 7 | 0.6 | 10 | 1.079% | 2.230% |
| 8 | 0.6 | 15 | 1.132% | 2.315% |
- Why the subset is everything: the B1 accept fraction and true risk per frame are properties of single frames, which
  have the same marginal distribution for every rho and m in this model, so the B1 curves are almost the same in all
  9 pairs. Within-sign correlation changes how variable the Cert count is, not the curve. The summary over the subset
  in (e) therefore equals the summary over all 18; both are reported as Part A says.

### Part B: results (filled from the printout after review)
- Date: 2026-11-15 / Stage: R9 Stage 10 (declared addition) / Run tag: runs/20261003_stage1/poc3/stage10
- Git commits: Part A a8d142f (definitions and table (c)); population-curve script a6ed1b3; boundary subset and curve
  file fece72c; runner and tests 43df8eb. One run, on 43df8eb, tracked files clean (only the new outputs untracked).
  Run command `py -3.12 -m src.r9_stage10`; 124.7 s; full suite before the run 560 passed.
- Code SHA-256: `src/r9_stage10.py` 3cd142a7f6435c8ee435ee5ba34b9c2f3382c37b8150650a1fe53e23018353ee,
  `src/r9_stage10_popcurves.py` 797aae0712ef69f2cbd1eb62968b8162c178423547d7ef5eb675fafa311bc72d, `src/r9_certify.py`
  d2fd6fd63a0a65b22c73b08722bc1aa5583b51619b5720f99f6224b4fef98075 (frozen; `check_frozen()` passed first).
- Inputs (SHA-256 checked before reading): `population_curves_stage10.json` ffa785ea...119f; `stage2_summary.json`
  8c152579...729f; `stage3b_summary.json` 7591a438...7367; `stage6_summary.json` 71be7957...185f. Predictions (c)
  recomputed within 0.005 pp. Curves recomputed from the seeds and equal to the curve file in all 9 pairs.
- Outputs: `stage10_summary.json` SHA-256 4f7ce88317ce251ef6e0e2aad305cb3d5b46999b04a8927b307b2edbed7d5a15 (git blob
  f342e8b3c82958fe3ac1730ac09c3d851ec48195); `stage10_printout.txt` SHA-256
  8212b93d2df1baa7f7ee8eb8f707b6a79210db7fb8b92a0131abce7c837c901b.
- Certified thresholds with no committed unit, all runs, both units, all rules: 0.

#### Prospective result (frame-level B1, exact binomial; boundary subset = all 18 settings, fixed in step 2)
| pair | rho | m | alpha | DE | predicted delta_eff | observed [95% CP] | gap (pp) |
|---|---|---|---|---|---|---|---|
| 0 | 0.1 | 5 | 1% | 1.40 | 13.94% | 4.50 [3.63, 5.50]% | -9.44 |
| 0 | 0.1 | 5 | 2% | 1.40 | 13.94% | 2.40 [1.77, 3.17]% | -11.54 |
| 1 | 0.1 | 10 | 1% | 1.90 | 17.63% | 3.45 [2.69, 4.35]% | -14.18 |
| 1 | 0.1 | 10 | 2% | 1.90 | 17.63% | 1.40 [0.93, 2.02]% | -16.23 |
| 2 | 0.1 | 15 | 1% | 2.40 | 20.41% | 4.10 [3.27, 5.06]% | -16.31 |
| 2 | 0.1 | 15 | 2% | 2.40 | 20.41% | 1.25 [0.81, 1.84]% | -19.16 |
| 3 | 0.2 | 5 | 1% | 1.80 | 16.97% | 6.20 [5.18, 7.35]% | -10.77 |
| 3 | 0.2 | 5 | 2% | 1.80 | 16.97% | 3.85 [3.05, 4.79]% | -13.12 |
| 4 | 0.2 | 10 | 1% | 2.80 | 22.19% | 6.90 [5.83, 8.10]% | -15.29 |
| 4 | 0.2 | 10 | 2% | 2.80 | 22.19% | 3.55 [2.78, 4.46]% | -18.64 |
| 5 | 0.2 | 15 | 1% | 3.80 | 25.55% | 6.15 [5.14, 7.29]% | -19.40 |
| 5 | 0.2 | 15 | 2% | 3.80 | 25.55% | 3.50 [2.74, 4.40]% | -22.05 |
| 6 | 0.6 | 5 | 1% | 3.40 | 24.35% | 15.35 [13.80, 17.01]% | -9.00 |
| 6 | 0.6 | 5 | 2% | 3.40 | 24.35% | 11.80 [10.42, 13.30]% | -12.55 |
| 7 | 0.6 | 10 | 1% | 6.40 | 30.62% | 20.60 [18.85, 22.44]% | -10.02 |
| 7 | 0.6 | 10 | 2% | 6.40 | 30.62% | 14.65 [13.13, 16.28]% | -15.97 |
| 8 | 0.6 | 15 | 1% | 9.40 | 33.80% | 20.35 [18.61, 22.18]% | -13.45 |
| 8 | 0.6 | 15 | 2% | 9.40 | 33.80% | 13.40 [11.94, 14.97]% | -20.40 |
- Summary, all 18 settings and boundary subset (identical, both 18): 18 of 18 within the bound by the Part A rule
  (CP lower <= delta_eff); the observed point was also below delta_eff in all 18. Mean absolute gap 14.86 pp, mean
  signed gap -14.86 pp; gaps from -9.00 to -22.05 pp.
- In every run the B1 binomial start was grid point 29, the start used in the diagnostic (h).
- B1 with HB (primary start): 0.10% to 12.45% (largest pair 8, 1%).
- Sign level, exact binomial: 0.00% in all 6 rho 0.1 settings; 0.00% to 5.30% [4.36, 6.37]% at rho 0.2 (largest pair
  3, 2%); 2.55% to 5.25% at rho 0.6. No sign-level CP upper bound above 0.12, so nothing at odds with Stage 2 V2. Sign
  level with HB (primary start): at most 1.10%. Certification rates and the other two HB starts are in the printout.

#### Retrospective (the formula was written after these results; not a test of it)
- Stage 2, all 36 settings x 2 tests (72 rows): 72 of 72 within the bound by the Part A rule. The observed point is at
  or below delta_eff in 71 of 72; the one exceedance is level 1, rho 0.45, fixed-30, alpha 2%, exact binomial:
  37.45% [35.32, 39.61]% against 36.62%. That is the Stage 2 frame-level worst case, which was known when the formula
  was written.
- GTSRB POC-2 Stage 3b (HB, DE 13.9, predicted 36.55%): 17.00% at 0.5%, 20.00% [18.27, 21.82]% at 1%, 0.00% at 2, 3
  and 5%. 5 of 5 within the bound.
- BelgiumTSC Stage 6 (DE 2.24, predicted 19.59% [12.58, 21.94]%): exact binomial 14.00% [12.51, 15.60]% at 0.5% and
  8.60% at 1%; HB 4.00% and 3.15%; 0.00% at 2, 3 and 5% for both (pool view error 1.45% below alpha). 10 of 10 within
  the bound.

#### Reading of the result
- The formula behaved as a conservative ceiling, not as a predictor. It was not exceeded beyond the CP interval in
  any of the 18 prospective settings, all of them in the boundary subset, but it overstated the observed rate by
  14.86 pp on average and by at least 9.00 pp in every setting.
- Observed rates fall with alpha, which the formula does not contain: in all 9 prospective pairs the rate at 2% is
  below the rate at 1%, by a factor of up to about 3.3 (pair 2: 4.10% at 1%, 1.25% at 2%). This does not hold in
  every retrospective setting (for example Stage 2 level 1 rises from 1% to 2%).
- The observed rate depends on more than DE. At rho 0.1 it does not rise with m (alpha 2%: 2.40%, 1.40%, 1.25% for
  m = 5, 10, 15, against predictions 13.94%, 17.63%, 20.41%; alpha 1%: 4.50%, 3.45%, 4.10%). A likely cause, not
  tested here, is that the frame count (784 x m) and the start point on the grid also shape the rate.
- Even with independent frames (DE = 1) the formula overstates: it gives 10%, while the Stage 2 rho 0 rows observed
  0.00% to 8.25% (exact binomial). The walk seldom certifies a threshold whose true risk is only just above alpha.
- Statement for the record: under the frozen engine and this simulator, delta_eff = Phi(Phi^-1(delta) / sqrt(DE)) is
  a ceiling on the frame-level violation rate that held prospectively; it is not a prediction of that rate.

- Checks: none scored (the outcome is reported, Part A (e)). check_frozen PASS; input hash checks PASS (4); curve
  recomputation PASS; prediction recomputation PASS.
- Decision taken and why: report as above. Nothing in Part A (a) to (h) was changed after the run.
- Deviations: the formula was evaluated with scipy while Part A was drafted (no data, no simulator), accepted by the
  author before the commit of Part A. The boundary diagnostic placed all 18 settings in the subset, so the subset
  summary equals the full summary (recorded before the run in (h) values).
- Next step: Stage 9 (idea text and gates), only when the author says so.
