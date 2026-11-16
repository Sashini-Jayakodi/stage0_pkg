## r13: pilot forecast with a bootstrap upper bound on rho_I (exploratory; simulated held-out and BelgiumTSC)

### Part A: fixed definitions (committed before any r13 code exists or any r13 number is computed)

Plan: `docs/poc3-implementation-plan-r9.md` (SHA-256 c0639973c70f35ac5cda1f11c49bf50a1639c632c48ccec8889d60a5ded03db4).
Exploratory addition after r11, r12 and r12b. It changes no frozen file (`src/r9_certify.py`, `tests/test_r9_certify.py`,
plan R9, `configs/frozen_r9_btsc.yaml`, `configs/frozen_r8.yaml`, plan R8), no POC-1 or POC-2 file and no earlier
output. delta = 0.1 throughout.

#### (a) History, stated before any r13 number
- The one change from r11 (the rho_I input) was chosen after the r11 held-out FAIL (MAE), the r12 GTSRB FAIL and the
  r12b post hoc diagnostic were seen. GTSRB is development data (set (c) is consistency only).
- BelgiumTSC: the forecast has never been run on it, but its observed view-level rates (exact binomial 14.00% at
  alpha 0.5% and 8.60% at 1%; Stage 6, `stage6_summary.json` 71be7957...185f) and its heuristic value (19.59%, Stage 10
  retrospective) were known when the criteria window in (d) was set. The window is the same fixed rule used for the
  simulated set (observed - 2 pp to observed + 5 pp); it was not tuned to the BelgiumTSC values. Set (b) is
  out-of-sample for the forecast's inputs, not blind to its targets.
- A pass is described as "conservative within 5 points", not "accurate". r11 judged accuracy (MAE <= 3 points) and
  failed; here MAE is reported for every set and not judged.

#### (b) The forecast: r11 Part A (b) and Amendment 1, with one change
Every step of r11 (b) 1-7 and Amendment 1 applies as written, except step 4's rho_I:
- j_hat = the r11 j* from the full pilot, held fixed. rho_I input = the one-sided 80% upper bootstrap bound of the
  pilot's rho_I at j_hat: 1,000 resamples of pilot clusters (signs, tracks or poles) with replacement; in each, rho_I by
  the r11 `icc_anova` at j_hat (real cluster sizes); resampled values <= 0 count as 0; a resample with no
  accepted-and-wrong frame at j_hat has no defined rho_I and counts as 0; the bound is the 80th percentile of the 1,000
  values (numpy default linear interpolation); a bound >= 1 uses the Amendment 1 limit.
- If the full pilot has zero errors at j_hat, the binomial model is used, as in r11 (every resample also has zero).
- j_hat and the bootstrap are computed separately for each test (the binomial and HB starts can differ), as r11's j*.
- Everything else (p = U_j_hat x f, N_g = round(m f), clipping at N_g, exact convolution, k_max per test) is unchanged.

#### (c) Sets and data
**(a) Simulated held-out.** rho in {0.25, 0.40} x m in {6, 20} x alpha in {0.5, 1, 2}% = 12 settings (4 pairs),
P_FRAME 7.4%, pilot 261 signs, Cert 784, 2,000 runs per setting; Stage 10 model, grid rule, B1 definition and
violation rule (r11 (c)); rho is the simulator's population wrong-indicator ICC.
- New values: m 6 and 20 never used (Stage 2: 3, view-mix, 30; Stage 10: 5, 10, 15; r11: 8, 12); rho 0.25 and 0.40
  never used in a committed setting (Stage 2: 0, 0.3, 0.45; Stage 10: 0.1, 0.2, 0.6; r11: 0.3, 0.5); rho 0.40 replaces
  the requested 0.45, which Stage 2 used. alpha 0.5% was never used in a simulated setting (Stage 2, Stage 10, r11: 1%
  and 2%).
- Pair index: 0 = (0.25, 6), 1 = (0.25, 20), 2 = (0.40, 6), 3 = (0.40, 20).
- Seeds: base 20261120 (unused by any earlier script): `default_rng([20261120, 13, pair, stream])`, stream 0 pilot grid
  (20,000 signs), 1 population (1,000,000 signs), 2 runs; bootstrap indices `default_rng([20261120, 13, pair, 3, run])`,
  one (1000, 261) array per run, shared by the three alphas and both tests.
- Heuristic (reported only; recomputed by the runner, stop if any differs by more than 0.005 pp):

| pair | rho | m | DE | heuristic delta_eff |
|---|---|---|---|---|
| 0 | 0.25 | 6 | 2.25 | 19.65% |
| 1 | 0.25 | 20 | 5.75 | 29.65% |
| 2 | 0.40 | 6 | 3.00 | 22.97% |
| 3 | 0.40 | 20 | 8.60 | 33.11% |

- Boundary diagnostic (Stage 10 Part A (h)) computed from the population curves before the runs and reported, so
  settings with no start-eligible risky threshold (observed rate near 0) are visible; no setting is dropped.

**(b) BelgiumTSC (out-of-sample for the forecast; see (a)).** Pilot = the 133 design poles; audit = the Stage 6 RQ1
draws (2,000 draws of 397 of the 530 pool poles, with replacement, `draw_counts(530, 2000, 397, (20261115, 6))`);
observed = the Stage 6 view-level B1 rates (exact binomial judged: 14.00% at 0.5%, 8.60% at 1%; HB reported: 4.00%,
3.15%). B1 on BelgiumTSC: each view is a unit, frozen 30-point grid (`frozen_r9_btsc.yaml` b1), frozen design
fractions f_d over 465 design views.
- Cluster = the pole (the audit draws whole poles).
- Views per pole vary (option 9A): m = round(m*), m* = sum(T^2) / sum(T) over the 133 pilot poles (T = views per pole);
  the model gives every Cert pole m views, Cert = 397 poles, N_c = 397 m; rho_I (point and bootstrap) by `icc_anova`
  with the real unequal pole sizes; f = accepted pilot views / 465. m* and m are computed by the runner from the pilot
  and printed. Known approximation, reported per draw: the model's 397 m views against the real view count of each
  draw (1,309 to 1,490 in Stage 6).
- Start in the forecast: `start_for(t, f_d, 397 m, alpha, 465)` with the frozen design fractions; the Stage 6 per-draw
  starts are reported beside it.
- Alphas 0.5% and 1% judged; 2% and 3% reported, not judged (observed 0.00% at both).
- Stop-check: the Stage 6 RQ1 B1 rows (violations and certified count, both tests, alphas 0.5, 1, 2, 3%) reproduced
  exactly from the hash-checked BelgiumTSC inputs, else the run stops. `check_frozen()` first.
- Heuristic (reported only): DE 2.24 -> 19.59%.

**(c) GTSRB (development data; consistency only; reported, not judged).** As r12: pilot = 261 design tracks, audit =
POC-2 Stage 3b draws, alpha 0.5, 1, 2%, both tests, with the r13 rule; plus the r12 1,000-resample pilot bootstrap
interval of the whole forecast (key [20261119, 12] as r12; the inner rho bootstrap uses `default_rng([20261120, 15,
outer])`, outer = 0 for the full pilot and 1..1000 for the outer resamples).

**(d) Descriptive pilot-size curve (GTSRB; no criteria).** Pilots of 100, 261, 400 and 600 tracks drawn from the
1,046 pool tracks without replacement, 500 pilots per size, `default_rng([20261120, 14, size, pilot])`; inner rho
bootstrap `default_rng([20261120, 16, size, pilot])`; r11 rule and r13 rule; alpha 0.5% and 1%, both tests; mean bias
(forecast - observed) and 10-90% range of the forecasts, against the observed Stage 3b rates from r12 (HB 17.00%,
20.00%; exact binomial 26.55%, 29.30%). The pilots come from the same pool the audit draws from, so they are not
independent of the audit population.

#### (d) Criteria (pre-registered; exact binomial primary; HB and heuristic reported only)
- (a) Simulated held-out (12 settings): in >= 10 of 12 settings, BOTH (i) the share of pilots with forecast >=
  observed - 2 pp is >= 80%, AND (ii) mean forecast - observed <= +5 pp.
- (b) BelgiumTSC (single pilot), at alpha 0.5% and at 1%: observed - 2 pp <= forecast <= observed + 5 pp. Both alphas
  must pass.
- Overall pass = (a) and (b). Every criterion and every setting reported separately. A pass is described as
  "conservative within 5 points", not "accurate". No change after any number is seen.
- Definitions: "observed" in (a) is the setting's exact-binomial B1 false-certificate rate over its 2,000 runs (one
  number per setting); "pilot" is each run's own pilot (2,000 per setting); "mean forecast" is the mean over those
  2,000 pilots. In (b), "forecast" is the single pilot's forecast and "observed" the Stage 6 rate. "pp" = percentage
  points; the bounds are inclusive as written.

#### (e) Reported beside the criteria (not judged)
For every set, setting and test: observed [95% CP]; mean, median and 10-90% of forecasts; MAE and mean signed gap; the
r11-rule forecast on the same pilots (so the effect of the one change is visible); heuristic; rho_I point and bootstrap
bound; case counts (Amendment 1 limit, zero errors, rho_I <= 0); boundary diagnostic (set (a)); view-count mismatch
(set (b)).

#### (f) Checks, tests, order
- Stop-checks: `check_frozen()`; input hashes; set (b) Stage 6 reproduction; set (c) r12 / Stage 3b reproduction;
  heuristic table recomputed.
- Tests (`tests/test_r13_*.py`, synthetic): bootstrap bound against a direct loop; the r13 forecast equals r11's when
  the bound is replaced by the point rho_I; m* against a hand value; criteria hand cases including the edges; tamper
  tests.
- Order: Part A committed before any code; code and tests; full suite; one run after the author approves the tests;
  whole printout; Part B after review.
- Deviation: `r13_` file names outside the POC-3 `r9_` rule, by the author's decision, as for r11 and r12.

#### Amendment 1 (16 Nov 2026, before any r13 run)
Set (b): the bootstrap of the 133 BelgiumTSC pilot poles uses `default_rng([20261120, 17]).integers(0, 133, (1000, 133))`,
one array shared by all alphas and both tests. Part A named no key for set (b); the base 20261120 is the Part A base,
and sub-key 17 is unused.
