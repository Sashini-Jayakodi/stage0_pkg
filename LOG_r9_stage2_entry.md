## POC-3 (R9) Stage 2: certification engine v2, V-checks V1 to V4

### Part A: fixed definitions (committed before any V-check code exists or any Cert simulation is run)

Plan: `docs/poc3-implementation-plan-r9.md` (SHA-256 c0639973c70f35ac5cda1f11c49bf50a1639c632c48ccec8889d60a5ded03db4),
sections 3, 5.1, 5.3, 6 and 9 Stage 2. Stage 2 uses synthetic data only. No GTSRB or BelgiumTSC file is opened, so no
Cert or Test number can exist in this stage. delta = 0.1 throughout. The population-curve script (commit 6043706) and
its design values (note in (f)) come before this commit; no other Stage 2 code exists yet.

#### (a) Engine: `src/r9_certify.py` (frozen after V1 to V4 pass, see (j))
- Input per threshold lambda_j (j = 0..G-1, ordered from the loosest, j = 0, to the strictest): n_c(j) = committed
  Cert units, k(j) = committed and wrong Cert units, n = all Cert units.
- Primary test (exact binomial): p_j = P(Binomial(n_c(j), alpha) <= k(j)) = scipy.stats.binom.cdf(k, n_c, alpha).
  H0: P(wrong | committed) >= alpha. If n_c(j) = 0, p_j = 1 by definition (explicit guard, not left to scipy).
- Comparator (Hoeffding-Bentkus, as POC-2): loss per unit L_i = wrong_i - alpha * committed_i + alpha, mean
  mu = alpha * (1 - n_c/n) + k/n, p_j = `src.certify.hb_pvalue(mu, n, alpha)` (imported, unchanged). If n_c(j) = 0,
  p_j = 1 (explicit guard; the formula gives the same).
- A threshold is certified when p_j <= delta (the same convention as POC-2, `src.r8_stage3_engine.walk_rows`).
- Power-aware start (plan 5.3), per alpha: from the design commit fractions f_d(j), start = the strictest j with
  f_d(j) * n * alpha >= 4. If no j qualifies, nothing is tested and nothing is certified.
- Fixed-sequence walk: test start, start-1, ..., 0 at level delta each, stop at the first p_j > delta. Output = the
  last threshold that passed, or "none". The same order is used for both tests.
- The start depends only on design fractions, n and alpha, never on Cert outcomes; a unit test checks this.

#### (b) Evidence and error model (POC-2 Stage 3a, `src/r8_stage3_synth.py`, imported, unchanged)
- x_it = a_i + e_it, a_i ~ N(m, r), e_it ~ N(0, 1 - r); a view or frame is wrong iff x_it < 0.
- Main level (level 0): P_FRAME = 0.074 and m = -Phi^-1(0.074), as `r8_stage3_synth.P_FRAME` and `.M`; r solved so
  the within-unit correlation of the wrong indicator equals rho, rho in {0, 0.3, 0.45}.
- Added level (level 1, declared addition, see (f)): P_FRAME = 0.05, m = -Phi^-1(0.05), r solved the same way.
- `src/r9_stage2_popcurves.py` holds the model with the frame error as an argument (m_of, icc_of_wrong, solve_r, draw).
  Tests check that at P_FRAME 0.074 these equal the r8 functions, and that for T = 30 the draw equals
  `r8_stage3_synth.sample` for the same generator state.
- Population calibration (view error, wrong-indicator ICC, mean views) is printed per (level, rho, cluster).

#### (c) Commit model (POC-2 Stage 3a commit rule, extended to T_i views)
- S_it = x_i1 + ... + x_it. The unit commits at the first t <= T_i with |S_it| >= lambda and is wrong iff S_it < 0
  there; if no t <= T_i passes, it escalates. Computed with `r8_stage3_synth.track_table` (views past T_i are zero, so
  S and its running maximum stay flat).
- Why this model: a strongly negative a_i (a hard sign) gives a large |S| with the wrong sign, and a near-zero a_i gives
  a small |S|, so confidence falls with difficulty without a new parameter. Commit score C_i = max_{t<=T_i} |S_it|.
- Sign-level grid (30 thresholds per (level, rho, cluster), the same for both alphas): from a pilot sample of 20,000
  units (stream 0), lambda_j = the (1 - f_j) quantile of C_i with f_j = 1.00 - 0.70 * j / 29, j = 0..29, and
  lambda_0 = 0, so pilot commit fractions run from 100% to 30%.
- B1 grid (frame level, (h)): from the same pilot sample, 30 thresholds on the per-view margin |x_it| over valid
  views, same rule (accept fractions 100% to 30%, lambda_0 = 0).

#### (d) Settings (36 = 2 levels x 3 rho x 2 alpha x 3 cluster settings)
| Cluster setting | Views per unit T_i | Cert units n | Design units per run |
|---|---|---|---|
| fixed-3 | 3 | 470 | 157 |
| view-mix | 1 / 2 / 3 / 4 with probabilities 14 / 8 / 816 / 10 out of 848, drawn per unit | 470 | 157 |
| fixed-30 | 30 | 784 | 261 |
- View-mix source: Stage 1 testing audit, views per sign unit (1 / 2 / 3 / 4+) = 14 / 8 / 816 / 10. These counts are
  before the F-A2 removal (848 units) and count one view per GT row, so they include the 14 duplicated rows
  (`LOG_r9_stage1_entry.md` Part B). 4+ is treated as 4.
- n = 470 is about 0.6 x 783 (the Stage 1 projection by sign units); 784 is the GTSRB Cert size. Design 157 is about
  0.2 x 783; 261 is the GTSRB design set.
- Power-aware start floor f_d >= 4 / (n * alpha): 85.1% (470, 1%), 42.6% (470, 2%), 51.0% (784, 1%), 25.5% (784, 2%).
- fixed-3 and fixed-30 are the plan's V2 settings. view-mix and level 1 are declared additions, counted in V2 like the
  others.

#### (e) Runs, population and seeds
- Seeds: numpy `default_rng([20261114, 9, level, pair, stream])`; pair = index of (rho, cluster) in the order rho 0,
  0.3, 0.45 x fixed-3, view-mix, fixed-30 (pair 0..8). Streams: 0 pilot grid, 1 population, 2 runs.
- Population: 1,000,000 units per (level, pair) (stream 1), drawn once. Sign-level true risk per threshold
  R(j) = (committed and wrong) / committed. B1 true risk per threshold = (accepted and wrong views) / accepted views.
  The runner recomputes the sign-level curves and checks them against the committed curve files (hash-checked first,
  hard rule 2).
- Runs: 2,000 per (level, pair) (stream 2). Each run draws a fresh design sample, then a fresh Cert sample. The same
  draws serve alpha 1% and 2%, both tests and B1 (alpha changes only the start and the test).
- Violation: a threshold was certified and its true risk R > alpha. Certifying nothing is not a violation. This equals
  the POC-2 definition (`src.r8_stage3_engine.violations` on mu_true = alpha * (1 - c) + w), because
  E[L] > alpha <=> P(wrong) > alpha * P(committed) <=> R > alpha; the code uses that function.
- Violation rate = violating runs / 2,000, with a two-sided 95% Clopper-Pearson interval
  (`src.r8_stage3_engine.cp_bounds`, through `rate_row`).

#### (f) Non-vacuity of V2 and the added error level
- Rule: in every setting the start-eligible grid points (population commit fraction >= 4 / (n * alpha)) should include
  at least one threshold with R > alpha and at least one with R < alpha.
- Design values (synthetic, no Cert run): population curves from `src/r9_stage2_popcurves.py` at P_FRAME 7.4%
  (level 0). Step 2 found no start-eligible threshold with R < alpha in 4 of 18 settings (rho 0.3 and 0.45, fixed-3
  and view-mix, alpha 1%); these stay in V2 as null settings, reported as they come out. Added level (declared
  addition, not a replacement): P_FRAME 5% (level 1) on all 18 settings, so V2 counts 36. Chosen as the largest of the
  candidates 5, 4, 3, 2% (levels 1 to 4) at which all 18 settings have eligible R above and below alpha; the
  candidates were run before this rule was written down, and all five curve files are kept. Seeds:
  [20261114, 9, level, pair, stream].
- Curve files (`runs/20261003_stage1/poc3/stage2/`, script SHA-256
  16ced3d98ef12e4ded20357f240f7ec07014908b52e4564b4168ef855530203d, commit 6043706; the script was first run before
  that commit, see Part B):
  | Level | P_FRAME | File | SHA-256 | Settings non-vacuous |
  |---|---|---|---|---|
  | 0 (main) | 7.4% | population_curves_L0.json | fd7fdb44acbf6b0809258969996a5a919e185a503d40a222236a3a9d3881b019 | 14 / 18 |
  | 1 (added) | 5% | population_curves_L1.json | 5cd5c3aa553f48fe9bd82e2b29066f29a59f9e0591d0143af35af0ee17986260 | 18 / 18 |
  | 2 (candidate) | 4% | population_curves_L2.json | f70734e1bab14a444dad7f3676d2655d8c38ad4ab349f92b181207b6ea038194 | 18 / 18 |
  | 3 (candidate) | 3% | population_curves_L3.json | e42ad42dfefe29773464d0fb520aaa6f066ab20640f62d4e9e03596ce9121d53 | 18 / 18 |
  | 4 (candidate) | 2% | population_curves_L4.json | ab761712d6ff9d1a66ed59bc8e33c526669eccb845bf12aea7e21292b84313f3 | 14 / 18 |
- Stated now: where only j = 0 (R about 7.4% or 5%) lies above alpha (all rho 0 settings, and several others), a
  violation needs tens of committed errors, so close to 0 violations is expected there. Those settings meet the rule
  above but do not stress the boundary.
- Part B reports the certification rate of each test in each setting. A rate under 50% is reported as it comes out.

#### (g) V-checks (plan Stage 2; must pass; a failure is a bug)
- V1: plan section 6 reproduced exactly. For n_c in {784, 700, 560, 470, 420, 300, 250} and alpha in
  {0.5, 1, 2, 3, 5}%: the largest k with p <= 0.1 ("none" if p > 0.1 at k = 0); and the smallest n_c with p <= 0.1 at
  k = 0: 460, 230, 114, 76, 45. All 40 values must equal the plan's.
- V2: in all 36 settings of (d), the sign-level exact-binomial violation rate has an upper 95% Clopper-Pearson bound
  <= 0.12.
- V3: exact-binomial p-values from the engine match exact rational arithmetic (Python `fractions`, no scipy) to
  1e-9 relative error, on these cases (n_c, k, alpha):
  (230, 0, 1%), (229, 0, 1%), (114, 0, 2%), (113, 0, 2%), (45, 0, 5%), (44, 0, 5%), (460, 0, 0.5%), (459, 0, 0.5%),
  (784, 3, 1%), (784, 4, 1%), (300, 2, 2%), (300, 3, 2%), (1, 1, 1%).
  The first eight sit on both sides of the zero-error limits; (114, 0, 2%) gives p about 0.09995 and tests the
  <= delta convention.
- V4: a threshold with n_c = 0 is never certified, by either test. Checked (i) directly on p for several n and alpha,
  (ii) inside a walk where the strictest thresholds commit nothing and the looser ones would pass, and (iii) in every
  V2 run (count of certified thresholds with n_c = 0 in the Cert sample must be 0, both tests, both units).

#### (h) Reported, not scored
- Frame-level (B1) violation rate in the same simulations. Acceptance as POC-2 Stage 3a: a view or frame is accepted
  iff |x_it| >= threshold and wrong iff x_it < 0; every view or frame is its own unit
  (`src.r8_stage3_synth.frame_counts` logic, over valid views only). B1 is certified with the R9 engine: exact binomial
  on accepted frames (n_c = accepted Cert frames, k = accepted and wrong) and Hoeffding-Bentkus with n = Cert frames;
  power-aware start from the design accept fractions with n = Cert frames in the run. Only the unit differs from the
  sign-level certificate. Violation as in (e), with the B1 true risk.
- Certification rate of exact binomial against Hoeffding-Bentkus per setting and unit (share of runs certifying
  anything, the median certified grid index), runs with no eligible start, and the mean true risk of the certified
  threshold over certifying runs, as POC-2.

#### (i) Code, tests and outputs
- `src/r9_certify.py`: engine (a), no file I/O.
- `src/r9_stage2_vchecks.py`: V1 to V4 runner and the V2 simulation (b) to (h). Imports `src.r9_stage2_popcurves`,
  `src.r8_stage3_synth`, `src.r8_stage3_engine` and `src.certify`; no r8 file is edited.
- `tests/test_r9_certify.py`, `tests/test_r9_stage2_vchecks.py`: brute-force references (p by explicit sum; walk and
  start by Python loops), V4 cases, a test that the start ignores Cert outcomes, a small V2 run, tamper tests (a changed
  curve file is refused; the runner refuses to write its summary if `src/r9_certify.py` changed during the run).
- Design-phase guard (hard rule 3): a test asserts that neither module imports torch, an embedding or cache loader, or
  `src.r8_stage2`, and that no path under `data/` or `runs/btsc` appears in them.
- Output: `runs/20261003_stage1/poc3/stage2/stage2_summary.json` (settings, seeds, grids, population calibration, rates,
  V1 to V4 results, SHA-256 of the code and of the curve files read). `README_r9_stage2.md` gives the run command.

#### (j) Freeze
- When V1 to V4 pass and the author has reviewed the printout: commit; record the SHA-256 (and git blob ID) of
  `src/r9_certify.py` and `tests/test_r9_certify.py` in `LOG.md`; push (approved). Neither file is edited after that.
- If a V-check fails: it is a bug. The fix is made before the freeze, the failed run stays in Part B, and V1 to V4 are
  re-run in full.

### Amendment 1 (14 Nov 2026, after the first synthetic V-check run, before the freeze)
Made after the first V-check run (code commit 2f18f95; V1 to V4 all PASS; summary kept as
`stage2_summary_run1.json`, SHA-256 daea75fed027b6aa7b4856c6f0f3d1cfeb2d3937b14a6a0e889b4eb0191ae9fe) and before the
engine freeze. Synthetic data only; no Cert or Test number of any dataset exists.

- Reason: with the binomial power-aware start, Hoeffding-Bentkus fails its first test at strict thresholds, because its
  loss carries alpha x (1 - commit fraction) for every uncommitted sign. The exact-vs-HB comparison therefore measured
  the start rule, not the test. Evidence from the first run (level 0, rho 0, fixed-3): sign_hb certified in 87.4% of
  runs at alpha 1% (start near 85% commit) and in 0.0% at alpha 2% (start near 43% commit), while the exact binomial
  certified in 99.5% and 100.0%. The same pattern held for B1_hb in all fixed-3 and view-mix settings (0.0 to 0.2%).
- New HB start: the strictest grid point j at which HB would certify with zero committed errors given the design
  commit fraction, that is hb_pvalue(alpha * (1 - f_d(j)), n, alpha) <= delta (`src.r9_certify.hb_start`). Computed
  only from design fractions, n and alpha, never from Cert outcomes. Used for sign_hb and B1_hb. The walk from that
  start is unchanged (relax one grid point at a time, stop at the first p > delta).
- This replaces "The same order is used for both tests" in (a). The binomial start, the binomial test, the grids,
  seeds, settings, V1 to V4 and everything else are unchanged.
- Runner change that follows from it: "runs with no eligible start" is counted per test.
- Tests added: the HB start against a loop; the HB start ignores Cert outcomes; a hand case (n = 470, alpha 2%, design
  fractions 1, 0.9, 0.7, 0.5, 0.3; exact zero-error HB p-values 7.5e-5, 0.00171, 0.0395, 0.231, 0.618 by
  `r8_stage3_engine.exact_hb`, so the HB start is index 2 while the binomial start is index 3).
- V1 to V4 are re-run in full after this amendment. The first run's printout and summary stay in Part B as recorded.

### Amendment 2 (14 Nov 2026, after the Amendment 1 re-run, before the freeze)
Made after the second V-check run (code commit 5716fc6; V1 to V4 all PASS; summary kept as
`stage2_summary_run2.json`, SHA-256 e809286c0fe7d4146da2153a8ff164c8d256130dbb69947a5bf771b328c182fe) and before the
engine freeze. Synthetic data only.

- Reason: the Amendment 1 start has no margin. A diagnostic on 500 synthetic runs (level 0, rho 0, fixed-3, alpha 1%;
  a separate script, not part of the record) showed that HB failed its first test at the Amendment 1 start in 184 of
  500 runs, mostly because the Cert commit fraction came out slightly below the design estimate; HB certified in 316 of
  500 runs from that start against 450 of 500 from the binomial start. In the second run sign_hb certified in about
  60% of runs in every rho 0 setting (e.g. level 0, rho 0, fixed-3, alpha 1%: 87.4% in the first run, 59.4% in the
  second).
- New primary HB start: the strictest grid point j at which HB would certify with zero committed errors at the lower
  one-sided 90% Clopper-Pearson bound of f_d(j), that is hb_pvalue(alpha * (1 - CP_lower(f_d(j))), n, alpha) <= delta,
  with CP_lower from n_design units at the sign level (157 or 261) and from the design frames at the frame level (B1)
  (`src.r9_certify.hb_start`, rule "hb"). Design data, n_design, n and alpha only, never Cert outcomes.
- The exact binomial start and test, the grids, seeds, settings and V1 to V4 are unchanged.
- Secondary analysis (here and in Stages 3 and 6): HB is reported under all three starts next to the exact binomial,
  so the comparison is shown to be robust or not to the start rule: "hb" (this amendment, primary), "hb_a1"
  (Amendment 1, zero margin) and "hb_bs" (the binomial power-aware start, as Part A was first written). The engine keeps
  all three rules.
- This is the last change to the comparator before the freeze. Whatever the re-run shows is reported as it comes out.
- Tests added: a hand case (n = 470, alpha 2%, n_design 157, design fractions 1, 0.9, 0.8, 0.7, 0.63, 0.45; each CP
  bound checked against its definition; exact zero-error HB p-values at the bounds give start index 3, against 4 for
  Amendment 1 and 5 for the binomial start), the start against a loop, the CP-margin start never stricter than the
  zero-margin one, and every HB start ignores Cert outcomes.
- V1 to V4 are re-run in full. The first and second runs' printouts and summaries stay in Part B as recorded.

### Part B: results (filled from the printout after the run)
- Re-run check of the design values (author's request, recorded either way): the population-curve script was first run
  before its commit 6043706 (process deviation from the working method). After that commit, all five levels were
  re-run with the same seeds into a separate folder from the committed script (SHA-256 16ced3d9...203d, unchanged).
  The file hashes did NOT match the recorded ones in any of the five files. Key-by-key comparison: the only differing
  field in every file is `seconds` (run time, e.g. L0 12.6 vs 14.1 s); grids, curves, calibration and step 2 results
  are identical. The recorded files are the ones committed and checked by the runner.
- Date: ____ / Stage: R9 Stage 2 / Run tag: runs/20261003_stage1/poc3/stage2
- Git commit of Part A: ____ | Git commit of the code: ____ (src/r9_certify.py SHA-256 ____;
  src/r9_stage2_vchecks.py SHA-256 ____). Run command: ____
- Output: stage2_summary.json SHA-256 ____
- V1: ____ (40 / 40 values)
- V2: ____ (settings with CP upper <= 0.12: ____ / 36; largest CP upper ____ in setting ____)
- V3: ____ (largest relative error ____)
- V4: ____
- Reported: sign-level exact binomial vs Hoeffding-Bentkus certification rates ____; B1 violation rates ____
- Decision taken and why: ____
- Deviations: ____
- Next step: ____
