## Stage 6: certification engine (synthetic validation, then Cert/Test)

- **Date / stage / run tag:** 2026-10-15 / Stage 6 / 20261003_stage1
- **Git commit before the runs:** ____ (commit the Stage 6 code first; `git rev-parse HEAD`)
- **Code / tests:** `src/certify.py` SHA-256 ____ ; `src/certify_synth.py` ____ ; `src/stage6.py` ____ ; `py -3.12 -m pytest tests -q` gives 97 passed, 0 skipped (torch imported normally on this run, including the Stage 1 smoke test).
- **Rules applied as pre-registered:** delta = 0.1 and beta = 5% unchanged; delta_k = delta/K; G1(iii) = some absolute alpha in {1, 2, 3}% certifiable in both strata in >= 4 of 5 reshuffles, eps primary with eps3 side by side; fixed-sequence ordering = Fit argmin then looser (plan 6.3, the declared deviation from R6 wording); per-track-mean base error primary, pooled sensitivity (Addendum A). No frozen rule changed.

### Step 1: synthetic validation (before any real data)

- **Decision (2026-10-11), synthetic design changed once, before any real data was touched.** First design (correct-frame margin mean 4, Beta-binomial clustering only) was uninformative, not an engine bug: the engine passed null, hand values, floor, ordering and the population cross-check, but nothing was certifiable at n = 784 and alpha = 2% (the HB test needs a mean shifted loss of about 0.9%, i.e. an accepted risk of about 0.8% at 90% coverage), so the non-vacuity check was empty, and the naive certifier never broke (violation <= 2%) because accepting only confident frames thins the errors and dilutes their intra-track correlation (ICC 0.3 becomes about 0.07 among accepted errors) while the HB bound is conservative. Final design: correct-frame margin mean 20 and an added "cluster" row (0.5% of tracks are confusable: every frame wrong, margins as confident as correct frames). `src/certify.py` was not changed. Plan wording for check 5 ("naive breaks at rho_sim = 0.3") is therefore met on the cluster design, not on the Beta rows.
- **Output file (SHA-256):** synthetic_validation.json ____ ; `ALL PASS: True`

| check | result | pass |
|---|---|---|
| 1 null: worst P(p <= 0.05) over 36 cases | 0.0192 | PASS |
| 2 hand values: max relative error | 1.6e-14 | PASS |
| 3 floor at n = 392: closed form / engine bisection | 0.761% / 0.761% (n = 450: 0.664%) | PASS |
| 6 ordering: Fit-ordered / strict-to-loose certified share | 100% / 0% | PASS |
| population cross-check: max abs difference | 0.00029 | PASS |

| design (1,000 runs) | track-level certified | violations | rate | CP95 upper | naive certified | naive violations | naive rate | naive CP95 |
|---|---|---|---|---|---|---|---|---|
| beta rho_sim = 0 | 100.0% | 0 | 0.000 | 0.004 | 100.0% | 0 | 0.000 | [0.000, 0.004] |
| beta rho_sim = 0.1 | 100.0% | 0 | 0.000 | 0.004 | 100.0% | 2 | 0.002 | [0.000, 0.007] |
| beta rho_sim = 0.3 | 100.0% | 0 | 0.000 | 0.004 | 100.0% | 11 | 0.011 | [0.006, 0.020] |
| cluster | 63.4% | 0 | 0.000 | 0.004 | 99.8% | 87 | 0.087 | [0.070, 0.106] |

- **Checks 4 and 5:** track-level valid on every row and non-vacuous: PASS ; naive clearly broken on the cluster design (CP lower 0.070 > 0.05): PASS.
- **Reading:** 0 violations of 1,000 on the track-level rows means the test is conservative (valid, not tight). The naive certifier breaks only when confident errors arrive in whole tracks; with Beta-binomial clustering alone the HB bound absorbs it.

### Step 2: real data

- **Prediction written BEFORE running step 2:** ____ (fill honestly; if none was written before the run, say so here)
- **Output file (SHA-256):** stage6_results.json ____
- **Checks:** Stage 4 split, tau, c\*, margin grids and E[L] curves reproduced (script asserts): passed ; certificates internally consistent (script asserts): passed ; synthetic validation hash matched (script asserts): passed ; no Test label read before decisions (unit test): passed.

**What the test demands for the real strata** (median n; mu* = largest mean shifted loss that passes; max accepted risk R at coverage 100% / 90% / 80%):

| stratum | n | alpha 1% | alpha 2% | alpha 3% | alpha 5% |
|---|---|---|---|---|---|
| easy | 487 | mu* 0.127%; R 0.13% / 0.03% / - | 0.662%; 0.66% / 0.51% / 0.33% | 1.315%; 1.31% / 1.13% / 0.89% | 2.875%; 2.87% / 2.64% / 2.34% |
| hard | 456 | 0.107%; 0.11% / 0.01% / - | 0.626%; 0.63% / 0.47% / 0.28% | 1.316%; 1.32% / 1.13% / 0.89% | 2.851%; 2.85% / 2.61% / 2.31% |

**G1(iii)** (some absolute alpha in {1, 2, 3}% certifiable in both strata, >= 4 of 5): 
- eps (primary): **0 of 5 -> FAIL**. The easy stratum certifies at 3% in seeds 0, 1, 3; the hard stratum never certifies at 3% (or at 1%, 2%) in any seed.
- eps3: **2 of 5 -> FAIL** (seeds 2 and 3, at 3%).

**Certified group-seeds / total, over 5 reshuffles (Test violations among certified groups in brackets; Test risk = realised track-averaged selective risk, overall median)**

| scheme | 1% | 2% | 3% | 5% | 0.5 x base (per-track) | 0.5 x base (pooled) |
|---|---|---|---|---|---|---|
| eps2 | 0/10 | 0/10 | 3/10 (0) | 10/10 (0) | 7/10 (0) | 2/10 (0) |
| eps3 | 0/10 | 0/10 | 7/10 (0) | 10/10 (0) | 8/10 (0) | 3/10 (0) |
| marginal | 0/5 | 3/5 (0) | 5/5 (0) | 5/5 (0) | 5/5 (0) | 4/5 (0) |
| Stallkamp-6 | 0/30 | 0/30 | 0/30 | 8/30 (0) | n/a | n/a |
| Mondrian-43 | 0/215 | 0/215 | 0/215 | 0/215 | n/a | n/a |

Test coverage (median, overall) at alpha = 5%: eps2 85.0%, eps3 82.8%, marginal 88.2%, Stallkamp-6 39.1%, Mondrian-43 0%. At alpha = 3%: eps2 37.0%, eps3 39.4%, marginal 81.1%. Test risk overall at alpha = 5%: eps2 1.09%, eps3 0.74%, marginal 1.81%. Addendum A caveat: the per-track and pooled base-error definitions disagree strongly on certifiability at 0.5 x base (eps2 7/10 vs 2/10, eps3 8/10 vs 3/10); both are reported, per-track stays primary.

**eps2, per reshuffle, alpha = 3% and 5%** (n tracks easy / hard; certified threshold; Test coverage; Test risk)
- alpha = 3%: seed 0 easy 493 certified (cov 80.2%, risk 0.83%), hard 456 UNCERTIFIABLE; seed 1 easy 480 certified (86.9%, 0.77%), hard 468 UNCERTIFIABLE; seed 2 both UNCERTIFIABLE; seed 3 easy 487 certified (83.7%, 0.32%), hard 444 UNCERTIFIABLE; seed 4 both UNCERTIFIABLE.
- alpha = 5%: all ten strata certified. Test coverage 88.0% to 92.8%; Test risk easy 1.33% to 2.10%, hard 0.59% to 2.98%. No Test violation.

**Floor table** (zero errors, all accepted, 1 - delta_k^(1/n)): eps2 easy n = 480 to 493, floor 0.61% to 0.62%; hard n = 444 to 481, floor 0.62% to 0.67%; eps3 easy n = 447 to 457 (0.65% to 0.67%), hard n = 408 to 418 (0.71% to 0.73%); marginal n about 781, floor 0.29% to 0.30%; Stallkamp-6 median n = 117, floor median 3.44% (min 1.57%, max 17.71%); Mondrian-43 median n = 17, floor median 30.0% (min 5.83%, max 86.75%).

**Descriptive scan, smallest certifiable alpha per reshuffle (%, step 0.5; not a certificate):** eps2 easy 3.0, 3.0, 2.5, 2.5, 2.5 ; eps2 hard 3.5, 3.5, 3.5, 3.5, 4.5 ; eps3 easy 2.5, 3.5, 3.0, 3.0, 3.0 ; eps3 hard 4.0, 3.0, 3.0, 3.0, 2.5 ; marginal 2.0, 2.0, 2.0, 2.5, 2.5 ; Stallkamp-6: speed limits 4.5 to 6.0, other prohibitory 6.5 to 10.0, derestriction none, mandatory 10.0 in one seed else none, danger 9.0 to 10.0 in three seeds else none, unique 3.5 to 4.0 ; Mondrian-43: no group certifiable up to 10% in any reshuffle.

**H1 preview** (marginal threshold applied to the eps2 strata on Test, track bootstrap B = 2000; point estimates):

| alpha | marginal certified | hard risk > alpha | hard 95% lower > alpha | easy risk > alpha | hard points % | easy points % |
|---|---|---|---|---|---|---|
| 1% | 0 of 5 | - | - | - | - | - |
| 2% | 3 of 5 | 0 | 0 | 0 | 0.00, 0.63, 0.09 | 0.88, 0.90, 0.47 |
| 3% | 5 of 5 | 0 | 0 | 0 | 0.16, 0.73, 0.27, 1.27, 1.28 | 1.46, 1.57, 1.18, 1.11, 2.08 |
| 5% | 5 of 5 | 0 | 0 | 0 | 1.34, 2.02, 0.99, 2.57, 3.23 | 2.96, 2.94, 3.66, 2.83, 3.57 |

H1 (the marginal promise visibly breaks for the hard group) has **no support** on clean data: the hard-stratum Test risk never exceeds alpha, and it is below the easy-stratum risk in 12 of the 13 seed-alpha pairs where the marginal threshold certified (the exception is seed 3 at alpha = 3%). Caveats: the 5 reshuffles overlap in tracks, there are about 130 Test tracks per stratum, and no confidence interval for the hard-minus-easy difference was computed; the direction is a point-estimate observation, consistent with G1(i) failing in Stage 5.

**Checks**
- [x] tests pass (97)
- [x] synthetic validation ALL PASS on this `certify.py`
- [x] Stage 4 reproduced
- [x] certificates consistent
- [x] no Test label read before decisions

- **Decision taken and why:** G1(i) FAIL (Stage 5), G1(ii) PASS, G1(iii) FAIL (eps 0/5, eps3 2/5). Per plan section 9: clean GTSRB is reported as a **certifiability frontier** (minimum certifiable alpha per stratum against n, floors, what the test demands), not as a demonstration of stratified certification; scenarios lead the main claim with alpha = 0.5 x base error; delta is not loosened, no R7 is declared, and the pre-registered G1(iii) alphas are not extended to 3.5% after the fact (the descriptive scan shows the hard stratum needs about 3.5%, reported as a frontier number only). Ranking of certifiability on clean data, by coverage at a given alpha: marginal > eps2/eps3 > Stallkamp-6 > Mondrian-43. The eps stratification beats the other group-conditional baselines but costs certifiability relative to no stratification, and H1 is not supported on clean data. What remains defensible and measured: track-level validity versus naive frame-level breaking (synthetic cluster design; rho = 0.446 on the real data), the floor and the frontier. Whether stratification earns its cost is left to the scenario cells (Stage 7).
- **Deviation from R6?** Only the Fit-ordered sequence (plan 6.3). The synthetic design change above is a validation-design decision, not a change to any frozen rule.
- **Next step:** Stage 7 (G2 freeze, Fit data only, before any scenario result on Cert or Test) and Stage 8 (latency).
