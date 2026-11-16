## r11: forecast of the frame-level false-certificate rate from a pilot upper bound (exploratory, synthetic)

### Part A: fixed definitions (committed before any r11 code exists or any r11 number is computed)

Plan: `docs/poc3-implementation-plan-r9.md` (SHA-256 c0639973c70f35ac5cda1f11c49bf50a1639c632c48ccec8889d60a5ded03db4).
Plan R9 has no such analysis; this is a declared exploratory addition to RQ1, following Stage 10. It changes no frozen
file (`src/r9_certify.py`, `tests/test_r9_certify.py`, plan R9, `configs/frozen_r9_btsc.yaml`) and no earlier output.
Synthetic data only; no GTSRB or BelgiumTSC data file is opened. delta = 0.1 throughout.

#### (a) Question and history
- Question: how closely does a forecast built only from a pilot (design) sample predict the frame-level (B1)
  false-certificate rate, compared with the design-effect heuristic delta_eff = Phi(Phi^-1(delta) / sqrt(DE))?
- History, stated before any run: a first version of the forecast (`src/r10_prelim_micro.py`, Part 3, GTSRB) chose the
  risky threshold from the pool's true risk and gave 47.3% against an observed 20.0% at alpha 1% (16.6% against 17.0%
  at 0.5%). The upper-bound rule in (b) was written after that result. The Stage 10 observed rates (18 settings) are
  also known. Both are therefore development data; only the held-out settings in (c) are out-of-sample.
- The held-out set is judged by the criteria in (e), fixed now. Everything else is reported, not judged. No result here
  changes any frozen value or earlier conclusion.

#### (b) The forecast (fixed now)
For one run i of one setting (rho, m, alpha) and one test t in {binom, hb}:
1. Pilot = that run's design sample: 261 signs x m frames, drawn as in Stage 10 (`simulate`). On the B1 grid of the
   setting, per threshold j: A_j = accepted pilot frames, W_j = accepted-and-wrong pilot frames.
2. U_j = one-sided upper 80% Clopper-Pearson bound of W_j / A_j: beta.ppf(0.80, W_j + 1, A_j - W_j); U_j = 1 when
   W_j = A_j or A_j = 0.
3. Walk order = the frozen engine's order for test t in that run: from `start_for(t, A/N_d, N_c, alpha, N_d)`
   (N_d = 261 m pilot frames, N_c = 784 m Cert frames; binom: power-aware start, hb: primary HB start, Amendment 2)
   down to index 0. j* = the first j in that order with U_j > alpha. If the start is -1 or no j qualifies, the run's
   forecast is 0 (the pilot predicts no risky threshold).
4. Inputs at j*: f = A_j* / (261 m); risk set to U_j*; per-frame accepted-wrong rate p = U_j* x f;
   rho_I = one-way ANOVA ICC of the accepted-and-wrong frame indicator over the 261 pilot signs at j* (groups of m
   frames; the `icc_anova` formula of r10 Part 3, copied into the r11 file, not imported). If W_j* = 0, or
   rho_I <= 0, the binomial model (rho_I = 0) is used, as in r10.
5. Model (the r10 Part 3 model, unchanged): every Cert sign has N_g = round(m x f) accepted frames; its accepted-wrong
   count is K_g ~ BetaBinomial(m, p, rho_I), clipped at N_g; 784 independent signs; total K = sum K_g;
   committed frames n_c = 784 N_g.
6. Run forecast = P(test t certifies at j* | model) = P(K <= k_max), with k_max the largest k at which t's p-value at
   j* is <= delta (binom: `C.binom_pvalue(k, n_c, alpha)`; hb: `C.hb_pvalue_counts(k, n_c, 784 m, alpha)`, both by
   search over k). The distribution of K is computed exactly (clipped beta-binomial pmf, convolved 784 times), not
   sampled, so the forecast has no seed. This is a calculation change from r10 (which sampled 20,000 sets), not a
   model change; a test checks it against sampling.
7. Setting forecast = mean of the run forecasts over the 2,000 runs; the median and the 10th-90th percentile range
   over runs are reported beside it.
- Limitation, j* only: a walk that certifies at j or at any looser threshold must first pass j, so for any fixed j,
  P(walk passes j) <= P(p_j <= delta). Hence, when j* lies at or before the true first risky threshold j_true (first
  index in walk order with population B1 risk > alpha), P(false certificate) <= P(p_j* <= delta): the j*-only quantity
  is an upper bound on the false-certificate probability. This bound holds for the true distribution of p_j*. The
  forecast evaluates P(p_j* <= delta) under the model in step 5, and j* comes from the pilot and can fall after
  j_true, so the forecast is an upper bound only where j* is at or before j_true and the model holds; neither is
  guaranteed. The share of runs with j* at or before j_true is reported per setting.
- Limitation, small m: clipping at N_g bends the model at small m (m = 5: N_g <= 5); the clipped probability mass is
  reported per setting.
- Heuristic: delta_eff = Phi(Phi^-1(0.1) / sqrt(DE)), DE = 1 + (m - 1) rho with the setting's population rho and m, as
  Stage 10. The same value applies to both tests and both alphas.

#### (c) Data (fixed now)
Draw count: 2,000 runs per setting (Stage 10 `RUNS` = 2,000; above the 200 floor). Cert 784 / design 261 signs,
P_FRAME = 0.074, model, grids, B1 definition and violation definition exactly as Stage 10 Part A (b). In every set, rho
is the simulator's population ICC of the wrong indicator over all frames (Stage 2 (b), `solve_r`); rho_I in (b) is a
different quantity, measured on the pilot.

| Set | Settings | Grid and population (streams 0, 1) | Runs (stream 2) | Role |
|---|---|---|---|---|
| Development | the 18 Stage 10 settings (rho 0.1/0.2/0.6 x m 5/10/15 x alpha 1/2%) | committed `population_curves_stage10.json` (ffa785ea...119f) | `[20261115, 10, pair, 2]`, regenerated; B1_binom and B1_hb violation counts must equal `stage10_summary.json` (4f7ce883...5a15) or the run stops | development (seen), reported |
| Replication | the same 18 settings | the same committed curves | `[20261118, 11, pair, 2]`, pair 0..8 as Stage 10 | new seeds, same process; reported with the (e) metrics, not judged |
| Held-out | rho 0.3/0.5 x m 8/12 x alpha 1/2% (8 settings, 4 pairs) | built here: `[20261118, 11, 100 + pair, 0]` pilot grid (20,000 signs), `[20261118, 11, 100 + pair, 1]` population (1,000,000 signs) | `[20261118, 11, 100 + pair, 2]` | out-of-sample; judged by (e) |

- Held-out pair index: 0 = (0.3, 8), 1 = (0.3, 12), 2 = (0.5, 8), 3 = (0.5, 12).
- Neither 0.3 nor 0.5 is a rho of the 18 development settings (0.1, 0.2, 0.6), so no value was shifted. rho = 0.3 was
  a Stage 2 setting (m 3, view-mix, 30); no m of 8 or 12 was used there, and the forecast was not compared with Stage 2.
- Seed base 20261118 has not been used by any earlier script (r9 used 20261114 and 20261115, r10 used 20261116 and
  20261117).
- Held-out heuristic values (fixed now; scipy.stats.norm; the runner recomputes them and stops if any differs by more
  than 0.005 percentage points):

| pair | rho | m | DE | heuristic delta_eff |
|---|---|---|---|---|
| 0 | 0.3 | 8 | 3.10 | 23.33% |
| 1 | 0.3 | 12 | 4.30 | 26.83% |
| 2 | 0.5 | 8 | 4.50 | 27.29% |
| 3 | 0.5 | 12 | 6.50 | 30.76% |

- The held-out boundary diagnostic (Stage 10 Part A (h), R_b and boundary flag) is reported for context.

#### (d) Outcome (reported for every set)
Per setting, per test (binom, hb), per set: observed B1 false-certificate rate with two-sided 95% Clopper-Pearson
interval (`src.r8_stage3_engine.rate_row`); UCB forecast (mean, median, 10-90% range over runs); heuristic; whether
each prediction lies inside the observed CP interval; signed gap (prediction - observed) in percentage points; share
of runs with forecast 0; share of runs with j* at or before j_true; clipped mass. Summary per set and test: the three
(e) quantities, count inside the CP interval, mean signed gap, for the forecast and the heuristic.

#### (e) Held-out pass criteria (fixed now, verbatim)
Exact binomial is primary and is the only target judged; HB is reported, not judged. The replication set is reported
with the same metrics but is not judged. Over the 8 held-out settings, with "observed" the exact-binomial B1
false-certificate rate and "forecast" the setting forecast of (b) 7:
- (a) MAE of the exact-binomial forecast vs observed <= 3 points
- (b) forecast closer to observed than the heuristic in >= 7 of 8 settings
- (c) forecast >= observed - 2 points in >= 7 of 8 settings
"Points" are percentage points. "Closer" means |forecast - observed| < |heuristic - observed| (a tie counts as not
closer). The forecast passes the held-out check when (a), (b) and (c) all hold; each is reported separately, and a
failed criterion is reported as failed.

#### (f) Checks (all must hold, else the run stops)
- `check_frozen()` first; both Stage 10 files hash-checked before reading.
- Development regeneration reproduces the Stage 10 B1_binom and B1_hb violation counts exactly.
- Held-out heuristic values recomputed and checked against the table in (c).
- Tests (`tests/test_r11_forecast_ucb.py`, synthetic): CP bound against a brute-force beta quantile; ICC against a
  brute-force ANOVA; exact convolution against 200,000 sampled Cert sets (within Monte Carlo error) and against direct
  enumeration for tiny n; k_max search against a scan of the p-value; j* walk against a brute-force loop; tamper tests
  (changed Stage 10 files are refused).

#### (g) Order of work
1. This Part A and `README_r11.md` committed before any r11 code exists.
2. Code and tests; full suite; commit.
3. One run; the whole printout kept; then stop. Part B only after the printout has been reviewed.

#### (h) Deviations from the project rules, declared
- File names `src/r11_forecast_ucb.py`, `tests/test_r11_forecast_ucb.py`, `LOG_r11_entry.md`, `README_r11.md` and
  output folder `runs/20261003_stage1/r11/` are outside the POC-3 `r9_` naming rule, by the author's decision.
