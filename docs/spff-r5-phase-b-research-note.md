# SPFF-R5 Phase B — Dependent-frame selection paths and pilot uncertainty

Status: exploratory mathematical and synthetic findings, 19 November 2026. This note is **not** a modification to the frozen POC-3 protocol or evidence of a confirmed IEEE Access novelty score.

## 1. Research object

Let `X_i = ((C_ij,E_ij) : j=0,...,G-1)` be the joint count signature for independent physical sign/pole `i`. `C_ij` is the number of frames accepted under threshold `j`, `E_ij` the accepted erroneous frames, and `0 <= E_ij <= C_ij <= T_i`. Under nested frame acceptance, both counts obey the relevant cross-threshold nesting and extra-frame feasibility constraints. The correlation of outcomes *within* one physical sign is unrestricted; distinct sampling units are assumed iid. This assumption should be revisited for multiple signs from one location or collection session.

For a hypothetical calibration of `n` independently sampled clusters, aggregate `C_j=sum_i C_ij` and `E_j=sum_i E_ij`. Apply the **naive frame-level** binomial p-value `BinomCDF(E_j; C_j, alpha)` and the frozen strict-to-loose fixed-sequence walk (zero accepted frames cannot certify). Let `J` be the *final* selected threshold, or `-1` when none certifies. For an independent deployment population, let `c_j=E[C_ij]`, `w_j=E[E_ij]`, and `t=E[T_i]>0`. Define

`e_j(q) = [w_j-alpha*c_j]_+ / t`,

`S_frame(q) = sum_{j>=0} Pr_q(J=j) * e_j(q)`,

`F_frame(q) = sum_{j>=0} Pr_q(J=j) * 1{c_j>0 and w_j>alpha*c_j}`.

Here S is expected *excess erroneous-accepted frames per population frame*, weighted by selection. F is the chance of selecting a threshold whose *population frame-conditional error risk* exceeds alpha. They are different; neither guarantees safety for the naive frame-level certificate.

If cluster size is not fixed, `t=E[T_i]` makes this a frame-weighted population estimand. Replacing it by a mean of per-cluster ratios would change the target. All path-policy definitions (threshold grid, order, start, alpha, delta) must be fixed independently of the independent pilot used for inference, or the adaptive procedure must be included explicitly in the inference target.

## 2. Exact prefix decomposition

Let `B_j` be the event that the binomial lower-tail p-value at threshold `j` is at most delta, and let `H_j=Pr(B_s & B_{s-1} & ... & B_j)` for fixed start `s`. Then

- `Pr(J=-1)=1-H_s`
- `Pr(J=s)=H_s-H_{s-1}` for `s>0`
- `Pr(J=j)=H_j-H_{j-1}` for `0<j<s`
- `Pr(J=0)=H_0`

with the evident two-case simplification when `s=0`. All these are exact identities, even for nonmonotone error paths and arbitrary cross-threshold dependence. They are existing probability identities, not a claimed new theorem. Prefix probabilities must be evaluated under a *joint cluster-level path law*, not by multiplying threshold-wise passing probabilities.

## 3. Constructive insufficiency of threshold-wise marginals (verified)

Consider two thresholds: loose `j=0` accepts 3 frames per cluster and strict `j=1` accepts 1 nested frame. A cluster signature is `(W0,W1)` because `(C0,C1)=(3,1)` is fixed. All listed signatures satisfy physically feasible nested errors: `0<=W1<=1`, `W1<=W0`, and `W0-W1<=2`.

| `(W0,W1)` | Population A | Population B |
|---|---:|---:|
| `(0,0)` | 0.44 | 0.44 |
| `(1,0)` | 0.02 | 0.26 |
| `(1,1)` | 0.24 | 0 |
| `(2,0)` | 0.25 | 0.01 |
| `(2,1)` | 0 | 0.24 |
| `(3,1)` | 0.05 | 0.05 |

Both populations have **identical full marginal distributions** at every threshold:

- Loose: `Pr(W0=0,1,2,3)=(.44,.26,.25,.05)`.
- Strict: `Pr(W1=0,1)=(.71,.29)`.
- Population risks: `r0=.91/3=.303333... > alpha=.30` (unsafe) and `r1=.29 < alpha=.30` (safe).

Take `n=7` clusters, `alpha=.30`, `delta=.10`, fixed order 1 then 0. The strict binomial test passes only with `sum W1=0` and the loose test passes only with `sum W0<=3`. An unsafe final certificate therefore occurs precisely on the intersection. Direct enumeration over `5^7` cluster sequences agrees with the following polynomial calculation:

`F = p0^7 + 7 p0^6 (r1+r2) + 21 p0^5 r1^2 + 42 p0^5 r1 r2 + 35 p0^4 r1^3`,

where `p0=Pr(W0=0,W1=0)`, `r1=Pr(W0=1,W1=0)` and `r2=Pr(W0=2,W1=0)`. Thus

- Population A: `F=0.02051947628544` (**2.052%**).
- Population B: `F=0.06517639096320` (**6.518%**).
- Both have identical threshold-wise marginal probabilities, risks, variances, and any ICC summary based only on within-threshold counts. The full-path failure probabilities differ by **4.466 percentage points**.
- With `T=3`, frame-normalized excess at the unsafe loose threshold equals `(.91-.30*3)/3 = 1/300`. So `S_A=0.0000683982542848`, `S_B=0.0002172546365440`.

**Inference:** No method that only uses the separate threshold-wise marginal laws can identify the full selected-path false-certificate probability in general. The joint cross-threshold cluster law is essential. This is a task-specific constructive example, not a claim that the general phenomenon of dependence/nonidentifiability is unprecedented.

## 4. Pilot-honest U-statistic interval: valid, but limited to m >= n

For a pilot `X_1,...,X_m`, with `m>=n`, let `h_j(X_{i1},...,X_{in})=1{J=j}`. The complete U-statistic `U_j` averages this bounded symmetric kernel over all size-`n` pilot subsets. It is unbiased for `pi_j=Pr(J=j)` without knowing a complete discrete signature alphabet. Standard bounded-kernel Hoeffding U-statistic concentration yields, jointly for all `G+1` final-selection categories, the radius

`eps_pi = sqrt( log(2(G+1)/eta_pi) / (2 floor(m/n)) )`.

For the **binary sign-level** reference (`0<=A,W<=1`), `d_j=W_j-alpha*A_j` has range of length 1. Its pilot empirical mean `dhat_j` obeys simultaneous Hoeffding bounds with radius

`eps_d = sqrt( log(2G/eta_d) / (2m) )`.

With `eta_pi+eta_d=eta`, optimize `sum_j pi_j [d_j]_+` over independent confidence intervals for the `pi_j` (subject to `pi>=0`, `sum pi=1`) and for the `d_j`. Two linear programs give a coverage-`1-eta` outer interval. When using the correctly specified independent-sign binomial LTT, it may additionally intersect the interval with the established `S <= (1-alpha)delta` structural cap. For general **frame-count clusters**, the numerator has range depending on maximal cluster size and the `E[T]` normalization also needs an independent confidence interval; the binary-sign proof must **not** be transplanted unchanged.

The U-statistic concentration is standard prior mathematical theory, not claimed as a new inequality. The result is a valid application to the exact path-selection functional when `n<=m` and required assumptions hold, but it does not solve historical `m=261, n=784` experiments.

### Independent prototype results

Synthetic binary-sign reference: 3 signature patterns, 3 thresholds, calibration n=8, alpha=delta=.30, true severity `0.001771875`, nominal interval coverage .95, and 100 independent pilots per sample size. A reproducible implementation and seeded experiments are in `spff_r5_phase_b_prototype.py`.

| Pilot `m` | R5 global-TV mean width | U-stat/LP mean width | Empirical U-stat coverage |
|---:|---:|---:|---:|
| 200 | 1.000000 | 0.081439 | 100/100 |
| 1000 | 0.662375 | 0.026940 | 100/100 |
| 3000 | 0.383100 | 0.013107 | 100/100 |

This independently reproducible, deliberately conservative LP variant gives slightly wider bounds than an earlier exploratory candidate (.07334, .02402, .01183), but still markedly improves on original TV intervals. A fair benchmark must also include the trivial structural-cap baseline `[0,(1-alpha)delta]=[0,.21]`, along with stronger nontrivial multinomial-functional intervals and matched-runtime comparisons. Coverage over 100 repetitions is a diagnostic, not a substitute for the proven confidence inequality.

## 5. Exploratory Gaussian full-path approximation: not pilot-honest

For cluster-count vector `X_i` with finite second moments, the multivariate CLT suggests approximating the full calibration sum by `N(n mu, n Sigma)`. Exact binomial tests are evaluated on rounded, physically clipped Gaussian total counts; selection is computed on the entire joint simulated path, not independently by threshold. This is a **heuristic finite-n implementation** of established Gaussian approximation, not a proof of finite-sample error or coverage, and clipping/rounding can introduce bias.

Six synthetic distributions were built from six variable-acceptance cluster-count patterns with 3 thresholds; `n=120`, `alpha=.12` or `.16`, delta=.1. The population-moment Gaussian path approximation, compared with 40,000-draw direct multinomial oracle per setting, had a maximum absolute selection-category difference between **0.00065 and 0.00530**. These gaps include Monte Carlo sampling error, not a certified approximation error.

However, estimating moments from only 261 independent pilot clusters (30 pilots per setting, 3,000 Gaussian draws each) produced much greater variation. The *mean maximum category error per pilot* ranged from approximately **0.028 to 0.188**, and the worst single-pilot errors ranged up to approximately **0.609** in the six settings. Some distributions were near selection boundaries. A few supplementary `n=784, m=261` probes likewise showed a large pilot-uncertainty problem. This is negative evidence against using plug-in Gaussian forecasting as if it were a pilot-honest confidence procedure.

## 6. Distinguish what is and is not established

**Established under stated assumptions:** exact prefix decomposition; the physically feasible counterexample; bounded-kernel U-statistic/mean confidence construction for sign-level severity and `m>=n`; reproducibility of the small synthetic reference; severe pilot uncertainty in exploratory Gaussian forecasts.

**Not established:** a new finite-sample interval with useful width for `n>m` under arbitrary cluster dependence; a strong error bound for multivariate Gaussian selection-path approximation; scalability to 30 thresholds or realistic variable-size sign clusters; real R5 traffic-sign results; independent empirical confirmation; IEEE Access novelty >=7.

The map `q -> F(q)` is discontinuous when a threshold population risk crosses alpha. Uniformly precise inference for F near a risk boundary can therefore fail even when S is continuous. The exact target, boundary behavior and required power/resolution conditions must be stated rather than expecting a uniformly narrow F interval.

## 7. Related work and originality risks

- Angelopoulos et al., *Learn then Test: Calibrating Predictive Algorithms to Achieve Risk Control*: https://arxiv.org/abs/2110.01052 — fixed-sequence LTT risk control is established.
- *Valid Per-Field Selective Risk Control for Document Extraction* (2026): https://arxiv.org/abs/2608.14639 — clustered selective-risk validation is established in a different application.
- Gan et al., *Certified Selective Automation of LLM Agent Evaluation* (2026): https://arxiv.org/abs/2609.34320 — cluster-aware bootstrap certificates and a coverage forecast are close antecedents.
- *Fixed-Sequence Confidence-Interval Calibration for Finite-Sample Selective Risk Control* (2026): https://www.mdpi.com/2571-905x/9/5/109 — CI-based fixed-sequence risk control and nonmonotone threshold analysis exist.
- Malloy et al., *Optimal Confidence Regions for the Multinomial Parameter*: https://arxiv.org/abs/2002.01044 — strong generic confidence-region baseline.
- Hoeffding U-statistic concentration is established: see e.g. https://economics.mit.edu/sites/default/files/2025-03/Theory_Transfer_0.pdf.

No claim of novel U-statistic inequalities or Gaussian simulation is warranted. The strongest defensible prospective contribution is **selection-path-identifiability analysis plus a practical pilot-honest method for correlated frame-count paths beyond what threshold-wise margins allow**, ideally with a useful `n>m` regime and rigorous numerical errors. That last part remains unsolved.

## 8. Gate assessment (exploratory method, not manuscript)

**Step 3 estimated idea publishability:** **60–65% — REVISE**, mainly because a practically relevant valid inference method for `n>m` and real-data evidence remain missing. This estimate is a judgment, not a measured publication probability.

**Step 4 current conceptual scores:** IEEE Access suitability **8.5/10**; novelty **6.5/10**; feasibility **6.5/10**; methodological soundness **7.0/10** (under clear scope). **FAIL** under the stated mandatory novelty threshold. The constructive path-dependence result improves the research argument, but standard U-statistic and Gaussian approximations do not yet justify novelty above 7.

## 9. Recommended next implementation, contingent on review

1. Preserve the local Phase A/stabilization work; avoid modifying frozen or historical files. Build independent `src/spff_r5_cluster_path.py` and `tests/test_spff_r5_cluster_path.py` for **exact cluster-count reference**: validation including nested physical count constraints for policy models where required, stable R9 p-values, fixed start including -1, full joint path, F, S, variable cluster sizes, and state guard. Use independent brute force for n<=4, with the Section 3 counterexample as a mandatory regression.
2. Add separate `src/spff_r5_phase_b_validation.py` for the bounded-kernel U-stat reference **only for m>=n**. Compare to original R5 TV, structural-cap baseline, and a general multinomial-region procedure; do not claim a new confidence theorem. Record m<n as unsupported rather than silently switching to a non-honest approximation.
3. Treat Gaussian full-path simulation as a descriptive comparator. If continued, add pilot uncertainty, stability diagnostics, Monte Carlo standard errors, and evidence near/discontinuous risk boundaries. Do not label it pilot-honest.
4. Do not run new GTSRB/BelgiumTSC R5 analyses or re-open the POC-3 freeze until a chosen method, baseline suite, and source integrity have been independently verified. Historical full-suite CRLF/hash failures remain reported.

**Decision:** Proceed to the exact joint cluster-count reference implementation and its independent oracle; defer any claim that a solution for the historical pilot/calibration regime or Gate 4 novelty PASS exists.
