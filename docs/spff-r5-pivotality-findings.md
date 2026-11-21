# SPFF-R5 Phase C — Exact selection-pivotality audit (2026-11-19)

## Status and reproducibility

This is a separate local mathematical experiment and does not alter `spff-r5`, POC-1/2/3, or prior results. Execute `python spff_r5_pivotality_audit.py` (requires NumPy/SciPy). The six-signature joint cluster law and the fixed-sequence thresholds are copied from the independently verified Phase B nested-frame counterexample.

The cluster accepts three frames at the loose threshold, one at the strict threshold; `n` iid clusters are calibrated by naive pooled-frame binomial tests at `alpha=.30`, `delta=.10`, visiting strict then loose. Each cluster's wrong-count signature is `(W_loose,W_strict)` in `{(0,0),(1,0),(1,1),(2,0),(2,1),(3,1)}`. Population A weights `(.44,.02,.24,.25,0,.05)`; B weights `(.44,.26,0,.01,.24,.05)`.

For a random sample of `n−1` clusters `Z` and independently sampled replacement candidates `X,X'~q`, define the **average replacement pivotality** `rho_n(q)=Pr[J(Z,X) != J(Z,X')]`. Define the **possible-pivotal-context probability** `gamma_n(q)=Pr_Z[there exist feasible x,y such that J(Z,x)!=J(Z,y)]`. The script exactly enumerates the `(n−1)`-cluster multinomial compositions, evaluating the frozen-R9-equivalent `scipy.stats.binom.cdf` decisions. `gamma` is conservative because it includes any feasible pattern regardless of its probability in the chosen population; it is not a finite-pilot uncertainty certificate.

## Exact numerical outcomes

| n | population | average replacement `rho_n` | possible-pivotal context `gamma_n` |
|---|---|---:|---:|
| 7 | A | .05983209 | .12810028 |
| 7 | B | .06110692 | .12810028 |
| 12 | A | .04886198 | .11069497 |
| 12 | B | .05289147 | .11703510 |
| 20 | A | .02032550 | .04609916 |
| 20 | B | .01961894 | .04561998 |

## Exact general coupling inequality, and why it is insufficient

Let `p,q` be arbitrary distributions on a *common fixed cluster-signature support*, let `q_t=(1−t)q+tp`, and let `pi_j(r)=Pr_{r^n}(J=j)` for the **same fixed** full-path selection function `J`. Let `gamma(r)=Pr_{Z~r^(n−1)}[there exist possible x,y in the common support such that J(Z,x)!=J(Z,y)]`. For any subset `B` of final-selection categories, differentiating the finite-degree polynomial `Pr_{q_t^n}(J in B)` and using total-variation duality gives

`|Pr_{p^n}(J in B)−Pr_{q^n}(J in B)| <= n TV(p,q) ∫_0^1 gamma(q_t) dt <= min(1,n TV(p,q) sup_t gamma(q_t)).`

Explanation: with `Z` fixed, `x -> 1{J(Z,x) in B}` is constant on non-pivotal contexts, so `E_p−E_q` vanishes there and has magnitude ≤ `TV(p,q)` on pivotal contexts. This remains valid for nonmonotone selection paths and correlated thresholds *within each independent cluster*. The proof describes a standard influence/coupling argument specialized to LTT, **not a demonstrated novel probability inequality**.

The bound is **not directly computable from a finite pilot** without confidence control of both `TV(p,q)` and `gamma` uniformly over the entire confidence region/path. For the A/B pair `TV(A,B)=.48` and even the `n=7` bound using `gamma=.1281` is approximately `.4304`, much larger than the true false-certificate probability difference `.0446569`. Therefore the test does not validate a useful forecasting bound. Averages `rho_n` cannot replace worst-case `gamma_n` without an additional proved distribution-comparison argument.

## Research decision

**Stop promoting pivotality alone as an IEEE novelty theorem.** It empirically confirms small *average* selection sensitivity in a small example but the immediately available bound is too loose. The remaining high-value candidate is a **pilot-honest, path-specific functional interval for `m<n`** using a low-dimensional, predeclared joint cluster summary (or a certified approximation and remainder), rigorously benchmarked against generic multinomial-region optimization, and with explicit near-boundary nonidentifiability caveats. This is a target for research, not a proven solution. The previously documented U-statistic (`m>=n`) and moment-Gaussian (`m<n`, not pilot-honest) methods remain baselines.

## Practical next coding experiment

Before spending GPU time, implement a low-dimensional **exact toy** where the joint distribution can be parameterized by six cluster categories. For fixed `n` and full selection map, construct simultaneous multinomial confidence regions from `m` independent pilot clusters; optimize the *entire selected-path severity functional* over those regions, with a certified global optimizer or exhaustive interval-subdivision enclosure. Compare against original R5 generic-TV interval, plug-in bootstrap (exploratory), and an independence/marginal-only model. Evaluate `m<n` including `m=261,n=784` only if the model's exact computation scales; otherwise label it an approximate experiment with a quantified remainder. Crucially, do not claim an optimized multinomial region is novel per se. Preserve negative findings and validate the counterexample.
