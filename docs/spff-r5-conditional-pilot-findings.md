# SPFF-R5 — Conditional finite-pilot full-path inference: rigorous restricted result

Date: 2026-11-19. Standalone mathematical experiment. No repository changes. **Scope: fixed-acceptance, two-threshold, six-support family with KNOWN and FIXED threshold-wise marginal error distributions. Not the historical 30-threshold setting.**

## Model

A physical-sign cluster yields nested wrong-frame counts `(L,S)` on loose (three accepted frames) and strict (one accepted frame). Feasible signatures are `(0,0),(1,0),(1,1),(2,0),(2,1),(3,1)`. Each of `n` calibration clusters is iid; frames within a sign may be dependent. The fixed frozen-R9-equivalent testing path visits strict then loose, using pooled-frame binomial lower tails with alpha=.30, delta=.10; strict passes if `S_total <= b`, loose passes if `L_total <= a`. The *final* selection is loose exactly when **both tests pass**, while strict alone is selected when strict passes but loose fails.

Consider the predeclared family, `t in [0,1]`:

`q(t)=(.44, .02+.24t, .24(1-t), .25-.24t, .24t, .05)`.

Both FULL per-threshold error-count marginal distributions are constant throughout this family; only their joint dependence changes. In particular, `q_2+q_4=.24`, and conditional on a pilot observation being in category 2 or 4, the chance it is category 4 is `t`. The family assumption is *not established* by observing that pilot marginals look similar. It must come from independent, justified design information or be explicitly treated as a restrictive thought experiment.

## Selection boundary identity and monotonicity

Let `a,b` be the respective pooled-count passing cutoffs at calibration size `n`. Define `F_n(t)=Pr_{q(t)^n}(L_total<=a,S_total<=b)`. With `Z=(L_Z,S_Z)` the totals from `n-1` iid clusters, the polynomial derivative obeys

`F'_n(t)=.24 n Pr_{q(t)^(n-1)}(L_Z=a-1, S_Z=b) >= 0`.

**Proof:** Differentiating one factor in the iid product measure yields `.24 n` times the expectation of the four-point switch contrast `f(Z+(1,0))+f(Z+(2,1))-f(Z+(1,1))-f(Z+(2,0))` for `f(l,s)=1{l<=a,s<=b}`. The contrast equals exactly `1{L_Z=a-1,S_Z=b}`. This is a special case of the classical supermodular / positive-dependence switch order; the novel aspect, if any, must be justified relative to prior selection procedures and inference literature. It is **not** a new general probability inequality.

`Pr(no certificate)` and `Pr(strict passes)` remain constant along the family because strict marginal counts do not change; strict-only selection decreases as loose selection increases. The loose threshold is population unsafe throughout because its frame-level risk is `.91/3=.303333... > alpha=.30`; the strict threshold is population safe because its risk is `.29 < alpha=.30`. Thus `F_n(t)` is exactly false-certification probability for this model, without a discontinuity along this fixed-risk family. Frame-normalized positive excess severity is `S_n(t)=F_n(t)*(.91-.90)/3=F_n(t)/300`.

## Exact finite-pilot inference, including m<n

Given `m` iid pilot clusters sampled from **the assumed family**, count `K` category-4 observations and `N` category-2-or-4 observations. Conditional on `N`, `K ~ Binomial(N,t)`. Construct a two-sided exact 95% Clopper-Pearson confidence interval `[t_L,t_U]` from `(K,N)` (return `[0,1]` if `N=0`). Its unconditional coverage is >=95% for every t and every pilot size m. By monotonicity, `[F_n(t_L),F_n(t_U)]` covers `F_n(t)` with >=95% probability, for any predeclared calibration size n, including `n>m`. The interval for severity follows by dividing by 300. This is an exact statistical-coverage assertion at the mathematical probability level; the current FFT endpoint implementation does **not** have a certified numerical rounding enclosure, so practical endpoints should be outward rounded or computed with validated arithmetic before calling the numerical implementation fully certified.

This conditional construction is a direct transformation of the classical Clopper-Pearson interval and a standard positive-dependence ordering property. **Do not claim this alone constitutes sufficient IEEE Access novelty.** If threshold-wise marginals are estimated rather than known, this 1D interval generally loses its guarantee, unless a joint confidence region also accounts for their uncertainty.

## Reproducible numerical results

With alpha=.30, delta=.10, `n=784`: pooled passing cutoffs `a=676` (loose), `b=218` (strict). FFT results (approximately 1e-13 absolute diagnostics, no rigorous error guarantee):

| t | False-certificate F | Analytic F' | Severity F/300 |
|---:|---:|---:|---:|
|0.00|.03534070998|.03298624517|.00011780237|
|0.25|.04402576433|.03650799854|.00014675255|
|0.50|.05359840726|.04005749991|.00017866136|
|0.75|.06400354686|.04289429210|.00021334516|
|1.00|.07456656991|.03903029846|.00024855523|

**Pilot simulation:** `m=261`, 15 independent seeded pilot replications for each t (45 total), exact conditional 95% intervals. The simulation is a computational smoke test of *a separately proven* coverage statement, not a statistical demonstration of 95% coverage from 45 runs.

| t | Covered/15 | Mean F interval width | Max F width | Mean width in t |
|---:|---:|---:|---:|---:|
|0.00|15/15|.00190285|.00224936|.0569948|
|0.50|14/15|.01014510|.01150451|.255246|
|1.00|15/15|.00242606|.00307817|.0598891|

All resulting F intervals are much narrower than the vacuous generic total-variation bound `[0,1]`, but that comparison is **not apples-to-apples**: the conditional interval assumes essentially all nuisance marginals are known, while the generic approach need not. The interval widths must not be presented as a practical improvement on the original historical pilot without justifying the extra assumptions.

Independent validation script checked: fixed complete marginal distributions for 11 values of t; analytic derivative against centered finite differences at 12 settings (`n=7,120,784`); and Clopper-Pearson exact conditional coverage at 126 (N,t) combinations (`N=1,2,5,15,63,261`). All checks passed. The numerical derivative discrepancies are below 1e-7 absolute in tested cases.

## Critical limitations and next research problem

1. **Model identification and generalizability:** The 1D family assumes known full per-threshold marginal count laws. Actual data have unknown marginal distributions, variable per-cluster accepted frame counts, varying track lengths, and approximately 30 thresholds; the current theorem does not handle them.
2. **Numerical certification:** FFT clamps negative coefficients, normalizes, and may put tiny spurious mass in impossible cells. Statistical coverage of ideal F endpoints is exact; computed FFT endpoints are not certified enclosures.
3. **Novelty:** Conditional binomial intervals and supermodular ordering are classical. The path-specific boundary derivative is a useful exact characterization for this setup, but its novelty relative to recent cluster-audit methods and risk certification requires literature comparison.
4. **Next mathematical target:** Derive an **honest nuisance-aware region** covering unknown marginal distributions jointly with dependence, then compute efficiently certified full-path extrema. Benchmark *under the same information assumptions* against general multinomial confidence regions. A feasible theoretical direction is a multidimensional generalized switch structure or an adaptive sensitivity bound, but no guarantee for the historical pipeline has yet been proved.

## Files and use

`spff_r5_conditional_pilot_prototype.py`: exact family definition, FFT oracle, derivative, conditional CP interval, reproducible pilot simulations. `spff_r5_conditional_pilot_verify.py`: independent derivative, fixed-marginal, and exact CP coverage checks. `spff_r5_conditional_pilot_results.json`: seeded 45-run summary. Run with NumPy and SciPy; no network, GPU, frozen POC edits, or commits required.
