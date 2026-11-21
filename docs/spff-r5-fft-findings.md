# SPFF-R5 — two-threshold joint generating-function feasibility test

Date 2026-11-19. Standalone exploratory calculation. No repository or frozen POC files modified.

## Research setting and mathematical method

Independent sign clusters have fixed nested acceptance counts (loose 3, strict 1), frame count 3, and a six-point feasible joint error signature `(W_loose,W_strict)` in `(0,0),(1,0),(1,1),(2,0),(2,1),(3,1)`. The two populations A and B share **identical full per-threshold marginal error distributions**, but different dependence. Their generating polynomials are `P_q(u,v) = sum_h q_h u^(W_loose,h) v^(W_strict,h)`. For `n` independent clusters, the coefficient of `u^a v^b` in `P_q(u,v)^n` equals the probability of observing aggregate errors `(a,b)`. Zero-padding to dimensions at least `(3n+1,n+1)` makes FFT-powered polynomial convolution non-aliased in exact arithmetic. A fixed-sequence binomial selection at strict then loose is then a sum of coefficients inside the rectangular joint pass region, with no cross-threshold independence assumption.

The implementation uses `scipy.fft.rfft2`, exponentiates its coefficients to `n`, applies `irfft2`, clamps negative roundoff values to zero and renormalizes. **This is numerically approximate** and does not provide certified floating-point remainder bounds; the observed negative mass is a diagnostic only. Although it scales to `n=784` for *this fixed-acceptance, two-threshold model*, it is not a general 30-threshold algorithm or a pilot-honest forecasting method.

## Results

`alpha=.30, delta=.10`. Each row reports `(Pr(no cert), Pr(select loose unsafe), Pr(select strict safe))`.

| n | Population | No certificate | Loose unsafe | Strict safe | Runtime (s) |
|---|---|---:|---:|---:|---:|
| 7 | A | .9090487984 | .0205194763 | .0704317253 | .0015 |
| 7 | B | .9090487984 | .0651763910 | .0257748106 | .0002 |
| 20 | A | .9566947363 | .0128282468 | .0304770169 | .0006 |
| 20 | B | .9566947363 | .0368382482 | .0064670155 | .0003 |
| 120 | A | .8573862612 | .0306923239 | .1119214149 | .0077 |
| 120 | B | .8573862612 | .0792424353 | .0633713035 | .0055 |
| 784 | A | .7563083850 | .0353407100 | .2083509050 | .2082 |
| 784 | B | .7563083850 | .0745665699 | .1691250451 | .1566 |

The n=7 outputs agree with independent exact enumeration to machine precision. A second independent 350,000-draw multinomial simulation of n=784 clusters gave standardized discrepancies below 1.5 standard errors across all six outcome probabilities. Estimated negative FFT roundoff mass at n=784 was ~5.1e-13 (A) and ~6.8e-13 (B).

## Implication and limitations

This **removes brute-force composition-count scaling as an obstacle for this highly restricted fixed-acceptance, two-threshold reference**. The difference in unsafe selection at n=784 remains 3.9226 percentage points despite identical threshold-wise marginal distributions, illustrating why the full joint path matters. It does not demonstrate a novel theorem: FFT/polynomial convolution and dependence counterexamples are established mathematical techniques. More important unresolved issues are (i) finite-pilot uncertainty when m=261<n=784, (ii) variable acceptance, nonconstant cluster sizes, and ~30 thresholds, (iii) certified numerical errors and reliable tail calculations, and (iv) validation on real data under a preregistered protocol.

## Reproduction

Run `python spff_r5_fft_path_experiment.py` and `python spff_r5_fft_verify.py` in the same folder with NumPy and SciPy. The verification script uses a seeded independent Monte Carlo simulation. No file writes are performed.
