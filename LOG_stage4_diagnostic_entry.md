## 2026-10-10 | Stage 4 add-on | Fit-only diagnostic (run tag 20261003_stage1)
- Code / commits: src/fitdiag.py ; tests: 48 passed, 1 warning (the known harmless train.py UserWarning). Addenda A and B and src/fitdiag.py were committed together in 8b512fc ("fitdiag", on branch bashee, after the pre-registration freeze 45a4d41). The order of that commit relative to the diagnostic run was not recorded; if the run came first, the Addenda were written in the working copy of LOG.md but not yet committed at run time. Either way the diagnostic is information only and changed no frozen rule.
- What was done: `py -3.12 -m src.fitdiag --run runs/20261003_stage1` (B = 2000 track-cluster bootstrap draws, Fit tracks resampled; output splits/fit_diagnostic.json). Verified the Stage 4 file hashes, recomputed splits and c* (matched). Fit frames only; no Cert or Test error computed.
- Status: information only (Addendum B). Nothing frozen was changed.

### A. Hard-minus-easy gap on Fit, percentage points (per-track mean = primary)

| variant | seed | gap | Fit SE | 95% CI | pooled gap |
|---|---|---|---|---|---|
| eps | 0 | +0.43 | 1.76 | [-3.27, +3.88] | +0.05 |
| eps | 1 | +0.61 | 1.85 | [-3.19, +4.11] | +1.44 |
| eps | 2 | +0.37 | 1.94 | [-3.38, +4.25] | -0.57 |
| eps | 3 | -0.35 | 1.66 | [-3.73, +3.00] | +0.58 |
| eps | 4 | +1.50 | 1.69 | [-1.81, +4.79] | +0.70 |
| eps3 | 0 | +1.28 | 2.26 | [-3.04, +5.56] | +0.15 |
| eps3 | 1 | +1.13 | 2.06 | [-2.89, +5.29] | +1.73 |
| eps3 | 2 | -0.52 | 2.27 | [-5.08, +3.89] | -0.37 |
| eps3 | 3 | +1.67 | 2.20 | [-2.40, +5.95] | +1.04 |
| eps3 | 4 | +1.85 | 2.16 | [-2.44, +6.07] | +1.17 |

- Means: eps gap +0.51 pp (Fit SE 1.78); eps3 gap +1.08 pp (Fit SE 2.19). Every Fit confidence interval contains 0.
- Rough read-across to Cert (Cert has 784 tracks vs 261, so SE about sqrt(261/784) = 0.58 times the Fit SE): Cert SE about 1.03 pp (eps) and 1.26 pp (eps3). G1(i) needs the 2.5th-percentile bound above 0, i.e. a gap of about 2.0 pp (eps) or 2.5 pp (eps3) just to clear it.
- Rough chance that G1(i) passes in a single reshuffle: 7.2% (eps), 13.5% (eps3). In >= 4 of 5 reshuffles: 0.0% (eps), 0.1% (eps3). These assume the true gap equals the Fit gap (itself noisy) and independent reshuffles (they are not). Order-of-magnitude only. Fit CIs do not rule out a true gap of 2-4 pp, so "unlikely", not "impossible".

### B. Error by confusability slice (8 slices of equal mass, mean over seeds)

eps (primary):

| slice | mass | per-track err | pooled err | share of Fit errors |
|---|---|---|---|---|
| 0 (lowest eps) | 11.8% | 5.82% | 3.11% | 7.8% |
| 1 | 13.3% | 9.18% | 5.65% | 15.4% |
| 2 | 12.6% | 11.91% | 4.86% | 12.5% |
| 3 | 12.1% | 7.77% | 4.63% | 11.6% |
| 4 | 11.2% | 9.84% | 4.01% | 9.4% |
| 5 | 13.2% | 12.83% | 5.57% | 15.5% |
| 6 | 14.8% | 12.33% | 5.94% | 18.2% |
| 7 (highest eps) | 10.9% | 10.03% | 4.13% | 9.6% |

eps3 (first fallback):

| slice | mass | per-track err | pooled err | share of Fit errors |
|---|---|---|---|---|
| 0 | 14.4% | 5.27% | 0.91% | 2.8% |
| 1 | 10.6% | 10.74% | 8.77% | 19.0% |
| 2 | 12.5% | 10.33% | 4.83% | 12.6% |
| 3 | 12.4% | 12.96% | 4.47% | 11.3% |
| 4 | 12.9% | 9.23% | 3.15% | 8.3% |
| 5 | 12.5% | 19.29% | 3.93% | 10.2% |
| 6 | 12.5% | 20.97% | 8.28% | 21.3% |
| 7 | 12.4% | 28.21% | 5.59% | 14.4% |

### What this says
- **eps does not separate error rates on Fit.** The pooled error is flat across all 8 slices (3.1-5.9%), and each slice holds about its fair share of errors. [Certain] for these numbers. The hard-minus-easy gap of about +0.5 pp is small against a Fit SE of 1.8 pp.
- **G1(i) is very likely to fail for eps, and probably for eps3** [Likely]. This is consistent with the Stage 3 warning that 94.4% of confusions sit inside the same Stallkamp subset: a pair-level rank does not add much at a median cut.
- **The eps3 per-track gradient (5% to 28% across slices) is mostly not real structure** [Likely]. The pooled error does not rise with eps3 (0.9% to 8.8%, no trend), while the per-track mean is 2 to 5 times higher in the upper slices (slice 7: 28.2% vs 5.6%). The per-track mean gives every track equal weight, including tracks that contribute only one or two frames to a slice (probably tracks whose predicted pair flips). Those fragments are noisy and error-prone, and they dominate the average. This also means my earlier remark that eps3 looks better on Fit is weaker than it sounded: the pooled eps3 gap is only +0.75 pp.
- **Cost of the committed definition:** the per-track mean has the larger SE and the more erratic slice profile. It is still the primary, because it matches the certificate's per-track loss. The pooled rate stays as the sensitivity (Addendum A applies if the two disagree).
- **G1(ii) and G1(iii) are not affected by this diagnostic.** G1(ii) should still pass easily; G1(iii) is decided by the Stage 6 certifier.
- Not tested yet [Guessing]: the fragment hypothesis. It can be checked descriptively (frames per track in a stratum, error by that count) without any Cert errors.

### Decision
- Proceed to Stage 5 exactly as pre-registered. No R7, no change to c*, strata, seeds or definitions. Reasons: the one visible gradient (eps3 per-track) looks like a fragment artifact, so re-tuning to it would chase noise; and Fit sets rotate across reshuffles, so a design tuned on Fit leaks into the Cert verdicts.
- Expected outcome, stated before Stage 5: G1(i) fails for eps (and probably eps3). Under the proposal's own G1 rule, a failure means clean GTSRB is reported as a certifiability frontier and the scenario cells become the primary evidence. That is a pre-registered outcome, not a deviation.
- Deviation from R6? No.
- Next step: Stage 5 (G1(i)/(ii) on Cert, rho and ICC, rank-of-error, eps3 ablation), plus one descriptive addition decided now: report the distribution of frames per track within each stratum (counts only, on Fit and on Cert) so the fragment effect is documented.
