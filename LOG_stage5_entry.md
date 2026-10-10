## 2026-10-11 | Stage 5 | G1(i), G1(ii), rho and rank-of-error on Cert (run tag 20261003_stage1)

- Config / code / checkpoint / git commit: constants = the `constants_used` block of stage5_results.json (plan and pre-registration values; no `stage5:` block added to poc.yaml) ; checkpoint 86527359bc61a5fbc4d48b7744e049b5ca3ff81cb016fb237298d0c007140bcc ; git commit: ____ (fill after committing the Stage 5 code: `git rev-parse HEAD`) ; stage5_results.json SHA-256: ____ (`Get-FileHash runs\20261003_stage1\splits\stage5_results.json`).
- Code change before this run (environment, not design): Windows Smart App Control blocked torch's unsigned DLLs (Code Integrity event 3077, policy VerifiedAndReputableDesktop, WinError 4551), which also stopped the numpy-only stages because they imported `sha256_file` from `src/train.py` (imports torch). Added `src/util.py` (same function, no torch) and changed ONE import line in `src/splits.py` and `src/fitdiag.py` (`src/stage5.py` is new). No logic changed; `train.py`, `head.py` and every Stage 1-4 output are untouched. Stage 5 asserts that the recomputed splits, tau and c* equal the saved Stage 4 files (they did). New SHA-256: splits.py 868a163af0994e8415de32235c03a76b9638c00366fab539be636f59c2d8aeaf (was 065e7047308a163c8baacd3a5030f766c6ac25d49aed75a1bcb0a7c348e3a1d7), fitdiag.py cb666b010f7aacf02633426765b24f64d7913bbf91f53ab45c8d47208b0c08f9, stage5.py 0aa94e8d26e5e24247557f824bf783a33463a0b676d31efc69c5b6fc45964212, util.py e4224c39386ee08b7d6db73fb27616a40e86c256ac51667ec7776d32f60b02da.
- Tests: `py -3.12 -m pytest tests -q` gives 48 passed, 2 skipped (the Stage 1 and Stage 3 test files skip themselves because torch cannot load on this machine; with torch the suite is 63 passed).
- What was done: `py -3.12 -m src.stage5 --run runs/20261003_stage1`. B = 2000 track-cluster bootstrap over Cert tracks (a second run with 2B draws as the stability check), strata from each seed's Fit c*, labels = head top-1 on gated-in Cert frames, no margin threshold. Labels read: Cert only, plus the dataset-level rho over all 39,209 frames. Test untouched.
- Rules applied exactly as pre-registered: G1(i) 2.5th percentile > 0 in >= 4 of 5; G1(ii) median >= 30 hard-group errors and >= 10 hard-error tracks; per-track mean primary, pooled sensitivity (Addendum A); eps primary, eps3 first fallback.

### G1(i): hard-minus-easy gap on Cert, percentage points (per-track mean = primary)

| variant | seed | easy | hard | gap | SE | 95% lower | pass | pooled gap | pooled lower | pooled pass |
|---|---|---|---|---|---|---|---|---|---|---|
| eps | 0 | 8.16 | 9.00 | +0.84 | 1.12 | -1.35 | no | +1.09 | -0.79 | no |
| eps | 1 | 5.70 | 5.72 | +0.02 | 1.06 | -1.97 | no | +0.96 | -0.48 | no |
| eps | 2 | 6.92 | 7.42 | +0.50 | 0.98 | -1.33 | no | +0.84 | -0.98 | no |
| eps | 3 | 6.73 | 6.87 | +0.14 | 1.08 | -2.07 | no | +1.01 | -0.79 | no |
| eps | 4 | 8.06 | 7.13 | -0.93 | 1.07 | -3.05 | no | -0.45 | -2.17 | no |
| eps3 | 0 | 8.77 | 10.05 | +1.28 | 1.37 | -1.44 | no | +1.35 | -0.65 | no |
| eps3 | 1 | 6.89 | 7.37 | +0.47 | 1.30 | -1.93 | no | +1.06 | -0.43 | no |
| eps3 | 2 | 7.81 | 8.85 | +1.04 | 1.36 | -1.55 | no | +1.01 | -1.03 | no |
| eps3 | 3 | 7.58 | 7.94 | +0.36 | 1.33 | -2.31 | no | +1.21 | -0.65 | no |
| eps3 | 4 | 8.44 | 7.46 | -0.98 | 1.27 | -3.36 | no | -0.25 | -2.07 | no |

- G1(i) verdict, eps (per-track, primary): 0 of 5 reshuffles pass -> FAIL. Pooled (sensitivity): 0 of 5 -> FAIL. Mean gap +0.11 pp per-track, +0.69 pp pooled. The two definitions agree, so the Addendum A caveat is not triggered.
- G1(i) verdict, eps3 (pre-registered first fallback): 0 of 5 -> FAIL. Pooled: 0 of 5 -> FAIL. Mean gap +0.43 pp per-track, +0.88 pp pooled.
- Bootstrap stable when B is doubled: True for every seed and both definitions.
- Not a near miss: the best 95% lower bound is -0.43 pp (eps3, pooled, seed 1); a pass would need the gap to exceed about 1.96 SE, i.e. about 1.9 to 2.7 pp, against observed gaps of -1.0 to +1.4 pp.

### G1(ii): hard group on Cert

| variant | hard-group frame errors, seeds 0-4 | distinct tracks with a hard error, seeds 0-4 | median errors / median tracks | verdict |
|---|---|---|---|---|
| eps | 722, 490, 718, 608, 511 | 156, 129, 158, 142, 144 | 608 / 144 | PASS (need >= 30 and >= 10) |
| eps3 | 747, 493, 701, 616, 525 | 158, 133, 156, 144, 143 | 616 / 144 | PASS |

### rho and design effect

- All 39,209 frames in 1,307 tracks: rho 0.446 [0.394, 0.495]; design effect (m = 30) 13.9; effective sample size 2,817 frames.
- Gated-in Cert frames per reshuffle, all strata: rho 0.377 to 0.438 (CI widths about 0.14 to 0.16), actual design effect 11.0 to 13.1, effective sample size 1,706 to 1,948 frames.
- By stratum (eps): easy rho 0.519 to 0.582, hard rho 0.289 to 0.380.
- ICC inside [-0.05, 1]: yes (dataset-level and Cert).

### Rank of error (gated-in wrong Cert frames, 5 overlapping reshuffles, 5,703 errors)

- True class is 2nd 58.1%, 3rd 20.0%, 4th or lower 21.9%.

### Fragment counts (counts only, no errors; seed 0, eps)

| group | tracks | mean frames per track | median | tracks with <= 2 frames |
|---|---|---|---|---|
| Fit easy | 160 | 23.8 | 30 | 4.4% |
| Fit hard | 148 | 25.0 | 30 | 4.7% |
| Cert easy | 493 | 23.1 | 29 | 5.1% |
| Cert hard | 456 | 24.1 | 30 | 3.9% |

- Cert tracks with gated-in frames in BOTH strata: 168 of 784. The other 616 (79%) sit in one stratum only.

### Checks
- [x] tests: 48 passed, 2 skipped (torch blocked; expected)
- [x] Stage 4 file hashes, splits, tau and c* matched (the script asserts)
- [x] bootstrap stable when B is doubled (all seeds, both definitions)
- [x] ICC inside [-0.05, 1]
- [x] all-frame head error 7.40% reconciles with the Stage 3 head accuracy (`reconciles_with_stage3_head_accuracy: True`)
- [x] no Test label read (enforced by a unit test; the one all-frame number is the dataset-level rho and the head error)

### What the numbers say
- [Certain] G1(i) fails for eps and for eps3, under both base-error definitions, in all 5 reshuffles. This is the outcome stated before Stage 5 in the Fit-only diagnostic entry. It is a pre-registered result, not a deviation.
- [Likely] The true clean-data gap is small, roughly 0 to 1 pp. The pooled gap is positive in 4 of 5 seeds for both variants (+0.7 and +0.9 pp on average), the same pattern as on Fit, so a weak positive effect is plausible but not detectable. With SE about 1.1 pp at 784 Cert tracks, confirming a 0.7 pp gap at 80% power would need an SE near 0.25 pp, about 19 times the tracks (roughly 15,000, against 1,307 in all of GTSRB). That ceiling belongs to the dataset, not to the method. [Guessing on the exact multiple: it assumes the same variance per track.]
- [Certain] G1(ii) passes by a wide margin (hundreds of hard-group errors over 130 to 160 tracks), so the hard stratum is not starved of errors.
- [Certain] rho = 0.446 over all frames. A frame-level certifier would treat 39,209 frames as independent when they carry the information of about 2,800 (design effect 13.9). This supports track-level certification (RG2) independently of G1.
- [Likely] Strata are mostly properties of the physical sign: 79% of Cert tracks fall in a single stratum, because eps depends on the (top-1, top-2) class pair, which is nearly constant within a track. Stratified certification will therefore behave like certifying groups of signs. The Stallkamp six-subset and class-conditional Mondrian baselines (RQ1b) are the decisive comparison; eps cannot be assumed to beat them.
- [Likely] Rank of error: 58% of errors have the true class 2nd, which supports the top-2 argument; 22% fall outside the top 3, where neither eps nor eps3 can describe the confusion.
- [Guessing] The easy stratum has higher rho (0.52 to 0.58) than the hard stratum (0.29 to 0.38): its errors sit in particular tracks (a few signs misread repeatedly), while hard-stratum errors are spread more widely. Descriptive only.
- The Fit diagnostic suggested the eps3 per-track gradient might be a fragment-track artifact. Fragments are few (about 4% to 5% of tracks have <= 2 frames in a stratum), so that explanation is at most partial. Error by fragment size was deliberately not computed.

### Decision
- Plan section 9, G1(i) fails: (1) Stage 3 Spearman and the pair-level check are already on record (Spearman 0.434; top 200 eps pairs hold 81.6% of confusion events), so no new action; (2) the pre-registered eps3 fallback also fails, so it does not rescue G1(i); (3) the scenario cells (RQ2) carry the main claim and clean GTSRB is reported as a certifiability frontier, which the proposal already states as the plan for a G1 failure; (4) this is the "negative result plus frontier" branch for the clean-data hypothesis.
- No R7, no change to c*, delta, beta, strata, seeds, definitions or aggregation rules.
- Deviation from R6? No.
- Next step: Stage 6 (certification engine). Synthetic validity tests must pass before any real data is used. G1(iii) and the floor table come from Stage 6.
