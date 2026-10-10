## 2026-10-10 | Stage 4 | reshuffles and Fit-side calibration (run tag 20261003_stage1)
- Config / code / checkpoint / git commit: constants = the `constants_used` block of stage4_summary.json (plan section 1 values, code defaults; no `splits:` block was added to poc.yaml) ; code = src/splits.py SHA-256 065e7047308a163c8baacd3a5030f766c6ac25d49aed75a1bcb0a7c348e3a1d7 (checked on the author's machine with Get-FileHash: identical) ; checkpoint 86527359bc61a5fbc4d48b7744e049b5ca3ff81cb016fb237298d0c007140bcc (verified by the script, which also verified the Stage 2 and Stage 3 file hashes) ; git commit: ______________
- What was done: `py -3.12 -m src.splits --run runs/20261003_stage1`. Track-level splits Fit/Cert/Test = 261/784/262 (seeds 0-4), open-set gate tau, cut-point c*, Fit base errors, 50-point margin grids and the plan 6.3 testing order, for eps (primary) and eps3 (first fallback). Output folder: runs/20261003_stage1/splits/.
- Not done on purpose: no Cert or Test error rate of any kind was computed.
- Two runs, one code change (read this):
  - Run 1 used the POOLED base error (weight 1/n_t per frame) for the exploratory alpha = 0.5 x base. After seeing the Fit base errors I decided the PER-TRACK MEAN is the primary definition (see Interpretation choices). src/splits.py was updated and the script re-run (run 2).
  - Only the "rel_0.5base" ordering changed (it now uses 0.5 x per-track mean; the pooled version is kept as "rel_0.5base_pooled"). Splits, tau, c*, easy-mass fractions, base errors, grids and the absolute-alpha orders are deterministic and do not depend on this. The tables below come from run 1; the run 2 printout was checked and matches them: [x] matches (c*, easy mass, rejection rates, per-track and pooled base errors identical).
  - Run 1 file hashes are superseded. Use the run 2 hashes below.
- Output file SHA-256 (run 2, from stage4_summary.json `files_sha256`):
  - seed0.npz 397611c5538cc9ebe7a0be768ada29cb0afa7c89bdc53511df3a2403001fea49
  - seed1.npz f4d3555bcc765e66fd8c7c1358e625f3dfe39d9e97dc098197320b38ec5055b4
  - seed2.npz f0f70752cb896e7317bad891574c74a099628313ef087a7bd1b74aaf417ac02e
  - seed3.npz cafb335d3eb2b1c9f27645b3b495afd47dca7f10be367c9d0a4a393289791991
  - seed4.npz eb53fbef50819d167fa9c225b0cac19b9f5d01e4154ea58c50169d9e5e94b154

### Result numbers (from splits/stage4_summary.json; Fit errors are on gated-in frames, no margin threshold)

Splits and gate. All seeds: tracks 261/784/262, conformal rank 249 of 261, min tracks per class per split 1/4/1. Frames per split about 7,830 / 23,520 / 7,860.

| seed | tau | Cert reject, frame-level | Cert reject, 1 frame/track | eps c* | easy mass Fit | easy mass Cert |
|---|---|---|---|---|---|---|
| 0 | 43.84 | 4.8% | 4.6% | -230.74 | 50.7% | 50.9% |
| 1 | 41.49 | 8.5% | 7.9% | -231.21 | 49.5% | 48.8% |
| 2 | 44.45 | 4.2% | 5.2% | -232.85 | 49.4% | 47.7% |
| 3 | 42.98 | 6.7% | 6.6% | -230.03 | 49.6% | 50.8% |
| 4 | 44.13 | 4.3% | 4.5% | -230.74 | 49.9% | 51.0% |

Fit base error per stratum, eps, in percent. PRIMARY = per-track mean; pooled = sensitivity. Gap = hard minus easy.

| seed | easy per-track | hard per-track | GAP per-track | easy pooled | hard pooled | gap pooled |
|---|---|---|---|---|---|---|
| 0 | 5.51 | 5.94 | +0.43 | 4.52 | 4.57 | +0.05 |
| 1 | 5.98 | 6.59 | +0.61 | 3.88 | 5.33 | +1.44 |
| 2 | 7.95 | 8.32 | +0.37 | 5.89 | 5.31 | -0.57 |
| 3 | 5.58 | 5.23 | -0.35 | 3.66 | 4.24 | +0.58 |
| 4 | 5.11 | 6.61 | +1.50 | 5.10 | 5.80 | +0.70 |
| mean | | | +0.51 | | | +0.44 |

Fit base error per stratum, eps3 (first fallback), in percent. c* = -224.77 in all five seeds.

| seed | easy per-track | hard per-track | GAP per-track | easy pooled | hard pooled | gap pooled |
|---|---|---|---|---|---|---|
| 0 | 6.29 | 7.57 | +1.28 | 4.47 | 4.62 | +0.15 |
| 1 | 5.85 | 6.97 | +1.13 | 3.75 | 5.48 | +1.73 |
| 2 | 9.58 | 9.06 | -0.52 | 5.78 | 5.41 | -0.37 |
| 3 | 5.38 | 7.05 | +1.67 | 3.42 | 4.47 | +1.04 |
| 4 | 6.74 | 8.59 | +1.85 | 4.87 | 6.04 | +1.17 |
| mean | | | +1.08 | | | +0.75 |

Easy-mass fraction for eps3: Fit 49.4-50.0%, Cert 49.1-51.1%.

Fit-based testing order: the start index (of 50 grid points) is 6-23 depending on stratum, variant and alpha, so the fixed-sequence test walks at most 7-24 thresholds. Estimated Fit E[L] at the start point for alpha = 3% is 0.007-0.013, far below alpha, so Fit shows a wide cushion. Fit is only a hint about Cert.

### Checks
- [x] splits disjoint, cover all 1,307 tracks, every class in every split (script asserts)
- [x] realised Fit split within 45/55 in all 5 seeds (49.4% to 50.7%)
- [x] Cert easy-mass 47.7% to 51.0% for eps, nowhere near the 35/65 warning line
- [~] Cert one-frame-per-track rejection: 4.5%, 4.6%, 5.2%, 6.6%, 7.9% (mean 5.8%). Two seeds are above 5% but this is within sampling noise (rough SD of one seed about 1.6 pp, because tau comes from only 261 tracks; seeds share tracks, so they are not independent). Gate not retuned.
- [x] frame-level Cert rejection maximum 8.5%, below the 10% NOTE line
- [x] no NOTE (lopsided split / rejection above 10%) was printed
- [x] no Cert/Test error computed
- [x] tests: `py -3.12 -m pytest tests -q` gives 41 passed, 0 failed, 0 skipped, 1 warning (2026-10-10, Python 3.12.10). The warning is the harmless NumPy non-writable-array UserWarning at src/train.py line 129 (old evaluate(); no effect on results or on the checkpoint). An earlier run gave 39 passed because tests/test_stage1.py was an older copy without the diagnostics and Stage 2 smoke tests; after replacing it the full suite passes.

### What the Fit numbers say (Fit only, NOT G1 evidence)
- The eps hard-minus-easy gap is small: mean +0.51 pp per-track (+0.44 pp pooled) on a base error of 5-8%. It is positive in 4 of 5 seeds under both definitions (negative: seed 3 per-track, seed 2 pooled). [Likely] this is a risk for G1(i). Cert has about 3x the tracks of Fit, but a 0.5 pp gap against track-clustered errors is probably well under one standard error [Guessing; no variance computed yet].
- eps3 shows a larger gap on Fit (+1.08 pp per-track, +0.75 pp pooled, positive in 4 of 5 seeds). Not a result. eps3 was named as the first fallback before this; Stage 5 reports eps and eps3 side by side.
- Why the median cut may be weak [Guessing]: confusions concentrate in a few hundred pairs (top 200 hold 81.6%), and the top-2 pair of a frame is made of near neighbours, so most frame mass already sits in high-eps pairs. A 50/50 mass cut can then fall where error rates are similar. A Fit-only table of error by eps-octile would test this. It may inform a future revision R7. It does not change c*.
- G1(ii) (>= 30 hard-group frame errors over >= 10 tracks on Cert) should be met easily: about 11,000 Cert hard-stratum frames at 4-6% base error [Likely].
- G1(iii) looks feasible on Fit, but Fit is optimistic because the margin threshold is chosen there. The Stage 6 certifier decides.
- Pooled vs per-track base error differ by 1-4 pp (e.g. seed 2 hard: 5.31 vs 8.32). Errors sit in tracks that contribute only a few frames to a stratum (probably tracks whose predicted pair flips); the per-track mean gives those tracks the same weight as full tracks.
- The relative alpha (0.5 x per-track base) comes out at about 2.6-4.8%, so some of these are above 3%. Only the absolute grid values 1%, 2%, 3% count for G1(iii) (pre-registration item 1). The relative alphas are exploratory.
- Margins and tau are on a large raw scale (margin thresholds in the hundreds, tau about 41-44) because the Stage 3 head is fitted on very tight clusters. Only ranks and quantiles are used, so nothing breaks, but these are not calibrated probabilities.

### Interpretation choices
- Base error (decided 2026-10-10 by the author): the PER-TRACK MEAN of in-stratum error rates is primary, for G1(i) and for alpha = 0.5 x base. Reason: the Stage 6 certificate averages per track, so this is the quantity it bounds. The pooled rate (weight 1/n_t per frame) is reported as a sensitivity. Disclosure: this was decided after seeing the Fit base errors. Both definitions give the same sign count for eps (4 of 5), so the choice was not driven by which one gives the larger gap.
- c* mass still uses weight 1/n_t per frame, as the plan words it. This is the c* rule, not the base-error definition, and is unchanged.
- Margin accept rule: accept if margin >= threshold; grid index 0 = loosest.
- Test order per plan 6.3: start at the Fit argmin of estimated E[L] with L_t = l_t + alpha, then looser. alpha in {0.01, 0.02, 0.03, 0.05, 0.5 x per-track base, 0.5 x pooled base}; all stored, none chosen here.
- eps3 = eps(y1,y2) + eps(y1,y3), summed in the log domain.

- Decision taken and why: keep c*, delta, beta, strata, reshuffles and ordering exactly as defined (plan section 6, Stage 4 and 6.3). Nothing re-tuned despite the small Fit gap.
- Deviation from R6? No. The per-track definition is an interpretation of an undefined plan term, not a design change.
- Next step: read and accept (or edit) the PRE-REGISTRATION block at the top of LOG.md, commit it, record the hash there and here: ______________. Then Stage 5.
