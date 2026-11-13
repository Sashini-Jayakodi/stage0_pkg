## POC-2 (R8) Stage 3: validity tests (3a synthetic, 3b resampling)

### Part A: fixed definitions (committed BEFORE any Stage 3 number was computed)
Confirm phase: F1 is closed (commit 3de4cb9). Nothing in `configs/frozen_r8.yaml` changes. delta = 0.1 throughout.
The only numbers computed before this commit are the B1 grid and order in (c), from the 261 design tracks only.

(a) **Freeze check.** Every Stage 3 script first checks the SHA-256 of `configs/frozen_r8.yaml`
(4f7a79156a4486b6b5f97515457513e6df2a2bfeed2a41b6fa05936bd1141d24) and of `docs/poc2-implementation-plan-r8.md`
(b0ad6292138732f40639753b6e5b76d050cfc30da137cc2b1d8f0c0e85174bd2), the values recorded at F1, and stops if either
differs (`src/r8_frozen.py`). Every other file read is hash-checked against the summary JSON that wrote it (hard rule 2).

(b) **Commit rule (track-level certifier).** Variant, working grid (Stage 1 indices), threshold order, alpha grid and
delta are read only from `frozen_r8.yaml`. The same frozen order is used at every alpha. Per draw: per-track loss
L_i = wrong_i - alpha * committed_i + alpha per lambda (tracks-by-lambdas table), p = `src.certify.hb_pvalue(mean L,
n_tracks, alpha)`, test in the frozen order at level delta, stop at the first p > delta; the certified threshold is the
last one that passed; if the first fails, nothing is certified. Tracks with no passing frame count as not committed.

(c) **B1 (frame-level Learn-then-Test).** Per-frame margin = top minus second of (log prior + frame log-likelihood),
i.e. the V1 score of a single frame. A frame is accepted if it passes the gate and its margin >= threshold
(gate-failing frames are never accepted); wrong = accepted and top class != label. Every frame is its own unit:
L_f = wrong_f - alpha * accepted_f + alpha, p = `hb_pvalue(mean L, n_frames, alpha)`, same fixed-sequence engine at delta.
  - Grid and order, from the 261 design tracks only, by the Stage 2 Part A rules applied to frames: candidate grid =
    the Stage 1 `make_grid` rule on the design per-frame margins (0 plus 199 log-spaced points); commit fraction =
    accepted frames / all design frames; working grid = rule (h) at the primary alpha 1% (c >= 70% and e <= alpha,
    30 points evenly by index); start = rule (c) (argmin of mean L at 1%, ties to the larger threshold), then relax
    one point at a time. The same order is used at every alpha.
  - Declared: B1 is named in `frozen_r8.yaml` only through plan section 9, so this grid and order are specified after
    F1. They use design data only and were fixed before any Stage 3 number. Values: see "B1 values" below.

(d) **True risk and violation.** True risk of a method at a threshold = the population mean of its loss at alpha
(track-level: mean of L_i over population tracks; B1: mean of L_f over population frames). A run violates if a
threshold was certified and its true mean loss > alpha (equivalently, selective error > alpha). A run that certifies
nothing does not violate. Violation rate = violating runs / runs.

(e) **3a synthetic** (new file `src/r8_stage3_synth.py`; `src/certify_synth.py` is not changed).
  - Evidence model: frame evidence x_it = a_i + e_it with latent track difficulty a_i ~ N(m, r) and e_it ~ N(0, 1 - r),
    30 frames per track. A frame is wrong iff x_it < 0. m is set so the frame error is 7.4% (plan's single-frame
    error). r is solved numerically so that the within-track correlation of the frame-wrong indicator equals
    rho_sim, for rho_sim in {0, 0.3, 0.45} (the quantity POC-1 measured as 0.446).
  - Why this model: in POC-1's simulator, a design with frames independent given a track error rate never broke the
    frame-level certifier (accepted errors lose their clustering); only confident whole-track errors did. Here a
    strongly negative a_i gives a confidently wrong whole track, as seen in the real design set (Stage 2 D3), without
    forcing it. No gate in 3a.
  - Commit rule: S_t = x_i1 + ... + x_it, margin |S_t|, commit at the first t with |S_t| >= lambda, wrong iff S_t < 0
    there. B1: frame margin |x_it|, wrong iff x_it < 0.
  - Grids: per rho, one fixed candidate grid per method from a pilot 261-track design sample (Stage 1 `make_grid`
    rule). Each run draws a fresh 261-track design sample and picks the working grid and order by rules (h) and (c)
    at alpha 1%, then a fresh 784-track Cert sample. 1,000 runs per rho. alpha = 1% (primary) only.
  - True risk: a 200,000-track population per rho, evaluated once on the candidate grids.
  - All seeds fixed in code.

(f) **3b resampling.** Population = the 1,046 non-design tracks; their commit-rule and B1 outcomes are computed here
for the first time (allowed after F1). Track-level outcomes from `track_table_pool.npz` (hash-checked against
`stage1_summary.json`) with labels from `train_release.npz`; B1 from per-frame margins of the pool frames. True risk
of every threshold is computed exactly on the 1,046 tracks (B1: on all their frames). 2,000 draws of 784 tracks with
replacement (fixed seed; a track drawn twice counts twice, its frames too). Each draw: track-level certifier at
n = 784 and B1 at n = frames in the draw. Reported for every alpha in the frozen grid; the primary alpha 1% is the
headline.

(g) **P1 scoring** (frozen P1; scope fixed here). At the primary alpha 1%, with two-sided 95% Clopper-Pearson bounds on
the violation rate:
  - Track-level part: upper bound <= delta + 0.02 = 0.12 in 3b AND in 3a at rho_sim 0, 0.3 and 0.45.
  - Frame-level part: lower bound > delta = 0.10 in 3b AND in 3a at rho_sim 0.45 (the real-data level).
  - 3a rho_sim = 0 is a negative control for B1 (frames independent, B1 expected valid); rho_sim = 0.3 is reported.
  - Other alphas are reported, not scored. P1 PASS only if every part holds; a failed part is reported as failed.

(h) **Extra checks.**
  - Three p-values hand-computed with exact arithmetic (Python `fractions`, no scipy): the zero-error case at n = 784,
    alpha = 1%; one case where the Hoeffding term is the minimum; one where the Bentkus term is. Engine values must
    match to 1e-9 relative.
  - Certificate B max-p rule on a synthetic case where the error promise holds and only the lateness promise is
    broken: p = max(p_error, p_late) must not certify there, and must certify when both promises hold.

(i) **If B1 does not visibly break on real data,** it is reported as measured (plan Stage 3); nothing is changed.

(j) **Outputs:** `runs/20261003_stage1/poc2/stage3/` b1_design.json (this step), stage3a_summary.json,
stage3b_summary.json, each with the hashes of what it read.

#### B1 values (from `src/r8_stage3_b1design.py`, design set only)
Computed 2026-11-14 after the Part A rules commit (eb4c332). Script SHA-256 a7084123571bc872c7bb08a3981bfe9ab1d72f5a56c5e37103ccfdf53f2a45bd;
output `runs/20261003_stage1/poc2/stage3/b1_design.json` SHA-256 8fb7e1c534ce1d923595e18794327afa70d654f033d4b55196a4dc2a169d22c9.
Files opened: seed0.npz, train_release.npz, scores_train_release.npz, frame_evidence.npz, head.npz,
track_table_design.npz (not track_table_pool.npz). 7,830 design frames of 261 tracks; 326 fail the gate (never accepted).
  - Working grid: 20 points (the useful range has 20 candidate points, so all are kept), candidate indices 131 to 150,
    thresholds 276.6808200078421, 283.49861752219556, 290.4844149830051, 297.64235214096715, 304.9766707559528,
    312.49171711065424, 320.19194458617164, 328.08191630106586, 336.1663078154409, 344.44990990165866,
    352.9376313833293, 361.634502044255, 370.5456756090572, 379.67643279725064, 389.03218445257073,
    398.61847474941555, 408.440984478297, 418.5055344122509, 428.8180887561991, 439.38475868130973.
  - Design-set frame coverage 83.19% (position 0) to 70.74% (position 19); frame selective error 0.921% to 0.253%.
  - Order: start at working position 19 (threshold 439.38475868130973; mean L 0.471% at 1%), then relax to position 0:
    [19, 18, ..., 0]. The start is the top of the working grid: rule (h)'s 70% coverage floor ends the range there.

### Part B: results (fill from the printout after the run)
- Date: ____ / Stage: R8 Stage 3 / Run tag: runs/20261003_stage1/poc2/stage3
- Git commit of Part A: ____ | Git commit of the code: ____
- 3a violation rates at 1% (track / B1), rho_sim 0, 0.3, 0.45: ____
- 3b violation rates per alpha (track / B1): ____
- P1: track part ____ ; frame part ____ -> PASS / FAIL
- Hand p-values: ____ | Certificate B max-p check: ____
- Environment observation: test_ledoit_wolf_matches_sklearn passes as of 2026-11-14 on the same scikit-learn 1.9.1 that was blocked during Stage 1, which points to the Application Control policy rather than a package upgrade; the Stage 1 entry is left unchanged.
- Checks: ____ — ALL PASS: ____
- Decision: ____
- Deviation from the plan? ____
- Next step: Stage 4 only after the author reviews Stage 3.
