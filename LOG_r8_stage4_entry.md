## POC-2 (R8) Stage 4: Certificate A on five re-splits

### Part A: fixed definitions (committed BEFORE any Stage 4 number was computed)
Confirm phase (F1 commit 3de4cb9). Nothing in `configs/frozen_r8.yaml` changes. delta = 0.1 throughout. The only
numbers computed before the values block below are the B2, B3 and B4 grids and orders, from the 261 design tracks only.

(a) **Freeze check.** Every Stage 4 script calls `load_frozen()` (`src/r8_frozen.py`) first and stops if
`frozen_r8.yaml` or the plan file differ from the F1 hashes. Every other file read is hash-checked against the summary
JSON that wrote it (hard rule 2).

(b) **Re-splits.** Population: the 1,046 non-design tracks (seed-0 POC-1 `track_split != 0`). The design set is in
neither Cert nor Test. For each seed s in {0, 1, 2, 3, 4}, independently:
  - Stratify by class. For class k with n_k pool tracks, the Test quota is q_k = n_k * 262 / 1046; take floor(q_k),
    then give the remaining Test places one at a time to the classes with the largest fractional parts (ties to the
    smaller class id), so the Test total is exactly 262 and Cert is exactly 784.
  - Every class must be in Cert: a class's Test count is capped at n_k - 1 (a class with one pool track goes to Cert
    only). Places removed by the cap go to the next-largest fractional parts among uncapped classes.
  - Within class k, tracks are ordered by track code, shuffled with `np.random.default_rng([20261114, s, k])`, and the
    first (Test count) tracks go to Test, the rest to Cert.
  - Checks: Cert and Test disjoint, their union = the 1,046 pool tracks, no design track in either, every class in
    Cert, sizes exactly 784 and 262.

(c) **Certification (commit rule).** At every alpha in the frozen grid {0.5, 1, 2, 3, 5}%, with the frozen variant (V1),
working grid (Stage 1 indices) and threshold order, read only via `load_frozen()`; the same order at every alpha.
Per-track loss per lambda, `hb_pvalue(mean L, 784, alpha)`, fixed sequence at delta, stop at the first p > delta;
certified threshold = last pass; if the first fails, nothing is certified at that alpha (every Test sign escalated).

(d) **Test metrics per re-split and alpha** (262 Test tracks; tracks with no passing frame count as not committed):
  - certified: yes / no, and the certified lambda;
  - Test coverage = committed Test tracks / 262 (0 if not certified);
  - error among committed signs = wrong / committed (undefined if nothing commits);
  - median commit frame (1-based position in frame order) and median ROI size at commit, over committed Test tracks;
  - track-level false-rejection rate = Test tracks with no passing frame / 262 (all GTSRB classes are known, so every
    rejection as unknown is a false rejection; it does not depend on lambda).

(e) **Baselines B2, B3, B4** (plan section 9), each a track-level certificate with the same engine as (c): one score
and one label per track, committed iff the score >= threshold, wrong iff committed and label != class, n = 784 Cert
tracks, same Test metrics as (d). Gate-failing frames never give a label.
  - **B2, one random frame per track.** One frame per track, chosen once for all 1,307 tracks: u_i =
    `np.random.default_rng(20261114).random(1307)` in track-code order, position = floor(u_i * frames in track i).
    Score = that frame's single-frame margin (top minus second of log prior + frame log-likelihood, as B1), label =
    its top class. If that frame fails the gate, the track is not committed. Commit frame = that position.
  - **B3, last frame only.** The frame with the largest frame index. Score and label as B2. If the last frame fails the
    gate, the track is not committed (no fall-back to an earlier frame). Commit frame = the last position.
  - **B4, majority vote** (Gross et al. 2023 style fusion). Each passing frame votes for its top class; label = the
    class with the most votes (ties to the smaller class id); score = vote margin = votes for the label minus votes
    for the runner-up (a count, 0 to 30). No passing frame = not committed. Commit frame = the last position (the vote
    needs every frame).
  - **Grids and orders, from the 261 design tracks only**, by Stage 2 Part A rules: candidate grid = the Stage 1
    `make_grid` rule on the design scores (0 plus 199 log-spaced points); commit fraction = committed / 261;
    working grid = rule (h) at the primary alpha 1%; start = rule (c); then relax. The same order at every alpha.
    If rule (h) finds no working grid for a baseline, that baseline is recorded as "no working grid at 1%" and
    certifies nothing at any alpha.
  - Declared: B2, B3 and B4 are named in `frozen_r8.yaml` only through plan section 9, so these definitions, grids
    and orders are specified after F1, from design data only, before any Stage 4 number.

(f) **Pass levels, exactly as frozen** (`frozen_r8.yaml` pass_levels), evaluated per re-split at the primary alpha 1%:
  - P2: the primary alpha is certified AND Test coverage >= c_min = 0.79272030651341; PASS if true in >= 4 of 5.
  - P3: commit-rule Test coverage > B2 Test coverage (strict; an uncertified method has coverage 0); PASS if true in
    >= 4 of 5.
  - P4: median commit frame of the committed Test tracks < 30 (before the last frame); a re-split where nothing is
    committed at 1% counts as not passing; PASS if true in >= 4 of 5.

(g) **Certifiability curve** (plan section 11, reported whatever P2 gives):
  - Per re-split: the smallest alpha in the frozen grid that is certified with all 784 Cert tracks.
  - Against n: nested Cert subsamples of n in {100, 200, 400, 784} (the first n tracks of a permutation of the Cert
    tracks drawn with `np.random.default_rng([20261115, s])`); the smallest certified alpha at each n, per re-split.

(h) **Integrity.** Stage 4 changes no frozen value. Failed checks and pass levels are reported as failed.

#### B2, B3, B4 values (from the design-set script, design set only)
____

### Part B: results (fill from the printout after the run)
- Date: ____ / Stage: R8 Stage 4 / Run tag: runs/20261003_stage1/poc2/stage4
- Git commit of Part A: ____ | Git commit of the code: ____
- Per re-split at 1%: certified ____ ; Test coverage ____ ; error among committed ____ ; median commit frame / size ____
- Smallest certified alpha per re-split: ____ ; curve against n: ____
- P2 ____ ; P3 ____ ; P4 ____
- Baselines B2 / B3 / B4 at 1%: ____
- Checks: ____ — ALL PASS: ____
- Decision: ____
- Deviation from the plan? ____
- Next step: Stage 5 only after the author reviews Stage 4.

### Amendment 1 to Stage 4 Part A (2026-11-14, before any Stage 4 number existed)
Made after the author's review of Part A (690713e) and before the B2/B3/B4 design script was run; no Stage 4 number,
design-set baseline number or pool outcome for Stage 4 existed. B3, P4 and the certifiability curve are unchanged.
(1) **B2** picks its one frame uniformly at random among the track's gate-passing frames: u_i =
`np.random.default_rng(20261114).random(1307)` in track-code order (same seed as before), and the chosen frame is the
passing frame at position floor(u_i * number of passing frames) in frame order. A track with no passing frame is
rejected as unknown (not committed), as for the commit rule. Commit frame = the chosen frame's position in the track.
Reason: a fair baseline; picking gated-out frames would lower B2's coverage artificially and bias P3.
(2) **B4** counts votes over gate-passing frames only; ties to the smaller class id. This confirms the rule already
written in Part A (e); it is not a change.
