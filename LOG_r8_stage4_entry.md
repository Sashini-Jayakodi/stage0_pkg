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
Computed 2026-11-14 after Amendment 1 (6db7d4b), with `src/r8_stage4_baselines_design.py` committed first (a74a164;
SHA-256 d0ca321e7b7f3141a2e1247b12864c63d08cd63dc573ff1769452ae939a8d1c3). Output
`runs/20261003_stage1/poc2/stage4/baselines_design.json` SHA-256 80fef0be5ac04f0fb911f20752b8efebdeffc7254d7c3e0e47a9c23ca99ad580.
Files opened: seed0.npz, train_release.npz, scores_train_release.npz, frame_evidence.npz, head.npz,
track_table_design.npz (not track_table_pool.npz). Thresholds below are rounded for reading; the exact values are in
the JSON, which the Stage 4 code reads and hash-checks.
  - **B2** (random passing frame): labelled tracks 261/261; design error committing every labelled track 6.51%.
    Working grid 21 points (the whole useful range), candidate indices 134 to 154, thresholds 285.851 to 461.744;
    design coverage 86.97% to 72.03%, error 0.881% to 0.000%. Start at working position 13 (threshold 390.400,
    c 80.46%, e 0.000%), then relax to 0.
  - **B3** (last frame): labelled tracks 250/261 (11 last frames fail the gate); design error committing every
    labelled track 3.60%. Working grid 30 points, candidate indices 88 to 136, thresholds 217.421 to 486.576;
    coverage 90.04% to 70.11%, error 0.426% to 0.000%. Start at working position 15 (threshold 330.761, c 83.14%,
    e 0.000%), then relax to 0.
  - **B4** (majority vote, margin in votes): labelled tracks 261/261; design error committing every labelled track
    2.30%. Working grid 30 points, candidate indices 45 to 192, thresholds 5.015 to 27.921; coverage 96.93% to
    72.41%, error 0.791% to 0.000%. Start at working position 26 (threshold 23.434, c 83.14%, e 0.000%), then relax
    to 0. The vote margin is a whole number, so neighbouring thresholds between the same two integers make the same
    decisions; the rules are applied as written.

### Part B: results (fill from the printout after the run)
- Date: 2026-11-14 / Stage: R8 Stage 4 / Run tag: runs/20261003_stage1/poc2/stage4
- Git commit of Part A: 690713e (rules), 6db7d4b (Amendment 1), 2a22adb (B2/B3/B4 values), 4631b32 (Amendment 2) |
  Git commit of the code: e31dff2 (src/r8_stage4.py SHA-256 fae67390a488b44c57f1386c9b256cb0fbbcf2444282f5724c2580aa3e6445c2)
- Output: stage4_summary.json SHA-256 e66e45f210ebae5e6d21a4d8247584e1affc2d9691a760427814c821b164d2cc (2.0 s).
  train_meta.csv SHA-256 4c06c3ad07a3b3b89faabcd49b26e8f6dbfff6460f25bf5b66730140de6a132e (no earlier record; its ROI
  boxes were cross-checked against the Stage 1 sizes, check PASS).
- Per re-split at 1%: commit rule certified in 0 of 5 (Test coverage 0% in every re-split, so no error, frame or size).
  B2 certified in 0 of 5. B3 certified in 5 of 5 (coverage 87.02, 88.55, 87.40, 90.46, 88.17%; error 0.00 to 1.27%;
  frame 30, 70 to 81 px). B4 certified in 5 of 5 (coverage 83.97, 82.06, 82.44, 82.44, 81.30%; error 0.00 to 0.45%;
  frame 30, 72 to 81 px).
- P2, P3, P4 all FAIL (0/5), because the commit rule was not certified at the primary alpha 1% in any re-split while
  B3 and B4 were.
- Certifiability curve (smallest certified alpha, n = 784): commit rule 2% in all five re-splits; B2 2% (seeds 0, 1, 4)
  and 3% (seeds 2, 3); B3 and B4 1% in all five. Commit rule against n (n = 100 / 200 / 400 / 784): seed 0 5/3/2/2%;
  seed 1 5/2/1/2%; seed 2 none/5/3/2%; seed 3 none/3/2/2%; seed 4 none/5/2/2%. Not monotone in n for seed 1 (the
  nested 400-track subsample certifies 1%, the full 784 does not): the curve is noisy at these sizes.
- 2% table (Test; coverage / error among committed / median commit frame / median size px):
  - seed 0: commit 93.51 / 1.22 / 10 / 26; B2 82.06 / 2.33 / 16 / 32; B3 89.31 / 0.43 / 30 / 70; B4 90.08 / 1.27 / 30 / 70
  - seed 1: commit 92.37 / 0.41 / 10 / 28; B2 82.06 / 0.93 / 16 / 34; B3 90.46 / 0.42 / 30 / 81; B4 89.31 / 0.00 / 30 / 81
  - seed 2: commit 93.51 / 1.22 / 10 / 27; B2 not certified; B3 90.08 / 0.85 / 30 / 75; B4 90.08 / 0.42 / 30 / 75
  - seed 3: commit 92.75 / 0.82 / 10 / 28; B2 not certified; B3 90.84 / 1.26 / 30 / 74.5; B4 91.22 / 0.84 / 30 / 75
  - seed 4: commit 91.22 / 0.42 / 11 / 26; B2 81.30 / 0.94 / 17 / 31; B3 90.08 / 0.85 / 30 / 71; B4 87.79 / 0.43 / 30 / 72
  Wherever the commit rule was certified (2%, 3%, 5%), its Test error among committed signs stayed below alpha
  (at most 1.22% at 2%, at most 2.01% at 3% and 5%).
- Amendment 2, descriptive only (not scored): at the smallest alpha certified by both the commit rule and B2 (2% for
  seeds 0, 1, 4; 3% for seeds 2, 3), the commit rule is ahead of B2 by 10 to 14 points (93.51 vs 82.06, 92.37 vs 82.06,
  95.04 vs 80.92, 95.04 vs 81.68, 91.22 vs 81.30%) at median commit frames 8 to 11, in every re-split.
- Limitation (noted, not changed): the frozen working grid did not reach thresholds conservative enough to approach
  last-frame behaviour. Its strictest point (lambda 12898.9) has a design-set median commit frame of 18, and the frozen
  order starts at working position 18 (lambda 8130.9) and only relaxes toward smaller lambda, so positions 19 to 29
  are never tested. The commit rule therefore always commits early (median frame 8 to 11 where certified), where the
  confidently wrong tracks make the 1% test fail, while B3 and B4 decide at frame 30.
- Track-level false-rejection rate (Test): 1.15, 0.38, 0.38, 0.00, 0.38% (seeds 0 to 4).
- Checks: frozen_grid_matches_stage1_grid, gate_flags_reproduce, pool_commit_table_reproduces, b1_top_equals_stored_top1,
  meta_roi_matches_stage1_sizes, b2_frames_all_pass_gate, pool_size_matches_frozen, splits_valid all PASS — ALL PASS
  (checks): True. P2, P3, P4 (pass levels) FAIL.
- Decision: P2, P3 and P4 reported as FAIL (0/5). Nothing frozen changed; the primary alpha stays 1%. Per plan section
  11, P2's failure is reported with the certifiability curve; P3's and P4's failures are reported as they are.
- Deviation from the plan? Declared in Part A: B2, B3, B4 definitions, grids and orders specified after F1 from design
  data only (Part A (e), Amendment 1); descriptive comparison added before any Stage 4 number (Amendment 2). No other.
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

### Amendment 2 to Stage 4 Part A (2026-11-14, before any Stage 4 number existed)
Made after the author's review and before any Stage 4 code was written; no Stage 4 number or pool outcome for Stage 4
existed. **P3 and P4 scoring are unchanged, as written in Part A (f)** (an uncertified method has coverage 0 for P3;
a re-split with nothing committed at 1% does not pass P4).
Descriptive only, not scored: each re-split also reports the P3 comparison (commit-rule Test coverage vs B2 Test
coverage) and the P4 comparison (median commit frame < 30) at the smallest alpha in the frozen grid that is certified
by both the commit rule and B2 in that re-split, or "none" if there is no common alpha.
