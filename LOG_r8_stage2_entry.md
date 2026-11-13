## POC-2 (R8) Stage 2: design phase and D-gate (design set only)

### Part A: fixed definitions (committed BEFORE any Stage 2 number was computed)
Written and committed before `src/r8_stage2.py` was run on real data. The git commit of this Part A is the proof of order.
Data: the 261 design tracks only (seed-0 Fit split, `track_split == 0` in `splits/seed0.npz`). delta = 0.1, beta = 5%,
n = 784 for every plug-in p-value. "Commit fraction" c = committed tracks / 261 (tracks with no passing frame count as
not committed). Grid = the Stage 1 lambda grid of the variant (200 points, `{V}_grid` in `track_table_design.npz`).

(a) **D1.** D1 passes if the upper 95% Clopper-Pearson bound of the design-set track error at the last frame
(V1, commit everything) is below the design-set single-frame error.
  - Track error at the last frame: share of design tracks with at least one passing frame whose V1 top class after
    their last frame (`V1_wrong_last`) differs from the label. Tracks with no passing frame are left out of this
    share (they are rejected as unknown, not misread) and their count is printed beside it.
  - Upper bound: the upper end of the two-sided 95% Clopper-Pearson interval (Beta quantile 0.975). This is the
    stricter of the two usual readings of "upper 95%".
  - Single-frame error: POC-1 head top-1 error on the gate-passing frames of those same design tracks (the evidence
    the commit rule uses). Chosen over "all design frames, no gate" because it is like-for-like and stricter: the
    gate's help is not credited to accumulation. The all-frames error is printed beside it for reference only.

(b) **Selective error** e = errors among committed tracks / committed tracks. If no track commits, e is undefined and
that lambda is not eligible in any rule that compares e.

(c) **Certificate A threshold order.** Start at the design-set argmin of mean L at the primary alpha, then relax one
grid point at a time.
  - mean L = (errors + alpha * (261 - committed)) / 261 = alpha * (1 - c) + c * e.
  - The argmin is taken over the frozen ~30-point grid of the chosen variant. Ties go to the larger lambda.
  - "Relax" = the next smaller lambda in that grid, down to the smallest grid point.

(d) **Certificate B size and lateness.** gamma = 10%. s = the smallest multiple of 5 pixels (5, 10, ... up to the
largest design-set ROI size, rounded up) at which the Certificate B plug-in (i) is predicted certifiable at some
working-grid lambda. If no s passes, Certificate B is recorded as "not predicted certifiable at gamma = 10%" and s
falls back to the plan's rule: the design-set median commit size at the start threshold (c), over committed tracks,
rounded up to the next 5 pixels. gamma is not changed in either case.
  - Why not the plan's median rule as primary: a track is late unless it commits at size <= s, so with s at the
    median commit size about half of the committed tracks plus all uncommitted tracks are late (near 50%), against
    gamma = 10%. That rule would decide R6 before any data is seen. Declared deviation from plan Stage 2 text
    ("For s, take the design-set median commit size"); allowed by plan section 2, which lets (s, gamma) be chosen
    in the design phase, and it follows the R6 mitigation (choose (s, gamma) with the same safety margin).
  - late_i(s) = 1 unless track i commits at a frame whose ROI size is at most s; that is,
    late_i = 1 - committed_i * [size at commit <= s]. ROI size = min(x2 - x1, y2 - y1), original pixels (Stage 1).
  - "Rounded up to the next 5 pixels" = smallest multiple of 5 that is >= the value (exact multiples unchanged).

(e) **D2 plug-in.** For each variant, each alpha in {0.5, 1, 2, 3, 5}% and each lambda of the Stage 1 grid:
mu = alpha * (1 - c) + c * e (mu = alpha when nothing commits), p = `src.certify.hb_pvalue(mu, 784, alpha)`. An alpha
is predicted certifiable for that variant if some lambda has p <= delta/10 = 0.01 and c >= 0.70. D2 passes if at least
one (variant, alpha) pair is predicted certifiable.

(f) **Primary alpha** = the smallest alpha in the grid that is predicted certifiable for at least one variant.

(g) **Variant** (plan section 5 rule, applied at the primary alpha): among the variants predicted certifiable at the
primary alpha, the one with the highest design-set commit fraction among lambdas whose selective error is at most
alpha/2. Ties go to V1, then V2, then V3.

(h) **Working lambda grid (~30 points)**, chosen variant only. Useful range = Stage 1 grid points with c >= 0.70 and
e <= primary alpha. Take 30 points evenly spaced by grid index from the lowest to the highest useful index (all of
them if there are 30 or fewer). The grid is a subset of the Stage 1 grid, so every row of it is the hash-checked
Stage 1 table.

(i) **Certificate B plug-in preview (risk R6).** For each working-grid lambda at the primary alpha and (s, gamma):
p_error as in (e), p_late = `hb_pvalue(mean late_i(s), 784, gamma)`, p = max(p_error, p_late). Predicted certifiable if
some lambda has p <= 0.01 (same safety margin as D2). Certificate B order = ascending design-set p, ties to the
larger lambda.

(j) **Pass levels.** P1 and the "4 of 5 re-splits" counts of P2 to P4 are as written in plan section 8.
P2: c_min = predicted coverage - 10 points, where predicted coverage = the design-set commit fraction at the threshold
where a fixed-sequence walk over the design-set plug-in p-values (n = 784, level delta = 0.1, order (c)) stops. This
is the threshold Certificate A would certify if Cert looked exactly like the design set.

(k) **D3** (informational). Frame wrong = POC-1 head top-1 (stored `top3[:, 0]`, no gate) differs from the label.
Count the design tracks with >= 25 wrong frames; for each, print wrong frames, passing frames, the last-frame margin
of V1, V2 and V3, and whether the last-frame top class is right.

(l) **Error-vs-frame curve** (descriptive). For each position t = 1..30 in frame order, among design tracks with at
least one passing frame by t: error of the top class at t for V1, V2 and V3, the single-frame head error at t, and
the share of tracks with evidence by t.

(m) **Integrity.** Only the 261 design tracks are used. `track_table_pool.npz` is never opened (the loader refuses
that path). Full-release arrays (embeddings, stored scores, gate flags) are cut to design rows right after loading,
before any computation. The design commit table recomputed in Stage 2 must equal the stored Stage 1 table exactly,
or the stage stops.

### Part B: results (fill from the printout after the run)
- Date: 2026-11-13 / Stage: R8 Stage 2 / Run tag: runs/20261003_stage1/poc2/stage2
- Git commit of Part A: f19cbfb (Amendment 1 to (d): ccc400a) | Git commit of the code: fe62b23
  (src/r8_stage2.py SHA-256 bff522fcadbecf7126b9b861d305358d281c4c88d0ed88475bcf0c0f3b417fa6)
- Output: stage2_summary.json SHA-256 332964ba26a977dda167d194e6954bca627a82eaebeec8825a66b270c4f96e74
- Input hashes checked against stage1_summary.json: all match. checkpoint 86527359...0bcc, train_release.npz 228fe5b2...d380,
  head.npz 5e85b1b0...a5c2, seed0.npz 397611c5...ea49, scores_train_release.npz 0be15ed0...eaac (vs head_summary.json),
  track_table_design.npz ad6fa0c5...b391, frame_evidence.npz f79033e8...e960. Files opened: seed0.npz, train_release.npz,
  scores_train_release.npz, frame_evidence.npz, head.npz, track_table_design.npz (track_table_pool.npz not opened).
- D1: track error at last frame 2.30% (6/261; 0 tracks with no passing frame) (upper CP bound 4.94%) vs single-frame
  error 4.54% (passing frames; all frames 6.33%, reference only) -> FAIL
- D2: predicted-certifiable (variant, alpha) pairs: V1 at 1, 2, 3, 5%; V2 at 5%; V3 at 2, 3, 5% -> PASS
  (V1 at 1%: lambda 7574, c 89.66%, e 0.00%, plug-in p 0.0054)
- D3: tracks wrong on >= 25 frames: 4 (19_00000 30/30, 39_00006 30/30, 40_00001 28/30, 41_00005 30/30); their V1
  last-frame margins: 2615.5, 671.9, 565.5, 7366.8 (all four wrong at the last frame in V1, V2 and V3)
- Error-vs-frame curve (V1, commit everything): 13.04% at frame 1, 8.14% at frame 10, 4.60% at frame 20, 2.30% at
  frame 30. V2 is identical to V1 at every frame (dividing all class scores by the same count does not change the
  top class). V3 ends at 2.30%.
- Primary alpha: 1% | Variant: V1 (only V1 is predicted certifiable at 1%; commit count at e <= alpha/2: 240) |
  Working grid: 30 points, lambda 3723.7 to 12898.9 (Stage 1 indices 132 to 167)
- Start threshold: working position 18, lambda 8130.9 (c = 89.27%, e = 0.00%; median commit frame 14, median size 29 px) |
  s = not certifiable at n = 784, gamma = 10%
- Note: the D2 best V1 lambda (7574, c 89.66%) is one of the 6 useful Stage 1 points dropped when rule (h) thins 36
  points to 30, so the start is 8130.9. This follows the committed rule; reported, not changed.
- Certificate B preview: min p = 0.241 -> predicted certifiable NO (risk R6)
- P2 c_min: 79.27% (predicted coverage 89.27%: the fixed-sequence walk at delta stops at working position 17)
- Checks: S1 PASS S2 PASS S3 PASS S4 PASS S5 PASS D1 FAIL D2 PASS — ALL PASS: False
- Tests: first run 3 Stage 2 test failures, caused by the fake-table helper making every track commit at lambda = 0; fixed in tests/test_r8_stage2.py only, src/r8_stage2.py unchanged; second run 197 passed, 1 known sklearn failure.
- Decision: D1 FAILED (track error 2.30%, 6/261; upper 95% CP 4.94% vs passing-frame error 4.54%). D2 PASSED (V1,
  alpha 1%, plug-in p 0.0054). Plan section 11 fallback 1 offers V2/V3 and a higher alpha, but neither can change D1
  (D1 is defined on V1 at the last frame; V2 and V3 have the same 2.30%). Declared deviation, made before any Cert or
  Test track is touched: proceed to F1 because D2 passed. The claim "accumulating evidence lowers error compared with
  a single frame" is dropped from all write-ups; D1 is reported in the paper as a pre-specified check that failed.
- Certificate B: no s is certifiable at gamma = 10% (best plug-in p 0.24). At the start threshold about 11% of tracks
  never commit, so the late fraction there is above gamma; at the lower thresholds where enough tracks commit (up to
  93.5%), the error part fails (plug-in p_error >= 0.24). gamma stays 10%. Stage 5 reports "not certifiable at
  n = 784" with the descriptive commit-fraction-vs-size curve. No amendment.
- Deviation from the plan? Yes. (1) Proceeding to F1 with D1 failed (plan section 8: "Do not go past it until D1 and
  D2 pass"); see Decision. (2) Part A (d) s rule and Amendment 1 (smallest certifiable s, no fallback) instead of the
  plan's median rule; declared before the run.
- Stage 2 status: checks not all passed (D1 FAIL, D2 PASS). Hard rule 5 and plan section 8 require all checks to pass; the author overrides this for D1 only, for the reasons in the Decision above, before any Cert or Test track is touched. Hard rule 5 itself is unchanged.
- Next step: the author reviews the D-gate; F1 (frozen_r8.yaml and the pre-registration block) only after that.

### Amendment 1 to Part A (d) (2026-11-13, before any Stage 2 run)
The fallback to the median rule is removed: if no s passes the Certificate B plug-in, s = not certifiable at n = 784,
with no substitute value. Reason: the median rule makes about half the tracks late by construction, so a fallback s
would be a value known to fail gamma. Made before `src/r8_stage2.py` was run on any data. No Stage 2 number existed,
and no Cert or Test track has been touched (plan section 2).
