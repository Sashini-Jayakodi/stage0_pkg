## POC-2 (R8) Stage 5: Certificate B status and the descriptive commit-by-size curve

### Part A: fixed definitions (committed BEFORE any Stage 5 number was computed)
Confirm phase (F1 commit 3de4cb9). Nothing in `configs/frozen_r8.yaml` changes. No new certification is run.

(a) **Certificate B, as frozen.** `frozen_r8.yaml`: s = "not certifiable at n = 784", gamma = 10%. Stage 5 reports that
status as frozen. s and gamma are not moved, and no joint certificate is run at any substitute s.

(b) **Descriptive curve (not scored, not a certificate).** For each re-split (the Stage 4 splits, seeds 0 to 4) and each
alpha at which the commit rule was certified in Stage 4 (2%, 3% and 5% in every re-split; not 0.5% or 1%), at the
Stage 4 certified threshold, over the 262 Test tracks:
  - committed by size s = the share of Test tracks with committed_i = 1 AND ROI size at commit <= s. This is
    1 - late_i(s) averaged over Test, with late_i(s) exactly as frozen (`frozen_r8.yaml` certificate_B.late_definition);
  - never committed = the share of Test tracks not committed (tracks with no passing frame included);
  - s = 5, 10, 15, ... px, one common range for every method, re-split and alpha: from 5 px up to the largest ROI
    size at commit among all curves, rounded up to the next multiple of 5 (beyond it every curve is flat);
  - ROI size = min(x2 - x1, y2 - y1) in original pixels (Stage 1 convention), at the commit frame.
The same curve for B2, B3 and B4 at their Stage 4 certified thresholds, at the same alphas. Where a baseline was not
certified at that alpha in that re-split (B2 at 2% in seeds 2 and 3), it is reported as "not certified", no curve.

(c) **Inputs and integrity.** `load_frozen()` first. `stage4_summary.json` is checked against the SHA-256 recorded in
Stage 4 Part B (e66e45f210ebae5e6d21a4d8247584e1affc2d9691a760427814c821b164d2cc). The certified threshold of every
(method, re-split, alpha) is read from it; nothing is re-certified.
  - The Stage 4 splits were not saved as files (Stage 4 regenerates them from the seeds). Stage 5 regenerates them with
    the same committed code: `src/r8_stage4.py` must match the SHA-256 recorded in stage4_summary.json
    (fae67390a488b44c57f1386c9b256cb0fbbcf2444282f5724c2580aa3e6445c2), and the stage stops otherwise.
  - Reproduction check: at every certified (method, re-split, alpha), the Test coverage recomputed from the
    regenerated split must equal the Stage 4 value exactly (it is also the right-hand end of the curve). Any mismatch
    stops the stage.
  - All other inputs are hash-checked as in Stage 4 (the Stage 3b pool loader and the train_meta.csv cross-check).

(d) **Summary numbers per curve (descriptive).**
  - The committed share at s = 25, 50 and 75 px.
  - **D90 size: the smallest s at which at least 90% of Test signs are committed (Test description, not a
    certificate).** s runs over the multiples of 5 px of (b). If a curve never reaches 90%, D90 is reported as
    "not reached".
  - Reported for the commit rule at every alpha where it was certified, and for B2, B3 and B4 at each alpha where both
    the commit rule and that baseline were certified in that re-split.
  - D90 is never described with any word suggesting a certified size; it is a Test description only.

(e) **Outputs.** `runs/20261003_stage1/poc2/stage5/stage5_summary.json` with the hashes of what it read. Stage 5 changes
no frozen value.

### Part B: results (fill from the printout after the run)
- Date: 2026-11-14 / Stage: R8 Stage 5 / Run tag: runs/20261003_stage1/poc2/stage5
- Git commit of Part A: 0a36313 | Git commit of the code: c5ffa9b (src/r8_stage5.py SHA-256
  e02d00a6b7e6659b07ada2020e479b88144c575172206ecaff7ebd49953ba5b2; src/r8_stage4.py matched fae67390...45c2)
- Output: stage5_summary.json SHA-256 3ec123ea567790997a9f81d58b9931d989d48aeb3cc49f252faa56277bee4c5b (1.0 s).
  stage4_summary.json matched e66e45f2...d2cc; every Stage 4 Test coverage reproduced exactly from the regenerated splits.
- Certificate B: not certifiable at n = 784 (frozen); gamma 10%. No joint certificate run; s and gamma not moved.
- Curves: committed share of the 262 Test tracks with ROI size at commit <= s, s = 5 to 185 px, at the Stage 4
  certified thresholds, for 2%, 3% and 5% (the alphas where the commit rule was certified in every re-split).
- D90 size (smallest s with >= 90% of Test signs committed; Test description, not a certificate), seeds 0 to 4:
  - 2%: commit 55, 60, 55, 55, 65 px; B2 not reached (seeds 0, 1, 4; not certified in seeds 2, 3); B3 not reached,
    165, 165, 170, 185 px; B4 165, not reached, 165, 165, not reached.
  - 3%: commit 45, 45, 45, 45, 50 px; B2 not reached in all five; B3 not reached, 165, 165, 170, 185 px;
    B4 130, 140, 135, 145, 165 px.
  - 5%: commit 45, 45, 45, 45, 50 px; B2 not reached in all five; B3 not reached, 165, 165, 170, 185 px;
    B4 125, 135, 130, 130, 135 px.
- Committed share at 25 / 50 / 75 px: commit rule 34.0 to 55.0% / 85.5 to 93.1% / 90.5 to 95.0%; B2 20.6 to 27.1% /
  62.6 to 72.9% / 74.1 to 79.4%; B3 0.8 to 2.7% / 12.2 to 17.2% / 38.6 to 49.2%; B4 1.2 to 2.7% / 12.2 to 19.5% /
  38.2 to 53.4% (ranges over re-splits and alphas where reported).
- Caveat 1: D90 must be read with coverage. The commit rule's Test coverage at 2% is only 91.2 to 93.5%, so D90 sits
  close to the coverage plateau; for seed 4 at 2% (coverage 91.22%) D90 is 65 px, and a slightly lower coverage would
  have made it "not reached". The same applies to B4 at 2% (not reached in seeds 1 and 4, coverage 89.31% and 87.79%).
- Caveat 2: at 5%, B4 covers more signs than the commit rule (B4 95.80 to 97.71%, commit rule 92.37 to 95.42%),
  consistent with plan section 9 ("do not claim more coverage than B4"). B4's D90 is still 125 to 135 px against the
  commit rule's 45 to 50 px, because B4 decides at the last frame.
- Structural note: B3 and B4 decide at the last frame by definition, so their curves trace the last-frame size
  distribution; the gap to the commit rule shows how much earlier it decides under a certificate, not a better
  classifier at equal size.
- Checks: frozen_grid_matches_stage1_grid, gate_flags_reproduce, pool_commit_table_reproduces, b1_top_equals_stored_top1,
  meta_roi_matches_stage1_sizes, b2_frames_all_pass_gate, stage4_coverages_reproduce, curve_right_end_equals_coverage
  all PASS — ALL PASS (checks): True.
- Decision: Certificate B reported as frozen ("not certifiable at n = 784"). The curves and D90 sizes are reported as
  Test descriptions, with both caveats. Nothing frozen changed.
- Deviation from the plan? No. (Splits regenerated rather than read from files, declared in Part A (c).)
- Next step: Stage 6 only after the author reviews Stage 5.
