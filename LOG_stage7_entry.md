## Stage 7, PART A: PRE-REGISTRATION (paste into LOG.md and COMMIT before running the calibration; the smoke test measures no accuracy and may come before or after)

- **Date / stage / run tag:** ____ / Stage 7 (G2) / 20261003_stage1
- **Git commit of this pre-registration (and of the Stage 7 code):** b2723e47e7d3be102a0ad226d637e4a1d60ddbbd
- **Bands (already item 4 of the pre-registration, restated):** mild = 3-8 pp drop in Fit accuracy, severe = 15-30 pp. Metric: top-1 accuracy of the tied-Gaussian head on the frames of the Fit tracks of reshuffle seed 0 (261 tracks), clean minus corrupted, same frames, every 3rd frame (`frame_idx % 3 == 0`, about 2,600 frames). Cert and Test pixels are not read in this stage.
- **Strength axis and grid:** continuous severity s in [0, 5] (s = 0 identity). Grid 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 1.25, ..., 5.0 (23 levels). Caps (strongest level a human still recognises, set from synthetic sign images BEFORE any accuracy was measured): rain 4.0, night 4.0, fog 5.0, side_view 5.0, motion_blur 5.0.
- **Selection rule (fixed now, constants in `src/stage7.py`):**
  1. mild = the level with drop closest to 5.5 pp among levels with drop in [3, 8]; severe = closest to 22.5 pp among levels with drop in [15, 30]. Ties go to the smaller strength.
  2. No level in the band: mild takes the closest level overall; severe takes the strongest level allowed by the cap if no level reaches 15 pp, else the closest to 22.5 pp. Both are logged as "band not reached" (plan section 7: cap at the strongest human-recognisable level and log it). No rule is changed after seeing numbers.
  3. Recertification cells = the two of the ten (5 scenarios x mild/severe) with the largest Fit drop. Ties: scenario order rain, fog, night, side_view, motion_blur, then severe before mild.
  4. Human check on the contact sheets (plan checks): every chosen severe level and side view at mild still recognisable. Failing it means the scenario is capped and logged, not a rule change.
- **Determinism rule:** per-track parameters seeded by (track, scenario, severity); per-frame randomness by (track, frame, scenario, severity); SHA-256 text keys. Re-rendering in another order must give identical bytes (checked by the smoke test and at the start of the calibration).
- **Declared deviations from the plan:**
  1. `fog` and `motion_blur` vendored from imagecorruptions 1.1.2 (cannot be imported on the current stack, plan risk R7; plan step 1 allows vendoring). Edits: `np.float_` -> `float`, global RNG -> explicit generator. Checked numerically equal to the original at severities 1-5 (max abs difference 0.0). Motion-blur radius and sigma halved for 112 px images.
  2. Continuous severity axis; anchors for rain, night and side view are my choice (they only shape the axis; frozen levels come from Fit accuracy drops).
  3. Head accuracy (not softmax) and every 3rd frame, for speed.
  4. Pins: albumentations 2.0.8 (RandomRain for rain, Affine with anisotropic scale and shear for side view); imagecorruptions not installed.
- **Predictions written before running (do not edit afterwards):** [Likely] fog, night and motion blur reach the severe band inside their caps. [Guessing ~50%] side view and rain do not reach 15 pp, and are logged as "band not reached". ____ (add your own)

---
## Stage 7, PART B: results (fill in after the runs)

### Smoke test
- Command: `py -3.12 -m src.stage7 --run runs/20261003_stage1 --smoke` ; deterministic under re-ordering: ____ ; `smoke.json` SHA-256 ____
- Contact sheet read by a human: each row looks like its name ____ ; sign recognisable at the strongest level shown ____ ; anything that looked wrong, and what was decided BEFORE the calibration ____

### Calibration
- Command: `py -3.12 -m src.stage7 --run runs/20261003_stage1` ; device ____ ; torch ____ ; albumentations ____ ; opencv ____ ; numpy ____ ; wall time ____
- Code: `src/scenarios.py` SHA-256 ____ ; `src/stage7.py` ____ ; `py -3.12 -m pytest tests -q` gives ____ passed, ____ skipped (expected 139 passed with torch; 122 passed, 4 skipped with torch blocked).
- Pipeline checks (script asserts): checkpoint hash ____ ; clean Fit accuracy ____ % on ____ frames ; max embedding difference vs Stage 2 ____ ; label agreement with Stage 3 ____ ; determinism ____
- Output: `stage7_results.json` SHA-256 ____

| scenario | level | severity | Fit drop (pp) | se | inside band | note |
|---|---|---|---|---|---|---|
| rain | mild | | | | | |
| rain | severe | | | | | |
| fog | mild | | | | | |
| fog | severe | | | | | |
| night | mild | | | | | |
| night | severe | | | | | |
| side_view | mild | | | | | |
| side_view | severe | | | | | |
| motion_blur | mild | | | | | |
| motion_blur | severe | | | | | |

- **Bands not reached:** ____ (logged, not fixed)
- **Recertification cells (largest Fit increase):** 1. ____ ; 2. ____
- **Human check on `chosen_levels_sheet.png`:** every severe level still recognisable ____ ; side view at mild recognisable ____ ; scenarios capped because of it ____

### Freeze
- Command: `py -3.12 -m src.stage7 --run runs/20261003_stage1 --freeze`
- `configs/frozen_g2.yaml` SHA-256 ____ ; **commit** ____ (`git rev-parse HEAD` after committing it). **G2 is closed from this commit.**
- No Cert or Test scenario number was computed before this commit: ____ (true by construction: nothing in Stage 7 reads them)

**Checks**
- [ ] tests pass
- [ ] contact sheet looks like the named effects
- [ ] side view keeps the sign recognisable at the mild level
- [ ] re-run gives identical hashes
- [ ] `frozen_g2.yaml` committed, hash recorded

- **Decision taken and why:** ____
- **Deviation from R6?** Only the declared ones in Part A; ____
- **Next step:** Stage 8 (latency micro-benchmark), then the proposal framing.
