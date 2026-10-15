# Stage 7: scenarios, Fit-only severity calibration, G2 freeze

Three commands, run in this order. Only the third one is irreversible.

| File | Goes to |
|---|---|
| `src/scenarios.py` | `src/` (new): the five corruptions, one strength axis, per-track weather, fully seeded. |
| `src/stage7.py` | `src/` (new): smoke test, calibration, freeze. |
| `tests/test_scenarios.py`, `tests/test_stage7.py` | `tests/` (new) |
| `LOG_stage7_entry.md` | Template. **Part A goes into `LOG.md` and is committed BEFORE the calibration run.** Part B after. |

Nothing existing is replaced. `LOG.md`, `README.md`, `configs/poc.yaml` and every Stage 1-6 file are untouched (no config edit is needed: the Stage 7 constants live in `src/stage7.py`, and a `stage7:` block in `poc.yaml` could override them, which you should not do).

## Install and test

```powershell
cd "D:\FYP Review\POC Implementation\stage0_package VI\stage0_pkg"
py -3.12 -m pip install albumentations==2.0.8 opencv-python-headless
py -3.12 -c "import albumentations, cv2; print(albumentations.__version__, cv2.__version__)"   # must print 2.0.8 and a cv2 version
py -3.12 -m pytest tests -q          # torch works: 139 passed (about 2.5 min) | torch blocked: 122 passed, 4 skipped
```

(`albumentations` also imports torch; `scenarios.py` hides torch from it when Windows blocks torch, so the smoke test works even then. OpenCV has its own DLLs, which I cannot test from here.)

If the import line fails with `simsimd` or `stringzilla`, run `py -3.12 -m pip install simsimd stringzilla` and try again. **If it fails with WinError 4551** (Smart App Control blocking another unsigned DLL, the same thing that blocked torch earlier): the smoke test needs only albumentations/OpenCV, the calibration needs torch too, so run both in Colab with the same `src/` and `runs/` folders (tell me and I will give you the Colab cells). Do not install `imagecorruptions`: it cannot be imported on a current numpy/scipy/setuptools (plan risk R7), which is why `fog` and `motion_blur` are vendored (see Deviations).

## Run

```powershell
git status                                                          # commit the Stage 7 code first
# 0. paste LOG_stage7_entry.md Part A into LOG.md, commit, put the hash in Part A.
py -3.12 -m src.stage7 --run runs/20261003_stage1 --smoke           # 1. seconds, no torch. Writes runs/20261003_stage1/stage7/contact_sheet.png
# LOOK at the sheet (see below). Then:
py -3.12 -m src.stage7 --run runs/20261003_stage1                   # 2. calibration, 15-40 min (mostly CPU corruption). Needs torch. Resumable.
# LOOK at chosen_levels_sheet.png and the printed table. Then, once and only once:
py -3.12 -m src.stage7 --run runs/20261003_stage1 --freeze          # 3. writes configs/frozen_g2.yaml. Refuses to overwrite.
git add configs/frozen_g2.yaml; git commit -m "G2 freeze"; git rev-parse HEAD     # this hash goes in LOG.md; G2 is closed
```

If the calibration crashes or the laptop sleeps, run the same command again: finished levels are read from `stage7/sweep_cache.jsonl` (the corruption is deterministic, so this changes nothing).

## What step 1 (smoke) shows

20 Fit images from 20 different classes, rows = clean + each scenario at strengths 1, 2.5 and the cap (rain 4, night 4, others 5), `contact_sheet.png`. It also re-renders every cell with the images in a different order and requires identical bytes (`smoke.json` has the hashes).

**Look for:** each row looks like its name; the sign is still recognisable to you at the cap. If a row is already unrecognisable at s = 2.5 or the cap looks destroyed, **tell me before step 2**: the plan's rule is to cap a scenario at the strongest human-recognisable level and log that, and that decision must be made before any accuracy is measured, not after. (I set the current caps from synthetic sign images, before touching your data; your real sheet is the first real look.)

## What step 2 (calibration) does

Fit tracks of reshuffle seed 0 only (261 tracks; every 3rd frame, about 2,600 frames). The images come from the cache with Fit rows only; Cert and Test pixels are never read (unit-tested). For each scenario and each level of the grid (0.1, 0.2, ..., 0.5, 0.75, 1.0, ..., up to the cap): corrupt, push through the frozen backbone and Gaussian head, record top-1 accuracy. **Drop = clean accuracy minus corrupted accuracy on the same frames, in percentage points**, with a track-bootstrap standard error for information.

Before measuring, it checks the pipeline: the backbone hash equals the frozen one; the clean Fit frames reproduce the Stage 2 embeddings (max difference < 1e-3) and the Stage 3 head labels (>= 99.9% agreement); corruption is bit-identical when frames are re-rendered in another order. Any failure stops the run.

Then the pre-registered rules (constants in `src/stage7.py`, same text as LOG Part A):

- **mild** = the level whose drop is closest to 5.5 pp among levels with a 3-8 pp drop; **severe** = closest to 22.5 pp among levels with a 15-30 pp drop. Ties go to the smaller strength.
- Nothing inside the band: mild takes the closest level overall; severe takes the strongest level allowed by the cap. Both are flagged `band_reached: false` and reported as such. No rule is changed afterwards.
- The two recertification cells = the two of the ten (5 scenarios x mild/severe) with the largest Fit drop. Ties: scenario order (rain, fog, night, side_view, motion_blur), then severe before mild.

It writes `stage7_results.json` (full sweep, choices, cells, versions, hashes) and `chosen_levels_sheet.png`. **Nothing is frozen yet.**

**Look for in the printed table and sheet:** (a) is the sign still recognisable at every chosen *severe* level (plan check), and *side view* at its *mild* level; (b) any `NO` in "inside band"; (c) any WARNING line about mild >= severe. If (a) fails for some scenario, that is a cap decision: send me the sheet, do not edit anything.

## What step 3 (freeze) does

Checks `stage7_results.json` exists, every scenario has mild < severe, and `src/scenarios.py` is byte-identical to the one used for the calibration. Then writes `configs/frozen_g2.yaml` (levels, drops, nominal parameters, the two cells, bands and selection rule, library versions, checkpoint hash, hashes of the results and of `scenarios.py`). It refuses to run if the file already exists. **Do not edit `scenarios.py` after step 2**: the freeze will refuse, and you would have to redo the calibration (which must then be logged as a change).

## Deviations from the plan (all go into LOG Part A)

1. **`fog` and `motion_blur` are vendored** from imagecorruptions 1.1.2 (Apache-2.0), as plan Stage 7 step 1 allows: `np.float_` became `float` and the global random state became an explicit generator. I verified the vendored functions return the same pixels as the original code (maximum absolute difference 0.0 at severities 1-5, run on a patched import). `imagecorruptions==1.1.2` itself is not installed.
2. **One continuous strength axis `s` in [0, 5]** with a grid, because the plan asks for a sweep over "a small range of parameter values" and imagecorruptions only takes the integers 1-5. For fog and motion blur the anchors s = 1..5 are the five imagecorruptions severities; the motion-blur radius and sigma are halved because our images are 112 px, not 224. For rain, night and side view the anchors are mine. They only shape the axis: the levels that get frozen are chosen from the Fit accuracy drops.
3. **Caps** rain = 4.0 and night = 4.0 (others 5.0), set from synthetic sign images before any accuracy was measured, because torrential rain at s = 5 destroyed the sign.
4. **Head accuracy, not softmax accuracy**, and **every 3rd frame**, to keep the sweep to minutes. The stride is recorded in the results and in the freeze file.
5. Night overlaps the brightness augmentation (already declared in Stage 1; brightness jitter was dropped, so there is no overlap in this checkpoint).

## What I expect (written before you run it)

[Likely] fog, night and motion blur reach the severe band inside their caps. [Guessing, ~50%] side view reaches 15 pp only near its cap, because GTSRB signs are not very sensitive to a 25-degree shear; rain is the other candidate for "band not reached" because its cap is 4. [Guessing] the two recertification cells will be severe levels of fog, night or motion blur. If none of this is true, that is information, not a problem.
