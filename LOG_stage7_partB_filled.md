## Stage 7, PART B: results (filled from the run of 2026-10-15; blanks marked ____ are for you)

- **Run tag:** 20261003_stage1 ; **date:** 2026-10-15 ; device cuda (RTX 5050) ; torch ____ ; albumentations 2.0.8 ; opencv ____ ; numpy ____ ; wall time ____
- **Commits:** code `967f9cf` ; pre-registration `b2723e4` ; fog cap 5 -> 4 `46a934f` (after the smoke sheet, before any accuracy) ; amendment 2 (rain axis) `e584f0a` (after run 1, Fit only) ; **G2 freeze `00e1664a6b66cd10869490309a84d586ffdf66f8`**
- **Hashes:** `configs/frozen_g2.yaml` SHA-256 B6C11EBADD66E3D89EBE7EA98647D3F12C1F61A96BDB98F956ACD9F0F959C667 ; `stage7_results.json` (run 2) 390ACF7069B6B77B12E6CF550073886664ADED1491AFA947B736753868BBCB11 ; `src/scenarios.py` ____ ; `src/stage7.py` ____ (`Get-FileHash`)
- **Tests:** Stage 7 files 43 passed after amendment 2 ; full suite `py -3.12 -m pytest tests -q`: ____ (expected 140 passed with torch)

### Smoke test
- Deterministic under re-ordering: yes (script asserts; 15 cells). Contact sheet: every row looks like its name. Fog at 5 was not recognisable (30, 100 and 80 signs gone) -> fog cap lowered to 4 BEFORE any accuracy was measured (commit 46a934f). Other caps unchanged at that point.

### Calibration (Fit seed 0, 261 tracks, 2,610 frames at stride 3)
- **Pipeline checks (script asserts):** checkpoint hash OK ; clean Fit accuracy 93.10% ; max embedding difference vs Stage 2 8.2e-05 ; label agreement with Stage 3 100.00% ; determinism re-check at every scenario's cap: identical.
- **Run 1 (kept in `runs/20261003_stage1/stage7_run1_rain_v1/`):** rain unusable (drop 65 pp at s = 0.1, 31 pp at s = 1, non-monotone; mild = severe = s 1.0) -> amendment 2. The other four scenarios were identical in run 2 (same drops to the last digit; only bootstrap standard errors differ because the bootstrap stream depends on call order).
- **Run 2 (final, frozen):**

| scenario | level | severity | Fit drop (pp) | se | inside band |
|---|---|---|---|---|---|
| rain | mild | 1.25 | 4.98 | 0.87 | yes |
| rain | severe | 3.0 | 23.03 | 1.76 | yes |
| fog | mild | 0.3 | 5.79 | 1.07 | yes |
| fog | severe | 0.75 | 19.85 | 1.78 | yes |
| night | mild | 0.75 | 6.55 | 1.17 | yes |
| night | severe | 1.75 | 20.50 | 2.04 | yes |
| side_view | mild | 2.25 | 5.82 | 0.85 | yes |
| side_view | severe | 4.0 | 21.88 | 1.48 | yes |
| motion_blur | mild | 4.0 | 6.78 | 1.16 | yes |
| motion_blur | severe | 5.0 | 10.54 | 1.32 | **NO** (no level reaches 15 pp; strongest level allowed by the cap, flagged) |

- **Band not reached:** motion_blur severe (10.54 pp at the cap s = 5.0, i.e. imagecorruptions severity 5 with kernel constants halved for 112 px). Logged, not fixed; changing the axis a second time would look like tuning.
- **Recertification cells (pre-registered rule: two largest Fit drops):** 1. rain / severe (s = 3.0, +23.03 pp) ; 2. side_view / severe (s = 4.0, +21.88 pp). **Caveat for the paper:** the four severe levels (fog 19.9, night 20.5, side view 21.9, rain 23.0) are within about one standard error of each other (se 1.5-2 pp), and each was selected as the grid point nearest 22.5 pp, so the choice of these two cells is a pre-registered tie-break, not evidence that rain and side view are the hardest scenarios. Do not claim that.
- **Human check on `chosen_levels_sheet.png`:** every severe level still recognisable ; side view at mild recognisable ; no scenario capped because of it. Observation: rain, night and fog severe look benign to a human but cost about 20 pp, i.e. the backbone is fragile to them (the dark signs, already dim when clean, are the first to go at fog 0.75).

### Freeze
- `py -3.12 -m src.stage7 --run runs/20261003_stage1 --freeze` -> `configs/frozen_g2.yaml` committed as `00e1664`. **G2 is closed from that commit.** No Cert or Test scenario number existed before it (Stage 7 never reads them).

**Checks**
- [x] tests pass
- [x] contact sheet looks like the named effects
- [x] side view keeps the sign recognisable at the mild level
- [x] re-run gives identical hashes (four unchanged scenarios reproduced exactly)
- [x] `frozen_g2.yaml` committed, hash recorded

- **Decision taken and why:** G2 frozen with the rain axis v2 and fog cap 4, per amendments 1-2 ; motion blur severe band not reached, logged.
- **Deviation from R6?** The declared ones in Part A plus amendments (fog cap, rain axis v2).
- **Next step:** Stage 8 (latency micro-benchmark), then the proposal framing.
