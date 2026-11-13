# Handoff from the claude.ai project "FYP Visual Distance" (13 Nov 2026)

## Where the project stands
- POC-1 (Revision 6, confusability-stratified certification) is finished. Result: two of three go/no-go criteria (G1)
  failed on clean data, because the hard group was not riskier than the easy group. The certifier itself was shown
  valid: on synthetic cluster data, frame-level certification broke (8.7% violations vs a 5% budget) and track-level
  had 0%. Measured within-track correlation: rho = 0.446, design effect 13.9. Frame head accuracy is 92.6%. There are
  3,176 frame errors spread over 438 of the 1,307 tracks.
- The project then moved to "Certify the Sign, Not the Frame" (proposal R8). The build plan is
  `docs/poc2-implementation-plan-r8.md`, a copy of the claude.ai project doc. It decides everything: stages, checks,
  gates, fallbacks.
- Plan decisions already taken: drop the softmax and fixed-K baselines; 6 adverse cells instead of 10; BelgiumTSC
  as the second dataset (RQ1 only).
- The end goal of POC-2 is the comparison table in plan section 9: what this method does better than B1 to B6.

## Stage status
| Stage | Status |
|---|---|
| 1 Accumulation and track table | **Done (13 Nov 2026). ALL PASS: True** (C1a to C6). Run on real data; entry appended to `LOG.md`. Outputs in `runs/20261003_stage1/poc2/stage1/`. |
| 2 Design phase and D-gate | Run 13 Nov 2026 (code `fe62b23`, definitions `f19cbfb` + `ccc400a`). D1 FAIL, D2 PASS (V1, alpha 1%). Certificate B: no s certifiable at gamma 10%. Details in `LOG_r8_stage2_entry.md` Part B. |
| F1 freeze | **Done 14 Nov 2026.** Commit `3de4cb99cfb0658e9d659bb1e31171e3187ff87c`. `configs/frozen_r8.yaml` SHA-256 `4f7a79156a4486b6b5f97515457513e6df2a2bfeed2a41b6fa05936bd1141d24`. Pre-registration block in `LOG.md`. Nothing frozen changes from here on; the plan file is frozen too (SHA-256 `b0ad6292...174bd2`). |
| 3 Validity tests | **Done 14 Nov 2026** (code `a044722`, Part B `263f3ec`). **P1 FAIL**: track_part_3b PASS, frame_part_3b PASS, track_part_3a PASS, frame_part_3a FAIL. 3a frame-part fail, as measured: B1's rule (h) found no usable threshold in most synthetic runs (629/1000 at rho 0.45), so its violation rate was capped by its 4.9% certifying share; when it did certify it violated in 48 of 49 runs (rho 0.45) and 17 of 19 (rho 0.3). Real data (3b, alpha 1%): B1 violation 20.0% [18.3, 21.8]% vs delta 10%; track-level 1.4% [0.9, 2.0]% (H1 supported on real data). For Stage 4: at 1% the track-level certifier certified in only 19.8% of 3b draws (pool mean loss at the start threshold 0.503% vs 0.11% on the design set); at 2% in 95.7% with 0 violations. P2 at 1% is at risk; reported with the certifiability curve (plan section 11); the primary alpha stays 1%. |
| 4 Certificate A, 5 re-splits | **Done 14 Nov 2026** (code `e31dff2`, Part B `1428fcc`). **P2, P3, P4 all FAIL (0/5)**: the commit rule was not certified at the primary alpha 1% in any re-split, while B3 (last frame) and B4 (majority vote) were certified at 1% in all five. Certifiability curve (n = 784): commit rule 2% in all five; B2 2-3%; B3 and B4 1%. At 2% the commit rule covers 91.2-93.5% of Test signs (error 0.41-1.22%) at median frame 10-11 and 26-28 px; B3 ~89-91% and B4 ~88-91% at frame 30 and 70-81 px; B2 81-82% where certified (not certified in seeds 2, 3). Amendment 2 (descriptive): commit rule ahead of B2 by 10-14 points at frames 8-11 in every re-split. Limitation (not changed): the frozen order starts at working position 18 and only relaxes, and the grid's strictest point still commits at median design frame 18, so the rule never approaches last-frame behaviour. |
| 5 Certificate B and commit-by-size curve | **Done 14 Nov 2026** (code `c5ffa9b`, Part B `f305454`). Certificate B reported as frozen: "not certifiable at n = 784", gamma 10%, no joint certificate run. Descriptive Test curves at the Stage 4 certified thresholds (2%, 3%, 5%). D90 size (smallest s with >= 90% of Test signs committed; Test description, not a certificate): commit rule 55-65 px at 2%, 45-50 px at 3% and 5%; B4 125-165 px or not reached; B3 165-185 px or not reached; B2 never reaches 90%. At 25 px the commit rule has committed 34-55% of Test signs vs 1-3% for B3/B4. Caveats: D90 must be read with coverage (commit-rule coverage at 2% only 91.2-93.5%); at 5% B4 covers more signs than the commit rule (95.8-97.7% vs 92.4-95.4%). B3/B4 decide at the last frame by definition, so the gap measures earliness, not better classification at equal size. |
| 6 Published baselines and comparison table | **Done 14 Nov 2026** (code `167557b`, Part B `87cfd70`). B5 = Ringel marginal, B5c = Ringel conditional (declared extra), B6 = Boursinos and Koutsoukos (ECDF sum/min, K = 5; K = 9 declared extra); papers in `docs/papers/` (git-ignored; citations, URLs, SHA-256 in `LOG_r8_stage6_entry.md`). **B6 exceeds the commit rule on coverage and earliness at both alphas** (1%: commit rule 0/5 certified vs B6 ~92.7% coverage at frame 5; 2%: B6 ~95.8% at frame 5 vs commit rule 92.7% at frame 10), but has no certificate: its Test error exceeded its target r in 1-3 of 5 re-splits per variant (up to 2.78% at r = 1%, 5.58% at r = 2%), while the commit rule stayed at 0.41-1.22% where certified. B5/B5c keep their gap small but their real error is 2.3-6.5%. Stage 6 checks: B5 own target FAIL (3/5), B5c PASS (5/5), B6 calibration direction PASS (5/5). Do not read B1's "0.5% certifiable" (17% violations in 3b) or the 0% row-2 values at 2% as validity evidence. |
| **POC-2 core** | **Stages 1-6 complete.** The comparison table of plan section 9 is in `LOG_r8_stage6_entry.md` Part B and `stage6_summary.json`. |
| 7 onward | Not started. Every confirm-phase script calls `load_frozen()` (`src/r8_frozen.py`) first. Stage 7 must follow the BelgiumTSC alpha rule below. |

**Stage 7 (BelgiumTSC) note:** BelgiumTSC's primary alpha is chosen by the Stage 2 D2 rule on BelgiumTSC's own design poles, at its own Cert size, and frozen before any BelgiumTSC Cert or Test pole is read; no GTSRB pool result is used.

Stage 2 status: checks not all passed (D1 FAIL, D2 PASS). Hard rule 5 and plan section 8 require all checks to pass; the author overrides this for D1 only, for the reasons in the Decision above, before any Cert or Test track is touched. Hard rule 5 itself is unchanged.
(The "Decision above" is the Decision line in `LOG_r8_stage2_entry.md` Part B.)

Stage 1 run command: `py -3.12 -m src.r8_stage1 --run runs/20261003_stage1 --config configs/poc.yaml`

### Stage 1 key results (from `stage1_summary.json`)
- Frame head accuracy 92.599%, equal to POC-1 (0.9259863806779056). tau = 43.8435.
- Gate: 4.836% of frames fail; 5 tracks have no passing frame (rejected as unknown).
- Track lengths {29: 1, 30: 1306}. ROI size vs frame index: median per-track Spearman 0.998, positive in 99.8% of tracks.
- Commit table: 200-point lambda grid per variant (V1, V2, V3), grid built from design-set margins only.
- Output SHA-256: design `ad6fa0c5...1f2b391`, pool `0afb7e89...2250db`, frame_evidence `f79033e8...a0e960`
  (full hashes in `stage1_summary.json` -> files_sha256). Stage 2 must check them before reading.
- Environment frozen in `env_poc2_stage1.txt` (Python 3.12.10, numpy 2.5.2, scipy 1.18.1, scikit-learn 1.9.1,
  torch 2.14.1+cu132).

### scikit-learn note (open environment issue)
- `tests/test_stage3.py::test_ledoit_wolf_matches_sklearn` now fails: Windows Application Control blocks a
  scikit-learn 1.9.1 DLL (`_middle_term_computer`). It passed in POC-1 (LOG.md Stage 4, 41 passed, 0 skipped).
  Full suite before the Stage 1 run: 169 passed, 1 failed.
- POC-1 never recorded its scikit-learn version, so it is unknown whether this comes from a package upgrade or a
  policy change.
- Stage 1 is unaffected (it does not refit the head or import sklearn). **Any POC-2 code must not import sklearn**
  while this block stands; if a later stage needs it, solve the block first and log it as an environment change.
  Do not edit the POC-1 test file (hard rule 1).

## Deviations already decided (logged in LOG_r8_stage1_entry.md)
1. Stage 1 writes the commit table as two files. `track_table_design.npz` covers the 261 design tracks, with
   `y_track` and `{V}_wrong`. `track_table_pool.npz` covers the 1,046 pool tracks and has no labels and no wrong.
   Why: the plan text asked for wrong on all tracks, which would break plan section 2. Pool errors are first computed
   in Stage 4, after F1.
2. ROI size = min(roi_x2 - roi_x1, roi_y2 - roi_y1) in original pixels (the plan names no convention). It is frozen
   with s at F1.
3. Before the first passing frame, a track has no evidence (S = NaN, M = -inf), so nothing can commit on the prior alone.

## POC-1 artefacts POC-2 reads (all under runs/20261003_stage1/ unless noted)
- `embeddings/train_release.npz`: emb (39209 x 512, l2-normalised), y, track (codes 0..1306), track_names, frame.
  Hash in `embeddings/embeddings_summary.json`, which also holds checkpoint_sha256 (starts `8652735`, ends `0bcc`).
- `embeddings/ft_train.npz`: the fine-tune frames (official test release).
- `head_emb/head.npz`: mu (43 x 512), W (whitening; Mahalanobis = Euclidean after `X @ W`), prec, priors, log_eps.
  `src.head.mahal_sq(head, X)` gives squared distances; `src.head.score` gives top3, lp3, margin, and s = sqrt(min D2),
  the gate score.
- `head_emb/scores_train_release.npz`: stored top3 (and s). Hashes in `head_emb/head_summary.json`, which also has
  `train_release.head_accuracy`.
- `splits/seed{0..4}.npz`: track_split (0 Fit, 1 Cert, 2 Test) and track_names. Hashes and the gate tau per seed are
  in `splits/stage4_summary.json` (`seeds[i].gate.tau`). The seed-0 Fit split is the R8 design set.
- `data/cache/train_meta.csv` (path from `configs/poc.yaml` -> data.cache_dir): one row per frame, in the same order
  as the embeddings. Columns: class_id, track_id, frame_idx, width, height, roi_x1, roi_y1, roi_x2, roi_y2, filename.
  `train_112.npy` is the image cache.
- `src/certify.py`: `hb_pvalue(mu_hat, n, alpha)` (Hoeffding-Bentkus for H0: E[L] >= alpha, L in [0,1]) and
  `floor_alpha`. Reuse `hb_pvalue` for Certificates A and B; it fits L_i = wrong_i - alpha*committed_i + alpha.
  `certify_sequence` is frame-based and is NOT the Certificate A engine (that engine needs per-track,
  per-lambda losses).
- `src/certify_synth.py`: the POC-1 synthetic validator (Stage 3a extends it in a new file).
- `configs/frozen_g2.yaml` (commit `00e1664`): frozen scenario severities. Scenario generator and latency benchmark
  are in POC-1 `src/`.
- `src/train.py`: `sha256_file`.
- POC-2 outputs go in `runs/20261003_stage1/poc2/stageN/`.

## Working agreements with the user
- The user is not strong in statistics. Explain results in plain words before the maths.
- Never suggest changing a frozen value or rule to get a better result. Negative results are reported as they are.
- Treat each stage as: build, then synthetic tests, then the user runs it on the laptop, then the user pastes the
  printout, then fill the LOG entry.
