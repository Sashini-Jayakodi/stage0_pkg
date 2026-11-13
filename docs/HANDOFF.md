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
| 1 Accumulation and track table | Code written (`src/r8_stage1.py`, `tests/test_r8_stage1.py`, 12 tests pass on synthetic data). **Not yet run on real data.** Next: run it and check `ALL PASS: True`. |
| 2 Design phase and D-gate | Not started. Design set only. Stop at the D-gate; D1 and D2 must pass before F1. |
| 3 onward | Not started. |

Stage 1 run command: `py -3.12 -m src.r8_stage1 --run runs/20261003_stage1 --config configs/poc.yaml`
Expected on real data: frame head accuracy equal to POC-1 (about 92.6%); tau = 43.8435; track lengths {29: 1, 30: 1306}.
If C1a or C1b fails, stop: the head or the cache is not the POC-1 version.

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
