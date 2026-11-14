# Handoff notes, project "FYP Visual Distance" (13 Nov 2026)

## Start here (POC-3)
- **Branch `poc3`** (tracks `origin/poc3`). POC-2 is finished and tagged `poc2-final` (`9a8d08c`); do not commit to
  `poc2`. Never touch the remote branch `origin/POC_2` (an older branch, at `2c9b74a`). Last commits: `8c6c902` (plan R9),
  `f5c49a8` (Stage 0 setup), `c0ffaaf` (handoff tidy), `5d0162e` (project rules and POC-3 file naming), `daf03b1`
  (wording rule narrowed; handoff and gitignore wording), `3fbb90f` (handoff working method section); Stage 2:
  `6043706`, `c70f15d`, `2f18f95`, `5716fc6`, `f19727c` and the freeze commit `99da0b5`; Stage 3: `9be85ac`,
  `aee1565`, `05c1df5`, `58cacfe`, `0c255f0`, `27ed9a7`, `515dcb7`; Stage 4: `1d9e00b` (Part A), `beeb114` (code) and
  the Part B commit.
- **Stages 0 to 4 done; the engine `src/r9_certify.py` is frozen.** Next step: **POC-3 Stage 5 (BelgiumTSC design
  phase and freeze F3)**, only when the author says so. Read plan R9 sections 2 and 3 first. Full suite at Stage 4: 435
  passed (one intermittent scikit-learn ImportError in a POC-1 test on the first run). BelgiumTSC testing labels are
  read only through `src/r9_btsc_guard.py` until F3. The POC-3 status table is below the POC-2 one.
- **Environment (checked 14 Nov 2026):** `py -3.12` (Python 3.12.10) with torch 2.14.1+cu132 (CUDA available, RTX 5050),
  numpy 2.5.2, scipy 1.18.1, scikit-learn 1.9.1 (imports fine again). Full test suite: `py -3.12 -m pytest tests -q
  -p no:cacheprovider`, 267 passed in about 70 s at `poc2-final`. Both PowerShell and Git Bash work; Git Bash has
  `pdftotext`/`pdfinfo`. PDFs are read as text extracts only (pdftoppm is not installed).
- **Data and artefacts:** POC-1 artefacts under `runs/20261003_stage1/` (list further down); image cache and
  `train_meta.csv` in `data/cache/` (`train_meta.csv` is tracked in git but has no recorded hash: POC-2 cross-checked its
  ROI boxes against the Stage 1 sizes instead). POC-2 outputs in `runs/20261003_stage1/poc2/stage1..6/`, each with a
  `*_summary.json` whose SHA-256 is in that stage's Part B. POC-3 outputs go in `runs/20261003_stage1/poc3/` and
  `runs/btsc/`.
- **Papers:** `docs/papers/` holds the Ringel et al. 2024 and Boursinos and Koutsoukos 2022 PDFs. Git-ignored (not
  redistributed); citations, source URLs and SHA-256 in `LOG_r8_stage6_entry.md`. On a fresh clone they must be
  downloaded again and hash-checked.
- **Reusable POC-2 code (import, never edit):** `src/r8_frozen.py` (freeze-guard pattern), `src/r8_stage2.py`
  (`open_npz` pool-file guard, design loader), `src/r8_stage3_engine.py` (vectorised Learn-then-Test, exact hand
  p-values), `src/r8_stage3_synth.py` (synthetic model with latent track difficulty, the base for R9 Stage 2),
  `src/r8_stage4.py` (Cert/Test splits; split membership is NOT saved: regenerate with `make_split` and check the
  Stage 4 code hash and coverages, as `src/r8_stage5.py` does), `src/r8_stage6_methods.py` (B5, B5c, B6 = R9's S2).
- **Gotchas:**
  - Windows line endings: git prints LF -> CRLF warnings. A file's SHA-256 is of its bytes on this machine; when a hash
    must be stable across machines, also record the git blob ID (as done for the R8 and R9 plans and frozen_r8.yaml).
  - POC-3 freeze rule: frozen POC-3 files (engine, engine tests, plan R9; `configs/frozen_r9_btsc.yaml` from Stage 5)
    are checked at run time by git blob ID through `src/r9_frozen.py` (added in Stage 3), not by SHA-256, so a
    line-ending-only difference does not stop a run and any other change does. Every r9 script that uses the engine
    calls it first. The recorded blob IDs are in `LOG.md`; SHA-256 values stay recorded for the bytes on this machine.
  - In POC-2, B6 was calibrated on Cert thirds (Stage 6 Amendment 1). Plan R9 section 5.2 says S2 is calibrated
    on the design set (two halves) so all Cert signs stay available for the certificate. This is a different rule; follow
    R9 and state the difference.
  - Downloads need the author's approval first, go into a new empty folder, and have their source URL and SHA-256
    recorded.
  - The shared local tool settings keep commit messages and pull requests free of trailer lines; personal local
    settings are git-ignored.

## Where the project stands
- POC-1 (Revision 6, confusability-stratified certification) is finished. Result: two of three go/no-go criteria (G1)
  failed on clean data, because the hard group was not riskier than the easy group. The certifier itself was shown
  valid: on synthetic cluster data, frame-level certification broke (8.7% violations vs a 5% budget) and track-level
  had 0%. Measured within-track correlation: rho = 0.446, design effect 13.9. Frame head accuracy is 92.6%. There are
  3,176 frame errors spread over 438 of the 1,307 tracks.
- The project then moved to "Certify the Sign, Not the Frame" (proposal R8). The build plan is
  `docs/poc2-implementation-plan-r8.md`, a copy of the project planning doc. It decides everything: stages, checks,
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

## POC-3 (plan R9) stage status
Plan: `docs/poc3-implementation-plan-r9.md` (SHA-256 `c0639973c70f35ac5cda1f11c49bf50a1639c632c48ccec8889d60a5ded03db4`,
committed in `8c6c902` on branch `poc3`). POC-2 is tagged `poc2-final` (`9a8d08c`); the POC-2 table above is the
finished record and is not updated further.

| Stage | Status |
|---|---|
| 0 Repo setup | **Done 14 Nov 2026.** `poc2-final` tagged and pushed; branch `poc3` created; plan R9 committed (`8c6c902`) and its SHA-256 recorded in `LOG.md`; POC-3 block added to the project rules file; local tool settings keep commit messages free of trailer lines. |
| 1 Data audit (F-gates) | **Done 14 Nov 2026** (Part A `9a228a1`, Amendments `5dc5032`, `6f05a92`; code `262c832`). **F-A1 PASS** (663 testing poles after removal, minimum 600), **F-A2 PASS after removal** (65 shared (pole, class) units removed), **F-A3 FAIL** (DITS page unreachable). BelgiumTSC zips in `data/btsc_raw/` (git-ignored; URLs and SHA-256 in `runs/btsc/stage1/downloads_manifest.json`). Testing release: 53 of 62 classes; 783 sign units after removal; 16 / 14 duplicated GT rows (count views by unique file name). Projected n_c about 279-329, so alpha 1% tolerates 0 errors. Details in `LOG_r9_stage1_entry.md` Part B. |
| 2 Certification engine v2 (V-checks) | **Done 14 Nov 2026; engine frozen** (Part A `c70f15d`, Amendments `5716fc6`, `f19727c`; final run on `f19727c`). **V1 to V4 all PASS** in all three runs: V1 40/40 plan section 6 values; V2 36/36 settings (2 error levels P_FRAME 7.4% and 5% x rho 0/0.3/0.45 x fixed-3 / view-mix / fixed-30 x alpha 1%/2%, 2,000 runs each), largest sign-level violation rate 5.35% [4.40, 6.43]%; V3 rel. error 6.9e-15; V4 PASS. Frozen: `src/r9_certify.py` SHA-256 `d2fd6fd6...8075` (blob `158a9c9e`), `tests/test_r9_certify.py` SHA-256 `3a163ab5...7813` (blob `ecb6e044`); full values in `LOG.md`. HB comparator: primary start = Amendment 2 (zero-error pass at the lower 90% CP bound of the design commit fraction); Stages 3 and 6 report HB under all three starts (`hb`, `hb_a1`, `hb_bs`). Reported: exact binomial certified at least as often as HB under every start in all 36 settings; frame-level B1 up to 37.45% violations (30 frames), 15.35% (3 views). Engine API: `certify(k, n_c, n, alpha, f_design, test, n_design=...)`; HB rules need `n_design` (design units, or design frames for B1). Full suite 327 passed. Details in `LOG_r9_stage2_entry.md` Part B. |
| 3 GTSRB exploratory run of the layer | **Done 14 Nov 2026; exploratory (post-hoc)** (Part A `9be85ac`, design values `58cacfe`, Amendment 1 `0c255f0`; code `aee1565`, `05c1df5`, `27ed9a7`). All integrity checks PASS (freeze by blob ID, 68 Stage 4 coverages reproduce, POC-2 pool checks). Exact binomial certifies 1% for every score (S1, S2-sum primary, S2-min, S3, S4) in 5/5 re-splits; 0.5% has no exact-binomial start at n = 784 (start rule, Amendment 1). At 1%: coverage S1 81.3%, S2-sum 90.8%, S2-min 89.3%, S3 85.9%, S4 56.5%; median commit frame 17 / 5 / 5 / 21 / 2. HB (primary start) certifies 1% in 5/5/2/1/0; exact binomial certifies wherever HB does, at the same or a looser point. Six certified Test cells above alpha, all with lower 95% CP bound below alpha. The 1% result is confounded with the changed test, grid and start vs POC-2. Freeze guard `src/r9_frozen.py` (blob IDs) is called first by every r9 script that uses the engine. Details in `LOG_r9_stage3_entry.md` Part B. |
| 4 BelgiumTSC model and cache | **Done 14 Nov 2026** (Part A `1d9e00b`, code `beeb114`). **V4.1 to V4.6 all PASS.** One view per image (class folder, file name); 65 Stage 1 units removed; testing split by pole (rarest-class strata, largest remainder, seed 0): design 133 poles / 157 units / 465 views / 40 classes, pool 530 / 626 / 1,860 / 53; 13 small classes in the pool only. 62-class ResNet-18, POC-1 recipe, whole images at 112 px, early stop on 10% of training poles: 18 epochs, best 13, validation 98.76%; checkpoint SHA-256 `83fdd0d1...b76b`. Head on the training part (shrinkage 0.0791). Design-pole accuracy (no threshold): 96.99% per view, 95.54% per unit at the last view (7 of 157 wrong). Pool: row counts only. Outputs in `runs/btsc/stage4/` (caches, embeddings, scores, checkpoint git-ignored; SHA-256 in `stage4_summary.json` `1c66a011...3a31` and Part B). Pre-F3 guard `src/r9_btsc_guard.py`. Details in `LOG_r9_stage4_entry.md` Part B. |
| 5 BelgiumTSC design phase and freeze F3 | Not started. |
| 6 BelgiumTSC confirmation | Not started. |
| 7 DITS earliness (only if F-A3 passes) | **Dropped** (F-A3 failed, plan section 11). Earliness evidence is GTSRB only, labelled exploratory. |
| 8 Final tables | Not started. |
| 9 Idea text and gates | Not started. |

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

### scikit-learn note (resolved)
- During POC-2 Stage 1 (13 Nov 2026), `tests/test_stage3.py::test_ledoit_wolf_matches_sklearn` failed because Windows
  Application Control blocked a scikit-learn 1.9.1 DLL (`_middle_term_computer`).
- Since 14 Nov 2026 it passes again on the same scikit-learn 1.9.1, which points to the Application Control policy,
  not a package upgrade (recorded in `LOG_r8_stage3_entry.md` Part B; the Stage 1 entry is left unchanged). Full suite
  at `poc2-final`: 267 passed, 0 failed.
- POC-2 code never imports sklearn. If POC-3 code needs it, check the import first; if the block returns, log it as an
  environment change. Do not edit the POC-1 test file (hard rule 1).

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

## Working method (as practised through POC-2; applied to POC-3)
- I never change a frozen value or rule to get a better result. Negative results and failed checks are logged in
  `LOG.md` as they came out.
- Order inside a stage:
  1. I write `LOG_r9_stageN_entry.md` Part A (every definition and open choice) and commit it **before** any number of
     that stage exists. Every choice the plan leaves open is decided and written into Part A before that commit.
  2. Values computed from design data go into Part A in a separate commit, with the script committed first.
  3. Code and synthetic-data tests (brute-force references, tamper tests); full suite; commit.
  4. I run on real data and keep the **whole** printout; then I stop.
  5. I fill Part B only after reviewing the whole printout; I commit it with the stage's `*_summary.json`, append the
     entry to `LOG.md` and update this file.
- Any change to a committed Part A before numbers exist is a dated "Amendment N" appended to the entry, never an edit.
- I review the diff before every commit. A commit stages only the files of that step and has no trailer lines.
- Every sentence written for the record is checked against the facts before it is committed.
- I stop after each step. The next stage, a real-data run, a push, a tag or a download each starts only as a separate
  decision.
