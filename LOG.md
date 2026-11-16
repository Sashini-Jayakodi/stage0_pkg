# LOG.md (decision and result log, plan section 12)

Rule: one entry per stage or decision. Commit this file after every entry.

---

## PRE-REGISTRATION (DRAFT: read, edit, then commit BEFORE running Stage 5; paste the commit hash)

Items 1-5 are the plan's recommendations. You must read and accept (or change) each one before committing;
changing them after you see Stage 5 numbers is the thing a reviewer can legitimately attack.
Items 6-7 were added on 2026-10-10 after the Stage 4 Fit numbers were seen (disclosed in item 7).

1. G1 aggregation over 5 reshuffles

   * G1(i) hard-minus-easy gap in base error, where "error" is the PER-TRACK MEAN base error (item 6), track-cluster bootstrap 95% lower bound > 0: passes if true in >= 4 of 5

   * G1(ii) Cert holds >= 30 hard-group frame errors over >= 10 distinct tracks: passes if the MEDIAN over 5 meets both

   * G1(iii) some alpha <= 3% certifiable in BOTH strata: passes if true in >= 4 of 5. Only the absolute grid values 1%, 2%, 3% count (clarification; the grid also holds 5% and the exploratory 0.5 x base alphas, which do not count)

2. Bootstrap: B = 2000, lower bound = 2.5th percentile

3. First fallback if G1(i) is weak: top-3 epsilon (eps3 = eps(y1,y2) + eps(y1,y3)). "Weak" = G1(i) fails by the rule in item 1 (clarification, no new threshold). eps stays the primary result either way; eps and eps3 are both reported in Stage 5.

4. G2 severity bands (measured on Fit only): mild = 3-8 pp accuracy drop, severe = 15-30 pp

5. Certification ordering (plan 6.3): order thresholds by Fit-estimated E[L], start at the Fit argmin, relax looser; this replaces "strict to loose" in R6

6. Base-error definition (decided 2026-10-10): per stratum, the base error is the mean over tracks that have >= 1 gated-in frame in the stratum of that track's in-stratum error rate (no margin threshold). It is used for G1(i) and for alpha = 0.5 x base. The pooled rate (frames weighted 1/n_t) is reported as a sensitivity row. Reason: the Stage 6 certificate averages per track. The c* rule is unchanged (gated-in Fit frames, weight 1/n_t, easy-side mass closest to 50%).

7. Disclosure of what was seen before this commit: the Stage 4 Fit-only results (no Cert or Test error has been computed). The eps hard-minus-easy gap on Fit is small (mean +0.51 pp per-track, +0.44 pp pooled; 4 of 5 seeds positive under both definitions); the eps3 gap is larger (+1.08 pp per-track). c*, delta = 0.1, beta = 5%, strata, reshuffles (seeds 0-4) and severities were NOT changed after seeing these numbers, and the choice in item 6 was made with both definitions showing the same sign count for eps.

Frozen at git commit: 45a4d41ab9e8d3b1f877555f58c4fd991804cdfe (2026-10-10; the pre-registration text as it stands in that commit is the frozen version. Any later change must be a dated addendum, committed before Stage 5.)

ADDENDA (dated, added AFTER the freeze commit above and committed before the diagnostic below was run and before Stage 5; the frozen text above is unchanged)

* Addendum A (2026-10-10): disagreement rule for item 6. If G1(i) passes under the per-track mean but fails under the pooled rate (or the reverse), both results are reported with that caveat. The per-track result stays the pre-registered primary.

* Addendum B (2026-10-10): the Fit-only diagnostic (src/fitdiag.py: track-cluster bootstrap of the hard-minus-easy gap on Fit, and error by confusability slice on Fit) is information only. It uses Fit frames only. No frozen rule (c*, delta, beta, strata, seeds, base-error definition, G1 aggregation) will be changed because of its output. Any different design would be a separate, declared revision (R7), written in this file and committed BEFORE Stage 5, with the leakage caveat that Fit sets rotate across the five reshuffles.


---

## 2026-10-02 | Stage 0 | data ingestion + track index

* Environment: Python 3.11.9; numpy 2.4.6, pandas 3.0.6, pillow 12.3.0, scipy 1.17.1, pyyaml 6.0.3, tqdm 4.70.1, matplotlib 3.11.2 (requirements-lock.txt; Stage 0 only, re-freeze after Stage 1)

* Git commit: 45a4d41ab9e8d3b1f877555f58c4fd991804cdfe (2026-10-10 snapshot that includes this stage)

* Data: GTSRB_Final_Training_Images.zip, GTSRB_Final_Test_Images.zip, GT-final_test.csv (labelled; used instead of the GT zip, same file)

* Result: train 39,209 images / 1,307 tracks / frames {30: 1306, 29: 1} / 43 classes; test 12,630 images / 43 classes; GT filenames == test zip images

* Checks: all assertions PASS (check_zips.py and src.stage0); track strips inspected, 5 tracks look like single signs

* Cache SHA-256 (112x112 uint8):

  * train_112.npy 67A541A74081E5FF51DBCDDF68547B5409E354F424FF0E826F01C409ECD56A53

  * test_112.npy E0B5986EF9D5F7515DAA87EB0A7F98ADCA80114D0871EA5C0B38AA7184BB992E

* Reproducibility: first run on Python 3.13 (numpy 2.5.3, scipy 1.18.1), rerun on 3.11.9 (numpy 2.4.6, scipy 1.17.1): both caches byte-identical, pHash result identical (449 pairs / 205 test images)

* Cross-release pHash flag: 449 pairs (Hamming <= 2), 205 test images = 1.62% of the test release (> 1% flag line in plan section 4). Hamming values are only 0 (20 pairs) or 2 (429 pairs); 70.6% of pairs share a class; matched test images are darker than typical (median brightness 47 vs 75).

* Diagnosis run 1 (contact sheet): top-10 closest pairs are real copies (8 at MAD 0.0, 2 at MAD 2.8, all class 14, same scene); random matched pairs are look-alikes (different sign values, 29% cross-class). The plan's premise 'each physical sign appears once' is therefore NOT strictly true. Exact copies are bounded by the 20 Hamming-0 pairs (identical pixels => identical pHash). Run 2 (stage0_package_IV, cache rebuilt from scratch, 449 pairs / 205 images reproduced a 3rd time): exact copies (MAD<=2) = 8 test images (0.06%), all class 14, one-to-one with 8 frames of ONE training track 14_00023 (train idx 22950-22969). Same-class pairs with MAD<=5 = 43 pairs / 26 test images: class 10 (30 pairs; very dark; the same train frame matches several different test images = look-alike pattern), class 14 (11), class 16 (2). Gray-zone sheet: class-10 pairs differ in scale/placement. STATUS: CLOSED for fine-tuning purposes (see decision).

* Decision: exclude from fine-tuning every test-release image with a same-class pHash partner at MAD <= 5 (26 images = 0.21%; data/cache/finetune_exclude_final.csv). Threshold chosen after seeing counts; impact bounded by 26 images. Residual risk: same-sign-different-frame copies that pHash misses; track 14_00023 is known-contaminated (optional later sensitivity: rerun without it). The plan's premise 'each physical sign appears once' is not strictly true: report in threats to validity.

* Deviation from R6? No design change. Process notes: initial run used Python 3.13 instead of the pinned 3.11 (fixed, results identical); phash threshold (Hamming <= 2) is my choice, not the plan's.

* Next step: finish the duplicate diagnosis, then Stage 1 (fine-tune)

---

## 2026-10-02 | Stage 1 | backbone fine-tune (run tag 20261003_stage1)

* Config hash (results.json `config_sha256`): 6c7404a0e7e3300de840d798dc2e0cb569aa199f6447dbbfb2d3ed7ea79356eb ; git commit (post-run snapshot): 6a0dbc25a0ff86bc1982f2581e68602991c7f46b

* Checkpoint: runs/20261003_stage1/best.pt (epoch 4), SHA-256 86527359bc61a5fbc4d48b7744e049b5ca3ff81cb016fb237298d0c007140bcc

* Data: official test release (12,630) minus 26 duplicate-list images = 12,604; stratified 90/10 by class -> train 11,344 / val 1,260 (seed 0).

* Model: ImageNet ResNet-18, new 43-way fc, 112x112, AdamW (backbone 2e-4, head 1e-3, wd 1e-4), 1 epoch warm-up + cosine, batch 128, AMP (CUDA), plain cross-entropy (no label smoothing), early stopping patience 5 on val accuracy.

* Augmentation (light only): random crop area 0.85-1.0, aspect 0.9-1.1, rotation +-10 deg, contrast +-20%.

  * Brightness jitter DROPPED: it overlaps the "night" scenario of the stress test.

  * Forbidden: flips (left/right signs change meaning), blur, fog, rain, noise, perspective, shear (these are the stress tests).

* Validation accuracy is optimistic (test release has no track IDs, so the same sign can sit in train and val; it reached 1.0000). Used only for stopping; never reported.

* Result: best epoch 4; 9 epochs run (early stop, ~40 s total). Sanity accuracy on the GTSRB training release = 0.9190 over 39,209 frames (labelled "backbone fine-tuned on the official test release", never "test accuracy"). Without the exact-copy track 14_00023: 0.9189.

* Worst classes: 21 (33.3%), 41 (36.3%), 37 (61.4%), 36 (62.6%), 39 (65.7%).

* Errors: 3,176 frames in 438 of 1,307 tracks (plan G1(ii) needs >= 30 errors over >= 10 tracks, so there is ample error mass).

* Band check: above the 90% bug line, below the 99.5% G1(ii)-risk line, so PASS under the plan rule. It is below the 95-99% range I had guessed (my guess, not a plan rule); margin above the bug line is only 1.9 pp. The script's first message said "inside expected band" for this run; that message was wrong and has been corrected.

* Diagnostics (scripts/stage1_diagnostics.py, read-only; tables in runs/20261003_stage1/diagnostics_*.csv):

  * By size (shorter side): 20-25 px 77.8%, 25-30 px 82.3%, 30-40 px 90.6%, 40-50 px 94.8%, 50-70 px 95.5%, >70 px 94.7%. By frame position: frames 0-4 82.9%, frames 15-29 about 95-96%.

  * About 5% of frames stay wrong even when large and close, so distance is not the main driver. Errors concentrate in look-alike pairs (2->5 146, 41->42 119, 36->35 113, 26->18 106, 21->19 105).

  * Hypothesis "train release vs test release differ in image size" CHECKED AND REJECTED: size distributions match to within 1.2 pp in every bin.

  * [Guessing, not verified] weakest classes are the ones with the fewest fine-tuning images.

* Decision: accept the checkpoint as is. NO re-tuning after seeing the training-release number (it is the pool for Fit/Cert/Test; plan section 5: fine-tune once). Checkpoint frozen from here on: read-only copy, hash recorded above, a hash check is done before every later stage.

* Threats to validity: classes 21 and 41 have about 11 and 8 tracks, so per-class statistics will be thin after the Fit/Cert/Test split; plan to pool with confusable partners. Weak classes are the confusable ones the project studies, but report the 91.9% with its label.

* Environment: Python 3.12 (patch version: 3.12.10), torch 2.14.1+cu132, torchvision 0.29.1+cu132, numpy 2.5.2 ; GPU NVIDIA GeForce RTX 5050 Laptop GPU (see results.json `versions`).

* Deviation from R6? No design change. Process notes: first smoke run used random weights (2.7% accuracy on 600 frames is expected); brightness-jitter drop is my recommendation, accepted by running with the config default.

* Next step: Stage 2 (512-d l2-normalised embeddings in fp32; must reproduce Stage 1 predictions exactly; refuses to run if the checkpoint hash differs from the one above).

---

## 2026-10-02 | Environment change | Python 3.11.9 -> 3.12

* Why: torch's DLLs were blocked by Windows Application Control under Python 3.11 (WinError 4551); torch 2.14.1+cu132 under Python 3.12 loads and runs on the GPU (matmul check passed, sm_120 in the arch list).

* Check: cache SHA-256 re-verified under Python 3.12, both MATCH the Stage 0 values (train 67A541A7...D56A53, test E0B5986E...BB992E). pytest: 17 passed under 3.12 (before the diagnostics test was added).

* New lock file: requirements-lock-py312.txt (Stage 0 entry above refers to the old 3.11 lock file).

* Deviation from R6? No design change.

* Versions after the change: Python 3.12.10, torch 2.14.1+cu132, torchvision 0.29.1+cu132, numpy 2.5.2 (the Stage 1 config hash is logged in the Stage 1 entry).

---

## 2026-10-03 | Stage 2 | embeddings from the frozen backbone (run tag 20261003_stage1)

* Checkpoint hash checked before running (equals the frozen value; script stops otherwise): 86527359bc61a5fbc4d48b7744e049b5ca3ff81cb016fb237298d0c007140bcc

* Git commit: 45a4d41ab9e8d3b1f877555f58c4fd991804cdfe (2026-10-10 snapshot that includes this stage)

* What was done: ResNet-18 in eval mode, fp32, no augmentation, over (a) the 11,344 fine-tune-train frames (official test release; split recomputed with the Stage 1 code and seed, counts matched Stage 1: pool 12,604 / train 11,344 / val 1,260) and (b) all 39,209 training-release frames. Saved 512-d l2-normalised embeddings, raw 512-d features, 43 logits, class / track / frame indices to `runs/20261003_stage1/embeddings/`. Raw features were saved so that plan Stage 3 mitigation 2 does not need a re-run.

* Checks (all PASS):

  * Row counts match the metadata: ft_train 11,344; train_release 39,209.
  * No NaN / inf.
  * Max |norm - 1|: ft_train 1.1e-08, train_release 1.3e-08 (limit 1e-5).
  * Feature really is the input of the last layer (`logits = fc(feature)`): max difference 2.5e-06 (ft_train), 3.3e-06 (train_release).
  * Argmax of the logits reproduces Stage 1 exactly: train-release accuracy 0.918998 = the logged Stage 1 value 0.9190 (identical predictions to the Stage 1 evaluation function, 0 differing frames).

* Accuracy on the frames the head will be fitted on (ft_train; the backbone was trained on these): 0.9996. Accuracy on the training release: 0.9190.

* Output file SHA-256 (`embeddings_summary.json` `files_sha256`):

  * `ft_train.npz`: a0c15f7d59f893a13794517c610bdcc320917eccc9990423a9577e96f654fdf7
  * `train_release.npz`: 228fe5b2ee29f59d5e3fac7e5b81ffd8b1a9b1f6b5747756f88711345034d380

* Threat to validity (plan-frozen, so only logged): the head is fitted on embeddings of frames the backbone was trained on (99.96% accurate) while it will be applied to frames at 91.9% accuracy. Those clusters can be tighter than for unseen frames, which could make distances and the open-set gate look sharper than they are. Stage 3 head-vs-softmax agreement on the training release will show whether this matters. The plan rule is not changed because of it.

* Environment: Python 3.12.10, torch 2.14.1+cu132, numpy 2.5.2, device cuda (RTX 5050 Laptop GPU). The Stage 0 lock file (3.11.9) had numpy 2.4.6; the cache hashes still matched, so no effect on results.

* Deviation from R6? No design change.

* Next step: backup `runs/20261003_stage1/embeddings` next to the checkpoint, then Stage 3 (tied-Gaussian head, epsilon table, confusability atlas).

---

## 2026-10-10 | Stage 3 | tied-Gaussian head, epsilon table, confusability atlas (run tag 20261003_stage1)
* Inputs: Stage 2 embeddings (script checked them against the frozen checkpoint hash 86527359bc61a5fbc4d48b7744e049b5ca3ff81cb016fb237298d0c007140bcc and the Stage 2 file hashes) ; git commit: 45a4d41ab9e8d3b1f877555f58c4fd991804cdfe (2026-10-10 snapshot that includes this stage)
* Features used: l2-normalised (emb), the plan default. Output folder: runs/20261003_stage1/head_emb/
* What was done: 43 class means + one pooled within-class covariance (Ledoit-Wolf) fitted on the 11,344 fine-tune-train embeddings only; empirical class priors; label = top-1 posterior; margin = top-1 minus top-2 log-posterior; OOD score s(x) = min Mahalanobis distance; eps_ij = Phi(-Delta_ij/2) stored as log eps (equal priors).
* Covariance: Ledoit-Wolf shrinkage 0.0208 ; condition number 151 (eigenvalues 2.3e-05 to 3.5e-03) ; finite: PASS
* Head vs softmax on the training release: head acc 0.9260 ; softmax acc 0.9190 ; label agreement 0.9651 ; softmax minus head -0.70 pp (head is better). More than 2 pp below softmax? NO, so no plan mitigation was needed.
* On the fit frames (ft_train): head acc 1.0000 ; softmax acc 0.9996 ; label agreement 0.9996
* Spearman(log eps, empirical symmetrised confusion rate, softmax labels) over 903 pairs: 0.434 (plan expectation roughly >= 0.3, a guess). Non-zero-confusion pairs only: 0.285. Pairs with zero confusion: 707 of 903 (heavy ties). Descriptive only; nothing tuned on it.
* Top-10 empirical confusion pairs found in the top-20 eps pairs: 0 of 10. This test is very strict (top 20 is only 2.2% of the 903 pairs); see the pair-rank result below, which shows the real confusions DO rank high in eps.
* Pairs where plain eps would underflow to 0: 0 (log eps ranges from -124.5 to -508.5)
* Ten most confusable pairs by eps (class i, class j, Delta, log eps, confusions, empirical rate): 3-5 Speed 60/Speed 80 (31.09, -124.47, 61, 1.87%); 5-8 Speed 80/Speed 120 (31.59, -128.41, 143, 4.37%); 2-5 Speed 50/Speed 80 (31.78, -129.97, 162, 3.94%); 4-8 Speed 70/Speed 120 (33.74, -146.04, 29, 0.86%); 2-4 Speed 50/Speed 70 (33.78, -146.41, 90, 2.13%); 4-5 Speed 70/Speed 80 (34.68, -154.13, 28, 0.73%); 1-5 Speed 30/Speed 80 (34.90, -156.00, 25, 0.61%); 2-3 Speed 50/Speed 60 (34.98, -156.73, 30, 0.82%); 7-8 Speed 100/Speed 120 (35.30, -159.57, 63, 2.21%); 3-8 Speed 60/Speed 120 (35.50, -161.36, 21, 0.74%). All ten are speed-limit pairs. Sign names are standard GTSRB labels written from memory; check against the signnames file before quoting.
* Atlas (atlas.png, top10_pairs.png): clear block structure; speed-limit, danger-sign and mandatory-sign groups are brighter (more confusable) than the rest. Top-10 sample images show dark and low-contrast frames for the confused speed signs.
* Output file SHA-256 (head_summary.json files_sha256):
  * head.npz 5e85b1b0ca9ee45b86bcf20fcf4fce454128f8b292d454b84da42fe12ff0a5c2
  * scores_train_release.npz 0be15ed0b850e24ddc69a7395e954372f127ae6a2387369d0e812f082f3ceaac
  * scores_ft_train.npz 5f538d97be01dbcab863ac24ce7f95ee0538ec6ce084b016275d5358c141a5bd
* Threat to validity carried from Stage 2 (head fitted on frames the backbone was trained on, 99.96% accurate): [Likely] visible in the absolute eps values. Delta about 31 gives eps of about 1e-54 for the closest pair (Speed 80 vs Speed 120), while the empirical confusion rate for that pair is 4.4%. eps is therefore used ONLY as a ranking (stratification), never reported as an error probability; the certificate counts real errors on held-out tracks and does not depend on eps being calibrated.
* Pair-rank check (scripts/stage3_pair_ranks.py, descriptive only, whole training release, softmax labels; total confusion events 3,176 = the Stage 1 error count, consistent):
  * The 15 most-confused real pairs have eps ranks 2, 3, 28, 35, 47, 77, 79, 122, 154, 157, 185, 200, 307, 427, 595 (of 903); 12 of 15 are inside the top 200. Highest real rate: 41-42 (26.3%, eps rank 35), then 19-21 (22.8%, rank 185).
  * Share of ALL confusion events inside the top-k eps pairs: top 20 = 31.0%, top 50 = 44.7%, top 100 = 60.7%, top 200 = 81.6%, top 451 (half of the pairs) = 90.9%.
  * 94.4% of confusion events are between classes of the same Stallkamp subset, while same-subset pairs are only 19.8% of all pairs.
  * Reading: eps rank is a useful ruler for where errors fall, so the earlier 0-of-10 worry about G1(i) was overstated. NOT proven: G1(i) is decided in Stage 5 on Fit/Cert/Test splits with frame-mass strata, not on these pair counts. Risk to remember: because most confusion is inside Stallkamp subsets, the six-subset baseline may capture much of the same structure; the value of eps over that baseline has to be shown in the full study, not assumed. No rule is changed; the pre-registered top-3 eps3 stays as the first fallback.
* Decision: keep the Gaussian head (plan section 6, Stage 3); no mitigations needed because the head is not below softmax. eps is a rank, not a probability.
* Deviation from R6? No design change.
* Next step: Stage 4 (reshuffles and Fit-side calibration). Commit the PRE-REGISTRATION block of LOG.md before Stage 5.

---

## 2026-10-10 | Stage 4 | reshuffles and Fit-side calibration (run tag 20261003_stage1)
* Config / code / checkpoint / git commit: constants = the `constants_used` block of stage4_summary.json (plan section 1 values, code defaults; no `splits:` block was added to poc.yaml) ; code = src/splits.py SHA-256 065e7047308a163c8baacd3a5030f766c6ac25d49aed75a1bcb0a7c348e3a1d7 (checked on the author's machine with Get-FileHash: identical) ; checkpoint 86527359bc61a5fbc4d48b7744e049b5ca3ff81cb016fb237298d0c007140bcc (verified by the script, which also verified the Stage 2 and Stage 3 file hashes) ; git commit: 45a4d41ab9e8d3b1f877555f58c4fd991804cdfe (2026-10-10 snapshot that includes this stage)
* What was done: `py -3.12 -m src.splits --run runs/20261003_stage1`. Track-level splits Fit/Cert/Test = 261/784/262 (seeds 0-4), open-set gate tau, cut-point c*, Fit base errors, 50-point margin grids and the plan 6.3 testing order, for eps (primary) and eps3 (first fallback). Output folder: runs/20261003_stage1/splits/.
* Not done on purpose: no Cert or Test error rate of any kind was computed.
* Two runs, one code change (read this):
  * Run 1 used the POOLED base error (weight 1/n_t per frame) for the exploratory alpha = 0.5 x base. After seeing the Fit base errors I decided the PER-TRACK MEAN is the primary definition (see Interpretation choices). src/splits.py was updated and the script re-run (run 2).
  * Only the "rel_0.5base" ordering changed (it now uses 0.5 x per-track mean; the pooled version is kept as "rel_0.5base_pooled"). Splits, tau, c*, easy-mass fractions, base errors, grids and the absolute-alpha orders are deterministic and do not depend on this. The tables below come from run 1; the run 2 printout was checked and matches them: [x] matches (c*, easy mass, rejection rates, per-track and pooled base errors identical).
  * Run 1 file hashes are superseded. Use the run 2 hashes below.
* Output file SHA-256 (run 2, from stage4_summary.json `files_sha256`):
  * seed0.npz 397611c5538cc9ebe7a0be768ada29cb0afa7c89bdc53511df3a2403001fea49
  * seed1.npz f4d3555bcc765e66fd8c7c1358e625f3dfe39d9e97dc098197320b38ec5055b4
  * seed2.npz f0f70752cb896e7317bad891574c74a099628313ef087a7bd1b74aaf417ac02e
  * seed3.npz cafb335d3eb2b1c9f27645b3b495afd47dca7f10be367c9d0a4a393289791991
  * seed4.npz eb53fbef50819d167fa9c225b0cac19b9f5d01e4154ea58c50169d9e5e94b154

### Result numbers (from splits/stage4_summary.json; Fit errors are on gated-in frames, no margin threshold)

Splits and gate. All seeds: tracks 261/784/262, conformal rank 249 of 261, min tracks per class per split 1/4/1. Frames per split about 7,830 / 23,520 / 7,860.

| seed | tau | Cert reject, frame-level | Cert reject, 1 frame/track | eps c* | easy mass Fit | easy mass Cert |
|---|---|---|---|---|---|---|
| 0 | 43.84 | 4.8% | 4.6% | -230.74 | 50.7% | 50.9% |
| 1 | 41.49 | 8.5% | 7.9% | -231.21 | 49.5% | 48.8% |
| 2 | 44.45 | 4.2% | 5.2% | -232.85 | 49.4% | 47.7% |
| 3 | 42.98 | 6.7% | 6.6% | -230.03 | 49.6% | 50.8% |
| 4 | 44.13 | 4.3% | 4.5% | -230.74 | 49.9% | 51.0% |

Fit base error per stratum, eps, in percent. PRIMARY = per-track mean; pooled = sensitivity. Gap = hard minus easy.

| seed | easy per-track | hard per-track | GAP per-track | easy pooled | hard pooled | gap pooled |
|---|---|---|---|---|---|---|
| 0 | 5.51 | 5.94 | +0.43 | 4.52 | 4.57 | +0.05 |
| 1 | 5.98 | 6.59 | +0.61 | 3.88 | 5.33 | +1.44 |
| 2 | 7.95 | 8.32 | +0.37 | 5.89 | 5.31 | -0.57 |
| 3 | 5.58 | 5.23 | -0.35 | 3.66 | 4.24 | +0.58 |
| 4 | 5.11 | 6.61 | +1.50 | 5.10 | 5.80 | +0.70 |
| mean | | | +0.51 | | | +0.44 |

Fit base error per stratum, eps3 (first fallback), in percent. c* = -224.77 in all five seeds.

| seed | easy per-track | hard per-track | GAP per-track | easy pooled | hard pooled | gap pooled |
|---|---|---|---|---|---|---|
| 0 | 6.29 | 7.57 | +1.28 | 4.47 | 4.62 | +0.15 |
| 1 | 5.85 | 6.97 | +1.13 | 3.75 | 5.48 | +1.73 |
| 2 | 9.58 | 9.06 | -0.52 | 5.78 | 5.41 | -0.37 |
| 3 | 5.38 | 7.05 | +1.67 | 3.42 | 4.47 | +1.04 |
| 4 | 6.74 | 8.59 | +1.85 | 4.87 | 6.04 | +1.17 |
| mean | | | +1.08 | | | +0.75 |

Easy-mass fraction for eps3: Fit 49.4-50.0%, Cert 49.1-51.1%.

Fit-based testing order: the start index (of 50 grid points) is 6-23 depending on stratum, variant and alpha, so the fixed-sequence test walks at most 7-24 thresholds. Estimated Fit E[L] at the start point for alpha = 3% is 0.007-0.013, far below alpha, so Fit shows a wide cushion. Fit is only a hint about Cert.

### Checks
* [x] splits disjoint, cover all 1,307 tracks, every class in every split (script asserts)
* [x] realised Fit split within 45/55 in all 5 seeds (49.4% to 50.7%)
* [x] Cert easy-mass 47.7% to 51.0% for eps, nowhere near the 35/65 warning line
* [~] Cert one-frame-per-track rejection: 4.5%, 4.6%, 5.2%, 6.6%, 7.9% (mean 5.8%). Two seeds are above 5% but this is within sampling noise (rough SD of one seed about 1.6 pp, because tau comes from only 261 tracks; seeds share tracks, so they are not independent). Gate not retuned.
* [x] frame-level Cert rejection maximum 8.5%, below the 10% NOTE line
* [x] no NOTE (lopsided split / rejection above 10%) was printed
* [x] no Cert/Test error computed
* [x] tests: `py -3.12 -m pytest tests -q` gives 41 passed, 0 failed, 0 skipped, 1 warning (2026-10-10, Python 3.12.10). The warning is the harmless NumPy non-writable-array UserWarning at src/train.py line 129 (old evaluate(); no effect on results or on the checkpoint). An earlier run gave 39 passed because tests/test_stage1.py was an older copy without the diagnostics and Stage 2 smoke tests; after replacing it the full suite passes.

### What the Fit numbers say (Fit only, NOT G1 evidence)
* The eps hard-minus-easy gap is small: mean +0.51 pp per-track (+0.44 pp pooled) on a base error of 5-8%. It is positive in 4 of 5 seeds under both definitions (negative: seed 3 per-track, seed 2 pooled). [Likely] this is a risk for G1(i). Cert has about 3x the tracks of Fit, but a 0.5 pp gap against track-clustered errors is probably well under one standard error [Guessing; no variance computed yet].
* eps3 shows a larger gap on Fit (+1.08 pp per-track, +0.75 pp pooled, positive in 4 of 5 seeds). Not a result. eps3 was named as the first fallback before this; Stage 5 reports eps and eps3 side by side.
* Why the median cut may be weak [Guessing]: confusions concentrate in a few hundred pairs (top 200 hold 81.6%), and the top-2 pair of a frame is made of near neighbours, so most frame mass already sits in high-eps pairs. A 50/50 mass cut can then fall where error rates are similar. A Fit-only table of error by eps-octile would test this. It may inform a future revision R7. It does not change c*.
* G1(ii) (>= 30 hard-group frame errors over >= 10 tracks on Cert) should be met easily: about 11,000 Cert hard-stratum frames at 4-6% base error [Likely].
* G1(iii) looks feasible on Fit, but Fit is optimistic because the margin threshold is chosen there. The Stage 6 certifier decides.
* Pooled vs per-track base error differ by 1-4 pp (e.g. seed 2 hard: 5.31 vs 8.32). Errors sit in tracks that contribute only a few frames to a stratum (probably tracks whose predicted pair flips); the per-track mean gives those tracks the same weight as full tracks.
* The relative alpha (0.5 x per-track base) comes out at about 2.6-4.8%, so some of these are above 3%. Only the absolute grid values 1%, 2%, 3% count for G1(iii) (pre-registration item 1). The relative alphas are exploratory.
* Margins and tau are on a large raw scale (margin thresholds in the hundreds, tau about 41-44) because the Stage 3 head is fitted on very tight clusters. Only ranks and quantiles are used, so nothing breaks, but these are not calibrated probabilities.

### Interpretation choices
* Base error (decided 2026-10-10 by the author): the PER-TRACK MEAN of in-stratum error rates is primary, for G1(i) and for alpha = 0.5 x base. Reason: the Stage 6 certificate averages per track, so this is the quantity it bounds. The pooled rate (weight 1/n_t per frame) is reported as a sensitivity. Disclosure: this was decided after seeing the Fit base errors. Both definitions give the same sign count for eps (4 of 5), so the choice was not driven by which one gives the larger gap.
* c* mass still uses weight 1/n_t per frame, as the plan words it. This is the c* rule, not the base-error definition, and is unchanged.
* Margin accept rule: accept if margin >= threshold; grid index 0 = loosest.
* Test order per plan 6.3: start at the Fit argmin of estimated E[L] with L_t = l_t + alpha, then looser. alpha in {0.01, 0.02, 0.03, 0.05, 0.5 x per-track base, 0.5 x pooled base}; all stored, none chosen here.
* eps3 = eps(y1,y2) + eps(y1,y3), summed in the log domain.

* Decision taken and why: keep c*, delta, beta, strata, reshuffles and ordering exactly as defined (plan section 6, Stage 4 and 6.3). Nothing re-tuned despite the small Fit gap.
* Deviation from R6? No. The per-track definition is an interpretation of an undefined plan term, not a design change.
* Next step: the pre-registration block was committed at 45a4d41ab9e8d3b1f877555f58c4fd991804cdfe before any Cert or Test error was computed. Stage 5 (G1(i)/(ii), rho and ICC, rank-of-error, eps3 ablation) may start. Optional beforehand, Fit only: track-cluster bootstrap of the hard-minus-easy gap and an error-by-eps-octile table (diagnostics; they change nothing that is frozen).

---

## 2026-10-10 | Stage 4 add-on | Fit-only diagnostic (run tag 20261003_stage1)
* Code / commits: src/fitdiag.py ; tests: 48 passed, 1 warning (the known harmless train.py UserWarning). Addenda A and B and src/fitdiag.py were committed together in 8b512fc ("fitdiag", on branch bashee, after the pre-registration freeze 45a4d41). The order of that commit relative to the diagnostic run was not recorded; if the run came first, the Addenda were written in the working copy of LOG.md but not yet committed at run time. Either way the diagnostic is information only and changed no frozen rule.
* What was done: `py -3.12 -m src.fitdiag --run runs/20261003_stage1` (B = 2000 track-cluster bootstrap draws, Fit tracks resampled; output splits/fit_diagnostic.json). Verified the Stage 4 file hashes, recomputed splits and c* (matched). Fit frames only; no Cert or Test error computed.
* Status: information only (Addendum B). Nothing frozen was changed.

### A. Hard-minus-easy gap on Fit, percentage points (per-track mean = primary)

| variant | seed | gap | Fit SE | 95% CI | pooled gap |
|---|---|---|---|---|---|
| eps | 0 | +0.43 | 1.76 | [-3.27, +3.88] | +0.05 |
| eps | 1 | +0.61 | 1.85 | [-3.19, +4.11] | +1.44 |
| eps | 2 | +0.37 | 1.94 | [-3.38, +4.25] | -0.57 |
| eps | 3 | -0.35 | 1.66 | [-3.73, +3.00] | +0.58 |
| eps | 4 | +1.50 | 1.69 | [-1.81, +4.79] | +0.70 |
| eps3 | 0 | +1.28 | 2.26 | [-3.04, +5.56] | +0.15 |
| eps3 | 1 | +1.13 | 2.06 | [-2.89, +5.29] | +1.73 |
| eps3 | 2 | -0.52 | 2.27 | [-5.08, +3.89] | -0.37 |
| eps3 | 3 | +1.67 | 2.20 | [-2.40, +5.95] | +1.04 |
| eps3 | 4 | +1.85 | 2.16 | [-2.44, +6.07] | +1.17 |

* Means: eps gap +0.51 pp (Fit SE 1.78); eps3 gap +1.08 pp (Fit SE 2.19). Every Fit confidence interval contains 0.
* Rough read-across to Cert (Cert has 784 tracks vs 261, so SE about sqrt(261/784) = 0.58 times the Fit SE): Cert SE about 1.03 pp (eps) and 1.26 pp (eps3). G1(i) needs the 2.5th-percentile bound above 0, i.e. a gap of about 2.0 pp (eps) or 2.5 pp (eps3) just to clear it.
* Rough chance that G1(i) passes in a single reshuffle: 7.2% (eps), 13.5% (eps3). In >= 4 of 5 reshuffles: 0.0% (eps), 0.1% (eps3). These assume the true gap equals the Fit gap (itself noisy) and independent reshuffles (they are not). Order-of-magnitude only. Fit CIs do not rule out a true gap of 2-4 pp, so "unlikely", not "impossible".

### B. Error by confusability slice (8 slices of equal mass, mean over seeds)

eps (primary):

| slice | mass | per-track err | pooled err | share of Fit errors |
|---|---|---|---|---|
| 0 (lowest eps) | 11.8% | 5.82% | 3.11% | 7.8% |
| 1 | 13.3% | 9.18% | 5.65% | 15.4% |
| 2 | 12.6% | 11.91% | 4.86% | 12.5% |
| 3 | 12.1% | 7.77% | 4.63% | 11.6% |
| 4 | 11.2% | 9.84% | 4.01% | 9.4% |
| 5 | 13.2% | 12.83% | 5.57% | 15.5% |
| 6 | 14.8% | 12.33% | 5.94% | 18.2% |
| 7 (highest eps) | 10.9% | 10.03% | 4.13% | 9.6% |

eps3 (first fallback):

| slice | mass | per-track err | pooled err | share of Fit errors |
|---|---|---|---|---|
| 0 | 14.4% | 5.27% | 0.91% | 2.8% |
| 1 | 10.6% | 10.74% | 8.77% | 19.0% |
| 2 | 12.5% | 10.33% | 4.83% | 12.6% |
| 3 | 12.4% | 12.96% | 4.47% | 11.3% |
| 4 | 12.9% | 9.23% | 3.15% | 8.3% |
| 5 | 12.5% | 19.29% | 3.93% | 10.2% |
| 6 | 12.5% | 20.97% | 8.28% | 21.3% |
| 7 | 12.4% | 28.21% | 5.59% | 14.4% |

### What this says
* **eps does not separate error rates on Fit.** The pooled error is flat across all 8 slices (3.1-5.9%), and each slice holds about its fair share of errors. [Certain] for these numbers. The hard-minus-easy gap of about +0.5 pp is small against a Fit SE of 1.8 pp.
* **G1(i) is very likely to fail for eps, and probably for eps3** [Likely]. This is consistent with the Stage 3 warning that 94.4% of confusions sit inside the same Stallkamp subset: a pair-level rank does not add much at a median cut.
* **The eps3 per-track gradient (5% to 28% across slices) is mostly not real structure** [Likely]. The pooled error does not rise with eps3 (0.9% to 8.8%, no trend), while the per-track mean is 2 to 5 times higher in the upper slices (slice 7: 28.2% vs 5.6%). The per-track mean gives every track equal weight, including tracks that contribute only one or two frames to a slice (probably tracks whose predicted pair flips). Those fragments are noisy and error-prone, and they dominate the average. This also means my earlier remark that eps3 looks better on Fit is weaker than it sounded: the pooled eps3 gap is only +0.75 pp.
* **Cost of the committed definition:** the per-track mean has the larger SE and the more erratic slice profile. It is still the primary, because it matches the certificate's per-track loss. The pooled rate stays as the sensitivity (Addendum A applies if the two disagree).
* **G1(ii) and G1(iii) are not affected by this diagnostic.** G1(ii) should still pass easily; G1(iii) is decided by the Stage 6 certifier.
* Not tested yet [Guessing]: the fragment hypothesis. It can be checked descriptively (frames per track in a stratum, error by that count) without any Cert errors.

### Decision
* Proceed to Stage 5 exactly as pre-registered. No R7, no change to c*, strata, seeds or definitions. Reasons: the one visible gradient (eps3 per-track) looks like a fragment artifact, so re-tuning to it would chase noise; and Fit sets rotate across reshuffles, so a design tuned on Fit leaks into the Cert verdicts.
* Expected outcome, stated before Stage 5: G1(i) fails for eps (and probably eps3). Under the proposal's own G1 rule, a failure means clean GTSRB is reported as a certifiability frontier and the scenario cells become the primary evidence. That is a pre-registered outcome, not a deviation.
* Deviation from R6? No.
* Next step: Stage 5 (G1(i)/(ii) on Cert, rho and ICC, rank-of-error, eps3 ablation), plus one descriptive addition decided now: report the distribution of frames per track within each stratum (counts only, on Fit and on Cert) so the fragment effect is documented.


---

## 2026-10-11 | Stage 5 | G1(i), G1(ii), rho and rank-of-error on Cert (run tag 20261003_stage1)

* Config / code / checkpoint / git commit: constants = the `constants_used` block of stage5_results.json (plan and pre-registration values; no `stage5:` block added to poc.yaml) ; checkpoint 86527359bc61a5fbc4d48b7744e049b5ca3ff81cb016fb237298d0c007140bcc ; git commit: ____ (fill after committing the Stage 5 code: `git rev-parse HEAD`) ; stage5_results.json SHA-256: ____ (`Get-FileHash runs\20261003_stage1\splits\stage5_results.json`).
* Code change before this run (environment, not design): Windows Smart App Control blocked torch's unsigned DLLs (Code Integrity event 3077, policy VerifiedAndReputableDesktop, WinError 4551), which also stopped the numpy-only stages because they imported `sha256_file` from `src/train.py` (imports torch). Added `src/util.py` (same function, no torch) and changed ONE import line in `src/splits.py` and `src/fitdiag.py` (`src/stage5.py` is new). No logic changed; `train.py`, `head.py` and every Stage 1-4 output are untouched. Stage 5 asserts that the recomputed splits, tau and c* equal the saved Stage 4 files (they did). New SHA-256: splits.py 868a163af0994e8415de32235c03a76b9638c00366fab539be636f59c2d8aeaf (was 065e7047308a163c8baacd3a5030f766c6ac25d49aed75a1bcb0a7c348e3a1d7), fitdiag.py cb666b010f7aacf02633426765b24f64d7913bbf91f53ab45c8d47208b0c08f9, stage5.py 0aa94e8d26e5e24247557f824bf783a33463a0b676d31efc69c5b6fc45964212, util.py e4224c39386ee08b7d6db73fb27616a40e86c256ac51667ec7776d32f60b02da.
* Tests: `py -3.12 -m pytest tests -q` gives 48 passed, 2 skipped (the Stage 1 and Stage 3 test files skip themselves because torch cannot load on this machine; with torch the suite is 63 passed).
* What was done: `py -3.12 -m src.stage5 --run runs/20261003_stage1`. B = 2000 track-cluster bootstrap over Cert tracks (a second run with 2B draws as the stability check), strata from each seed's Fit c*, labels = head top-1 on gated-in Cert frames, no margin threshold. Labels read: Cert only, plus the dataset-level rho over all 39,209 frames. Test untouched.
* Rules applied exactly as pre-registered: G1(i) 2.5th percentile > 0 in >= 4 of 5; G1(ii) median >= 30 hard-group errors and >= 10 hard-error tracks; per-track mean primary, pooled sensitivity (Addendum A); eps primary, eps3 first fallback.

### G1(i): hard-minus-easy gap on Cert, percentage points (per-track mean = primary)

| variant | seed | easy | hard | gap | SE | 95% lower | pass | pooled gap | pooled lower | pooled pass |
|---|---|---|---|---|---|---|---|---|---|---|
| eps | 0 | 8.16 | 9.00 | +0.84 | 1.12 | -1.35 | no | +1.09 | -0.79 | no |
| eps | 1 | 5.70 | 5.72 | +0.02 | 1.06 | -1.97 | no | +0.96 | -0.48 | no |
| eps | 2 | 6.92 | 7.42 | +0.50 | 0.98 | -1.33 | no | +0.84 | -0.98 | no |
| eps | 3 | 6.73 | 6.87 | +0.14 | 1.08 | -2.07 | no | +1.01 | -0.79 | no |
| eps | 4 | 8.06 | 7.13 | -0.93 | 1.07 | -3.05 | no | -0.45 | -2.17 | no |
| eps3 | 0 | 8.77 | 10.05 | +1.28 | 1.37 | -1.44 | no | +1.35 | -0.65 | no |
| eps3 | 1 | 6.89 | 7.37 | +0.47 | 1.30 | -1.93 | no | +1.06 | -0.43 | no |
| eps3 | 2 | 7.81 | 8.85 | +1.04 | 1.36 | -1.55 | no | +1.01 | -1.03 | no |
| eps3 | 3 | 7.58 | 7.94 | +0.36 | 1.33 | -2.31 | no | +1.21 | -0.65 | no |
| eps3 | 4 | 8.44 | 7.46 | -0.98 | 1.27 | -3.36 | no | -0.25 | -2.07 | no |

* G1(i) verdict, eps (per-track, primary): 0 of 5 reshuffles pass -> FAIL. Pooled (sensitivity): 0 of 5 -> FAIL. Mean gap +0.11 pp per-track, +0.69 pp pooled. The two definitions agree, so the Addendum A caveat is not triggered.
* G1(i) verdict, eps3 (pre-registered first fallback): 0 of 5 -> FAIL. Pooled: 0 of 5 -> FAIL. Mean gap +0.43 pp per-track, +0.88 pp pooled.
* Bootstrap stable when B is doubled: True for every seed and both definitions.
* Not a near miss: the best 95% lower bound is -0.43 pp (eps3, pooled, seed 1); a pass would need the gap to exceed about 1.96 SE, i.e. about 1.9 to 2.7 pp, against observed gaps of -1.0 to +1.4 pp.

### G1(ii): hard group on Cert

| variant | hard-group frame errors, seeds 0-4 | distinct tracks with a hard error, seeds 0-4 | median errors / median tracks | verdict |
|---|---|---|---|---|
| eps | 722, 490, 718, 608, 511 | 156, 129, 158, 142, 144 | 608 / 144 | PASS (need >= 30 and >= 10) |
| eps3 | 747, 493, 701, 616, 525 | 158, 133, 156, 144, 143 | 616 / 144 | PASS |

### rho and design effect

* All 39,209 frames in 1,307 tracks: rho 0.446 [0.394, 0.495]; design effect (m = 30) 13.9; effective sample size 2,817 frames.
* Gated-in Cert frames per reshuffle, all strata: rho 0.377 to 0.438 (CI widths about 0.14 to 0.16), actual design effect 11.0 to 13.1, effective sample size 1,706 to 1,948 frames.
* By stratum (eps): easy rho 0.519 to 0.582, hard rho 0.289 to 0.380.
* ICC inside [-0.05, 1]: yes (dataset-level and Cert).

### Rank of error (gated-in wrong Cert frames, 5 overlapping reshuffles, 5,703 errors)

* True class is 2nd 58.1%, 3rd 20.0%, 4th or lower 21.9%.

### Fragment counts (counts only, no errors; seed 0, eps)

| group | tracks | mean frames per track | median | tracks with <= 2 frames |
|---|---|---|---|---|
| Fit easy | 160 | 23.8 | 30 | 4.4% |
| Fit hard | 148 | 25.0 | 30 | 4.7% |
| Cert easy | 493 | 23.1 | 29 | 5.1% |
| Cert hard | 456 | 24.1 | 30 | 3.9% |

* Cert tracks with gated-in frames in BOTH strata: 168 of 784. The other 616 (79%) sit in one stratum only.

### Checks
* [x] tests: 48 passed, 2 skipped (torch blocked; expected)
* [x] Stage 4 file hashes, splits, tau and c* matched (the script asserts)
* [x] bootstrap stable when B is doubled (all seeds, both definitions)
* [x] ICC inside [-0.05, 1]
* [x] all-frame head error 7.40% reconciles with the Stage 3 head accuracy (`reconciles_with_stage3_head_accuracy: True`)
* [x] no Test label read (enforced by a unit test; the one all-frame number is the dataset-level rho and the head error)

### What the numbers say
* [Certain] G1(i) fails for eps and for eps3, under both base-error definitions, in all 5 reshuffles. This is the outcome stated before Stage 5 in the Fit-only diagnostic entry. It is a pre-registered result, not a deviation.
* [Likely] The true clean-data gap is small, roughly 0 to 1 pp. The pooled gap is positive in 4 of 5 seeds for both variants (+0.7 and +0.9 pp on average), the same pattern as on Fit, so a weak positive effect is plausible but not detectable. With SE about 1.1 pp at 784 Cert tracks, confirming a 0.7 pp gap at 80% power would need an SE near 0.25 pp, about 19 times the tracks (roughly 15,000, against 1,307 in all of GTSRB). That ceiling belongs to the dataset, not to the method. [Guessing on the exact multiple: it assumes the same variance per track.]
* [Certain] G1(ii) passes by a wide margin (hundreds of hard-group errors over 130 to 160 tracks), so the hard stratum is not starved of errors.
* [Certain] rho = 0.446 over all frames. A frame-level certifier would treat 39,209 frames as independent when they carry the information of about 2,800 (design effect 13.9). This supports track-level certification (RG2) independently of G1.
* [Likely] Strata are mostly properties of the physical sign: 79% of Cert tracks fall in a single stratum, because eps depends on the (top-1, top-2) class pair, which is nearly constant within a track. Stratified certification will therefore behave like certifying groups of signs. The Stallkamp six-subset and class-conditional Mondrian baselines (RQ1b) are the decisive comparison; eps cannot be assumed to beat them.
* [Likely] Rank of error: 58% of errors have the true class 2nd, which supports the top-2 argument; 22% fall outside the top 3, where neither eps nor eps3 can describe the confusion.
* [Guessing] The easy stratum has higher rho (0.52 to 0.58) than the hard stratum (0.29 to 0.38): its errors sit in particular tracks (a few signs misread repeatedly), while hard-stratum errors are spread more widely. Descriptive only.
* The Fit diagnostic suggested the eps3 per-track gradient might be a fragment-track artifact. Fragments are few (about 4% to 5% of tracks have <= 2 frames in a stratum), so that explanation is at most partial. Error by fragment size was deliberately not computed.

### Decision
* Plan section 9, G1(i) fails: (1) Stage 3 Spearman and the pair-level check are already on record (Spearman 0.434; top 200 eps pairs hold 81.6% of confusion events), so no new action; (2) the pre-registered eps3 fallback also fails, so it does not rescue G1(i); (3) the scenario cells (RQ2) carry the main claim and clean GTSRB is reported as a certifiability frontier, which the proposal already states as the plan for a G1 failure; (4) this is the "negative result plus frontier" branch for the clean-data hypothesis.
* No R7, no change to c*, delta, beta, strata, seeds, definitions or aggregation rules.
* Deviation from R6? No.
* Next step: Stage 6 (certification engine). Synthetic validity tests must pass before any real data is used. G1(iii) and the floor table come from Stage 6.

---

## TEMPLATE (copy for each new entry)

## <date> | Stage N | <run tag>

* Config hash / checkpoint hash / git commit:

* What was done (one line):

* Result numbers (from the results JSON):

* Checks (pass/fail each):

* Decision taken and why (cite the plan section):

* Deviation from R6? yes/no (what, why):

* Next step:

---
## Stage 7, PART A: PRE-REGISTRATION (paste into LOG.md and COMMIT before running the calibration; the smoke test measures no accuracy and may come before or after)

- **Date / stage / run tag:** ____ / Stage 7 (G2) / 20261003_stage1
- **Git commit of this pre-registration (and of the Stage 7 code):** ____
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


- AMENDMENT to Stage 7 Part A (2026-10-15, after the smoke contact sheet, BEFORE any accuracy was measured): fog cap lowered 5.0 -> 4.0 because at s=5 several signs (30, 100, 80) are not recognisable on the real Fit sheet. Rain 4.0, night 4.0, side_view 5.0, motion_blur 5.0 unchanged. No other rule changed.

- **AMENDMENT 2 to Stage 7 Part A (date ____, after the first Fit calibration, before any Cert or Test scenario number exists).**
  - **What the first sweep showed (run 1, kept as evidence; `runs/20261003_stage1/stage7_run1_rain_v1/`):** rain was unusable as a strength axis. Fit drop was 65 pp at s = 0.1, 31 pp at s = 1.0 and 75 pp at s = 4.0, non-monotone (28% accuracy at s = 0.1, 51% at s = 0.3, 62% at s = 1.0), so no level lay inside either band and mild = severe = s 1.0 (the script flagged it and `--freeze` would have refused). Cause: albumentations RandomRain draws drops at a fixed 200 gray, the integer `blur_value` steps 1 -> 2 -> 3 at s = 0.25 and 0.75 changed the effective drop brightness by factors of 2-3, and the drop count changed at s = 1.5 and 2.5. The other four scenarios behaved: fog mild 0.3 (5.79 pp) / severe 0.75 (19.85 pp); night 0.75 (6.55) / 1.75 (20.50); side view 2.25 (5.82) / 4.0 (21.88); motion blur mild 4.0 (6.78), severe band not reached (max 10.54 pp at the cap 5.0).
  - **Change (rain only):** strength = brightness of the drops (`drop_color` gray 0, 15, 40, 80, 140, 200 at s = 0..5, interpolated); drop count (rain_type "default", 37 drops), drop width and blur (3) constant; drop length 8-16; image brightness 1.0-0.9; rain cap 4.0 -> 5.0. Nothing else changed: fog, night, side view and motion blur code, the bands, the selection rule, the tie-breaks, the cells rule, the grid, the stride.
  - **Why this is not tuning on the test:** the change is to the definition of one strength axis (declared in Part A as my choice), it was made on Fit data only, Cert and Test pixels were never read, and the reason is a property of the axis (non-monotone, nothing below the severe band), not a preference for a particular level. Motion blur's missed severe band is NOT being "fixed": it stays logged as band not reached. A second axis change would look like tuning.
  - **Consequence:** the calibration is re-run in full on the new `scenarios.py` (the sweep cache refuses results from other code). The four unchanged scenarios must reproduce run 1 exactly (same drops to the last digit); if they do not, stop and send me both files.
  - Predictions in Part A were: fog, night and motion blur reach the severe band [wrong for motion blur: 10.5 pp]; side view and rain do not [wrong for side view, 21.9 pp; rain was wrong in the other direction: everything above the band].


---

## POC-2 (R8) Stage 1: accumulation and track table

- Date: 2026-11-13 / Stage: R8 Stage 1 / Run tag: runs/20261003_stage1/poc2/stage1
- Checkpoint hash: (stage1_summary.json -> inputs_sha256.checkpoint) 86527359bc61a5fbc4d48b7744e049b5ca3ff81cb016fb237298d0c007140bcc
- Git commit of the code: a3e0b48ec41f313b2b7d7690ae66e2f409050cde
- What was done: per-frame log-likelihoods, gate, running scores for V1/V2/V3, tracks-by-lambdas commit table (200-point grid per variant, from design-set margins only).
- Result numbers (from stage1_summary.json):
  - tau: 43.8435 | frames failing gate: 4.836% | tracks with no passing frame: 5
  - Frame head accuracy: 92.599% (POC-1: 0.9259863806779056)
  - ROI vs frame index, median per-track Spearman: 0.998 (positive in 99.8% of tracks)
  - Track lengths: {29: 1, 30: 1306}
- Checks: C1a PASS C1b PASS C1c PASS C2 PASS C3 PASS C4 PASS C5 PASS C6 PASS — ALL PASS: True
- Output file SHA-256 (files_sha256): design ad6fa0c59f65697f107b0cb6b3dbef25b4b6680f8618fe38c50bd770d1f2b391 / pool 0afb7e89052c393d695a8cb7e5796e4e811b1ff9c3c6bd392bf3cd10742250db / frame_evidence f79033e8cee5ba73a6b22d1d33b91227ff7cf628d95dc616a28824ad68a0e960
- Decision: the commit table is written as two files. The 261 design tracks get labels and 'wrong'; the 1,046 pool tracks are label-free. Why: plan section 2 forbids knowing commit-rule outcomes on Cert/Test before F1.
- Deviation from the plan? Yes. Section 10 Stage 1 asks for 'committed, wrong' on all tracks; 'wrong' is computed on design tracks only, and on pool tracks only after F1 (Stage 4). This is stricter than the plan and changes no result.
- Deviation from the plan? Yes (detail). ROI size = min(x2 - x1, y2 - y1) in original pixels. The plan says "shorter side" without a convention. This convention is fixed here and frozen with s at F1.
- Environment note: `py -3.12 -m pytest tests -q` gave 169 passed, 1 failed before this run. The failure, `tests/test_stage3.py::test_ledoit_wolf_matches_sklearn`, is `ImportError: DLL load failed while importing _middle_term_computer: An Application Control policy has blocked this file` (scikit-learn 1.9.1), reproduced standalone with `py -3.12 -c "import sklearn.covariance"`. LOG.md Stage 4 entry (line 260) records 41 passed, 0 failed, 0 skipped under the same Python 3.12.10 during POC-1, so this test passed cleanly then. Cause not established: scikit-learn was not version-recorded in POC-1, so either a package upgrade or an Application Control policy change could explain it. The author approved running Stage 1 despite this failure because Stage 1 does not refit the head or use sklearn: it hash-checks `head.npz` against the frozen checkpoint and cross-checks C1a/C1b against stored POC-1 outputs only. The test file itself was not edited.
- Environment versions (`env_poc2_stage1.txt`, pip freeze, Python 3.12.10): numpy 2.5.2, scipy 1.18.1, scikit-learn 1.9.1, torch 2.14.1+cu132. Compared with POC-1: numpy 2.5.2 is the same as in the Python 3.12 entries of LOG.md (lines 116, 134, 163). scipy: LOG.md only records 1.17.1 (Stage 0 lock, Python 3.11.9, line 46) and 1.18.1 (first Python 3.13 run, line 62); the POC-1 Stage 7 and 8 result JSONs (Python 3.12.10) record scipy 1.18.1, so no change. scikit-learn: no version is recorded anywhere in POC-1 (LOG.md or run JSONs), so it cannot be shown whether the sklearn DLL block comes from a package upgrade or from an Application Control policy change.
- Next step: Stage 2 (design phase and D-gate), design set only.

---

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

---

## POC-2 (R8) Pre-registration block, gate F1 (plan section 13), 2026-11-14

Written before any Cert or Test outcome of the commit rule exists. Every value below was chosen on the 261 design tracks
only, and is in `configs/frozen_r8.yaml` (committed together with this block). Nothing here changes after this commit.
delta = 0.1 and beta = 5% in every phase.

- **Plan file:** `docs/poc2-implementation-plan-r8.md`, SHA-256 b0ad6292138732f40639753b6e5b76d050cfc30da137cc2b1d8f0c0e85174bd2
  (working-copy bytes), git blob 51d0aa50ce4486b851f3ec8d5b95e1281a6b1359. Items 8 and 9 refer to its sections 9 and 11,
  so the plan file is frozen from this commit on.

1. **Score variant: V1 (sum).** Chosen by plan section 5's rule at the primary alpha (LOG_r8_stage2_entry.md Part A (g)):
   V1 is the only variant predicted certifiable at 1% (commit count at e <= alpha/2: 240). Design-set table (best eligible
   lambda per cell, plug-in p at n = 784): V1 passes at 1, 2, 3, 5% (1%: lambda 7574, c 89.66%, e 0.00%, p 0.0054);
   V2 passes at 5% only; V3 passes at 2, 3, 5%. Full table in `runs/20261003_stage1/poc2/stage2/stage2_summary.json`
   (SHA-256 332964ba26a977dda167d194e6954bca627a82eaebeec8825a66b270c4f96e74).
2. **Lambda grid and threshold order.** 30 points of the Stage 1 V1 grid, indices 132 to 167, lambda 3723.7 to 12898.9
   (full list in frozen_r8.yaml). Certificate A order: start at working position 18 (lambda 8130.9, the design-set argmin
   of mean L at 1%, ties to the larger lambda), then relax one point at a time to position 0; stop at the first p > delta.
3. **Alpha grid and primary alpha.** {0.5, 1, 2, 3, 5}%; primary alpha 1% (the smallest alpha predicted certifiable).
4. **(s, gamma) for Certificate B.** gamma = 10%. s = not certifiable at n = 784: no s passed the design-set plug-in
   (best p 0.241 against 0.01). At the start threshold about 11% of tracks never commit, so the late fraction there is
   above gamma; at the lower thresholds where enough tracks commit (up to 93.5%), the error part fails. Stage 5 reports
   "not certifiable at n = 784" with the descriptive commit-fraction-vs-size curve. gamma and s do not change.
5. **Gate threshold tau = 43.84354019165039** (POC-1 seed-0 value, fitted on the design set).
6. **Six adverse cells**, severities unchanged from `configs/frozen_g2.yaml` (commit 00e1664): rain severe 3.0, side view
   severe 4.0, fog severe 0.75, night severe 1.75, rain mild 1.25, side view mild 2.25. Rule (plan section 4): every
   scenario whose severe band was reached at G2, plus the mild level of the two frozen recertification scenarios; motion
   blur's severe band was not reached, so it is dropped. This is the declared amendment plan section 4 requires before
   any scenario result on Cert or Test exists. Recertification curves: rain severe and side view severe.
7. **Pass levels.** P1: track-level violation rate upper Clopper-Pearson bound <= delta + 0.02; frame-level lower bound
   > delta. P2: primary alpha certified and Test sign coverage >= c_min = 0.79272030651341 (79.27%, design-set predicted
   coverage 89.27% minus 10 points), in at least 4 of 5 re-splits. P3: commit-rule Test coverage above B2 at the primary
   alpha, in at least 4 of 5. P4: median commit frame before the last frame, in at least 4 of 5.
8. **Baselines B1 to B6** exactly as in plan section 9 (plan file hash above).
9. **Fallbacks** as in plan section 11 (plan file hash above).

**Design-phase checks and declared deviation.** D1 FAILED: track error at the last frame 2.30% (6/261), upper 95%
Clopper-Pearson 4.94%, against the passing-frame single-frame error 4.54%. D2 PASSED (V1, alpha 1%, plug-in p 0.0054).
Plan section 11 fallback 1 (V2/V3, higher alpha) cannot change D1. The author overrides D1 only and proceeds to F1
because D2 passed, before any Cert or Test track is touched. Hard rule 5 and plan section 8 are unchanged. The claim
"accumulating evidence lowers error compared with a single frame" is dropped from all write-ups; D1 is reported in the
paper as a pre-specified check that failed.

- **F1 record (2026-11-14):** F1 commit 3de4cb99cfb0658e9d659bb1e31171e3187ff87c. `configs/frozen_r8.yaml` SHA-256
  4f7a79156a4486b6b5f97515457513e6df2a2bfeed2a41b6fa05936bd1141d24 (working-copy bytes), git blob
  4797e118953a96f0d7f8bc951f8c8ae356028558. Stage 3 onward must check this hash before reading the file.

---

## POC-2 (R8) Stage 3: validity tests (3a synthetic, 3b resampling)

### Part A: fixed definitions (committed BEFORE any Stage 3 number was computed)
Confirm phase: F1 is closed (commit 3de4cb9). Nothing in `configs/frozen_r8.yaml` changes. delta = 0.1 throughout.
The only numbers computed before this commit are the B1 grid and order in (c), from the 261 design tracks only.

(a) **Freeze check.** Every Stage 3 script first checks the SHA-256 of `configs/frozen_r8.yaml`
(4f7a79156a4486b6b5f97515457513e6df2a2bfeed2a41b6fa05936bd1141d24) and of `docs/poc2-implementation-plan-r8.md`
(b0ad6292138732f40639753b6e5b76d050cfc30da137cc2b1d8f0c0e85174bd2), the values recorded at F1, and stops if either
differs (`src/r8_frozen.py`). Every other file read is hash-checked against the summary JSON that wrote it (hard rule 2).

(b) **Commit rule (track-level certifier).** Variant, working grid (Stage 1 indices), threshold order, alpha grid and
delta are read only from `frozen_r8.yaml`. The same frozen order is used at every alpha. Per draw: per-track loss
L_i = wrong_i - alpha * committed_i + alpha per lambda (tracks-by-lambdas table), p = `src.certify.hb_pvalue(mean L,
n_tracks, alpha)`, test in the frozen order at level delta, stop at the first p > delta; the certified threshold is the
last one that passed; if the first fails, nothing is certified. Tracks with no passing frame count as not committed.

(c) **B1 (frame-level Learn-then-Test).** Per-frame margin = top minus second of (log prior + frame log-likelihood),
i.e. the V1 score of a single frame. A frame is accepted if it passes the gate and its margin >= threshold
(gate-failing frames are never accepted); wrong = accepted and top class != label. Every frame is its own unit:
L_f = wrong_f - alpha * accepted_f + alpha, p = `hb_pvalue(mean L, n_frames, alpha)`, same fixed-sequence engine at delta.
  - Grid and order, from the 261 design tracks only, by the Stage 2 Part A rules applied to frames: candidate grid =
    the Stage 1 `make_grid` rule on the design per-frame margins (0 plus 199 log-spaced points); commit fraction =
    accepted frames / all design frames; working grid = rule (h) at the primary alpha 1% (c >= 70% and e <= alpha,
    30 points evenly by index); start = rule (c) (argmin of mean L at 1%, ties to the larger threshold), then relax
    one point at a time. The same order is used at every alpha.
  - Declared: B1 is named in `frozen_r8.yaml` only through plan section 9, so this grid and order are specified after
    F1. They use design data only and were fixed before any Stage 3 number. Values: see "B1 values" below.

(d) **True risk and violation.** True risk of a method at a threshold = the population mean of its loss at alpha
(track-level: mean of L_i over population tracks; B1: mean of L_f over population frames). A run violates if a
threshold was certified and its true mean loss > alpha (equivalently, selective error > alpha). A run that certifies
nothing does not violate. Violation rate = violating runs / runs.

(e) **3a synthetic** (new file `src/r8_stage3_synth.py`; `src/certify_synth.py` is not changed).
  - Evidence model: frame evidence x_it = a_i + e_it with latent track difficulty a_i ~ N(m, r) and e_it ~ N(0, 1 - r),
    30 frames per track. A frame is wrong iff x_it < 0. m is set so the frame error is 7.4% (plan's single-frame
    error). r is solved numerically so that the within-track correlation of the frame-wrong indicator equals
    rho_sim, for rho_sim in {0, 0.3, 0.45} (the quantity POC-1 measured as 0.446).
  - Why this model: in POC-1's simulator, a design with frames independent given a track error rate never broke the
    frame-level certifier (accepted errors lose their clustering); only confident whole-track errors did. Here a
    strongly negative a_i gives a confidently wrong whole track, as seen in the real design set (Stage 2 D3), without
    forcing it. No gate in 3a.
  - Commit rule: S_t = x_i1 + ... + x_it, margin |S_t|, commit at the first t with |S_t| >= lambda, wrong iff S_t < 0
    there. B1: frame margin |x_it|, wrong iff x_it < 0.
  - Grids: per rho, one fixed candidate grid per method from a pilot 261-track design sample (Stage 1 `make_grid`
    rule). Each run draws a fresh 261-track design sample and picks the working grid and order by rules (h) and (c)
    at alpha 1%, then a fresh 784-track Cert sample. 1,000 runs per rho. alpha = 1% (primary) only.
  - True risk: a 200,000-track population per rho, evaluated once on the candidate grids.
  - All seeds fixed in code.

(f) **3b resampling.** Population = the 1,046 non-design tracks; their commit-rule and B1 outcomes are computed here
for the first time (allowed after F1). Track-level outcomes from `track_table_pool.npz` (hash-checked against
`stage1_summary.json`) with labels from `train_release.npz`; B1 from per-frame margins of the pool frames. True risk
of every threshold is computed exactly on the 1,046 tracks (B1: on all their frames). 2,000 draws of 784 tracks with
replacement (fixed seed; a track drawn twice counts twice, its frames too). Each draw: track-level certifier at
n = 784 and B1 at n = frames in the draw. Reported for every alpha in the frozen grid; the primary alpha 1% is the
headline.

(g) **P1 scoring** (frozen P1; scope fixed here). At the primary alpha 1%, with two-sided 95% Clopper-Pearson bounds on
the violation rate:
  - Track-level part: upper bound <= delta + 0.02 = 0.12 in 3b AND in 3a at rho_sim 0, 0.3 and 0.45.
  - Frame-level part: lower bound > delta = 0.10 in 3b AND in 3a at rho_sim 0.45 (the real-data level).
  - 3a rho_sim = 0 is a negative control for B1 (frames independent, B1 expected valid); rho_sim = 0.3 is reported.
  - Other alphas are reported, not scored. P1 PASS only if every part holds; a failed part is reported as failed.

(h) **Extra checks.**
  - Three p-values hand-computed with exact arithmetic (Python `fractions`, no scipy): the zero-error case at n = 784,
    alpha = 1%; one case where the Hoeffding term is the minimum; one where the Bentkus term is. Engine values must
    match to 1e-9 relative.
  - Certificate B max-p rule on a synthetic case where the error promise holds and only the lateness promise is
    broken: p = max(p_error, p_late) must not certify there, and must certify when both promises hold.

(i) **If B1 does not visibly break on real data,** it is reported as measured (plan Stage 3); nothing is changed.

(j) **Outputs:** `runs/20261003_stage1/poc2/stage3/` b1_design.json (this step), stage3a_summary.json,
stage3b_summary.json, each with the hashes of what it read.

#### B1 values (from `src/r8_stage3_b1design.py`, design set only)
Computed 2026-11-14 after the Part A rules commit (eb4c332). Script SHA-256 a7084123571bc872c7bb08a3981bfe9ab1d72f5a56c5e37103ccfdf53f2a45bd;
output `runs/20261003_stage1/poc2/stage3/b1_design.json` SHA-256 8fb7e1c534ce1d923595e18794327afa70d654f033d4b55196a4dc2a169d22c9.
Files opened: seed0.npz, train_release.npz, scores_train_release.npz, frame_evidence.npz, head.npz,
track_table_design.npz (not track_table_pool.npz). 7,830 design frames of 261 tracks; 326 fail the gate (never accepted).
  - Working grid: 20 points (the useful range has 20 candidate points, so all are kept), candidate indices 131 to 150,
    thresholds 276.6808200078421, 283.49861752219556, 290.4844149830051, 297.64235214096715, 304.9766707559528,
    312.49171711065424, 320.19194458617164, 328.08191630106586, 336.1663078154409, 344.44990990165866,
    352.9376313833293, 361.634502044255, 370.5456756090572, 379.67643279725064, 389.03218445257073,
    398.61847474941555, 408.440984478297, 418.5055344122509, 428.8180887561991, 439.38475868130973.
  - Design-set frame coverage 83.19% (position 0) to 70.74% (position 19); frame selective error 0.921% to 0.253%.
  - Order: start at working position 19 (threshold 439.38475868130973; mean L 0.471% at 1%), then relax to position 0:
    [19, 18, ..., 0]. The start is the top of the working grid: rule (h)'s 70% coverage floor ends the range there.

### Part B: results (fill from the printout after the run)
- Date: 2026-11-14 / Stage: R8 Stage 3 / Run tag: runs/20261003_stage1/poc2/stage3
- Git commit of Part A: eb4c332 (rules), d9b1e48 (B1 values), b35805e (b1_design.json) | Git commit of the code: a044722
- Outputs: stage3a_summary.json SHA-256 b6631a1f617ce619616ee58e76f85bd88fe5338366d3489a1313d85b0b8c420d (180.5 s);
  stage3b_summary.json SHA-256 7591a438b569e7231f3a40d8da6420a81b6f65791e707df4c0101338a0bf7367 (records the 3a hash).
  Both record frozen_r8.yaml SHA-256 4f7a7915...1d24 (freeze check passed).
- 3a violation rates at 1% (track / B1; certifying share in brackets), 1,000 runs each:
  - rho_sim 0 (r 0.000; realised frame error 7.40%, wrong-indicator correlation 0.000): track 0.20% [CP 0.02, 0.72]%
    (certified 100.0%) / B1 0.00% [0.00, 0.37]% (certified 0.0%; empty B1 working grid in 924 runs)
  - rho_sim 0.3 (r 0.599; 7.44%, 0.300): track 0.80% [0.35, 1.57]% (22.4%; 9 empty grids) / B1 1.70% [0.99, 2.71]%
    (1.9%; 691 empty grids)
  - rho_sim 0.45 (r 0.762; 7.41%, 0.451): track 0.20% [0.02, 0.72]% (2.5%; 96 empty grids) / B1 4.80% [3.56, 6.31]%
    (4.9%; 629 empty grids)
- 3b violation rates per alpha (track / B1; certifying share), 2,000 draws of 784 from the 1,046 pool tracks:
  0.5%: 1.90% [1.35, 2.60]% (4.7%) / 17.00% [15.38, 18.72]% (60.0%);
  1%: 1.40% [0.93, 2.02]% (19.8%) / 20.00% [18.27, 21.82]% (99.9%);
  2%, 3%, 5%: 0.00% [0.00, 0.18]% for both (track certified 95.7%, 100%, 100%; B1 100%).
- P1: FAIL (track_part_3b PASS, frame_part_3b PASS, track_part_3a PASS, frame_part_3a FAIL).
- Reason for the 3a frame-part fail, as measured: B1's rule (h) found no usable threshold in most synthetic runs
  (629/1000 at rho 0.45), so its violation rate was capped by its 4.9% certifying share; when it did certify it violated
  in 48 of 49 runs (rho 0.45) and 17 of 19 (rho 0.3). Reported as failed; nothing changed.
- Real-data result (3b, alpha 1%): B1 violation 20.0% [18.3, 21.8]% vs delta 10%; track-level 1.4% [0.9, 2.0]%.
  H1 supported on real data.
- Observation for Stage 4: at alpha 1% the track-level certifier certified in 19.8% of 3b draws (pool mean loss at the
  start threshold 0.503% vs 0.11% on the design set); at 2% in 95.7% with 0 violations. P2 at 1% is at risk; per plan
  section 11 it will be reported with the certifiability curve; the primary alpha stays 1%.
- Hand p-values: PASS (3 exact-arithmetic cases, relative error <= 5.1e-14; min term hoeffding, hoeffding, bentkus) |
  Certificate B max-p check: PASS (p_error 0.00182; lateness broken -> p 1, not certified; both kept -> p 0.00182)
- Environment observation: test_ledoit_wolf_matches_sklearn passes as of 2026-11-14 on the same scikit-learn 1.9.1 that was blocked during Stage 1, which points to the Application Control policy rather than a package upgrade; the Stage 1 entry is left unchanged.
- Checks: 3b frozen_grid_matches_stage1_grid PASS, gate_flags_reproduce PASS, pool_commit_table_reproduces PASS,
  b1_top_equals_stored_top1 PASS, pool_size_matches_frozen PASS — ALL PASS (checks): True. P1 (a pass level, not a
  check) FAIL, as above.
- Decision: P1 reported as FAIL. Nothing frozen changed. Proceed to Stage 4 with the frozen primary alpha 1%.
- Deviation from the plan? Declared in Part A: B1 grid and order specified after F1 from design data only (Part A (c));
  P1 scope fixed in Part A (g) before any Stage 3 number. No other deviation.
- Next step: Stage 4 only after the author reviews Stage 3.

---

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

---

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

---

## POC-2 (R8) Stage 6: published baselines B5 (Ringel et al.) and B6 (Boursinos and Koutsoukos), comparison table

### Part A: fixed definitions (committed BEFORE any B5/B6 number was computed)
Confirm phase (F1 commit 3de4cb9). Nothing in `configs/frozen_r8.yaml` changes. delta = 0.1 throughout. Every decision
below was made by the author before any B5 or B6 number existed. Every value fixed from data is fixed from the 261
design tracks only (values block below, computed after this rules block was committed).

#### Papers (local copies in `docs/papers/`, git-ignored, not redistributed)
- **[R]** L. Ringel, R. Cohen, D. Freedman, M. Elad, Y. Romano, "Early Time Classification with Accumulated Accuracy Gap
  Control," Proc. 41st Int. Conf. on Machine Learning (ICML), PMLR vol. 235, pp. 42584-42600, 2024. Published PMLR
  version, from https://proceedings.mlr.press/v235/ringel24a.html via its PDF link
  https://raw.githubusercontent.com/mlresearch/v235/main/assets/ringel24a/ringel24a.pdf (the guessed
  .../v235/ringel24a/ringel24a.pdf returned 404). Local `docs/papers/ringel24a.pdf`, SHA-256
  cc1ffe7245b21ab282569a522b3b308493c1b1e982cc20bc06098359614b1039. Page numbers below are the PDF's printed pages 1-17.
- **[BK]** D. Boursinos, X. Koutsoukos, "Selective Classification of Sequential Data Using Inductive Conformal
  Prediction," 2022 IEEE Int. Conf. on Assured Autonomy (ICAA), pp. 46-55, doi:10.1109/ICAA52185.2022.00015 (year
  confirmed 2022 on the PDF). From
  https://cs.vuse.vanderbilt.edu/koutsoxd/www/Publications/Selective_Classification_of_Sequential_Data_Using_Inductive_Conformal_Prediction.pdf
  (IEEE Xplore print). Local `docs/papers/boursinos_koutsoukos_selective_classification.pdf`, SHA-256
  af0f13ce31e6c8fadad7e13f04d8445bf811c183569c10b7be2f022a1ba5b557. Page numbers below are printed pages 46-55.

#### Guiding rule for every point the papers leave open (author's decision)
Choose the option most faithful to the paper and not weaker than the paper's own setup, fixed from the 261 design tracks
only, and write the reason. Where we cannot be faithful, it is a declared difference.

(a) **Freeze check and inputs.** Every Stage 6 script calls `load_frozen()` first. All inputs are hash-checked as in
Stages 4 and 5 (pool loader, train_meta.csv cross-check, stage4_summary.json e66e45f2...d2cc, r8_stage4.py
fae67390...45c2, baselines_design.json, b1_design.json, stage3a/3b summaries); the Stage 4 splits are regenerated with
the hash-checked Stage 4 code and must reproduce every Stage 4 Test coverage exactly. The two PDFs are hash-checked
against the SHA-256 values above. Cert = calibration, Test = evaluation, in every re-split (seeds 0 to 4).

(b) **Common to B5 and B5c (Ringel et al.).**
  - Classifier output per frame t: the V1 accumulated score of Stage 1 ("their rule on the same accumulated score",
    plan section 9). Prediction Y_t = V1 top class at t. Before a track's first passing frame there is no evidence:
    no prediction, and no stop is possible there (as in Stage 1).
  - Y_full = the V1 top class after the track's last frame (its own t_max: 30 frames, or 29 for the one short track).
    The paper's rule always stops at t_max at the latest ([R] eq. 6, p. 3), so B5/B5c never escalate.
  - Gate: gate-failing frames add no evidence (same accumulated score as the commit rule). A track with no passing
    frame has no prediction at any frame: it is counted as not committed (rejected as unknown) and its gap loss is 0
    (the full prediction cannot be correct).
  - Loss: L_gap = (1[Y = Y_full] - 1[Y = Y_early])_+ per track ([R] eq. 3, p. 2).
  - **Score in [0, 1].** [R] uses a confidence in [0, 1], e.g. the largest class probability ([R] p. 3, §5.2 p. 8).
    Our accumulated V1 margins are in log-likelihood units and reach thousands, so the largest posterior saturates at
    1.0 (share measured on the design set below). Faithful fix: pi(X_t) = F_design(M_t), the empirical CDF of the V1
    margins M_t over all design-set frames with evidence (F(x) = share of design margins <= x); pi = -1 before the first
    passing frame (cannot stop). This is a monotone transform, so the stopping rule is the paper's rule on the same
    score, the [0, 1] scale and the grid resolution are the paper's, and the guarantee holds for any score.
  - Grid: Lambda = {0, 0.01, ..., 1} plus infinity (never stop early), Delta = 0.01 as in [R] §5, p. 7.
  - p-value: CDF_bin(n * R_hat; n, alpha), the exact binomial tail for a 0/1 loss ([R] §3, p. 4).
  - alpha = the gap tolerance, run at every alpha of the frozen grid {0.5, 1, 2, 3, 5}%; the primary alpha 1% is the
    pre-registered setting ([R] does not require a value; its experiments use 10% and 5%, Table 1 p. 8, Table C.3
    p. 13). delta = 0.1 (project-wide). **Declared difference:** [R] uses delta = 1% (§5, p. 7).

(c) **B5 = Ringel marginal method** (pre-registered as named in plan section 9). [R] §3 p. 4 and Algorithm A.3 p. 12:
one threshold lambda for all frames; stop at the first t with pi(X_t) >= lambda, or at t_max. Fixed-sequence test over
Lambda from the largest value down at level delta, stop at the first p > delta; lambda_hat = the smallest value that
passed, or infinity if the first fails. Calibration set = all 784 Cert tracks ([R] §5.1: D_cal = D_cal-1 union D_cal-2).

(d) **B5c = Ringel conditional method** (declared additional baseline: [R] presents the marginal method as a warm-up,
§3 p. 4, and calls the conditional one its main contribution, p. 2). [R] §4, Algorithms 1 and 2 (pp. 5-7), with one
threshold per frame index t = 1..30 (lambda_t; a track stops at its own t_max at the latest).
  - Cert split into D_cal-1 and D_cal-2 of 392 tracks each (equal halves, as in [R] §5.1 p. 7): the first 392 tracks of
    a permutation of the Cert tracks drawn with `np.random.default_rng([20261116, seed])`.
  - Stage 1 (D_cal-1): for t = 1..30, lambda_t = the smallest grid value with empirical accumulated gap R^t_gap <= alpha
    among tracks halted by t; infinity if none or if no track halts by t (Algorithm 1).
  - Stage 2 (D_cal-2): test the configurations from t = 30 down to 1, each through its finer hypotheses t' >= t, with
    the max-p rule and the binomial p-value; stop at the first failure (Algorithm 2).

(e) **B6 = Boursinos and Koutsoukos selective classifier** ([BK] §IV pp. 47-48, §VI pp. 49-52).
  - **Per-frame p-values (ICP).** Nonconformity of class j for a frame = d_j / min_{k != j} d_k, the nearest-centroid
    ratio of [BK] eq. 9 (p. 50), with d_k = the head's Mahalanobis distance to class mean k (sqrt of `mahal_sq`).
    p_j = share of calibration nonconformity scores >= the frame's score for class j ([BK] eq. 10, p. 50; no +1).
    **Declared difference:** [BK] computes the distances in a Siamese-network embedding ([BK] §V, p. 48); we use the
    frozen ResNet-18 embedding with the head's whitening (Mahalanobis = Euclidean after whitening), the same formula.
  - **Data roles, design set only** (all disjoint by track; [BK] §VII-A p. 52 also keeps calibration and ECDF
    sequences disjoint): the 261 design tracks are permuted with `np.random.default_rng(20261117)` and split into three
    parts of 87 tracks: ICP calibration (nonconformity scores of the true class), ECDF calibration (the combined
    null p-values for eq. 22), and selection (a, b and theta). Only gate-passing frames are used in all three.
    The head was fitted on the fine-tune frames, so those are not used as ICP calibration data (they are its proper
    training set in [BK]'s sense).
  - **Window.** The last K passing frames up to t; no decision before K passing frames exist (t < K: the window is not
    full; [BK] defines combinations only for full windows, §VI-B p. 50). K = 5 (pre-registered, plan section 9);
    K = 9 (declared extra: the window of [BK]'s risk-coverage figure, Fig. 6 p. 54). [BK] evaluates K = 2 to 9 and
    recommends none (Tables III-IV, p. 54), so K = 5 is the plan's choice, not the paper's.
  - **Combination (selective classifier).** ECDF Sum and ECDF Min ([BK] eq. 22 p. 52; the only methods [BK] uses for
    its risk-coverage results, §VII-D p. 53, Tables III-IV): p_comb(j) = F(f(p_j over the window)), f = sum or min,
    F = the empirical CDF of f over all full windows of the true class in the ECDF-calibration tracks.
    **Interpretation of plan section 9:** "arithmetic mean" there is read as [BK]'s arithmetic-mean merging function
    (2 x mean, eq. 15, p. 51), which [BK] evaluates only for calibration (Table III, p. 54). It is reported only in the
    calibration check (g), not used as a selective classifier.
  - **Assurance evaluator.** credibility = the largest combined p-value, confidence = 1 - the second largest ([BK]
    eqs. 1-2, p. 47); k = a * credibility + b * confidence ([BK] eq. 7, p. 48). a, b by grid search over [-1, 1] in
    steps of 0.1 (441 pairs; [BK] gives no resolution) minimising the AURC ([BK] eq. 6, p. 48) over the selection-split
    frames with full windows; ties go to the first pair in (a, b) ascending order.
  - **Threshold from the risk-coverage curve** ([BK] p. 48: theta such that (r_hat, c_hat) = (r, c)). Per-track
    adaptation (declared: [BK] decides per frame only): a track commits at its first full-window frame with k > theta,
    with label = the class with the largest combined p-value there; otherwise it escalates. theta = the smallest value
    (maximum coverage) whose track-level selective error on the selection split is <= r, with r = alpha for each alpha
    in the frozen grid; if no theta meets r, B6 commits nothing at that alpha. No certificate (as in [BK]).

(f) **Test metrics** (per re-split, per alpha, 262 Test tracks), for B5, B5c and the four B6 variants (ECDF Sum and
ECDF Min, K = 5 and 9): coverage, error among committed signs, median commit (stop) frame and median ROI size at commit,
as in Stage 4 Part A (d). For B5/B5c also the Test accuracy gap: marginal, and accumulated over the 20% and 50%
earliest-stopping Test tracks ([R] Table 1, p. 8).

(g) **Stage 6 checks** (plan section 10), at the primary alpha 1%, PASS if true in >= 4 of 5 re-splits:
  - B5 meets its own target: Test marginal accuracy gap <= alpha.
  - B5c meets its own target: Test accumulated gap over the 20% earliest, over the 50% earliest, and marginal, all <= alpha.
  - B6 combination improves calibration in the direction [BK] reports ("ECDF based methods consistently improve the
    calibration over the baseline", p. 53): ECE of ECDF Sum and of ECDF Min (K = 5) < ECE of single-frame ICP, on the
    Test frames with full windows. ECE as [BK] eq. 24 (p. 53): prediction set = classes with p > eps,
    eps_i = i/1000, i = 1..200, ECE = mean_i |acc(eps_i) + eps_i - 1|. The arithmetic-mean merge (2 x mean, eq. 15) and
    K = 9 are reported, not scored.
  - Also reported at 2%. A failed check is reported as failed.

(h) **Comparison table (plan section 9).** Columns: Ours (commit rule), B1, B2, B3, B4, B5, B5c, B6 (ECDF Sum, K = 5),
B6 (ECDF Min, K = 5); the K = 9 variants in a second table. Two alphas: **1% (primary, pre-registered)** and **2%**
(declared extra: the smallest alpha certified for the commit rule in all five re-splits, Stage 4). Numbers are the
**median (min to max) over the five re-splits**; a method certified in fewer than five re-splits shows "k/5 certified",
with coverage 0 for the others and error / frame / size medians over the certified re-splits only.
Rows, and where each number comes from:
  1. Type of promise: certified real error among committed signs (Ours, B2, B3, B4); certified per-frame error with
     frames assumed independent (B1); accuracy gap to the full-sequence answer, marginal (B5) or conditional on the
     accumulated halt time (B5c); none (B6). From the methods' definitions.
  2. Valid on linked frames: violation rate vs delta at the alpha of the column, from Stage 3 (3b real data; 3a at
     rho_sim 0.45) for Ours and B1. B2, B3, B4: "valid by construction (track-level certificate), not measured".
     B5, B5c: "valid for its own target under i.i.d. tracks, not measured". B6: "no promise".
  3. Sign coverage at the alpha (Test): Stage 4 (Ours, B2, B3, B4), Stage 6 (B5, B5c, B6). B1: "n/a - B1 certifies
     frames, not signs, and was not run on the Stage 4 splits" (author's decision: B1 numbers come from Stage 3 only).
  4. Error among committed signs (Test): as row 3.
  5. Smallest certifiable alpha at n = 784: Stage 4 certifiability curve (Ours, B2, B3, B4); B1: the smallest alpha
     at which B1 certified in at least half of the Stage 3b draws; B5, B5c, B6: "n/a (no certified real error)".
  6. Median commit frame (Test): Stage 4 (Ours, B2, B3, B4), Stage 6 (B5, B5c, B6); B1 n/a.
  7. Median sign size at commit (px, Test): Stages 4 and 5 sources as row 6; B1 n/a.
  8. Can escalate: yes (Ours, B1, B2, B3, B4, B6); no (B5, B5c: always stop by t_max).

(i) **Integrity.** Stage 6 changes no frozen value and no earlier result; Stage 4 numbers are read, not recomputed
(only reproduced as a check). B5, B5c and B6 are specified after F1 from the papers and design data only.

#### Design-set values (from the Stage 6 design script, design set only)
Computed 2026-11-14 after Amendment 1 (12d6bdb), with `src/r8_stage6_design.py` committed first (abe7522; SHA-256
4d67843a43f0a9654a8f639ab8ac4d925ed39f75fdbfe9f10f1ed72044115367). Output
`runs/20261003_stage1/poc2/stage6/b5_design.json` SHA-256 39d4e9b6f614f3062ef3e7af98a66ef2b4549a07e80106b827083a9b4c3e1924.
Both paper hashes checked. Files opened: seed0.npz, train_release.npz, scores_train_release.npz, frame_evidence.npz,
head.npz, track_table_design.npz (not track_table_pool.npz).
  - B5/B5c score transform: F_design = the sorted V1 margins of the 7,706 design frames with evidence (stored in full in
    the JSON); margin 0.9752 to 39,770; quantiles 1/10/25/50/75/90/99% = 63.94, 699, 2,606, 7,529, 13,890, 21,370,
    32,680. Each grid step of 0.01 in pi is one percentile of the design margins.
  - Reason, measured: the largest V1 posterior is exactly 1.0 in float64 in 7,674 of the 7,706 design frames with
    evidence (99.58%), so the paper's "largest class probability" would give nearly every frame the same score.
  - B6: nothing fitted on the design set (Amendment 1).

### Part B: results (fill from the printout after the run)
- Date: 2026-11-14 / Stage: R8 Stage 6 / Run tag: runs/20261003_stage1/poc2/stage6
- Git commit of Part A: 2127a4b (rules), 12d6bdb (Amendment 1), d1b6a9f (design-set values) | Git commit of the code:
  167557b (src/r8_stage6.py SHA-256 87a4afe1047fffd6df87ac01568d7c68435bb257270f4b50d7715e0f4bdeb507;
  src/r8_stage6_methods.py 2b4fbdad8eae0daef7b3ddc22208b789a9cd1275d6bdc5ab36a97c3481c31ab4)
- Output: stage6_summary.json SHA-256 59ef05a43e43963bc3cf5a7911b7d262234005df1d4cc8909f5f5e18e9d1c25c (28.2 s). All
  Stage 4 Test coverages reproduced from the regenerated splits; both paper hashes checked.
- **B6 exceeds the commit rule on coverage and earliness at both alphas.** At 1% the commit rule certifies nothing
  (0/5) while B6 (K = 5) covers 92.7% of Test signs (median; ECDF sum 92.4-95.8%, ECDF min 92.7-95.4%) at median frame
  5 and 22-24 px. At 2% B6 (K = 5) covers 95.8% (sum 93.9-96.9%, min 94.7-97.7%) at frame 5 and 22-24 px, against the
  commit rule's 92.7% (91.2-93.5%) at frame 10 (10-11) and 27 px (26-28). B6 gives no certificate.
- **B6 Test error above its target r** (error among committed Test signs, seeds 0-4):
  - r = 1%: ECDF sum K5 1.24, 0.82, 0.81, 1.59, 1.23% (3/5 above 1%); ECDF min K5 1.23, 0.82, 0.81, 0.80, 2.06% (2/5);
    ECDF sum K9 1.23, 0.82, 0.41, 1.20, 0.41% (2/5); ECDF min K9 1.21, 1.21, 0.82, 2.78, 0.83% (3/5).
  - r = 2%: ECDF sum K5 1.63, 1.20, 1.59, 1.57, 4.78% (1/5); ECDF min K5 1.61, 1.59, 1.20, 1.95, 3.98% (1/5);
    ECDF sum K9 1.61, 1.59, 1.57, 2.75, 5.58% (2/5); ECDF min K9 1.21, 1.62, 1.98, 3.16, 5.18% (2/5).
  - For comparison, the commit rule's Test error where certified (2%) is 0.41-1.22%. These are Test descriptions over
    five re-splits, not violation rates like Stage 3.
- **B5 / B5c real error vs their gap target** (alpha = gap tolerance; seeds 0-4):
  - alpha 1%: B5 Test gap 0.76, 0.38, 0.38, 1.15, 2.29% but real error among committed 3.09, 2.30, 2.30, 3.44, 4.98%;
    B5c gap 0.00, 0.76, 0.38, 0.00, 0.00% but real error 2.32, 2.68, 2.30, 2.29, 2.68%.
  - alpha 2%: B5 gap 0.76, 1.53, 0.38, 1.53, 3.82%, real error 3.09, 3.45, 2.30, 3.82, 6.51%; B5c gap 0.76, 1.53, 0.38,
    1.53, 1.15%, real error 3.09, 3.45, 2.30, 3.82, 3.83%.
  - Controlling the gap to the full-sequence answer does not control the real error (plan section 9: B5 bounds a gap,
    not the real error). B5/B5c cover 98.9-100% (they never escalate). B5c stops late where few thresholds are finite
    (2/30 finite in seeds 3 and 4 at 1%: median frame 29, 64-67 px).
- **Stage 6 checks** (Part A (g), alpha 1%, >= 4 of 5):
  - B5 own target (Test marginal gap <= 1%): PASS, PASS, PASS, fail, fail -> 3/5 -> **FAIL**.
  - B5c own target (gap over the 20% earliest, the 50% earliest, and marginal, all <= 1%): **PASS 5/5**.
  - B6 calibration direction (ECE of ECDF sum and ECDF min, K = 5, below single-frame ICP): **PASS 5/5**
    (e.g. seed 0: 0.0422 -> 0.0170 / 0.0148). Reported, not scored: the arithmetic-mean merge (x2) is worse than single
    ICP in every re-split (ECE 0.066-0.082 vs 0.021-0.053); K = 9 also improves with ECDF sum and min.
- **Comparison table, alpha 1% (primary), B6 K = 5; median (min to max) over 5 re-splits:**
  - Promise: Ours, B2, B3, B4 certified real error; B1 certified per-frame error (frames assumed independent); B5 gap,
    marginal; B5c gap, conditional on halt time; B6 none.
  - Valid on linked frames: Ours 3b 1.4% [0.9, 2.0], 3a rho .45 0.2%; B1 3b 20.0% [18.3, 21.8], 3a rho .45 4.8% (delta
    10%); B2-B4 valid by construction, not measured; B5, B5c valid for own target, not measured; B6 no promise.
  - Coverage: Ours 0.0% (0/5 certified); B1 n/a; B2 0.0% (0/5); B3 88.2% (87.0-90.5); B4 82.4% (81.3-84.0); B5 99.6%
    (98.9-100.0); B5c 99.6% (98.9-100.0); B6 sum 92.7% (92.4-95.8); B6 min 92.7% (92.7-95.4).
  - Error among committed: Ours n/a; B2 n/a; B3 0.43% (0.00-1.27); B4 0.00% (0.00-0.45); B5 3.09% (2.30-4.98); B5c
    2.32% (2.29-2.68); B6 sum 1.23% (0.81-1.59); B6 min 0.82% (0.80-2.06).
  - Smallest certifiable alpha (n = 784): Ours 2.0%; B1 0.5% (3b rule); B2 2.0% (2.0-3.0); B3 1.0%; B4 1.0%; B5, B5c,
    B6 n/a.
  - Median commit frame: B3 30; B4 30; B5 7 (5-8); B5c 19 (8-29); B6 5. Median size px: B3 75 (70-81); B4 76 (72-81);
    B5 24 (22-25); B5c 35 (26-67); B6 sum 24 (22-24); B6 min 23.5 (22-24). Ours, B2: n/a (not certified).
  - Can escalate: yes for all except B5 and B5c.
- **Comparison table, alpha 2% (declared extra), B6 K = 5:** coverage Ours 92.7% (91.2-93.5, 5/5); B2 81.3%
  (0.0-82.1, 3/5); B3 90.1% (89.3-90.8); B4 90.1% (87.8-91.2); B5 99.6%; B5c 99.6%; B6 sum 95.8% (93.9-96.9); B6 min
  95.8% (94.7-97.7). Error: Ours 0.82% (0.41-1.22); B2 0.94% (0.93-2.33); B3 0.85% (0.42-1.26); B4 0.43% (0.00-1.27);
  B5 3.45% (2.30-6.51); B5c 3.45% (2.30-3.83); B6 sum 1.59% (1.20-4.78); B6 min 1.61% (1.20-3.98). Frame: Ours 10
  (10-11); B2 16 (16-17); B3 30; B4 30; B5 5 (4-5); B5c 11 (2-18); B6 5. Size px: Ours 27 (26-28); B2 32 (31-34); B3
  74.5 (70-81); B4 75 (70-81); B5 23 (21-23); B5c 26 (22-34); B6 23. Row 2 at 2%: Ours and B1 3b 0.0% [0.0, 0.2].
- **K = 9 tables** (both alphas) are in stage6_summary.json (tables "K9_0.01" and "K9_0.02"): B6 coverage 93.5-96.2%
  (medians) at frame 9 and 25-26 px; errors as listed above.
- **Two cells that could be misread:**
  - B1 "smallest certifiable alpha 0.5%" follows the Part A rule (certified in at least half of the 3b draws: 60.0%),
    but B1 broke its promise in 17.0% of the 3b draws at 0.5% (delta 10%); its certificate at 0.5% is not trustworthy.
  - Row 2 at 2% shows 0.0% violations for both Ours and B1 because every threshold certified at 2% in 3b had true risk
    below 2%; it does not show B1 is valid at 2% in general.
- Checks: frozen_grid_matches_stage1_grid, gate_flags_reproduce, pool_commit_table_reproduces, b1_top_equals_stored_top1,
  meta_roi_matches_stage1_sizes, b2_frames_all_pass_gate, stage4_coverages_reproduce all PASS — ALL PASS (checks): True.
  Stage 6 checks: B5 FAIL (3/5), B5c PASS, B6 calibration direction PASS.
- Decision: results reported as measured; nothing frozen changed. In the write-up, the commit rule's edge over B6 is
  the certificate (a promise on real error that held where certified), not coverage or earliness.
- Deviation from the plan? Declared in Part A and Amendment 1: B5c and the K = 9 B6 variants added; B6 reads the plan's
  "arithmetic mean" as [BK]'s merge (calibration check only); score transform for B5/B5c; per-track adaptation of B6;
  B6 calibrated on Cert thirds. No other.
- Next step: Stage 7 only after the author reviews Stage 6.

### Amendment 1 to Stage 6 Part A (2026-11-14, before the design script was written or run)
Made after the author's review of Part A (2127a4b); no B5, B5c or B6 number and no Stage 6 design-set value existed.
Replaces the "Data roles, design set only" item of (e). Everything else in (e) is unchanged.
- **B6 gets the same calibration budget as the other methods.** In each Stage 4 re-split, the 784 Cert tracks are
  permuted with `np.random.default_rng([20261118, seed])` and split into three disjoint parts of 261, 261 and 262 tracks
  (in that order): ICP calibration (nonconformity scores of the true class), ECDF calibration (the combined null
  p-values for [BK] eq. 22), and selection of a, b and theta. Only gate-passing frames are used. Test is never used.
- Reason: under the original (e) B6 had about a third of the calibration data the other methods get (87 design tracks
  per role), which would weaken it for reasons unrelated to the method.
- What is still fixed without Cert data: nothing in B6 is fitted on the design set. The (a, b) grid (step 0.1 over
  [-1, 1]), the window sizes K = 5 and 9, and the split seed are constants stated in Part A, not fitted. The only
  design-set quantity left in Stage 6 is the B5/B5c score transform of (b) (design ECDF of the V1 margin).

---

## POC-3 (R9) Stage 0: repo setup

- Date: 2026-11-14 / Stage: R9 Stage 0 / Run tag: none (no data read)
- Config hash / checkpoint hash / git commit: plan `docs/poc3-implementation-plan-r9.md` SHA-256
  c0639973c70f35ac5cda1f11c49bf50a1639c632c48ccec8889d60a5ded03db4 (working-copy bytes), git blob
  0abe7fa51af3b499cd79218e46c83790dbfa2e99 (line-ending independent); committed in 8c6c902 on branch `poc3`. POC-2
  tagged `poc2-final` at 9a8d08c and pushed. No checkpoint used.
- What was done: tagged and pushed `poc2-final`; created and pushed branch `poc3`; committed plan R9 and recorded its
  hash; added the POC-3 block to `CLAUDE.md`; created `.claude/settings.json` with commit/PR attribution off; added the
  POC-3 status table to `docs/HANDOFF.md`.
- Result numbers: none (setup only).
- Checks: plan file committed and identical to its source (same SHA-256 as both copies in `POC 3 plan - FYP`) — PASS;
  plan hash recorded here and in `docs/HANDOFF.md` — PASS.
- Decision taken and why: plan R9 section 9, Stage 0.
- Deviation from the plan? No.
- Next step: Stage 1 (data audit, F-gates); not started.

---

## POC-3 (R9) Stage 1: data audit (BelgiumTSC, DITS), F-gates

- Date: 2026-11-14 / Stage: R9 Stage 1 / Run tag: runs/btsc/stage1
- Config hash / checkpoint hash / git commit: Part A 9a228a1, Amendment 1 5dc5032, Amendment 2 6f05a92; code 262c832
  (src/r9_stage1_audit.py SHA-256 3f1df93bf2623b86b5f18545b0f10970b60af4d718ce489cd147926d3176acf9); breakdown 852dedc.
  Inputs: BelgiumTSC_Training.zip SHA-256 e5f6a33c18716917d69989488b9f6af573821a7126ca1390af7e75403af1ee61,
  BelgiumTSC_Testing.zip SHA-256 1811447452f02ede83c0237f2852fa8bd278f39e40ddcb1ba01c2ed212e735fa (from
  https://btsd.ethz.ch/shareddata/). Output stage1_summary.json SHA-256
  77ba6a7042a471fa3320059d85a5ec967a8d30a613355c05c74815d03277a0dd. No checkpoint used.
- What was done: downloaded and integrity-checked the BelgiumTSC releases, audited poles, sign units and shared units,
  retried the DITS page; full record in `LOG_r9_stage1_entry.md`.
- Result numbers (from stage1_summary.json):
  - Images 4,575 / 2,520 (training / testing; plan 4,575 / 2,520; readmes 4,591 / 2,534 because of 16 / 14 duplicated
    GT rows). Listed-size check FAIL for both zips; zip integrity and image counts PASS (Amendment 1).
  - Testing: 701 poles, 848 sign units (pole, class), 53 of 62 classes; 128 poles carry more than one class.
  - Shared units: check 1 only 5, check 2 only 59, both 1; 65 removed. All 60 check-2 units matched by class, size and
    ROI only, none byte-identical (descriptive, after the gate result, changes nothing).
  - After removal: 663 testing poles, 783 sign units. Projection: Cert 398 poles (n_c 279 at 70%) or about 470 units
    (n_c about 329); alpha 1% tolerates 0 errors at either size, 2% tolerates 2, 0.5% not certifiable.
- Checks: F-A1 PASS (663 >= 600) | F-A2 PASS after removal (65 units) | F-A3 FAIL (DITS page unreachable on HTTPS and
  HTTP, 2026-11-14 11:46 UTC, curl exit 28).
- Decision taken and why: BelgiumTSC is the fresh dataset for Stages 4 to 6; Stage 7 dropped (plan R9 sections 9 and
  11). Stage 4 counts views by unique file name and stratifies by class where possible.
- Deviation from the plan? Yes: size check replaced by integrity conditions (Amendment 1); analysis unit (pole, class)
  from Stage 4 on, F-A1 still counts poles; two shared-unit checks; view order = annotation order, no camera index
  (Amendment 2). Details and process deviations in `LOG_r9_stage1_entry.md` Part B.
- Next step: Stage 2 (certification engine v2, V-checks); not started.

## POC-3 (R9) Stage 2: certification engine v2, V-checks V1 to V4

- Date: 2026-11-14 / Stage: R9 Stage 2 / Run tag: runs/20261003_stage1/poc3/stage2
- Config hash / checkpoint hash / git commit: Part A c70f15d, Amendment 1 5716fc6, Amendment 2 f19727c; population
  curves 6043706 (population_curves_L0.json SHA-256 fd7fdb44acbf6b0809258969996a5a919e185a503d40a222236a3a9d3881b019,
  L1 5cd5c3aa553f48fe9bd82e2b29066f29a59f9e0591d0143af35af0ee17986260). Final run on f19727c; output
  stage2_summary.json SHA-256 8c152579609bb972b394d8c145746eae2975f7e7c79fae640ba829f46ee3729f (earlier runs:
  stage2_summary_run1.json daea75fe...9fe, stage2_summary_run2.json e809286c...82fe). Synthetic data only; no
  checkpoint, no dataset file.
- What was done: built the exact-binomial engine with the power-aware start and fixed-sequence walk, the
  Hoeffding-Bentkus comparator and a simulator on the POC-2 Stage 3a model (36 settings: 2 error levels x 3 rho x
  3 cluster settings x 2 alpha, 2,000 runs each); ran V1 to V4 three times (Part A, Amendment 1, Amendment 2); full
  record in `LOG_r9_stage2_entry.md`.
- Result numbers (final run; V-check results identical in all three runs):
  - V1 40 / 40 plan section 6 values; V3 largest relative error 6.9e-15; V4 0 certified thresholds with no committed
    unit.
  - V2: largest sign-level violation rate 5.35% [4.40, 6.43]% (level 0, rho 0.45, fixed-30, alpha 2%).
  - Reported: frame-level B1 up to 37.45% violations with 30 frames and up to 15.35% with 3 views; the exact binomial
    certified at least as often as HB under all three HB starts in all 36 settings.
- Checks: V1 PASS | V2 PASS (36 / 36) | V3 PASS | V4 PASS.
- Decision taken and why: engine frozen (plan Stage 2). src/r9_certify.py SHA-256
  d2fd6fd63a0a65b22c73b08722bc1aa5583b51619b5720f99f6224b4fef98075, git blob 158a9c9e32ab8ec1f568f75b5c35c9992711db74;
  tests/test_r9_certify.py SHA-256 3a163ab536a2fe8c5a3c7b0ce14ef533a432d2558739b563cf176b127cdf7813, git blob
  ecb6e04429883ddbf298b22b803cd3bc70c959ef. Neither file changes from here on. Primary HB start: Amendment 2; Stages 3
  and 6 report HB under all three starts.
- Deviation from the plan? Yes: a declared added error level (P_FRAME 5%) and a view-mix cluster setting in V2; the HB
  comparator start changed twice before the freeze (Amendments 1 and 2), the exact binomial never; the population-curve
  script was run before its commit. Details in `LOG_r9_stage2_entry.md` Part B.
- Next step: Stage 3 (GTSRB exploratory run of the layer), only when the author says so.

## POC-3 (R9) Stage 3: GTSRB run of the layer on four scores (exploratory, post-hoc)

- Date: 2026-11-14 / Stage: R9 Stage 3 / Run tag: runs/20261003_stage1/poc3/stage3
- Config hash / checkpoint hash / git commit: Part A 9be85ac, design-set values 58cacfe (stage3_design.json SHA-256
  798585428c4cc345a760e347ded0c37f16c46f65a2ecf7dfa50473bacfb8e62b), Amendment 1 0c255f0; code aee1565 (freeze guard
  `src/r9_frozen.py`), 05c1df5 (scores, design script), 27ed9a7 (runner, SHA-256
  45a712ca65c0bf55c012cb0a41984a0a6a00411083540bf040628bb9d243ff7c). Output stage3_summary.json SHA-256
  aad2c03ff94487bbc4743dbedf6b76a3e51a4167f50a8d3b831ba7360fa052e2; printout stage3_printout.txt SHA-256
  21d57b80b500a5c5c56fc0dbfd28f05a4621b8daf5bc87e8624817ffb6283d69. POC-1 checkpoint 8652735...0bcc via the POC-2
  inputs (hash-checked).
- What was done (all exploratory, post-hoc): wrapped S1 (summed Gaussian margin), S2-sum (primary) and S2-min
  (Boursinos and Koutsoukos, K = 5, calibrated on two design halves), S3 (running vote margin) and S4 (current-frame
  margin) in the frozen engine; 30-point design grids, power-aware starts; certified on the five regenerated POC-2
  Stage 4 re-splits (Cert 784 / Test 262) at alpha 0.5 to 5% with the exact binomial and HB under three starts; full
  record in `LOG_r9_stage3_entry.md`.
- Result numbers (exact binomial; median over 5 re-splits):
  - Smallest certifiable alpha: 1% for every score in all 5 re-splits. alpha 0.5% has no exact-binomial start at
    n = 784 (Amendment 1).
  - At 1%: coverage S1 81.30%, S2-sum 90.84%, S2-min 89.31%, S3 85.88%, S4 56.49%; median commit frame 17 / 5 / 5 / 21
    / 2. At 2%: coverage 93.51 / 92.75 / 92.75 / 91.22 / 77.10%.
  - HB certified 1% in 5 / 5 / 2 / 1 / 0 re-splits (primary start) for S1 / S2-sum / S2-min / S3 / S4; the exact
    binomial certified wherever HB did, at the same or a looser point, from 1% to 5%.
  - Six certified Test cells above alpha (S1 3% s4 3.63%; S2-sum 2% s4 2.44%; S4 1% s0 1.57%; S4 2% s0 2.15%; S4 2%
    s4 3.00%; S4 3% s4 3.27%); every lower 95% CP bound is below alpha (descriptive, added at Part B).
- Checks: freeze check PASS | all hash checks PASS | 68 Stage 4 Test coverages reproduce PASS | POC-2 pool checks
  PASS | design starts recompute PASS. No V-checks in this stage (R-results only).
- Decision taken and why: results reported as run (plan R9 section 3). 1% in 5/5 for every score is confounded with
  the changed test, grid and start against POC-2.
- Deviation from the plan? Yes, declared in Part A and Amendment 1: S3 and S4 sequential; S2 in two variants (S2-sum
  primary, chosen before any Cert number); S2 calibrated on design halves of 141 / 120 tracks; no exact-binomial start
  at 0.5%. The review list of above-alpha cells missed S4 2% seed 0; added in Part B.
- Next step: Stage 4 (BelgiumTSC model and cache), only when the author says so.

## POC-3 (R9) Stage 4: BelgiumTSC model, cache, design/pool split, embeddings and head

- Date: 2026-11-14 / Stage: R9 Stage 4 / Run tag: runs/btsc/stage4
- Config hash / checkpoint hash / git commit: Part A 1d9e00b; code beeb114 (HEAD for all six steps; configs/
  r9_btsc_stage4.yaml SHA-256 f92bcc7403953873620c398ffdfdb93bf3c55c2a12db6698793dee502786a7a8). Checkpoint
  `runs/btsc/stage4/model/best.pt` SHA-256 83fdd0d1975f0b05f06ef45f8bfa45dc31846570f61db1be8da0299b06c5b76b (ImageNet
  weights resnet18-f37072fd.pth SHA-256 f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec, local copy).
  Output stage4_summary.json SHA-256 1c66a0119d39473f818f1add56c8a878d32ddfcc91aef4b0f5a9bd76568a3a31; split_poles.json
  SHA-256 d0d39881bd394579502aa1b63b684a0460445a67f097e708d481d7062b78c44e. Inputs hash-checked against the Stage 1
  summary (77ba6a70...a0dd) and manifest.
- What was done: one view per image (class folder, file name); dropped the 65 Stage 1 removed units; split the testing
  poles by pole (rarest-class strata, largest remainder, seed 0) into design and pool; cached whole images at 112 x 112;
  fine-tuned a 62-class ResNet-18 once on the training release with the POC-1 recipe (early stopping on 10% of
  training poles); extracted embeddings; fitted the tied-Gaussian head on the training part; added the pre-F3 guard
  `src/r9_btsc_guard.py` (pool labels refused until configs/frozen_r9_btsc.yaml is frozen). Full record in
  `LOG_r9_stage4_entry.md`.
- Result numbers (from stage4_summary.json and the step summaries):
  - Counts: training 4,575 views, 62 classes, 1,364 poles, 1,545 units; testing 2,520 views, 53 classes, 701 poles,
    848 units; after removal 2,325 views, 663 poles, 783 units. Duplicated GT rows dropped 16 / 14.
  - Split: design 133 poles, 157 units, 465 views, 40 classes; pool 530 poles, 626 units, 1,860 views, 53 classes.
    13 classes (0, 2, 3, 5, 12, 20, 25, 27, 43, 44, 46, 49, 51; 1 to 3 units each) are in the pool only.
  - Training: 18 epochs, early stop; best epoch 13, validation accuracy 0.9876 (training-release poles).
  - Head: shrinkage 0.0791; training validation head 0.9814, softmax 0.9876.
  - Design-pole accuracy (no threshold): per view head 0.9699 (451 / 465), per unit at the last view head 0.9554
    (150 / 157). Pool: row counts only.
- Checks: V4.1 PASS | V4.2 PASS | V4.3 PASS | V4.4 PASS | V4.5 PASS | V4.6 PASS.
- Decision taken and why: Part A decisions 1 to 5 (whole image, head on the training part, design accuracy without
  threshold, removed units not cached, seed 0), taken before the Part A commit. Caches, embeddings, scores and the
  checkpoint stay git-ignored; their SHA-256 values are in `LOG_r9_stage4_entry.md` Part B and stage4_summary.json.
- Deviation from the plan? Yes: whole released images cached without a further ROI crop; split by pole with
  (pole, class) as the unit; early stopping on a by-pole training-release split; training values in
  train_summary.json instead of results.json. Run facts: `worktree clean False` in every step because the Stage 4
  outputs were untracked (HEAD beeb114 throughout); a PyTorch non-writable-array warning in the train step; one
  intermittent scikit-learn ImportError in a POC-1 test before the run, 435 passed on rerun. Details in
  `LOG_r9_stage4_entry.md` Part B.
- Next step: Stage 5 (BelgiumTSC design phase and freeze F3), only when the author says so.

## POC-3 (R9) Pre-registration block, freeze F3 (plan R9 Stage 5), 2026-11-15

Written before any Cert or Test number on BelgiumTSC pool poles exists. Every value below was chosen on the 133 design
poles (157 units, 465 views) or from pool structure only, and is in `configs/frozen_r9_btsc.yaml` (committed together
with this block and its entry in `FROZEN_BLOBS`). Nothing here changes after this commit. delta = 0.1 in every phase.
Definitions: `LOG_r9_stage5_entry.md` Part A (commit 019f782) and its design-pole values (commit ec85800).

- **Plan and engine:** plan R9 git blob 0abe7fa51af3b499cd79218e46c83790dbfa2e99; engine `src/r9_certify.py` blob
  158a9c9e32ab8ec1f568f75b5c35c9992711db74; engine tests blob ecb6e04429883ddbf298b22b803cd3bc70c959ef.
- **Design-values file:** `runs/btsc/stage5/stage5_design.json` SHA-256
  115f98025f60b14aedad9c0d75df6d5deafc568d77b427e2bf9804be1d854a11, git blob b06a09f8c64bbd789bd02addab0b518ecde66d0e.
  The yaml was written by `src/r9_stage5_freeze.py` (commit 986a8e7) from a recomputation that reproduced this file
  exactly.

1. **Unit and views.** Unit = (pole, class); view order = ascending YYYYY; commit time = views used; no gate.
2. **Scores.** S1 (summed head log-likelihood margin), S2-sum (primary S2) and S2-min (window of at most the last 3
   views, one ECDF per window length), S3 (running vote margin), S4 (current-view margin). S2 halves by pole, seed
   20261115: A 66 poles (226 ICP scores), B 67 poles (ECDF sizes 80 / 80 / 79); (a, b) = (0, 0.1) for both variants.
   Stated: on the design units S2's confidence takes only two values, so its grid has 3 points and its strictest point
   commits 97.45% (S2-sum) / 95.54% (S2-min).
3. **Grids.** Rule of Part A (f) on the 157 design units; distinct points S1 30, S2-sum 3, S2-min 3, S3 4, S4 30.
   Values in the yaml.
4. **Alpha grid and primary alpha.** {0.5, 1, 2, 3, 5}%. Rule 3 (n_min = 464, f_d >= 70%, UCB90 x floor(f_d x n_min)
   <= k_max): predicted certifiable at 3% for S1 and S4 (0 / 150 committed design errors, UCB90 1.523%, n_c 443,
   6.75 <= 8) and at 5% for S1, S2-sum, S2-min and S4; S3 at no alpha (5%: 15.54 > 15). **Primary alpha 3%.** F-D1
   PASS; 10% not added. Rule 3 ignores the walk path (Part A (i), Q5).
5. **Re-splits.** Seeds 0 to 4; Test = 133 pool poles by the Stage 4 rule (25%), Cert = 397 poles. Cert / Test units:
   471 / 155, 472 / 154, 464 / 162, 471 / 155, 468 / 158. Pole lists in the yaml.
6. **Tests and starts.** Exact binomial (primary) and Hoeffding-Bentkus (starts hb primary, hb_a1, hb_bs), each with
   that seed's Cert unit count; start indices per (seed, score, alpha, rule) in the yaml. alpha 0.5% has no
   exact-binomial start in any seed ("no start possible", not tested and failed).
7. **RQ1 resampling (Stage 6).** 2,000 draws of 397 pool poles with replacement (seed [20261115, 6, d]); sign-level
   certificate counted by unit; view-level B1 on the S4 single-view margin with its 30-point view grid; all score
   variants and B1, every alpha with a start, exact binomial and HB primary start; all reported.
8. **Stage 6 V-check.** Upper 95% Clopper-Pearson bound of the sign-level violation rate <= 0.12, scored for every score
   variant at the primary alpha 3%, exact binomial. All other cells reported, not scored.
9. **Baselines.** B2 (one random view per unit, S4 margin; seeds [20261115, 7, s], design draw [20261115, 7, 100];
   30-point design grid); B5 and B5c (Ringel et al., POC-2 code; pi = F_design of the 465 design S1 margins; grid
   0 to 1 by 0.01 plus infinity; B5c thresholds for views 1 to 3, D_cal halves 199 / 198 poles, seed [20261116, s]);
   B6 (uncertified S2-sum and S2-min; theta on half B: 0.09875 at 0.5% and 1%, -inf at 2%, 3% and 5%).
10. **Uncertified reference.** Loosest grid point with design committed error <= alpha (indices in the yaml).
11. **Fallbacks** as in plan section 11.

**Design-phase measurements (reported, not used for any choice).** Within-unit error correlation rho 0.631
[0.126, 0.887], design effect 2.24 [1.25, 2.74] at 2.96 views per unit; within-pole rho 0.515 [-0.627, 1.000]
(23 poles with more than one unit).

**S2 resolution on BelgiumTSC (design-phase finding).** S2 (both variants) has 3 distinct grid points on BelgiumTSC
design units: (a, b) = (0, 0.1); the second-largest ICP p-value is 0 for 418/465 views; each ECDF rests on about 80
windows; 6 of 7 S2 view-1 errors follow tied combined p-values broken to the smaller class ID. Recorded before F3 as a
finding about the score with a small calibration set; S2 is not amended (decision recorded here). The tie counts come
from a check run after the recorded design run, on design units only, with the recorded S2 fit (both variants: 9 of
157 units tied at view 1, 7 wrong, 6 of them tied).

**Design file and yaml.** `stage5_design.json` stores every non-finite number as null, so a B6 threshold of -inf
("commit at view 1") and None ("commit nothing") cannot be told apart in that file. The yaml is therefore written from
a recomputation of the design phase from the hash-checked Stage 4 inputs, which was checked equal to the hash-checked
design file in every field except the run fields (time, HEAD, worktree state, seconds); the yaml keeps -inf and null
apart.

- **F3 record (2026-11-15):** F3 commit = the commit that adds this block and the `FROZEN_BLOBS` entry (its hash is
  recorded in `LOG_r9_stage5_entry.md` Part B and `docs/HANDOFF.md`). `configs/frozen_r9_btsc.yaml` SHA-256
  93e030fdf85e6416691df1ddc0185286095be2c8254e571ed4d8897f3f645932 (working-copy bytes), git blob
  f4d37e5d73ef230e91689be5303e5eac51b15450. Stage 6 checks it through `src/r9_frozen.py` before reading it.

## POC-3 (R9) Stage 5: BelgiumTSC design phase and freeze F3

- Date: 2026-11-15 / Stage: R9 Stage 5 / Run tag: runs/btsc/stage5
- Config hash / checkpoint hash / git commit: Part A 019f782; code b63546a (design run HEAD); design-pole values
  ec85800 (stage5_design.json SHA-256 115f98025f60b14aedad9c0d75df6d5deafc568d77b427e2bf9804be1d854a11, git blob
  b06a09f8c64bbd789bd02addab0b518ecde66d0e); freeze code 986a8e7; F3 commit e94e6670987f7e9abad8382d848012a6a928407a
  (configs/frozen_r9_btsc.yaml SHA-256 93e030fdf85e6416691df1ddc0185286095be2c8254e571ed4d8897f3f645932, git blob
  f4d37e5d73ef230e91689be5303e5eac51b15450); tests after F3 1355c7d. Stage 4 inputs hash-checked against
  stage4_summary.json (1c66a011...3a31). Checkpoint 83fdd0d1...b76b via the Stage 4 embeddings.
- What was done: on the 133 design poles only, computed the S1 to S4 sequences (S2 with at most the last 3 views and
  one ECDF per window length, calibrated on two halves by pole), 30-point grids, design statistics, the Cert/Test
  re-splits from pool structure, starts per seed, Rule 3, within-sign correlation and the baseline design values;
  wrote configs/frozen_r9_btsc.yaml from a recomputation checked equal to the design file; committed it with the
  pre-registration block above and its FROZEN_BLOBS entry (F3). Full record in `LOG_r9_stage5_entry.md`.
- Result numbers:
  - Grids: S1 30, S2-sum 3, S2-min 3, S3 4, S4 30 distinct points; e_d at view 1 2.55% (S1, S3, S4), 4.46% (S2).
  - Cert / Test units per seed: 471 / 155, 472 / 154, 464 / 162, 471 / 155, 468 / 158; n_min 464.
  - Rule 3: predicted at 3% for S1 and S4 (0 / 150, UCB90 1.523%, n_c 443, 6.75 <= 8); at 5% also S2-sum and S2-min;
    S3 at no alpha. Primary alpha 3%.
  - Within-unit rho 0.631 [0.126, 0.887], design effect 2.24 [1.25, 2.74]; within-pole rho 0.515 [-0.627, 1.000].
  - S2: (a, b) = (0, 0.1); second-largest ICP p-value 0 for 418 / 465 views; 6 of 7 view-1 errors from tied combined
    p-values (post-run check on design units, not in the printout).
- Checks: integrity (m) PASS | F-D1 PASS | freeze-step reproduction PASS | check_frozen after F3 PASS (4 files).
- Decision taken and why: S2 frozen as fitted, its low resolution reported as a design-phase finding (author's
  decision before F3). F3 closed; nothing in the frozen yaml changes from here on.
- Deviation from the plan? Yes: S2 window of at most 3 views with per-length ECDFs; Rule 3 with n_min and a single
  grid point; view order ascending YYYYY; no stage5_summary.json (its contents are in stage5_design.json and the yaml);
  head margins compared exactly in float32; tests that read the live FROZEN_BLOBS pinned to an explicit pre-F3 state
  at F3 (guard code unchanged). Details in `LOG_r9_stage5_entry.md` Part B.
- Next step: Stage 6 (BelgiumTSC confirmation), only when the author says so.

## POC-3 (R9) Stage 6: BelgiumTSC confirmation

- Date: 2026-11-15 / Stage: R9 Stage 6 / Run tag: runs/btsc/stage6
- Config hash / checkpoint hash / git commit: Part A 89a8c42; code 34c3492 (run HEAD); `configs/frozen_r9_btsc.yaml`
  blob f4d37e5d73ef230e91689be5303e5eac51b15450 checked by `check_frozen()` (4 files); Stage 4 inputs and
  `stage5_design.json` hash-checked; `stage6_summary.json` SHA-256
  71be795779554a62546b1e03089222b53c35bc5a8ffac4bb2406c470b39e185f (blob 28ceaa6ae6cdd8964b0018510612fe2840dab3e2);
  `stage6_printout.txt` SHA-256 b2d6eef1a8f11d673882b01d08ff4ecca0d6c4eb57bdad04bfbd844c1898f30c.
- What was done: one run on the BelgiumTSC pool with every value from the frozen yaml: RQ1 resampling (2,000 draws of
  397 pool poles; S1 to S4 and view-level B1; every alpha; exact binomial and HB primary start), the pre-registered
  V-check, RQ2 on the 5 frozen re-splits (every score, alpha and rule; uncertified reference), B2, B5, B5c and B6, the
  plan section 8 table and the S2 against B6 table. Full record in `LOG_r9_stage6_entry.md`.
- Result numbers:
  - V-check PASS, 5 of 5 cells (3%, exact binomial): S1 0.00% (2000 certified), S2-sum 3.25% [2.52, 4.12] (1196),
    S2-min 3.25% [2.52, 4.12] (1920), S3 0.00% (1989), S4 0.00% (2000).
  - RQ1: view-level B1 14.00% [12.51, 15.60]% at 0.5% (above delta) and 8.60% [7.41, 9.92]% at 1%; largest
    sign-level rate 1.45% at 1% and 3.25% at any alpha. Design effect 2.24 [1.25, 2.74].
  - RQ2, smallest certifiable alpha (exact binomial, 5 seeds): S1 and S4 1%; S3 2%; S2-min 2% (one seed 3%); S2-sum
    3% (two seeds 5%). HB primary start certified 1% only in seed 4 (S1, S4); HB never certified where the exact
    binomial did not.
  - At 3% S1, S3 and S4 certify commit-all at view 1 (pool view-1 error 1.44%); coverage 100%, Test error 0.65%
    (0.62-1.90%). At 1% S1 and S4 certify in 5/5 at coverage 93.55% and 95.48% (Test error up to 1.32%, seed 4).
  - B6 above its target r: 2 of 5 re-splits at 3%, 5 of 5 at 2% (worst 3.23%). B5/B5c gap at most 0.65%, real error
    up to 1.90%.
- Checks: integrity all True | V-check PASS (5/5).
- Decision taken and why: the primary alpha stays 3% as frozen; the 3% table does not separate the methods, so the 1%
  table is reported beside it as pre-registered, non-scored cells.
- Deviation from the plan? Yes: the runner's `worktree clean` flag reads False because it creates its own output
  folder before `git status` (tree clean before the run; no number changed; not rerun); the 1% comparison table was
  assembled after the run from the summary JSON by `src/r9_stage6_tables.py` (commit 2392433; Part A promised it in
  the printout);
  B5/B5c "certified" read as a finite threshold. Details in `LOG_r9_stage6_entry.md` Part B.
- Next step: Stage 8 (final tables), only when the author says so. Stage 7 dropped (F-A3 failed).

## POC-3 (R9) Stage 8: final tables

- Date: 2026-11-15 / Stage: R9 Stage 8 / Output: docs/poc3-results-tables.md
- Config hash / checkpoint hash / git commit: code fd60e28 (`src/r9_stage8_tables.py` SHA-256
  3740264e458f536234bc04e5caa32a5274eebf4b5fc563345046a52c0f127242); output `docs/poc3-results-tables.md` SHA-256
  ae889e2e8765b1efabb917aea6e3db780c0897105c53cc411b0229feeca1ed07 (blob 3c6fed2750650a38453c089a82d9e2b39834033a).
  `check_frozen()` PASS (4 files). Sources hash-checked: R9 Stage 1, 2, 3 and 6 summaries; POC-2 Stage 3b, 4 and 6
  summaries; POC-1 stage5_results.json (git blob, no recorded SHA-256); LOG_r9_stage6_entry.md (git blob).
- What was done: built the Stage 8 tables from the committed summaries by code, with no new analysis: GTSRB
  comparison at 1% (exploratory), BelgiumTSC comparison at the primary 3% with the Stage 6 statement and at 1%
  (pre-registered, not the primary alpha), RQ1 validity (GTSRB real, simulation, BelgiumTSC), exact binomial against
  HB (three starts), S2 against B6 at every alpha, Ringel gap against real error, and limitations. Full record in
  `LOG_r9_stage8_entry.md`.
- Result numbers: none new; every value is as recorded in the cited summaries and Part B entries.
- Checks: check_frozen PASS | source hash checks PASS (9) | tests/test_r9_stage8_tables.py 22 passed | full suite
  530 passed before the table 1 header change.
- Decision taken and why: the GTSRB baseline columns are POC-2 records (HB for B1 and B2; B6 calibrated on Cert
  thirds), because R9 Stage 3 ran S1 to S4 only. They are marked "POC-2 record" in their headers, each column has a
  "Source / test" row, and the note under the table states the differences.
- Deviation from the plan? Yes: no Part A committed before the code (no new number exists in this stage); the
  optional adverse-cells table is not built; Stage 7 dropped (F-A3 failed), so there is no DITS table.
- Next step: Stage 9 (idea text and gates), only when the author says so.

## POC-3 (R9) Stage 10: prospective check of the effective-delta formula (declared addition, synthetic)

- Date: 2026-11-15 / Stage: R9 Stage 10 / Run tag: runs/20261003_stage1/poc3/stage10
- Config hash / checkpoint hash / git commit: Part A a8d142f (formula, settings, predicted delta_eff); population-curve
  script a6ed1b3; boundary subset fece72c (population_curves_stage10.json SHA-256
  ffa785eafc545a45b4a2d364b072111f5f8dd9440d4b0f308753e2dcce0b119f); runner 43df8eb (`src/r9_stage10.py` SHA-256
  3cd142a7f6435c8ee435ee5ba34b9c2f3382c37b8150650a1fe53e23018353ee). Output stage10_summary.json SHA-256
  4f7ce88317ce251ef6e0e2aad305cb3d5b46999b04a8927b307b2edbed7d5a15 (blob f342e8b3c82958fe3ac1730ac09c3d851ec48195);
  printout SHA-256 8212b93d2df1baa7f7ee8eb8f707b6a79210db7fb8b92a0131abce7c837c901b. Synthetic data and committed
  summaries only; no checkpoint, no dataset file. Frozen engine checked by blob ID first.
- What was done: fixed delta_eff = Phi(Phi^-1(delta) / sqrt(1 + (m - 1) rho)) and its predictions for 18 new settings
  (rho 0.1/0.2/0.6 x m 5/10/15 x alpha 1%/2%, P_FRAME 7.4%, Cert 784 / design 261) before any run; computed the
  boundary subset from population curves before any Cert sample; ran the Stage 2 simulator and frozen engine once,
  2,000 runs per setting; tabulated Stage 2, GTSRB POC-2 Stage 3b and BelgiumTSC Stage 6 retrospectively. Full record
  in `LOG_r9_stage10_entry.md`.
- Result numbers:
  - Prospective (frame-level B1, exact binomial): 18 of 18 settings within the bound (CP lower <= delta_eff), all 18 in
    the boundary subset; observed 1.25% to 20.60% against predicted 13.94% to 33.80%; mean gap -14.86 pp (absolute
    14.86), from -9.00 to -22.05 pp.
  - Observed rates fall with alpha in all 9 pairs (up to about 3.3 times lower at 2%) and do not rise with m at rho 0.1.
  - Sign level, exact binomial: at most 5.30% [4.36, 6.37]%; no CP upper bound above 0.12.
  - Retrospective: Stage 2 72 of 72 rows within the bound, point above prediction in 1 of 72 (level 1, rho 0.45,
    fixed-30, alpha 2%: 37.45% against 36.62%, the worst case known when the formula was written); GTSRB 3b 5 of 5;
    BelgiumTSC Stage 6 10 of 10.
- Checks: none scored (reported outcome). check_frozen PASS | input hash checks PASS | curve and prediction
  recomputation PASS.
- Decision taken and why: the formula is reported as a conservative ceiling on the frame-level violation rate, not as
  a predictor: it held in every prospective setting but overstated the rate by 14.86 pp on average, and the rate
  depends on alpha and on more than DE.
- Deviation from the plan? Yes: Stage 10 is not in plan R9 (declared addition; no frozen file or earlier result
  changed). The formula was evaluated with scipy while drafting Part A (accepted by the author). The boundary subset
  is all 18 settings, so its summary equals the full one.
- Next step: Stage 9 (idea text and gates), only when the author says so.

## r11: pilot upper-bound forecast of the frame-level false-certificate rate (exploratory, synthetic)

- Date: 2026-11-16 / Stage: r11 (exploratory addition after POC-3 Stage 10) / Run tag: runs/20261003_stage1/r11
- Config hash / checkpoint hash / git commit: Part A 867db93; Amendment 1 and run code 109ec7b
  (`src/r11_forecast_ucb.py` SHA-256 69293b526557a227e7dfbd065ecbdebc98aafe611bd8e85bc3369f096dc8f421). Output
  r11_summary.json SHA-256 b11c81adf4ae9a7521b3e37934400f0044a3d2bfc2242acbc823630ccf534509 (after the label
  correction; as written b6c9c5017b5300b78f3d7f70f3ec172613a6a3b0d060eab8f976d3644c824335). Synthetic data only.
- What was done: forecast of the B1 false-certificate rate from each run's pilot (one-sided upper 80% CP bound, j*,
  beta-binomial with pilot rho_I), against observed rates and the design-effect heuristic; development (Stage 10,
  reproduced exactly), replication (new seeds) and held-out (rho 0.3/0.5 x m 8/12 x alpha 1/2%) sets, 2,000 runs each.
  Full record in `LOG_r11_entry.md`.
- Result numbers: held-out, exact binomial: MAE 3.18 pp (heuristic 15.48 pp); forecast closer than the heuristic in
  8 of 8; forecast >= observed - 2 pp in 8 of 8.
- Checks: held-out criteria (Part A (e)): (a) FAIL (MAE 3.18 > 3.00), (b) PASS, (c) PASS; **held-out check FAIL**.
  check_frozen PASS | input hash checks PASS | Stage 10 reproduction PASS.
- Decision taken and why: recorded as failed; no criterion changed, nothing rerun.
- Deviation from the plan? Yes: not in plan R9 (exploratory addition; `r11_` names by the author's decision). Label
  correction in the two output files after the run (Part B, Deviations).
- Next step: none set; the author decides.

## r12: the r11 forecast on the GTSRB frame-level audit (exploratory, development data)

- Date: 2026-11-16 / Stage: r12 (exploratory addition after r11) / Run tag: runs/20261003_stage1/r12
- Config hash / checkpoint hash / git commit: Part A 6491dc8; run code 53f3da6 (`src/r12_gtsrb_forecast.py` SHA-256
  615b6f537c78a4d3b229c34a02d38685ce7a0f65cc4668af61b9df235850d22d). Output r12_summary.json SHA-256
  3d559b2fa5ec9de5d25b2d4e01079a083003ff45bee0c2f0b4c5f4a06d18488a. POC-1/POC-2 inputs hash-checked by the POC-2
  loaders; checkpoint 86527359...0bcc (unchanged, not reloaded).
- What was done: the r11 forecast, unchanged, with the 261 design tracks as the single pilot, against the POC-2 Stage
  3b B1 rates (HB, reproduced exactly) and a new exact-binomial B1 rate on the same 2,000 draws; 1,000-resample pilot
  bootstrap. GTSRB is development data for the forecast (Part A (a)). Full record in `LOG_r12_entry.md`.
- Result numbers: HB forecast 6.61% against observed 17.00% at alpha 0.5% (gap -10.39 pp); 16.94% against 20.00% at
  1% (gap -3.06 pp); 2%: 0.00% and 0.00%.
- Checks: Part A (h) stop-checks all PASS. Criteria (Part A (f)): (a) and (b) FAIL at 0.5% and at 1%; **overall FAIL**.
- Decision taken and why: recorded as failed; no criterion changed, nothing rerun.
- Deviation from the plan? Yes: not in plan R9 (exploratory addition; `r12_` names by the author's decision).
  `README_r12.md` not yet written.
- Next step: a post hoc diagnostic (r12b), labelled as such, only as the author directs.

## r12b: POST HOC DIAGNOSTIC of the r12 result (GTSRB development data)

- Date: 2026-11-16 / Stage: r12b (post hoc diagnostic, no criteria) / Run tag: runs/20261003_stage1/r12b
- Config hash / checkpoint hash / git commit: script daa19b0 (run 1) and the fixed script committed with this entry
  (run 2; `src/r12b_diagnostic.py` SHA-256 8a93fda029ef3de94678086ca68d546590db481eafd488cd60cb81230a01e44d). Output
  r12b_summary.json SHA-256 268464cd4bb5d8b8426f6509c4174b2df7dc962f05e4608415da0fc776ee0f61 (run 2; run 1
  374935c37aacf7f45062b5b2095412b59d72de062adeb48a83d622ee4a79fcfd, not committed).
- What was done: oracle forecast at the true first unsafe threshold with pool inputs; beta-binomial fit check of
  per-track wrong counts; split of acceptance against errors at jt. Written after the r12 result; not a test.
  Full record in `LOG_r12b_entry.md`.
- Result numbers: HB oracle forecast (fixed N_g) 16.83% against observed 17.00% at alpha 0.5% and 21.80% against
  20.00% at 1% (pilot forecasts 6.61% and 16.94%).
- Checks: none scored (diagnostic). Input hash checks PASS; POC-2 pool checks PASS.
- Decision taken and why: run 1 chi-square pooling did not match its stated rule (middle bins below 5 kept); fixed and
  rerun once on the author's instruction; every non-chi-square number identical between runs.
- Deviation from the plan? Yes: not in plan R9; post hoc diagnostic with no Part A, labelled as such.
- Next step: none set; the author decides.

## r13: pilot forecast with a bootstrap upper bound on rho_I (exploratory; simulated held-out and BelgiumTSC)

- Date: 2026-11-16 / Stage: r13 (exploratory addition after r12b) / Run tag: runs/20261003_stage1/r13
- Config hash / checkpoint hash / git commit: Part A 8c44fde; code 16085fd; Amendment 1 2a78cc2 (run on 2a78cc2).
  Output r13_summary.json SHA-256 f51abd7d5a0a6f4643707dae6fb5ccb2ba918322fb07b4d0294f01a3938e269f (after the label
  correction; as written 76cf350cd1e930c9e5e474c96c3b6b7497d5328d69352983ed21879e2d84305b).
- What was done: the r11 forecast with one change (rho_I input = 80th percentile of a 1,000-resample cluster bootstrap)
  on (a) 12 new simulated settings, (b) BelgiumTSC (pilot 133 design poles, Stage 6 RQ1 draws), (c) GTSRB (development
  data, reported) and (d) a GTSRB pilot-size curve (descriptive). Full record in `LOG_r13_entry.md`.
- Result numbers: (a) 0 of 12 settings met both (i) and (ii); (b) forecast 2.69% against observed 14.00% at 0.5%,
  2.73% against 8.60% at 1% (exact binomial).
- Checks: stop-checks all PASS. Criteria (Part A (d)): (a) FAIL, (b) FAIL at both alphas; **overall FAIL**.
- Decision taken and why: recorded as failed; no criterion changed, nothing rerun.
- Deviation from the plan? Yes: not in plan R9 (exploratory addition; `r13_` names by the author's decision). Label
  correction in the two output files after the run (Part B, Deviations).
- Next step: none set; the author decides.

## r10 (preliminary): sign-level certificate of frame-weighted (micro) risk on GTSRB — EXPLORATORY

- Date: 2026-11-16 (run; outputs written 14:12) / Stage: r10 preliminary, exploratory addition / Run tag:
  runs/20261003_stage1/prelim_r10
- Status: EXPLORATORY (post hoc). No Part A and no pre-registration: script, settings and reading were all written
  without a prior record, and this entry was written after the run. GTSRB tracks had all been seen before. Nothing
  here is a test.
- Config hash / checkpoint hash / git commit: `src/r10_prelim_micro.py` SHA-256
  8b397c213aa01f0e58b81492fe2747c8102eafa7a2f72f84182cf2747730385b (equal to the code hash recorded in the summary);
  outputs `prelim_r10_summary.json` SHA-256 670a3e0702784d99988b688f1256d138ba994c8b4d085a621cec32146ad3d3f4,
  `prelim_r10_report.md` SHA-256 d99ac9eace400b1156e39dd43822ff72ef669920466c68cdf07ae997fba2d234; checkpoint
  86527359...0bcc (hash-checked through the POC-2 loaders, not reloaded). Committed with this entry.
- What was done: Part 1, on the POC-2 Stage 3b draws: tests of the micro (frame-weighted) selective risk per track
  (Hoeffding, empirical Bernstein, WSR betting, WSR hedged half) against the stored B1 and track-level rows. Part 2:
  the same on the five POC-2 Stage 4 re-splits. Part 3: a first beta-binomial forecast of the B1 violation rate (j*
  from the pool's true risk), the forecast later revised in r11.
- Result numbers (alpha 1%, Part 1): micro-risk tests certified in 0.0% (Hoeffding), 0.0% (empirical Bernstein),
  5.8% (WSR betting) and 0.0% (WSR hedged) of 2,000 draws, each with 0 violations; stored track-level certificate
  certified in 19.8% with 1.40% violations; stored B1 20.00% violations. Part 3 forecast 16.60% (alpha 0.5%) and 47.30%
  (alpha 1%) against observed 17.00% and 20.00%.
- Checks: the script's own 14 integrity checks all PASS (Stage 3b draws and rows reproduce, Stage 4 splits
  reproduce, scalar references for the vectorised tests). No pytest file exists for this script (the project rule
  asks for one); not added.
- Decision taken and why: committed for traceability, because its numbers are cited; labelled exploratory.
- Deviation from the plan? Yes: not in plan R9; no pre-registration; `r10_` names outside the POC-3 `r9_` rule.
- Next step: none.

## a0: silent-error check on the GTSRB pool — EXPLORATORY

- Date: 2026-11-17 / Stage: a0, exploratory addition / Run tag: runs/20261003_stage1/a0
- Status: EXPLORATORY (post hoc). No Part A, no pre-registration, no certificate claim and no criteria; this entry
  was written after the run. GTSRB tracks had all been seen before. Nothing here is a test.
- Config hash / checkpoint hash / git commit: `src/a0_silent_error_check.py` SHA-256
  018d47c2e72d53a9a5b25c59d23e785f9c2e44a75525f65662868f7aa6c4baaf (equal to the code hash in the summary);
  `tests/test_a0_silent_error_check.py` 9779a4b84e73dc6747842b94660faa709b448f87541ab632ed18c20220f2e710; outputs
  `a0_summary.json` 4c4c907ee19a493ec0df0e188f6f34aa51f3e350af156985e235bfc4d8791928, `a0_printout.txt`
  258cce4f116c6104c9aa02ce03d5f58c53665595c7bec688ad429d02dc0fff0d. Frozen blobs checked first (plan R9, engine,
  engine tests, frozen_r9_btsc.yaml); inputs hash-checked through the POC-2 pool loader and the Stage 3 design file
  (798585428c4c...e62b). Committed with this entry.
- What was done: on the 1,046 pool tracks, consensus yhat_t = majority vote of the head's frame predictions over all
  frames (gate-failing frames included; ties: larger summed pre-gate V1 margin, then smaller class id). For S1, S2-sum,
  S2-min, S3, S4 and alpha 1/2/3/5%, the threshold certified by the frozen exact-binomial rule with the whole pool as
  the Cert set (n = 1,046, power-aware start at n = 1,046; Stage 3 grids and design fractions). At that threshold:
  committed error e, visible v (committed label != consensus), silent s (committed label = consensus != label);
  q = E[s] per track with two-sided 95% CP; gap = alpha * CP_lower(round(E[c] 784), 784, delta/3) -
  CP_upper(round(E[v] 784), 784, delta/3), one-sided bounds; n_A, n_A0 (search to 5,000) and n_full (exact binomial
  at delta with rounded counts, search to 1,000,000), rounding half up. The certificate and the measurement use the
  same tracks (in-sample).
- Result numbers: consensus wrong on 41 of 1,046 tracks (3.92%); 4 tracks had a vote tie. e == v on
  consensus-correct tracks in every row (0 mismatches). S1 and S3 at 1-3%: v = 0, so every committed error was silent
  (S1 4/907 at 1%, 11/974 at 2-3%; S3 3/897, 12/953). n_A was larger than n_full in every row where both exist
  (ratio 3.51 to 21.82); at 1% n_A was not found up to 5,000 for S1, S2-sum, S2-min, S3, and was impossible for S4
  (gap -0.032%, commit rate 43.4%). S4 was impossible (gap <= q) at 1-3% and not found at 5%. n_full ranged 138 to 893.
- Checks: 41 of 41 script checks PASS (frame top class equals the stored top1; committed tables agree with the
  Stage 3 table code; mismatch zero in all 20 rows). 17 synthetic tests (brute-force references, tamper test) pass.
- Decision taken and why: committed for traceability; labelled exploratory. No reading is attached.
- Deviation from the plan? Yes: not in plan R9; no pre-registration; `a0_` names outside the POC-3 `r9_` rule
  (author's naming); output folder beside `prelim_r10/` under runs/20261003_stage1/ (new folder, no POC-1 file changed).
- Next step: none set; the author decides.

## c0: class-group view of the full-pool certified thresholds on GTSRB — EXPLORATORY

- Date: 2026-11-17 / Stage: c0, exploratory addition / Run tag: runs/20261003_stage1/c0
- Status: EXPLORATORY (post hoc). No Part A, no pre-registration, no certificate claim and no criteria; this entry
  was written after the run. GTSRB tracks had all been seen before. Nothing here is a test.
- Config hash / checkpoint hash / git commit: `src/c0_class_check.py` SHA-256
  1e9eca889fa71b4aa4002cd45f4629403820f053f201fe3c7528bf3529604e90 (equal to the code hash in the summary);
  `tests/test_c0_class_check.py` 889eff72b74de3e94ab62ec4097d84afb0fbb5f62d82a5055344f42dd28c177e; outputs
  `c0_summary.json` ade431f022ed0cfe39bf91a2aedf0550098d2f186e286393ffe1a2773ab1ad44, `c0_printout.txt`
  ade32fc90c7ce476a5674182234e8b87754aea5dbc0d3cea32e714d2b8c5dea5. Frozen blobs checked first; inputs hash-checked
  through the POC-2 pool loader, the Stage 3 design file, `track_table_design.npz` (Stage 1 summary) and
  `a0_summary.json` (4c4c907e...1928, a0 entry). Committed with this entry.
- Class ids: GTSRB ships class ids only, no name file. The names in `src/head.py` were checked before the run by
  viewing three cached crops per class: 0-5 speed limits 20/30/50/60/70/80, 6 end of speed limit 80, 7 and 8 speed
  limits 100/120, 12 priority road, 13 yield, 14 stop, 17 no entry. Class 6 is a derestriction sign, so the
  speed-limit group is {0-5, 7, 8} and class 6 is in "others" (author's decision; the request said 0-8). The crop
  sheet is not part of the record (the image cache has no recorded hash).
- What was done: thresholds as in a0 (frozen exact-binomial rule, whole pool as Cert, n = 1,046, Stage 3 grids and
  design fractions); j, committed tracks and errors equal a0 in all 20 score x alpha cells. Two groupings:
  (1) g1 stop, g2 yield, g3 no entry, g4 speed limits, g5 priority road, g6 others (G = 6); (2) r1 right-of-way
  {14, 13, 17, 12}, r2 speed limits, r3 others (G = 3). Per group, score (S1, S2-sum, S2-min, S3, S4) and alpha
  (1/2/3/5%): tracks in pool, design and per 784-track draw; committed tracks and errors on the pool; ratio of group
  error to pooled committed error. Minimum committed tracks for a direct exact binomial test on the group's committed
  tracks (no fixed-sequence walk), p = P(Binom(m, alpha) <= k), level 0.05 (scheme b) or 0.1/(2G) (scheme a,
  Bonferroni): m0 with k = 0; with the observed group error, k = round half up(e_g m), the first rejecting m and the
  smallest m from which every larger m up to 1,000,000 rejects (author's addition; used for feasibility).
  "Impossible" when e_g >= alpha. Feasible = 784 n_g / 1,046 >= minimum.
- Result numbers: pool / design tracks per group: stop 21/5, yield 57/15, no entry 30/7, speed limits 341/85, priority
  road 56/14, others 541/135. m0 (scheme b) 299/149/99/59 at 1/2/3/5%; scheme a 477/237/158/94 (G = 6) and
  408/203/135/80 (G = 3). The right-of-way classes had 0 committed errors in every score x alpha cell. The four
  single-class groups are not feasible in any cell (expected committed per draw at most 42.7). r1 is feasible with
  zero errors at 3% (scheme b) and 5% (both schemes). Others: group error 1.40 to 2.03 times the pooled committed
  error, not feasible with its observed error in any cell; for S4 at 2/3/5% its error is at or above alpha
  (impossible). Speed limits: ratio 0 to 0.88; S4 at 1% commits no speed-limit track.
- Checks: 94 of 94 script checks PASS (13 class names, all 43 classes present, j and counts equal a0, group counts
  sum to the pooled counts). 17 synthetic tests (brute-force references, tamper test) pass; full suite 679 passed.
- Decision taken and why: committed for traceability; labelled exploratory. No reading is attached.
- Deviation from the plan? Yes: not in plan R9; no pre-registration; `c0_` names outside the POC-3 `r9_` rule
  (author's naming); output folder beside `a0/` under runs/20261003_stage1/ (new folder, no POC-1 file changed).
- Next step: none set; the author decides.

## c1: transfer bound from the pooled committed error to class groups on GTSRB — EXPLORATORY

- Date: 2026-11-17 / Stage: c1, exploratory addition / Run tag: runs/20261003_stage1/c1
- Status: EXPLORATORY (post hoc). No Part A, no pre-registration, no certificate claim and no criteria; this entry
  was written after the run. GTSRB tracks had all been seen before. Nothing here is a test.
- Config hash / checkpoint hash / git commit: `src/c1_transfer_check.py` SHA-256
  f8d747b95b6345676e6d9f7bd2d71f36bd87aeedf784344698309ab1d9a56425 (equal to the code hash in the summary);
  `tests/test_c1_transfer_check.py` 0af360146bc7a60e5281b1b6858f23d25acc394782901c188522804a3e9bcc72; outputs
  `c1_summary.json` 25e13397af3d267efba8705420e59f181308a5149e7517d788607271e096f776, `c1_printout.txt`
  cf9206a3b00ffc8c1613a20e6af2690d10a52a119a277b2e5e9041f9206fa993. Frozen blobs checked first; the only input is
  `c0_summary.json` (ade431f0...ad44, c0 entry), hash-checked; nothing recomputed from the pool. Committed with this
  entry.
- What was done: c0 Grouping 2 (r1 right-of-way {14, 13, 17, 12}, r2 speed limits {0-5, 7, 8}, r3 others incl.
  class 6), S1, S2-sum, S2-min, S3, S4, alpha 1/2/3/5%, at the full-pool certified thresholds (n = 1,046). Per group:
  share of committed tracks n_g / n_c and pooled committed error k / n_c. Transfer bound U_g = CP_upper(k_N, 784, 0.025)
  / CP_lower(w_N, 784, 0.025), one-sided Clopper-Pearson, with k_N = round half up(784 k / 1,046) and
  w_N = round half up(784 n_g / 1,046); it uses only the pooled error mass (all errors placed in the group, worst
  case), so a group's own observed error does not enter. Transfer certifies if U_g < alpha. Largest pooled committed
  error the transfer accepts (commit fraction and w_N held fixed): x_max = largest error count at 784 with
  CP_upper(x) < alpha CP_lower(w_N), by search, and the supremum e_max = (x_max + 0.5) / (784 n_c / 1,046). r2 only:
  c0 direct test at its observed error, scheme b (level 0.05), expected committed per 784 draw against the stable
  minimum m_all, printed FEASIBLE or NOT.
- Result numbers: U_g >= alpha in all 60 rows (transfer certifies no group in any cell); smallest U_g / alpha 1.53
  (r3, S1 at 3%: U_g 4.60%). At 1% no pooled error count is accepted for any group (x_max none, including 0 errors). r1 is accepted
  only at 5% and only with 0 errors at 784 (e_max 0.066%); r2 e_max 0.068-0.070% at 2%, 0.203-0.210% at 3%,
  0.595-0.598% at 5% (S4 at 2%: none); r3 e_max up to 1.262% (5%). Shares: r1 16.0-35.2%, r2 0.0-33.4%, r3
  49.3-64.8%. r2 direct test FEASIBLE in 11 of 20 cells (S1 3, 5%; S2-sum 2, 3, 5%; S2-min 2, 3%; S3 3, 5%; S4 3, 5%),
  NOT in 9 (all five scores at 1%; S1 2%, S2-min 5%, S3 2%, S4 2%).
- Checks: 62 of 62 script checks PASS (c0 all_pass, Grouping 2 class lists, U_g < alpha exactly when k_N <= x_max in
  every row). 14 synthetic tests (brute-force references, tamper test) pass; full suite 693 passed.
- Decision taken and why: committed for traceability; labelled exploratory. No reading is attached.
- Deviation from the plan? Yes: not in plan R9; no pre-registration; `c1_` names outside the POC-3 `r9_` rule
  (author's naming); output folder beside `c0/` under runs/20261003_stage1/ (new folder, no POC-1 file changed).
- Next step: none set; the author decides.

## d0: corruption envelope of the sign-level and frame-level certificate on GTSRB — EXPLORATORY (definitions, before the run)

- Date: 2026-11-17 / Stage: d0, exploratory addition / Run tag: runs/20261003_stage1/d0
- Status: EXPLORATORY (post hoc). Not in plan R9; no criteria. This block records the definitions and the author's
  choices before any d0 number exists; the results follow in a separate entry after the run. GTSRB tracks had all
  been seen before.
- Environment change (author's approval): imagecorruptions 1.1.2, scikit-image 0.26.0, opencv-python 5.0.0.93,
  imageio 2.38.2, lazy-loader 0.6, tifffile 2026.11.12, installed from PyPI wheels (files.pythonhosted.org) pinned with
  `--require-hashes`. Wheels kept in `data/d0_wheels/` (git-ignored); their SHA-256 values are in
  `src/d0_envelope_check.py` (WHEELS_SHA256) and are checked at run time. No numpy or scikit-image downgrade. No
  import was blocked by Windows Application Control.
- Patch (`src/d0_corrupt_compat.py`): imagecorruptions 1.1.2 `fog` fails on numpy 2 (`np.float_` was removed). The
  only change: `plasma_fractal` allocates its heightmap with `np.float64` instead of `np.float_` (the same dtype in
  numpy < 2). Check: the patched fog equals the unmodified package fog (old alias restored for the call, same seed)
  bit for bit at severities 1-5 on 3 images, as a synthetic test and again at run time on 3 pool cache images. The
  other four corruptions run unmodified.
- Definitions:
  - Conditions: clean, plus fog, motion_blur, gaussian_noise, contrast and brightness at severity 1-5, applied to every
    frame of each of the 1,046 pool tracks (same corruption and severity for the whole track). Images are the 112 x 112
    cache (`train_112.npy`, SHA-256 checked): whole frames resized, no ROI crop. The package's severity constants were
    set for 224 px ImageNet images, so severity k here is not ImageNet severity k. Seed:
    `np.random.seed([20261117, track code])` once per track, frames in frame order, the same seed in every condition.
  - Unchanged: the frozen POC-1 ResNet-18 (checkpoint 8652735...0bcc), the head, the gate tau (the gate is recomputed
    on the corrupted frames), and the R9 Stage 3 design values (grids, design commit fractions, S2-sum model; all
    fitted on clean design tracks). Scores S1, S2-sum, S3, S4; alpha 2, 3, 5%; delta 0.1.
  - Cert sets (author's choice): the 5 POC-2 Stage 4 re-splits (Cert 784 tracks each), regenerated and checked as in
    R9 Stage 3. Pool risk is measured on all 1,046 corrupted pool tracks, which include the 784 Cert tracks.
  - Sign level: the frozen exact-binomial certificate (`src/r9_certify.py`, rule binom), n = 784.
  - Frame level (author's choice: per score, frame units): same engine and grid. A frame of a Cert track is committed
    at lambda when the running score value at that frame exists and is >= lambda; it is wrong when the running label
    at that frame is not the track's class. n = frames of the Cert tracks. Start from the design frame commit fraction
    on the 261 clean design tracks (design frames as n_design).
  - Flags (author's choice, recorded before the run): primary = a frame-level certification whose pool frame risk
    (wrong committed frames / committed frames) is above alpha; secondary = its pool sign risk (wrong committed tracks /
    committed tracks at that threshold) above alpha. A sign-level certification is flagged when its pool sign risk is
    above alpha.
  - Frame error = top-1 head error over all pool frames (before the gate).
  - Envelope boundary per corruption x score x alpha x level x re-split: the last severity s with severities 1..s all
    certified (0 if severity 1 is not); the minimum over re-splits is also printed.
  - Run-time checks: freeze blobs; hashes of every input; POC-2 pool reproduction; Stage 4 coverages reproduce; design
    commit fractions reproduce; clean re-embedding of the pool frames within 1e-3 of the stored embeddings; clean
    sign-level certified indices equal R9 Stage 3 (`stage3_summary.json`); fog patch reference.
  - GPU time = wall time of the embedding forward passes, transfer included.
- Code: `src/d0_envelope_check.py`, `src/d0_corrupt_compat.py`, `tests/test_d0_envelope_check.py` (17 synthetic tests:
  brute-force references, fog patch reference, seed determinism, tamper test). Code hashes go in the results entry.
- Deviation from the plan? Yes: not in plan R9; `d0_` names outside the POC-3 `r9_` rule (author's naming); output
  folder `runs/20261003_stage1/d0/` (new folder, no POC-1 file changed).
- Next step: one run; the whole printout goes to the author; then the results entry.
