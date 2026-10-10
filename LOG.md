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

## TEMPLATE (copy for each new entry)

## <date> | Stage N | <run tag>

* Config hash / checkpoint hash / git commit:

* What was done (one line):

* Result numbers (from the results JSON):

* Checks (pass/fail each):

* Decision taken and why (cite the plan section):

* Deviation from R6? yes/no (what, why):

* Next step:
