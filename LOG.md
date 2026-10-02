# LOG.md (decision and result log, plan section 12)

Rule: one entry per stage or decision. Commit this file after every entry.

---

## PRE-REGISTRATION (DRAFT: edit now, then commit BEFORE running Stage 5; paste the commit hash)

Values below are the plan's recommendations, not decisions yet. Changing them after you
see Stage 5 numbers is the thing a reviewer can legitimately attack.

1. G1 aggregation over 5 reshuffles

   * G1(i) hard-minus-easy track-weighted error gap, track-cluster bootstrap 95% lower bound > 0: passes if true in >= 4 of 5

   * G1(ii) Cert holds >= 30 hard-group frame errors over >= 10 distinct tracks: passes if the MEDIAN over 5 meets both

   * G1(iii) some alpha <= 3% certifiable in BOTH strata: passes if true in >= 4 of 5

2. Bootstrap: B = 2000, lower bound = 2.5th percentile

3. First fallback if G1(i) is weak: top-3 epsilon (eps3 = eps(y1,y2) + eps(y1,y3))

4. G2 severity bands (measured on Fit only): mild = 3-8 pp accuracy drop, severe = 15-30 pp

5. Certification ordering (plan 6.3): order thresholds by Fit-estimated E[L], start at the Fit argmin, relax looser; this replaces "strict to loose" in R6

Frozen at git commit: ______________

---

## 2026-10-02 | Stage 0 | data ingestion + track index

* Environment: Python 3.11.9; numpy 2.4.6, pandas 3.0.6, pillow 12.3.0, scipy 1.17.1, pyyaml 6.0.3, tqdm 4.70.1, matplotlib 3.11.2 (requirements-lock.txt; Stage 0 only, re-freeze after Stage 1)

* Git commit: ______________

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

* config hash: 6c7404a0e7e3300de840d798dc2e0cb569aa199f6447dbbfb2d3ed7ea79356eb

* Python: 3.12.10

* torchvision: 0.29.1+cu132

* numpy: 2.5.2


## 2026-10-03 | Stage 2 | embeddings from the frozen backbone (run tag 20261003_stage1)

* Checkpoint hash checked before running (equals the frozen value; script stops otherwise): 86527359bc61a5fbc4d48b7744e049b5ca3ff81cb016fb237298d0c007140bcc

* Git commit: ______________

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

## TEMPLATE (copy for each new entry)

## <date> | Stage N | <run tag>

* Config hash / checkpoint hash / git commit:

* What was done (one line):

* Result numbers (from the results JSON):

* Checks (pass/fail each):

* Decision taken and why (cite the plan section):

* Deviation from R6? yes/no (what, why, supervisor told?):

* Next step:
