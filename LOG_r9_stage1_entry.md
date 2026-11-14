## POC-3 (R9) Stage 1: data audit (BelgiumTSC, DITS), F-gates F-A1 to F-A3

### Part A: fixed definitions (committed BEFORE any file is downloaded or any count exists)

Plan: `docs/poc3-implementation-plan-r9.md` (SHA-256 c0639973c70f35ac5cda1f11c49bf50a1639c632c48ccec8889d60a5ded03db4),
section 9 Stage 1, section 7 (datasets), section 11 (fallbacks). Stage 1 is a structure audit only: no model, no
embedding, no prediction and no design/pool split is made here (the split is Stage 4). No Cert or Test number can exist
in this stage.

#### (a) Download sources (approved by the author before any download)
| File | URL | Size listed on the source page | Where the URL comes from |
|---|---|---|---|
| BelgiumTSC training release | https://btsd.ethz.ch/shareddata/BelgiumTSC/BelgiumTSC_Training.zip | 171.3 MBytes | Link target `../shareddata/BelgiumTSC/BelgiumTSC_Training.zip` on https://btsd.ethz.ch/shareddata/ (BelgiumTS page, maintained by Radu Timofte), resolved against that page |
| BelgiumTSC testing release | https://btsd.ethz.ch/shareddata/BelgiumTSC/BelgiumTSC_Testing.zip | 76.5 MBytes | Link target `../shareddata/BelgiumTSC/BelgiumTSC_Testing.zip` on the same page |
| DITS dataset page | http://users.diag.uniroma1.it/bloisi/ds/dits.html | none | Authors' own page (Sapienza University of Rome; Youssef, Albani, Nardi and Bloisi, ACIVS 2016) |

- Source pages opened on 2026-11-14 (HTML only, no data file):
  - https://btsd.ethz.ch/shareddata/ answered (HTTP 200). It states no licence and no terms of use. Its only condition
    is: "If you use the material from this dataset, please cite one of the following publications, accordingly:".
    The cited publications are recorded in Part B.
  - The DITS page did not answer: the connection to users.diag.uniroma1.it was refused on port 443 and timed out on
    port 80.
- DITS is taken only from the authors' own page. Copies on Roboflow or any other re-upload site are not used.
- Only the files in this table are downloaded. Any other file (a readme, a licence text, a DITS archive linked from its
  page) is added to this table by a dated Amendment before it is downloaded.
- If a source page forbids research use, its files are not downloaded and the gate concerned is treated as failed.

#### (b) Target folders and download record
- BelgiumTSC: new empty folder `data/btsc_raw/`. The zips go in `data/btsc_raw/zips/`; each zip is extracted into its
  own new empty folder `data/btsc_raw/extracted/<zip name without .zip>/`.
- DITS: new empty folder `data/dits_raw/`, same layout (`zips/`, `extracted/`).
- Each folder must not exist before the download; the download step stops if it does.
- `.gitignore` gets `data/btsc_raw/` and `data/dits_raw/` (committed with this Part A), so no dataset file is tracked
  or redistributed.
- For every downloaded file: source URL, download date (UTC), size in bytes and SHA-256 are written to
  `runs/btsc/stage1/downloads_manifest.json` at download time and copied into Part B.
- Size check after download: each BelgiumTSC zip's size in bytes must match the listed size (171.3 and 76.5 MBytes).
  The page does not say whether "MBytes" means 10^6 or 2^20 bytes, so the check passes if bytes / 10^6 or
  bytes / 2^20, rounded to one decimal, equals the listed value. A mismatch stops the stage before extraction and is
  reported.
- The audit script checks every zip's SHA-256 against that manifest before reading it and stops if any differs
  (hard rule 2). It reads the zips' contents from the extracted folders only after the zip hashes pass, and records the
  SHA-256 of every GT csv file it reads.

#### (c) What is counted (per release: training, testing)
1. Images: number of image files, compared with the plan's 4,575 (training) and 2,520 (testing) "as distributed".
   A difference is reported, not corrected.
2. Classes: number of class folders and of distinct class IDs in the ground-truth csv files (expected 62).
3. Poles and sign units (definitions in (d)): number of pole numbers; number of sign units; views per sign unit
   (distribution of counts 1, 2, 3, 4+); sign units per class; the number of poles carrying more than one class and
   how many extra sign units they add (sign units minus pole numbers).
4. Consistency: every image has exactly one ground-truth row and every row has an image; class folder = class ID in
   the csv; every file name parses under the rule in (d). Each failure is counted and listed.
5. Size: shorter ROI side, min(Roi.X2 - Roi.X1, Roi.Y2 - Roi.Y1) in original pixels (the POC-2 convention), as
   quantiles over views, per release. Structure only; used later for the "sign size at commit" metric.
6. Cross-release overlap (F-A2), by the rule in (e).
7. DITS (if the page gives access): how the data is packaged, whether track IDs exist, frames per track, classes,
   number of tracks and of frames, day/night/fog labels. Recorded as found; DITS is not split or modelled in Stage 1.

#### (d) Pole and camera IDs from file names
- Expected pattern (to be confirmed from the readme or dataset page): files `PPPPP_CCCCC.ppm` inside class folders
  `00000` to `00061`, with `PPPPP` the pole number and `CCCCC` the camera (view) number, plus one ground-truth csv per
  class folder. [Guessing: from prior knowledge of the release, not yet seen.] The BelgiumTS page describes cameras
  only for the detection sequences ("The cameras are taken in clockwise order 0-7, are mounted in pairs, one pair on
  each side of the recording van."); it says nothing about BelgiumTSC file names or pole numbers.
- Rule: the meaning of each number is taken from the readme or the dataset page, quoted in Part B. If neither states
  it, the pattern is accepted only if all of these hold, and each is reported:
  - every file name in both releases matches `^\d{5}_\d{5}\.ppm$`;
  - within a release, no (pole, camera) pair occurs twice inside one class folder;
  - views per pole is mostly small (the plan expects about 3).
  If the meaning cannot be confirmed by the readme or by these checks, poles are "not recoverable" and F-A1 fails.
- Sign unit (author's decision): unit = (pole number, class). A pole can carry several physical signs, and the plan's
  unit is one physical sign (section 7). Every count, split and certificate from Stage 4 on uses this unit. The pole
  number alone is kept for the shared-pole test (e).
- View order (used from Stage 5 on): ascending camera number within a sign unit. Fixed here so it is not chosen after
  seeing results.

#### (e) Shared poles between releases (F-A2)
- Shared-pole test (author's decision): a testing-release sign unit is shared if a check that applies flags it.
  - Check 1 (number): the same pole number and the same class occur in both releases. Check 1 applies only if the
    readme states that pole numbers are global across the training and testing releases. Otherwise pole numbers may
    restart in each release, a match may be a coincidence, and check 2 alone decides.
  - Check 2 (content): a testing view is byte-identical to a training view, or has the same class, image width,
    image height and ROI box (Roi.X1, Roi.Y1, Roi.X2, Roi.Y2) as a training view. A flagged view flags its sign unit.
    Check 2 always applies.
- Shared sign units are removed from the testing release (the training release is used only to fit the model in
  Stage 4 and is not otherwise changed).
- Part B reports whether check 1 applied (with the readme sentence, or "not stated"), the number flagged by check 1
  only, by check 2 only, by both, and the number removed. Check 1's count is reported even when it does not apply,
  with what it then means (pole-number coincidences between releases, not shared signs).

#### (f) F-gates (plan section 9 Stage 1; fallbacks plan section 11)
| ID | Gate | Measured as | If it fails |
|---|---|---|---|
| F-A1 | BelgiumTSC poles can be recovered and at least 600 testing-release poles exist | (d) passes; count of testing-release pole numbers with at least one sign unit left after the F-A2 removal (poles, as the plan says; not sign units) | Fallback 1 |
| F-A2 | No pole is shared between the training and testing release, or shared poles can be removed | (e); if shared sign units exist, they are removed and F-A1 is checked on the rest | Remove shared poles; if fewer than 600 remain, Fallback 1 |
| F-A3 | DITS can be downloaded and has track IDs | Only from the authors' own page in (a); access works without breaking the source's terms, and track IDs can be read. Page access rule below | Drop Stage 7. Not a failure |

- Analysis unit: (pole number, class) sign units are the unit from Stage 4 on (d). F-A1 counts poles only. Part B
  reports both counts.
- DITS page access (author's decision): one retry on the day of the Stage 1 run, first HTTPS, then HTTP, at the URL in
  (a). If both fail, F-A3 fails and Stage 7 is dropped (plan section 11). Both attempts are recorded in Part B with
  time (UTC) and the error.

- Fallback 1 (plan section 11): use DITS as the fresh dataset if F-A3 passed. If neither is usable, POC-3 stops at
  Stage 3 and the paper is the GTSRB study with the layer labelled exploratory, plus the POC-2 record; the
  single-dataset limit is stated.
- The 600 threshold is the plan's and is not changed by anything in this entry.
- A gate that fails is reported as failed, and its fallback is followed as written.

#### (g) Expected limit (stated before any count; not a gate and not a change to F-A1)
- Cert is about 60% of all testing-release poles (design 20%; Cert 75% of the 80% pool). With about 840 testing poles
  (2,520 views at about 3 views per pole) [Guessing until counted], Cert is about 500 poles, and at a 70% commit
  fraction n_c is about 350.
- Exact binomial at delta = 0.1 (plan section 6, recomputed for these sizes): alpha = 1% certifies only with 0 errors
  for n_c below 388, and with at most 1 error from 388 up to about 500. At n_c about 350 the tolerated errors are 0 (1%),
  3 (2%), 6 (3%), 11 (5%); alpha = 0.5% is not certifiable below n_c = 460. At the F-A1 minimum (600 poles, Cert 360,
  n_c about 252 at 70%): 0 (1%), 1 (2%), 3 (3%), 7 (5%).
- Part B reports, next to the section 6 table: the testing-pole count after F-A2, the expected Cert size
  (0.6 x that count) and the expected n_c at a 70% commit fraction, with the tolerated errors at each alpha.

#### (h) Code, tests and outputs
- Code: `src/r9_stage1_audit.py` (reads file names, ground-truth csv files and, for check 2 in (e), image bytes and headers;
  no model, no torch import). A test asserts that the module imports neither torch nor any POC-1/POC-2 model or
  embedding loader, as the design-phase guard for this stage (hard rule 3).
- Tests: `tests/test_r9_stage1.py` on synthetic folder trees: file name parsing, pole/view counting against a
  brute-force count, multi-class poles, shared-pole detection, missing ground-truth rows, and tamper tests (a changed
  zip or csv is refused).
- Output: `runs/btsc/stage1/stage1_summary.json` (counts, checks, gate results, SHA-256 of every input) and
  `runs/btsc/stage1/downloads_manifest.json`. `README_r9_stage1.md` gives the run command.

### Amendment 1 (14 Nov 2026, after the size check)
Made after the two BelgiumTSC zips were downloaded (Part A commit 9a228a1) and the listed-size check in (b) failed. No
zip had been opened or extracted, and no count of any kind existed.

- The listed-size check failed for both zips. The FAIL stands and is reported in Part B as recorded:

  | File | Bytes | Bytes / 10^6 | Bytes / 2^20 | Listed (MBytes) | Size check |
  |---|---|---|---|---|---|
  | BelgiumTSC_Training.zip | 174,298,785 | 174.3 | 166.2 | 171.3 | FAIL |
  | BelgiumTSC_Testing.zip | 77,877,833 | 77.9 | 74.3 | 76.5 | FAIL |

- Both files are about 1.8% larger than listed in decimal MB. This points to the page's size convention or an older
  build of the zips, not to a broken download; it is not proven. The page gives no hash to check against.
- Both downloads returned HTTP 200 from the URLs in (a) with no redirect, and the byte counts reported by the download
  equal the file sizes on disk. The first download step did not record the download tool's own exit code; the
  download step is fixed so the manifest records it from now on.
- New conditions to proceed, fixed before either is run. They replace the size check as the condition to proceed;
  they do not turn the size check into a pass.
  - (a) Zip integrity: every member's CRC passes (Python `zipfile.ZipFile.testzip`, read-only, nothing written). The
    zip's SHA-256 is checked against `runs/btsc/stage1/downloads_manifest.json` first. If (a) fails: stop, download
    that zip again once with the fixed download step, record both attempts, and run (a) again.
  - (b) Image counts after extraction equal the counts in plan section 7: 4,575 training and 2,520 testing images. An
    image is a file with extension `.ppm` (any case). Only images are counted at this step; no pole, camera or class
    count is made until (b) is recorded. If (b) fails: stop and report; the stage does not proceed.
  - Extraction (only after (a) passes) goes into `data/btsc_raw/extracted/<zip name without .zip>/`, each a new empty
    folder, as in (b) of Part A.
- This amendment replaces (c) item 1 ("a difference is reported, not corrected") for the image count: a difference now
  stops the stage.
- Part B reports the size mismatch, conditions (a) and (b) and their results as facts.

### Amendment 2 (14 Nov 2026, before any pole/class count)
Made after reading the two readmes and the header lines of the GT csv files for their format, and before the audit was
run. No pole, class or shared-unit count had been made; the only counts so far are the Amendment 1 image counts.

- Check 1 applies. Reason: the readme states that XXXXX "represents the pole number from where is in the full
  annotations of BelgiumTS database", that is, one numbering for the whole database, which meets the rule in (e).
  Decided from the readme text only. Check 1 and check 2 counts are reported separately in Part B, as already set.
- View order: BelgiumTSC has no camera index. YYYYY is "a running number for the views where the traffic sign is
  annotated. There is no temporal order of the images, but they were where extracted in the order of the annotations
  on the original pole from BelgiumTS." (readme, quoted as written). View order for all later stages = ascending YYYYY.
  This replaces "camera order" and "camera number" wherever Part A or plan Stage 5 uses them.
- Views per pole: reported as median and range (minimum, maximum), with no pass/fail. This replaces the "mostly small"
  condition in (d).
- The (g) projection uses poles, as written. Part B notes that from Stage 4 the split is by (pole, class) units, so the
  real Cert size may be larger.
- Part B reports the readme totals (4,591 training / 2,534 testing images) against the zip counts (4,575 / 2,520) as
  a fact. The zip counts are used.

### Part B: results (filled from the printout after the run)
- Date: 2026-11-14 (run 11:46:48 UTC) / Stage: R9 Stage 1 / Run tag: runs/btsc/stage1
- Git commit of Part A: 9a228a1 (definitions), 5dc5032 (Amendment 1), 6f05a92 (Amendment 2) | Git commit of the code:
  262c832 (src/r9_stage1_audit.py SHA-256 3f1df93bf2623b86b5f18545b0f10970b60af4d718ce489cd147926d3176acf9;
  src/r9_stage1_integrity.py 1f8a1dd67ff0618759b7d3eb15d3a75b9e5a1864c033fba41f536cff4bb2a0e3;
  src/r9_stage1_download.py 4fd5887d649f9e9ab402d1faee60cddb2becf564105b13c4af280deae683cc3b). Working tree clean at
  run time. Run command: `py -3.12 -m src.r9_stage1_audit --check1-applies yes --dits-retry`.
- Downloads (from downloads_manifest.json, SHA-256 06c90353f75a8c4133991eb32dad2281d922622cecc0591a1b9b5bc819ae4ca3),
  one attempt each, HTTP 200, no redirect:
  - BelgiumTSC_Training.zip: 174,298,785 bytes (174.3 x 10^6, 166.2 x 2^20; listed 171.3 MBytes; size check FAIL),
    SHA-256 e5f6a33c18716917d69989488b9f6af573821a7126ca1390af7e75403af1ee61; 2026-11-14 11:18:30 to 11:21:49 UTC.
  - BelgiumTSC_Testing.zip: 77,877,833 bytes (77.9 x 10^6, 74.3 x 2^20; listed 76.5 MBytes; size check FAIL),
    SHA-256 1811447452f02ede83c0237f2852fa8bd278f39e40ddcb1ba01c2ed212e735fa; 11:21:49 to 11:22:33 UTC.
  - The first download step did not record curl's own exit code (recorded as null in the manifest).
  - DITS: nothing downloaded (F-A3 below).
  - Licence / terms: the BelgiumTS page states no licence and no terms of use; it asks users to cite. Publications named
    in both readmes: R. Timofte and L. Van Gool, "Sparse representation based projections", BMVC 2011; R. Timofte,
    K. Zimmermann and L. Van Gool, "Multi-view Traffic Sign Detection, Recognition, and 3D Localisation", Machine
    Vision and Applications, 2011, doi 10.1007/s00138-011-0391-3.
- Amendment 1 conditions: (a) zip integrity PASS for both (training 4,701 members, testing 2,646 members, no bad
  member; first attempt, no re-download); (b) image counts PASS: training 4,575 (plan 4,575), testing 2,520 (plan 2,520).
- Output: stage1_summary.json SHA-256 77ba6a7042a471fa3320059d85a5ec967a8d30a613355c05c74815d03277a0dd.
  Descriptive breakdown (below): check2_breakdown.json SHA-256
  08aa904ba6e0bd0a9d6fa5ec4850773f791722a71e0d3c95ef27d7c54e96a50d (src/r9_stage1_breakdown.py, commit 852dedc,
  SHA-256 d696bd2ec0c2a802d8b20f0616b5d7b01c02b9277d14ed08200123fff8e2abc0).
- Pole and view encoding (readme, both releases, quoted as written): "The first part, XXXXX, represents the pole number
  from where is in the full annotations of BelgiumTS database. All images of one class with identical pole numbers
  originate from the same pole and more represent one single physical traffic sign. The second part, YYYYY, is a
  running number for the views where the traffic sign is annotated." There is no camera index (Amendment 2).
- Counts, training / testing:
  - images 4,575 / 2,520 (plan 4,575 / 2,520; readmes state 4,591 / 2,534, see below);
  - class folders 62 / 62; distinct class IDs 62 / 53 (9 classes have no testing image);
  - pole numbers 1,364 / 701; sign units (pole, class) 1,545 / 848;
  - poles with more than one class 162 / 128, adding 181 / 147 sign units;
  - views per sign unit (1 / 2 / 3 / 4+): training 18 / 64 / 1,432 / 31; testing 14 / 8 / 816 / 10;
  - views per pole: median 3 / 3, range 1-15 / 1-12;
  - ROI shorter side (px), quantiles 0 / 10 / 25 / 50 / 75 / 90 / 100%: training 9 / 41 / 56 / 78 / 108 / 146 / 438;
    testing 10 / 35 / 46 / 65 / 99 / 139 / 416;
  - consistency failures: 16 / 14 images with more than one GT row and 16 / 14 repeated (pole, view) pairs within a
    class folder; no other failure type (no unparsed name, no image without a row, no row without an image, no class
    mismatch). Examples: training 00056/01199_0000[0-5], 00057/00472_0000[0-3]; testing 00035/02261_0000[0-1],
    00035/02263_0000[0-5], 00038/00027_0000[0-1].
  - The duplicated rows account exactly for the readme totals: 4,575 + 16 = 4,591 and 2,520 + 14 = 2,534. The zip
    file counts are used. The audit counts one view per GT row, so the views-per-unit and views-per-pole figures above
    include the duplicates; pole and unit counts do not depend on them. Stage 4 counts views by unique file name.
- Shared sign units: check 1 applied (Amendment 2: one pole numbering for the whole BelgiumTS database). Check 1 only 5,
  check 2 only 59, both 1; check 1 total 6, check 2 total 60; removed 65. Check 1's count means 6 testing sign units
  carry the same pole number and class as a training sign unit, read as the same physical sign.
  - Descriptive breakdown, after the gate result, changes nothing: of the 60 units flagged by check 2, 0 have a
    byte-identical training image and 60 are matched only by class, image width, image height and ROI box. Whether
    these are the same physical signs re-encoded or coincidences between different signs is not established; all 60
    stay removed as the rule in (e) requires.
- After removal, testing release: 663 poles (F-A1 count) / 783 sign units.
- Expected limit (g), next to plan section 6:
  - By poles (as written): Cert 398 (0.6 x 663), n_c 279 at a 70% commit fraction; tolerated errors at alpha
    0.5 / 1 / 2 / 3 / 5%: not certifiable / 0 / 2 / 4 / 8.
  - By sign units (Amendment 2 note; the Stage 4 split is by (pole, class) units, so the real Cert size may be larger):
    Cert about 470 (0.6 x 783), n_c about 329; tolerated errors: not certifiable / 0 / 2 / 5 / 10.
  - At either size alpha = 1% certifies only with 0 committed errors (n_c below 388), and 0.5% is out of reach.
- DITS page retry on the run day: HTTPS 2026-11-14 11:46:03 UTC, curl exit 28, "Failed to connect to
  users.diag.uniroma1.it:443 after 21542 ms: Could not connect to server" / HTTP 11:46:25 UTC, curl exit 28, "Failed
  to connect to users.diag.uniroma1.it:80 after 21059 ms: Could not connect to server". DITS structure: not known.
- Checks: F-A1 PASS (663 testing poles after removal, minimum 600) | F-A2 PASS after removal (65 shared units removed;
  663 poles remain, so the "fewer than 600" fallback is not triggered) | F-A3 FAIL (authors' page unreachable on both
  retries).
- Decision taken and why: BelgiumTSC is the fresh dataset for Stages 4 to 6 (F-A1 and F-A2 passed, plan section 9).
  Stage 7 (DITS) is dropped (F-A3 failed, plan section 11); earliness evidence stays GTSRB only, labelled exploratory,
  and this is stated as a limitation. Stage 4 stratifies by class where possible; the testing release has 53 of 62
  classes.
- Deviations: Amendment 1 conditions (a) and (b) were run on 2026-11-14 on code committed afterwards, unchanged
  (src/r9_stage1_integrity.py SHA-256 1f8a1dd67ff0618759b7d3eb15d3a75b9e5a1864c033fba41f536cff4bb2a0e3,
  src/r9_stage1_download.py SHA-256 4fd5887d649f9e9ab402d1faee60cddb2becf564105b13c4af280deae683cc3b; committed with
  tests/test_r9_stage1_integrity.py in the commit that adds this line, 88741f3). These were integrity checks only (zip
  CRCs and the .ppm image count); no research count was made. src/r9_stage1_download.py was not run.
- Deviations: commit 262c832 added the views-per-pole minimum and maximum to the audit code (with a test) after
  Amendment 2 asked for median and range, and before the audit was run. Its commit message was not set in advance.
- Deviations: the check-2 breakdown (852dedc) was added after the gate result as a descriptive read-only step; it
  changes no removal and no gate.
- Deviation from the plan? Yes. (1) The listed-size check was replaced as the condition to proceed by zip integrity
  and image counts (Amendment 1); the size FAIL stands. (2) The analysis unit from Stage 4 on is (pole, class), since
  a pole can carry several signs; F-A1 still counts poles as the plan says. (3) Shared units are found by two checks
  (pole number and class; image content), not only by pole number. (4) BelgiumTSC has no camera index; view order is
  ascending view number in annotation order (Amendment 2), replacing "camera index" in plan Stage 5.
- Next step: Stage 2 (certification engine v2, V-checks), only when the author says so.
