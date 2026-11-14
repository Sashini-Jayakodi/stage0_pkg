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

### Part B: results (fill from the printout after the run)
- Date: ____ / Stage: R9 Stage 1 / Run tag: runs/btsc/stage1
- Git commit of Part A: ____ | Git commit of the code: ____ (src/r9_stage1_audit.py SHA-256 ____)
- Downloads (from downloads_manifest.json): BelgiumTSC_Training.zip ____ bytes (listed 171.3 MBytes; size check ____),
  SHA-256 ____; BelgiumTSC_Testing.zip ____ bytes (listed 76.5 MBytes; size check ____), SHA-256 ____; DITS ____.
  Licence / terms: ____; publications the BelgiumTS page asks to cite: ____
- Amendment 1 conditions: (a) zip integrity, training ____ / testing ____ (attempts ____); (b) image counts, training
  ____ (plan 4,575) / testing ____ (plan 2,520): ____
- Output: stage1_summary.json SHA-256 ____
- Pole and camera encoding (source quoted): ____
- Counts, training / testing: images ____ / ____ (plan 4,575 / 2,520); classes ____ / ____; pole numbers ____ / ____;
  sign units ____ / ____; views per sign unit ____ / ____; poles with more than one class ____ / ____ (extra sign units
  ____ / ____); consistency failures ____ / ____
- Shared sign units: check 1 applied? ____ (readme sentence or "not stated"); check 1 only ____, check 2 only ____,
  both ____, total removed ____; what check 1's count means here: ____
- After removal, testing release: poles ____ (F-A1 count) / sign units ____
- Expected limit (g): Cert size ____; n_c at 70% ____; tolerated errors at 0.5/1/2/3/5%: ____
- DITS page retry on the run day: HTTPS ____ (time UTC, error) / HTTP ____ (time UTC, error); DITS structure: ____
- Checks: F-A1 ____ F-A2 ____ F-A3 ____
- Decision taken and why: ____
- Deviation from the plan? ____
- Next step: ____
