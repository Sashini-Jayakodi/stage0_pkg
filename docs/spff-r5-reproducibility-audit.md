# SPFF-R5 reproducibility and repository-integrity audit

Audit date: 2026-11-22 (Asia/Colombo). Scope: existing synthetic research artifacts and repository integrity; no new methods, traffic-sign experiments, or manuscript edits. The recovery continuation below supersedes the earlier missing-evidence disposition retained later in this file.

## Recovery continuation: current findings

**[Certain] All substantive numerical and experimental claims in the identified manuscript have now been reproduced under the stated restricted synthetic models.** Recovered source and transparently reconstructed verification resolve the three former evidence gaps. Two last-digit rounding differences in one table are documented below; neither changes a substantive result. No material numerical conflict was found. This conclusion concerns numerical reproducibility, not independent peer review, novelty, real-data validation, or certified floating-point enclosures.

The user confirmed the intended manuscript as `SPFF_R5_IEEE_Revised_Source.tex`, alongside the recovery ZIP in `POC 3 plan - FYP/IEEE paper revisions`. Its title is **Forecasting False Certificates Along Fixed-Sequence Paths: Sharp Coupling Bounds, a General Inference Barrier, and Pilot-Honest Auditing**. Its n=784, m=261, two-threshold endpoints, three-threshold examples, and boundary constant match the requested SPFF-R5 target. SHA-256: `28201ee5725c148b2fffc083f97d7f7a80ea75770a99cfaa3a8962647ea830b6`. The manuscript was read only and is excluded from the research commit.

### Recovery verification and provenance

The ZIP was located before extraction. At the user's explicit direction it was then extracted into a **new** sibling directory `SPFF_R5_Missing_Evidence_Recovery/SPFF_R5_Recovery_Checks`, with a refusal to overwrite and archive path-containment checks. All seven entries in `SHA256SUMS.txt` passed both ZIP-byte and extracted-file checks. ZIP SHA-256: `3281cdee2cb857f69aaa064ee6b1418321fb743a1b2a1665fc383f978d057a46`. The manifest is a package-integrity check, not an external authenticity signature. `README_RECOVERY.md` was read before execution.

The supplied package, isolated execution copies, and local reruns are kept separate under `runs/spff_r5/recovery_audit_20261122/` (abbreviated **R** below). The executed sources were hash-matched to the extracted package; no modified or newer source was overwritten. The non-FFT program writes JSON beside its source, so it was run from an isolated copy rather than over the supplied result. A separate verified copy of the supplied JSON was retained before execution.

| File/evidence | Provenance and integration |
|---|---|
| `experiments/spff_r5_generality_probe.py` | Recovered original according to the recovery README and user provenance; supplied SHA-256 `44082a919551db3333c30b3cdd97549afb3b630980d98d96062e9c25784a5a4e`. The earlier named generality ZIP itself was not located, so its historical byte identity cannot be independently rechecked against that ZIP. |
| `experiments/spff_r5_conditional_pilot_prototype.py` | Existing copy was already byte-identical to recovery and to the original conditional package; preserved without replacement. SHA-256 `9d647570eda5a893be09a4ddfd1be2aad11f8c90fb188ee85b1fcaaf7bee9a64`. |
| `experiments/spff_r5_nonfft_recurrence_check.py` | **New reconstructed independent verification**, never described as recovered original. Supplied SHA-256 `5a89c136f8ee58c1d1c60bc5b8599b0af4a77da70edec2769842dba0586ce0ba`; integrated byte-identically. Its tested domain here is n=784 and the five stated couplings, not arbitrary arguments to the helper. |
| `R/supplied/nonfft_reproduction_results.json` and two supplied logs | **New supplied container reruns**, not original historical runs. Python 3.13.5, NumPy 2.3.5, SciPy 1.17.0 according to supplied metadata. Original bytes retained. |
| `experiments/spff_r5_manuscript_checks.py` | **New reconstructed verification** for rational sums, frozen-selector enumeration, explicitly defined coverage grid, and Wilson intervals. It tests existing manuscript claims and introduces no new model or research experiment. |
| `experiments/spff_r5_phase_b_prototype.py` | Original source copied without edits from the existing Phase B/C archive; SHA-256 `670e5e366bff2b4c43ccf89eb02acbc19bf4f9888568c7f42f78b091fbcb8fde`. Earlier exploratory/negative results remain distinct from the manuscript's final examples. |
| `runs/spff_r5/pivotality_archived_results.txt` | Original archive-only output copied unchanged from the pivotality ZIP; SHA-256 `27912aea4e19b726e36c7219540ba0a44ab2cc2e4c2e332a28f88baa80fcdb25`. |

The original non-FFT historical source/log and original extensive coverage-grid script/log remain unavailable. Replacement checks now reproduce their numerical claims; they do not recover their execution history. The reconstructed coverage grid is N=0..261 and t=0,.01,...,1; its match to the paper's exact count and minimum is evidence of numerical reproducibility, not proof of the original grid specification. Earlier raw FFT/MC execution logs are likewise not recovered; both routes have independently recorded reruns. These provenance limitations must accompany any future release. They do not imply that original artifacts were recovered.

### Actual missing-component results

- Three-threshold F: **0.002515049999999998** at n=7 and **0.01639500952499995** at n=10. Risks are (1/12,3/14,1/3). All eight signatures pass nesting checks; full outcome distributions match the manuscript's printed values. Probability-sum discrepancies are below 4e-15.
- Boundary at n=784: cutoff **218**, boundary passing probability **0.09588941721833312**, expected-width lower-bound constant **0.08630047549649981** (8.630047549649981 pp). The script evaluates `(1-2*.05)*b_n`; it does not empirically prove the theorem. The manuscript's two-point argument and assumptions were inspected separately; no numerical conflict was found. Supplied versus local constant differs by about 2.4e-16.
- Non-FFT endpoints: **0.035340709976372696** and **0.07456656990588575**. Across t=(0,.25,.5,.75,1), maximum |F_nonFFT-F_FFT| is **5.0105752880114096e-14**; maximum derivative discrepancy is **2.435551760271437e-15**. Both meet the manuscript's reported approximate differences. The recurrence uses nonnegative forward convolution and discards only counts outside the pass rectangle; subsequent nonnegative increments cannot return them to that rectangle.
- Non-FFT identification width: **3.922585992951306 pp**; half-width/minimax arithmetic: **1.961292996475653 pp**. FFT values differ only in the last few numerical digits. These calculations are consistent with the supplied headline precision.

Local runtime including interpreter/import startup: generality **1.8727 seconds**; five-point non-FFT/FFT check **11.3731 seconds**. Per-point times, including its FFT comparator, are **1.9163, 2.3105, 2.2484, 2.2232, 1.9264 seconds**. These are measurements of this rerun, not replacements for historical runtime claims.

### Complete manuscript numerical mapping

All paths below are repository-relative. **R** is `runs/spff_r5/recovery_audit_20261122`. Two-threshold parameters are n as stated, alpha=.30, delta=.10, strict-then-loose order, acceptance (1 strict,3 loose), and q(t) from `weights(t)`. Deterministic checks have no random seed. Original historical outputs, supplied new reruns, and local reruns must not be conflated.

| Manuscript location/claim | Script, command/function, parameters, seed | Output and status |
|---|---|---|
| Full fixed marginals, means .91/.29, risks .30333/.29, excess 1/300 | `conditional_pilot_verify.py:check_fixed_marginals`; q(t), 11 t points; deterministic; mathematical definitions inspected | `R/conditional-verify-local.txt`; PASS. Source equations give exact means/excess. |
| Table I positioning/literature | Not a numerical experiment | Outside numerical verification; no new literature-validation claim. |
| Table II final-selection distributions n=7,120,784 at t=0,1 | `py -B experiments/spff_r5_fft_path_experiment.py`; `oracle(q,n)`; deterministic | Prior audit `fft.txt`, plus current five-point FFT comparator and exact small-n checks; reproduced. Two n=784 no-certificate entries have the last-digit rounding issue described below. |
| Main endpoints, cutoffs (676,218), interval width .03922586, minimax 1.9613 pp, endpoint ratio ~2.11 | `py -B spff_r5_nonfft_recurrence_check.py` in isolated execution directory; `full_path`; deterministic n=784 | `R/execution/nonfft_reproduction_results.json`, `R/nonfft-local.txt`; reproduced. Ratio ~2.109934122878. Complete-coupling and monotonicity arguments support interpretation; five grid points alone are not a proof of sharpness. |
| Figure 1 five F values and derivatives .03299,.04289,.03903 | Same reconstructed recurrence/FFT check; t=0,.25,.5,.75,1 | Five machine-readable rows in local non-FFT JSON; plot coordinates agree at displayed precision. |
| Exact rational n=7 F/derivative agreement ~5.6e-17 | `py -B experiments/spff_r5_manuscript_checks.py --output <new-output.json>`; exact integer multinomial sums divided by 100**n, t=0,.25,.5,.75,1; deterministic | `R/manuscript-checks.json:small_n`; max difference **5.551115123125783e-17**. New check, not a historical source recovery. |
| 78,125 ordered paths per endpoint with frozen R9 selector | Same script `small_n()`, n=7, each endpoint's five positive-mass signatures; fixed start=1; `r9_certify.binom_pvalue` and `walk`, cached by identical totals | **78,125 per endpoint**, zero selector disagreements, exact total mass 1. `R/manuscript-checks.json:small_n.ordered_endpoints`; full selection max FFT discrepancy 6.25e-17; independent cluster engine max discrepancy 4.45e-16. |
| Independent cluster-count engine and variable acceptance | `py -B -m pytest tests/test_spff_r5_path.py tests/test_spff_r5_cluster_path.py -q -p no:cacheprovider`; independent enumeration, n=0..4 and n=7 counterexample, fixed-start and invalid-input cases | `R/focused-tests.txt`; 140 passed. Frozen selector and POC files unchanged. |
| Three-threshold full outcome distributions and risks at n=7,10; eight appendix signatures | `py -B spff_r5_generality_probe.py` in execution directory; alpha=.30,delta=.10, Q=(.30,.05,.05,.10,.10,.10,.20,.10), three frames, strict-to-loose ascending columns; deterministic | `R/generality-local.txt`; both distributions and signatures match Section VII and appendix. Composition counts C(14,7)=3432 and C(17,7)=19448. |
| Safety-boundary b_n=.0958894172, lower bound .0863004755 | Same probe `exact_single_cutoff(784)`; eta=.05; deterministic | `R/generality-local.txt`; n=7 and n=120 supplementary constants also recorded; n=784 headline reproduced. |
| Table III conditional pilot coverage/mean and max widths | `conditional_pilot_prototype.run(reps=15,m=261,n=784,seed=20261119)`, t=0,.5,1, shared RNG in that order | Historical `experiments/spff_r5_conditional_pilot_results.json`, byte-identical ZIP copy; local `R/conditional-pilot-local.json`; 15/15,14/15,15/15 and all widths reproduced to floating precision. Main-block default is 40, so an explicit 15 is required. |
| Conditional derivative and finite differences | `py -B experiments/spff_r5_conditional_pilot_verify.py`; n=7,120,784 and t=.1,.4,.7,.9; h=1e-4 | `R/conditional-verify-local.txt`; all 12 comparisons pass <1e-7. |
| 26,462 conditional-binomial cases, minimum .95004809 | `spff_r5_manuscript_checks.py:coverage_grid`; N=0..261, 101 t values, eta=.05; deterministic | `R/manuscript-checks.json:conditional_binomial`; **26,462 pass**, min **.9500480881074119 at N=248,t=.67**. Includes N=0. **34,453** distinct (N,K) endpoint pairs matched the existing prototype; binomial-tail inversion residual <=1.680e-14. Original 126-setting check also passes separately. |
| 350,000 draws per endpoint, all six outcomes inside pointwise 95% Wilson intervals, max z=1.453 | `spff_r5_manuscript_checks.py:monte_carlo`, n=784, seed=20261119, shared RNG t=0 then t=1; exact original draw order preserved | `R/manuscript-checks.json:monte_carlo`; all six interval inclusions pass, max |z|=**1.4525991506522342**. Original MC script and first-audit transcript retained. |
| FFT negative mass ~5.07e-13/6.79e-13 and ~.11 sec per endpoint | First-audit FFT executable, same restricted model | `runs/spff_r5/audit_20261122/fft.txt`; masses 5.0681e-13/6.7883e-13; times .1201/.1221 sec. Approximate runtime is machine/load dependent. No certified error-enclosure claim. |
| Hypothetical tolerance .05 between endpoints | Arithmetic using reproduced endpoints | Both endpoints straddle .05; no empirical resolution/inconclusive-rate experiment is claimed or performed. |
| General confidence-set inversion and severity stability propositions | Mathematical representations and stated proofs inspected; no numerical experiment or tractability promise in manuscript | Not promoted to a new empirical result. No 30-threshold or real-image result is inferred. |
| Appendix verification table | Combination of the exact-path, cluster, recurrence, MC and coverage checks above | Every numerical verification target reproduced; historical execution provenance remains limited as described. |

**Minor printed-rounding finding:** Table II prints the n=784 no-certificate probability as `.75630839` for both endpoints. Rerun values are `.7563083849888659` and `.7563083849888446`; conventional rounding to eight decimal places gives `.75630838`. Absolute differences from the printed entries are about 5.011e-9, just over half one last-place unit. This does not affect the reported F endpoints, width, inference, or conclusions. It is recorded as a nonmaterial last-digit discrepancy; the manuscript was not edited. The substantive numerical findings are reproducible, but literal identity of every printed last digit is not claimed.

### Commands, outputs and environment

The current environment is unchanged from the first audit: Python 3.13.14, NumPy 2.5.3, SciPy 1.18.1 on Windows 11, Intel i7-13650HX, 20 logical processors, 16,873,545,728 bytes RAM. No GPU was used. Measured dependency pins are in `configs/spff_r5_audit_requirements.txt`; these are **current** reproduction versions, not historical-run metadata. Pytest 9.2.0 is loaded from the isolated temporary installation, leaving project dependencies untouched.

To run both recovery programs safely from repository sources, use a **fresh** directory because the recurrence writes a source-adjacent JSON. These commands intentionally refuse a pre-existing rerun folder:

```powershell
New-Item -ItemType Directory runs/spff_r5/recovery_rerun -ErrorAction Stop | Out-Null
Copy-Item experiments/spff_r5_generality_probe.py runs/spff_r5/recovery_rerun/
Copy-Item experiments/spff_r5_nonfft_recurrence_check.py runs/spff_r5/recovery_rerun/
Copy-Item experiments/spff_r5_conditional_pilot_prototype.py runs/spff_r5/recovery_rerun/
py -B runs/spff_r5/recovery_rerun/spff_r5_generality_probe.py
py -B runs/spff_r5/recovery_rerun/spff_r5_nonfft_recurrence_check.py
py -B experiments/spff_r5_manuscript_checks.py --output runs/spff_r5/recovery_rerun/manuscript-checks.json
py -B experiments/spff_r5_conditional_pilot_verify.py
```

Use the explicit 15-replicate function invocation in the earlier record to reproduce Table III; do not run its 40-replicate default and relabel the output. The source-adjacent recurrence output contains an inherited container-oriented provenance string. Local run location and current environment are established by this report, the execution record and manifest, not by treating that inherited label as a historical fact.

Actual test results in this continuation:

```text
Focused R5: 140 passed in 3.75s.
Full repository attempt: 298 passed, 6 skipped, 26 collection errors in 53.60s.
```

The 140 are included in the 298, not 438 distinct tests. The full suite used `py -B -m pytest tests -q -p no:cacheprovider --continue-on-collection-errors`, with plugin autoload disabled and temporary pytest on PYTHONPATH as in the earlier record. The errors are missing historical dependencies, primarily torch and imagecorruptions; the full repository suite is **not green**. No frozen test or dependency failure was hidden or repaired. The standalone manuscript verifier completed in **4.7245 seconds**, separately from pytest counts; the 26,462 grid cases are not reported as pytest tests. The conditional verifier took **3.2181 seconds** including startup, and the 45-run pilot calculation **10.1405 seconds**.

### Security, repository state and release decision

All seven recovery hashes passed; existing source and archive hashes from the first audit remained unchanged. The new verification files and candidate research diffs were reviewed for credential assignments, GitHub/AWS token patterns, private-key markers, email addresses, personal profile paths and unrelated artifacts. No secret-pattern hit was found. `R/security-review.json` lists only the pre-existing mathematical audit's personal profile paths (without printing their values).

SHA-256 values in the candidate inventory describe the measured local file bytes. Git normalizes CRLF to LF for some text evidence on commit; the local staged-security record captures both byte hashes. All three recovery Python sources retain the exact supplied hashes in the staged blobs. This line-ending normalization changes no experimental value or source logic.

`docs/spff-r5-mathematical-audit.md` remains unchanged and is excluded from the commit because it contains unnecessary personal machine paths. Original ZIPs, the external manuscript and recovery ZIP, redundant execution/source snapshots, Python/pytest caches, datasets, checkpoints, virtual environments, and unrelated POC assets are also excluded. The supplied README/manifest and small provenance logs/results are retained separately from local reruns. Original archive-only Phase B source and pivotality output were copied without changing their archives.

The continuation started on `spff-r5` at `681c1ad305769edd462b61c957338d7cc9501b54`, with the same two pre-existing modified files and untracked research/audit files recorded below. The live remote was rechecked at that same base. There is no branch divergence or source conflict. The release gate is satisfied for the manuscript's **substantive synthetic numerical results** with the disclosed nonmaterial rounding difference, replacement provenance and full-suite dependency limitation. No manuscript, main branch, previous POC implementation or historical experimental output was changed.

The audit commit is identified unambiguously by `git log -1 --format=%H -- docs/spff-r5-reproducibility-audit.md`; embedding its own hash in this document would change that hash. Final commit, push outcome, exact staged-file manifest and remaining local exclusions are recorded in the final delivery and the local post-commit record `R/final-delivery.json`. The pre-commit snapshot and staged security report are saved beside it. Untracked original archives and excluded historical/privacy material intentionally remain in the worktree.

## Earlier audit record (before recovery; superseded where noted)

## Disposition

**[Certain] Commit and push withheld under the requested stop condition for missing essential files.** The available FFT endpoints, conditional pilot summary, and focused tests reproduce successfully. However, the non-FFT n=784 recurrence, final three-threshold variable-acceptance experiment, generality probe, and approximately 8.63 percentage-point safety-boundary lower-bound evidence could not be located. The manuscript itself was not supplied, so a complete table-by-table manuscript reconciliation is not possible. This report maps the requested headline results and every experiment family identified in the available R5 records; it does not certify every manuscript claim.

No files were staged, committed, pushed, switched, discarded, or overwritten. All pre-existing research files and ZIP archives remain byte-for-byte unchanged. New audit outputs are separate in `runs/spff_r5/audit_20261122/`. No frozen POC implementation, historical record, main branch, repository visibility, or pull request was changed.

## Git state

- Root: the supplied `stage0_pkg_spff_r5_audit` working directory. It is a linked worktree; the other worktree was not modified.
- Branch throughout: `spff-r5`.
- Initial and final HEAD: `681c1ad305769edd462b61c957338d7cc9501b54`.
- Cached `origin/spff-r5`: same commit; `git rev-list --left-right --count HEAD...origin/spff-r5` returned `0 0`.
- Live `git ls-remote --heads origin spff-r5` also returned that full hash. No fetch/pull was necessary to change the worktree; no new local commits exist on this branch relative to its remote.
- Other local branches: `POC_2`, `bashee`, `main`, `poc2`, `poc3`. Remote-tracking branches: those six names plus `spff-r5`; `origin/HEAD` points to `origin/main`. All were left untouched.
- Initial and final index: empty staged diff. `git diff --check` passed, with LF-to-CRLF notices only.
- Initial unstaged diff: `src/spff_r5_path.py` (73 insertions, 26 deletions) and `tests/test_spff_r5_path.py` (171 insertions, 10 deletions). These are existing local input validation, stable binomial-tail, fixed-start, and independent-oracle test changes; they were inspected and preserved.

Initial porcelain status:

```text
 M src/spff_r5_path.py
 M tests/test_spff_r5_path.py
?? docs/spff-r5-conditional-pilot-findings.md
?? docs/spff-r5-fft-findings.md
?? docs/spff-r5-mathematical-audit.md
?? docs/spff-r5-phase-b-research-note.md
?? docs/spff-r5-pivotality-findings.md
?? experiments/spff_r5_conditional_pilot_prototype.py
?? experiments/spff_r5_conditional_pilot_results.json
?? experiments/spff_r5_conditional_pilot_verify.py
?? experiments/spff_r5_fft_path_experiment.py
?? experiments/spff_r5_fft_verify.py
?? experiments/spff_r5_pivotality_audit.py
?? spff_r5_conditional_pilot_package.zip
?? spff_r5_hardening_patch.zip
?? spff_r5_phase_b_c_research_bundle.zip
?? spff_r5_pivotality_package.zip
?? src/spff_r5_cluster_path.py
?? tests/test_spff_r5_cluster_path.py
```

Final status retains all entries above and adds this untracked report and the untracked audit-output directory. The complete final per-file status is saved in `runs/spff_r5/audit_20261122/final-status.txt`. Files committed/pushed: **none**.

## Discovery and archive provenance

Recursive filesystem discovery included ignored files and ZIP member inventories. `inventory.json` records original relative paths, sizes, SHA-256 hashes, tracked/ignored status, archive members, and duplicate comparisons. No research file was relocated or extracted over an existing file.

| Original location | Initial state and evidence |
|---|---|
| `src/spff_r5_path.py` | Tracked, modified; binary sign-signature reference |
| `tests/test_spff_r5_path.py` | Tracked, modified; frozen-R9 comparison and validation tests |
| `src/spff_r5_benchmark.py` | Tracked, unchanged; finite-pilot synthetic benchmark |
| `runs/spff_r5/finite_pilot_synthetic.json` | Tracked, unchanged; original n=8 benchmark summary |
| `docs/spff-r5-results.md` | Tracked, unchanged; benchmark results/limitations |
| `docs/spff-r5-handoff.md` | Tracked, unchanged; historical implementation/provenance instructions |
| `src/spff_r5_cluster_path.py` | Untracked; byte-identical to hardening ZIP source |
| `tests/test_spff_r5_cluster_path.py` | Untracked; byte-identical to hardening ZIP tests |
| `experiments/spff_r5_fft_path_experiment.py` | Untracked; FFT oracle, no matching archive found |
| `experiments/spff_r5_fft_verify.py` | Untracked; 350,000 Monte Carlo draws per endpoint |
| `experiments/spff_r5_conditional_pilot_prototype.py` | Untracked; byte-identical to conditional ZIP member |
| `experiments/spff_r5_conditional_pilot_verify.py` | Untracked; byte-identical to conditional ZIP member |
| `experiments/spff_r5_conditional_pilot_results.json` | Untracked; byte-identical to conditional ZIP member |
| `experiments/spff_r5_pivotality_audit.py` | Untracked; byte-identical to pivotality ZIP member |
| `docs/spff-r5-conditional-pilot-findings.md` | Untracked; byte-identical to conditional ZIP findings |
| `docs/spff-r5-fft-findings.md` | Untracked; historical FFT/MC summary; no raw original FFT transcript found |
| `docs/spff-r5-mathematical-audit.md` | Untracked; historical Phase A audit; contains local paths |
| `docs/spff-r5-phase-b-research-note.md` | Untracked; byte-identical to Phase B ZIP note |
| `docs/spff-r5-pivotality-findings.md` | Untracked; byte-identical to pivotality ZIP findings |

The inventory has 18 files whose names contain SPFF; the benchmark JSON is additionally identified above because its parent directory, rather than its filename, contains SPFF.

All four root ZIPs are untracked, small (4,207 to 24,669 bytes), and preserved locally:

- `spff_r5_conditional_pilot_package.zip`: four members, all present byte-identically as code, findings, or JSON in the table.
- `spff_r5_hardening_patch.zip`: current cluster source/tests plus README, pytest cache, and compiled Python caches. Its README identifies the older Phase B/C bundle as the base and explains int64 overflow checks and the no-start short circuit. Its historical claim of 28 passed is not substituted for this audit's test result.
- `spff_r5_phase_b_c_research_bundle.zip`: older cluster source/tests, the research note, and `spff_r5_research_bundle/phase_b/spff_r5_phase_b_prototype.py`. The prototype exists **only in this archive**. Source differences against the hardening version match the README's three changes and added tests; they are explained version history, not an unresolved conflict. No version was overwritten.
- `spff_r5_pivotality_package.zip`: identical script/findings and the original `spff_r5_pivotality_results.txt`, which is **archive-only**. It records n=7,12,20 replacement-change and pivotal-context probabilities for A/B. It was inspected, not replaced with a new record.

Ignored R5 artifacts found: compiled Python files in `src/__pycache__`, `tests/__pycache__`, and `experiments/__pycache__`. Pytest caches are also excluded by `.gitignore`. Git reported permission warnings for the user-level global ignore and `.pytest_cache`; no configuration was changed to bypass them. Classification uses repository rules and observed files, with that global-ignore access limitation.

Missing essential evidence: `spff_r5_generality_probe.py`; a distinct non-FFT n=784 recurrence implementation and its original output; the final three-threshold experiment script/output; the safety-boundary derivation/implementation/output; original raw FFT and MC transcripts. No R5-specific configuration file or final manuscript was found. The archived Phase B prototype is not missing, but is not yet a standalone repository file.

## Result-to-code-to-output mapping

All A/B computations below use iid clusters with fixed acceptance counts (3 loose, 1 strict), cluster size 3, error patterns `(0,0),(1,0),(1,1),(2,0),(2,1),(3,1)`, alpha=.30 and delta=.10, strict-to-loose walk. A has q=(.44,.02,.24,.25,0,.05); B has q=(.44,.26,0,.01,.24,.05). These are synthetic pooled-frame certificate calculations, not a valid sign-level safety guarantee or a real traffic-sign experiment.

| Result/experiment | Responsible code/input/seed | Original output | Audit verification |
|---|---|---|---|
| Exact n=7 enumeration, A/B F=.02051947628544/.06517639096320 | Archive-only Phase B prototype `exact_counterexample()`; enumerates 5^7 sequences per population; deterministic | Phase B research note; no original raw enumeration transcript located | Existing function executed unchanged from ZIP in memory; its formula assertions passed; `exact-n7.txt` |
| Independent cluster-count computation | `src/spff_r5_cluster_path.py:exact_cluster_path`; six-point model above, n=7, deterministic | Phase B note; source preserved in base/hardening ZIPs | Existing n=7 test passed; tests also compare independent brute force at n=0..4, all four starts, variable acceptance/nonmonotone errors |
| FFT generating function n=7,20,120,784 | `experiments/spff_r5_fft_path_experiment.py:oracle`; deterministic | `docs/spff-r5-fft-findings.md` summary | All eight A/B rows rerun; `fft.txt` |
| F(0), F(1), main calibration n=784 | FFT oracle above; `conditional_pilot_prototype.full_path`; family q(t) in `weights(t)` | Conditional JSON `true_F` and findings | F(0)=.03534070997642254; F(1)=.07456656990593194; cutoff loose=676, strict=218 |
| Identification width and minimax error | Arithmetic on the above endpoints; no random seed | Supplied headline request; endpoint JSON supports arithmetic | 100*(F1-F0)=3.9225859929509403 pp; 50*(F1-F0)=1.9612929964754702 pp. This verifies arithmetic, not an independently supplied minimax theorem/proof |
| Non-FFT recurrence at n=784 | Not located; the small-state cluster reference is not evidence that this separate large-n calculation was archived | Not located | Unverified; no replacement algorithm developed |
| 350,000 MC draws per endpoint | `experiments/spff_r5_fft_verify.py`; n=784; seed 20261119; one RNG, A then B | FFT findings report standardized discrepancies below 1.5; raw original transcript absent | Rerun completed; A max absolute difference .0010540992745801114, z=(-1.453,.263,1.416); B .0001692406090622045, z=(-.055,-.291,.267); `monte-carlo.txt` |
| Conditional pilot inference m=261 | `conditional_pilot_prototype.run(reps=15,m=261,n=784,seed=20261119)`; t=0,.5,1, 15 pilots each, known fixed marginals | `experiments/spff_r5_conditional_pilot_results.json`, identical ZIP copy | Coverage counts 15/15,14/15,15/15; maximum numeric row differences 2.73e-15,4.69e-16,1.39e-17; timing intentionally excluded; new JSON separate |
| Conditional coverage/derivative/marginals | `experiments/spff_r5_conditional_pilot_verify.py`; deterministic | Conditional findings summary | PASS fixed marginals at 11 t values, derivative at 12 settings, conditional CP coverage at 126 settings; `conditional-verify.txt` |
| Three-threshold variable-acceptance final experiment | `spff_r5_generality_probe.py` absent. Earlier three-threshold support and six Gaussian cases exist in archived Phase B prototype | Earlier Phase B note only | Earlier code is not presumed to be the missing final experiment; unresolved |
| Boundary expected-width lower bound ~8.63 pp | Not located | Not located | Unverified; not inferred from other numbers |
| Preliminary finite-pilot severity benchmark | `src/spff_r5_benchmark.py:run`; seed 20261119; n=8; q=(.5,.2,.3), alpha=delta=.3, m=200/1000/3000; 30 pilots, 40 bootstraps | `runs/spff_r5/finite_pilot_synthetic.json`; results/handoff | Archive inspected; full stochastic benchmark not rerun; existing fixed-start test reproduces S=.001771875 |
| Phase B U-statistic and Gaussian negative results | Archive-only prototype: `u_stat_experiment(reps=100)`, seed 20261119; `gaussian_experiment()`, seeds 123+i,456+i,900+i; n=120,m=261, 40,000 oracle/Gaussian draws, 30 pilots, 3,000 pilot forecast draws | Phase B note; no original raw output located | Code/record inspected, not rerun. n<=m limitation and large plug-in errors preserved; supplementary n=784 probes have no located standalone reproduction record |
| Pivotality negative findings | `experiments/spff_r5_pivotality_audit.py:audit`; deterministic n=7,12,20, A/B | Pivotality ZIP `spff_r5_pivotality_results.txt` and findings | Original transcript inspected; not rerun |

FFT coefficient calculations clamp negative mass and normalize, without certified numerical enclosures. At n=784 this rerun observed negative mass 5.0681e-13 (A), 6.7883e-13 (B). Post-normalization error is not a certified bound on pre-normalization or endpoint error. MC agreement is diagnostic, not a proof of numerical error bounds. Conditional inference assumes known nuisance marginals; 45 pilot runs do not establish nominal coverage empirically. Existing limitations and negative results were preserved.

## Exact reproduction commands

Run from the repository root with NumPy, SciPy, and pytest available. `py` selected Python 3.13 during this audit; the documented historical command is `py -3.12`, which is not registered here. Commands below never target original outputs.

```powershell
py -B -m pytest tests/test_spff_r5_path.py tests/test_spff_r5_cluster_path.py -q -p no:cacheprovider
py -B experiments/spff_r5_fft_path_experiment.py
py -B experiments/spff_r5_fft_verify.py
py -B experiments/spff_r5_conditional_pilot_verify.py
```

The initial pytest attempt failed because pytest was absent. Pytest 9.2.0 and dependencies were installed only in a dedicated temporary directory. The restricted attempt could not access that installation. The successful test invocation used that directory with normal filesystem access and these environment settings, without modifying repository dependencies:

```powershell
$env:PYTHONPATH = "$env:TEMP\spff-r5-audit-20261122-deps"
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = '1'
py -B -m pytest tests/test_spff_r5_path.py tests/test_spff_r5_cluster_path.py -q -p no:cacheprovider
```

Actual outcome: **140 passed in 3.46s**. Failure and success transcripts are retained separately. The entire historical test suite was not run because the task concerns R5 and the current runtime lacks historical dependencies including torch.

To repeat the exact archived enumeration without extracting or changing source:

```powershell
@'
import zipfile
with zipfile.ZipFile('spff_r5_phase_b_c_research_bundle.zip') as z:
    source=z.read('spff_r5_research_bundle/phase_b/spff_r5_phase_b_prototype.py')
namespace={'__name__':'archived_phase_b_audit'}
exec(compile(source,'archived_phase_b_prototype.py','exec'),namespace)
print(namespace['exact_counterexample']())
'@ | py -B -
```

To repeat the original 15-replicate conditional calculation (the script's main block instead defaults to 40):

```powershell
@'
import sys
sys.path.insert(0,'experiments')
from spff_r5_conditional_pilot_prototype import run
rows,seconds=run(reps=15,m=261,n=784,seed=20261119)
print(rows)
print(seconds)
'@ | py -B -
```

The audit saved those rows to `conditional-pilot-rerun.json` and numeric comparisons to `conditional-pilot-comparison.txt`, leaving the original JSON intact. The archived JSON does not embed a seed; the seed above is the implementation default and its rerun matches the archived rows to floating precision. That match does not prove the unrecorded original invocation.

Additional existing commands, identified but **not executed** in this audit:

```powershell
py -B -m src.spff_r5_benchmark --replicates 30 --boots 40 --output runs/spff_r5/audit_20261122/finite-pilot-rerun.json
py -B experiments/spff_r5_pivotality_audit.py
```

The archived Phase B prototype can likewise be loaded as above and its existing `u_stat_experiment()` / `gaussian_experiment()` called. No exact command can be provided for the missing final recurrence, generality, or boundary artifacts.

## Computing environments

**Current, measured:** Windows 11 build 26200, AMD64; Python 3.13.14, MSC v.1944; NumPy 2.5.3; SciPy 1.18.1. Processor registry: 13th Gen Intel Core i7-13650HX, 20 logical processors. Windows memory API: 16,873,545,728 bytes physical RAM (15.7147 GiB). CIM OS/CPU/RAM/GPU queries were denied; alternate OS/CPU/RAM APIs supplied the above. GPU model was not verified and `nvidia-smi` was unavailable. No GPU computation was used. Torch/psutil were absent; pytest was initially absent, then 9.2.0 was installed in the isolated temporary directory. Exact test-only dependency versions are recorded in `inventory.json`.

**Historical, documented rather than remeasured:** `docs/HANDOFF.md` reports a 2026-11-14 Windows environment with Python 3.12.10, NumPy 2.5.2, SciPy 1.18.1, scikit-learn 1.9.1, torch 2.14.1+cu132, CUDA available, RTX 5050. `CLAUDE.md` records 16 GB RAM and an 8 GB RTX 5050. `env_poc2_stage1.txt` is a prior POC environment record. None of these proves the environment used to create each standalone R5 ZIP. The hardening archive contains CPython 3.13/pytest 9.0.2 cache names; local ignored caches contain CPython 3.12/pytest 9.1.1 names. These are provenance clues, not a complete original runtime manifest. The original R5 CPU, RAM, GPU usage, and full dependency set remain unverified.

## Security and exclusion review

Reviewed the unstaged research diff, research scripts/tests/JSON/notes, archive text members, and file inventory. Automated scans for credential assignments, common GitHub/AWS token formats, private-key markers and email addresses found no secret-pattern or private-email hits in the research candidates. This is a scoped inspection, not a guarantee that the entire historical repository is secret-free. The staged diff was empty throughout.

Local machine paths are present in the historical mathematical audit at lines 16,38,236. They describe the worktree and prior permission warnings; one includes the local Windows profile name. They are unnecessary for publication and should be reviewed for redaction before any future commit. Original text was preserved, not silently sanitized. Several automated absolute-path hits were DOI/HTTPS URL false positives. No `.env` file, credential file, dataset, checkpoint, environment directory, or unrelated personal file was selected for staging.

Excluded from any proposed commit: all four ZIPs (packaging duplicates and caches; archive-only essential text still requires careful integration), all Python/pytest caches, raw datasets/images, checkpoints, virtual environments, large generated artifacts, and unrelated POC files. No existing SPFF file or original output was deleted. Audit reruns are small text/JSON evidence and remain local/untracked pending resolution of the stop conditions. No security-clean staged research change or commit is claimed.

## Remaining work before release

1. Locate the actual final generality/three-threshold, non-FFT recurrence, and safety-boundary source and original outputs; resolve their relation to manuscript tables and proofs. Do not reconstruct missing historical evidence from desired numbers.
2. Supply the final manuscript or its complete result inventory for exhaustive reconciliation. Current endpoint agreement alone does not validate all manuscript results or theoretical optimality claims.
3. Review the unnecessary personal machine paths before publishing the historical audit. Preserve an original copy if a redacted repository copy is authorized.
4. Integrate individually reviewed archive-only source/output as separate files, preserving archive provenance; record a reproducible dependency/environment manifest based on evidence.
5. Only after resolving essential gaps, stage individually reviewed research files, inspect the staged diff/security findings, commit on `spff-r5`, recheck the live remote, and push only a safe fast-forward. Network access was sufficient to verify the remote, but no push was attempted because the evidence gate failed.
