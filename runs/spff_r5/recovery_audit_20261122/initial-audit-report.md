# SPFF-R5 reproducibility and repository-integrity audit

Audit date: 2026-11-22 (Asia/Colombo). Scope: existing synthetic research artifacts and repository integrity; no new methods, traffic-sign experiments, or manuscript edits.

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
