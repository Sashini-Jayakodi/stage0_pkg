# SPFF-R5 evidence recovery — supplementary verification

This package addresses three checks flagged missing in the author's local reproducibility audit. It contains no real-image experiment and does not alter original research archives. Do not interpret these new checks as an independent human peer review or rigorous floating-point error certification.

## Provenance

- `spff_r5_generality_probe.py`: recovered byte-for-byte from `SPFF_R5_Generality_Research_Package.zip`. The same script computes both the three-threshold variable-acceptance example and the one-threshold safety-boundary lower-bound constants.
- `spff_r5_conditional_pilot_prototype.py`: recovered byte-for-byte from `spff_r5_conditional_pilot_package.zip`; used as the archived FFT comparator.
- `spff_r5_nonfft_recurrence_check.py`: newly reconstructed independent recurrence, not a recovered original historical script. Uses nonnegative forward convolution, truncating only counts that have already exceeded the joint-pass rectangle; later increments cannot return those counts to the rectangle. The derivative is evaluated from the boundary coefficient after n-1 increments. The separate FFT comparator checks the same five t values (0, .25, .5, .75, 1).
- `generality_reproduction_log.txt`, `nonfft_reproduction_log.txt`, `nonfft_reproduction_results.json`: outputs from new container reruns. The manuscript's prior historical execution logs are not asserted to have been recovered.

## Run commands

In an isolated copy of this directory, with Python, NumPy and SciPy installed:

```sh
python spff_r5_generality_probe.py
python spff_r5_nonfft_recurrence_check.py
```

The second command writes `nonfft_reproduction_results.json` alongside its source. Preserve the supplied JSON before rerunning if you want to compare records. Environment used for these new runs: Python 3.13.5, NumPy 2.3.5, SciPy 1.17.0. This is not the original Windows environment.

## Observed results

- Three thresholds: risks 0.08333333, 0.21428571, 0.33333333. Unsafe-final-selection probabilities 0.00251505 at n=7 and 0.016395009525 at n=10. Both full outcome distributions reproduce the manuscript to its printed precision.
- One-threshold boundary, alpha=.30, delta=.10, eta=.05, n=784: cutoff 218; boundary pass probability 0.09588941721833338; theoretical expected-width lower bound 0.08630047549650005 (8.6300 percentage points). The script evaluates the bound, not a proof of it.
- Two thresholds, n=784, cutoffs a=676 and b=218: non-FFT endpoints 0.035340709976372696 and 0.07456656990588575. Maximum absolute discrepancy across the five grid points: approximately 5.011e-14 in unsafe-selection probability and 2.436e-15 in the derivative. Agreement is numerical, not a certified enclosure.

## Integration and remaining checks

1. Extract outside the Git worktree, compare hashes of files already present locally, and keep the original versions. Do not overwrite a modified local file.
2. Add the recovered/new files and logs to an explicitly designated R5 verification location on `spff-r5`, preserving import compatibility. Keep historical results separate from new reruns. Use the source-hash manifest to verify copies.
3. Update the local reproducibility audit with separate original/recovered/new statuses. The 140-test result and other historical claims must be supported by their actual local logs. Verify any remaining manuscript claims (including extensive conditional coverage grid and exact small-n/frozen-selector comparison) rather than assuming these three checks verify them.
4. Review the staged files and security findings. Commit only verified research files on the private branch after resolving conflicts or unexplained discrepancies. Do not alter old frozen project code or publish the repository. No commit or push was performed in preparing this package.
5. If historical recurrence source remains absent, describe the reconstructed check and its actual observed differences in the revised manuscript, instead of attributing its output to the original execution. Full mathematical proof review remains separate.
