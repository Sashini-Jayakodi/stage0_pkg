# SPFF-R5: Exact path computation and finite-pilot diagnostic

## Scope
This is an exploratory extension based on the frozen POC-3 `poc3` branch. It does not change or replace the existing POC-3 certification engine, pre-registration, result tables, or recorded confirmatory findings.

## Implemented
- Joint physical-sign signatures contain binary acceptance and wrong-commit indicators at every threshold.
- An exact reference dynamic program propagates the joint count state and applies the existing strict-to-loose fixed-sequence selection convention.
- Conditional wrong-sign excess severity is weighted by the probability that each threshold is finally selected.
- A conservative finite-pilot interval uses simultaneous coordinate-wise concentration bounds and total-variation coupling. The signature alphabet must be complete and fixed independently of pilot observations.
- A synthetic benchmark compares this conservative bound with percentile bootstrap intervals; percentile intervals are not finite-sample guarantees.

## Checks and results
- Six local tests passed. Three tests compare selected threshold probabilities, false-selection probabilities, and excess severity against direct enumeration for 2, 4 and 6 calibration signs. Other tests cover variable acceptance, nonzero errors, input validity, and the state-count guard.
- Synthetic benchmark uses 3 signature patterns, 3 thresholds, 8 independent calibration signs, alpha=0.30, delta=0.30, 30 pilot replicates per pilot size, and 40 percentile-bootstrap draws per replicate.
- Exact reference population excess severity: 0.001771875 (probability scale).

| Pilot signs | Point-estimate mean absolute error | Conservative interval empirical coverage | Conservative mean interval width | Percentile bootstrap empirical coverage |
|---:|---:|---:|---:|---:|
| 200 | 0.000686812 | 30/30 | 1.0000 | 29/30 |
| 1,000 | 0.000223979 | 30/30 | 0.6623 | 25/30 |
| 3,000 | 0.000070260 | 30/30 | 0.3831 | 28/30 |

Widths are on a 0–1 probability scale. Conservative coverage of 30/30 is not evidence of useful precision: the intervals were extremely broad. Bootstrap empirical coverage is noisy with only 30 replicates and 40 draws per replicate.

## Limits
This is a fixed three-signature synthetic illustration, not a traffic-sign dataset result or a general 30-threshold scalability demonstration. The dynamic program has a state-count guard and may become infeasible. The conservative finite-pilot interval is valid under its stated i.i.d. fixed-alphabet assumptions but is not practically resolved in this benchmark. No IEEE Access novelty threshold is claimed.

## Next experiments
- Develop and prove a sharper pilot-honest interval exploiting the selected-path functional rather than multiplying a full-distribution bound.
- Compare against general multinomial-functional inference and optimized cluster/bootstrap alternatives at matched measured runtime.
- Test full variable acceptance with realistic sign tracks, more score thresholds, and actual pilot/pool separation.
