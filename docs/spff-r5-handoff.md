# SPFF-R5 research and implementation handoff

Status date: 2026-11-19. This document is the continuity record for the new SPFF-R5 research implementation. It is not a modification to the frozen POC-3 protocol.

## 1. Repository and branch

- Repository: `https://github.com/Sashini-Jayakodi/stage0_pkg` (private; connected GitHub access required).
- Historical POC-3 branch: `poc3`; source at branch creation: `0d13f23782db58ef5a30f1b191bd3b02cbb2739b`.
- Active new research branch: **`spff-r5`**, branched from `poc3`. New changes belong only on `spff-r5`.
- POC-2: frozen; POC-3: stages 0–6, 8 and 10 completed, later exploratory r11–r13 forecasting work exists. POC-3 Stage 9 remains separate.
- Do not modify `main`, `poc3`, `poc2`, `POC_2`, POC-1/2 source, or frozen POC-3 files.
- The original plan and freeze rules are in `CLAUDE.md`, `docs/HANDOFF.md`, `docs/poc3-implementation-plan-r9.md`, `LOG.md`, and `configs/frozen_r9_btsc.yaml`. Read them before touching original project data.
- New code uses `src/spff_r5_*.py`; tests use `tests/test_spff_r5_*.py`; results are confined to `runs/spff_r5/` and new `docs/spff-r5-*.md`. Keep original histories and logs intact.
- No pull requests or GitHub discussion comments. Use concise technical commit subjects. Do not insert attribution statements about code-generation tools in source, docstrings, documentation, or commits. Preserve accurate experimental provenance and all negative results.

## 2. Research objective and gate

Target venue: IEEE Access. Mandatory Step 4 threshold: suitability >= 7/10 and **novelty >= 7/10**. A higher novelty score cannot be produced by renaming established methods; the contribution must enable a demonstrably new result, algorithmic capability, or inference method beyond prior work.

Historical Step 4 results:
- BC-SPFF Revision 2: suitability 8.5, novelty 6.5, feasibility 6.5, methodology 7.5. Conditional Monte Carlo is established; no runtime-matched and finite-pilot demonstration.
- IAPF Revision 3: suitability 8.0, novelty 5.5, feasibility 7.0, methodology 6.8. Established exact prefixes, confidence intervals, and information-theoretic bounds did not add up to a distinctive contribution.
- SPFF Revision 4: suitability 8.5, novelty 6.5, feasibility 7.5, methodology 7.5. A new selected-path excess-severity estimand and stability result improved specificity, but pilot-honest inference and a convincing comparison with general multinomial-functional inference were missing. Its path engine still had restricted acceptance patterns.
- SPFF-R5 is exploratory and has NOT earned a novelty >=7 assessment.

Central target: forecast the harm of false selective-risk certificates, not only their frequency, for fixed-sequence Learn-then-Test (LTT) selection on correlated frames grouped by independent physical signs.

Let a physical-sign signature at threshold j contain two indicators (A_j, W_j): sign accepted and accepted-and-wrong, with 0 <= W_j <= A_j <= 1. For population signature distribution q, a_j = E_q[A_j], w_j = E_q[W_j]; conditional committed error risk r_j = w_j/a_j if a_j > 0. Define threshold excess e_j(q) = max(w_j - alpha*a_j, 0), which includes acceptance mass. For a fixed threshold order and calibration sample of n independent signs, let J be the **final selected threshold**, or -1 on no certification. The selected-path excess-severity target is

  S(q) = sum_j P_q(J=j) * max(w_j - alpha*a_j, 0).

Keep the separate probability of selecting an unsafe threshold:

  F(q) = sum_j P_q(J=j) * 1{a_j > 0, w_j > alpha*a_j}.

These targets are distinct and should both be reported. The selected policy is determined by the actual LTT procedure, not a threshold-wise independent approximation. A threshold with zero accepted calibration signs is never certified.

The existing R5 dynamic program matches the sign-level exact binomial lower-tail test and the fixed-sequence walk. Distinguish sign-level inference from treating within-sign frames as independent. The path starts at a fixed or design-derived index; ensure any design-derived choices remain independent of pilot/confirmation outcomes as required.

## 3. Implemented new branch files (verified 2026-11-19)

- `src/spff_r5_path.py`: exact reference propagation of full joint threshold count states, final selected threshold distribution, S(q), F(q), and a preliminary conservative finite-pilot interval.
- `tests/test_spff_r5_path.py`: direct brute-force comparisons for n=2,4,6 plus input, acceptance, state guard, and pilot-interval checks.
- `src/spff_r5_benchmark.py`: reproducible synthetic finite-pilot experiment.
- `runs/spff_r5/finite_pilot_synthetic.json`: generated synthetic numbers.
- `docs/spff-r5-results.md`: results and limitations.

At the prior session's end, branch head was `0213749099e3e7e112d6967280733e87e1062871`, 5 commits ahead of `poc3`, no commits behind; verify before work. Six local focused tests passed; the entire repository suite was NOT rerun in that session. No real GTSRB or BelgiumTSC R5 experiments have been run. Do not imply otherwise.

The exact reference uses a state count guard (`max_states=250000` by default); it is **not** evidence of scalable 30-threshold inference. A state explosion is a reported limitation. No novel confidence bound has been established.

## 4. Preliminary finite-pilot synthetic evidence

Fixed 3-pattern signature distribution q=(0.5,0.2,0.3), 3 thresholds, 8 calibration signs, alpha=0.30, delta=0.30; 30 independent pilots at each size, 40 percentile bootstrap draws per pilot. Exact population S=0.001771875 (probability scale). The values are synthetic and do not represent traffic-sign data.

| Independent pilot signs | Point-estimate MAE, probability units | Conservative interval empirical coverage | Conservative mean interval width, probability units | Percentile bootstrap empirical coverage |
| ---: | ---: | ---: | ---: | ---: |
| 200 | 0.000686812 | 30/30 | 1.0000 | 29/30 |
| 1000 | 0.000223979 | 30/30 | 0.6623 | 25/30 |
| 3000 | 0.000070260 | 30/30 | 0.3831 | 28/30 |

Conservative intervals are much too wide to resolve the risk. All 30/30 covering does not establish useful precision; 30 replicates are insufficient for strong empirical coverage claims. A percentile bootstrap interval is not a finite-sample confidence guarantee. The finite-pilot radius currently uses coordinate-wise Hoeffding bounds plus a total-variation coupling argument; independently audit its coverage proof, assumptions, and constants.

A prior Revision 4 synthetic run used 80 signs and 30 thresholds, but only 10 all-commit signature patterns; it is not evidence that the R5 general acceptance engine handles realistic 30-threshold datasets. Do not combine the two experiments as if they had identical assumptions.

## 5. Reviewer-prioritized research problems

1. **Pilot-honest confidence intervals for S(q).** Derive and test a demonstrably tighter, statistically justified method; account for the pilot-estimated signature distribution AND the selection probabilities. A confidence interval must retain its stated guarantee under the precise assumptions, not merely cover a few simulated cases.
2. **Strong inference baselines.** Compare with general multinomial-functional confidence procedures (e.g. likelihood/profile or valid concentration-based approaches), full-path sign-level/bootstrap approaches, percentile bootstrap (descriptive), and the existing conservative interval. Compare coverage, width, resolution, MAE and measured runtime at the same resource budget.
3. **General variable-acceptance path engine.** Correctly handle changing numbers of accepted signs, nonzero wrong signs, fixed pre-registered threshold orders, zero-acceptance cases and nonmonotone error paths. Verify with brute force, then investigate mathematically justified state compression or bounded approximation with explicit numerical error.
4. **Real data.** Use already available GTSRB and BelgiumTSC data and frozen integrity records only in ways that do not imply an unearned new independent confirmation. All newly chosen R5 analyses on already inspected data are exploratory or post-hoc. Maintain sign-level splits and leakage protection. Real model/cache files may only exist in the local Windows workstation.
5. **Novelty audit.** Explicitly identify what an established generic method cannot do, or what theoretical property is new. Additional datasets, framework names and slight MC speedups alone do not justify novelty >=7.

## 6. Exact phase-by-phase implementation work order

### Phase A — Repository and mathematical audit: do this first and STOP

- Check `git status`; verify branch `spff-r5`; inspect exact repository files.
- Read `CLAUDE.md`, `docs/HANDOFF.md`, `docs/poc3-implementation-plan-r9.md`, `docs/spff-r5-results.md`, `src/spff_r5_path.py`, `src/spff_r5_benchmark.py`, `tests/test_spff_r5_path.py`, `src/r9_certify.py`, and forecasting r11/r12/r12b/r13 with results.
- Independently verify exact path calculation against brute-force enumeration for variable acceptance and nonzero errors. Check zero-accepted sign behavior; sign independence vs intra-sign dependence; mathematical definitions of S and F.
- Audit and prove or refute the current finite-pilot confidence bound, including whether the signature alphabet is fixed independently of pilot data and whether all possible patterns are represented.
- Identify scope for efficient path calculation and any errors, confounds, or incorrect inference claims.
- Run existing focused tests (and, if feasible, full tests); state what was and was not run. Create `docs/spff-r5-mathematical-audit.md` with findings, precise next proposal, and clearly ranked options.
- **Stop after Phase A; do not silently implement subsequent phases or alter original POC files.**

### Phase B — Develop and evaluate novel inference candidate(s), after review

- Formulate at least two meaningful candidate paths: a selection-functional-aware sensitivity/localization approach and a rigorous multinomial likelihood/profile or other valid inference approach.
- Derive assumptions, finite-sample or carefully qualified asymptotic coverage, and how selection-path information is exploited. Compare analytically with generic inference.
- Reject unsupported novelty labels or invalid confidence claims.

### Phase C — Efficient general path evaluation, after review

- Keep a brute-force exact reference for small cases. Implement independently verified compression or truncation with rigorous remainder bounds for medium cases.
- Benchmark thresholds 3/5/10/30, varying acceptance/error-pattern support and sample sizes. Measure runtime, peak memory and output error against the reference or independent oracle.
- Report exact tractability limits without silently substituting approximation.

### Phase D — Finite-pilot and real-data experiments, after review

- Predefine synthetic generating distributions and varied risk-boundary distances, pilot n in 100/261/500/1000/3000 if feasible, calibration sizes and threshold grids.
- Compare confidence coverage with binomial uncertainty intervals, interval widths, resolution/abstention rates, forecast MAE, runtime-matched efficiency and worst-case behavior; no cherry-picked exclusions.
- Run eligible exploratory GTSRB and BelgiumTSC studies through existing hashed data loaders where local files exist; keep prior frozen experiments unchanged.

### Phase E — Quality and reproducibility

- Tests for each theorem-relevant edge case and numerical implementation. Save seeds, parameters, environment and complete output JSON.
- State failures, nonconvergence and limits as findings. Do not claim full empirical validation without actually running experiments.

### Phase F — Independent novelty review

- Re-evaluate technical novelty against clustered LTT, bootstrap, conditional MC, selective risk, multinomial-functional inference, and related safety work; verify literature externally.
- Apply strict IEEE Access Step 4 scores: suitability, novelty, feasibility, methodology. PASS only if suitability >=7 and novelty >=7. A score is evidence-driven, not guaranteed.

## 7. Local VS Code usage

Before running anything, in the existing local repo terminal:

```powershell
git status
git fetch origin
git switch spff-r5
```

If the local branch is absent, use `git switch --track origin/spff-r5`. If `git status` reports uncommitted changes, stop to preserve them rather than using `reset --hard` or discarding work. Verify `git branch --show-current`, then `git pull --ff-only`.

Local Python environment recorded in `docs/HANDOFF.md`: Windows, `py -3.12`, RTX 5050; use `py -3.12 -m pytest tests/test_spff_r5_path.py -q` for focused tests. The complete test suite is larger and may require local datasets/checkpoints.

## 8. New chat starting point

Start by opening the `spff-r5` branch and this exact file. Reinspect source, do not rely on a prior conversation summary as a substitute. Begin Phase A only and deliver a precise audit (including any weaknesses with the claimed confidence proof). Wait for an explicit request before starting Phase B.
