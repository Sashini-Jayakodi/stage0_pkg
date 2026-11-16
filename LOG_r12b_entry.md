## r12b: POST HOC DIAGNOSTIC of the r12 result (GTSRB development data)

### Definition (no Part A)
Written after the r12 result (Part B, 18d1f29) was seen, on the author's instruction; labelled POST HOC DIAGNOSTIC: no
criteria, not a test, not a revised method. At the true first unsafe threshold jt of the POC-2 B1 order, on the 1,046
pool tracks, alpha 0.5% and 1%: (1) oracle forecast, r11 (b) 4-6 with pool risk, pool f and pool rho_I at jt;
(2) fit of per-track accepted-and-wrong counts against the clipped beta-binomial(30, pool rate, pool rho_I), with
category counts and a chi-square p-value; (3) split at jt (the author's choice among the readings offered): errors
given the real acceptance (fit against BB(A_g, pool risk, rho), and a Monte Carlo oracle forecast with real accepted
counts, key [20261119, 13], 20,000 Cert sets per cell), and acceptance A_g against the fixed N_g. Two rho readings are
shown in (3): rho_I (all-frame ICC, as in the model) and the ICC among accepted frames. Code: `src/r12b_diagnostic.py`;
tests: `tests/test_r12b_diagnostic.py`.

### Part B: results (filled after review)

#### Runs
- Run 1, 16 Nov 2026, on commit daa19b0 (`src/r12b_diagnostic.py` SHA-256
  74c4be1f0f35bb5a770ebddc85dc387b4bbece71f70988efecd9823ecac37cdc), 9.5 s, exit code 0. Outputs: `r12b_summary.json`
  SHA-256 374935c37aacf7f45062b5b2095412b59d72de062adeb48a83d622ee4a79fcfd, `r12b_report.md` SHA-256
  b0d5e70a4ac32faf7bc67954a1c3ee56abc7b8105fe26b418acad5e41abc4e1c. Not committed; replaced by run 2.
- Run 2 (the one rerun the author allowed), 16 Nov 2026, after the fix below (`src/r12b_diagnostic.py` SHA-256
  8a93fda029ef3de94678086ca68d546590db481eafd488cd60cb81230a01e44d), 11.5 s, exit code 0. Outputs (committed):
  `r12b_summary.json` SHA-256 268464cd4bb5d8b8426f6509c4174b2df7dc962f05e4608415da0fc776ee0f61, `r12b_report.md`
  SHA-256 485c0ab7f4bfe309865f0e6fbaabe4172b819464ea0e48c609556d77de355414.
- Integrity: `r12_summary.json` and `stage3b_summary.json` SHA-256 checked; POC-2 pool checks True; jt equals the r12
  value at both alphas.

#### Defect and fix
- Defect (run 1): the chi-square merged bins only at the ends (from the top, and the bottom if needed), so middle bins
  with expected counts below 5 were kept (for example 3.9, 2.7, 2.0, 1.5 in the first fit at alpha 0.5%), while the
  printout's Notes line stated that every expected count was >= 5. The run-1 chi-square statistics, degrees of freedom
  and p-values were therefore not computed under the stated rule and are not used.
- Fix: adjacent bins are merged anywhere: walking from the top value down, bins are added into a group until its
  expected count reaches 5, then a new group starts; a bottom leftover below 5 joins the group above it. Tests added:
  small expected counts in the middle (hand case), a bottom leftover case, and a random check that every group reaches
  5 and the totals are kept. 10 of 10 r12b tests pass.
- Check: every number outside the chi-square is identical in runs 1 and 2 (summary JSON compared with the chi-square
  blocks, code hashes and run time removed: no difference; report lines other than the chi-square, Notes and Output
  lines: identical).
- Earlier, before run 1: while the tests were first written, a test caught a merge bug in which a combined
  `a[-2] += a.pop()` statement added each merged bin into the wrong bin; it was fixed before run 1 and no real-data
  number was produced by it.

#### Chi-square lines of run 2, verbatim
```
alpha 0.5%, (2) clipped BB:          chi-square 5.36, df 2, p 0.0684; bins (lower edges, last open) [0, 1, 2, 4, 7]; obs [1007, 20, 6, 8, 5]; exp [1010.3, 13.7, 10.3, 6.1, 5.6]
alpha 0.5%, (3i) rho_I:              chi-square 3.93, df 2, p 0.14; bins [0, 1, 2, 4, 7]; obs [1007, 20, 6, 8, 5]; exp [1006.5, 16.3, 11.6, 6.5, 5.1]
alpha 0.5%, (3i) rho among accepted: chi-square 147.21, df 0, p n/a (df <= 0); bins [0, 1, 10]; obs [1007, 35, 4]; exp [1035.0, 5.8, 5.2]
alpha 1.0%, (2) clipped BB:          chi-square 23.43, df 5, p 0.000279; bins (lower edges, last open) [0, 1, 2, 3, 4, 5, 7, 11]; obs [962, 46, 13, 5, 1, 5, 10, 4]; exp [976.5, 25.4, 12.1, 7.6, 5.3, 6.9, 7.0, 5.2]
alpha 1.0%, (3i) rho_I:              chi-square 17.51, df 5, p 0.00362; bins [0, 1, 2, 3, 4, 5, 7, 10]; obs [962, 46, 13, 5, 1, 5, 9, 5]; exp [970.6, 29.2, 13.6, 8.3, 5.7, 7.2, 5.7, 5.7]
alpha 1.0%, (3i) rho among accepted: chi-square 309.04, df 3, p 1.1e-66; bins [0, 1, 2, 5, 10, 19]; obs [962, 46, 19, 14, 5, 0]; exp [1017.4, 6.1, 6.8, 5.0, 5.3, 5.4]
```
The labels before each colon are added here to say which fit the line belongs to; the text after the colon is
verbatim. At alpha 0.5%, rho among accepted frames, the merge leaves 3 groups, so df = 3 - 1 - 2 = 0 and no p-value is
given.

#### Forecasts and observed rates (unchanged between runs; full printout in `r12b_report.md`)
```
--- alpha 0.5%: jt = 12 (threshold 370.55) ---
        hb |  17.00% [15.38, 18.72]% |       6.61% |           16.83% |                 15.29% |                   30.67%
     binom |  26.55% [24.63, 28.54]% |      13.33% |           25.49% |                 23.92% |                   36.14%
--- alpha 1.0%: jt = 1 (threshold 283.50) ---
        hb |  20.00% [18.27, 21.82]% |      16.94% |           21.80% |                 18.34% |                   30.46%
     binom |  29.30% [27.31, 31.35]% |      25.48% |           30.71% |                 27.72% |                   36.34%
```
Columns: test | observed [95% CP] | pilot (r12) | oracle fixed N_g | oracle real acceptance, rho_I | oracle real
acceptance, rho among accepted frames.
