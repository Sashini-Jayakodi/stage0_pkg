# EXPLORATORY (post-hoc, preliminary; GTSRB tracks all seen before)

Sign-level certificate of frame-weighted (micro) selective risk, GTSRB, delta = 0.1.

## Part 1: validity and power (Stage 3b design: 2,000 draws of 784 of the 1,046 pool tracks, seed 20261115)

Violation = a threshold is certified and its pool micro risk > alpha. Coverage = pool frame coverage at the certified threshold, mean over certifying draws. Stored rows are copied from stage3b_summary.json.

| alpha | test | viol | violation rate | 95% CP | certify share | mean frame cov. | mean true risk (cert.) |
|---|---|---|---|---|---|---|---|
| 0.5% | (i) Hoeffding | 0 | 0.00% | [0.00%, 0.18%] | 0.0% | - | - |
| 0.5% | (ii) Emp. Bernstein | 0 | 0.00% | [0.00%, 0.18%] | 0.0% | - | - |
| 0.5% | (iii) WSR betting | 0 | 0.00% | [0.00%, 0.18%] | 0.0% | - | - |
| 0.5% | (iii-h) WSR hedged half | 0 | 0.00% | [0.00%, 0.18%] | 0.0% | - | - |
| 0.5% | B1 frame-level HB (recomputed) | 340 | 17.00% | [15.38%, 18.72%] | 60.0% | 74.7% | 0.473% |
| 0.5% | stored B1 frame-level (Stage 3b) | 340 | 17.00% | [15.38%, 18.72%] | 60.0% | - | 0.482% |
| 0.5% | stored track-level (Stage 3b) | 38 | 1.90% | [1.35%, 2.60%] | 4.7% | - | 0.513% |
| 1.0% | (i) Hoeffding | 0 | 0.00% | [0.00%, 0.18%] | 0.0% | - | - |
| 1.0% | (ii) Emp. Bernstein | 0 | 0.00% | [0.00%, 0.18%] | 0.0% | - | - |
| 1.0% | (iii) WSR betting | 0 | 0.00% | [0.00%, 0.18%] | 5.8% | 73.6% | 0.434% |
| 1.0% | (iii-h) WSR hedged half | 0 | 0.00% | [0.00%, 0.18%] | 0.0% | - | - |
| 1.0% | B1 frame-level HB (recomputed) | 400 | 20.00% | [18.27%, 21.82%] | 99.9% | 80.3% | 0.836% |
| 1.0% | stored B1 frame-level (Stage 3b) | 400 | 20.00% | [18.27%, 21.82%] | 99.9% | - | 0.871% |
| 1.0% | stored track-level (Stage 3b) | 28 | 1.40% | [0.93%, 2.02%] | 19.8% | - | 0.630% |
| 2.0% | (i) Hoeffding | 0 | 0.00% | [0.00%, 0.18%] | 0.0% | - | - |
| 2.0% | (ii) Emp. Bernstein | 0 | 0.00% | [0.00%, 0.18%] | 65.1% | 76.7% | 0.579% |
| 2.0% | (iii) WSR betting | 0 | 0.00% | [0.00%, 0.18%] | 100.0% | 82.1% | 1.038% |
| 2.0% | (iii-h) WSR hedged half | 0 | 0.00% | [0.00%, 0.18%] | 99.9% | 81.4% | 0.955% |
| 2.0% | B1 frame-level HB (recomputed) | 0 | 0.00% | [0.00%, 0.18%] | 100.0% | 82.4% | 1.072% |
| 2.0% | stored B1 frame-level (Stage 3b) | 0 | 0.00% | [0.00%, 0.18%] | 100.0% | - | 1.235% |
| 2.0% | stored track-level (Stage 3b) | 0 | 0.00% | [0.00%, 0.18%] | 95.7% | - | 1.037% |
| 3.0% | (i) Hoeffding | 0 | 0.00% | [0.00%, 0.18%] | 0.0% | - | - |
| 3.0% | (ii) Emp. Bernstein | 0 | 0.00% | [0.00%, 0.18%] | 100.0% | 82.2% | 1.051% |
| 3.0% | (iii) WSR betting | 0 | 0.00% | [0.00%, 0.18%] | 100.0% | 82.4% | 1.072% |
| 3.0% | (iii-h) WSR hedged half | 0 | 0.00% | [0.00%, 0.18%] | 100.0% | 82.4% | 1.072% |
| 3.0% | B1 frame-level HB (recomputed) | 0 | 0.00% | [0.00%, 0.18%] | 100.0% | 82.4% | 1.072% |
| 3.0% | stored B1 frame-level (Stage 3b) | 0 | 0.00% | [0.00%, 0.18%] | 100.0% | - | 1.412% |
| 3.0% | stored track-level (Stage 3b) | 0 | 0.00% | [0.00%, 0.18%] | 100.0% | - | 1.546% |
| 5.0% | (i) Hoeffding | 0 | 0.00% | [0.00%, 0.18%] | 0.0% | - | - |
| 5.0% | (ii) Emp. Bernstein | 0 | 0.00% | [0.00%, 0.18%] | 100.0% | 82.4% | 1.072% |
| 5.0% | (iii) WSR betting | 0 | 0.00% | [0.00%, 0.18%] | 100.0% | 82.4% | 1.072% |
| 5.0% | (iii-h) WSR hedged half | 0 | 0.00% | [0.00%, 0.18%] | 100.0% | 82.4% | 1.072% |
| 5.0% | B1 frame-level HB (recomputed) | 0 | 0.00% | [0.00%, 0.18%] | 100.0% | 82.4% | 1.072% |
| 5.0% | stored B1 frame-level (Stage 3b) | 0 | 0.00% | [0.00%, 0.18%] | 100.0% | - | 1.764% |
| 5.0% | stored track-level (Stage 3b) | 0 | 0.00% | [0.00%, 0.18%] | 100.0% | - | 1.816% |

(iii) = theta 1 (one-sided). (iii-h) = sensitivity: the downward half of the theta 1/2 hedged process (L = ln(2/delta), threshold 2/delta). Track-level stored row: its unit is the track commit rule, so no frame coverage is given. Last column: pool micro risk (W/A) for the rows computed here; for the stored rows it is the stored mean loss L, a different scale (violations and certify shares reproduce exactly).

## Part 2: the five POC-2 Stage 4 Cert/Test re-splits (regenerated, checked against Stage 4)

Cert 784 tracks, Test 262. WSR visits Cert tracks in a seeded random order (salt 20261116, seed). Not certified: coverage 0, risk '-'.

| alpha | test | seed | certified pos. | threshold | Cert micro risk | Test micro risk | Test frame cov. |
|---|---|---|---|---|---|---|---|
| 1% | (ii) Emp. Bernstein | 0 | - | - | - | - | 0.0% |
| 1% | (ii) Emp. Bernstein | 1 | - | - | - | - | 0.0% |
| 1% | (ii) Emp. Bernstein | 2 | - | - | - | - | 0.0% |
| 1% | (ii) Emp. Bernstein | 3 | - | - | - | - | 0.0% |
| 1% | (ii) Emp. Bernstein | 4 | - | - | - | - | 0.0% |
| 1% | (iii) WSR betting | 0 | - | - | - | - | 0.0% |
| 1% | (iii) WSR betting | 1 | - | - | - | - | 0.0% |
| 1% | (iii) WSR betting | 2 | - | - | - | - | 0.0% |
| 1% | (iii) WSR betting | 3 | - | - | - | - | 0.0% |
| 1% | (iii) WSR betting | 4 | - | - | - | - | 0.0% |
| 2% | (ii) Emp. Bernstein | 0 | 12 | 370.5 | 0.454% | 0.761% | 76.9% |
| 2% | (ii) Emp. Bernstein | 1 | 15 | 398.6 | 0.469% | 0.404% | 75.6% |
| 2% | (ii) Emp. Bernstein | 2 | 14 | 389.0 | 0.457% | 0.524% | 75.3% |
| 2% | (ii) Emp. Bernstein | 3 | 19 | 439.4 | 0.377% | 0.304% | 71.0% |
| 2% | (ii) Emp. Bernstein | 4 | 19 | 439.4 | 0.359% | 0.358% | 71.1% |
| 2% | (iii) WSR betting | 0 | 0 | 276.7 | 1.058% | 1.113% | 82.3% |
| 2% | (iii) WSR betting | 1 | 0 | 276.7 | 1.171% | 0.779% | 83.3% |
| 2% | (iii) WSR betting | 2 | 0 | 276.7 | 1.105% | 0.971% | 82.5% |
| 2% | (iii) WSR betting | 3 | 0 | 276.7 | 1.052% | 1.129% | 83.4% |
| 2% | (iii) WSR betting | 4 | 0 | 276.7 | 1.070% | 1.078% | 82.6% |
| 3% | (ii) Emp. Bernstein | 0 | 0 | 276.7 | 1.058% | 1.113% | 82.3% |
| 3% | (ii) Emp. Bernstein | 1 | 0 | 276.7 | 1.171% | 0.779% | 83.3% |
| 3% | (ii) Emp. Bernstein | 2 | 0 | 276.7 | 1.105% | 0.971% | 82.5% |
| 3% | (ii) Emp. Bernstein | 3 | 0 | 276.7 | 1.052% | 1.129% | 83.4% |
| 3% | (ii) Emp. Bernstein | 4 | 0 | 276.7 | 1.070% | 1.078% | 82.6% |
| 3% | (iii) WSR betting | 0 | 0 | 276.7 | 1.058% | 1.113% | 82.3% |
| 3% | (iii) WSR betting | 1 | 0 | 276.7 | 1.171% | 0.779% | 83.3% |
| 3% | (iii) WSR betting | 2 | 0 | 276.7 | 1.105% | 0.971% | 82.5% |
| 3% | (iii) WSR betting | 3 | 0 | 276.7 | 1.052% | 1.129% | 83.4% |
| 3% | (iii) WSR betting | 4 | 0 | 276.7 | 1.070% | 1.078% | 82.6% |
| 5% | (ii) Emp. Bernstein | 0 | 0 | 276.7 | 1.058% | 1.113% | 82.3% |
| 5% | (ii) Emp. Bernstein | 1 | 0 | 276.7 | 1.171% | 0.779% | 83.3% |
| 5% | (ii) Emp. Bernstein | 2 | 0 | 276.7 | 1.105% | 0.971% | 82.5% |
| 5% | (ii) Emp. Bernstein | 3 | 0 | 276.7 | 1.052% | 1.129% | 83.4% |
| 5% | (ii) Emp. Bernstein | 4 | 0 | 276.7 | 1.070% | 1.078% | 82.6% |
| 5% | (iii) WSR betting | 0 | 0 | 276.7 | 1.058% | 1.113% | 82.3% |
| 5% | (iii) WSR betting | 1 | 0 | 276.7 | 1.171% | 0.779% | 83.3% |
| 5% | (iii) WSR betting | 2 | 0 | 276.7 | 1.105% | 0.971% | 82.5% |
| 5% | (iii) WSR betting | 3 | 0 | 276.7 | 1.052% | 1.129% | 83.4% |
| 5% | (iii) WSR betting | 4 | 0 | 276.7 | 1.070% | 1.078% | 82.6% |

Summary over the five re-splits, median (min-max):

| alpha | test | certified | position | Test micro risk | Test frame coverage |
|---|---|---|---|---|---|
| 1% | (ii) Emp. Bernstein | 0/5 | - | - | 0.0% (0.0-0.0) |
| 1% | (iii) WSR betting | 0/5 | - | - | 0.0% (0.0-0.0) |
| 2% | (ii) Emp. Bernstein | 5/5 | 15 (12-19) | 0.404% (0.304-0.761) | 75.3% (71.0-76.9) |
| 2% | (iii) WSR betting | 5/5 | 0 (0-0) | 1.078% (0.779-1.129) | 82.6% (82.3-83.4) |
| 3% | (ii) Emp. Bernstein | 5/5 | 0 (0-0) | 1.078% (0.779-1.129) | 82.6% (82.3-83.4) |
| 3% | (iii) WSR betting | 5/5 | 0 (0-0) | 1.078% (0.779-1.129) | 82.6% (82.3-83.4) |
| 5% | (ii) Emp. Bernstein | 5/5 | 0 (0-0) | 1.078% (0.779-1.129) | 82.6% (82.3-83.4) |
| 5% | (iii) WSR betting | 5/5 | 0 (0-0) | 1.078% (0.779-1.129) | 82.6% (82.3-83.4) |

## Part 3: forecast of the B1 violation rate (preliminary)

j* = first threshold in the frozen order with pool micro risk > alpha. Design quantities from the 261 design tracks. 20,000 simulated Cert sets of 784 tracks x 30 frames; accepted frames fixed at round(30 x design acceptance); accepted-wrong counts beta-binomial(30, design rate, rho_I), clipped at N_g; forecast = P(Stage 3b HB frame-level p-value at j* <= delta). Heuristic = Phi(Phi^-1(0.1)/sqrt(13.9)). Seed 20261117.

| alpha | j* | threshold | pool micro risk at j* | design accept. | design acc.-wrong / frame | design micro risk | rho_I | N_g | sim micro risk | clipped | forecast | observed 3b B1 | heuristic |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.5% | 12 | 370.5 | 0.531% | 77.0% | 0.383% | 0.498% | 0.124 | 23 | 0.500% | 16 | 16.60% | 17.00% | 36.55% |
| 1.0% | 1 | 283.5 | 1.030% | 82.9% | 0.728% | 0.879% | 0.199 | 25 | 0.874% | 258 | 47.30% | 20.00% | 36.55% |

## Sanity: per-threshold pool micro risk (1,046 pool tracks) and design values

| pos. | threshold | pool accepted | pool acc.-wrong | pool micro risk | pool frame cov. | design micro risk | design frame cov. |
|---|---|---|---|---|---|---|---|
| 19 | 439.38 | 22304 | 80 | 0.359% | 71.1% | 0.253% | 70.7% |
| 18 | 428.82 | 22616 | 88 | 0.389% | 72.1% | 0.303% | 71.8% |
| 17 | 418.51 | 22885 | 95 | 0.415% | 72.9% | 0.316% | 72.6% |
| 16 | 408.44 | 23148 | 100 | 0.432% | 73.8% | 0.313% | 73.5% |
| 15 | 398.62 | 23415 | 106 | 0.453% | 74.6% | 0.360% | 74.4% |
| 14 | 389.03 | 23650 | 112 | 0.474% | 75.4% | 0.373% | 75.3% |
| 13 | 379.68 | 23862 | 118 | 0.495% | 76.0% | 0.469% | 76.3% |
| 12 | 370.55 | 24104 | 128 | 0.531% | 76.8% | 0.498% | 77.0% |
| 11 | 361.63 | 24300 | 142 | 0.584% | 77.4% | 0.543% | 77.7% |
| 10 | 352.94 | 24483 | 151 | 0.617% | 78.0% | 0.554% | 78.4% |
| 9 | 344.45 | 24637 | 162 | 0.658% | 78.5% | 0.582% | 79.0% |
| 8 | 336.17 | 24779 | 174 | 0.702% | 79.0% | 0.627% | 79.5% |
| 7 | 328.08 | 24931 | 185 | 0.742% | 79.5% | 0.623% | 80.0% |
| 6 | 320.19 | 25086 | 195 | 0.777% | 79.9% | 0.650% | 80.6% |
| 5 | 312.49 | 25218 | 209 | 0.829% | 80.4% | 0.725% | 81.1% |
| 4 | 304.98 | 25358 | 219 | 0.864% | 80.8% | 0.751% | 81.6% |
| 3 | 297.64 | 25489 | 235 | 0.922% | 81.2% | 0.840% | 82.1% |
| 2 | 290.48 | 25616 | 247 | 0.964% | 81.6% | 0.868% | 82.4% |
| 1 | 283.50 | 25735 | 265 | 1.030% | 82.0% | 0.879% | 82.9% |
| 0 | 276.68 | 25846 | 277 | 1.072% | 82.4% | 0.921% | 83.2% |

## Sanity: first 10 pool tracks, Z_g at j* = 1 for alpha = 1%

| pool idx | track | frames | N_g | K_g | Z_g |
|---|---|---|---|---|---|
| 0 | 00_00001 | 30 | 18 | 0 | -0.006000 |
| 1 | 00_00002 | 30 | 2 | 2 | +0.066000 |
| 2 | 00_00003 | 30 | 8 | 1 | +0.030667 |
| 3 | 00_00004 | 30 | 0 | 0 | +0.000000 |
| 4 | 00_00006 | 30 | 1 | 1 | +0.033000 |
| 5 | 01_00000 | 30 | 30 | 0 | -0.010000 |
| 6 | 01_00001 | 30 | 15 | 0 | -0.005000 |
| 7 | 01_00002 | 30 | 30 | 0 | -0.010000 |
| 8 | 01_00003 | 30 | 25 | 0 | -0.008333 |
| 9 | 01_00004 | 30 | 25 | 0 | -0.008333 |

## Checks

- PASS frozen_grid_matches_stage1_grid
- PASS gate_flags_reproduce
- PASS pool_commit_table_reproduces
- PASS b1_top_equals_stored_top1
- PASS design_tracks_261
- PASS design_frames_match_b1_design
- PASS design_work_e_reproduces
- PASS design_work_c_reproduces
- PASS draws_equal_stage3b_weights
- PASS stage3b_rows_reproduce
- PASS b1_true_risk_equals_micro_risk_side
- PASS wsr_eb_match_scalar_reference
- PASS Z_within_range_width_1
- PASS stage4_splits_reproduce

ALL CHECKS PASS: True

Input SHA-256:

- checkpoint: 86527359bc61a5fbc4d48b7744e049b5ca3ff81cb016fb237298d0c007140bcc
- train_release.npz: 228fe5b2ee29f59d5e3fac7e5b81ffd8b1a9b1f6b5747756f88711345034d380
- head.npz: 5e85b1b0ca9ee45b86bcf20fcf4fce454128f8b292d454b84da42fe12ff0a5c2
- seed0.npz: 397611c5538cc9ebe7a0be768ada29cb0afa7c89bdc53511df3a2403001fea49
- track_table_pool.npz: 0afb7e89052c393d695a8cb7e5796e4e811b1ff9c3c6bd392bf3cd10742250db
- frame_evidence.npz: f79033e8cee5ba73a6b22d1d33b91227ff7cf628d95dc616a28824ad68a0e960
- scores_train_release.npz: 0be15ed0b850e24ddc69a7395e954372f127ae6a2387369d0e812f082f3ceaac
- b1_design.json: 8fb7e1c534ce1d923595e18794327afa70d654f033d4b55196a4dc2a169d22c9
- stage3b_summary.json: 7591a438b569e7231f3a40d8da6420a81b6f65791e707df4c0101338a0bf7367
- stage4_summary.json: e66e45f210ebae5e6d21a4d8247584e1affc2d9691a760427814c821b164d2cc
- stage1_summary.json: bfe2c26973727c5cc84cce61f150fbe8cbfb919cadb12ea6a5cc93a52c423882
- head_summary.json: eecfded822569f64a185fcfcefda5dd2ff6344e043fd0e136f655dd76f570e89
- frozen_r8.yaml: 4f7a79156a4486b6b5f97515457513e6df2a2bfeed2a41b6fa05936bd1141d24
- r8_stage4.py: fae67390a488b44c57f1386c9b256cb0fbbcf2444282f5724c2580aa3e6445c2
