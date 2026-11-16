=== r12b: POST HOC DIAGNOSTIC (GTSRB development data; written after the r12 result; no criteria; not a revised method) ===
Pool 1,046 tracks (frames per track: {29: 1, 30: 1045}); model m = 30; Cert 784; delta 0.1. Observed rates and pilot forecasts from r12_summary.json (SHA-256 checked). Monte Carlo (item 3) key [20261119, 13], 20,000 Cert sets per cell.
POC-2 pool checks: True

--- alpha 0.5%: jt = 12 (threshold 370.55) ---
  Pool at jt: risk  0.531%, f   76.8%, per-frame accepted-wrong  0.408%, rho_I 0.2174, rho among accepted frames 0.7491, N_g 23
  1. Forecasts and observed rates:
      test |          observed [95% CP] | pilot (r12) | oracle fixed N_g | oracle real acc. rho_I | oracle real acc. rho_acc
        hb |  17.00% [15.38, 18.72]% |       6.61% |           16.83% |                 15.29% |                   30.67%
     binom |  26.55% [24.63, 28.54]% |      13.33% |           25.49% |                 23.92% |                   36.14%
     (oracle fixed N_g: k_max hb 70, binom 77; clipped mass per track 4.15e-05)
  2. Fit at jt: per-track accepted-and-wrong count k against clipped BB(30, 0.408%, rho_I 0.2174) at N_g 23 (largest observed k: 16)
                                                                k=0: observed  1007, expected   1010.28
                                                               k>=1: observed    39, expected     35.72
                                                           k>=N_g/2: observed     2, expected      1.80
                                                              k=N_g: observed     0, expected      0.07
     k>N_g (observed only; the clipped model puts this mass at N_g): observed     0, expected      0.00
     chi-square 5.36, df 2, p 0.0684; bins (lower edges, last open) [0, 1, 2, 4, 7]; obs [1007, 20, 6, 8, 5]; exp [1010.3, 13.7, 10.3, 6.1, 5.6]
  3(i). Errors given the real acceptance: k against BB(A_g, pool risk, rho), summed over tracks
     rho_I (all-frame ICC, as in the model) = 0.2174:
                                                                  k=0: observed  1007, expected   1006.49
                                                                 k>=1: observed    39, expected     39.51
                                                             k>=N_g/2: observed     2, expected      1.45
                                                                k=N_g: observed     0, expected      0.02
       k>N_g (observed only; the clipped model puts this mass at N_g): observed     0, expected      0.03
       chi-square 3.93, df 2, p 0.14; bins [0, 1, 2, 4, 7]; obs [1007, 20, 6, 8, 5]; exp [1006.5, 16.3, 11.6, 6.5, 5.1]
     rho among accepted frames = 0.7491:
                                                                  k=0: observed  1007, expected   1034.95
                                                                 k>=1: observed    39, expected     11.05
                                                             k>=N_g/2: observed     2, expected      4.70
                                                                k=N_g: observed     0, expected      0.23
       k>N_g (observed only; the clipped model puts this mass at N_g): observed     0, expected      2.06
       chi-square 147.21, df 0, p n/a (df <= 0); bins [0, 1, 10]; obs [1007, 35, 4]; exp [1035.0, 5.8, 5.2]
  3(ii). Acceptance at jt: A_g mean 23.04, sd 8.89, quantiles 10/25/50/75/90 8/19/27/30/30 against the fixed N_g 23; A_g = 0 in    2.9%, < N_g/2 in   13.7%, < N_g in   32.8%, = 30 in   36.3%; wrong frames in tracks with A_g < N_g: 121 of 128

--- alpha 1.0%: jt = 1 (threshold 283.50) ---
  Pool at jt: risk  1.030%, f   82.0%, per-frame accepted-wrong  0.845%, rho_I 0.2318, rho among accepted frames 0.6369, N_g 25
  1. Forecasts and observed rates:
      test |          observed [95% CP] | pilot (r12) | oracle fixed N_g | oracle real acc. rho_I | oracle real acc. rho_acc
        hb |  20.00% [18.27, 21.82]% |      16.94% |           21.80% |                 18.34% |                   30.46%
     binom |  29.30% [27.31, 31.35]% |      25.48% |           30.71% |                 27.72% |                   36.34%
     (oracle fixed N_g: k_max hb 167, binom 177; clipped mass per track 4.60e-05)
  2. Fit at jt: per-track accepted-and-wrong count k against clipped BB(30, 0.845%, rho_I 0.2318) at N_g 25 (largest observed k: 18)
                                                                k=0: observed   962, expected    976.53
                                                               k>=1: observed    84, expected     69.47
                                                           k>=N_g/2: observed     4, expected      3.38
                                                              k=N_g: observed     0, expected      0.08
     k>N_g (observed only; the clipped model puts this mass at N_g): observed     0, expected      0.00
     chi-square 23.43, df 5, p 0.000279; bins (lower edges, last open) [0, 1, 2, 3, 4, 5, 7, 11]; obs [962, 46, 13, 5, 1, 5, 10, 4]; exp [976.5, 25.4, 12.1, 7.6, 5.3, 6.9, 7.0, 5.2]
  3(i). Errors given the real acceptance: k against BB(A_g, pool risk, rho), summed over tracks
     rho_I (all-frame ICC, as in the model) = 0.2318:
                                                                  k=0: observed   962, expected    970.57
                                                                 k>=1: observed    84, expected     75.43
                                                             k>=N_g/2: observed     4, expected      2.82
                                                                k=N_g: observed     0, expected      0.02
       k>N_g (observed only; the clipped model puts this mass at N_g): observed     0, expected      0.03
       chi-square 17.51, df 5, p 0.00362; bins [0, 1, 2, 3, 4, 5, 7, 10]; obs [962, 46, 13, 5, 1, 5, 9, 5]; exp [970.6, 29.2, 13.6, 8.3, 5.7, 7.2, 5.7, 5.7]
     rho among accepted frames = 0.6369:
                                                                  k=0: observed   962, expected   1017.40
                                                                 k>=1: observed    84, expected     28.60
                                                             k>=N_g/2: observed     4, expected      8.70
                                                                k=N_g: observed     0, expected      0.43
       k>N_g (observed only; the clipped model puts this mass at N_g): observed     0, expected      2.32
       chi-square 309.04, df 3, p 1.1e-66; bins [0, 1, 2, 5, 10, 19]; obs [962, 46, 19, 14, 5, 0]; exp [1017.4, 6.1, 6.8, 5.0, 5.3, 5.4]
  3(ii). Acceptance at jt: A_g mean 24.60, sd 7.78, quantiles 10/25/50/75/90 13/22/28/30/30 against the fixed N_g 25; A_g = 0 in    1.4%, < N_g/2 in    9.8%, < N_g in   31.8%, = 30 in   41.9%; wrong frames in tracks with A_g < N_g: 245 of 265

Notes: chi-square bins: adjacent bins merged (top value down) until every expected count >= 5; df = bins - 1 - 2 (rate and rho taken from the same pool tracks; rho by ANOVA, not maximum likelihood, so the df is approximate). The model gives every track 30 frames; the pool has one 29-frame track. Item 3 Monte Carlo uses HB over 784 x 30 frames, as r11. This is a POST HOC DIAGNOSTIC: no criterion, no revised method.
