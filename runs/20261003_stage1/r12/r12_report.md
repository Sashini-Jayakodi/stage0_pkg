=== r12: r11 forecast on the GTSRB frame-level audit (EXPLORATORY (GTSRB development data); LOG_r12_entry.md Part A, commit 6491dc8) ===
delta 0.1; pilot 261 design tracks x 30 frames; audit = POC-2 Stage 3b draws (2,000 x 784 of 1,046 pool tracks, seed 20261115); bootstrap 1000 resamples, key [20261119, 12]; heuristic DE 13.9.
Checks (Part A (h)): frozen_grid_matches_stage1_grid PASS; gate_flags_reproduce PASS; pool_commit_table_reproduces PASS; b1_top_equals_stored_top1 PASS; design_tracks_261 PASS; design_frames_30_each PASS; design_work_c_reproduces PASS; design_work_e_reproduces PASS; start_19_every_alpha_and_test PASS; stage3b_b1_hb_rows_reproduce PASS

  alpha  test | observed               CP95   cert | forecast       boot p10-p90 boot med |   heur. |    gap | j*    U_j*      f   rho_I        case N_g k_max     clip | jt j*<=jt pool R(j*)
   0.5%    hb |  17.00% [15.38, 18.72]%  60.0% |   6.61% [  1.64,  16.29]%   9.05% |  36.55% | -10.39 | 13  0.559%  76.3%  0.1156   betabinom   23    70 6.24e-07 | 12    yes    0.495%
   0.5% binom |  26.55% [24.63, 28.54]%  73.1% |  13.33% [  4.75,  27.22]%  17.77% |  36.55% | -13.22 | 13  0.559%  76.3%  0.1156   betabinom   23    77 6.24e-07 | 12    yes    0.495%
   1.0%    hb |  20.00% [18.27, 21.82]%  99.9% |  16.94% [  0.00,  19.44]%   5.59% |  36.55% |  -3.06 |  0  1.035%  83.2%  0.1945   betabinom   25   167 1.53e-05 |  1     no    1.072%
   1.0% binom |  29.30% [27.31, 31.35]% 100.0% |  25.48% [  0.00,  29.39]%  10.63% |  36.55% |  -3.82 |  0  1.035%  83.2%  0.1945   betabinom   25   177 1.53e-05 |  1     no    1.072%
   2.0%    hb |   0.00% [ 0.00,  0.18]% 100.0% |   0.00% [  0.00,   0.00]%   0.00% |  36.55% |  +0.00 | -1     n/a    n/a     n/a        None  n/a   n/a 0.00e+00 | -1    n/a       n/a
   2.0% binom |   0.00% [ 0.00,  0.18]% 100.0% |   0.00% [  0.00,   0.00]%   0.00% |  36.55% |  +0.00 | -1     n/a    n/a     n/a        None  n/a   n/a 0.00e+00 | -1    n/a       n/a

Bootstrap detail (counts of 1,000): forecast zero share; rho_I case counts; resamples whose start is not 19
   0.5%    hb: zero   2.4%; betabinom 976, rho_I<=0 0, zero errors 0, rho_I>=1 0, no j* 24; start != 19: 0
   0.5% binom: zero   2.4%; betabinom 976, rho_I<=0 0, zero errors 0, rho_I>=1 0, no j* 24; start != 19: 0
   1.0%    hb: zero  48.6%; betabinom 514, rho_I<=0 0, zero errors 0, rho_I>=1 0, no j* 486; start != 19: 0
   1.0% binom: zero  48.6%; betabinom 514, rho_I<=0 0, zero errors 0, rho_I>=1 0, no j* 486; start != 19: 0
   2.0%    hb: zero  99.8%; betabinom 2, rho_I<=0 0, zero errors 0, rho_I>=1 0, no j* 998; start != 19: 0
   2.0% binom: zero  99.8%; betabinom 2, rho_I<=0 0, zero errors 0, rho_I>=1 0, no j* 998; start != 19: 0

CRITERIA (Part A (f); HB only; consistency check on development data; 2% and exact binomial reported only):
  alpha 0.5%: forecast 6.61%, observed 17.00%, gap -10.39 pp
    (a) |forecast - observed| <= 3 points: FAIL
    (b) forecast >= observed - 2 points: FAIL
  alpha 1.0%: forecast 16.94%, observed 20.00%, gap -3.06 pp
    (a) |forecast - observed| <= 3 points: FAIL
    (b) forecast >= observed - 2 points: FAIL
  Overall (all four must pass): FAIL
