## Stage 8, backbone RUN 3 (paste into LOG.md; filled from the printout of 2026-10-16 ~01:00; blanks ____ are for you)

- **Code:** patch 2 (`src/latency.py` SHA-256 ____), commit ____ ; amendment 1 committed BEFORE the run: ____ (yes / no) ; torch 2.14.1+cu132 (default threads 14) ; numpy ____ ; GPU NVIDIA GeForce RTX 5050 Laptop GPU ; CPU 13th Gen Intel Core i7-13650HX ; thread env in the shell: `OMP_NUM_THREADS` unset, `MKL_NUM_THREADS` unset, `OPENBLAS_NUM_THREADS` 1 (set by the script) ; all-threads row = 14 threads (confirmed by the script)
- **Conditions:** plugged in ____ ; Best performance ____ ; apps closed ____ ; window in the foreground the whole time ____ ; anything else ____

| row | forward median / p95 (ms) | end-to-end median / p95 (ms) | e2e round medians (ms) | forward round medians (ms) | canary (ms) | script says |
|---|---|---|---|---|---|---|
| CPU 1 thread | 100.94 / 110.20 | 105.09 / 116.20 | 107.91, **25.83**, 105.08 | 102.23, 53.71, 101.83 | 20.4, 67.1, 63.8, 65.9 | NOT STABLE |
| CPU 14 threads | 44.75 / 48.82 | 49.01 / 52.64 | 45.69, 49.63, 49.69 | 42.55, 44.68, 45.88 | 20.0, 64.5, 67.9, 63.5 | NOT STABLE |
| GPU | 15.03 / 21.28 | 17.48 / 24.96 | 17.37, 17.47, 17.58 | 15.27, 14.76, 15.19 | 60.8, 65.6, 65.2, 54.7 | "stable" (**wrong**, see below) |

- **Parts of the end-to-end call (median ms):** 1 thread: preprocess 0.89 / backbone + feature copy 102.67 / decision layer 0.790 ; 14 threads: 1.36 / 46.28 / 0.823 ; GPU: 1.32 / 15.30 / 0.676.
- **Earlier estimate "under 0.05 s per photo end to end on a normal CPU":** NOT JUDGED (the script's own rule: row not stable). Median of round medians 49.6 ms, in a machine state that was about 4x slower than its fast state, so the number says nothing about the estimate either way.
- **Prediction check (written before the run):** all-threads row stable [65%] -> NO ; all-threads end-to-end 12-20 ms and under 50 ms [85%] -> NO (49.6 ms, in the slow state) ; 1-thread end-to-end 23-27 ms [70%] -> NO as a pooled number (105 ms), but round 2 alone was 25.8 ms ; GPU end-to-end 4-6 ms -> NO (17.5 ms). All four missed, for the same reason.

### What run 3 shows (all runs kept)
1. **The machine changes speed during the run.** The fixed CPU workload (canary, a pure-Python loop that touches no torch and no numpy) took 20.0-20.4 ms at the start of both CPU rows and 54.7-67.9 ms (2.7-3.4x slower) after the first round of every row, including the GPU row, which never saw a fast reading. In the 1-thread row the middle round was fast again for its end-to-end block (25.8 ms) and the forward block of that round sat in between (53.7 ms), so the slow state comes and goes. [Certain for the readings; the cause is not identified.]
2. **The slow state explains every anomaly seen so far.** Slow-state versus earlier fast-state forward times: 1 thread 100.9 vs 23.2 ms (4.4x), 14 threads 44.8 vs 9.7 ms (4.6x), GPU path 15.0 vs 3.85 ms (3.9x). Run 1's all-threads end-to-end of 47.2 ms is almost exactly run 3's all-threads forward in the slow state (44.8 ms). So my explanation of run 1 (numpy BLAS fighting torch's thread pool) is **not supported** and is probably wrong; the MKL_NUM_THREADS bug of patch 1 was real but is a separate matter. [Likely]
3. **Two mistakes of mine in the instrument.** (a) The stability flag compares a row only with itself, so the GPU row was called "stable" while running at about 3x its fast-state speed (canary 55-66 ms against 20 ms). (b) Amendment 1 said the forward-only medians "have agreed across runs" and would be the fallback numbers. They have not (see item 2), so that fallback is void.
4. **What survives.** The decision layer (numpy, 1 thread) was measured in its own process before any of this; p95 87.6 us (float64) and 42.5 us (float32) against a 1 ms requirement. Even against the cheapest backbone time ever observed (GPU forward 3.85 ms, fast state) the decision layer is at most 2.3% of one image's compute; against the CPU all-threads forward in the fast state (9.7 ms) it is 0.9%. That statement does not depend on which machine state the backbone runs were in. [Likely, for a laptop; not a service-level claim.]
5. **What is unresolved.** Absolute backbone and end-to-end times on this laptop, and the 50 ms estimate. The laptop is not a controlled instrument until the slowdown is explained.

- **Decision taken and why:** ____ (see next step)
- **Deviation from R6?** ____ (none beyond amendment 1)
- **Next step:** run `py -3.12 -m src.cpu_probe` (about 90 s, no model) to find out whether the CPU is slow when idle, slows under sustained load, or oscillates. Then either (a) fix the cause and do ONE more run under a rule written before it, or (b) close Stage 8 with the decision layer as the result and the backbone times reported as unresolved, with runs 1-3 and the probe attached. No repeat runs until a number passes.
