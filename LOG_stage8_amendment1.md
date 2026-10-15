## Stage 8, AMENDMENT 1: backbone protocol (paste into LOG.md and COMMIT before the third backbone run)

- **Date / stage:** ____ / Stage 8 (latency), backbone part only. **The decision-layer protocol, run and result are untouched** (p95 float64 87.6 us, float32 42.5 us, stable, reproduces Stage 3; requirement 1 ms).
- **Git commit (patch 2 code and this text):** ____

### What happened (all runs are kept, none discarded)
- **Run 1** (`--backbone`, 14 default torch threads): forward-only numbers were plausible (all-threads forward about 9.7 ms). The all-threads **end-to-end** median was 47.2 ms, about 37 ms above the forward time. My explanation: numpy's BLAS threads and torch's thread pool competing for the same cores inside the end-to-end call. [Likely, not proven: run 1 had no per-part breakdown.]
- **Run 2** (after my patch 1): the patch set `MKL_NUM_THREADS=1` as well as `OPENBLAS_NUM_THREADS=1`. Torch reads `MKL_NUM_THREADS` as its default thread count, so torch reported 1 thread (it had reported 14), and the row labelled "all threads" ran on **one** thread. That is my bug, confirmed in the sandbox (torch 2.14.1: default 2 threads; with `MKL_NUM_THREADS=1` or `OMP_NUM_THREADS=1` it drops to 1; with `OPENBLAS_NUM_THREADS=1` alone it stays at 2). Separately, in that run the second row's end-to-end median was 102 ms against 25 ms in the first row at the same nominal thread count, although its forward-only block (24 ms) matched the first row; its parts (preprocess 1.92 vs 0.63 ms, backbone 99 vs 24 ms, decision layer 0.72 vs 0.24 ms) all slowed by about 3-4x together, and the GPU row was slower too (end-to-end 17 ms; p95 of the forward block 16 ms against a 3.9 ms median). All components slowing together points at the machine's state (clock, power mode, core type, background load), not at the code. [Likely; the cause of this slowdown is not identified.] **The verdict printed in run 2 ("DOES NOT HOLD, 102 ms") is void.**

### What changes (the instrument, not the question)
1. Only numpy's OpenBLAS is pinned to 1 thread; `MKL_NUM_THREADS` / `OMP_NUM_THREADS` are never set by the script. Each torch row sets its thread count with `torch.set_num_threads` and the script asserts it took. If torch reports 1 thread on a multi-core machine, the all-threads row uses half the logical CPUs and prints a warning.
2. Every row is 3 rounds of (forward block, end-to-end block): 3 x 200 timed calls on CPU, 3 x 400 on GPU, 20 warm-up calls per round on CPU (50 in the first round); on GPU 100 in the first round and 30 after. Per-round and pooled numbers are both stored.
3. A fixed single-thread CPU workload (canary) is timed before the row and after every round.
4. **Stable** = the end-to-end round medians, the forward round medians and the canary times each agree within a factor of 1.25. Fixed now, before the run.
5. **Rule for the earlier estimate ("under 0.05 s per photo end to end on a normal CPU"):** judged on the median of the three round medians of the all-threads end-to-end row, **only if that row is stable**; otherwise the result is "not reliable, rerun", and the estimate stays unjudged. A rerun is logged next to the earlier ones, never instead of them.
6. The model input is made contiguous (standard `to_tensor` preprocessing). An existing `latency_backbone.json` is renamed (`latency_backbone_earlier_<time>.json`), not overwritten.

### Predictions for the third run (written before running; do not edit afterwards)
- All-threads row is stable on a quiet, plugged-in machine: [Likely ~65%].
- All-threads end-to-end median 12-20 ms and under 50 ms: [Likely ~85%]. 1-thread end-to-end median 23-27 ms: [Likely ~70%]. GPU end-to-end 4-6 ms: [Guessing].
- If a row is NOT STABLE again, I will report that the laptop is not a controlled instrument for this measurement and give the forward-only medians (which have agreed across runs) as the backbone numbers, with the end-to-end estimate left unjudged. I will not run it repeatedly until a pass appears.

### Conditions (fill in before the run)
- Plugged in ____ ; power mode "Best performance" ____ ; other apps closed ____ ; plain PowerShell window ____ ; `$env:OMP_NUM_THREADS` and `$env:MKL_NUM_THREADS` empty ____ ; anything else running ____

---
## Stage 8, PART B (backbone, run 3): results (fill in after the run)

- **Code:** `src/latency.py` SHA-256 ____ ; `py -3.12 -m pytest tests/test_latency.py -q` gives ____ passed, ____ skipped ; `latency_backbone.json` SHA-256 ____ ; earlier files kept: ____
- **Hardware / versions:** printed at the top of the run and stored in the JSON: CPU ____ ; logical CPUs ____ ; torch ____ ; GPU ____ ; "all threads" = ____ threads

| row | forward median / p95 (ms) | end-to-end median / p95 (ms) | round medians, end-to-end (ms) | canary (ms) | stable |
|---|---|---|---|---|---|
| CPU 1 thread | | | | | |
| CPU ____ threads | | | | | |
| GPU ____ | | | | | |

- **Parts of the end-to-end call (median ms): preprocess ____ / backbone + feature copy ____ / decision layer ____** (all-threads row)
- **Earlier estimate "under 0.05 s per photo end to end on a normal CPU":** ____ (holds / does not hold / not reliable) ; median of round medians ____ ms
- **Runs so far, all kept:** run 1 (47.2 ms all-threads end-to-end, inflated by a thread conflict) ; run 2 (void: one-thread row mislabelled, machine slowed mid-run) ; run 3 (this one)
- **Anything unusual:** ____
- **Prediction check:** stable ____ ; 12-20 ms and under 50 ms ____ ; 1-thread 23-27 ms ____ ; GPU 4-6 ms ____
- **Decision taken and why:** ____
- **Deviation from R6?** ____ (the backbone instrument changed after two runs, as declared above)
- **Next step:** POC wrap-up: G1 and G2 summary for the proposal, then the full-study plan.
