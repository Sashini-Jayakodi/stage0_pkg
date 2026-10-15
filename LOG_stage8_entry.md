## Stage 8, PART A: PRE-REGISTRATION (paste into LOG.md and COMMIT before running)

- **Date / stage / run tag:** ____ / Stage 8 (latency) / 20261003_stage1
- **Git commit (code and this pre-registration):** ____
- **Decision layer:** whiten -> 43 squared Mahalanobis distances -> log-posterior -> margin -> eps lookup -> gate check -> group -> threshold check, numpy only, one CPU thread, batch 1, input a cached 512-d L2-normalised embedding. Checked against the stored Stage 3 scores on every fine-tune embedding before timing (labels 100%, accept decisions vs the offline rule >= 99.95%).
- **Protocol (fixed now):** 1,000 warm-up calls, 10,000 timed calls, each timed individually, GC off, 10,000 distinct real embeddings, batch 1. Median, p95, p99, mean, max. **Requirement: decision-layer p95 <= 1 ms** (primary = float64, ONE block of 10,000; four more blocks as a stability check, worst p95 also reported; float32 reported alongside). Backbone: median and p95 per image for CPU 1 thread, CPU all threads, GPU if present, and end-to-end (image to decision); no threshold. Earlier estimate to check: under 0.05 s per photo end to end on a normal CPU.
- **Conditions I will set:** plugged in, Best performance power mode, other apps closed, plain PowerShell window. ____ (note anything different)
- **Predictions (do not edit afterwards):** decision layer passes with a wide margin [Likely ~90%]; all-threads end-to-end CPU median under 50 ms [Likely ~75%]. ____ (add your own)

---
## Stage 8, PART B: results (fill in after the runs)

- **Hardware:** CPU ____ ; logical CPUs ____ ; RAM ____ ; GPU ____ ; OS ____ ; Python ____ ; numpy ____ ; torch ____ (all printed in the JSON files)
- **Code:** `src/latency.py` SHA-256 ____ ; `py -3.12 -m pytest tests/test_latency.py -q` gives ____ passed, ____ skipped. Outputs: `latency_decision.json` SHA-256 ____ ; `latency_backbone.json` ____

### Decision layer (numpy, 1 thread, batch 1)
- **Reproduces Stage 3 on ____ embeddings:** top-1 ____ %, top-2 ____ %, max |margin difference| ____ , accept decisions vs offline rule ____ %.

| variant | constants | median (us) | p95 (us) | p99 (us) | mean (us) | max (us) | worst p95 over 5 blocks (us) | p95 <= 1 ms | stable |
|---|---|---|---|---|---|---|---|---|---|
| float64 (primary) | ____ MB | | | | | | | | |
| float32 | ____ MB | | | | | | | | |

- Timer overhead ____ us. Threads: ____ (from `threadpools` in the JSON).

### Backbone (ResNet-18, 112 x 112, batch 1)

| row | forward median / p95 (ms) | end-to-end image to decision median / p95 (ms) |
|---|---|---|
| CPU 1 thread | | |
| CPU ____ threads | | |
| GPU ____ | | |

- **Earlier estimate "under 0.05 s per photo end to end on a normal CPU":** ____ (all-threads end-to-end median ____ ms)
- **Anything unusual (noisy run, other apps, thermal throttling, rerun):** ____ (all runs are reported, none discarded)

**Checks**
- [ ] tests pass
- [ ] decision layer reproduces Stage 3 (script asserts)
- [ ] p95 <= 1 ms in the primary block, and stable across the five blocks
- [ ] numbers copied from the JSON, with hardware and versions

- **Decision taken and why:** ____
- **Deviation from R6?** ____ (none expected)
- **Next step:** POC wrap-up: G1 and G2 summary for the proposal, then the full-study plan.
