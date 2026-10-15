## Stage 8, CLOSE-OUT (paste into LOG.md; filled from the mains probe of 2026-10-16 ~01:20; blanks ____ are for you)

- **Mains probe** (`runs/20261003_stage1/stage8/cpu_probe.json`; the battery probe is kept as `cpu_probe_earlier_*.json`)
  - Power at the probe: **plugged in, Battery Saver off**, battery 10% (charging), plan "Balanced". Windows "Best performance" mode set at the time: ____ (yes / no; `powercfg` cannot show it, so this line is yours to fill).
  - Canary pattern `FFFFFFFFFFSSSFFFFF`. Fast windows 10.4-11.5 ms; three slow windows (50-65 s into the probe) at 31.5-31.9 ms, 3.0x slower; then fast again for the last 20 s.
- **Rule from Amendment 2, point 2:** a mains probe with any S closes the backbone question. It has S. **Run 4 is not done. The decision-layer confirmatory run is not done** (it was tied to an all-F probe, and the first run's pass does not depend on it, see below).
- **Prediction check:** "mains probe is all F" [Likely ~70%] -> NO. The other predictions in Amendment 2 do not apply (no further runs).

### What the two probes show together
| | fast windows | slow windows | slow / fast |
|---|---|---|---|
| battery, Battery Saver on (01:12) | 18.7-20.3 ms | 70.4-73.9 ms (45-65 s) | 3.8x |
| mains, Battery Saver off (01:20) | 10.4-11.5 ms | 31.5-31.9 ms (50-65 s) | 3.0x |

1. **The speed this laptop calls "fast" on battery is about 1.8x slower than on mains** (about 19 ms against 10.5 ms for the same loop). [Certain for the readings.]
2. **Backbone run 3's canary values (20.0-20.4 ms fast, 54.7-67.9 ms slow) match the battery probe, not the mains probe.** So run 3 was [Likely] run in the battery-capped regime. The 20.0 ms "fast-state reference" I put in `src/cpu_probe.py` (`REFERENCE_MS`) and used in my own reasoning was therefore the battery regime; the mains figure is about 10.5 ms. Not patched, because no further runs follow. Anyone reusing the probe should pass `--reference-ms 10.5`.
3. **Mains power did not remove the slow episodes.** They are shorter (15 s against 20 s) and milder (3.0x against 3.8x). In both probes the slow episode starts about 25-30 s into sustained single-thread load and ends at about the moment the probe pauses for its own measurement call. [Guessing] a core-placement or scheduler effect (the thread being moved to a slower core) or periodic background work, not heat; the evidence cannot separate these. **The cause is not identified and I am not pursuing it for the POC.**
4. **Consequence:** this laptop is not a controlled instrument for backbone latency. Runs 1-3 stay in the log as they are, each with its own explanation; none is quoted as a speed.

### What may and may not be claimed (for the proposal and the paper)
- **May:** the decision layer, as pre-registered and measured in isolation (numpy, 1 thread, batch 1, warm loop, 10,000 distinct embeddings): p95 87.6 us (float64), 42.5 us (float32, 1.15 MB of constants), about 0.29 M multiply-adds, requirement 1 ms met with an 11x margin, stable over five blocks, reproducing the Stage 3 scores (labels 100%). Throttling can only raise a measured latency, so unmet power conditions at that run can make the figure pessimistic, never optimistic. State the hardware (i7-13650HX laptop) and that the CPU speed was not controlled.
- **Must add (in-situ caveat):** inside the end-to-end call, the same decision code (plus the L2 normalisation and the float cast) took **0.24-0.82 ms (median)**, 3-9x the isolated p95: 1.0% of the end-to-end time on CPU in the one fast-state row and 0.7-1.7% in slow-state CPU rows, 3.8-3.9% on the GPU path. [Likely] cause: cold caches after a ResNet forward pass (the 512 x 512 float64 matrix is 2 MB); not tested. So the honest sentence is "about a millisecond at most, and a few percent of the end-to-end time in every state observed", not "2% of the backbone". The earlier figure I gave (at most 2.3% of the cheapest backbone time) used the isolated number and was too optimistic for the GPU path. The in-situ medians are below the 1 ms line, but their p95 was not recorded, and the slow-state medians (0.65-0.82 ms) are close to it.
- **May not:** any absolute backbone or end-to-end latency, and the earlier estimate "under 0.05 s per photo end to end on a normal CPU" (**not judged**; the observed all-threads forward times spanned 9.7-44.8 ms and end-to-end 47-50 ms depending on the machine state). Report it as "not measured under controlled conditions on the development laptop; to be measured on a fixed-clock machine in the full study".

### Run list (all kept)
| run | what | status |
|---|---|---|
| decision layer, first run | 5 blocks, power state not recorded | **valid, primary** |
| backbone 1 | all-threads end-to-end 47.2 ms, explanation (BLAS conflict) not supported | unusable as a speed |
| backbone 2 (patch 1) | torch clamped to 1 thread by my env-var bug; machine slowed mid-run | void |
| backbone 3 (patch 2) | rows NOT STABLE; GPU row flagged stable but about 3x slow; battery-capped regime | unusable as a speed |
| probe 1 (battery, saver on) | `FFFFFFFFFSSSSFFFFF` | diagnosis |
| probe 2 (mains) | `FFFFFFFFFFSSSFFFFF` | diagnosis; closes the question |

- **Instrument changes after seeing data (all declared):** patch 1 (env pins; my bug), patch 2 (rounds, canary, stability, verdict rule), patch 3 (power gate, reference gate, keep-earlier files), the probe. None touches the decision-layer arithmetic or its protocol.
- **Decision taken and why:** Stage 8 closed. The decision layer is the Stage 8 result. Backbone latency reported as unresolved on this machine, per the rule written before the mains probe; no further runs until a machine with fixed clocks is available (full study).
- **Deviation from R6?** Backbone numbers not obtained (declared above); decision layer as pre-registered.
- **Next step:** POC wrap-up: one-page G1 and G2 summary for the proposal, then the full-study plan (with a latency line: backbone timing on a fixed-clock machine, decision layer measured both isolated and in situ with p95).
