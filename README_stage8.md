# Stage 8: latency micro-benchmark

Two commands, two separate processes on purpose. The decision-layer timing never imports torch (so one thread really means one thread, and it works even if torch is blocked); the backbone timing needs torch.

| File | Goes to |
|---|---|
| `src/latency.py` | `src/` (new) |
| `tests/test_latency.py` | `tests/` (new) |
| `LOG_stage8_entry.md` | Template. **Part A goes into `LOG.md` and is committed BEFORE the run.** Part B after. |

Nothing existing is replaced. Everything it reads is hash-checked against the Stage 2-4 summaries; no Cert or Test label is read.

## Run

```powershell
cd "D:\FYP Review\POC Implementation\stage0_package VI\stage0_pkg"
py -3.12 -m pytest tests/test_latency.py -q                         # 13 passed (11 passed, 2 skipped if torch is blocked)
# paste LOG_stage8_entry.md Part A into LOG.md and commit it (30 seconds)
# then, for honest numbers: plug the laptop in, Windows power mode "Best performance", close browsers/IDEs/Teams, run from a plain PowerShell window
py -3.12 -m src.latency --run runs/20261003_stage1                  # decision layer, about 20-40 s
py -3.12 -m src.latency --run runs/20261003_stage1 --backbone       # backbone, about 2-4 min, needs torch. Do not touch the laptop while it runs.
```

Outputs: `runs/20261003_stage1/stage8/latency_decision.json` and `latency_backbone.json` (hardware, thread settings, library versions, every block). Send me both printouts.

## Backbone protocol, patch 2 (read this if you ran `--backbone` before)

Two backbone runs exist and **both are kept**. Run 1: the all-threads end-to-end row (47.2 ms) was inflated by a thread conflict between torch and numpy's BLAS; the forward-only number (9.7 ms) was fine. Run 2 (after patch 1): my fix set `MKL_NUM_THREADS=1`, which torch reads as its default thread count, so the "all threads" row silently ran on **one** thread, and the machine also slowed by 3-4x partway through (preprocess, backbone and decision layer all slowed together, and the GPU row with them). Its "DOES NOT HOLD (102 ms)" verdict is void.

Patch 2 changes the instrument, not the question:

- Only numpy's own OpenBLAS is pinned to one thread; torch's thread count is set explicitly per row, **verified** after setting, and the all-threads row refuses to be a one-thread row (it falls back to half the logical CPUs and says so).
- Each row is 3 rounds of (forward block, end-to-end block): 3 x 200 timed calls on CPU (3 x 400 on GPU), 20 warm-up per round (50 in the first). Per-round and pooled numbers are both in the JSON.
- A fixed single-thread CPU workload (the "canary") is timed before the row and after every round. If it, or the round medians, move by more than 25% the row is marked **NOT STABLE**.
- The "under 50 ms end to end" estimate is judged on the median of the three round medians of the all-threads row, and only if that row is stable. Otherwise the printout says NOT RELIABLE and you rerun. A rerun never replaces an earlier one; all go in the log.
- The model input is now made contiguous (as torchvision's `to_tensor` does); this is the standard preprocessing and can only help.

The decision-layer numbers are unaffected and valid; do not rerun them.

## What the decision layer is

The part that runs after the backbone for one image, from a cached 512-d L2-normalised embedding: whiten (512 x 512) -> 43 squared Mahalanobis distances -> log-posterior and top-1 posterior -> margin (top-1 minus top-2) -> epsilon lookup `log_eps[top1, top2]` -> gate check (`sqrt(min d2) <= tau`) -> group (hard iff `log_eps > c*`) -> threshold check -> accept or escalate. About 0.29 M multiply-adds.

**It is checked before it is timed.** The deployed function is run on every fine-tune embedding (about 11,000) and must reproduce the stored Stage 3 scores: top-1 and top-2 labels 100%, margin and gate score within float32 storage error, and its accept/escalate decision must equal the offline Stage 4/6 rule on the stored arrays (>= 99.95%, the slack is float32 rounding at exact thresholds). So what is timed is what was certified. The check differs from the Stage 3 code on purpose in one way: the deployed form precomputes `mu W` and uses a vector-matrix product, and the unit tests compare it with an independent route (squared distances through the precision matrix).

## Rules (pre-registered, constants in the code)

- 1,000 warm-up calls, then 10,000 timed calls, **each call timed individually** (`perf_counter_ns`), garbage collection off while timing, a different real embedding for every call (10,000 distinct), batch 1, one thread.
- Reported: median, p95, p99, mean, max. **Requirement: p95 <= 1 ms.** Primary = float64 (the arithmetic of Stage 3); float32 (1.15 MB, the size the plan assumed) is printed next to it.
- The verdict uses ONE block of 10,000 calls. Four more blocks are run as a stability check and the worst p95 is reported too ("stable" = all five blocks pass). I report both rather than picking the best.
- The margin thresholds inside the function are placeholders (medians of the Stage 4 grids); the cost does not depend on their value and no Stage 6 certificate is needed.
- Backbone (ResNet-18, 112 x 112, batch 1): median and p95 per image for CPU with 1 thread, CPU with all threads, and GPU if present (synchronised before the clock stops), plus an **end-to-end** row (uint8 image -> normalise -> backbone -> L2 -> decision layer). No threshold. The plan's earlier estimate "under 0.05 s per photo end to end on a normal CPU" is checked against the all-threads end-to-end median.

## Read this before you quote the numbers

- These are in-process Python/numpy times on **your laptop**. They exclude Python start-up, image decoding, disk and network. A p95 that passes 1 ms by 5x here says the decision layer is not the bottleneck; it does not say a production service meets a latency SLA.
- Windows timing is jittery. If the primary p95 is between 0.5 and 1 ms, or "NOT stable", rerun on mains power with the apps closed and tell me; do not cherry-pick the best run, report all of them.
- The max (tens of microseconds to a few ms) is dominated by the operating system and says little about the code.

## What I expect (written before you run it)

[Likely, ~90%] the decision layer passes by a wide margin (my 2-core cloud machine gives about 0.2 ms p95 for float64 and 0.15 ms for float32). [Likely, ~75%] the all-threads end-to-end CPU median is under 50 ms; [Guessing] the 1-thread end-to-end median is 20-60 ms and the GPU backbone 2-5 ms. If the CPU estimate fails, that is a number for the paper, not a failure of the project.
