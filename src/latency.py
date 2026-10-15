"""Stage 8: latency micro-benchmark (plan section 8). Two commands, two separate processes on purpose:

  py -3.12 -m src.latency --run runs/20261003_stage1               DECISION LAYER. numpy only (no torch is imported), one CPU thread, batch 1.
  py -3.12 -m src.latency --run runs/20261003_stage1 --backbone    BACKBONE (needs torch): CPU 1 thread, CPU all threads, GPU if present, plus
                                                                   an end-to-end CPU number (uint8 image -> embedding -> decision).

Decision layer = what runs after the backbone for ONE image, from a cached 512-d L2-normalised embedding x:
  whiten (x W) -> 43 squared Mahalanobis distances -> log-posterior and top-1 posterior -> margin (top1 - top2) -> eps lookup log_eps[top1, top2]
  -> gate check (sqrt(min d2) <= tau) -> group (hard iff log_eps > c*) -> threshold check (margin >= the group's threshold) -> accept or escalate.
It is the same arithmetic as Stage 3 (`head.score`) and Stage 4/6 (gate, c*, accept rule); before timing it, it is checked against the stored Stage 3 scores
on every fine-tune embedding (labels must agree 100%), so the thing that is timed is known to be the thing that was certified.

Rules fixed BEFORE running (constants below, not options): 1,000 warm-up calls then 10,000 timed calls, each call timed on its own with perf_counter_ns, garbage
collection off while timing, a different real embedding for every call (10,000 distinct), median and p95 (and p99, mean, max) reported, requirement p95 <= 1 ms
for the decision layer (primary = float64, which mirrors Stage 3 exactly; float32 = 1.1 MB, what the plan's size estimate assumes, is reported next to it).
The primary number is ONE block of 10,000 calls; four more blocks are run as a stability check and the worst p95 is reported too.
The backbone has no threshold: median and p95 per image. Backbone protocol (patch 2, written after two backbone runs, both kept; see README_stage8.md): each row = 3 rounds of
(forward block, end-to-end block), per-round and pooled numbers, a fixed CPU canary timed before and after every round, a row is "stable" only if the round medians and the canary
agree within 25%, and the 50 ms estimate gets a verdict only from a stable all-threads row. The margin THRESHOLDS used here are placeholders (medians of the Stage 4 grids): the cost of the
decision does not depend on their value, and no certified threshold from Stage 6 is needed or read.
"""
import os

import sys  # noqa: E402

if "--backbone" not in sys.argv:                          # decision-layer mode: ONE thread, everything (torch is never imported here)
    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[_v] = "1"                              # before numpy is imported (works when run as `python -m src.latency`); run_decision also uses threadpoolctl if present
else:                                                     # backbone mode: pin ONLY numpy's own OpenBLAS, so the decision layer inside the end-to-end row does not fight torch for cores.
    os.environ["OPENBLAS_NUM_THREADS"] = "1"              # NEVER set MKL_NUM_THREADS / OMP_NUM_THREADS here: torch reads both as its default thread count (patch 1 did, and silently made
                                                          # the 'all threads' row single-threaded). Thread counts of the torch rows are set explicitly with torch.set_num_threads and verified.

import argparse  # noqa: E402
import gc  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import yaml  # noqa: E402

from src.util import sha256_file  # noqa: E402

C8 = {"warmup": 1000, "timed": 10000, "repeats": 5, "req_p95_ms": 1.0, "seed": 0,
      "bb_rounds": 3, "bb_cpu_warmup": 50, "bb_cpu_round_warmup": 20, "bb_cpu_timed": 200,       # backbone (patch 2): 3 rounds per row, forward block then end-to-end block in each round
      "bb_gpu_warmup": 100, "bb_gpu_round_warmup": 30, "bb_gpu_timed": 400, "bb_stable_ratio": 1.25,
      "e2e_cpu_ms_estimate": 50.0,
      "repro_label_agreement": 1.0, "repro_margin_tol": 1e-3, "repro_decision_agreement": 0.9995}


# ------------------------------------------------------------------ the decision layer
class DecisionLayer:
    """Deployment form of the Stage 3-6 decision: constants are precomputed once (Mw = mu W, |Mw|^2, log priors), then each call is a few small numpy operations."""

    def __init__(self, head, log_eps, tau, c_star, thr_easy, thr_hard, dtype=np.float64):
        self.dtype = np.dtype(dtype)
        self.W = np.ascontiguousarray(head["W"], dtype=self.dtype)
        Mw = head["mu"].astype(np.float64) @ head["W"].astype(np.float64)
        self.Mw = np.ascontiguousarray(Mw, dtype=self.dtype)
        self.m2 = np.ascontiguousarray((Mw ** 2).sum(1), dtype=self.dtype)
        self.log_priors = np.log(head["priors"].astype(np.float64)).astype(self.dtype)
        self.log_eps = np.ascontiguousarray(np.nan_to_num(log_eps, nan=0.0), dtype=np.float64)       # diagonal is never read (top-1 != top-2)
        self.tau, self.c_star = float(tau), float(c_star)
        self.thr = (float(thr_easy), float(thr_hard))
        self.neg_inf = self.dtype.type(-np.inf)

    @property
    def n_bytes(self):
        return int(self.W.nbytes + self.Mw.nbytes + self.m2.nbytes + self.log_priors.nbytes + self.log_eps.nbytes)

    def __call__(self, x):
        xw = np.asarray(x, dtype=self.dtype) @ self.W                          # whiten
        d2 = self.m2 + xw @ xw - 2.0 * (self.Mw @ xw)                          # 43 squared Mahalanobis distances
        np.maximum(d2, 0.0, out=d2)
        lp = self.log_priors - 0.5 * d2                                        # unnormalised log-posterior (the normaliser cancels in the margin)
        i1 = int(lp.argmax())
        v1 = lp[i1]
        post1 = 1.0 / float(np.exp(lp - v1).sum())                             # top-1 posterior (log-sum-exp)
        lp[i1] = self.neg_inf
        i2 = int(lp.argmax())
        margin = float(v1 - lp[i2])                                            # top-1 minus top-2 log-posterior
        log_eps = self.log_eps[i1, i2]
        s = float(np.sqrt(d2.min()))
        gate_ok = s <= self.tau
        hard = log_eps > self.c_star
        accept = bool(gate_ok and margin >= self.thr[1 if hard else 0])
        return accept, i1, i2, margin, float(log_eps), s, post1, bool(hard), bool(gate_ok)


def load_decision_inputs(run, cfg_all, seed=0):
    """Frozen Stage 3 head, Stage 4 gate and cut for reshuffle `seed`, with every file hash-checked. No Cert or Test label is read."""
    run = Path(run)
    es = json.loads((run / "embeddings" / "embeddings_summary.json").read_text())
    want = (cfg_all["embed"].get("expected_checkpoint_sha256") or "").lower()
    assert es["checkpoint_sha256"].lower() == want, "embeddings are not from the frozen checkpoint"
    assert sha256_file(run / "embeddings" / "ft_train.npz") == es["files_sha256"]["ft_train.npz"], "ft_train.npz changed since Stage 2"
    hs = json.loads((run / "head_emb" / "head_summary.json").read_text())
    for f in ("head.npz", "scores_train_release.npz"):
        assert sha256_file(run / "head_emb" / f) == hs["files_sha256"][f], f"{f} changed since Stage 3"
    if (run / "head_emb" / "scores_ft_train.npz").exists() and "scores_ft_train.npz" in hs["files_sha256"]:
        assert sha256_file(run / "head_emb" / "scores_ft_train.npz") == hs["files_sha256"]["scores_ft_train.npz"], "scores_ft_train.npz changed since Stage 3"
    sp = run / "splits"
    summ = json.loads((sp / "stage4_summary.json").read_text())
    f = f"seed{seed}.npz"
    assert sha256_file(sp / f) == summ["files_sha256"][f], f"{f} changed since Stage 4"
    sres = next(r for r in summ["seeds"] if r["seed"] == seed)
    S = np.load(sp / f)
    H = np.load(run / "head_emb" / "head.npz")
    return {"head": {k: H[k] for k in ("mu", "W", "priors")}, "log_eps": H["log_eps"], "tau": float(sres["gate"]["tau"]), "c_star": float(sres["eps2"]["c_star"]),
            "thr_easy": float(np.median(S["eps2_easy_margin_grid"])), "thr_hard": float(np.median(S["eps2_hard_margin_grid"])),
            "emb": np.load(run / "embeddings" / "ft_train.npz")["emb"], "scores": np.load(run / "head_emb" / "scores_ft_train.npz"),
            "files": {"head.npz": hs["files_sha256"]["head.npz"], f: summ["files_sha256"][f]}}


def reproduce_stage3(layer, inp, c8):
    """Run the deployed function on EVERY fine-tune embedding and compare with the stored Stage 3 scores and the offline Stage 4/6 accept rule."""
    sc, emb = inp["scores"], inp["emb"]
    n = len(emb)
    out = [layer(emb[i]) for i in range(n)]
    acc, i1, i2, mg, le, s = (np.array([o[k] for o in out]) for k in (0, 1, 2, 3, 4, 5))
    top3 = sc["top3"].astype(int)
    r = {"n": int(n), "top1_agreement": float((i1 == top3[:, 0]).mean()), "top2_agreement": float((i2 == top3[:, 1]).mean()),
         "max_abs_margin_diff": float(np.abs(mg - sc["margin"]).max()), "max_abs_s_diff": float(np.abs(s - sc["s"]).max()),
         "max_abs_log_eps_diff": float(np.abs(le - sc["log_eps12"]).max())}
    tol = c8["repro_margin_tol"]                                               # stored values are float32: allow a small relative error on top of the absolute one
    r["margin_within_tol"] = bool((np.abs(mg - sc["margin"]) <= tol + 1e-5 * np.abs(sc["margin"])).all())
    r["s_within_tol"] = bool((np.abs(s - sc["s"]) <= tol + 1e-5 * np.abs(sc["s"])).all())
    hard = sc["log_eps12"].astype(float) > layer.c_star                        # offline rule from the stored arrays (what Stage 6 does with accept_mask)
    off = (sc["s"].astype(float) <= layer.tau) & (sc["margin"].astype(float) >= np.where(hard, layer.thr[1], layer.thr[0]))
    r["decision_agreement_with_offline_rule"] = float((acc == off).mean())
    r["accept_rate"] = float(acc.mean())
    return r


# ------------------------------------------------------------------ timing
def time_calls(fn, xs, warmup, timed, sync=None):
    """Per-call wall time in ns. Each call timed on its own; GC off while timing; `sync` (GPU) is called before the clock stops."""
    n = len(xs)
    for i in range(warmup):
        fn(xs[i % n])
    if sync:
        sync()
    ts = np.empty(timed, dtype=np.int64)
    pc = time.perf_counter_ns
    was = gc.isenabled()
    gc.disable()
    try:
        for i in range(timed):
            x = xs[i % n]
            t0 = pc()
            fn(x)
            if sync:
                sync()
            ts[i] = pc() - t0
    finally:
        if was:
            gc.enable()
    return ts


def summarize(ns):
    ms = np.asarray(ns, dtype=np.float64) / 1e6
    return {"n": int(len(ms)), "median_ms": float(np.median(ms)), "p95_ms": float(np.percentile(ms, 95)), "p99_ms": float(np.percentile(ms, 99)),
            "mean_ms": float(ms.mean()), "max_ms": float(ms.max()), "min_ms": float(ms.min()), "std_ms": float(ms.std(ddof=1)) if len(ms) > 1 else 0.0}


def cpu_name():
    try:
        if sys.platform.startswith("win"):
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as k:
                return winreg.QueryValueEx(k, "ProcessorNameString")[0].strip()
        if Path("/proc/cpuinfo").exists():
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or "unknown"


def hardware_info():
    info = {"platform": platform.platform(), "python": platform.python_version(), "cpu": cpu_name(), "logical_cpus": os.cpu_count(), "numpy": np.__version__,
            "thread_env": {v: os.environ.get(v) for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}}
    try:
        import scipy
        info["scipy"] = scipy.__version__
    except Exception:
        pass
    return info


def run_decision(run, cfg_all, c8, out, n_override=None):
    limit = None
    try:
        from threadpoolctl import threadpool_info, threadpool_limits
        limit = threadpool_limits(limits=1)
        pools = [{k: p.get(k) for k in ("internal_api", "num_threads", "version")} for p in threadpool_info()]
    except Exception:
        pools = "threadpoolctl not installed: single thread enforced through the environment variables set before numpy was imported"
    inp = load_decision_inputs(run, cfg_all, c8["seed"])
    emb = inp["emb"]
    rng = np.random.default_rng([c8["seed"], 88])
    n_timed, n_warm = (n_override["timed"], n_override["warmup"]) if n_override else (c8["timed"], c8["warmup"])
    pick = rng.permutation(len(emb))[:n_timed]
    xs = [emb[i] for i in pick]                                                # float32 as stored: the cast to float64 is part of the timed call
    result = {"constants": {k: c8[k] for k in ("warmup", "timed", "repeats", "req_p95_ms")} if not n_override else dict(n_override),
              "hardware": hardware_info(), "threadpools": pools, "distinct_embeddings": int(len(set(pick.tolist()))),
              "decision_inputs": {"tau": inp["tau"], "c_star": inp["c_star"], "thr_easy_placeholder": inp["thr_easy"], "thr_hard_placeholder": inp["thr_hard"], "files_sha256": inp["files"]},
              "variants": {}}
    timer = summarize(time_calls(lambda x: None, xs, 100, 5000))
    result["timer_overhead_median_ms"] = timer["median_ms"]
    for name, dt in (("float64", np.float64), ("float32", np.float32)):
        layer = DecisionLayer(inp["head"], inp["log_eps"], inp["tau"], inp["c_star"], inp["thr_easy"], inp["thr_hard"], dt)
        repro = reproduce_stage3(layer, inp, c8)
        if name == "float64":                                                  # the primary variant must be the certified arithmetic
            assert repro["top1_agreement"] >= c8["repro_label_agreement"] and repro["top2_agreement"] >= c8["repro_label_agreement"], f"decision layer disagrees with Stage 3 labels: {repro}"
            assert repro["margin_within_tol"] and repro["s_within_tol"], f"decision layer disagrees with Stage 3 scores: {repro}"
            assert repro["decision_agreement_with_offline_rule"] >= c8["repro_decision_agreement"], f"accept decisions differ from the offline rule: {repro}"
        blocks = []
        for b in range(c8["repeats"] if not n_override else n_override["repeats"]):
            blocks.append(summarize(time_calls(layer, xs, n_warm, n_timed)))
        primary = blocks[0]
        result["variants"][name] = {"constant_bytes": layer.n_bytes, "reproduces_stage3": repro, "primary_block": primary, "all_blocks": blocks,
                                    "worst_p95_ms_over_blocks": max(b["p95_ms"] for b in blocks),
                                    "meets_requirement_primary": bool(primary["p95_ms"] <= c8["req_p95_ms"]),
                                    "stable_all_blocks_meet_requirement": bool(all(b["p95_ms"] <= c8["req_p95_ms"] for b in blocks))}
    if limit is not None:
        limit.unregister()
    out.mkdir(parents=True, exist_ok=True)
    p = out / "latency_decision.json"
    result["code_sha256"] = sha256_file(Path(__file__))
    p.write_text(json.dumps(result, indent=2))
    report_decision(result)
    return result


def report_decision(r):
    print("\n=== Stage 8: decision layer (numpy, 1 thread, batch 1) ===")
    h = r["hardware"]
    print(f"CPU {h['cpu']} | {h['logical_cpus']} logical | {h['platform']} | python {h['python']} numpy {h['numpy']} | timer overhead {1000 * r['timer_overhead_median_ms']:.2f} us")
    print(f"{r['distinct_embeddings']} distinct real embeddings; warm-up {r['constants'].get('warmup')}, timed {r['constants'].get('timed')} per block, {r['constants'].get('repeats')} blocks")
    for name, v in r["variants"].items():
        b, rep = v["primary_block"], v["reproduces_stage3"]
        print(f"\n{name}: constants {v['constant_bytes'] / 1e6:.2f} MB | reproduces Stage 3 on {rep['n']} embeddings: top-1 {100 * rep['top1_agreement']:.3f}%, top-2 {100 * rep['top2_agreement']:.3f}%, "
              f"max |margin diff| {rep['max_abs_margin_diff']:.1e}, decisions vs offline rule {100 * rep['decision_agreement_with_offline_rule']:.3f}%")
        print(f"  PRIMARY block: median {1000 * b['median_ms']:.1f} us | p95 {1000 * b['p95_ms']:.1f} us | p99 {1000 * b['p99_ms']:.1f} us | mean {1000 * b['mean_ms']:.1f} us | max {1000 * b['max_ms']:.1f} us")
        print(f"  worst p95 over {len(v['all_blocks'])} blocks: {1000 * v['worst_p95_ms_over_blocks']:.1f} us | requirement p95 <= {1000 * r['constants'].get('req_p95_ms', 1.0):.0f} us: "
              f"{'PASS' if v['meets_requirement_primary'] else 'FAIL'} (primary), {'stable' if v['stable_all_blocks_meet_requirement'] else 'NOT stable'} over all blocks")
    print("\nNEXT: py -3.12 -m src.latency --run <run> --backbone (needs torch), then fill LOG_stage8_entry.md.")


# ------------------------------------------------------------------ backbone (torch)
def canary_ms(reps=7):
    """A fixed single-thread CPU workload (pure-Python integer loop, no BLAS, no torch). Its time changes if the machine changes state mid-run (clock speed, power mode,
    performance vs efficiency core). Timed before the row and after every round; a change of more than bb_stable_ratio flags the row as not comparable with itself."""
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter_ns()
        s = 0
        for i in range(300000):
            s += i * i
        ts.append(time.perf_counter_ns() - t0)
    return float(min(ts)) / 1e6


def all_threads(default_threads, cpu_count):
    """Thread count of the 'all threads' row: torch's own default when it is above 1. If torch reports 1 on a multi-core machine something has clamped it (an OMP_/MKL_NUM_THREADS
    variable in the shell): fall back to half the logical CPUs and say so, instead of silently labelling a 1-thread run 'all threads'."""
    if default_threads > 1 or (cpu_count or 1) <= 1:
        return int(default_threads), False
    return max(2, (cpu_count or 2) // 2), True


def estimate_verdict(round_medians_ms, stable, limit_ms):
    """Earlier estimate 'under 50 ms end to end on a normal CPU'. Verdict = median of the round medians of the all-threads end-to-end row, and ONLY if the row was stable
    (rounds agree within bb_stable_ratio, canary unchanged); otherwise None = 'not reliable, rerun'."""
    m = float(np.median(round_medians_ms))
    return {"median_ms_all_threads": m, "round_medians_ms": [float(x) for x in round_medians_ms], "stable": bool(stable),
            "holds": (bool(m < limit_ms) if stable else None)}


def run_backbone(run, cfg_all, c8, device_arg, out, n_override=None):
    import torch
    from src import train as TR
    run = Path(run)
    want = (cfg_all["embed"].get("expected_checkpoint_sha256") or "").lower()
    ck = run / "best.pt"
    sha = sha256_file(ck)
    assert sha.lower() == want, f"CHECKPOINT HASH MISMATCH: {sha} != {want}"
    size = yaml.safe_load((run / "config_used.yaml").read_text())["image_size"]
    inp = load_decision_inputs(run, cfg_all, c8["seed"])
    layer = DecisionLayer(inp["head"], inp["log_eps"], inp["tau"], inp["c_star"], inp["thr_easy"], inp["thr_hard"], np.float64)
    X = np.load(Path(cfg_all["data"]["cache_dir"]) / f"train_{size}.npy", mmap_mode="r")
    imgs = [np.array(X[i]) for i in np.random.default_rng([c8["seed"], 99]).permutation(len(X))[:200]]      # pixels only, no labels
    g = lambda k: (n_override or {}).get(k, c8[k])                                                           # noqa: E731
    default_threads = torch.get_num_threads()
    n_all, fell_back = all_threads(default_threads, os.cpu_count())
    if fell_back:
        print(f"WARNING: torch reports {default_threads} thread(s) on a {os.cpu_count()}-CPU machine (check OMP_NUM_THREADS / MKL_NUM_THREADS in this shell); "
              f"the 'all threads' row uses {n_all} threads instead.", flush=True)
    result = {"hardware": {**hardware_info(), "torch": torch.__version__, "torch_default_threads": default_threads, "cuda": torch.cuda.is_available(),
                           "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},
              "all_threads_row": {"threads": n_all, "fell_back_from_default": fell_back},
              "protocol": {k: g(k) for k in ("bb_rounds", "bb_cpu_warmup", "bb_cpu_round_warmup", "bb_cpu_timed", "bb_gpu_warmup", "bb_gpu_round_warmup", "bb_gpu_timed", "bb_stable_ratio")},
              "checkpoint_sha256": sha, "image_size": size, "input": f"1 x 3 x {size} x {size}, batch 1, contiguous", "rows": {}}
    grab = {}
    ratio = g("bb_stable_ratio")

    def build(device):
        m = TR.build_model(pretrained=False).to(device)
        m.load_state_dict(torch.load(ck, map_location=device, weights_only=False)["model"])
        m.eval()
        m.avgpool.register_forward_hook(lambda mod, i, o: grab.__setitem__("f", o.flatten(1)))
        return m

    def to_input(img, device):                                                                               # uint8 HxWx3 -> normalised 1x3xHxW float, contiguous (as torchvision's to_tensor)
        return TR.normalize(torch.from_numpy(img).to(device).permute(2, 0, 1).contiguous()[None].float().div_(255))

    def make_fns(model, device):
        tens = [to_input(i, device) for i in imgs]
        parts = []                                                                                           # per e2e call: (preprocess, backbone + feature copy, decision) in ns

        def fwd(t):
            with torch.no_grad():
                model(t)

        def e2e(img):
            t0 = time.perf_counter_ns()
            with torch.no_grad():
                x = to_input(img, device)
                t1 = time.perf_counter_ns()
                model(x)
                f = grab["f"][0].float().cpu().numpy()                                                       # on GPU the copy back also waits for the kernels
                t2 = time.perf_counter_ns()
            res = layer(f / np.linalg.norm(f.astype(np.float64)))
            parts.append((t1 - t0, t2 - t1, time.perf_counter_ns() - t2))
            return res
        return tens, fwd, e2e, parts

    def spread(v):
        v = [float(x) for x in v]
        return max(v) / min(v) if min(v) > 0 else float("inf")

    def measure_row(label, device, threads, sync, w0, w, t):
        """3 rounds, each: forward block then end-to-end block (the same model, same inputs). Per-round numbers kept; pooled numbers over all rounds; canary before and after every round."""
        if threads is not None:
            torch.set_num_threads(threads)
            assert torch.get_num_threads() == threads, f"torch did not accept {threads} threads"
        model = build(device)
        tens, fwd, e2e, parts = make_fns(model, device)
        canaries, rounds, fwd_all, e2e_all, bd_all = [canary_ms()], [], [], [], []
        for r in range(g("bb_rounds")):
            fw = time_calls(fwd, tens, w0 if r == 0 else w, t, sync=sync)
            parts.clear()
            ee = time_calls(e2e, imgs, w0 if r == 0 else w, t, sync=sync)
            bd = np.array(parts[-t:], dtype=np.float64) / 1e6
            canaries.append(canary_ms())
            rounds.append({"backbone_forward": summarize(fw), "end_to_end_image_to_decision": summarize(ee),
                           "breakdown_median_ms": {k: float(np.median(bd[:, i])) for i, k in enumerate(("preprocess", "backbone_and_feature_copy", "decision_layer"))}})
            fwd_all.append(fw), e2e_all.append(ee), bd_all.append(bd)
        e2e_med = [x["end_to_end_image_to_decision"]["median_ms"] for x in rounds]
        fwd_med = [x["backbone_forward"]["median_ms"] for x in rounds]
        sp = {"end_to_end_round_medians": spread(e2e_med), "forward_round_medians": spread(fwd_med), "canary": spread(canaries)}
        bd_pool = np.concatenate(bd_all)
        row = {"threads": threads if threads is not None else None,
               "threads_confirmed": torch.get_num_threads() if device.type == "cpu" else None,
               "canary_ms": canaries, "round_spread": sp, "stable": bool(max(sp.values()) <= ratio),
               "rounds": rounds,
               "backbone_forward": summarize(np.concatenate(fwd_all)), "end_to_end_image_to_decision": summarize(np.concatenate(e2e_all)),
               "end_to_end_breakdown_median_ms": {k: float(np.median(bd_pool[:, i])) for i, k in enumerate(("preprocess", "backbone_and_feature_copy", "decision_layer"))},
               "end_to_end_round_medians_ms": e2e_med, "forward_round_medians_ms": fwd_med}
        result["rows"][label] = row
        a, b = row["backbone_forward"], row["end_to_end_image_to_decision"]
        print(f"{label}: forward median {a['median_ms']:.2f} p95 {a['p95_ms']:.2f} ms | end-to-end median {b['median_ms']:.2f} ms | rounds (e2e median) "
              f"{', '.join(f'{x:.1f}' for x in e2e_med)} | canary {', '.join(f'{x:.1f}' for x in canaries)} ms | {'stable' if row['stable'] else 'NOT STABLE'}", flush=True)
        return row

    cw0, cw, ct = g("bb_cpu_warmup"), g("bb_cpu_round_warmup"), g("bb_cpu_timed")
    cpu = torch.device("cpu")
    measure_row("cpu_1_thread", cpu, 1, None, cw0, cw, ct)
    if n_all > 1:
        measure_row(f"cpu_{n_all}_threads", cpu, n_all, None, cw0, cw, ct)
    if torch.cuda.is_available() and device_arg != "cpu":
        torch.set_num_threads(n_all)
        measure_row("gpu", torch.device("cuda"), None, torch.cuda.synchronize, g("bb_gpu_warmup"), g("bb_gpu_round_warmup"), g("bb_gpu_timed"))
    key = f"cpu_{n_all}_threads" if n_all > 1 else "cpu_1_thread"
    row = result["rows"][key]
    result["earlier_estimate_under_50ms_end_to_end_cpu"] = {**estimate_verdict(row["end_to_end_round_medians_ms"], row["stable"], c8["e2e_cpu_ms_estimate"]),
                                                            "row": key, "threads": n_all if n_all > 1 else 1}
    out.mkdir(parents=True, exist_ok=True)
    result["code_sha256"] = sha256_file(Path(__file__))
    dst = out / "latency_backbone.json"
    if dst.exists():                                                                                  # all runs are kept: an earlier result is renamed, never overwritten
        dst.rename(out / f"latency_backbone_earlier_{time.strftime('%Y%m%d_%H%M%S', time.localtime(dst.stat().st_mtime))}.json")
    dst.write_text(json.dumps(result, indent=2))
    report_backbone(result)
    return result


def report_backbone(r):
    h = r["hardware"]
    print("\n=== Stage 8: backbone (ResNet-18, batch 1, no threshold) ===")
    print(f"CPU {h['cpu']} | torch {h['torch']} (default threads {h['torch_default_threads']}; 'all threads' row = {r['all_threads_row']['threads']}) | GPU {h['gpu']} | input {r['input']}")
    print(f"thread env in this shell: {h['thread_env']}")
    print("  row                    forward median / p95 (ms)     end-to-end image->decision median / p95 (ms)   stable")
    for k, v in r["rows"].items():
        a, b = v["backbone_forward"], v["end_to_end_image_to_decision"]
        print(f"  {k:<22} {a['median_ms']:>8.2f} / {a['p95_ms']:<8.2f}            {b['median_ms']:>8.2f} / {b['p95_ms']:<8.2f}        {'yes' if v['stable'] else 'NO'}")
        d = v["end_to_end_breakdown_median_ms"]
        print(f"  {'':<22} end-to-end parts (median ms): preprocess {d['preprocess']:.2f} | backbone + feature copy {d['backbone_and_feature_copy']:.2f} | decision layer {d['decision_layer']:.3f}")
        print(f"  {'':<22} round medians: forward {', '.join(f'{x:.2f}' for x in v['forward_round_medians_ms'])} | end-to-end {', '.join(f'{x:.2f}' for x in v['end_to_end_round_medians_ms'])} | "
              f"canary {', '.join(f'{x:.1f}' for x in v['canary_ms'])} ms | spreads {', '.join(f'{k}={x:.2f}' for k, x in v['round_spread'].items())}")
    e = r["earlier_estimate_under_50ms_end_to_end_cpu"]
    verdict = "NOT RELIABLE: the row is not stable (see spreads above); rerun with the machine plugged in and idle" if e["holds"] is None else ("holds" if e["holds"] else "DOES NOT HOLD")
    print(f"\nearlier estimate 'under 0.05 s per photo end to end on a normal CPU': {verdict} ({e['median_ms_all_threads']:.1f} ms = median of round medians, {e['threads']} threads, row {e['row']})")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--config", default="configs/poc.yaml")
    ap.add_argument("--backbone", action="store_true", help="time the backbone (needs torch); default is the numpy-only decision layer")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu"], help="with --backbone: 'cpu' skips the GPU row")
    ap.add_argument("--out", help="default <run>/stage8")
    ap.add_argument("--quick", action="store_true", help="tiny call counts, for a smoke check only (results are NOT the Stage 8 numbers)")
    a = ap.parse_args(argv)
    cfg_all = yaml.safe_load(open(a.config))
    c8 = {**C8, **cfg_all.get("stage8", {})}
    out = Path(a.out) if a.out else Path(a.run) / "stage8"
    quick = {"warmup": 20, "timed": 200, "repeats": 2, "bb_rounds": 2, "bb_cpu_warmup": 2, "bb_cpu_round_warmup": 1, "bb_cpu_timed": 10, "bb_gpu_warmup": 2, "bb_gpu_round_warmup": 1, "bb_gpu_timed": 10, "bb_stable_ratio": 1e9, "req_p95_ms": c8["req_p95_ms"]} if a.quick else None
    if a.backbone:
        return run_backbone(a.run, cfg_all, c8, a.device, out, quick)
    return run_decision(a.run, cfg_all, c8, out, quick)


if __name__ == "__main__":
    main()
