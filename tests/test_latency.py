"""Stage 8: the deployed decision layer equals the certified arithmetic (checked against an independent route), timing helpers, hash guards, torch-free process."""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import latency as L  # noqa: E402
from src.util import sha256_file  # noqa: E402

K, P = 43, 512


def _head(seed=0):
    rng = np.random.default_rng(seed)
    mu = rng.normal(size=(K, P))
    mu /= np.linalg.norm(mu, axis=1, keepdims=True)
    A = rng.normal(size=(P, P)) / np.sqrt(P)
    cov = 4e-4 * (A @ A.T) + 1e-4 * np.eye(P)
    ev, V = np.linalg.eigh(cov)
    W = (V * ev ** -0.5) @ V.T                                   # whitening (as Stage 3)
    prec = (V / ev) @ V.T                                        # precision (the independent route)
    pri = rng.random(K) + 0.5
    pri /= pri.sum()
    le = -rng.random((K, K)) * 30
    le = (le + le.T) / 2
    np.fill_diagonal(le, np.nan)
    return {"mu": mu, "W": W, "prec": prec, "priors": pri}, le


def _emb(head, n, seed=1, noise=0.012):
    rng = np.random.default_rng(seed)
    c = rng.integers(0, K, n)
    x = head["mu"][c] + rng.normal(size=(n, P)) * noise
    return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)


def _reference(head, le, X):
    """Stage 3 quantities by a different route: squared distances with the PRECISION matrix, plain sorting."""
    X = X.astype(np.float64)
    D2 = np.stack([(((X - m) @ head["prec"]) * (X - m)).sum(1) for m in head["mu"]], 1)
    lp = np.log(head["priors"])[None, :] - 0.5 * D2
    lpn = lp - np.log(np.exp(lp - lp.max(1, keepdims=True)).sum(1, keepdims=True)) - lp.max(1, keepdims=True)
    top = np.argsort(-lp, axis=1)[:, :2]
    marg = lpn[np.arange(len(X)), top[:, 0]] - lpn[np.arange(len(X)), top[:, 1]]
    return {"i1": top[:, 0], "i2": top[:, 1], "margin": marg, "s": np.sqrt(D2.min(1)), "log_eps": le[top[:, 0], top[:, 1]],
            "post1": np.exp(lpn[np.arange(len(X)), top[:, 0]])}


def test_decision_layer_matches_an_independent_reference():
    head, le = _head()
    X = _emb(head, 300)
    ref = _reference(head, le, X)
    tau, c_star = float(np.median(ref["s"])), float(np.median(ref["log_eps"]))
    tE, tH = float(np.median(ref["margin"])) * 0.8, float(np.median(ref["margin"])) * 1.2
    layer = L.DecisionLayer(head, le, tau, c_star, tE, tH)
    out = [layer(x) for x in X]
    assert (np.array([o[1] for o in out]) == ref["i1"]).all() and (np.array([o[2] for o in out]) == ref["i2"]).all()
    assert np.allclose([o[3] for o in out], ref["margin"], rtol=1e-8, atol=1e-8)
    assert np.allclose([o[5] for o in out], ref["s"], rtol=1e-8, atol=1e-8)
    assert np.allclose([o[4] for o in out], ref["log_eps"]) and np.allclose([o[6] for o in out], ref["post1"], atol=1e-10)
    hard = ref["log_eps"] > c_star
    exp_accept = (ref["s"] <= tau) & (ref["margin"] >= np.where(hard, tH, tE))
    assert (np.array([o[0] for o in out]) == exp_accept).all() and (np.array([o[7] for o in out]) == hard).all()
    assert 0.05 < exp_accept.mean() < 0.95                                  # the test exercises both outcomes


def test_gate_and_threshold_edges_and_the_group_specific_threshold():
    head, le = _head(2)
    x = _emb(head, 1, seed=5)[0]
    ref = _reference(head, le, x[None])
    s, m, e = float(ref["s"][0]), float(ref["margin"][0]), float(ref["log_eps"][0])
    base = dict(head=head, log_eps=le)
    assert L.DecisionLayer(**base, tau=s + 1e-9, c_star=e + 1, thr_easy=m - 1e-6, thr_hard=1e9)(x)[0] is True           # easy group, margin above its threshold
    assert L.DecisionLayer(**base, tau=s - 1e-6, c_star=e + 1, thr_easy=m - 1e-6, thr_hard=1e9)(x)[0] is False          # gate rejects: s > tau
    assert L.DecisionLayer(**base, tau=s + 1e-9, c_star=e + 1, thr_easy=m + 1e-6, thr_hard=1e9)(x)[0] is False          # margin below the easy threshold
    r = L.DecisionLayer(**base, tau=s + 1e-9, c_star=e - 1, thr_easy=1e9, thr_hard=m - 1e-6)(x)                         # hard group (log_eps > c*): uses thr_hard
    assert r[0] is True and r[7] is True and r[8] is True
    assert L.DecisionLayer(**base, tau=s + 1e-9, c_star=e - 1, thr_easy=-1e9, thr_hard=m + 1e-6)(x)[0] is False


def test_float32_variant_is_smaller_and_agrees_on_labels():
    head, le = _head()
    X = _emb(head, 200)
    a, b = L.DecisionLayer(head, le, 1e9, 0, 0, 0, np.float64), L.DecisionLayer(head, le, 1e9, 0, 0, 0, np.float32)
    assert b.n_bytes < a.n_bytes
    assert np.mean([a(x)[1] == b(x)[1] for x in X]) >= 0.99


def test_summarize_and_time_calls():
    s = L.summarize(np.arange(1, 101) * 1_000_000)                          # 1..100 ms
    assert s["n"] == 100 and abs(s["median_ms"] - 50.5) < 1e-9 and abs(s["p95_ms"] - 95.05) < 1e-9 and s["max_ms"] == 100 and s["min_ms"] == 1
    calls, syncs = [], []
    import gc
    gc.enable()
    ts = L.time_calls(lambda x: calls.append(x), [1, 2, 3], 7, 20, sync=lambda: syncs.append(1))
    assert len(calls) == 27 and calls[:3] == [1, 2, 3] and len(ts) == 20 and (ts >= 0).all() and gc.isenabled()          # warm-up + timed, GC restored
    assert len(syncs) == 1 + 20


# ---------------------------------------------------------------- a fake run in the real file layout
def _fake_run(tmp_path, n=400, with_model=False):
    head, le = _head()
    X = _emb(head, n, seed=3)
    ref = _reference(head, le, X)
    top3 = np.stack([ref["i1"], ref["i2"], (ref["i2"] + 1) % K], 1).astype(np.int16)
    run = tmp_path / "run"
    for d in ("embeddings", "head_emb", "splits"):
        (run / d).mkdir(parents=True)
    np.savez(run / "embeddings" / "ft_train.npz", emb=X, y=np.zeros(n, int))
    cfg = {"embed": {"expected_checkpoint_sha256": "ab" * 32}, "data": {"cache_dir": str(tmp_path / "cache")}}
    np.savez(run / "head_emb" / "head.npz", mu=head["mu"], W=head["W"], prec=head["prec"], priors=head["priors"], log_eps=le)
    np.savez(run / "head_emb" / "scores_train_release.npz", top3=top3[:5])
    np.savez(run / "head_emb" / "scores_ft_train.npz", top3=top3, margin=ref["margin"].astype(np.float32), s=ref["s"].astype(np.float32),
             log_eps12=ref["log_eps"].astype(np.float32), y=np.zeros(n, int))
    np.savez(run / "splits" / "seed0.npz", eps2_easy_margin_grid=np.linspace(0, 6, 20), eps2_hard_margin_grid=np.linspace(0, 9, 20), track_split=np.zeros(3, np.int8))
    tau, cs = float(np.quantile(ref["s"], 0.9)), float(np.median(ref["log_eps"]))
    (run / "splits" / "stage4_summary.json").write_text(json.dumps(
        {"files_sha256": {"seed0.npz": sha256_file(run / "splits" / "seed0.npz")}, "seeds": [{"seed": 0, "gate": {"tau": tau}, "eps2": {"c_star": cs}}]}))
    (run / "head_emb" / "head_summary.json").write_text(json.dumps(
        {"files_sha256": {f: sha256_file(run / "head_emb" / f) for f in ("head.npz", "scores_train_release.npz", "scores_ft_train.npz")}}))
    if with_model:
        torch = pytest.importorskip("torch")
        from src import train as TR
        size = 32
        torch.manual_seed(0)
        m = TR.build_model(pretrained=False)
        torch.save({"model": m.state_dict()}, run / "best.pt")
        cfg["embed"]["expected_checkpoint_sha256"] = sha256_file(run / "best.pt")
        (run / "config_used.yaml").write_text(yaml.safe_dump({"image_size": size}))
        (tmp_path / "cache").mkdir()
        np.save(tmp_path / "cache" / f"train_{size}.npy", np.random.default_rng(0).integers(0, 255, (30, size, size, 3), dtype=np.uint8))
    (run / "embeddings" / "embeddings_summary.json").write_text(json.dumps(
        {"checkpoint_sha256": cfg["embed"]["expected_checkpoint_sha256"], "files_sha256": {"ft_train.npz": sha256_file(run / "embeddings" / "ft_train.npz")}}))
    cfgp = tmp_path / "c.yaml"
    cfgp.write_text(yaml.safe_dump(cfg))
    return run, cfgp, cfg, ref


def test_decision_run_end_to_end_quick(tmp_path):
    run, cfgp, cfg, _ = _fake_run(tmp_path)
    r = L.main(["--run", str(run), "--config", str(cfgp), "--quick", "--out", str(tmp_path / "o")])
    assert (tmp_path / "o" / "latency_decision.json").exists()
    v64, v32 = r["variants"]["float64"], r["variants"]["float32"]
    assert v64["reproduces_stage3"]["top1_agreement"] == 1.0 and v64["reproduces_stage3"]["n"] == 400
    assert v64["reproduces_stage3"]["decision_agreement_with_offline_rule"] >= 0.9995
    assert len(v64["all_blocks"]) == 2 and v64["primary_block"] == v64["all_blocks"][0] and v32["constant_bytes"] < v64["constant_bytes"]
    assert v64["primary_block"]["n"] == 200 and r["distinct_embeddings"] == 200
    assert r["hardware"]["cpu"] and r["decision_inputs"]["tau"] > 0
    assert set(r["decision_inputs"]["files_sha256"]) == {"head.npz", "seed0.npz"}


def test_guards_refuse_changed_files_wrong_checkpoint_and_a_wrong_decision_layer(tmp_path):
    run, cfgp, cfg, ref = _fake_run(tmp_path)
    cfg2 = json.loads(json.dumps(cfg)); cfg2["embed"]["expected_checkpoint_sha256"] = "cd" * 32
    (tmp_path / "c2.yaml").write_text(yaml.safe_dump(cfg2))
    with pytest.raises(AssertionError, match="frozen checkpoint"):
        L.main(["--run", str(run), "--config", str(tmp_path / "c2.yaml"), "--quick", "--out", str(tmp_path / "o")])
    np.savez(run / "head_emb" / "head.npz", **{k: v for k, v in np.load(run / "head_emb" / "head.npz").items()}, extra=np.zeros(1))        # file changed after Stage 3
    with pytest.raises(AssertionError, match="changed since Stage 3"):
        L.main(["--run", str(run), "--config", str(cfgp), "--quick", "--out", str(tmp_path / "o")])
    # a stored Stage 3 score that the deployed function does not reproduce (hashes updated so only the reproduction check can catch it)
    (tmp_path / "b").mkdir()
    run2, cfgp2, _, _ = _fake_run(tmp_path / "b")
    sc = dict(np.load(run2 / "head_emb" / "scores_ft_train.npz"))
    sc["top3"] = sc["top3"].copy(); sc["top3"][:5, 0] = (sc["top3"][:5, 0] + 1) % K
    np.savez(run2 / "head_emb" / "scores_ft_train.npz", **sc)
    hs = json.loads((run2 / "head_emb" / "head_summary.json").read_text())
    hs["files_sha256"]["scores_ft_train.npz"] = sha256_file(run2 / "head_emb" / "scores_ft_train.npz")
    (run2 / "head_emb" / "head_summary.json").write_text(json.dumps(hs))
    with pytest.raises(AssertionError, match="disagrees with Stage 3 labels"):
        L.main(["--run", str(run2), "--config", str(cfgp2), "--quick", "--out", str(tmp_path / "o2")])


def test_decision_mode_is_single_threaded_and_never_imports_torch(tmp_path):
    run, cfgp, cfg, _ = _fake_run(tmp_path)
    code = ("import sys, os; sys.argv=['x']; from src import latency as L; "
            f"L.main(['--run', r'{run}', '--config', r'{cfgp}', '--quick', '--out', r'{tmp_path / 'o'}']); "
            "print('OMP', os.environ['OMP_NUM_THREADS'], 'OPENBLAS', os.environ['OPENBLAS_NUM_THREADS'], 'torch_loaded', 'torch' in sys.modules)")
    clean = {k: v for k, v in os.environ.items() if k not in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}
    r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=clean, capture_output=True, text=True)
    assert r.returncode == 0 and "OMP 1 OPENBLAS 1 torch_loaded False" in r.stdout, (r.stdout[-600:], r.stderr[-800:])
    # and with torch blocked the way Windows blocks it
    (tmp_path / "blk" / "torch").mkdir(parents=True)
    (tmp_path / "blk" / "torch" / "__init__.py").write_text("raise OSError('[WinError 4551] Application Control policy has blocked this file')")
    env = {**clean, "PYTHONPATH": os.pathsep.join([str(tmp_path / "blk"), str(ROOT)])}
    r = subprocess.run([sys.executable, "-m", "src.latency", "--run", str(run), "--config", str(cfgp), "--quick", "--out", str(tmp_path / "o2")], cwd=ROOT, env=env, capture_output=True, text=True)
    assert r.returncode == 0 and "PRIMARY block" in r.stdout, (r.stdout[-600:], r.stderr[-800:])


def test_backbone_timing_end_to_end_on_a_tiny_fake_run(tmp_path):
    try:
        import torch  # noqa: F401
        import torchvision  # noqa: F401
    except (ImportError, OSError) as e:
        pytest.skip(f"torch cannot be loaded here ({type(e).__name__})")
    run, cfgp, cfg, _ = _fake_run(tmp_path, with_model=True)
    r = L.main(["--run", str(run), "--config", str(cfgp), "--backbone", "--device", "cpu", "--quick", "--out", str(tmp_path / "o")])
    assert (tmp_path / "o" / "latency_backbone.json").exists() and "cpu_1_thread" in r["rows"]
    row = r["rows"]["cpu_1_thread"]
    assert row["backbone_forward"]["n"] == 20 and row["end_to_end_image_to_decision"]["median_ms"] >= row["backbone_forward"]["median_ms"] * 0.5
    assert "gpu" not in r["rows"] and r["input"].startswith("1 x 3 x 32 x 32")
    assert len(row["rounds"]) == 2 and len(row["canary_ms"]) == 3 and row["threads"] == 1 and row["threads_confirmed"] == 1 and row["stable"] is True
    assert set(row["round_spread"]) == {"end_to_end_round_medians", "forward_round_medians", "canary"} and row["backbone_forward"]["n"] == 20
    n_all = r["all_threads_row"]["threads"]
    if n_all > 1:
        assert r["rows"][f"cpu_{n_all}_threads"]["threads_confirmed"] == n_all      # the all-threads row really ran with that many threads
    assert r["earlier_estimate_under_50ms_end_to_end_cpu"]["holds"] in (True, False) and "round_medians_ms" in r["earlier_estimate_under_50ms_end_to_end_cpu"]
    bd = row["end_to_end_breakdown_median_ms"]
    assert set(bd) == {"preprocess", "backbone_and_feature_copy", "decision_layer"} and all(v >= 0 for v in bd.values())
    assert abs(sum(bd.values()) - row["end_to_end_image_to_decision"]["median_ms"]) < 0.5 * row["end_to_end_image_to_decision"]["median_ms"] + 1.0
    first = (tmp_path / "o" / "latency_backbone.json").read_bytes()                  # a second run into the same folder must keep the first result, not overwrite it
    L.main(["--run", str(run), "--config", str(cfgp), "--backbone", "--device", "cpu", "--quick", "--out", str(tmp_path / "o")])
    kept = list((tmp_path / "o").glob("latency_backbone_earlier_*.json"))
    assert len(kept) == 1 and kept[0].read_bytes() == first and (tmp_path / "o" / "latency_backbone.json").read_bytes() != b""
    cfg["embed"]["expected_checkpoint_sha256"] = "cd" * 32
    (tmp_path / "c2.yaml").write_text(yaml.safe_dump(cfg))
    with pytest.raises(AssertionError, match="CHECKPOINT HASH MISMATCH"):
        L.main(["--run", str(run), "--config", str(tmp_path / "c2.yaml"), "--backbone", "--quick", "--out", str(tmp_path / "o3")])


def test_backbone_mode_pins_numpy_blas_only_and_leaves_torch_threads_alone():
    """Regression for patch 1: MKL_NUM_THREADS / OMP_NUM_THREADS are read by torch as its default thread count, so they must NOT be set in backbone mode."""
    code = "import sys, os; sys.argv=['x', '--backbone']; import src.latency; print('BLAS', os.environ.get('OPENBLAS_NUM_THREADS'), os.environ.get('MKL_NUM_THREADS'), 'OMP', os.environ.get('OMP_NUM_THREADS'))"
    clean = {k: v for k, v in os.environ.items() if k not in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}
    r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=clean, capture_output=True, text=True)
    assert r.returncode == 0 and "BLAS 1 None OMP None" in r.stdout, (r.stdout, r.stderr[-500:])


def test_torch_default_thread_count_is_the_same_with_and_without_importing_latency_in_backbone_mode():
    try:
        import torch  # noqa: F401
    except (ImportError, OSError) as e:
        pytest.skip(f"torch cannot be loaded here ({type(e).__name__})")
    clean = {k: v for k, v in os.environ.items() if k not in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}
    plain = subprocess.run([sys.executable, "-c", "import torch; print(torch.get_num_threads())"], cwd=ROOT, env=clean, capture_output=True, text=True)
    via = subprocess.run([sys.executable, "-c", "import sys; sys.argv=['x','--backbone']; import src.latency, torch; print(torch.get_num_threads())"], cwd=ROOT, env=clean, capture_output=True, text=True)
    assert plain.returncode == 0 and via.returncode == 0 and plain.stdout.strip() == via.stdout.strip(), (plain.stdout, plain.stderr[-300:], via.stdout, via.stderr[-300:])


def test_all_threads_row_is_never_silently_a_one_thread_row():
    assert L.all_threads(14, 20) == (14, False)
    assert L.all_threads(1, 20) == (10, True)            # something clamped torch on a multi-core machine: fall back and say so
    assert L.all_threads(1, 1) == (1, False)             # a genuinely single-CPU machine
    assert L.all_threads(1, 2) == (2, True)


def test_estimate_verdict_is_withheld_when_the_row_is_not_stable():
    ok = L.estimate_verdict([40.0, 44.0, 42.0], True, 50.0)
    assert ok["holds"] is True and ok["median_ms_all_threads"] == 42.0
    assert L.estimate_verdict([60.0, 55.0, 58.0], True, 50.0)["holds"] is False
    bad = L.estimate_verdict([25.0, 25.0, 102.0], False, 50.0)
    assert bad["holds"] is None and bad["round_medians_ms"] == [25.0, 25.0, 102.0]


def test_canary_is_a_positive_time_and_repeatable_to_within_a_factor_of_two():
    a, b = L.canary_ms(3), L.canary_ms(3)
    assert a > 0 and b > 0 and 0.5 < a / b < 2.0
