"""Tests for POC-2 Stage 3 (engine, 3a synthetic, 3b resampling, B1 design) on synthetic data. No real GTSRB is read.
Brute-force references are plain loops written from LOG_r8_stage3_entry.md Part A, independently of the vectorised code."""
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src import certify as C  # noqa: E402
from src import r8_frozen as FZ  # noqa: E402
from src import r8_stage1 as R1  # noqa: E402
from src import r8_stage2 as S2  # noqa: E402
from src import r8_stage3_b1design as B1D  # noqa: E402
from src import r8_stage3_engine as E  # noqa: E402
from src import r8_stage3_resample as RS  # noqa: E402
from src import r8_stage3_synth as SY  # noqa: E402
from src.util import sha256_file  # noqa: E402
import test_r8_stage1 as T1  # noqa: E402

rng = np.random.default_rng(11)


# ------------------------------------------------------------------ brute-force references
def brute_walk(p, order, delta):
    last = -1
    for j in order:
        if p[j] > delta:
            break
        last = j
    return last


def brute_track_mu(committed, wrong, counts, alpha):
    """Expand the draw into explicit units (a track drawn twice appears twice) and average L_i."""
    units = [i for i, k in enumerate(counts) for _ in range(int(k))]
    return np.array([np.mean([float(wrong[i, j] and committed[i, j]) - alpha * float(committed[i, j]) + alpha for i in units])
                     for j in range(committed.shape[1])]), len(units)


def brute_frame_mu(acc_frames, wrong_frames, counts, alpha):
    """acc_frames[i]: (frames_i, G) booleans of track i. Every frame of every drawn copy is one unit."""
    L = []
    for i, k in enumerate(counts):
        for _ in range(int(k)):
            for f in range(acc_frames[i].shape[0]):
                L.append([float(wrong_frames[i][f, j]) - alpha * float(acc_frames[i][f, j]) + alpha
                          for j in range(acc_frames[i].shape[1])])
    L = np.array(L)
    return L.mean(0), len(L)


# ------------------------------------------------------------------ engine
def test_walk_rows_matches_brute_force():
    for _ in range(200):
        G = int(rng.integers(1, 12))
        P = rng.random((5, G)) * 0.3
        order = list(rng.permutation(G))
        got = E.walk_rows(P, order, 0.1)
        assert list(got) == [brute_walk(P[i], order, 0.1) for i in range(5)]


def test_track_and_frame_mu_match_brute_force_with_repeated_draws():
    T, G, a = 15, 6, 0.02
    nf = rng.integers(1, 6, T)
    acc_frames = [rng.random((n, G)) < np.linspace(0.9, 0.3, G) for n in nf]
    wrong_frames = [acc & (rng.random(acc.shape) < 0.2) for acc in acc_frames]
    committed = rng.random((T, G)) < 0.7
    wrong = committed & (rng.random((T, G)) < 0.15)
    A = np.array([x.sum(0) for x in acc_frames])
    W = np.array([x.sum(0) for x in wrong_frames])
    for _ in range(10):
        counts = np.bincount(rng.integers(0, T, 20), minlength=T)
        mu, n = E.track_mu(committed, wrong, counts[None, :], a)
        mb, nb = brute_track_mu(committed, wrong, counts, a)
        assert n[0, 0] == nb and np.allclose(mu[0], mb, atol=1e-14)
        mu, n = E.frame_mu(A, W, nf, counts[None, :], a)
        mb, nb = brute_frame_mu(acc_frames, wrong_frames, counts, a)
        assert n[0, 0] == nb and np.allclose(mu[0], mb, atol=1e-14)


def test_b1_equals_track_level_when_each_track_has_one_frame():
    """With one frame per track, 'every frame its own unit' is the same as 'every track its own unit'. Both must also
    equal the POC-1 engine (certify.certify_sequence)."""
    T, a, delta = 400, 0.05, 0.1
    margin = rng.exponential(5.0, T)
    wrong_f = rng.random(T) < np.exp(-margin / 2) * 0.4                 # low margins are wrong more often
    grid = np.concatenate([[0.0], np.quantile(margin, np.linspace(0.05, 0.95, 15))])
    acc = margin[:, None] >= grid[None, :]
    order = list(range(len(grid) - 1, -1, -1))
    mu_t, n_t = E.track_mu(acc, acc & wrong_f[:, None], np.ones((1, T)), a)
    mu_b, n_b = E.frame_mu(acc.astype(int), (acc & wrong_f[:, None]).astype(int), np.ones(T), np.ones((1, T)), a)
    assert n_t[0, 0] == n_b[0, 0] == T and np.allclose(mu_t, mu_b, atol=1e-15)
    ct, cb = E.certify(mu_t, n_t, a, order, delta)[0], E.certify(mu_b, n_b, a, order, delta)[0]
    ref = C.certify_sequence(np.arange(T), margin, wrong_f, grid, order, a, delta)
    assert ct == cb == (ref["grid_index"] if ref["certified"] else -1)


def test_violation_rule():
    mu_true = np.array([0.02, 0.01, 0.0099, 0.0101])
    assert list(E.violations([-1, 0, 1, 2, 3], mu_true, 0.01)) == [False, True, False, False, True]


def test_hand_pvalues_exact_and_cert_b_maxp():
    h = E.check_hand()
    assert h["pass"], h
    assert [c["min_term"] for c in h["cases"]] == ["hoeffding", "hoeffding", "bentkus"]
    assert E.check_cert_b()["pass"]


def test_cert_b_maxp_only_lateness_broken_does_not_certify():
    p = E.cert_b_pvalue(0.0005, np.array([0.03, 0.10, 0.20]), 784, 0.01, 0.10)
    assert p[0] <= 0.1 and p[1] > 0.1 and p[2] > 0.1


# ------------------------------------------------------------------ violation counting end to end (brute force)
def test_resample_violation_counts_match_brute_force():
    T, Gt, Gb, delta = 40, 6, 5, 0.1
    committed = rng.random((T, Gt)) < np.linspace(1.0, 0.5, Gt)
    wrong = committed & (rng.random((T, Gt)) < np.linspace(0.06, 0.0, Gt))
    nf = rng.integers(2, 6, T)
    acc_frames = [rng.random((n, Gb)) < np.linspace(1.0, 0.6, Gb) for n in nf]
    wrong_frames = [x & (rng.random(x.shape) < np.linspace(0.08, 0.005, Gb)) for x in acc_frames]
    P = {"committed": committed, "wrong": wrong, "A": np.array([x.sum(0) for x in acc_frames]),
         "W": np.array([x.sum(0) for x in wrong_frames]), "NF": nf.astype(float), "n_tracks": T}
    cfg = {"certificate_A": {"threshold_order": {"order_positions": [5, 4, 3, 2, 1, 0]}}}
    b1 = {"order_positions": [4, 3, 2, 1, 0]}
    Wd = RS.draw_weights(T, 30, 150, 3)
    for a in (0.02, 0.05):
        res = RS.resample(P, cfg, b1, Wd, [a], delta)[a]
        true_t = brute_track_mu(committed, wrong, np.ones(T), a)[0]
        true_b = brute_frame_mu(acc_frames, wrong_frames, np.ones(T), a)[0]
        vt = vb = 0
        for counts in Wd:
            mu, n = brute_track_mu(committed, wrong, counts, a)
            j = brute_walk([float(C.hb_pvalue(m, n, a)) for m in mu], [5, 4, 3, 2, 1, 0], delta)
            vt += int(j >= 0 and true_t[j] > a + 1e-12)
            mu, n = brute_frame_mu(acc_frames, wrong_frames, counts, a)
            j = brute_walk([float(C.hb_pvalue(m, n, a)) for m in mu], [4, 3, 2, 1, 0], delta)
            vb += int(j >= 0 and true_b[j] > a + 1e-12)
        assert res["track"]["violations"] == vt and res["B1"]["violations"] == vb


# ------------------------------------------------------------------ 3a synthetic model
def test_synthetic_tables_match_brute_force():
    X = SY.sample(25, SY.solve_r(0.3), rng)
    grid = np.concatenate([[0.0], np.geomspace(0.5, 40, 12)])
    com, wr = SY.track_table(X, grid)
    A, W = SY.frame_counts(X, grid)
    for i in range(25):
        S = np.cumsum(X[i])
        for j, lam in enumerate(grid):
            t = next((t for t in range(30) if abs(S[t]) >= lam), None)
            assert com[i, j] == (t is not None) and wr[i, j] == (t is not None and S[t] < 0)
            assert A[i, j] == sum(abs(x) >= lam for x in X[i]) and W[i, j] == sum(abs(x) >= lam and x < 0 for x in X[i])


@pytest.mark.parametrize("rho", [0.0, 0.3, 0.45])
def test_calibration_gives_frame_error_and_wrong_icc(rho):
    r = SY.solve_r(rho)
    assert abs(SY.icc_of_wrong(r) - rho) < 1e-6
    g = np.array([0.0, 1.0])
    _, _, cal = SY.population_risk(r, g, g, 0.01, np.random.default_rng(5), n_pop=40_000)
    assert abs(cal["frame_error"] - 0.074) < 0.004 and abs(cal["wrong_icc"] - rho) < 0.03


def test_synthetic_population_risk_matches_direct_count():
    r = SY.solve_r(0.45)
    X = SY.sample(3000, r, np.random.default_rng(9))
    gt, gb = np.array([0.0, 3.0, 10.0]), np.array([0.0, 1.0, 2.5])
    mu_t, mu_b, _ = SY.population_risk(r, gt, gb, 0.01, np.random.default_rng(9), n_pop=3000)
    c, w = SY.track_table(X, gt)
    assert np.allclose(mu_t, 0.01 * (1 - c.mean(0)) + w.mean(0))
    A, W = SY.frame_counts(X, gb)
    assert np.allclose(mu_b, (W.sum(0) + 0.01 * (X.size - A.sum(0))) / X.size)


def test_run_rho_small_is_well_formed():
    row = SY.run_rho(0.45, 2, 0.01, 0.1, runs=8, n_pop=3000)
    for m in ("track", "B1"):
        assert row[m]["runs"] == 8 and 0 <= row[m]["violation_rate"] <= 1


# ------------------------------------------------------------------ freeze guard and fake-run end to end (3b)
def fake_root(tmp, cfg):
    root = tmp / "root"
    (root / "configs").mkdir(parents=True)
    (root / "docs").mkdir()
    (root / "configs" / "frozen_r8.yaml").write_text(yaml.safe_dump(cfg))
    (root / "docs" / "poc2-implementation-plan-r8.md").write_text("plan\n")
    return {"root": root, "frozen_sha": sha256_file(root / "configs" / "frozen_r8.yaml"),
            "plan_sha": sha256_file(root / "docs" / "poc2-implementation-plan-r8.md")}


def test_load_frozen_refuses_changed_files(tmp_path):
    args = fake_root(tmp_path, {"r8_frozen": True})
    assert FZ.load_frozen(**args)["r8_frozen"] is True
    (args["root"] / "docs" / "poc2-implementation-plan-r8.md").write_text("plan edited\n")
    with pytest.raises(AssertionError, match="plan file differs"):
        FZ.load_frozen(**args)
    with pytest.raises(AssertionError, match="frozen_r8.yaml differs"):
        FZ.load_frozen(args["root"], "0" * 64, args["plan_sha"])


def test_real_freeze_guard_accepts_the_committed_files():
    assert FZ.load_frozen()["commit_rule"]["variant"] == "V1"


def make_3b(tmp, monkeypatch):
    monkeypatch.setattr(R1, "N_GRID", 40)
    orig = R1.run_checks

    def rc(inp, B):
        res, ok = orig(inp, B)
        ok["C6_design_set_is_261_tracks"] = int(B["design"].sum()) == 6
        return res, ok
    monkeypatch.setattr(R1, "run_checks", rc)
    run, cfgp, n = T1.fake_run(tmp)
    assert R1.main(["--run", str(run), "--config", str(cfgp)])["all_pass"]
    grid = np.load(run / "poc2" / "stage1" / "track_table_pool.npz")["V1_grid"]
    idx = list(range(0, 10))
    cfg = {"r8_frozen": True, "splits": {"pool_tracks": n - 6, "cert_tracks": n - 6},
           "fixed_in_every_phase": {"delta": 0.1}, "certificate_B": {"gamma": 0.1},
           "certificate_A": {"primary_alpha": 0.01, "alpha_grid": [0.01, 0.05],
                             "threshold_order": {"start_position": 9, "order_positions": list(range(9, -1, -1))}},
           "lambda_grid": {"stage1_indices": idx, "values": [float(grid[i]) for i in idx]}}
    args = fake_root(tmp, cfg)
    s3 = run / "poc2" / "stage3"
    s3.mkdir(parents=True)
    (s3 / "b1_design.json").write_text(json.dumps({"work_thresholds": [0.0, 1.0, 5.0, 10.0, 50.0],
                                                  "order_positions": [4, 3, 2, 1, 0], "start_position": 4}))
    monkeypatch.setattr(RS, "FROZEN_ARGS", args)
    monkeypatch.setattr(RS, "B1_DESIGN_SHA256", sha256_file(s3 / "b1_design.json"))
    monkeypatch.setattr(RS, "N_DRAWS", 40)
    return run, n


def test_3b_end_to_end_on_fake_run(tmp_path, monkeypatch):
    run, n = make_3b(tmp_path, monkeypatch)
    s = RS.main(["--run", str(run)])
    assert s["all_pass"], s["checks"]
    assert s["p1"]["complete"] is False                                 # no 3a summary in the fake run
    r = s["results"]["0.01"]
    assert r["track"]["runs"] == 40 and r["B1"]["runs"] == 40
    split = np.load(run / "splits" / "seed0.npz")
    pool = np.load(run / "poc2" / "stage1" / "track_table_pool.npz")
    assert not set(pool["track_names"]) & set(split["track_names"][split["track_split"] == 0])


@pytest.mark.parametrize("rel,msg", [
    ("poc2/stage1/track_table_pool.npz", "track_table_pool.npz changed since Stage 1"),
    ("poc2/stage1/frame_evidence.npz", "frame_evidence.npz changed since Stage 1"),
    ("embeddings/train_release.npz", "train_release.npz changed since Stage 1"),
    ("poc2/stage3/b1_design.json", "b1_design.json differs"),
])
def test_3b_refuses_tampered_inputs(tmp_path, monkeypatch, rel, msg):
    run, _ = make_3b(tmp_path, monkeypatch)
    p = run / rel
    p.write_bytes(p.read_bytes() + b" ")
    with pytest.raises(AssertionError, match=msg):
        RS.main(["--run", str(run)])


def test_3b_refuses_changed_frozen_file(tmp_path, monkeypatch):
    run, _ = make_3b(tmp_path, monkeypatch)
    fz = RS.FROZEN_ARGS["root"] / "configs" / "frozen_r8.yaml"
    fz.write_text(fz.read_text() + "# edited\n")
    with pytest.raises(AssertionError, match="frozen_r8.yaml differs"):
        RS.main(["--run", str(run)])


def test_3a_refuses_changed_frozen_file(tmp_path, monkeypatch):
    args = fake_root(tmp_path, {"r8_frozen": True})
    (args["root"] / "configs" / "frozen_r8.yaml").write_text("r8_frozen: true\n# edited\n")
    monkeypatch.setattr(SY, "FROZEN_ARGS", args)
    with pytest.raises(AssertionError, match="frozen_r8.yaml differs"):
        SY.main(["--run", str(tmp_path)])


def test_b1_design_rules_on_synthetic_frames():
    """b1_design applies Stage 2 rules (h) and (c) to frames: all working points have c >= 70% and e <= alpha."""
    m = np.where(rng.random(5000) < 0.05, -np.inf, rng.exponential(50.0, 5000))
    wrong = rng.random(5000) < 0.08 * np.exp(-np.nan_to_num(m, neginf=0) / 8)
    r = B1D.b1_design(m, wrong, 0.01)
    st = r["stats"]
    assert len(r["work"]) and (st["c"][r["work"]] >= 0.70 - 1e-12).all() and (st["e"][r["work"]] <= 0.01 + 1e-12).all()
    mu = S2.mean_L(st, 0.01)[r["work"]]
    assert mu[r["start"]] == mu.min() and r["order"] == list(range(r["start"], -1, -1))
