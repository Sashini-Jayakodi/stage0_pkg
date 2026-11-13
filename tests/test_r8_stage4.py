"""Tests for POC-2 Stage 4 (src/r8_stage4.py, src/r8_stage4_baselines_design.py) on synthetic data. No real GTSRB.
Brute-force references are plain loops written from LOG_r8_stage4_entry.md Part A and its Amendments 1 and 2."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src import certify as C  # noqa: E402
from src import r8_stage1 as R1  # noqa: E402
from src import r8_stage3_resample as RS  # noqa: E402
from src import r8_stage4 as S4  # noqa: E402
from src import r8_stage4_baselines_design as BD  # noqa: E402
from src.util import sha256_file  # noqa: E402
import test_r8_stage1 as T1  # noqa: E402
import test_r8_stage3 as T3  # noqa: E402

rng = np.random.default_rng(21)


# ------------------------------------------------------------------ re-splits (Part A (b))
def brute_test_counts(n_k, n_test, n_pool):
    """Independent loop: floor quotas capped at n_k - 1, then one place at a time by largest fractional part."""
    keys = sorted(n_k)
    frac = {k: n_k[k] * n_test / n_pool - int(n_k[k] * n_test / n_pool) for k in keys}
    t = {k: min(int(n_k[k] * n_test / n_pool), n_k[k] - 1) for k in keys}
    ranked = sorted(keys, key=lambda k: (-frac[k], k))
    while sum(t.values()) < n_test:
        for k in ranked:
            if sum(t.values()) < n_test and t[k] < n_k[k] - 1:
                t[k] += 1
    return t


@pytest.mark.parametrize("seed", range(6))
def test_test_counts_match_brute_force_and_keep_every_class_in_cert(seed):
    r = np.random.default_rng(seed)
    n_k = {int(k): int(v) for k, v in enumerate(r.integers(1, 40, 43))}
    n_k[3] = 1                                                        # a class with a single pool track
    n_pool = sum(n_k.values())
    n_test = int(round(n_pool * 262 / 1046))
    t = S4.test_counts(n_k, n_test, n_pool)
    assert t == brute_test_counts(n_k, n_test, n_pool)
    assert sum(t.values()) == n_test and all(t[k] <= n_k[k] - 1 for k in n_k) and t[3] == 0


def test_make_split_is_stratified_deterministic_and_disjoint():
    y = np.repeat(np.arange(10), rng.integers(2, 30, 10))
    n_test = int(round(len(y) * 262 / 1046))
    a, b = S4.make_split(y, 0, n_test), S4.make_split(y, 0, n_test)
    assert (a == b).all() and a.sum() == n_test and not (a & ~a).any()
    counts = S4.test_counts({int(k): int((y == k).sum()) for k in np.unique(y)}, n_test, len(y))
    assert all(int(a[y == k].sum()) == counts[int(k)] for k in np.unique(y))
    assert set(np.unique(y[~a])) == set(np.unique(y))                # every class in Cert
    assert not (S4.make_split(y, 1, n_test) == a).all()               # another seed, another split


# ------------------------------------------------------------------ certification and metrics
def tab_from_scores(score, label, y, grid, order, pos=None, size=None):
    com = score[:, None] >= grid[None, :]
    n = len(score)
    pos = np.ones(n, int) if pos is None else pos
    size = np.full(n, 30.0) if size is None else size
    return {"committed": com, "wrong": com & (label != y)[:, None], "pos": np.repeat(pos[:, None], len(grid), 1),
            "size": np.repeat(size[:, None], len(grid), 1), "order": order}


def test_certify_on_equals_poc1_engine():
    n = 500
    score = rng.exponential(10.0, n)
    y = rng.integers(0, 5, n)
    label = np.where(rng.random(n) < 0.3 * np.exp(-score / 3), (y + 1) % 5, y)
    grid = np.concatenate([[0.0], np.quantile(score, np.linspace(0.05, 0.9, 12))])
    order = list(range(len(grid) - 1, -1, -1))
    tab = tab_from_scores(score, label, y, grid, order)
    cert = rng.random(n) < 0.75
    for a in (0.01, 0.03, 0.05):
        ref = C.certify_sequence(np.flatnonzero(cert), score[cert], (label != y)[cert], grid, order, a, 0.1)
        assert S4.certify_on(tab, cert, a, 0.1) == (ref["grid_index"] if ref["certified"] else -1)


def test_test_metrics_match_brute_force():
    n = 60
    score, y = rng.exponential(5.0, n), rng.integers(0, 3, n)
    label = np.where(rng.random(n) < 0.2, (y + 1) % 3, y)
    pos, size = rng.integers(1, 31, n), rng.integers(15, 90, n).astype(float)
    grid = np.array([0.0, 2.0, 5.0])
    tab = tab_from_scores(score, label, y, grid, [2, 1, 0], pos, size)
    test = rng.random(n) < 0.4
    for j in (0, 1, 2):
        m = S4.test_metrics(tab, j, test)
        idx = [i for i in range(n) if test[i] and score[i] >= grid[j]]
        assert m["coverage"] == len(idx) / test.sum() and m["committed"] == len(idx)
        assert m["error_committed"] == pytest.approx(sum(label[i] != y[i] for i in idx) / len(idx))
        assert m["median_commit_frame"] == np.median([pos[i] for i in idx]) and m["median_size"] == np.median([size[i] for i in idx])
    m = S4.test_metrics(tab, -1, test)
    assert m["certified"] is False and m["coverage"] == 0.0 and m["error_committed"] is None


def test_smallest_certified_and_nested_curve():
    n = 800
    score, y = rng.exponential(10.0, n), rng.integers(0, 5, n)
    label = np.where(rng.random(n) < 0.05 * np.exp(-score / 5), (y + 1) % 5, y)
    grid = np.concatenate([[0.0], np.quantile(score, np.linspace(0.05, 0.6, 10))])
    tab = tab_from_scores(score, label, y, grid, list(range(len(grid) - 1, -1, -1)))
    cert = np.ones(n, bool)
    alphas = [0.005, 0.01, 0.02, 0.03, 0.05]
    sm = S4.smallest_certified(tab, cert, alphas, 0.1)
    brute = next((a for a in alphas if S4.certify_on(tab, cert, a, 0.1) >= 0), None)
    assert sm == brute
    cv = S4.curve_vs_n(tab, np.arange(n), 0, alphas, 0.1, n)
    assert set(cv) == set(S4.CURVE_N) and cv[784] == S4.smallest_certified(
        tab, np.isin(np.arange(n), np.arange(n)[np.random.default_rng([S4.CURVE_SALT, 0]).permutation(n)][:784]), alphas, 0.1)


# ------------------------------------------------------------------ pass levels (Part A (f)) and Amendment 2
def seed_row(c_cert, c_cov, c_fr, b_cert, b_cov, a=0.01):
    mk = lambda cert, cov, fr: {"certified": cert, "coverage": cov if cert else 0.0, "committed": 10 if cert and cov > 0 else 0,
                                "median_commit_frame": fr if cert and cov > 0 else None}
    return {"alphas": {a: {"commit": mk(c_cert, c_cov, c_fr), "B2": mk(b_cert, b_cov, 30)}}}


def test_p_checks_follow_part_a_table():
    rows = [seed_row(True, 0.85, 14, True, 0.70),     # both certified: coverage vs coverage
            seed_row(True, 0.85, 14, False, 0.0),     # B2 not certified: P3 passes (0.85 > 0), Part A as written
            seed_row(False, 0.0, None, True, 0.70),   # commit rule not certified: P3 and P4 fail
            seed_row(False, 0.0, None, False, 0.0),   # neither: 0 > 0 is false; P4 fails
            seed_row(True, 0.75, 30, True, 0.80)]     # certified but below c_min, behind B2, median at the last frame
    P = S4.p_checks(rows, 0.01, 0.79272030651341)
    assert P["P2"] == [True, True, False, False, False]
    assert P["P3"] == [True, True, False, False, False]
    assert P["P4"] == [True, True, False, False, False]


def test_descriptive_common_alpha():
    s = {"alphas": {}}
    for a, (cc, bc) in zip((0.005, 0.01, 0.02), ((True, False), (False, True), (True, True))):
        s["alphas"][a] = seed_row(cc, 0.9, 12, bc, 0.8, a)["alphas"][a]
    dc = S4.descriptive_common(s, [0.005, 0.01, 0.02])
    assert dc["alpha"] == 0.02 and dc["p3_holds"] and dc["p4_holds"]
    assert S4.descriptive_common({"alphas": {0.01: seed_row(True, 0.9, 12, False, 0, 0.01)["alphas"][0.01]}}, [0.01]) == "none"


# ------------------------------------------------------------------ baselines (Part A (e), Amendment 1)
def test_b2_picks_only_passing_frames_uniformly_and_rejects_tracks_without_any():
    rows = [np.arange(i * 30, i * 30 + 30) for i in range(400)]
    passed = rng.random(400 * 30) < 0.6
    passed[:30] = False                                               # track 0 has no passing frame
    pos = BD.b2_positions(1307, rows, passed, np.arange(400))
    assert pos[0] == -1 and all(passed[r[p]] for r, p in zip(rows[1:], pos[1:]))
    u = np.random.default_rng(BD.B2_SEED).random(1307)
    for i in range(1, 400):                                           # brute: k-th passing frame, k = floor(u * n_passing)
        ok = [f for f in range(30) if passed[rows[i][f]]]
        assert pos[i] == ok[int(u[i] * len(ok))]


def test_b3_and_b4_rules_brute_force():
    rows = [np.arange(i * 6, i * 6 + 6) for i in range(50)]
    passed = rng.random(300) < 0.7
    top = rng.integers(0, 4, 300)
    m = rng.exponential(5.0, 300)
    B = BD.baseline_scores(rows, m, top, passed, BD.b2_positions(1307, rows, passed, np.arange(50)), K=4)
    for i, r in enumerate(rows):
        last = r[-1]
        assert (B["B3"]["label"][i] == (top[last] if passed[last] else -1)) and B["B3"]["pos"][i] == 6
        votes = [0] * 4
        for f in r:
            if passed[f]:
                votes[top[f]] += 1
        if sum(votes) == 0:
            assert np.isneginf(B["B4"]["score"][i])
            continue
        best = max(votes)
        lab = votes.index(best)                                       # first maximum = smallest class id
        assert B["B4"]["label"][i] == lab and B["B4"]["score"][i] == best - sorted(votes)[-2]


# ------------------------------------------------------------------ end to end on a fake run folder
def make_4(tmp, monkeypatch, n_tracks=40, K=4):
    monkeypatch.setattr(R1, "N_GRID", 40)
    orig = R1.run_checks
    n_design = n_tracks // 2

    def rc(inp, B):
        res, ok = orig(inp, B)
        ok["C6_design_set_is_261_tracks"] = int(B["design"].sum()) == n_design
        return res, ok
    monkeypatch.setattr(R1, "run_checks", rc)
    run, cfgp, _ = T1.fake_run(tmp, n_tracks=n_tracks, K=K)
    assert R1.main(["--run", str(run), "--config", str(cfgp)])["all_pass"]
    grid = np.load(run / "poc2" / "stage1" / "track_table_pool.npz")["V1_grid"]
    n_pool = n_tracks - n_design
    n_test = 5
    cfg = {"r8_frozen": True, "splits": {"pool_tracks": n_pool, "cert_tracks": n_pool - n_test, "test_tracks": n_test},
           "fixed_in_every_phase": {"delta": 0.1}, "certificate_B": {"gamma": 0.1},
           "certificate_A": {"primary_alpha": 0.01, "alpha_grid": [0.01, 0.05],
                             "threshold_order": {"start_position": 9, "order_positions": list(range(9, -1, -1))}},
           "lambda_grid": {"stage1_indices": list(range(10)), "values": [float(grid[i]) for i in range(10)]},
           "pass_levels": {"P2": {"c_min": 0.5, "resplits_required": 4}, "P3": {"resplits_required": 4},
                           "P4": {"resplits_required": 4}}}
    args = T3.fake_root(tmp, cfg)
    s3, s4 = run / "poc2" / "stage3", run / "poc2" / "stage4"
    s3.mkdir(parents=True)
    s4.mkdir(parents=True)
    (s3 / "b1_design.json").write_text(json.dumps({"work_thresholds": [0.0, 5.0], "order_positions": [1, 0], "start_position": 1}))
    base = {b: {"work_thresholds": [0.0, 1.0, 3.0], "order_positions": [2, 1, 0], "start_position": 2} for b in BD.BASELINES}
    (s4 / "baselines_design.json").write_text(json.dumps({"baselines": base}))
    monkeypatch.setattr(RS, "B1_DESIGN_SHA256", sha256_file(s3 / "b1_design.json"))
    monkeypatch.setattr(S4, "BASELINES_SHA256", sha256_file(s4 / "baselines_design.json"))
    monkeypatch.setattr(S4, "FROZEN_ARGS", args)
    monkeypatch.setattr(RS, "FROZEN_ARGS", args)
    return run, cfgp


def test_stage4_end_to_end_on_fake_run(tmp_path, monkeypatch):
    run, cfgp = make_4(tmp_path, monkeypatch)
    s = S4.main(["--run", str(run), "--config", str(cfgp)])
    assert s["all_pass"], s["checks"]
    assert len(s["per_seed"]) == 5 and all(r["n_test"] == 5 and r["n_cert"] == 15 for r in s["per_seed"])
    assert set(s["pass_levels"]) == {"P2", "P3", "P4"}
    assert (run / "poc2" / "stage4" / "stage4_summary.json").exists()


def test_stage4_refuses_changed_baselines_file(tmp_path, monkeypatch):
    run, cfgp = make_4(tmp_path, monkeypatch)
    p = run / "poc2" / "stage4" / "baselines_design.json"
    p.write_text(p.read_text() + " ")
    with pytest.raises(AssertionError, match="baselines_design.json differs"):
        S4.main(["--run", str(run), "--config", str(cfgp)])


def test_stage4_refuses_changed_frozen_file(tmp_path, monkeypatch):
    run, cfgp = make_4(tmp_path, monkeypatch)
    fz = S4.FROZEN_ARGS["root"] / "configs" / "frozen_r8.yaml"
    fz.write_text(fz.read_text() + "# edited\n")
    with pytest.raises(AssertionError, match="frozen_r8.yaml differs"):
        S4.main(["--run", str(run), "--config", str(cfgp)])


def test_stage4_detects_changed_roi_in_meta(tmp_path, monkeypatch):
    """train_meta.csv has no recorded hash, so its ROI boxes are cross-checked against Stage 1's stored sizes."""
    run, cfgp = make_4(tmp_path, monkeypatch)
    mp = Path(yaml.safe_load(cfgp.read_text())["data"]["cache_dir"]) / "train_meta.csv"
    meta = pd.read_csv(mp)
    meta["roi_x2"] = meta["roi_x2"] + 7
    meta["roi_y2"] = meta["roi_y2"] + 7
    meta.to_csv(mp, index=False)
    assert S4.main(["--run", str(run), "--config", str(cfgp)])["checks"]["meta_roi_matches_stage1_sizes"] is False
