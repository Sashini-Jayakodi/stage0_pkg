"""Tests for POC-3 Stage 3 scores and design phase (src/r9_stage3_scores.py, src/r9_stage3_design.py), synthetic only.
Brute-force references are plain loops written from LOG_r9_stage3_entry.md Part A, independent of the vectorised code."""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src import r8_stage1 as R1  # noqa: E402
from src import r8_stage2 as S2  # noqa: E402
from src import r9_certify as C  # noqa: E402
from src import r9_frozen as FZ  # noqa: E402
from src import r9_stage3_design as D  # noqa: E402
from src import r9_stage3_scores as SC  # noqa: E402
from src.util import sha256_file  # noqa: E402
import test_r8_stage1 as T1  # noqa: E402

rng = np.random.default_rng(11)


# ------------------------------------------------------------------ brute-force references
def brute_first(conf, lam):
    for t, c in enumerate(conf):
        if np.isfinite(c) and c >= lam:
            return t
    return len(conf)


def test_first_commit_matches_loop():
    for _ in range(300):
        L = int(rng.integers(1, 12))
        conf = rng.normal(size=L)
        conf[rng.random(L) < 0.4] = -np.inf
        grid = np.concatenate([[-np.inf], np.sort(rng.normal(size=6))])
        got = SC.first_commit(conf, grid)
        assert list(got) == [brute_first(conf, l) for l in grid]


def test_lambda0_never_commits_a_track_without_value():
    assert SC.first_commit(np.full(5, -np.inf), np.array([-np.inf, 0.0]))[0] == 5


def test_s3_running_vote_matches_recount():
    for _ in range(100):
        L, K = 15, 5
        F = {"rows": [np.arange(L)], "passed": rng.random(L) < 0.7, "top": rng.integers(0, K, L)}
        conf, lab = SC.seq_s3(F, 0, K=K)
        for t in range(L):
            v = [int(sum(F["passed"][u] and F["top"][u] == k for u in range(t + 1))) for k in range(K)]
            if sum(v) == 0:
                assert conf[t] == -np.inf and lab[t] == -1
                continue
            best = max(v)
            l = v.index(best)                                           # first maximum = smallest class id
            runner = sorted(v)[-2]
            assert lab[t] == l and conf[t] == best - runner


def test_s4_has_values_at_passing_frames_only():
    F = {"rows": [np.arange(6)], "passed": np.array([0, 1, 1, 0, 1, 0], bool),
         "m": np.array([-np.inf, 2.0, 0.5, -np.inf, 3.0, -np.inf]), "top": np.array([4, 1, 2, 3, 1, 0])}
    conf, lab = SC.seq_s4(F, 0)
    assert list(np.isfinite(conf)) == list(F["passed"]) and list(lab) == [-1, 1, 2, -1, 1, -1]


def brute_grid(cmax, n_points=30, f_low=0.3):
    N = len(cmax)
    vals = sorted(cmax, reverse=True)
    g = {-math.inf}
    for j in range(1, n_points):
        f = 1 - (1 - f_low) * j / (n_points - 1)
        m = 1
        while m < f * N - 1e-9:
            m += 1
        g.add(vals[m - 1])
    return sorted(g)


def test_grid_matches_brute_and_commit_fractions():
    for N in (261, 50, 7):
        cmax = np.round(rng.normal(size=N), 1)                         # ties on purpose
        cmax[:2] = -np.inf
        g = SC.make_grid(cmax)
        assert list(g) == brute_grid(list(cmax))
        for lam in g[1:]:
            assert (cmax >= lam).sum() >= math.ceil(F_LOW_MIN * N - 1e-9)


F_LOW_MIN = 0.30


def test_discrete_scores_give_fewer_points():
    cmax = rng.integers(25, 31, 261).astype(float)
    assert len(SC.make_grid(cmax)) <= 7


def test_design_stats_and_reference_match_loops():
    n, G = 40, 9
    com = rng.random((n, G)) < np.linspace(0.95, 0.3, G)
    wrong = rng.random((n, G)) < 0.1
    pos = np.where(com, rng.integers(1, 31, (n, G)), 0)
    st = SC.design_stats({"committed": com, "wrong": wrong, "pos": pos})
    for j in range(G):
        c = [i for i in range(n) if com[i, j]]
        w = [i for i in c if wrong[i, j]]
        assert st["committed"][j] == len(c) and st["wrong"][j] == len(w)
        assert st["f_d"][j] == pytest.approx(len(c) / n)
        if c:
            assert st["e_d"][j] == pytest.approx(len(w) / len(c))
            assert st["median_pos"][j] == pytest.approx(np.median([pos[i, j] for i in c]))
    ref = SC.reference(st, alphas=(0.05, 0.1, 0.2))
    for a in (0.05, 0.1, 0.2):
        want = -1
        for j in range(G):
            if st["committed"][j] and st["wrong"][j] / st["committed"][j] <= a:
                want = j
                break
        assert ref[a] == want


def test_binomial_start_matches_loop_and_half_percent_has_none():
    f_d = np.sort(rng.uniform(0.3, 1.0, 25))[::-1]
    f_d[0] = 1.0
    S = SC.starts(f_d, 261)
    for a in SC.ALPHAS:
        want = -1
        for j in range(len(f_d)):
            if f_d[j] * 784 * a >= 4:
                want = j
        assert S[a]["binom"] == want == S[a]["hb_bs"]
        assert S[a]["hb"] == C.start_for("hb", f_d, 784, a, n_design=261)
    assert S[0.005]["binom"] == -1                                     # 784 * 0.005 * 1.0 = 3.92 < 4


def test_halves_disjoint_cover_and_alternate_by_class():
    y = rng.integers(0, 43, 261)
    A, B = SC.halves(y)
    assert len(set(A) & set(B)) == 0 and sorted(set(A) | set(B)) == list(range(261))
    for k in np.unique(y):
        a, b = np.isin(A, np.flatnonzero(y == k)).sum(), np.isin(B, np.flatnonzero(y == k)).sum()
        assert a - b in (0, 1)
    A2, B2 = SC.halves(y)
    assert (A == A2).all() and (B == B2).all()


# ------------------------------------------------------------------ end-to-end on a fake run (24 tracks, 4 classes)
N_FAKE_DESIGN = 12


def make_run(tmp, monkeypatch):
    """POC-2 fake run with few classes so every class has tracks in both S2 halves."""
    monkeypatch.setattr(R1, "N_GRID", 40)
    monkeypatch.setattr(R1, "K", 4)
    orig = R1.run_checks

    def rc(inp, B):
        res, ok = orig(inp, B)
        ok["C6_design_set_is_261_tracks"] = int(B["design"].sum()) == N_FAKE_DESIGN
        return res, ok
    monkeypatch.setattr(R1, "run_checks", rc)
    run, cfgp, _ = T1.fake_run(tmp, n_tracks=24, K=4)
    assert R1.main(["--run", str(run), "--config", str(cfgp)])["all_pass"]
    monkeypatch.setattr(S2, "N_DESIGN", N_FAKE_DESIGN)
    return run


def test_end_to_end_design_only(tmp_path, monkeypatch):
    run = make_run(tmp_path, monkeypatch)
    (run / "poc2" / "stage1" / S2.POOL_FILE).unlink()                 # the design phase must not need it
    s = D.main(["--run", str(run)])
    out = run / "poc3" / "stage3" / "stage3_design.json"
    assert out.exists() and S2.POOL_FILE not in s["files_opened"]
    js = json.loads(out.read_text())
    assert js["frozen_blobs"] == FZ.FROZEN_BLOBS and js["primary_s2"] == "S2-sum"
    for sc in SC.SCORES:
        g = js["scores"][sc]["grid"]
        assert g[0] is None and all(x is not None for x in g[1:])     # only lambda_0 is -inf
        assert js["scores"][sc]["starts"]["0.005"]["binom"] == -1
    assert set(js["halves"]["A_codes"]).isdisjoint(js["halves"]["B_codes"])


def test_s1_equals_stage1_v1_table(tmp_path, monkeypatch):
    run = make_run(tmp_path, monkeypatch)
    S2.OPENED.clear()
    d = S2.load_inputs(run)
    F = SC.frame_inputs(d["head"], d["emb"], d["passed"], d["track"], d["frame"])
    tb = d["table"]
    seqs = [SC.seq_s1(F, i) for i in range(len(F["rows"]))]
    tab = SC.track_table(seqs, tb["y_track"], tb["V1_grid"])          # Stage 1 grid starts at 0 = first passing frame
    com = tab["committed"]
    assert (com == tb["V1_committed"]).all()
    assert (tab["pos"][com] == tb["V1_pos"][com]).all()                # Stage 1 marks "not committed" as -1, here 0
    assert (tab["wrong"] == tb["V1_wrong"]).all()


def test_s2_values_sit_at_window_ends(tmp_path, monkeypatch):
    run = make_run(tmp_path, monkeypatch)
    S2.OPENED.clear()
    d = S2.load_inputs(run)
    F = SC.frame_inputs(d["head"], d["emb"], d["passed"], d["track"], d["frame"])
    y = np.array([d["y"][r[0]] for r in F["rows"]])
    A, B = SC.halves(y)
    model = SC.fit_s2(F, y, A, B, "sum")
    for i, r in enumerate(F["rows"]):
        conf, _ = SC.seq_s2(F, i, model)
        npass = np.cumsum(d["passed"][r])
        want = d["passed"][r] & (npass >= SC.K_WINDOW)
        assert list(np.isfinite(conf)) == list(want)


def test_tampered_design_table_is_refused(tmp_path, monkeypatch):
    run = make_run(tmp_path, monkeypatch)
    p = run / "poc2" / "stage1" / S2.DESIGN_FILE
    p.write_bytes(p.read_bytes() + b"x")
    with pytest.raises(AssertionError, match="changed since Stage 1"):
        D.main(["--run", str(run)])


def test_edited_table_with_rehashed_summary_is_refused(tmp_path, monkeypatch):
    run = make_run(tmp_path, monkeypatch)
    dp = run / "poc2" / "stage1" / S2.DESIGN_FILE
    T = dict(np.load(dp))
    j = int(np.flatnonzero(T["V1_committed"][0])[0])
    T["V1_pred"][0, j] = (T["V1_pred"][0, j] + 1) % 4
    np.savez_compressed(dp, **T)
    sp = run / "poc2" / "stage1" / "stage1_summary.json"
    s = json.loads(sp.read_text())
    s["files_sha256"][S2.DESIGN_FILE] = sha256_file(dp)
    sp.write_text(json.dumps(s))
    with pytest.raises(AssertionError, match="do not reproduce Stage 1"):
        D.main(["--run", str(run)])


def test_freeze_check_runs_before_any_data(tmp_path, monkeypatch):
    run = make_run(tmp_path, monkeypatch)
    S2.OPENED.clear()
    bad = dict(FZ.FROZEN_BLOBS, **{FZ.ENGINE: "0" * 40})
    monkeypatch.setattr(FZ, "FROZEN_BLOBS", bad)
    with pytest.raises(AssertionError, match="frozen blob ID"):
        D.main(["--run", str(run)])
    assert S2.OPENED == []


def test_design_modules_name_no_pool_or_btsc_path():
    for f in ("r9_stage3_design.py", "r9_stage3_scores.py"):
        src = (ROOT / "src" / f).read_text(encoding="utf-8")
        for bad in ("track_table_pool", "btsc_raw", "runs/btsc", "import torch", "stage4_summary", "make_split"):
            assert bad not in src, (f, bad)
