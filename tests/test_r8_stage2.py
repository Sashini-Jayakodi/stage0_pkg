"""Tests for POC-2 Stage 2 (src/r8_stage2.py) on synthetic data. No real GTSRB is read.
Brute-force references are plain loops written from the Part A definitions, independently of the vectorised code."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import binom

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src import r8_stage1 as R1  # noqa: E402
from src import r8_stage2 as S2  # noqa: E402
from src.certify import hb_pvalue  # noqa: E402
from src.util import sha256_file  # noqa: E402
import test_r8_stage1 as T1  # noqa: E402  (reuse the fake POC-1 run folder)

rng = np.random.default_rng(7)


# ------------------------------------------------------------------ synthetic commit tables
def synth_table(n=261, G=40, commit_frac=0.9, err=0.01, seed=0, L=30):
    """A Stage-1-shaped table built the same way as Stage 1: per-track margin paths, first_commit per lambda."""
    r = np.random.default_rng(seed)
    grid = np.concatenate([[0.0], np.geomspace(0.5, 200, G - 1)])
    committed = np.zeros((n, G), bool)
    pos = np.full((n, G), -1, int)
    size = np.full((n, G), np.nan)
    wrong = np.zeros((n, G), bool)
    for i in range(n):
        reach = r.random() < commit_frac                                  # tracks that ever get evidence
        # margins are >= 0, so a track with any evidence commits at lambda = 0; a never-committing track must have none
        M = np.cumsum(r.exponential(3.0, L)) if reach else np.full(L, -np.inf)
        sizes = 15 + 3 * np.arange(L) + r.integers(0, 3)
        idx = R1.first_commit(M, grid)
        c = idx < L
        bad = r.random() < err
        committed[i], pos[i, c], size[i, c], wrong[i] = c, idx[c] + 1, sizes[idx[c]], c & bad
    return {"grid": grid, "committed": committed, "wrong": wrong, "size": size, "pos": pos}


def stats_of(t):
    return S2.grid_stats(t["committed"], t["wrong"], t["size"], t["pos"])


# ------------------------------------------------------------------ (b), (e): per-lambda statistics and plug-in
def test_grid_stats_match_brute_force():
    t = synth_table(n=40, G=12, commit_frac=0.6, err=0.2)
    st = stats_of(t)
    for j in range(12):
        com = [i for i in range(40) if t["committed"][i, j]]
        assert st["k"][j] == len(com) and st["c"][j] == len(com) / 40
        w = sum(1 for i in com if t["wrong"][i, j])
        assert st["w"][j] == w
        if com:
            assert st["e"][j] == w / len(com)
            assert st["med_pos"][j] == np.median([t["pos"][i, j] for i in com])
            assert st["med_size"][j] == np.median([t["size"][i, j] for i in com])
        else:
            assert np.isnan(st["e"][j]) and np.isnan(st["med_size"][j])


def test_mean_L_is_the_mean_of_per_track_losses():
    t = synth_table(n=50, G=10, commit_frac=0.7, err=0.3)
    st = stats_of(t)
    for a in S2.ALPHAS:
        brute = [np.mean([float(t["wrong"][i, j]) - a * float(t["committed"][i, j]) + a for i in range(50)]) for j in range(10)]
        assert np.allclose(S2.mean_L(st, a), brute, atol=1e-15)


def test_d2_pass_rule_matches_brute_force():
    stats = {v: stats_of(synth_table(seed=s, commit_frac=cf, err=e)) for v, s, cf, e in
             zip(S2.VARIANTS, (1, 2, 3), (0.95, 0.75, 0.6), (0.0, 0.02, 0.0))}
    d2 = S2.d2_table(stats)
    for v, st in stats.items():
        for a in S2.ALPHAS:
            brute = any(hb_pvalue(S2.mean_L(st, a)[j], 784, a) <= 0.01 and st["c"][j] >= 0.70 for j in range(len(st["c"])))
            assert d2[v][a]["pass"] == brute


# ------------------------------------------------------------------ (a) D1 bound
@pytest.mark.parametrize("k,n", [(0, 50), (3, 50), (10, 256), (49, 50)])
def test_cp_upper_matches_bisection(k, n):
    lo, hi = k / n, 1.0
    for _ in range(200):                                                  # P(Bin(n, u) <= k) = 0.025 at the upper end
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if binom.cdf(k, n, mid) > 0.025 else (lo, mid)
    assert abs(S2.cp_upper(k, n) - hi) < 1e-9
    assert S2.cp_upper(n, n) == 1.0


# ------------------------------------------------------------------ (d) lateness
def test_late_fraction_matches_definition():
    """Brute force walks the frames: late unless the track commits at a frame whose ROI size is <= s."""
    n, L = 30, 30
    grid = np.array([0.0, 2.0, 8.0, 30.0, 1e9])
    Ms, sizes = [], []
    com = np.zeros((n, len(grid)), bool)
    size = np.full((n, len(grid)), np.nan)
    for i in range(n):
        M = np.cumsum(rng.exponential(1.0, L))
        M[: rng.integers(0, 4)] = -np.inf                                 # some tracks start with no evidence
        sz = np.sort(rng.integers(10, 120, L)).astype(float)
        Ms.append(M); sizes.append(sz)
        idx = R1.first_commit(M, grid)
        c = idx < L
        com[i], size[i, c] = c, sz[idx[c]]
    for s in (5, 20, 45, 100, 500):
        brute = []
        for j, lam in enumerate(grid):
            late = 0
            for M, sz in zip(Ms, sizes):
                t_commit = next((t for t in range(L) if M[t] >= lam), None)
                late += 0 if (t_commit is not None and sz[t_commit] <= s) else 1
            brute.append(late / n)
        assert np.allclose(S2.late_frac(com, size, s), brute)


def test_ceil5():
    assert [S2.ceil5(x) for x in (0.2, 5.0, 5.5, 29.0, 30.0, 30.5)] == [5, 5, 10, 30, 30, 35]


# ------------------------------------------------------------------ (c), (g), (h): order, variant, grid
def test_working_grid_rule_brute_force():
    st = stats_of(synth_table(G=120, commit_frac=0.95, err=0.004))
    for a in (0.01, 0.05):
        w = S2.working_grid(st, a)
        useful = [j for j in range(120) if st["c"][j] >= 0.70 and st["k"][j] > 0 and st["e"][j] <= a]
        lo, hi = min(useful), max(useful)
        expect = list(range(lo, hi + 1)) if hi - lo + 1 <= 30 else sorted(set(int(round(x)) for x in np.linspace(lo, hi, 30)))
        assert list(w) == expect and len(w) <= 30 and w[0] == lo and w[-1] == hi


def test_working_grid_empty_when_nothing_useful():
    st = stats_of(synth_table(commit_frac=0.3))
    assert len(S2.working_grid(st, 0.01)) == 0


def test_start_ties_go_to_larger_lambda_and_order_relaxes():
    start, order = S2.start_and_order(np.array([0.03, 0.01, 0.02, 0.01, 0.04]))
    assert start == 3 and order == [3, 2, 1, 0]


def test_walk_stops_at_first_failure():
    p = np.array([0.01, 0.5, 0.02, 0.03])
    assert S2.walk(p, [3, 2, 1, 0], 0.1) == 2                           # 3 and 2 pass, 1 fails, 0 is never reached
    assert S2.walk(p, [1, 0], 0.1) is None


def test_variant_tie_goes_to_V1():
    t = synth_table(commit_frac=0.95, err=0.0)
    stats = {v: stats_of(t) for v in S2.VARIANTS}
    d2 = S2.d2_table(stats)
    a = S2.choose_primary_alpha(d2)
    assert S2.choose_variant(stats, d2, a)[0] == "V1"


def test_variant_rule_prefers_higher_commit_count_at_half_alpha():
    stats = {"V1": stats_of(synth_table(seed=1, commit_frac=0.80, err=0.0)),
             "V2": stats_of(synth_table(seed=1, commit_frac=0.95, err=0.0)),
             "V3": stats_of(synth_table(seed=1, commit_frac=0.85, err=0.0))}
    d2 = S2.d2_table(stats)
    assert S2.choose_variant(stats, d2, S2.choose_primary_alpha(d2))[0] == "V2"


# ------------------------------------------------------------------ (d) s rule, Amendment 1 (no fallback)
def test_choose_s_is_the_smallest_passing_multiple_of_5():
    t = synth_table(commit_frac=0.995, err=0.0, seed=4)
    st = stats_of(t)
    a = 0.05
    w = S2.working_grid(st, a)
    com, size = t["committed"][:, w], t["size"][:, w]
    p_err = hb_pvalue(S2.mean_L(st, a)[w], 784, a)
    s, scan = S2.choose_s(com, size, p_err)
    brute = None
    for s_try in range(5, 1000, 5):                                       # plain loop over every s and every lambda
        ok = False
        for j in range(len(w)):
            late = np.mean([0 if (com[i, j] and size[i, j] <= s_try) else 1 for i in range(com.shape[0])])
            if max(p_err[j], hb_pvalue(late, 784, 0.10)) <= 0.01:
                ok = True
        if ok:
            brute = s_try
            break
    assert brute is not None and s == brute and scan[-1][0] == s and scan[-1][1] <= 0.01


def test_no_s_qualifies_gives_none_and_no_substitute():
    """D2 passes (85% commit, no errors) but at least 15% of tracks are always late, so no s can meet gamma = 10%."""
    t = synth_table(commit_frac=0.85, err=0.0, seed=5)
    tables = {v: t for v in S2.VARIANTS}
    stats = {v: stats_of(t) for v in S2.VARIANTS}
    d2 = S2.d2_table(stats)
    assert S2.choose_primary_alpha(d2) is not None
    ch = S2.design_choices(stats, tables, d2)
    assert ch["s"] is None and ch["s_certifiable"] is False
    assert ch["pB"] is None and ch["b_order"] == [] and ch["b_report"] is None
    assert len(ch["s_scan"]) > 0 and all(mp > 0.01 for _, mp in ch["s_scan"])
    txt = json.dumps(S2.jsonable(ch))
    assert "median" not in txt and "fallback" not in txt                 # no median-rule value anywhere in the summary


# ------------------------------------------------------------------ end-to-end on a fake run folder
def make_stage1(tmp, monkeypatch):
    monkeypatch.setattr(R1, "N_GRID", 40)
    orig = R1.run_checks

    def rc(inp, B):
        res, ok = orig(inp, B)
        ok["C6_design_set_is_261_tracks"] = int(B["design"].sum()) == 6  # the fake run has 6 design tracks
        return res, ok
    monkeypatch.setattr(R1, "run_checks", rc)
    run, cfgp, _ = T1.fake_run(tmp)
    assert R1.main(["--run", str(run), "--config", str(cfgp)])["all_pass"]
    monkeypatch.setattr(S2, "N_DESIGN", 6)
    return run


def test_end_to_end_reproduces_stage1_and_never_opens_pool(tmp_path, monkeypatch):
    run = make_stage1(tmp_path, monkeypatch)
    (run / "poc2" / "stage1" / S2.POOL_FILE).unlink()                    # Stage 2 must not need it at all
    frozen = ROOT / "configs" / "frozen_r8.yaml"
    existed = frozen.exists()
    s = S2.main(["--run", str(run)])
    assert frozen.exists() == existed                                     # Stage 2 never writes the freeze file
    for k in ("S1_inputs_match_stage1_hashes", "S2_design_set_only", "S3_gate_flags_reproduce",
              "S4_commit_table_reproduces", "S5_pool_file_never_opened"):
        assert s["checks"][k], k
    assert S2.POOL_FILE not in s["files_opened"]
    assert (run / "poc2" / "stage2" / "stage2_summary.json").exists()


def test_d1_and_curve_match_brute_force_on_fake_run(tmp_path, monkeypatch):
    run = make_stage1(tmp_path, monkeypatch)
    S2.OPENED.clear()
    d = S2.load_inputs(run)
    rc = S2.recompute(d)
    D1 = S2.d1(d, rc)
    tb = d["table"]
    wrong_tracks = [bool(tb["V1_top_last"][i] != tb["y_track"][i]) for i in range(6) if tb["n_passing"][i] > 0]
    assert D1["k"] == sum(wrong_tracks) and D1["n"] == len(wrong_tracks)
    pf = [d["top1"][i] != d["y"][i] for i in range(len(d["y"])) if d["passed"][i]]
    assert D1["frame_err_passing"] == pytest.approx(np.mean(pf))
    cv = S2.curve(d, rc)
    # brute: V1 top class at position t from the Stage 1 brute-force scorer
    ll, _ = R1.frame_evidence(d["head"], d["emb"])
    lp = np.log(d["head"]["priors"])
    for t in (1, 10, 29):
        errs = []
        for i, r in enumerate(rc["rows"]):
            if len(r) < t or not d["passed"][r[:t]].any():
                continue
            S = T1.brute_scores(ll[r], d["passed"][r], lp, "V1")[t - 1]
            errs.append(int(np.nanargmax(S)) != tb["y_track"][i])
        assert cv[t - 1]["V1"] == pytest.approx(np.mean(errs))


def test_pool_file_is_refused_by_the_loader(tmp_path):
    p = tmp_path / S2.POOL_FILE
    np.savez(p, x=np.zeros(1))
    with pytest.raises(AssertionError, match="never open the pool table"):
        S2.open_npz(p)


@pytest.mark.parametrize("rel,msg", [
    ("poc2/stage1/track_table_design.npz", "track_table_design.npz changed since Stage 1"),
    ("poc2/stage1/frame_evidence.npz", "frame_evidence.npz changed since Stage 1"),
    ("embeddings/train_release.npz", "train_release.npz changed since Stage 1"),
    ("head_emb/head.npz", "head.npz changed since Stage 1"),
    ("splits/seed0.npz", "seed0.npz changed since Stage 1"),
    ("head_emb/scores_train_release.npz", "scores_train_release.npz changed since Stage 3"),
])
def test_tampered_inputs_are_refused(tmp_path, monkeypatch, rel, msg):
    run = make_stage1(tmp_path, monkeypatch)
    p = run / rel
    p.write_bytes(p.read_bytes() + b"x")
    with pytest.raises(AssertionError, match=msg):
        S2.main(["--run", str(run)])


def test_failed_stage1_is_refused(tmp_path, monkeypatch):
    run = make_stage1(tmp_path, monkeypatch)
    sp = run / "poc2" / "stage1" / "stage1_summary.json"
    s = json.loads(sp.read_text())
    s["all_pass"] = False
    sp.write_text(json.dumps(s))
    with pytest.raises(AssertionError, match="Stage 1 did not pass"):
        S2.main(["--run", str(run)])


def test_edited_table_with_rehashed_summary_fails_reproduction(tmp_path, monkeypatch):
    """Even if someone edits the design table AND updates its hash, the recomputation catches it (check S4)."""
    run = make_stage1(tmp_path, monkeypatch)
    dp = run / "poc2" / "stage1" / S2.DESIGN_FILE
    T = dict(np.load(dp))
    j = int(np.flatnonzero(T["V1_committed"][0])[0])
    T["V1_pred"][0, j] = (T["V1_pred"][0, j] + 1) % R1.K
    np.savez_compressed(dp, **T)
    sp = run / "poc2" / "stage1" / "stage1_summary.json"
    s = json.loads(sp.read_text())
    s["files_sha256"][S2.DESIGN_FILE] = sha256_file(dp)
    sp.write_text(json.dumps(s))
    assert S2.main(["--run", str(run)])["checks"]["S4_commit_table_reproduces"] is False
