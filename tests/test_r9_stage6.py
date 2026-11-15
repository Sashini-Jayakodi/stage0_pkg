"""Tests for R9 Stage 6 on synthetic data: the RQ1 draws, per-pole sums, certification and violation counting against
brute-force references, the V-check bound, Test metrics, the Ringel steps with views beyond tmax, B5c parts by pole,
B6, and the runner end to end on a synthetic Stage 4 folder with a synthetic frozen yaml. Tamper tests: changed pool
files, design file, yaml values, starts and splits are refused; check_frozen() runs first; pool labels need F3.
No real BelgiumTSC file is read."""
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml
from scipy.stats import beta, binom

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import r8_stage6_methods as M  # noqa: E402
from src import r9_frozen as FZ  # noqa: E402
from src import r9_stage4_data as D4  # noqa: E402
from src import r9_stage5_design as DS  # noqa: E402
from src import r9_stage5_freeze as FR  # noqa: E402
from src import r9_stage5_scores as SC  # noqa: E402
from src import r9_stage6 as R6  # noqa: E402
from src import r9_stage6_methods as SM  # noqa: E402
from src.head import score  # noqa: E402
from src.r9_btsc_guard import GuardedTestLabels  # noqa: E402
from src.r9_frozen import blob_id  # noqa: E402
from src.r9_stage4_model import fit_head_k  # noqa: E402
from src.util import sha256_file  # noqa: E402


# ---------------------------------------------------------------- brute-force references
def brute_certify(k, n_c, n, alpha, f_d, delta=0.1):
    """Exact binomial with the power-aware start, written out with a loop."""
    p = [1.0 if c == 0 else binom.cdf(w, c, alpha) for w, c in zip(k, n_c)]
    start = -1
    for j in range(len(f_d)):
        if f_d[j] * n * alpha >= 4:
            start = j
    best = -1
    for j in range(start, -1, -1):
        if p[j] > delta:
            break
        best = j
    return best


def random_table(rng, n, G, p_wrong=0.03):
    """Nested commit sets (stricter thresholds commit a subset) with wrong flags fixed per item."""
    conf = rng.random(n)
    grid = np.concatenate([[-np.inf], np.linspace(0.05, 0.9, G - 1)])
    com = conf[:, None] >= grid[None, :]
    wrong = (rng.random(n) < p_wrong)[:, None] & com
    return {"committed": com, "wrong": wrong, "pos": np.where(com, 1, 0)}


# ---------------------------------------------------------------- RQ1 parts
def test_draw_counts_match_direct_draws():
    c = SM.draw_counts(7, 5, 6, (1, 2))
    for d in range(5):
        idx = np.random.default_rng([1, 2, d]).integers(0, 7, size=6)
        assert c[d].tolist() == [int((idx == p).sum()) for p in range(7)]
    assert (c.sum(1) == 6).all()


def test_per_pole_and_draw_sums_bruteforce():
    rng = np.random.default_rng(3)
    item_pole = rng.integers(0, 9, 40)
    tab = random_table(rng, 40, 6, 0.2)
    Cp, Wp, n_items = SM.per_pole(tab, item_pole, 9)
    counts = SM.draw_counts(9, 4, 9, (5,))
    K, NC, N = SM.draw_sums(counts, Cp, Wp, n_items)
    for d in range(4):
        idx = np.random.default_rng([5, d]).integers(0, 9, size=9)
        items = np.concatenate([np.flatnonzero(item_pole == p) for p in idx])    # a pole twice: its items twice
        assert N[d] == len(items)
        assert NC[d].tolist() == tab["committed"][items].sum(0).tolist()
        assert K[d].tolist() == (tab["committed"] & tab["wrong"])[items].sum(0).tolist()


def test_rq1_cell_against_bruteforce():
    rng = np.random.default_rng(4)
    G, D = 8, 60
    f_d = np.linspace(1.0, 0.3, G)
    NC = np.sort(rng.integers(150, 500, size=(D, G)), axis=1)[:, ::-1]
    K = rng.binomial(NC, 0.02)
    N = NC[:, 0] + rng.integers(0, 5, D)
    risk = np.linspace(0.04, 0.0, G)
    for a in (0.01, 0.03, 0.05):
        cell = SM.rq1_cell(K, NC, N, a, f_d, "binom", 157, risk)
        ref = [brute_certify(K[d], NC[d], N[d], a, f_d) for d in range(D)]
        assert cell["certified"] == sum(r >= 0 for r in ref)
        assert cell["violations"] == sum(r >= 0 and risk[r] > a for r in ref)
        assert sum(cell["index_counts"].values()) == cell["certified"]
        ok = [r for r in ref if r >= 0]
        if ok:
            assert np.isclose(cell["mean_true_risk"], np.mean(risk[ok]))


def test_rq1_cell_counts_no_start_and_hb():
    G, D = 4, 10
    NC = np.full((D, G), 100)
    K = np.zeros((D, G), int)
    cell = SM.rq1_cell(K, NC, np.full(D, 100), 0.005, np.ones(G), "binom", 157, np.zeros(G))
    assert cell["no_start"] == D and cell["certified"] == 0 and cell["violations"] == 0
    cell = SM.rq1_cell(K, NC, np.full(D, 100), 0.05, np.ones(G), "hb", 157, np.zeros(G))
    assert cell["draws"] == D


def test_v_check_bound():
    def cell(v, D=2000):
        hi = 1.0 if v == D else float(beta.ppf(0.975, v + 1, D - v))
        return {"cp95": [0.0, hi]}
    assert SM.v_check(cell(0))                                  # never certifies: passes (bound 0.18%)
    assert np.isclose(cell(0)["cp95"][1], 0.0018427, atol=1e-6)
    v = max(x for x in range(400) if beta.ppf(0.975, x + 1, 2000 - x) <= 0.12)
    assert SM.v_check(cell(v)) and not SM.v_check(cell(v + 1))
    assert SM.V_BOUND == 0.12


# ---------------------------------------------------------------- RQ2 parts
def test_certify_split_and_test_metrics_bruteforce():
    rng = np.random.default_rng(5)
    n, G = 300, 6
    tab = random_table(rng, n, G, 0.05)
    tab["pos"] = np.where(tab["committed"], rng.integers(1, 4, (n, 1)), 0)
    tab["size"] = np.where(tab["committed"], rng.integers(10, 60, (n, 1)).astype(float), np.nan)
    cert = rng.random(n) < 0.7
    test = ~cert
    f_d = tab["committed"].mean(0)
    for a in (0.03, 0.05, 0.1):
        j = SM.certify_split(tab, cert, a, f_d, "binom", 157)
        k = (tab["committed"] & tab["wrong"])[cert].sum(0)
        assert j == brute_certify(k, tab["committed"][cert].sum(0), cert.sum(), a, f_d)
    s_values = list(range(5, 65, 5))
    m = SM.test_metrics(tab, 2, test, s_values)
    com, wr = tab["committed"][test, 2], tab["wrong"][test, 2]
    assert m["committed"] == com.sum() and m["wrong"] == (com & wr).sum()
    assert np.isclose(m["coverage"], com.mean()) and np.isclose(m["error"], (com & wr).sum() / com.sum())
    assert m["median_view"] == np.median(tab["pos"][test, 2][com])
    sz = tab["size"][test, 2]
    d90 = next((s for s in s_values if np.mean(com & (sz <= s)) >= 0.9), "not reached")
    assert m["d90"] == d90
    lo, hi = m["cp95"]
    k_, c_ = m["wrong"], m["committed"]
    assert np.isclose(hi, beta.ppf(0.975, k_ + 1, c_ - k_))
    none = SM.test_metrics(tab, -1, test, s_values)
    assert none["coverage"] == 0.0 and none["error"] is None


def test_smallest_alpha():
    al = (0.005, 0.01, 0.02)
    assert SM.smallest_alpha({0.005: -1, 0.01: 3, 0.02: 2}, al) == 0.01
    assert SM.smallest_alpha({a: -1 for a in al}, al) is None


# ---------------------------------------------------------------- Ringel with views as time
def ringel_units(rng, n, max_len):
    lens = rng.integers(1, max_len + 1, n)
    pis = [np.round(rng.random(L), 2) for L in lens]
    preds = [rng.integers(0, 3, L) for L in lens]
    y = rng.integers(0, 3, n)
    yfull = np.array([p[-1] for p in preds])
    return pis, preds, y, yfull


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_screen_and_test_equal_poc2_for_short_units(seed):
    rng = np.random.default_rng(seed)
    pis, preds, y, yfull = ringel_units(rng, 80, 3)
    for a in (0.05, 0.1, 0.2):
        cand = SM.screen_ext(pis, preds, y, yfull, a, 3)
        assert np.array_equal(cand, M.stage1_screen(pis, preds, y, yfull, a, 3))
        assert np.array_equal(SM.test_ext(pis, preds, y, yfull, cand, a, 0.1, 3),
                              M.stage2_test(pis, preds, y, yfull, cand, a, 0.1, 3))


def test_lambda_tmax_applies_beyond_tmax():
    assert SM.ext([0.5, 0.7, 0.9], 6).tolist() == [0.5, 0.7, 0.9, 0.9, 0.9, 0.9]
    pis = [np.array([0.1, 0.1, 0.1, 0.95, 0.2, 0.2]), np.array([0.1, 0.1, 0.1, 0.5])]
    assert SM.halts_ext(pis, [0.5, 0.5, 0.9]).tolist() == [3, 3]      # 4th view passes 0.9; last view forced
    assert SM.halts_ext(pis, [0.5, 0.5, np.inf]).tolist() == [5, 3]


def test_screen_counts_long_units_at_last_step():
    """At the last screening step every still-running unit halts somewhere (at or after view tmax)."""
    pis = [np.array([0.0, 0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0])]
    preds = [np.array([1, 1, 1, 0]), np.array([0, 0, 0])]
    y, yfull = np.array([0, 0]), np.array([0, 0])
    # unit 0: Y_full = 0 is right, early label 1 is wrong at views 1-3, so stopping before view 4 costs a gap loss
    lam = SM.screen_ext(pis, preds, y, yfull, 0.0, 3)
    assert np.isinf(lam[0]) and np.isinf(lam[1])
    assert lam[2] > 0.0                                                # g = 0 would stop unit 0 at view 3 (loss 1)


def test_b5c_parts_by_pole():
    cert_poles = list(range(10, 407))
    unit_pole = np.array([p for p in cert_poles for _ in range(1 + p % 2)])
    cert_units = np.arange(len(unit_pole))
    d1, d2, h = SM.b5c_parts(cert_poles, unit_pole, cert_units, 0)
    assert h == 199 and len(set(unit_pole[d1])) == 199 and len(set(unit_pole[d2])) == 198
    assert not set(unit_pole[d1]) & set(unit_pole[d2]) and len(d1) + len(d2) == len(unit_pole)
    perm = np.array(cert_poles)[np.random.default_rng([SC.B5C_SALT, 0]).permutation(397)]
    assert set(unit_pole[d1]) == set(perm[:199].tolist())


def test_b6_table_bruteforce():
    seqs = [(np.array([0.1, 0.3, 0.9]), np.array([2, 1, 1])), (np.array([0.05]), np.array([0])),
            (np.array([0.2, 0.2]), np.array([3, 3]))]
    y = np.array([1, 0, 3])
    rows = [np.array([0, 1, 2]), np.array([3]), np.array([4, 5])]
    size = np.arange(10, 70, 10.0)
    t = SM.b6_table(seqs, y, 0.15, rows, size)
    assert t["committed"][:, 0].tolist() == [True, False, True]
    assert t["pos"][:, 0].tolist() == [2, 0, 1] and t["wrong"][:, 0].tolist() == [False, False, False]
    assert t["size"][0, 0] == 20 and t["size"][2, 0] == 50
    t = SM.b6_table(seqs, y, -np.inf, rows, size)                   # commit at view 1
    assert t["committed"].all() and t["pos"][:, 0].tolist() == [1, 1, 1] and t["wrong"][0, 0]
    assert not SM.b6_table(seqs, y, None, rows, size)["committed"].any()


def test_unit_table_size_of_committing_view():
    seqs = [(np.array([0.1, 0.5, 0.2]), np.array([0, 0, 1]))]
    t = SM.unit_table(seqs, np.array([0]), np.array([-np.inf, 0.3, 0.6]), [np.array([4, 5, 6])],
                      np.arange(100.0))
    assert t["committed"][0].tolist() == [True, True, False]
    assert t["size"][0, 0] == 4 and t["size"][0, 1] == 5 and np.isnan(t["size"][0, 2])


# ---------------------------------------------------------------- synthetic Stage 4 folder with pool files
def make_stage4(tmp, n_design_poles=60, n_pool_poles=120, K=8, d=12, seed=0):
    rng = np.random.default_rng(seed)
    base = Path(tmp) / "stage4"
    (base / "embeddings").mkdir(parents=True)
    (base / "head").mkdir()
    rows, p = [], 1
    for side, n_p in (("design", n_design_poles), ("pool", n_pool_poles)):
        for _ in range(n_p):
            for c in sorted(set(rng.integers(0, K, 1 + int(rng.random() < 0.2)).tolist())):
                for v in range(int(rng.choice([1, 2, 3, 3, 3, 3, 4]))):
                    rows.append({"relpath": f"T/{c:05d}/{p:05d}_{v:05d}.ppm", "folder": f"{c:05d}",
                                 "file": f"{p:05d}_{v:05d}.ppm", "pole": p, "view": v, "cls": c, "width": 50,
                                 "height": 50, "roi_x1": 1, "roi_y1": 1, "roi_x2": 40, "roi_y2": 40,
                                 "size": 20 + 7 * v + int(rng.integers(0, 5)), "part": side})
            p += 1
    D4.write_table(base / "test_views.csv", rows)
    design = sorted({r["pole"] for r in rows if r["part"] == "design"})
    pool = sorted({r["pole"] for r in rows if r["part"] == "pool"})
    (base / "split_poles.json").write_text(json.dumps({"design_poles": design, "pool_poles": pool}, indent=1) + "\n")
    fit_y = np.repeat(np.arange(K), 30)
    means = rng.normal(size=(K, d)) * 2.5
    head = fit_head_k(means[fit_y] + rng.normal(size=(len(fit_y), d)), fit_y, K)
    np.savez(base / "head" / "head.npz", mu=head["mu"], W=head["W"], prec=head["prec"], priors=head["priors"],
             shrinkage=head["shrinkage"])
    for side in ("design", "pool"):
        idx = np.array([i for i, r in enumerate(rows) if r["part"] == side])
        y = np.array([rows[i]["cls"] for i in idx])
        emb = (means[y] + rng.normal(size=(len(idx), d)) * 1.3).astype(np.float32)
        get = lambda k: np.array([rows[i][k] for i in idx])
        extra = {"y": y} if side == "design" else {}
        np.savez(base / "embeddings" / f"test_{side}.npz", emb=emb, row=idx, pole=get("pole"), view=get("view"),
                 size=get("size"), unit=np.arange(len(idx)), **extra)
        sc = score(head, emb)
        np.savez(base / "head" / f"scores_test_{side}.npz", top3=sc["top3"], lp3=sc["lp3"], margin=sc["margin"],
                 s=sc["s"], row=idx, pole=get("pole"), view=get("view"), size=get("size"), **extra)
    files = {rel: sha256_file(base / rel) for rel in (*DS.INPUTS, *R6.POOL_INPUTS)}
    (base / "stage4_summary.json").write_text(json.dumps({"files_sha256": files}, indent=2) + "\n")
    blobs = {rel: blob_id(base / rel) for rel in ("stage4_summary.json", "split_poles.json", "test_views.csv")}
    return base, sha256_file(base / "stage4_summary.json"), blobs


def expected_design(d):
    E = d["emb"]
    sc = score(d["head"], E["emb"])
    last = {}
    for i, u in enumerate(zip(E["pole"].tolist(), E["y"].tolist())):
        if u not in last or E["view"][i] > E["view"][last[u]]:
            last[u] = i
    pool_units = {(r["pole"], r["cls"]) for r in d["rows"] if r["part"] == "pool"}
    pool_views = sum(1 for r in d["rows"] if r["part"] == "pool")
    return ({"design_poles": len(set(E["pole"].tolist())), "design_units": len(last), "design_views": len(E["row"]),
             "pool_poles": len({p for p, _ in pool_units}), "pool_units": len(pool_units),
             "views_correct": int((sc["top3"][:, 0] == E["y"]).sum()),
             "units_correct_last_view": int(sum(sc["top3"][i, 0] == u[1] for u, i in last.items()))},
            {"pool_poles": len({p for p, _ in pool_units}), "pool_units": len(pool_units), "pool_views": pool_views})


@pytest.fixture
def synth(tmp_path, monkeypatch):
    """Synthetic Stage 4 folder, its recorded design file and a frozen yaml written by the real freeze step."""
    base, s4, blobs = make_stage4(tmp_path)
    monkeypatch.setattr(DS, "STAGE4_SUMMARY_SHA256", s4)
    monkeypatch.setattr(DS, "TEXT_BLOBS", blobs)
    monkeypatch.setattr(SC, "BOOT_DRAWS", 40)
    monkeypatch.setattr(DS, "stop", lambda m: (_ for _ in ()).throw(SystemExit(m)))
    monkeypatch.setattr(R6, "stop", lambda m: (_ for _ in ()).throw(SystemExit(m)))
    d = DS.load_stage4(base, s4, blobs)
    exp_d, exp_p = expected_design(d)
    monkeypatch.setattr(DS, "EXPECTED", exp_d)
    monkeypatch.setattr(R6, "EXPECTED_POOL", exp_p)
    monkeypatch.setattr(R6, "N_DESIGN_UNITS", exp_d["design_units"])
    R = DS.run_design(d)
    monkeypatch.chdir(ROOT)
    code = {f: sha256_file(ROOT / f) for f in DS.CODE}
    dpath = tmp_path / "stage5_design.json"
    dpath.write_text(json.dumps(DS.to_json(R, d, {"x": "y"}, code, 0.0), indent=1) + "\n")
    design = json.loads(dpath.read_text())
    ytext = yaml.safe_dump(FR.build(R, design, sha256_file(dpath), blob_id(dpath)), sort_keys=False,
                           default_flow_style=None, width=120)
    Y = yaml.safe_load(ytext)
    P = R6.open_pool(base, json.loads((base / "stage4_summary.json").read_text()))
    return {"base": base, "d": d, "R": R, "design": design, "dpath": dpath, "ytext": ytext, "Y": Y, "P": P}


def fast(Y, draws=60):
    Y = copy.deepcopy(Y)
    Y["rq1"]["draws"] = draws
    return Y


def run_synth(s, Y=None, ytext=None):
    return R6.run(s["d"], s["P"], fast(s["Y"]) if Y is None else Y, s["ytext"] if ytext is None else ytext,
                  s["design"], sha256_file(s["dpath"]), blob_id(s["dpath"]))


def test_run_end_to_end_synthetic(synth):
    res = run_synth(synth)
    assert all(res["integrity"].values())
    Y = synth["Y"]
    alphas = [float(a) for a in Y["alphas"]]
    assert set(res["rq1"]) == set(SC.SCORES) | {"B1"}
    for name, r in res["rq1"].items():
        for a in alphas:
            for rule in SM.RQ1_RULES:
                c = r["cells"][a][rule]
                assert c["draws"] == 60 and 0 <= c["violations"] <= c["certified"] <= 60
    assert set(res["v_check"]["cells"]) == set(SC.SCORES)
    assert len(res["rq2"]) == 5
    for seed, r in res["rq2"].items():
        sp = Y["resplits"]["split"][str(seed)]
        assert r["n_cert"] == sp["n_cert_units"] and r["n_test"] == sp["n_test_units"]
        for a in alphas:
            assert r["b6"]["S2-sum"][a]["theta"] == Y["b6"]["theta"]["S2-sum"][SM.akey(a)]
            assert r["b5"][a]["coverage"] == 1.0                    # Ringel never escalates
    assert set(res["comparison"]) == set(R6.COLS)
    assert res["comparison"]["B1"]["Test coverage"] == R6.NOT_RUN_B1
    assert res["comparison"]["B6 S2-sum"]["Violation rate vs delta (resampling)"] == R6.NOT_RQ1
    json.dumps(R6.to_json(res, Y, {}, {}, synth["d"], 0.0))       # the output is valid JSON


def test_rq1_sums_match_direct_recount(synth):
    """One RQ1 draw recounted by hand from the units of the drawn poles."""
    Y = fast(synth["Y"], draws=1)
    R = synth["R"]
    T = R6.pool_tables(synth["d"], synth["P"], R, Y)
    out = R6.run_rq1(T, Y, [0.05])
    unit_pole = np.array([p for p, _ in T["units"]])
    poles = np.unique(unit_pole)
    idx = np.random.default_rng([*Y["rq1"]["salt"], 0]).integers(0, len(poles), size=Y["rq1"]["poles_per_draw"])
    items = np.concatenate([np.flatnonzero(unit_pole == poles[i]) for i in idx])
    tab = T["tabs"]["S1"]
    k = (tab["committed"] & tab["wrong"])[items].sum(0)
    j = brute_certify(k, tab["committed"][items].sum(0), len(items), 0.05, np.array(Y["scores"]["S1"]["f_d"]))
    assert out["S1"]["cells"][0.05]["binom"]["certified"] == int(j >= 0)


# ---------------------------------------------------------------- tamper and order checks
def test_changed_yaml_value_refused(synth):
    bad = synth["ytext"].replace("delta: 0.1\n", "delta: 0.2\n", 1)
    assert bad != synth["ytext"]
    with pytest.raises(SystemExit, match="does not reproduce"):
        run_synth(synth, ytext=bad)


def test_changed_start_refused(synth):
    Y = fast(synth["Y"])
    st = Y["scores"]["S1"]["starts"]["0"][SM.akey(float(Y["alphas"][2]))]
    st["binom"] += 1
    with pytest.raises(SystemExit, match="differs from the frozen start"):
        run_synth(synth, Y=Y)


def test_changed_b2_start_refused(synth):
    Y = fast(synth["Y"])
    Y["b2"]["starts"]["1"][SM.akey(float(Y["alphas"][3]))]["hb"] += 1
    with pytest.raises(SystemExit, match="B2"):
        run_synth(synth, Y=Y)


def test_split_not_partitioning_refused(synth):
    Y = fast(synth["Y"])
    sp = Y["resplits"]["split"]["2"]
    sp["test_poles"] = sp["test_poles"] + [sp["cert_poles"][0]]
    with pytest.raises(SystemExit, match="partition"):
        run_synth(synth, Y=Y)


def test_split_unit_count_refused(synth):
    Y = fast(synth["Y"])
    Y["resplits"]["split"]["0"]["n_cert_units"] += 1
    with pytest.raises(SystemExit, match="unit counts"):
        run_synth(synth, Y=Y)


@pytest.mark.parametrize("rel", ["embeddings/test_pool.npz", "head/scores_test_pool.npz"])
def test_tampered_pool_file_refused(synth, rel):
    p = synth["base"] / rel
    b = bytearray(p.read_bytes())
    b[len(b) // 2] = (b[len(b) // 2] + 1) % 256
    p.write_bytes(bytes(b))
    with pytest.raises(SystemExit):
        R6.open_pool(synth["base"], json.loads((synth["base"] / "stage4_summary.json").read_text()))


def test_tampered_design_file_refused(synth):
    p = synth["dpath"]
    sha, blob = sha256_file(p), blob_id(p)
    p.write_text(p.read_text().replace('"n_min": ', '"n_min": 1'))
    with pytest.raises(SystemExit):
        FR.load_design(p, sha, blob)


def test_pool_labels_need_f3(synth, monkeypatch):
    monkeypatch.setattr(R6, "f3_closed", lambda: False)
    with pytest.raises(AssertionError, match="F3 is not closed"):
        R6.pool_tables(synth["d"], synth["P"], synth["R"], fast(synth["Y"]))


def test_guard_refuses_pool_labels_before_f3():
    g = GuardedTestLabels([1, 2, 3], [5, 6, 7], design_poles=[1], blobs={})
    with pytest.raises(AssertionError):
        g.labels([1])


def test_check_frozen_runs_first(monkeypatch, tmp_path):
    calls = []

    def fake():
        calls.append("frozen")
        raise RuntimeError("frozen check called")
    monkeypatch.setattr(R6.FZ, "check_frozen", fake)
    monkeypatch.setattr(R6, "load_yaml", lambda *a: calls.append("yaml"))
    with pytest.raises(RuntimeError, match="frozen check called"):
        R6.main(["--stage4", str(tmp_path / "none"), "--out", str(tmp_path / "out")])
    assert calls == ["frozen"] and not (tmp_path / "out").exists()


def test_changed_frozen_yaml_refused_by_check_frozen(tmp_path):
    rel = "configs/frozen_r9_btsc.yaml"
    (tmp_path / "configs").mkdir()
    src = (ROOT / rel).read_bytes()
    (tmp_path / rel).write_bytes(src)
    want = {rel: FZ.FROZEN_BLOBS[rel]}
    assert FZ.check_frozen(root=tmp_path, blobs=want)[rel] == want[rel]
    (tmp_path / rel).write_bytes(src.replace(b"primary_alpha: '0.03'", b"primary_alpha: '0.02'"))
    with pytest.raises(AssertionError, match="differs from its frozen blob ID"):
        FZ.check_frozen(root=tmp_path, blobs=want)


def test_real_frozen_set_has_four_files_and_the_yaml():
    blobs = FZ.check_frozen()
    assert len(blobs) == 4 and R6.YAML_PATH.as_posix() in blobs


def test_main_refuses_overwrite(tmp_path):
    out = tmp_path / "stage6"
    out.mkdir()
    (out / "stage6_summary.json").write_text("{}")
    with pytest.raises(SystemExit):
        R6.main(["--stage4", str(tmp_path / "none"), "--out", str(out)])
