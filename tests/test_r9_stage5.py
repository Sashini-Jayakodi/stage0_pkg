"""Tests for R9 Stage 5 on synthetic data: sequences and the variable-window S2 against brute-force references,
per-length ECDF fit, Rule 3 against a loop reference and its arithmetic limits, re-splits, the ANOVA ICC and the
pole bootstrap, B2 and B6, input hash checks with tamper tests, refusal of pool files, the label guard and the design
step end to end. No real BelgiumTSC file is read."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import beta, binom

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import r9_stage4_data as D4  # noqa: E402
from src import r9_stage5_design as DS  # noqa: E402
from src import r9_stage5_scores as SC  # noqa: E402
from src.head import score  # noqa: E402
from src.r9_btsc_guard import GuardedTestLabels  # noqa: E402
from src.r9_frozen import blob_id  # noqa: E402
from src.r9_stage4_model import fit_head_k  # noqa: E402
from src.util import sha256_file  # noqa: E402


# ---------------------------------------------------------------- synthetic per-view inputs
def synth_V(n_views=40, K=6, seed=0):
    rng = np.random.default_rng(seed)
    D2 = rng.gamma(2.0, 3.0, size=(n_views, K))
    priors = rng.dirichlet(np.ones(K))
    ll = -0.5 * D2
    lp = np.log(priors) + ll
    s = np.sort(lp, 1)
    from src.r8_stage6_methods import nonconformity
    return {"D2": D2, "ll": ll, "log_prior": np.log(priors), "m": s[:, -1] - s[:, -2], "top": lp.argmax(1),
            "nc": nonconformity(D2)}


def test_view_inputs_match_head_score():
    rng = np.random.default_rng(1)
    K, d = 5, 8
    emb = rng.normal(size=(200, d))
    y = np.repeat(np.arange(K), 40)
    emb += 2 * np.eye(K, d)[y]
    head = fit_head_k(emb, y, K)
    V = SC.view_inputs(head, emb)
    sc = score(head, emb)
    assert np.array_equal(V["top"], sc["top3"][:, 0])
    assert np.allclose(V["m"], sc["margin"], atol=1e-4)


def test_unit_rows_order():
    pole = [5, 5, 3, 5, 3, 5]
    key = [1, 1, 2, 0, 2, 1]
    view = [9, 2, 4, 1, 0, 5]
    units, rows = SC.unit_rows(pole, key, view)
    assert units == [(3, 2), (5, 0), (5, 1)]
    assert [r.tolist() for r in rows] == [[4, 2], [3], [1, 5, 0]]


def test_s1_s3_s4_bruteforce():
    V = synth_V()
    r = np.array([3, 7, 1, 20])
    m1, l1 = SC.seq_s1(V, r)
    for v in range(len(r)):
        S = V["log_prior"] + V["ll"][r[: v + 1]].sum(0)
        o = np.sort(S)
        assert np.isclose(m1[v], o[-1] - o[-2]) and l1[v] == np.argmax(S)
    m3, l3 = SC.seq_s3(V, r)
    for v in range(len(r)):
        cnt = np.bincount(V["top"][r[: v + 1]], minlength=V["ll"].shape[1])
        best = min(range(len(cnt)), key=lambda c: (-cnt[c], c))
        second = sorted(cnt)[-2]
        assert l3[v] == best and m3[v] == cnt[best] - second
    m4, l4 = SC.seq_s4(V, r)
    assert np.array_equal(m4, V["m"][r]) and np.array_equal(l4, V["top"][r])


def test_windows_q2():
    assert [w.tolist() for w in SC.windows(5)] == [[0], [0, 1], [0, 1, 2], [1, 2, 3], [2, 3, 4]]
    assert [w.tolist() for w in SC.windows(1)] == [[0]]


def brute_s2(V, r, model):
    """Loop reference: ICP p-value per class, window of at most 3 views, ECDF of the window's length."""
    cal = model["cal"]
    K = V["nc"].shape[1]
    conf, lab = [], []
    for v in range(len(r)):
        w = list(range(max(0, v - 2), v + 1))
        p = np.array([[np.mean(cal >= V["nc"][r[i], k]) for k in range(K)] for i in w])
        comb = p.sum(0) if model["combo"] == "sum" else p.min(0)
        ec = model["ecdf"][len(w)]
        pc = np.array([np.mean(ec <= c) for c in comb])
        o = np.sort(pc)
        conf.append(model["a"] * o[-1] + model["b"] * (1 - o[-2]))
        lab.append(int(np.argmax(pc)))
    return np.array(conf), np.array(lab)


@pytest.mark.parametrize("combo", ["sum", "min"])
def test_seq_s2_bruteforce(combo):
    V = synth_V(seed=3)
    rng = np.random.default_rng(4)
    model = {"cal": np.sort(rng.gamma(1.0, 1.0, 50)), "combo": combo, "a": 0.3, "b": 0.8,
             "ecdf": {L: np.sort(rng.uniform(0, L if combo == "sum" else 1, 30)) for L in (1, 2, 3)}}
    for r in (np.array([0]), np.array([4, 2]), np.array([1, 5, 9]), np.array([2, 8, 11, 30, 31])):
        c, l = SC.seq_s2(V, r, model)
        cb, lb = brute_s2(V, r, model)
        assert np.allclose(c, cb, atol=1e-12) and np.array_equal(l, lb)


def test_fit_s2_ecdf_per_length_bruteforce():
    V = synth_V(n_views=60, seed=5)
    lengths = [1, 2, 3, 3, 4, 3, 5, 3, 2, 3, 3, 3, 3, 3, 3, 3, 3, 1]
    rows, start = [], 0
    for L in lengths:
        rows.append(np.arange(start, start + L))
        start += L
    y = np.array([int(V["top"][r[0]]) for r in rows])
    A, B = np.arange(0, 8), np.arange(8, len(rows))
    m = SC.fit_s2(V, rows, y, A, B, "sum")
    cal = np.sort(np.concatenate([V["nc"][rows[i], y[i]] for i in A]))
    assert np.array_equal(m["cal"], cal)
    want = {1: [], 2: [], 3: []}
    for i in B:
        for v in range(len(rows[i])):
            w = list(range(max(0, v - 2), v + 1))
            p = [np.mean(cal >= V["nc"][rows[i][j], y[i]]) for j in w]
            want[len(w)].append(sum(p))
    for L in (1, 2, 3):
        assert np.allclose(m["ecdf"][L], np.sort(want[L]))
    assert -1 <= m["a"] <= 1 and -1 <= m["b"] <= 1


def test_fit_s2_stops_without_length3():
    V = synth_V(seed=6)
    rows = [np.array([0]), np.array([1, 2]), np.array([3]), np.array([4, 5])]
    y = np.array([0, 0, 0, 0])
    with pytest.raises(AssertionError):
        SC.fit_s2(V, rows, y, np.array([0]), np.array([1, 2, 3]), "sum")


# ---------------------------------------------------------------- Rule 3
def kmax_ref(nc, alpha, delta=0.1):
    if nc <= 0 or binom.cdf(0, nc, alpha) > delta:
        return None
    k = 0
    while binom.cdf(k + 1, nc, alpha) <= delta:
        k += 1
    return k


def test_rule3_against_loop_reference():
    rng = np.random.default_rng(7)
    for _ in range(30):
        nd = 157
        com = np.sort(rng.integers(30, nd + 1, 12))[::-1]
        wrong = np.minimum(rng.integers(0, 6, 12), com)
        stats = {"n": nd, "committed": com, "wrong": wrong}
        n_min = int(rng.integers(440, 500))
        for a in SC.ALPHAS + (0.10,):
            r = SC.rule3(stats, n_min, a)
            ok_ref = []
            for c, w in zip(com, wrong):
                if c / nd < 0.70 - 1e-12:
                    continue
                nc = int(np.floor(c / nd * n_min + 1e-9))
                u = 1.0 if w >= c else beta.ppf(0.9, w + 1, c - w)
                km = kmax_ref(nc, a)
                ok_ref.append(km is not None and u * nc <= km)
            assert r["predicted"] == any(ok_ref)
            assert len(r["points"]) == len(ok_ref)


def zero_error_stats(nd=157, wrong=0):
    com = np.arange(nd, int(np.ceil(0.7 * nd)) - 1, -1)
    return {"n": nd, "committed": com, "wrong": np.minimum(wrong, com)}


@pytest.mark.parametrize("n_min", [460, 470, 480])
def test_rule3_arithmetic_limits(n_min):
    """Part A (i): with 157 design units, 0.5/1/2% are never predicted; 3% needs 0 errors; 5% allows at most 2."""
    for a in (0.005, 0.01, 0.02):
        assert not SC.rule3(zero_error_stats(), n_min, a)["predicted"]
    assert SC.rule3(zero_error_stats(), n_min, 0.03)["predicted"]
    assert not SC.rule3(zero_error_stats(wrong=1), n_min, 0.03)["predicted"]
    assert SC.rule3(zero_error_stats(wrong=2), n_min, 0.05)["predicted"]
    assert not SC.rule3(zero_error_stats(wrong=3), n_min, 0.05)["predicted"]


def test_rule3_needs_70_percent_and_reports_deciding_point():
    stats = {"n": 100, "committed": np.array([100, 80, 69]), "wrong": np.array([10, 0, 0])}
    r = SC.rule3(stats, 470, 0.05)
    assert [p["j"] for p in r["points"]] == [0, 1]                  # 69% is below 70%
    assert r["predicted"] and r["deciding"]["j"] == 1
    assert SC.primary_alpha({"S1": {0.03: {"predicted": False}, 0.05: {"predicted": True}},
                             "S2": {0.03: {"predicted": False}, 0.05: {"predicted": False}}}, (0.03, 0.05)) == 0.05
    assert SC.primary_alpha({"S1": {0.05: {"predicted": False}}}, (0.05,)) is None


def test_starts_use_each_seed_n():
    f_d = np.linspace(1.0, 0.3, 30)
    st = SC.starts(f_d, {0: 450, 1: 500}, 157, (0.02,))
    from src.r9_certify import power_start
    assert st[0][0.02]["binom"] == power_start(f_d, 450, 0.02)
    assert st[1][0.02]["binom"] == power_start(f_d, 500, 0.02)


# ---------------------------------------------------------------- re-splits
def test_resplits_by_pole():
    rng = np.random.default_rng(8)
    units = sorted({(int(p), int(c)) for p, c in zip(rng.integers(0, 120, 200), rng.integers(0, 15, 200))})
    rs = SC.resplits(units, seeds=(0, 1))
    poles = {p for p, _ in units}
    for s, r in rs.items():
        assert len(r["test_poles"]) == D4.round_half_up(len(poles), 4)
        assert set(r["test_poles"]) | set(r["cert_poles"]) == poles
        assert not set(r["test_poles"]) & set(r["cert_poles"])
        assert r["n_cert_units"] == sum(p in set(r["cert_poles"]) for p, _ in units)
        assert r["n_cert_units"] + r["n_test_units"] == len(units)
        assert r["test_poles"] == D4.split_by_pole(units, (1, 4), s)
    assert rs[0]["test_poles"] != rs[1]["test_poles"]


# ---------------------------------------------------------------- correlation
def icc_ref(groups):
    """Direct one-way ANOVA ICC from the 0/1 values per cluster (unequal sizes)."""
    allv = np.concatenate(groups)
    N, k = len(allv), len(groups)
    gm = allv.mean()
    msb = sum(len(g) * (g.mean() - gm) ** 2 for g in groups) / (k - 1)
    msw = sum(((g - g.mean()) ** 2).sum() for g in groups) / (N - k)
    n0 = (N - sum(len(g) ** 2 for g in groups) / N) / (k - 1)
    return (msb - msw) / (msb + (n0 - 1) * msw)


def test_rho_matches_direct_anova_and_bootstrap_weights():
    rng = np.random.default_rng(9)
    groups = [(rng.random(int(n)) < rng.beta(1, 8)).astype(float) for n in rng.integers(1, 5, 80)]
    n = np.array([len(g) for g in groups])
    e = np.array([g.sum() for g in groups])
    pole = np.repeat(np.arange(40), 2)
    r = SC.rho_ci(n, e, pole, (1, 2, 3), n_draws=200)
    assert np.isclose(r["rho"], icc_ref(groups))
    assert r["ci95"][0] <= r["ci95"][1]
    W = SC.pole_boot_weights(pole, 50, (1, 2, 3))
    assert W.shape == (50, 80) and (W[:, 0::2] == W[:, 1::2]).all()   # both clusters of a pole share its count
    assert (W[:, 0::2].sum(1) == 40).all()                              # 40 poles drawn per bootstrap sample
    assert np.isclose(SC.design_effect(0.45, 3), 1.9)


# ---------------------------------------------------------------- baselines
def test_b2_commits_only_at_drawn_view():
    V = synth_V(seed=10)
    r = np.array([2, 5, 9])
    conf, lab = SC.seq_b2(V, r, 1)
    assert np.isneginf(conf[[0, 2]]).all() and conf[1] == V["m"][5] and lab[1] == V["top"][5]
    grid = np.array([-np.inf, conf[1] - 1, conf[1] + 1])
    tab = SC.track_table([(conf, lab)], np.array([lab[1]]), grid)
    assert tab["committed"][0].tolist() == [True, True, False] and tab["pos"][0, 0] == 2
    k = SC.b2_draw([1, 3, 3, 4], np.random.default_rng(0))
    assert k[0] == 0 and (k < np.array([1, 3, 3, 4])).all()


def test_b6_theta_bruteforce():
    rng = np.random.default_rng(11)
    seqs = [(rng.random(3), rng.integers(0, 3, 3)) for _ in range(40)]
    y = rng.integers(0, 3, 40)
    th = SC.b6_theta(seqs, y, (0.05, 0.2))
    cands = np.unique(np.concatenate([[-np.inf]] + [s[0] for s in seqs]))
    for a, t in th.items():
        best = None
        for c in cands:
            com = wr = 0
            for (k, l), yy in zip(seqs, y):
                hit = np.flatnonzero(k > c)
                if len(hit):
                    com += 1
                    wr += l[hit[0]] != yy
            if com and wr <= a * com + 1e-12:
                best = c
                break
        assert (t is None and best is None) or np.isclose(t, best)


# ---------------------------------------------------------------- inputs: hash checks and tamper tests
def make_stage4(tmp, n_design_poles=30, n_pool_poles=60, K=8, d=12, seed=0):
    """A small synthetic Stage 4 output folder: view table, split lists, design embeddings, head, design scores."""
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
                                 "size": 20 + v, "part": side})
            p += 1
    D4.write_table(base / "test_views.csv", rows)
    design = sorted({r["pole"] for r in rows if r["part"] == "design"})
    pool = sorted({r["pole"] for r in rows if r["part"] == "pool"})
    (base / "split_poles.json").write_text(json.dumps({"design_poles": design, "pool_poles": pool}, indent=1) + "\n")
    fit_y = np.repeat(np.arange(K), 30)
    means = rng.normal(size=(K, d)) * 2.5
    fit_emb = means[fit_y] + rng.normal(size=(len(fit_y), d))
    head = fit_head_k(fit_emb, fit_y, K)
    np.savez(base / "head" / "head.npz", mu=head["mu"], W=head["W"], prec=head["prec"], priors=head["priors"],
             shrinkage=head["shrinkage"])
    di = np.array([i for i, r in enumerate(rows) if r["part"] == "design"])
    y = np.array([rows[i]["cls"] for i in di])
    emb = (means[y] + rng.normal(size=(len(di), d)) * 1.3).astype(np.float32)
    get = lambda k: np.array([rows[i][k] for i in di])
    np.savez(base / "embeddings" / "test_design.npz", emb=emb, row=di, pole=get("pole"), view=get("view"),
             size=get("size"), unit=np.arange(len(di)), y=y)
    sc = score(head, emb)
    np.savez(base / "head" / "scores_test_design.npz", top3=sc["top3"], lp3=sc["lp3"], margin=sc["margin"],
             s=sc["s"], row=di, pole=get("pole"), view=get("view"), size=get("size"), y=y)
    files = {rel: sha256_file(base / rel) for rel in DS.INPUTS}
    (base / "stage4_summary.json").write_text(json.dumps({"files_sha256": files}, indent=2) + "\n")
    blobs = {rel: blob_id(base / rel) for rel in ("stage4_summary.json", "split_poles.json", "test_views.csv")}
    return base, sha256_file(base / "stage4_summary.json"), blobs


def test_load_stage4_ok_and_records_files(tmp_path):
    base, s4, blobs = make_stage4(tmp_path)
    DS.OPENED.clear()
    d = DS.load_stage4(base, s4, blobs)
    assert set(DS.OPENED) == {"stage4_summary.json", *DS.INPUTS}
    assert all(v == "sha256" for v in d["checks"].values())


def test_line_ending_only_change_passes_by_blob(tmp_path):
    base, s4, blobs = make_stage4(tmp_path)
    p = base / "test_views.csv"
    b = p.read_bytes()
    p.write_bytes(b.replace(b"\r\n", b"\n") if b"\r\n" in b else b.replace(b"\n", b"\r\n"))
    d = DS.load_stage4(base, s4, blobs)
    assert d["checks"]["test_views.csv"] == "blob"


@pytest.mark.parametrize("rel", ["test_views.csv", "split_poles.json", "embeddings/test_design.npz",
                                 "head/head.npz", "head/scores_test_design.npz", "stage4_summary.json"])
def test_tampered_input_refused(tmp_path, rel):
    base, s4, blobs = make_stage4(tmp_path)
    p = base / rel
    b = bytearray(p.read_bytes())
    i = len(b) // 2
    while b[i] in b"\r\n":                         # change a real character, not a line ending
        i += 1
    b[i] = (b[i] + 1) % 256
    p.write_bytes(bytes(b))
    with pytest.raises(SystemExit):
        DS.load_stage4(base, s4, blobs)


@pytest.mark.parametrize("rel", ["embeddings/test_pool.npz", "head/scores_test_pool.npz", "cache/test_112.npy",
                                 "model/best.pt"])
def test_pool_and_model_files_refused(tmp_path, rel):
    with pytest.raises(AssertionError):
        DS.open_input(tmp_path, rel, {"files_sha256": {rel: "0"}})


def test_guard_refuses_pool_labels():
    g = GuardedTestLabels([1, 1, 2, 3], [5, 5, 6, 7], design_poles=[1, 2], blobs={})
    assert g.labels([0, 2]).tolist() == [5, 6]
    with pytest.raises(AssertionError):
        g.labels([3])


# ---------------------------------------------------------------- design step end to end (synthetic)
def test_run_design_synthetic(tmp_path, monkeypatch):
    base, s4, blobs = make_stage4(tmp_path, n_design_poles=60, n_pool_poles=120)
    d = DS.load_stage4(base, s4, blobs)
    rows = d["rows"]
    # facts of this synthetic set replace the real Stage 4 counts
    E = d["emb"]
    sc = score(d["head"], E["emb"])
    units = {(int(p), int(c)) for p, c in zip(E["pole"], E["y"])}
    last = {}
    for i, u in enumerate(zip(E["pole"].tolist(), E["y"].tolist())):
        if u not in last or E["view"][i] > E["view"][last[u]]:
            last[u] = i
    pool_units = {(r["pole"], r["cls"]) for r in rows if r["part"] == "pool"}
    monkeypatch.setattr(DS, "EXPECTED", {
        "design_poles": len(set(E["pole"].tolist())), "design_units": len(units), "design_views": len(E["row"]),
        "pool_poles": len({p for p, _ in pool_units}), "pool_units": len(pool_units),
        "views_correct": int((sc["top3"][:, 0] == E["y"]).sum()),
        "units_correct_last_view": int(sum(sc["top3"][i, 0] == u[1] for u, i in last.items()))})
    monkeypatch.setattr(SC, "BOOT_DRAWS", 100)
    R = DS.run_design(d)
    assert all(v for k, v in R["integrity"].items() if not k.endswith("max_abs"))
    nd = R["facts"]["design_units"]
    for s in SC.SCORES:
        g, st = R["scores"][s]["grid"], R["scores"][s]["stats"]
        assert np.isneginf(g[0]) and np.all(np.diff(g) > 0) and len(g) <= SC.N_POINTS
        assert st["n"] == nd and st["committed"][0] == nd                  # no gate: every unit commits at lambda_0
        for seed in SC.SEEDS:
            for a in R["alphas"]:
                assert set(R["scores"][s]["starts"][seed][a]) == set(SC.RULES)
    assert len(R["scores"]["S3"]["grid"]) <= 5
    assert R["n_min"] == min(r["n_cert_units"] for r in R["resplits"].values())
    assert R["fd1"] == any(R["rule3"][s][a]["predicted"] for s in SC.SCORES for a in SC.FD1_ALPHAS)
    if not R["fd1"]:
        assert SC.ALPHA_EXTRA in R["alphas"]
    h = R["halves"]
    assert not set(h["A_poles"]) & set(h["B_poles"]) and h["A_units"] + h["B_units"] == nd
    assert len(R["b5"]["f_design"]) == R["facts"]["design_views"]
    json.dumps(DS.to_json(R, d, {}, {}, 0.0))                              # the output is valid JSON


def test_main_refuses_overwrite(tmp_path, monkeypatch):
    monkeypatch.setattr(DS, "f3_closed", lambda: False)        # pre-F3 state, to reach the overwrite check
    out = tmp_path / "stage5"
    out.mkdir()
    (out / "stage5_design.json").write_text("{}")
    with pytest.raises(SystemExit):
        DS.main(["--stage4", str(tmp_path / "none"), "--out", str(out)])


def test_design_step_refused_after_f3(tmp_path):
    # the real repo is past F3: the design phase is over and the design step must not run
    with pytest.raises(AssertionError, match="F3 is closed"):
        DS.main(["--stage4", str(tmp_path / "none"), "--out", str(tmp_path / "out")])


# ---------------------------------------------------------------- freeze step (src/r9_stage5_freeze.py)
from src import r9_stage5_freeze as FR  # noqa: E402
import yaml  # noqa: E402


def synthetic_design(tmp_path, monkeypatch):
    """A synthetic Stage 4 folder, its design run R and the recorded design file (as the design step writes it)."""
    base, s4, blobs = make_stage4(tmp_path, n_design_poles=60, n_pool_poles=120)
    d = DS.load_stage4(base, s4, blobs)
    monkeypatch.setattr(DS, "STAGE4_SUMMARY_SHA256", s4)
    monkeypatch.setattr(DS, "TEXT_BLOBS", blobs)
    monkeypatch.setattr(SC, "BOOT_DRAWS", 50)
    monkeypatch.setattr(DS, "stop", lambda m: (_ for _ in ()).throw(SystemExit(m)))
    E = d["emb"]
    sc = score(d["head"], E["emb"])
    last = {}
    for i, u in enumerate(zip(E["pole"].tolist(), E["y"].tolist())):
        if u not in last or E["view"][i] > E["view"][last[u]]:
            last[u] = i
    pool_units = {(r["pole"], r["cls"]) for r in d["rows"] if r["part"] == "pool"}
    monkeypatch.setattr(DS, "EXPECTED", {
        "design_poles": len(set(E["pole"].tolist())), "design_units": len(last), "design_views": len(E["row"]),
        "pool_poles": len({p for p, _ in pool_units}), "pool_units": len(pool_units),
        "views_correct": int((sc["top3"][:, 0] == E["y"]).sum()),
        "units_correct_last_view": int(sum(sc["top3"][i, 0] == u[1] for u, i in last.items()))})
    R = DS.run_design(d)
    code = {f: sha256_file(ROOT / f) for f in DS.CODE}
    path = tmp_path / "stage5_design.json"
    path.write_text(json.dumps(DS.to_json(R, d, {"x": "y"}, code, 0.0), indent=1) + "\n")
    return base, d, R, path


def test_plain_keeps_inf():
    assert FR.plain({1: np.array([-np.inf, 1.5]), "t": np.float64(-np.inf), "n": None}) == \
        {"1": [-np.inf, 1.5], "t": -np.inf, "n": None}


def test_freeze_reproduce_and_build(tmp_path, monkeypatch):
    base, d, R, path = synthetic_design(tmp_path, monkeypatch)
    monkeypatch.chdir(ROOT)
    design, how = FR.load_design(path, sha256_file(path), blob_id(path))
    assert how == "sha256"
    FR.reproduce(R, d, design, {"x": "y"})                               # same design: passes
    y = yaml.safe_load(yaml.safe_dump(FR.build(R, design, "s", "b"), sort_keys=False, default_flow_style=None))
    for s in SC.SCORES:
        assert y["scores"][s]["grid"][0] == -np.inf
        assert y["scores"][s]["grid"][1:] == R["scores"][s]["grid"][1:].tolist()
    for s, th in R["b6"].items():                                        # -inf and None stay apart in the yaml
        for a, t in th.items():
            assert y["b6"]["theta"][s][f"{a:g}"] == t
    assert y["resplits"]["split"]["0"]["test_poles"] == R["resplits"][0]["test_poles"]
    assert y["s2_halves"]["B_poles"] == R["halves"]["B_poles"]
    assert y["primary_alpha"] == (None if R["primary_alpha"] is None else f"{R['primary_alpha']:g}")
    assert set(y["s2"]["S2-sum"]["ecdf"]) == {"1", "2", "3"}


def test_freeze_refuses_changed_design(tmp_path, monkeypatch):
    base, d, R, path = synthetic_design(tmp_path, monkeypatch)
    monkeypatch.chdir(ROOT)
    sha, blob = sha256_file(path), blob_id(path)
    obj = json.loads(path.read_text())
    obj["n_min"] += 1
    path.write_text(json.dumps(obj, indent=1) + "\n")
    with pytest.raises(SystemExit):                                      # hash check
        FR.load_design(path, sha, blob)
    design, _ = FR.load_design(path, sha256_file(path), blob_id(path))   # hash accepted, content changed
    with pytest.raises(SystemExit):                                      # reproduction check
        FR.reproduce(R, d, design, {"x": "y"})


def test_freeze_refuses_existing_yaml(tmp_path, monkeypatch):
    monkeypatch.setattr(FR, "f3_closed", lambda: False)        # pre-F3 state, to reach the overwrite check
    y = tmp_path / "frozen.yaml"
    y.write_text("x: 1\n")
    with pytest.raises(SystemExit):
        FR.main(["--yaml", str(y), "--design", str(tmp_path / "none.json")])


def test_freeze_refused_after_f3(tmp_path):
    with pytest.raises(AssertionError, match="F3 is already closed"):
        FR.main(["--yaml", str(tmp_path / "y.yaml"), "--design", str(tmp_path / "none.json")])
