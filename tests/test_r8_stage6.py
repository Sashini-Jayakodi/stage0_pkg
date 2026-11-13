"""Tests for POC-2 Stage 6 (src/r8_stage6_methods.py, src/r8_stage6.py, src/r8_stage6_design.py) on synthetic data.
Brute-force references are literal transcriptions of Ringel et al. 2024 (Algorithm A.3, Algorithms 1 and 2) and of the
Boursinos and Koutsoukos 2022 formulas, written with plain loops and no scipy where a p-value is involved."""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src import r8_stage4 as S4  # noqa: E402
from src import r8_stage6 as S6  # noqa: E402
from src import r8_stage6_design as B5D  # noqa: E402
from src import r8_stage6_methods as M  # noqa: E402
from src.util import sha256_file  # noqa: E402
import test_r8_stage5 as T5  # noqa: E402

rng = np.random.default_rng(41)


# ------------------------------------------------------------------ synthetic Ringel tracks
def ringel_tracks(n, tmax=30, seed=0):
    r = np.random.default_rng(seed)
    pis, preds, y = [], [], r.integers(0, 4, n)
    for i in range(n):
        L = tmax - (1 if i == 0 else 0)                              # one short track, as in GTSRB
        start = int(r.integers(0, 3))                                # frames before the first evidence
        p = np.full(L, -1.0)
        p[start:] = np.clip(np.cumsum(r.random(L - start) * 0.12), 0, 1)
        flip = r.random() < 0.3                                      # early frames sometimes wrong, later right
        pred = np.full(L, -1)
        pred[start:] = y[i]
        if flip:
            k = int(r.integers(start, L))
            pred[start:k] = (y[i] + 1) % 4
        if r.random() < 0.08:                                        # some tracks wrong to the end
            pred[start:] = (y[i] + 2) % 4
        if r.random() < 0.03:                                        # a track that never has evidence
            p[:] = -1.0
            pred[:] = -1
        pis.append(p)
        preds.append(pred)
    yfull = np.array([q[-1] for q in preds])
    return pis, preds, y, yfull


def brute_halt(p, lam_vec):
    for t in range(len(p)):
        if p[t] >= lam_vec[t]:
            return t
    return len(p) - 1


def brute_loss(y, yfull, ystop):
    return max(0, int(y == yfull) - int(y == ystop))


def brute_binom_cdf(k, n, a):
    return float(sum(math.comb(n, i) * a ** i * (1 - a) ** (n - i) for i in range(k + 1)))


# ------------------------------------------------------------------ Ringel: marginal (Algorithm A.3)
def brute_marginal(pis, preds, y, yfull, alpha, delta):
    lam_hat = math.inf
    for j in range(100, -1, -1):                                     # lambda = 1, 0.99, ..., 0
        lam = j / 100
        k = sum(brute_loss(y[i], yfull[i], preds[i][brute_halt(pis[i], [lam] * len(pis[i]))]) for i in range(len(pis)))
        if brute_binom_cdf(k, len(pis), alpha) > delta:
            break
        lam_hat = lam
    return lam_hat


@pytest.mark.parametrize("seed,alpha", [(0, 0.05), (1, 0.1), (2, 0.2), (3, 0.01)])
def test_marginal_fixed_sequence_matches_algorithm_a3(seed, alpha):
    pis, preds, y, yfull = ringel_tracks(120, seed=seed)
    assert M.marginal_calibrate(pis, preds, y, yfull, alpha, 0.1) == brute_marginal(pis, preds, y, yfull, alpha, 0.1)


# ------------------------------------------------------------------ Ringel: conditional (Algorithms 1 and 2)
def brute_alg1(pis, preds, y, yfull, alpha, tmax):
    lam = [math.inf] * tmax
    for t in range(1, tmax + 1):
        lam[t - 1] = math.inf
        for j in range(0, 101):
            lam[t - 1] = j / 100
            I = [i for i in range(len(pis)) if brute_halt(pis[i], lam) <= t - 1]
            if not I:
                lam[t - 1] = math.inf
                break
            R = sum(brute_loss(y[i], yfull[i], preds[i][brute_halt(pis[i], lam)]) for i in I) / len(I)
            if R <= alpha:
                break
        else:
            lam[t - 1] = math.inf                                    # no lambda met the target
    return np.array(lam)


def brute_alg2(pis, preds, y, yfull, cand, alpha, delta, tmax):
    lam_hat = [math.inf] * tmax
    for t in range(tmax, 0, -1):
        conf = [math.inf] * (t - 1) + list(cand[t - 1:])
        for tp in range(t, tmax + 1):
            I = [i for i in range(len(pis)) if brute_halt(pis[i], conf) <= tp - 1]
            if not I:
                return np.array(lam_hat)
            k = sum(brute_loss(y[i], yfull[i], preds[i][brute_halt(pis[i], conf)]) for i in I)
            if brute_binom_cdf(k, len(I), alpha) > delta:
                return np.array(lam_hat)
        lam_hat = conf
    return np.array(lam_hat)


@pytest.mark.parametrize("seed,alpha", [(0, 0.1), (1, 0.2), (4, 0.05)])
def test_conditional_follows_algorithms_1_and_2(seed, alpha):
    pis, preds, y, yfull = ringel_tracks(80, seed=seed)
    cand = M.stage1_screen(pis, preds, y, yfull, alpha, 30)
    assert np.array_equal(cand, brute_alg1(pis, preds, y, yfull, alpha, 30))
    lam = M.stage2_test(pis, preds, y, yfull, cand, alpha, 0.1, 30)
    assert np.array_equal(lam, brute_alg2(pis, preds, y, yfull, cand, alpha, 0.1, 30))
    idx = np.arange(80)
    full, c = M.conditional_calibrate(pis, preds, y, yfull, idx[:40], idx[40:], alpha, 0.1, 30)
    sub = lambda I: ([pis[i] for i in I], [preds[i] for i in I], y[I], yfull[I])
    assert np.array_equal(c, brute_alg1(*sub(idx[:40]), alpha, 30))
    assert np.array_equal(full, brute_alg2(*sub(idx[40:]), c, alpha, 0.1, 30))


def test_ringel_test_metrics_brute_force():
    pis, preds, y, yfull = ringel_tracks(60, seed=7)
    sizes = [np.arange(len(p)) * 3.0 + 15 for p in pis]
    lam = np.full(30, 0.5)
    r = M.ringel_test(pis, preds, y, yfull, lam, sizes)
    h = [brute_halt(p, lam) for p in pis]
    L = [brute_loss(y[i], yfull[i], preds[i][h[i]]) for i in range(60)]
    com = [preds[i][h[i]] >= 0 for i in range(60)]
    order = sorted(range(60), key=lambda i: h[i])                    # stable: ties keep track order
    assert r["gap_marginal"] == pytest.approx(np.mean(L))
    assert r["gap_20pct_earliest"] == pytest.approx(np.mean([L[i] for i in order[:12]]))
    assert r["gap_50pct_earliest"] == pytest.approx(np.mean([L[i] for i in order[:30]]))
    assert r["coverage"] == pytest.approx(np.mean(com))
    assert r["median_commit_frame"] == np.median([h[i] + 1 for i in range(60) if com[i]])


def test_no_stop_before_evidence_and_never_escalates():
    p = np.array([-1.0, -1.0, 0.2, 0.9])
    assert M.halt_index(p, np.zeros(4)) == 2                         # lambda = 0 still needs evidence
    assert M.halt_index(p, np.full(4, np.inf)) == 3                  # always stops by the last frame


# ------------------------------------------------------------------ B6 building blocks
def test_nonconformity_and_icp_pvalues_brute_force():
    D2 = rng.random((50, 6)) * 20
    nc = M.nonconformity(D2)
    d = np.sqrt(D2)
    for i in range(50):
        for j in range(6):
            assert nc[i, j] == pytest.approx(d[i, j] / min(d[i, k] for k in range(6) if k != j))
    cal = np.sort(rng.random(200) * 2)
    P = M.icp_pvalues(nc, cal)
    for i in range(0, 50, 7):
        for j in range(6):
            assert P[i, j] == sum(c >= nc[i, j] for c in cal) / 200


def test_ecdf_cred_conf_aurc_ece_brute_force():
    vals = np.sort(rng.random(100))
    assert M.ecdf(vals, 0.5) == sum(v <= 0.5 for v in vals) / 100
    pc = rng.random((20, 5))
    c1, c2, lab = M.cred_conf(pc)
    for i in range(20):
        s = sorted(pc[i])
        assert c1[i] == s[-1] and c2[i] == 1 - s[-2] and lab[i] == int(np.argmax(pc[i]))
    k, cor = rng.random(40), rng.random(40) < 0.8
    order = sorted(range(40), key=lambda i: -k[i])
    brute = np.mean([sum(not cor[order[j]] for j in range(i + 1)) / (i + 1) for i in range(40)])
    assert M.aurc(k, cor) == pytest.approx(brute)
    P, y = rng.random((30, 4)), rng.integers(0, 4, 30)
    acc = [np.mean([P[i, y[i]] > e for i in range(30)]) for e in M.EPS]
    assert M.ece(P, y) == pytest.approx(np.mean([abs(a + e - 1) for a, e in zip(acc, M.EPS)]))


def test_theta_rule_brute_force():
    ks = [rng.random(int(rng.integers(0, 8))) for _ in range(40)]
    ls = [rng.integers(0, 3, len(k)) for k in ks]
    y = rng.integers(0, 3, 40)
    cands, com, wr = M.theta_table(ks, ls, y)
    for r in (0.0, 0.2, 0.5):
        th = M.choose_theta(cands, com, wr, r)
        best = None
        for c in sorted(set([-np.inf] + [v for k in ks for v in k])):
            n = w = 0
            for kk, ll, yy in zip(ks, ls, y):
                hit = [j for j in range(len(kk)) if kk[j] > c]
                if hit:
                    n += 1
                    w += ll[hit[0]] != yy
            if n and w / n <= r + 1e-12:
                best = c
                break
        assert th == best


def test_b6_parts_are_cert_thirds():
    cert = np.sort(rng.choice(1046, 784, replace=False))
    p = M.b6_parts(cert, 0)
    assert [len(p[k]) for k in ("icp", "ecdf", "select")] == [261, 261, 262]
    allp = np.concatenate(list(p.values()))
    assert len(np.unique(allp)) == 784 and set(allp) == set(cert)


def test_b6_never_reads_a_test_track():
    """Poison every non-Cert track (nonconformity scores and labels): the fitted B6 model must not change."""
    T, F, C = 60, 12, 5
    rows = [np.arange(i * F, (i + 1) * F) for i in range(T)]
    passed = [r[rng.random(F) < 0.85] for r in rows]
    y = rng.integers(0, C, T)
    nc = rng.random((T * F, C)) + 0.5
    for i in range(T):
        nc[rows[i], y[i]] -= 0.4                                    # the true class conforms better
    cert = np.arange(45)
    parts = M.b6_parts(cert, 3)
    m1 = M.b6_fit(nc, passed, y, parts, 3, "sum", [0.05, 0.2])
    nc2, y2 = nc.copy(), y.copy()
    for i in range(45, T):                                           # the 'Test' tracks
        nc2[rows[i]] = rng.random((F, C)) * 9
        y2[i] = (y[i] + 1) % C
    m2 = M.b6_fit(nc2, passed, y2, parts, 3, "sum", [0.05, 0.2])
    for k in ("cal", "ecdf"):
        assert np.array_equal(m1[k], m2[k])
    assert (m1["a"], m1["b"], m1["theta"]) == (m2["a"], m2["b"], m2["theta"])


def test_b6_decision_commits_at_first_window_above_theta():
    seq = (np.array([0.1, 0.7, 0.9]), np.array([2, 1, 0]), np.array([10, 11, 12]))
    assert M.b6_decide(seq, 0.5) == (1, 11)
    assert M.b6_decide(seq, 0.95) == (-1, -1) and M.b6_decide(seq, None) == (-1, -1)


def test_papers_hash_check_refuses_other_files(tmp_path):
    for rel in B5D.PAPERS:
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_bytes(b"%PDF-not the paper")
    with pytest.raises(AssertionError, match="differs from the hash in Stage 6 Part A"):
        B5D.check_papers(tmp_path)


def test_design_score_transform():
    F = np.sort(rng.random(500) * 100)
    s = B5D.score(np.array([-np.inf, 0.0, 50.0, 1e9]), F)
    assert s[0] == -1.0 and s[1] == 0.0 and s[2] == pytest.approx(np.mean(F <= 50.0)) and s[3] == 1.0


# ------------------------------------------------------------------ end to end on a fake run folder
def make_6(tmp, monkeypatch):
    run, cfgp = T5.make_5(tmp, monkeypatch)
    s3, s6 = run / "poc2" / "stage3", run / "poc2" / "stage6"
    s6.mkdir(parents=True, exist_ok=True)
    (s6 / "b5_design.json").write_text(json.dumps({"F_margins": list(np.sort(rng.random(300) * 50))}))
    res = {a: {"track": {"violation_rate": 0.01, "cp95": [0.0, 0.02], "certified_share": 1.0},
               "B1": {"violation_rate": 0.2, "cp95": [0.18, 0.22], "certified_share": 1.0}} for a in ("0.01", "0.3", "0.5")}
    (s3 / "stage3b_summary.json").write_text(json.dumps({"results": res}))
    (s3 / "stage3a_summary.json").write_text(json.dumps({"rows": [{"rho_sim": 0.45, "track": {"violation_rate": 0.002},
                                                                   "B1": {"violation_rate": 0.05}}]}))
    s4p = run / "poc2" / "stage4" / "stage4_summary.json"
    for name, p in (("STAGE4_SUMMARY_SHA256", s4p), ("B5_DESIGN_SHA256", s6 / "b5_design.json"),
                    ("STAGE3A_SHA256", s3 / "stage3a_summary.json"), ("STAGE3B_SHA256", s3 / "stage3b_summary.json")):
        monkeypatch.setattr(S6, name, sha256_file(p))
    monkeypatch.setattr(S6, "STAGE4_CODE_SHA256", sha256_file(Path(S4.__file__)))
    monkeypatch.setattr(S6, "FROZEN_ARGS", S4.FROZEN_ARGS)
    monkeypatch.setattr(S6, "check_papers", lambda: None)            # the PDFs are git-ignored; checked separately
    monkeypatch.setattr(S6, "TABLE_ALPHAS", (0.01, 0.3))
    return run, cfgp


def test_stage6_end_to_end_on_fake_run(tmp_path, monkeypatch):
    run, cfgp = make_6(tmp_path, monkeypatch)
    s = S6.main(["--run", str(run), "--config", str(cfgp)])
    assert s["all_pass"], s["checks"]
    assert len(s["per_seed"]) == 5 and set(s["stage6_checks"]) == {"B5_own_target", "B5c_own_target", "B6_calibration_direction_K5"}
    t = s["tables"]["main_0.3"]
    assert t["columns"][:7] == ["Ours", "B1", "B2", "B3", "B4", "B5", "B5c"] and len(t["rows"]) == 8
    assert (run / "poc2" / "stage6" / "stage6_summary.json").exists()


def test_stage6_refuses_changed_inputs(tmp_path, monkeypatch):
    run, cfgp = make_6(tmp_path, monkeypatch)
    p = run / "poc2" / "stage6" / "b5_design.json"
    p.write_text(p.read_text() + " ")
    with pytest.raises(AssertionError, match="b5_design.json differs"):
        S6.main(["--run", str(run), "--config", str(cfgp)])


def test_stage6_stops_if_stage4_does_not_reproduce(tmp_path, monkeypatch):
    run, cfgp = make_6(tmp_path, monkeypatch)
    p = run / "poc2" / "stage4" / "stage4_summary.json"
    s4 = json.loads(p.read_text())
    done = False
    for r in s4["per_seed"]:
        for by_m in r["alphas"].values():
            if by_m["commit"]["certified"] and not done:
                by_m["commit"]["coverage"] += 0.01
                done = True
    assert done
    p.write_text(json.dumps(s4))
    monkeypatch.setattr(S6, "STAGE4_SUMMARY_SHA256", sha256_file(p))
    with pytest.raises(AssertionError, match="do not reproduce"):
        S6.main(["--run", str(run), "--config", str(cfgp)])
