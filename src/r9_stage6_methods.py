"""POC-3 (R9) Stage 6, BelgiumTSC confirmation: tables, RQ1 resampling, certification on the frozen re-splits, Test
metrics and the baselines (LOG_r9_stage6_entry.md Part A (c) to (j)). Pure numpy/scipy; no file reading, so every
part can be checked on synthetic data against a brute-force reference.

Every number that is chosen comes from configs/frozen_r9_btsc.yaml; these functions only apply frozen rules.
Losses always come from the units-by-lambdas tables (hard rule 4), never from one score per unit.
"""
import numpy as np

from src import r8_stage5 as S5
from src import r8_stage6_methods as M
from src import r9_certify as C
from src import r9_stage5_scores as SC
from src.r8_stage3_engine import TOL, cp_bounds

RQ1_RULES = ("binom", "hb")                # Part A (d): exact binomial and HB under its primary start
V_BOUND = round(C.DELTA + 0.02, 10)        # Part A (e): upper 95% CP bound of the violation rate <= 0.12
INF = np.inf


def akey(a):
    """Alpha as the yaml writes it ('0.03')."""
    return f"{a:g}"


# ------------------------------------------------------------------ tables (c)
def unit_table(seqs, y, grid, rows, size_view):
    """Units-by-lambdas table (Stage 3/5 track_table) plus the size of the committing view (NaN if not committed)."""
    tab = SC.track_table(seqs, y, grid)
    size = np.full(tab["pos"].shape, np.nan)
    for i, r in enumerate(rows):
        c = tab["committed"][i]
        size[i, c] = size_view[r[tab["pos"][i, c] - 1]]
    tab["size"] = size
    return tab


def b1_seqs(V):
    """B1: every view is its own unit with the S4 single-view margin (as the Stage 5 design step)."""
    return [(np.array([V["m"][i]], float), np.array([V["top"][i]])) for i in range(len(V["m"]))]


def true_risk(tab):
    """Population risk per grid point: wrong / committed over every population item counted once (NaN if none)."""
    com = tab["committed"].sum(0)
    w = (tab["committed"] & tab["wrong"]).sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(com > 0, w / np.maximum(com, 1), np.nan), com / tab["committed"].shape[0]


# ------------------------------------------------------------------ RQ1 resampling (d)
def draw_counts(n_poles, n_draws, per_draw, salt):
    """How often each pole (sorted pole order) is drawn in draw d: default_rng([*salt, d]).integers(0, P, per_draw)."""
    out = np.empty((n_draws, n_poles), np.int64)
    for d in range(n_draws):
        rng = np.random.default_rng([*salt, d])
        out[d] = np.bincount(rng.integers(0, n_poles, size=per_draw), minlength=n_poles)
    return out


def per_pole(tab, item_pole, n_poles):
    """Committed and wrong counts per (pole, grid point), and items per pole. item_pole: pole index of each item."""
    com = tab["committed"].astype(np.int64)
    wr = (tab["committed"] & tab["wrong"]).astype(np.int64)
    Cp = np.zeros((n_poles, com.shape[1]), np.int64)
    Wp = np.zeros_like(Cp)
    np.add.at(Cp, item_pole, com)
    np.add.at(Wp, item_pole, wr)
    return Cp, Wp, np.bincount(item_pole, minlength=n_poles)


def draw_sums(counts, Cp, Wp, n_items):
    """k(j), n_c(j) and n per draw: a pole drawn twice contributes all its items twice."""
    return counts @ Wp, counts @ Cp, counts @ n_items


def rq1_cell(K, NC, N, alpha, f_d, rule, n_design, risk):
    """One RQ1 cell: certify every draw with the frozen engine (start from the draw's n), then count violations.
    Violation (Stage 2 definition): a grid point was certified and its true risk exceeds alpha."""
    D = len(N)
    cert = np.array([C.certify(K[d], NC[d], int(N[d]), alpha, f_d, test=rule, n_design=n_design) for d in range(D)])
    start = np.array([C.start_for(rule, f_d, int(N[d]), alpha, n_design=n_design) for d in range(D)])
    ok = cert >= 0
    viol = np.zeros(D, bool)
    viol[ok] = risk[cert[ok]] > alpha + TOL
    v = int(viol.sum())
    lo, hi = cp_bounds(v, D)
    idx, cnt = np.unique(cert[ok], return_counts=True)
    return {"draws": D, "no_start": int((start < 0).sum()), "certified": int(ok.sum()), "violations": v,
            "rate": v / D, "cp95": [lo, hi],
            "mean_true_risk": float(risk[cert[ok]].mean()) if ok.any() else None,
            "index_counts": {int(i): int(c) for i, c in zip(idx, cnt)}}


def v_check(cell, bound=V_BOUND):
    """Part A (e): PASS if the upper end of the two-sided 95% CP interval of the violation rate is <= 0.12."""
    return bool(cell["cp95"][1] <= bound + TOL)


# ------------------------------------------------------------------ RQ2 on the re-splits (f), (g)
def cert_counts(tab, mask):
    com = tab["committed"][mask]
    return (com & tab["wrong"][mask]).sum(0), com.sum(0)


def certify_split(tab, cert_mask, alpha, f_d, rule, n_design):
    k, n_c = cert_counts(tab, cert_mask)
    return int(C.certify(k, n_c, int(cert_mask.sum()), alpha, f_d, test=rule, n_design=n_design))


def test_metrics(tab, j, test_mask, s_values):
    """Plan 5.4 metrics on Test units at grid index j (j < 0: nothing certified, nothing commits)."""
    nt = int(test_mask.sum())
    if j < 0:
        return {"j": None, "coverage": 0.0, "escalation": 1.0, "error": None, "cp95": None, "committed": 0,
                "wrong": 0, "median_view": None, "median_size": None, "d90": S5.NOT_REACHED, "n_test": nt}
    com, wr = tab["committed"][test_mask, j], tab["wrong"][test_mask, j]
    sz = tab["size"][test_mask, j]
    nc, k = int(com.sum()), int((com & wr).sum())
    return {"j": int(j), "coverage": nc / nt, "escalation": 1.0 - nc / nt, "error": k / nc if nc else None,
            "cp95": list(cp_bounds(k, nc)) if nc else None, "committed": nc, "wrong": k,
            "median_view": float(np.median(tab["pos"][test_mask, j][com])) if nc else None,
            "median_size": float(np.median(sz[com])) if nc else None,
            "d90": S5.d90(S5.committed_by_size(com, sz, s_values), s_values), "n_test": nt}


def smallest_alpha(cert_by_alpha, alphas):
    """Smallest alpha with a certified grid index; None if none."""
    for a in sorted(alphas):
        if cert_by_alpha[a] >= 0:
            return a
    return None


# ------------------------------------------------------------------ B5 / B5c with views as the time axis (h)
def ext(lam_vec, length):
    """Thresholds per view index; lambda_tmax is reused from view tmax + 1 on (Stage 5 (k))."""
    v = np.asarray(lam_vec, float)
    return np.concatenate([v, np.full(max(0, length - len(v)), v[-1])])


def halts_ext(pis, lam_vec):
    return np.array([M.halt_index(p, ext(lam_vec, len(p))) for p in pis])


def screen_ext(pis, preds, y, yfull, alpha, tmax, grid=M.GRID):
    """Ringel Algorithm 1 (candidate screening) written out step by step, for units that may have more views than
    tmax: lambda_tmax applies from view tmax on and the halting time is capped at tmax. For units with at most tmax
    views it equals src.r8_stage6_methods.stage1_screen (checked in the tests)."""
    lam = np.full(tmax, INF)
    for t in range(tmax):
        chosen = INF
        for g in grid:
            v = lam.copy()
            v[t] = g
            h = halts_ext(pis, v)
            I = np.flatnonzero(np.minimum(h, tmax - 1) <= t)
            if not len(I):
                chosen = INF                                          # no empirical risk can be formed
                break
            L, _ = M.losses_at(h[I], [preds[i] for i in I], y[I], yfull[I])
            if L.sum() / len(I) <= alpha:
                chosen = float(g)
                break
        lam[t] = chosen
    return lam


def test_ext(pis, preds, y, yfull, cand, alpha, delta, tmax):
    """Ringel Algorithm 2 (testing) with the same extension; equals stage2_test for units with at most tmax views."""
    lam_hat = np.full(tmax, INF)
    for t in range(tmax, 0, -1):
        conf = np.full(tmax, INF)
        conf[t - 1:] = cand[t - 1:]
        h = halts_ext(pis, conf)
        hc = np.minimum(h, tmax - 1)
        for tp in range(t, tmax + 1):
            I = np.flatnonzero(hc <= tp - 1)
            if not len(I):
                return lam_hat
            L, _ = M.losses_at(h[I], [preds[i] for i in I], y[I], yfull[I])
            if M.binom_p(int(L.sum()), len(I), alpha) > delta:
                return lam_hat
        lam_hat = conf
    return lam_hat


def conditional_ext(pis, preds, y, yfull, d1, d2, alpha, delta, tmax):
    sub = lambda idx: ([pis[i] for i in idx], [preds[i] for i in idx], y[idx], yfull[idx])
    cand = screen_ext(*sub(d1), alpha, tmax)
    return test_ext(*sub(d2), cand, alpha, delta, tmax), cand


def b5c_parts(cert_poles, unit_pole, cert_units, seed, salt=SC.B5C_SALT):
    """D_cal-1 = units of the first round-half-up(P / 2) Cert poles of a permutation from default_rng([salt, seed]);
    D_cal-2 = the units of the other Cert poles (by pole, Stage 5 (k))."""
    perm = np.asarray(cert_poles)[np.random.default_rng([salt, seed]).permutation(len(cert_poles))]
    h = (len(perm) + 1) // 2
    p1 = set(perm[:h].tolist())
    d1 = np.array([i for i in cert_units if unit_pole[i] in p1], int)
    d2 = np.array([i for i in cert_units if unit_pole[i] not in p1], int)
    return d1, d2, h


def ringel_eval(pis, preds, y, yfull, lam_vec, sizes):
    """Test evaluation (POC-2 ringel_test) with thresholds extended to every unit's length; adds the wrong count."""
    L = max(len(p) for p in pis)
    r = M.ringel_test(pis, preds, y, yfull, ext(lam_vec, L), sizes)
    h = halts_ext(pis, lam_vec)
    ystop = np.array([preds[i][h[i]] for i in range(len(h))])
    com = ystop >= 0
    r["wrong"] = int((ystop[com] != y[com]).sum())
    r["n_test"] = len(pis)
    r["cp95"] = list(cp_bounds(r["wrong"], r["committed"])) if r["committed"] else None
    return r


# ------------------------------------------------------------------ B6 (h)
def b6_table(seqs, y, theta, rows, size_view):
    """B6: commit at the first view with confidence > theta (POC-2 rule); theta None = commit nothing.
    One-column table so test_metrics applies with j = 0."""
    n = len(seqs)
    com, wrong, pos = np.zeros((n, 1), bool), np.zeros((n, 1), bool), np.zeros((n, 1), int)
    size = np.full((n, 1), np.nan)
    if theta is not None:
        for i, (conf, lab) in enumerate(seqs):
            hit = np.flatnonzero(conf > theta)
            if len(hit):
                com[i, 0], pos[i, 0] = True, hit[0] + 1
                wrong[i, 0] = lab[hit[0]] != y[i]
                size[i, 0] = size_view[rows[i][hit[0]]]
    return {"committed": com, "wrong": wrong, "pos": pos, "size": size}


# ------------------------------------------------------------------ summaries for the tables (i), (j)
def med_range(vals):
    """(median, min, max, count) over the values that exist; None if none."""
    v = [x for x in vals if x is not None]
    if not v:
        return None
    return float(np.median(v)), float(min(v)), float(max(v)), len(v)
