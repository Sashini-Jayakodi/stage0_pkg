"""EXPLORATORY (post-hoc, preliminary). Sign-level certificate of frame-weighted (micro) selective risk on GTSRB.

Every GTSRB track has been seen before, so nothing here is confirmatory. Stored outputs only: no frozen file and no
earlier result is changed; outputs go to runs/20261003_stage1/prelim_r10/.

Unit = track g. At B1 threshold j (POC-2 Stage 3 Part A (c): a frame is accepted if it passes the gate and its single-
frame V1 margin >= threshold; wrong = accepted and top class != label):
    N_g = accepted frames, K_g = accepted-and-wrong frames, Z_g = (K_g - alpha * N_g) / 30   (Z_g in [-alpha, 1 - alpha])
H0: E[Z] >= 0, i.e. pooled micro risk E[K] / E[N] >= alpha. Rejecting H0 at j certifies micro risk < alpha at j.
Tests at level delta: (i) Hoeffding, (ii) empirical Bernstein (Maurer and Pontil 2009, Thm 4), (iii) betting test of
Waudby-Smith and Ramdas 2024 on X = Z + alpha in [0, 1] against mean alpha, predictable plug-in bet. Fixed-sequence
testing in the B1 order of b1_design.json (position 19 down to 0); certify the last rejected threshold.

Part 1: the POC-2 Stage 3b resampling design (same seed, same draws). Part 2: the five POC-2 Stage 4 Cert/Test
re-splits (regenerated, checked against Stage 4). Part 3: a beta-binomial forecast of the B1 violation rate built from
design-track quantities.

Usage:  py -3.12 -m src.r10_prelim_micro --run runs/20261003_stage1
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
from scipy.stats import norm

from src import r8_frozen as FZ
from src import r8_stage1 as R1
from src import r8_stage2 as S2
from src import r8_stage3_engine as E
from src import r8_stage3_resample as RS
from src import r8_stage4 as S4
from src.certify import hb_pvalue
from src.r8_frozen import load_frozen
from src.r8_stage3_b1design import frame_margins
from src.util import sha256_file

LABEL = "EXPLORATORY (post-hoc, preliminary; GTSRB tracks all seen before)"
STAGE3B_SHA256 = "7591a438b569e7231f3a40d8da6420a81b6f65791e707df4c0101338a0bf7367"   # POC-2 Stage 3 Part B
STAGE4_SHA256 = "e66e45f210ebae5e6d21a4d8247584e1affc2d9691a760427814c821b164d2cc"    # POC-2 Stage 4 Part B
STAGE4_CODE_SHA256 = "fae67390a488b44c57f1386c9b256cb0fbbcf2444282f5724c2580aa3e6445c2"
FRAMES = 30                      # Z denominator and simulated track length
ALPHAS = (0.005, 0.01, 0.02, 0.03, 0.05)
ALPHAS_P2 = (0.01, 0.02, 0.03, 0.05)
ALPHAS_P3 = (0.005, 0.01)
WSR_C = 0.5                      # bet truncation c (Waudby-Smith and Ramdas suggest 1/2 or 3/4)
ORDER_SALT = 20261116            # random visiting order of Cert tracks for the sequential test (Part 2)
FORECAST_SEED = 20261117
N_FORECAST = 20000
HEURISTIC_DEFF = 13.9            # POC-1 design effect


# ------------------------------------------------------------------ the three tests (vectorised over draws x thresholds)
def hoeffding_reject(mean, G, delta):
    return mean <= -math.sqrt(math.log(1 / delta) / (2 * G))


def eb_reject(mean, var, G, delta):
    """Maurer-Pontil Thm 4 upper bound on E[Z]; Z has range width 1, so the bound applies after a shift."""
    L = math.log(2 / delta)
    return mean + np.sqrt(2 * np.maximum(var, 0.0) * L / G) + 7 * L / (3 * (G - 1)) < 0


def wsr_reject(X, idx, m, delta, hedge=False, c=WSR_C):
    """One-sided betting test of H0: E[X] >= m, X in [0, 1]. X: (T, J) table per unit; idx: (D, G) visiting order.
    Capital K_t = prod (1 - lam_i (X_i - m)), lam_i >= 0 predictable, so K is a nonnegative supermartingale under H0
    (Ville: P(sup K >= 1/delta) <= delta). Plug-in bet lam_t = sqrt(2 L / (sig2_{t-1} t ln(1+t))) ^ c/(1-m), with
    mu_t = (1/2 + sum X) / (t+1), sig2_t = (1/4 + sum (X_i - mu_i)^2) / (t+1). hedge=False: theta = 1, L = ln(1/delta),
    threshold 1/delta. hedge=True: the downward half of the theta = 1/2 hedged process, L = ln(2/delta), threshold 2/delta.
    Returns (reject (D, J), final log capital (D, J))."""
    D, G = idx.shape
    J = X.shape[1]
    L = math.log(2 / delta) if hedge else math.log(1 / delta)
    log_thr = math.log(2 / delta) if hedge else math.log(1 / delta)
    S = np.zeros((D, J)); SS = np.zeros((D, J)); sig2 = np.full((D, J), 0.25)
    logK = np.zeros((D, J)); rej = np.zeros((D, J), bool)
    cap = c / (1 - m)
    for t in range(1, G + 1):
        x = X[idx[:, t - 1]]
        lam = np.minimum(np.sqrt(2 * L / (sig2 * t * math.log(1 + t))), cap)
        logK += np.log1p(-lam * (x - m))
        rej |= logK >= log_thr
        S += x
        mu = (0.5 + S) / (t + 1)
        SS += (x - mu) ** 2
        sig2 = (0.25 + SS) / (t + 1)
    return rej, logK


def wsr_scalar(xs, m, delta, hedge=False, c=WSR_C):
    """Brute-force single-sequence reference for wsr_reject."""
    L = math.log(2 / delta) if hedge else math.log(1 / delta)
    thr = (2 / delta) if hedge else (1 / delta)
    S = SS = 0.0; sig2 = 0.25; K = 1.0; rej = False
    for t, x in enumerate(xs, 1):
        lam = min(math.sqrt(2 * L / (sig2 * t * math.log(1 + t))), c / (1 - m))
        K *= 1 - lam * (x - m)
        rej = rej or K >= thr
        S += x
        mu = (0.5 + S) / (t + 1)
        SS += (x - mu) ** 2
        sig2 = (0.25 + SS) / (t + 1)
    return rej, math.log(K)


def walk_bool(R, order):
    """Fixed sequence on reject flags: certified column = last rejected before the first non-rejection; -1 if none."""
    order = np.asarray(order, int)
    n = np.cumprod(R[:, order], axis=1).sum(1)
    out = np.full(len(R), -1, int)
    out[n > 0] = order[n[n > 0] - 1]
    return out


def icc_anova(groups):
    """One-way ANOVA intraclass correlation for 0/1 values in groups of unequal size. groups: list of (n_i, k_i)."""
    n = np.array([g[0] for g in groups], float); k = np.array([g[1] for g in groups], float)
    N, G = n.sum(), len(n)
    grand = k.sum() / N
    msb = ((k - n * grand) ** 2 / n).sum() / (G - 1)
    msw = (k - k ** 2 / n).sum() / (N - G)
    n0 = (N - (n ** 2).sum() / N) / (G - 1)
    return float((msb - msw) / (msb + (n0 - 1) * msw))


# ------------------------------------------------------------------ inputs
def load_all(run, cfg):
    run = Path(run)
    d = RS.load_pool(run, cfg)                                         # hash-checks Stage 1 / head / split / b1_design
    P, checks = RS.pool_tables(d, cfg)                                 # pool A, W, NF per track at the 20 B1 thresholds
    s3b_p = run / "poc2" / "stage3" / "stage3b_summary.json"
    s4_p = run / "poc2" / "stage4" / "stage4_summary.json"
    assert sha256_file(s3b_p) == STAGE3B_SHA256, "stage3b_summary.json differs from its recorded hash"
    assert sha256_file(s4_p) == STAGE4_SHA256, "stage4_summary.json differs from its recorded hash"
    s3b, s4 = json.loads(s3b_p.read_text()), json.loads(s4_p.read_text())
    assert s4["code_sha256"]["r8_stage4.py"] == STAGE4_CODE_SHA256 == sha256_file(Path(S4.__file__)), "Stage 4 code changed"

    # design tracks (seed-0 Fit split): frame acceptance at the same thresholds, for Part 3 only
    Sp = np.load(run / "splits" / "seed0.npz")
    split = np.asarray(Sp["track_split"])
    E_ = np.load(run / "embeddings" / "train_release.npz")
    track_all = E_["track"].astype(int)
    rows = np.flatnonzero(split[track_all] == 0)
    ev = np.load(run / "poc2" / "stage1" / RS.EVID_FILE)
    m, top = frame_margins(d["head"], E_["emb"][rows], ev["passed"][rows])
    yd, td = E_["y"][rows].astype(int), track_all[rows]
    codes, inv = np.unique(td, return_inverse=True)
    thr = np.asarray(d["b1"]["work_thresholds"], float)
    acc = m[:, None] >= thr[None, :]
    wr = acc & (top != yd)[:, None]
    des = {"inv": inv, "acc": acc, "wr": wr, "n_tracks": len(codes),
           "A": np.stack([np.bincount(inv, weights=acc[:, j], minlength=len(codes)) for j in range(len(thr))], 1),
           "W": np.stack([np.bincount(inv, weights=wr[:, j], minlength=len(codes)) for j in range(len(thr))], 1),
           "NF": np.bincount(inv, minlength=len(codes)).astype(float)}
    checks["design_tracks_261"] = des["n_tracks"] == 261
    checks["design_frames_match_b1_design"] = int(des["NF"].sum()) == int(d["b1"]["n_frames"])
    # b1_design.json work_e is the accepted error on the design frames at alpha 1%: reproduce it
    checks["design_work_e_reproduces"] = bool(np.allclose(des["W"].sum(0) / des["A"].sum(0), d["b1"]["work_e"], rtol=0, atol=1e-12))
    checks["design_work_c_reproduces"] = bool(np.allclose(des["A"].sum(0) / des["NF"].sum(), d["b1"]["work_c"], rtol=0, atol=1e-12))

    rows_p = R1.track_rows(d["track"], d["frame"])
    y_track = np.array([d["y"][r[0]] for r in rows_p])
    ins = dict(d["inputs_sha256"])
    ins.update({"stage3b_summary.json": STAGE3B_SHA256, "stage4_summary.json": STAGE4_SHA256,
                "stage1_summary.json": sha256_file(run / "poc2" / "stage1" / "stage1_summary.json"),
                "head_summary.json": sha256_file(run / "head_emb" / "head_summary.json"),
                "frozen_r8.yaml": sha256_file(FZ.ROOT / FZ.FROZEN), "r8_stage4.py": STAGE4_CODE_SHA256})
    return d, P, des, y_track, s3b, s4, checks, ins


def draw_indices(n_tracks, n_cert, draws, seed):
    """The Stage 3b draws in drawn order (same generator calls as RS.draw_weights)."""
    rng = np.random.default_rng(seed)
    return np.stack([rng.integers(0, n_tracks, n_cert) for _ in range(draws)])


# ------------------------------------------------------------------ helpers
def pct(x, nd=2):
    return "-" if x is None else f"{100 * x:.{nd}f}%"


def med_rng(v, nd=2, scale=100, suf="%"):
    v = [x for x in v if x is not None]
    if not v:
        return "-"
    return f"{scale * np.median(v):.{nd}f}{suf} ({scale * min(v):.{nd}f}-{scale * max(v):.{nd}f})"


def md(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return out


# ------------------------------------------------------------------ main
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    a = ap.parse_args(argv)
    t0 = time.time()
    cfg = load_frozen()
    delta = float(cfg["fixed_in_every_phase"]["delta"])
    assert delta == 0.1
    run = Path(a.run)
    d, P, des, y_track, s3b, s4, checks, ins = load_all(run, cfg)
    b1 = d["b1"]
    order, thr = b1["order_positions"], np.asarray(b1["work_thresholds"], float)
    assert order == list(range(19, -1, -1)) and b1["start_position"] == 19
    T, J = P["n_tracks"], len(thr)
    A, W, NF = P["A"], P["W"], P["NF"]
    pool_risk = W.sum(0) / A.sum(0)
    pool_cov = A.sum(0) / NF.sum()
    out = [f"# {LABEL}", "", "Sign-level certificate of frame-weighted (micro) selective risk, GTSRB, delta = 0.1.", ""]

    # ---- integrity: Stage 3b draws and stored rows reproduce
    n_cert = int(s3b["config"]["n_cert"])
    idx = draw_indices(T, n_cert, int(s3b["config"]["draws"]), int(s3b["config"]["seed"]))
    Wd = np.stack([np.bincount(r, minlength=T) for r in idx]).astype(float)
    checks["draws_equal_stage3b_weights"] = bool(np.array_equal(Wd, RS.draw_weights(T, n_cert, len(idx), int(s3b["config"]["seed"]))))
    res = RS.resample(P, cfg, b1, Wd, list(ALPHAS), delta)
    same = True
    for al in ALPHAS:
        st = s3b["results"][str(al)]
        for mth in ("track", "B1"):
            same &= res[al][mth]["violations"] == st[mth]["violations"] and \
                abs(res[al][mth]["certified_share"] - st[mth]["certified_share"]) < 1e-12
    checks["stage3b_rows_reproduce"] = bool(same)
    checks["b1_true_risk_equals_micro_risk_side"] = bool(all(
        np.array_equal(np.asarray(res[al]["B1"]["true_mean_L"]) > al + E.TOL, pool_risk > al + E.TOL) for al in ALPHAS))

    # ---- brute-force references for the vectorised tests (3 draws, alpha 1%)
    Z1 = (W - 0.01 * A) / FRAMES
    ok = True
    for r in range(3):
        for j in range(J):
            xs = list(Z1[idx[r], j] + 0.01)
            rr, lk = wsr_scalar(xs, 0.01, delta)
            rv, lkv = wsr_reject(Z1[:, [j]] + 0.01, idx[r:r + 1], 0.01, delta)
            ok &= rr == bool(rv[0, 0]) and abs(lk - lkv[0, 0]) < 1e-9
            z = np.array(xs) - 0.01
            ok &= bool(eb_reject(np.array([z.mean()]), np.array([z.var(ddof=1)]), len(z), delta)[0]) == \
                (z.mean() + math.sqrt(2 * z.var(ddof=1) * math.log(20) / len(z)) + 7 * math.log(20) / (3 * (len(z) - 1)) < 0)
    checks["wsr_eb_match_scalar_reference"] = bool(ok)
    zr = [(Z1[:, j].min(), Z1[:, j].max()) for j in range(J)]
    checks["Z_within_range_width_1"] = bool(all(lo >= -0.01 - 1e-12 and hi <= 0.99 + 1e-12 for lo, hi in zr))

    # ---- Part 1
    G = n_cert
    p1 = {}
    for al in ALPHAS:
        Z = (W - al * A) / FRAMES
        mean = Wd @ Z / G
        var = (Wd @ Z ** 2 - G * mean ** 2) / (G - 1)
        rej = {"(i) Hoeffding": hoeffding_reject(mean, G, delta), "(ii) Emp. Bernstein": eb_reject(mean, var, G, delta),
               "(iii) WSR betting": wsr_reject(Z + al, idx, al, delta)[0],
               "(iii-h) WSR hedged half": wsr_reject(Z + al, idx, al, delta, hedge=True)[0]}
        mu_b, n_b = E.frame_mu(A, W, NF, Wd, al)
        cb = E.certify(mu_b, n_b, al, order, delta)                    # B1 recomputed, for its frame coverage
        p1[al] = {}
        for name, R in list(rej.items()) + [("B1 frame-level HB (recomputed)", None)]:
            cert = cb if R is None else walk_bool(R, order)
            row = E.rate_row(cert, pool_risk, al)
            okc = cert >= 0
            row["mean_frame_coverage_certified"] = float(pool_cov[cert[okc]].mean()) if okc.any() else None
            row["certified_position_counts"] = np.bincount(cert[okc], minlength=J).tolist()
            p1[al][name] = row
    out += ["## Part 1: validity and power (Stage 3b design: 2,000 draws of 784 of the 1,046 pool tracks, seed "
            f"{s3b['config']['seed']})", "",
            "Violation = a threshold is certified and its pool micro risk > alpha. Coverage = pool frame coverage at the "
            "certified threshold, mean over certifying draws. Stored rows are copied from stage3b_summary.json.", ""]
    rows = []
    for al in ALPHAS:
        for name, r in p1[al].items():
            rows.append([pct(al, 1), name, f"{r['violations']}", pct(r["violation_rate"]),
                         f"[{pct(r['cp95'][0])}, {pct(r['cp95'][1])}]", pct(r["certified_share"], 1),
                         pct(r["mean_frame_coverage_certified"], 1), pct(r["mean_true_risk_certified"], 3)])
        for mth, lab in (("B1", "stored B1 frame-level (Stage 3b)"), ("track", "stored track-level (Stage 3b)")):
            st = s3b["results"][str(al)][mth]
            rows.append([pct(al, 1), lab, f"{st['violations']}", pct(st["violation_rate"]),
                         f"[{pct(st['cp95'][0])}, {pct(st['cp95'][1])}]", pct(st["certified_share"], 1), "-",
                         pct(st["mean_true_risk_certified"], 3)])
    out += md(["alpha", "test", "viol", "violation rate", "95% CP", "certify share", "mean frame cov.", "mean true risk (cert.)"], rows)
    out += ["", "(iii) = theta 1 (one-sided). (iii-h) = sensitivity: the downward half of the theta 1/2 hedged process "
            "(L = ln(2/delta), threshold 2/delta). Track-level stored row: its unit is the track commit rule, so no frame "
            "coverage is given. Last column: pool micro risk (W/A) for the rows computed here; for the stored rows it is "
            "the stored mean loss L, a different scale (violations and certify shares reproduce exactly).", ""]

    # ---- Part 2
    tests, rep = {}, True
    n_test = int(s4["per_seed"][0]["n_test"])
    for row in s4["per_seed"]:
        sd = int(row["seed"])
        te = S4.make_split(y_track, sd, n_test)
        rep &= int(te.sum()) == int(row["n_test"]) and int((~te).sum()) == int(row["n_cert"])
        for by_m in row["alphas"].values():
            r4 = by_m["commit"]
            if r4["certified"]:
                rep &= float(P["committed"][te, int(r4["grid_position"])].sum() / te.sum()) == float(r4["coverage"])
        tests[sd] = te
    checks["stage4_splits_reproduce"] = bool(rep)
    p2 = {}
    for al in ALPHAS_P2:
        Z = (W - al * A) / FRAMES
        p2[al] = {}
        for sd, te in tests.items():
            ci = np.flatnonzero(~te)
            ci = ci[np.random.default_rng([ORDER_SALT, sd]).permutation(len(ci))]
            Gc = len(ci)
            zc = Z[ci]
            mean, var = zc.mean(0), zc.var(0, ddof=1)
            rr = {"(ii) Emp. Bernstein": eb_reject(mean[None], var[None], Gc, delta),
                  "(iii) WSR betting": wsr_reject(Z + al, ci[None], al, delta)[0]}
            for name, R in rr.items():
                j = int(walk_bool(R, order)[0])
                rec = {"position": None, "threshold": None, "test_micro_risk": None, "test_frame_coverage": 0.0,
                       "cert_micro_risk": None}
                if j >= 0:
                    At, Wt = A[te, j].sum(), W[te, j].sum()
                    rec = {"position": j, "threshold": float(thr[j]), "test_micro_risk": float(Wt / At) if At else None,
                           "test_frame_coverage": float(At / NF[te].sum()),
                           "cert_micro_risk": float(W[ci, j].sum() / A[ci, j].sum())}
                p2[al].setdefault(name, {})[sd] = rec
    out += ["## Part 2: the five POC-2 Stage 4 Cert/Test re-splits (regenerated, checked against Stage 4)", "",
            f"Cert 784 tracks, Test {n_test}. WSR visits Cert tracks in a seeded random order (salt {ORDER_SALT}, seed). "
            "Not certified: coverage 0, risk '-'.", ""]
    rows = []
    for al in ALPHAS_P2:
        for name in p2[al]:
            for sd in tests:
                r = p2[al][name][sd]
                rows.append([pct(al, 0), name, sd, "-" if r["position"] is None else r["position"],
                             "-" if r["threshold"] is None else f"{r['threshold']:.1f}", pct(r["cert_micro_risk"], 3),
                             pct(r["test_micro_risk"], 3), pct(r["test_frame_coverage"], 1)])
    out += md(["alpha", "test", "seed", "certified pos.", "threshold", "Cert micro risk", "Test micro risk", "Test frame cov."], rows)
    out += ["", "Summary over the five re-splits, median (min-max):", ""]
    rows = []
    for al in ALPHAS_P2:
        for name in p2[al]:
            v = list(p2[al][name].values())
            ncert = sum(r["position"] is not None for r in v)
            rows.append([pct(al, 0), name, f"{ncert}/5",
                         med_rng([r["position"] for r in v], 0, 1, ""), med_rng([r["test_micro_risk"] for r in v], 3),
                         med_rng([r["test_frame_coverage"] for r in v], 1)])
    out += md(["alpha", "test", "certified", "position", "Test micro risk", "Test frame coverage"], rows)
    out += [""]

    # ---- Part 3
    rng = np.random.default_rng(FORECAST_SEED)
    heur = float(norm.cdf(norm.ppf(delta) / math.sqrt(HEURISTIC_DEFF)))
    p3 = {}
    for al in ALPHAS_P3:
        jstar = next((j for j in order if pool_risk[j] > al + E.TOL), None)
        rec = {"j_star": jstar}
        if jstar is not None:
            f_acc = float(des["A"][:, jstar].sum() / des["NF"].sum())
            p_aw = float(des["W"][:, jstar].sum() / des["NF"].sum())
            rho = icc_anova([(int(n), int(k)) for n, k in zip(des["NF"], des["W"][:, jstar])])
            Ng = int(round(FRAMES * f_acc))
            hits, clipped, above = 0, 0, 0
            for ch in range(N_FORECAST // 2000):
                if p_aw == 0:
                    K = np.zeros((2000, G))
                elif rho > 0:
                    s = 1 / rho - 1
                    K = rng.binomial(FRAMES, rng.beta(p_aw * s, (1 - p_aw) * s, size=(2000, G))).astype(float)
                else:
                    K = rng.binomial(FRAMES, p_aw, size=(2000, G)).astype(float)
                clipped += int((K > Ng).sum())
                K = np.minimum(K, Ng)
                nfr = FRAMES * G
                mu = (K.sum(1) + al * (nfr - G * Ng)) / nfr
                hits += int((hb_pvalue(mu, nfr, al) <= delta).sum())
                above += int((K.sum(1) / (G * Ng) > al).sum())
            rec.update({"threshold": float(thr[jstar]), "pool_micro_risk": float(pool_risk[jstar]),
                        "pool_frame_coverage": float(pool_cov[jstar]), "design_acceptance": f_acc,
                        "design_accepted_wrong_per_frame": p_aw,
                        "design_micro_risk": float(des["W"][:, jstar].sum() / des["A"][:, jstar].sum()),
                        "design_rho_I": rho, "N_g": Ng, "implied_sim_micro_risk": p_aw * FRAMES / Ng,
                        "forecast_P_p_le_delta": hits / N_FORECAST, "clipped_track_draws": clipped,
                        "share_sims_micro_risk_above_alpha": above / N_FORECAST})
        rec["observed_stage3b_B1_violation"] = float(s3b["results"][str(al)]["B1"]["violation_rate"])
        rec["heuristic"] = heur
        p3[al] = rec
    out += ["## Part 3: forecast of the B1 violation rate (preliminary)", "",
            f"j* = first threshold in the frozen order with pool micro risk > alpha. Design quantities from the 261 "
            f"design tracks. {N_FORECAST:,} simulated Cert sets of 784 tracks x 30 frames; accepted frames fixed at "
            f"round(30 x design acceptance); accepted-wrong counts beta-binomial(30, design rate, rho_I), clipped at "
            f"N_g; forecast = P(Stage 3b HB frame-level p-value at j* <= delta). Heuristic = "
            f"Phi(Phi^-1(0.1)/sqrt({HEURISTIC_DEFF})). Seed {FORECAST_SEED}.", ""]
    rows = []
    for al in ALPHAS_P3:
        r = p3[al]
        if r["j_star"] is None:
            rows.append([pct(al, 1), "none"] + ["-"] * 10 + [pct(r["observed_stage3b_B1_violation"]), pct(heur)])
            continue
        rows.append([pct(al, 1), r["j_star"], f"{r['threshold']:.1f}", pct(r["pool_micro_risk"], 3),
                     pct(r["design_acceptance"], 1), pct(r["design_accepted_wrong_per_frame"], 3),
                     pct(r["design_micro_risk"], 3), f"{r['design_rho_I']:.3f}", r["N_g"],
                     pct(r["implied_sim_micro_risk"], 3), r["clipped_track_draws"], pct(r["forecast_P_p_le_delta"]),
                     pct(r["observed_stage3b_B1_violation"]), pct(heur)])
    out += md(["alpha", "j*", "threshold", "pool micro risk at j*", "design accept.", "design acc.-wrong / frame",
               "design micro risk", "rho_I", "N_g", "sim micro risk", "clipped", "forecast", "observed 3b B1", "heuristic"], rows)
    out += [""]

    # ---- sanity prints
    out += ["## Sanity: per-threshold pool micro risk (1,046 pool tracks) and design values", ""]
    rows = [[j, f"{thr[j]:.2f}", int(A[:, j].sum()), int(W[:, j].sum()), pct(pool_risk[j], 3), pct(pool_cov[j], 1),
             pct(des["W"][:, j].sum() / des["A"][:, j].sum(), 3), pct(des["A"][:, j].sum() / des["NF"].sum(), 1)]
            for j in order]
    out += md(["pos.", "threshold", "pool accepted", "pool acc.-wrong", "pool micro risk", "pool frame cov.",
               "design micro risk", "design frame cov."], rows)
    js = p3[0.01]["j_star"]
    out += ["", f"## Sanity: first 10 pool tracks, Z_g at j* = {js} for alpha = 1%", ""]
    names = np.asarray(d["table"]["track_names"]).astype(str)
    if js is not None:
        rows = [[i, names[i], int(NF[i]), int(A[i, js]), int(W[i, js]), f"{(W[i, js] - 0.01 * A[i, js]) / FRAMES:+.6f}"]
                for i in range(10)]
        out += md(["pool idx", "track", "frames", "N_g", "K_g", "Z_g"], rows)
    out += ["", "## Checks", ""] + [f"- {'PASS' if v else 'FAIL'} {k}" for k, v in checks.items()]
    all_pass = all(checks.values())
    out += ["", f"ALL CHECKS PASS: {all_pass}", "", "Input SHA-256:", ""] + [f"- {k}: {v}" for k, v in ins.items()]

    text = "\n".join(out)
    print(text)
    od = run / "prelim_r10"
    od.mkdir(exist_ok=True)
    summary = {"label": LABEL, "code_sha256": {"r10_prelim_micro.py": sha256_file(Path(__file__))}, "inputs_sha256": ins,
               "config": {"delta": delta, "frames": FRAMES, "alphas_part1": ALPHAS, "alphas_part2": ALPHAS_P2,
                          "alphas_part3": ALPHAS_P3, "wsr_c": WSR_C, "order_salt": ORDER_SALT,
                          "forecast_seed": FORECAST_SEED, "n_forecast": N_FORECAST, "stage3b_seed": s3b["config"]["seed"]},
               "pool_micro_risk": pool_risk, "pool_frame_coverage": pool_cov,
               "part1": {str(k): v for k, v in p1.items()},
               "part2": {str(k): {n: {str(s): r for s, r in v.items()} for n, v in by.items()} for k, by in p2.items()},
               "part3": {str(k): v for k, v in p3.items()}, "checks": checks, "all_pass": all_pass,
               "seconds": round(time.time() - t0, 1)}
    (od / "prelim_r10_summary.json").write_text(json.dumps(S2.jsonable(summary), indent=2))
    (od / "prelim_r10_report.md").write_text(text + "\n")
    print(f"\nOutput: {od / 'prelim_r10_summary.json'} and prelim_r10_report.md ({summary['seconds']} s). {LABEL}")
    return summary


if __name__ == "__main__":
    main()
