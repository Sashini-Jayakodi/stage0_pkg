"""r13 (exploratory): the r11 forecast with a bootstrap upper bound on rho_I. Definitions: LOG_r13_entry.md Part A
(commit 8c44fde). One change from r11 + Amendment 1: rho_I input = 80th percentile of 1,000 cluster-bootstrap rho_I
values at j_hat (values <= 0 and zero-error resamples count as 0). Everything else as r11.

Sets: (a) simulated held-out, judged; (b) BelgiumTSC, judged (out-of-sample for the forecast, not blind to its
targets); (c) GTSRB development data, reported; (d) GTSRB pilot-size curve, descriptive.
Criteria: exact binomial only; a pass is "conservative within 5 points", not "accurate".

Usage:  py -3.12 -m src.r13_forecast --run runs/20261003_stage1
Output: runs/20261003_stage1/r13/r13_summary.json and r13_report.md
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from src import r11_forecast_ucb as F
from src import r12_gtsrb_forecast as G
from src import r8_stage2 as S2
from src import r8_stage3_engine as E
from src import r8_stage3_resample as RS
from src import r8_stage3_synth as R8S
from src import r9_certify as C
from src import r9_stage10 as S10
from src import r9_stage10_popcurves as Q
from src import r9_stage2_popcurves as P
from src import r9_stage2_vchecks as V
from src.r8_frozen import load_frozen
from src.r9_frozen import check_frozen
from src.util import sha256_file

LABEL = "EXPLORATORY; LOG_r13_entry.md Part A, commit 8c44fde"
BOOT_N, BOOT_Q = 1000, 80.0                     # Part A (b)
SEED = 20261120                                 # Part A (c): unused by any earlier script
# set (a)
SIM_RHOS, SIM_MS, SIM_ALPHAS = (0.25, 0.40), (6, 20), (0.005, 0.01, 0.02)
SIM_PAIRS = [(rho, m) for rho in SIM_RHOS for m in SIM_MS]       # pair 0..3 as Part A (c)
SIM_HEUR_PCT = {(0.25, 6): 19.65, (0.25, 20): 29.65, (0.40, 6): 22.97, (0.40, 20): 33.11}
SIM_RUNS, SIM_ND, SIM_NC = 2000, 261, 784
# set (b)
BTSC_ALPHAS, BTSC_JUDGED = (0.005, 0.01, 0.02, 0.03), (0.005, 0.01)
BTSC_POLES_PER_DRAW, BTSC_DE = 397, 2.24
BTSC_BOOT_KEY = (SEED, 17)                      # not named in Part A; see the r13 Amendment 1 draft
STAGE6_SUMMARY = Path("runs/btsc/stage6/stage6_summary.json")
STAGE6_SHA256 = "71be795779554a62546b1e03089222b53c35bc5a8ffac4bb2406c470b39e185f"   # LOG.md, Stage 6
# set (c)/(d)
GT_ALPHAS = (0.005, 0.01, 0.02)
CURVE_SIZES, CURVE_PILOTS, CURVE_ALPHAS = (100, 261, 400, 600), 500, (0.005, 0.01)
R12_SUMMARY_SHA256 = "3d559b2fa5ec9de5d25b2d4e01079a083003ff45bee0c2f0b4c5f4a06d18488a"
TESTS = ("binom", "hb")
CRIT_UNDER_PP, CRIT_OVER_PP, CRIT_SHARE, CRIT_MIN_SETTINGS = 2.0, 5.0, 0.80, 10
HEUR_TOL_PCT = 0.005
EPS_PP = 1e-9                                   # inclusive bounds: ignore float noise (100 * 0.14 = 14.000000000000002)
OUT = Path("runs/20261003_stage1/r13")


# ------------------------------------------------------------------ the forecast (tested on synthetic data)
def icc_batch(n, k):
    """Row-wise one-way ANOVA ICC (the r11 icc_anova formula) for B resamples. n, k: (B, G) cluster sizes and counts.
    Rows with no counts get 0 (Part A (b)); any other undefined row stops the run."""
    n, k = n.astype(float), k.astype(float)
    Gc = n.shape[1]
    N = n.sum(1)
    tot = k.sum(1)
    grand = tot / N
    msb = ((k - n * grand[:, None]) ** 2 / n).sum(1) / (Gc - 1)
    msw = (k - k ** 2 / n).sum(1) / (N - Gc)
    n0 = (N - (n ** 2).sum(1) / N) / (Gc - 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        rho = (msb - msw) / (msb + (n0 - 1) * msw)
    rho = np.where(tot == 0, 0.0, rho)
    if not np.isfinite(rho).all():
        raise SystemExit("STOP: a bootstrap resample with errors has an undefined rho_I (Part A (b) does not define it)")
    return rho


def boot_upper(sizes, k, idx, q=BOOT_Q):
    """80th percentile of the resampled rho_I, with values <= 0 counted as 0. idx: (B, G) cluster indices."""
    vals = np.maximum(icc_batch(np.asarray(sizes)[idx], np.asarray(k)[idx]), 0.0)
    return float(np.percentile(vals, q)), vals


def forecast(acc_g, wr_g, sizes, m, alpha, test, n_cert, rule, boot_idx=None):
    """r11 (b) 1-6 for one pilot with real cluster sizes; rule 'r11' (point rho_I) or 'r13' (bootstrap upper bound).
    With equal sizes m this is r11's run_forecast (tested)."""
    sizes = np.asarray(sizes, int)
    A, W = acc_g.sum(0), wr_g.sum(0)
    Ntot = int(sizes.sum())
    start = C.start_for(test, A / Ntot, n_cert * m, alpha, Ntot)
    U = F.cp_upper(W, A)
    js = F.j_star(U, start, alpha)
    rec = {"start": int(start), "j_star": js, "forecast": 0.0, "clipped": 0.0, "rho_point": None, "rho_in": None,
           "case": None, "N_g": None}
    if js < 0:
        return rec
    f = float(A[js] / Ntot)
    p = float(U[js] * f)
    n_g = int(round(m * f))
    if W[js] > 0:
        rho_pt = F.icc_anova([(int(s), int(c)) for s, c in zip(sizes, wr_g[:, js])])
        if not np.isfinite(rho_pt):
            raise SystemExit(f"STOP: rho_I = {rho_pt} at j_hat = {js}")
        rho_in = rho_pt if rule == "r11" else boot_upper(sizes, wr_g[:, js], boot_idx)[0]
        case = "rho_ge_1" if rho_in >= 1 else ("rho_le_0" if rho_in <= 0 else "betabinom")
    else:
        rho_pt, rho_in, case = 0.0, 0.0, "zero_errors"
    rec.update({"rho_point": rho_pt, "rho_in": rho_in, "case": case, "N_g": n_g, "f": f, "p": p, "U": float(U[js])})
    if n_g == 0:
        return rec
    pmf, clip = F.clipped_pmf(m, p, rho_in if rho_in > 0 else 0.0, n_g)
    km = F.k_max(n_cert * n_g, n_cert * m, alpha, test)
    rec.update({"clipped": clip, "k_max": km, "forecast": F.total_cdf(pmf, n_cert, km)})
    return rec


def m_star(sizes):
    """sum(T^2) / sum(T) (Stage 10 convention for clusters of varying size)."""
    s = np.asarray(sizes, float)
    return float((s ** 2).sum() / s.sum())


def crit_sim(forecasts, observed):
    """Part A (d)(a) for one setting: (i) share of pilots with F >= O - 2 pp is >= 80%; (ii) mean F - O <= +5 pp."""
    f = 100 * np.asarray(forecasts, float)
    o = 100 * observed
    share = float(np.mean(f >= o - CRIT_UNDER_PP - EPS_PP))
    gap = float(f.mean() - o)
    ok_i, ok_ii = share >= CRIT_SHARE, gap <= CRIT_OVER_PP + EPS_PP
    return {"share_not_below": share, "mean_gap_pp": gap, "i": bool(ok_i), "ii": bool(ok_ii), "both": bool(ok_i and ok_ii)}


def crit_btsc(forecast_, observed):
    """Part A (d)(b) for one alpha: O - 2 pp <= F <= O + 5 pp (inclusive)."""
    f, o = 100 * forecast_, 100 * observed
    return {"gap_pp": f - o, "pass": bool(o - CRIT_UNDER_PP - EPS_PP <= f <= o + CRIT_OVER_PP + EPS_PP)}


def check_heuristics():
    for (rho, m), want in SIM_HEUR_PCT.items():
        got = 100 * Q.delta_eff(rho, m)
        if abs(got - want) > HEUR_TOL_PCT:
            raise SystemExit(f"STOP: heuristic({rho}, {m}) = {got:.4f}% differs from Part A {want}%")


def pct(x, nd=2):
    return "    n/a" if x is None else f"{100 * x:6.{nd}f}%"


def cp(c):
    return f"[{100 * c[0]:5.2f}, {100 * c[1]:5.2f}]%"


# ------------------------------------------------------------------ set (a): simulated held-out
def rng_sim(pair, stream, run=None):
    return np.random.default_rng([SEED, 13, pair, stream] + ([] if run is None else [run]))


def sim_pair(pair, rho, m, runs, n_pop):
    cv = F.b1_curves(rho, m, lambda s: rng_sim(pair, s), n_pop)
    bgrid, acc, risk = cv["b1_grid"], cv["b1_pop_accept"], cv["b1_true_risk"]
    mu_true = {a: S10.mean_loss(a, acc, risk) for a in SIM_ALPHAS}
    rng = rng_sim(pair, 2)
    Nd, Nc = SIM_ND * m, SIM_NC * m
    sizes = np.full(SIM_ND, m)
    cert = {(a, t): np.full(runs, -1) for a in SIM_ALPHAS for t in TESTS}
    fc = {(a, t, r): [] for a in SIM_ALPHAS for t in TESTS for r in ("r11", "r13")}
    for i in range(runs):
        Xd = Q.draw_fixed(SIM_ND, m, cv["r_latent"], cv["mu"], rng)
        Xc = Q.draw_fixed(SIM_NC, m, cv["r_latent"], cv["mu"], rng)
        acc_g, wr_g = R8S.frame_counts(Xd, bgrid)
        Ad = acc_g.sum(0)
        Ac, Wc, _ = V.frame_totals(Xc, np.full(SIM_NC, m), bgrid)
        bidx = rng_sim(pair, 3, i).integers(0, SIM_ND, (BOOT_N, SIM_ND))
        for a in SIM_ALPHAS:
            for t in TESTS:
                cert[(a, t)][i] = C.certify(Wc, Ac, Nc, a, Ad / Nd, t, n_design=Nd)
                for r in ("r11", "r13"):
                    fc[(a, t, r)].append(forecast(acc_g, wr_g, sizes, m, a, t, SIM_NC, r, bidx))
    settings = []
    for a in SIM_ALPHAS:
        rows = {}
        for t in TESTS:
            obs = E.rate_row(cert[(a, t)], mu_true[a], a)
            row = {"observed": obs["violation_rate"], "cp95": obs["cp95"], "certified_share": obs["certified_share"]}
            for r in ("r11", "r13"):
                f = np.array([x["forecast"] for x in fc[(a, t, r)]])
                cases = [x["case"] for x in fc[(a, t, r)]]
                row[r] = {"mean": float(f.mean()), "median": float(np.median(f)), "p10": float(np.quantile(f, 0.1)),
                          "p90": float(np.quantile(f, 0.9)), "crit": crit_sim(f, obs["violation_rate"]),
                          "cases": {c: cases.count(c) for c in ("betabinom", "rho_le_0", "zero_errors", "rho_ge_1", None)},
                          "rho_in_median": float(np.median([x["rho_in"] for x in fc[(a, t, r)] if x["rho_in"] is not None]))
                          if any(x["rho_in"] is not None for x in fc[(a, t, r)]) else None,
                          "rho_point_median": float(np.median([x["rho_point"] for x in fc[(a, t, r)] if x["rho_point"] is not None]))
                          if any(x["rho_point"] is not None for x in fc[(a, t, r)]) else None}
            rows[t] = row
        settings.append({"pair": pair, "rho": rho, "m": m, "alpha": a, "heuristic": Q.delta_eff(rho, m),
                         "boundary": Q.boundary(acc, risk, Nc, a), "rows": rows})
    return settings, {"rho": rho, "m": m, "r_latent": cv["r_latent"], "calibration": cv["calibration"]}


# ------------------------------------------------------------------ set (b): BelgiumTSC
def load_btsc():
    """Stage 6 loaders (hash- and blob-checked), the design-pole B1 table (pilot) and the pool B1 table (audit)."""
    from src import r9_frozen as FZ
    from src import r9_stage5_design as DS
    from src import r9_stage5_freeze as FR
    from src import r9_stage5_scores as SC
    from src import r9_stage6 as S6
    from src import r9_stage6_methods as SM
    from src.r9_btsc_guard import GuardedTestLabels
    Y, yaml_text = S6.load_yaml()
    design, _ = FR.load_design(S6.STAGE5_DESIGN, S6.DESIGN_SHA256, S6.DESIGN_BLOB)
    d = DS.load_stage4()
    summary = json.loads((Path(DS.STAGE4) / "stage4_summary.json").read_text())
    Pp = S6.open_pool(DS.STAGE4, summary)
    R = S6.reproduce_yaml(d, design, yaml_text, sha256_file(S6.STAGE5_DESIGN), FZ.blob_id(S6.STAGE5_DESIGN))
    T = S6.pool_tables(d, Pp, R, Y)
    grid = np.array(Y["b1"]["grid"], float)
    # pilot: the design views through the guard, B1 single-view table on the frozen grid, grouped by design pole
    Ed, rows = d["emb"], d["rows"]
    guard = GuardedTestLabels([r["pole"] for r in rows], [r["cls"] for r in rows], sorted(d["poles"]["design_poles"]))
    y_view = guard.labels(Ed["row"])
    Vd = SC.view_inputs(d["head"], Ed["emb"])
    tab_d = SC.track_table(SM.b1_seqs(Vd), y_view, grid)
    dpoles = np.unique(Ed["pole"])
    Cp_d, Wp_d, T_d = SM.per_pole(tab_d, np.searchsorted(dpoles, Ed["pole"]), len(dpoles))
    checks = {"design_views_465": int(T_d.sum()) == int(Y["b1"]["n_design_views"]) == 465,
              "design_poles_133": len(dpoles) == 133,
              "design_f_d_reproduces": bool(np.allclose(Cp_d.sum(0) / T_d.sum(), Y["b1"]["f_d"], rtol=0, atol=1e-12))}
    # audit: Stage 6 RQ1 draws on the pool B1 table
    ppoles = np.unique(T["view_pole"])
    Cp, Wp, n_items = SM.per_pole(T["b1"], np.searchsorted(ppoles, T["view_pole"]), len(ppoles))
    rq = Y["rq1"]
    counts = SM.draw_counts(len(ppoles), int(rq["draws"]), int(rq["poles_per_draw"]), tuple(rq["salt"]))
    K, NC, N = SM.draw_sums(counts, Cp, Wp, n_items)
    risk, f_pool = SM.true_risk(T["b1"])
    return {"Y": Y, "grid": grid, "Cp_d": Cp_d, "Wp_d": Wp_d, "T_d": T_d, "K": K, "NC": NC, "N": N, "risk": risk,
            "f_pool": f_pool, "SM": SM, "checks": checks, "n_pool_poles": len(ppoles),
            "inputs_sha256": d["inputs_sha256"]}


def check_btsc(B, s6):
    """Stop-check before any forecast: the Stage 6 RQ1 B1 cells (violations and certified count) reproduce exactly."""
    SM = B["SM"]
    f_d = np.array(B["Y"]["b1"]["f_d"], float)
    nd = int(B["Y"]["b1"]["n_design_views"])
    cells, repro = {}, True
    for a in BTSC_ALPHAS:
        for t in TESTS:
            c = SM.rq1_cell(B["K"], B["NC"], B["N"], a, f_d, t, nd, B["risk"])
            want = s6["rq1"]["B1"]["cells"][str(a)][t]
            repro &= c["violations"] == want["violations"] and c["certified"] == want["certified"]
            cells[(a, t)] = c
    B["checks"]["stage6_rq1_b1_reproduces"] = bool(repro)
    if not all(B["checks"].values()):
        raise SystemExit(f"STOP: a BelgiumTSC check failed: {B['checks']}")
    return cells


def run_btsc(B, cells):
    f_d = np.array(B["Y"]["b1"]["f_d"], float)
    nd = int(B["Y"]["b1"]["n_design_views"])
    ms = m_star(B["T_d"])
    m = int(round(ms))
    bidx = np.random.default_rng(list(BTSC_BOOT_KEY)).integers(0, len(B["T_d"]), (BOOT_N, len(B["T_d"])))
    out = {"m_star": ms, "m": m, "model_views_per_draw": BTSC_POLES_PER_DRAW * m,
           "real_views_per_draw": {"min": int(B["N"].min()), "mean": float(B["N"].mean()), "max": int(B["N"].max())},
           "mismatch_views": {"min": int(B["N"].min() - BTSC_POLES_PER_DRAW * m),
                              "mean": float(B["N"].mean() - BTSC_POLES_PER_DRAW * m),
                              "max": int(B["N"].max() - BTSC_POLES_PER_DRAW * m)},
           "heuristic": S10.de_to_delta_eff(BTSC_DE), "rows": {}}
    for a in BTSC_ALPHAS:
        for t in TESTS:
            c = cells[(a, t)]
            r = {"observed": c["rate"], "cp95": c["cp95"], "certified": c["certified"], "violations": c["violations"]}
            for rule in ("r11", "r13"):
                r[rule] = forecast(B["Cp_d"], B["Wp_d"], B["T_d"], m, a, t, BTSC_POLES_PER_DRAW, rule, bidx)
            st = np.array([C.start_for(t, f_d, int(n), a, nd) for n in B["N"]])
            vals, cnt = np.unique(st, return_counts=True)
            r["stage6_starts"] = {int(v): int(k) for v, k in zip(vals, cnt)}
            r["j_true"] = F.j_true(S10.mean_loss(a, B["f_pool"], B["risk"]), r["r13"]["start"], a)
            if a in BTSC_JUDGED and t == "binom":
                r["crit"] = crit_btsc(r["r13"]["forecast"], c["rate"])
            out["rows"][f"{a}_{t}"] = r
    return out


# ------------------------------------------------------------------ sets (c) and (d): GTSRB
def load_gtsrb(run, cfg, r12):
    """Load and stop-check before any r13 forecast: Stage 3b HB and r12 exact-binomial observed rates and the r12
    (r11-rule) pilot forecasts reproduce exactly (all are recorded r12 numbers)."""
    d = RS.load_pool(run, cfg)
    Pp, checks = RS.pool_tables(d, cfg)
    Ad, Wdes, NFd = G.load_design(run, d)
    s3b = G.load_stage3b(Path(run) / "poc2" / "stage3" / "stage3b_summary.json")
    T = Pp["n_tracks"]
    Wd = RS.draw_weights(T, 784, 2000, int(s3b["config"]["seed"]))
    res = RS.resample(Pp, cfg, d["b1"], Wd, list(GT_ALPHAS), C.DELTA)
    pop = np.ones((1, T))
    obs, repro = {}, all(checks.values())
    for a in GT_ALPHAS:
        mu_true = E.frame_mu(Pp["A"], Pp["W"], Pp["NF"], pop, a)[0][0]
        obs[(a, "hb")] = res[a]["B1"]["violation_rate"]
        obs[(a, "binom")] = E.rate_row(G.observed_binom(Pp["A"], Pp["W"], Wd, a), mu_true, a)["violation_rate"]
        for t in TESTS:
            repro &= obs[(a, t)] == r12["rows"][f"{a}_{t}"]["observed"]["violation_rate"]
            repro &= forecast(Ad, Wdes, NFd, G.M, a, t, G.N_CERT, "r11")["forecast"] == r12["rows"][f"{a}_{t}"]["forecast"]
    if not repro:
        raise SystemExit("STOP: set (c) does not reproduce r12 / Stage 3b")
    return {"Pp": Pp, "Ad": Ad, "Wdes": Wdes, "NFd": NFd, "obs": obs, "T": T}


def run_gtsrb(L):
    Pp, Ad, Wdes, NFd, obs, T = L["Pp"], L["Ad"], L["Wdes"], L["NFd"], L["obs"], L["T"]
    outer = np.random.default_rng(list(G.BOOT_KEY)).integers(0, G.N_DESIGN, (G.BOOT_N, G.N_DESIGN))
    rows = {}
    for a in GT_ALPHAS:
        for t in TESTS:
            inner0 = np.random.default_rng([SEED, 15, 0]).integers(0, G.N_DESIGN, (BOOT_N, G.N_DESIGN))
            r11 = forecast(Ad, Wdes, NFd, G.M, a, t, G.N_CERT, "r11")
            r13 = forecast(Ad, Wdes, NFd, G.M, a, t, G.N_CERT, "r13", inner0)
            fb = []
            for b, idx in enumerate(outer):
                inner = np.random.default_rng([SEED, 15, b + 1]).integers(0, G.N_DESIGN, (BOOT_N, G.N_DESIGN))
                fb.append(forecast(Ad[idx], Wdes[idx], NFd[idx], G.M, a, t, G.N_CERT, "r13", inner)["forecast"])
            fb = np.array(fb)
            rows[(a, t)] = {"observed": obs[(a, t)], "r11": r11, "r13": r13, "boot_p10": float(np.quantile(fb, 0.1)),
                            "boot_p90": float(np.quantile(fb, 0.9)), "boot_median": float(np.median(fb))}
    # (d) pilot-size curve from the pool tracks
    curve = {}
    for size in CURVE_SIZES:
        per = {(a, t, r): [] for a in CURVE_ALPHAS for t in TESTS for r in ("r11", "r13")}
        for pi in range(CURVE_PILOTS):
            sel = np.random.default_rng([SEED, 14, size, pi]).choice(T, size, replace=False)
            inner = np.random.default_rng([SEED, 16, size, pi]).integers(0, size, (BOOT_N, size))
            Ag, Wg, Ng = Pp["A"][sel].astype(int), Pp["W"][sel].astype(int), Pp["NF"][sel].astype(int)
            for a in CURVE_ALPHAS:
                for t in TESTS:
                    for r in ("r11", "r13"):
                        per[(a, t, r)].append(forecast(Ag, Wg, Ng, G.M, a, t, G.N_CERT, r, inner)["forecast"])
        for (a, t, r), v in per.items():
            v = np.array(v)
            o = obs[(a, t)]
            curve[(size, a, t, r)] = {"mean_bias": float(v.mean() - o), "p10": float(np.quantile(v, 0.1)),
                                      "p90": float(np.quantile(v, 0.9)), "mean": float(v.mean())}
    return rows, curve, obs


# ------------------------------------------------------------------ main
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/20261003_stage1")
    ap.add_argument("--runs", type=int, default=SIM_RUNS)
    ap.add_argument("--n-pop", type=int, default=P.N_POP)
    ap.add_argument("--stage6", default=str(STAGE6_SUMMARY))
    ap.add_argument("--r12", default=None)
    ap.add_argument("--sets", default="abcd")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--check-only", action="store_true")
    a = ap.parse_args(argv)
    t0 = time.time()
    run = Path(a.run)
    frozen = check_frozen()
    cfg = load_frozen()
    check_heuristics()
    s6 = S10.load_checked(a.stage6, STAGE6_SHA256)[0]
    r12 = S10.load_checked(a.r12 or run / "r12" / "r12_summary.json", R12_SUMMARY_SHA256)[0]
    engine_sha = sha256_file(Path(C.__file__))
    out = [f"=== r13: forecast with a bootstrap upper bound on rho_I ({LABEL}) ===",
           f"delta {C.DELTA}; bootstrap {BOOT_N} resamples, {BOOT_Q:g}th percentile; seed base {SEED}. Criteria: exact "
           "binomial only; a pass means 'conservative within 5 points', not 'accurate'.", ""]
    summary = {"label": LABEL, "frozen_blobs": frozen}

    # every real-data load and stop-check runs before any forecast is computed (a crash or failed check here prints
    # no r13 number); --check-only stops after this block and writes nothing
    B = cells = L = None
    if "b" in a.sets:
        B = load_btsc()
        cells = check_btsc(B, s6)
        print(f"  set (b) checks: {B['checks']}", flush=True)
    if "c" in a.sets or "d" in a.sets:
        L = load_gtsrb(run, cfg, r12)
        print("  set (c)/(d) checks: Stage 3b HB, r12 exact-binomial and r12 pilot forecasts reproduce: True", flush=True)
    if a.check_only:
        print(f"CHECK-ONLY: all loads and stop-checks passed ({time.time() - t0:.1f} s); no forecast computed, nothing written.")
        return None

    sim_settings, sim_cal = [], []
    if "a" in a.sets:
        for pair, (rho, m) in enumerate(SIM_PAIRS):
            st, cal = sim_pair(pair, rho, m, a.runs, a.n_pop)
            sim_settings += st
            sim_cal.append(cal)
            print(f"  set (a) pair {pair} rho {rho:.2f} m {m:2d}: done ({time.time() - t0:.0f} s)", flush=True)
        out.append("--- SET (a): SIMULATED HELD-OUT (judged) ---")
        for c in sim_cal:
            out.append(f"  rho {c['rho']:.2f} m {c['m']:2d}: r_latent {c['r_latent']:.4f}, view error "
                       f"{100 * c['calibration']['view_error']:.3f}%, wrong ICC {c['calibration']['wrong_icc']:.4f}")
        out.append(f"  {'rho':>4} {'m':>3} {'alpha':>5} {'test':>5} | {'observed':>7} {'CP95':>16} | rule | {'mean F':>7} "
                   f"{'median':>7} {'p10-p90':>17} | share>=O-2 meanF-O | (i) (ii) | heur. | bnd R_b | cases bb/le0/zero/ge1/none | rho_in med")
        for s in sim_settings:
            b = s["boundary"]
            rb = "n/a" if b["R_b"] is None else f"{100 * b['R_b']:.3f}%"
            for t in TESTS:
                r = s["rows"][t]
                for rule in ("r11", "r13"):
                    x = r[rule]
                    c, cs = x["crit"], x["cases"]
                    rh = "n/a" if x["rho_in_median"] is None else f"{x['rho_in_median']:.4f}"
                    out.append(
                        f"  {s['rho']:4.2f} {s['m']:3d} {100 * s['alpha']:4.1f}% {t:>5} | {pct(r['observed'])} {cp(r['cp95'])} | "
                        f"{rule} | {pct(x['mean'])} {pct(x['median'])} [{100 * x['p10']:5.2f}, {100 * x['p90']:5.2f}]% | "
                        f"{100 * c['share_not_below']:6.1f}% {c['mean_gap_pp']:+7.2f} | {'Y' if c['i'] else 'N':>3} "
                        f"{'Y' if c['ii'] else 'N':>4} | {pct(s['heuristic'])} | {'yes' if b['boundary'] else 'no'} {rb} | "
                        f"{cs['betabinom']}/{cs['rho_le_0']}/{cs['zero_errors']}/{cs['rho_ge_1']}/{cs[None]} | {rh}")
        judged = [s["rows"]["binom"]["r13"]["crit"] for s in sim_settings]
        n_both = sum(c["both"] for c in judged)
        crit_a = n_both >= CRIT_MIN_SETTINGS
        gaps = [s["rows"]["binom"]["r13"]["mean"] - s["rows"]["binom"]["observed"] for s in sim_settings]
        out.append(f"  MAE (reported, not judged), exact binomial, r13: {100 * np.mean(np.abs(gaps)):.2f} pp over "
                   f"{len(gaps)} settings; r11 rule: "
                   f"{100 * np.mean([abs(s['rows']['binom']['r11']['mean'] - s['rows']['binom']['observed']) for s in sim_settings]):.2f} pp")
        out.append(f"  CRITERION (a): settings with (i) and (ii), exact binomial, r13: {n_both} of {len(judged)} "
                   f"(need >= {CRIT_MIN_SETTINGS}) -> {'PASS' if crit_a else 'FAIL'}")
        out.append("")
        summary["set_a"] = {"settings": sim_settings, "calibration": sim_cal, "n_both": n_both, "pass": crit_a}

    if "b" in a.sets:
        bt = run_btsc(B, cells)
        out.append("--- SET (b): BELGIUMTSC (judged; out-of-sample for the forecast, not blind to its targets) ---")
        out.append(f"  Checks: {B['checks']}")
        out.append(f"  Pilot 133 design poles, 465 views; m* = {bt['m_star']:.4f}, m = {bt['m']}; model views per draw "
                   f"{bt['model_views_per_draw']} against real {bt['real_views_per_draw']} (mismatch real - model: "
                   f"{bt['mismatch_views']}; known approximation)")
        out.append(f"  {'alpha':>5} {'test':>5} | {'observed':>7} {'CP95':>16} | rule | {'forecast':>8} | j_hat start "
                   f"{'U':>7} {'f':>6} rho_pt rho_in case N_g k_max | j_true | Stage 6 starts")
        for key, r in bt["rows"].items():
            al, t = key.split("_")
            for rule in ("r11", "r13"):
                x = r[rule]
                u = "n/a" if x["j_star"] < 0 else f"{100 * x['U']:.3f}%"
                ff = "n/a" if x["j_star"] < 0 else f"{100 * x['f']:.1f}%"
                rp = "n/a" if x["rho_point"] is None else f"{x['rho_point']:.4f}"
                ri = "n/a" if x["rho_in"] is None else f"{x['rho_in']:.4f}"
                out.append(f"  {100 * float(al):4.1f}% {t:>5} | {pct(r['observed'])} {cp(r['cp95'])} | {rule} | "
                           f"{pct(x['forecast'])} | {x['j_star']:2d} {x['start']:2d} {u} {ff} {rp} {ri} {x['case']} "
                           f"{x['N_g']} {x.get('k_max')} | {r['j_true']} | {r['stage6_starts']}")
        cb = {al: bt["rows"][f"{al}_binom"]["crit"] for al in BTSC_JUDGED}
        crit_b = all(c["pass"] for c in cb.values())
        gb = [bt["rows"][f"{al}_binom"]["r13"]["forecast"] - bt["rows"][f"{al}_binom"]["observed"] for al in BTSC_JUDGED]
        out.append(f"  MAE (reported, not judged), exact binomial, r13, alpha 0.5% and 1%: {100 * np.mean(np.abs(gb)):.2f} pp")
        out.append(f"  Heuristic (reported only): {pct(bt['heuristic'])}")
        for al, c in cb.items():
            o = bt["rows"][f"{al}_binom"]["observed"]
            f_ = bt["rows"][f"{al}_binom"]["r13"]["forecast"]
            out.append(f"  CRITERION (b) alpha {100 * al:.1f}%: observed {100 * o:.2f}%, forecast {100 * f_:.2f}%, gap "
                       f"{c['gap_pp']:+.2f} pp; window [{100 * o - CRIT_UNDER_PP:.2f}%, {100 * o + CRIT_OVER_PP:.2f}%] -> "
                       f"{'PASS' if c['pass'] else 'FAIL'}")
        out.append(f"  CRITERION (b): {'PASS' if crit_b else 'FAIL'}")
        out.append("")
        summary["set_b"] = {**bt, "checks": B["checks"], "crit": {str(k): v for k, v in cb.items()}, "pass": crit_b}

    if "c" in a.sets or "d" in a.sets:
        rows, curve, obs = run_gtsrb(L)
        out.append("--- SET (c): GTSRB (development data; reported only) ---")
        for (al, t), r in rows.items():
            out.append(f"  {100 * al:4.1f}% {t:>5}: observed {pct(r['observed'])}; r11 rule {pct(r['r11']['forecast'])} "
                       f"(rho {r['r11']['rho_in']}); r13 rule {pct(r['r13']['forecast'])} (rho_in {r['r13']['rho_in']}); "
                       f"r13 pilot bootstrap 10-90% [{100 * r['boot_p10']:.2f}, {100 * r['boot_p90']:.2f}]%, median "
                       f"{pct(r['boot_median'])}")
        out.append("--- SET (d): GTSRB PILOT-SIZE CURVE (descriptive; pilots from the pool, not independent of it) ---")
        for (size, al, t, rule), x in curve.items():
            out.append(f"  size {size:3d} {100 * al:4.1f}% {t:>5} {rule}: mean {pct(x['mean'])}, mean bias "
                       f"{100 * x['mean_bias']:+6.2f} pp, 10-90% [{100 * x['p10']:5.2f}, {100 * x['p90']:5.2f}]%")
        out.append("")
        summary["set_c"] = {f"{k[0]}_{k[1]}": v for k, v in rows.items()}
        summary["set_d"] = {f"{k[0]}_{k[1]}_{k[2]}_{k[3]}": v for k, v in curve.items()}

    if a.sets == "abcd":
        overall = summary["set_a"]["pass"] and summary["set_b"]["pass"]
        out += ["OVERALL (Part A (d)): (a) " + ("PASS" if summary["set_a"]["pass"] else "FAIL") + ", (b) "
                + ("PASS" if summary["set_b"]["pass"] else "FAIL") + f" -> {'PASS' if overall else 'FAIL'}"
                + " (a pass means 'conservative within 5 points', not 'accurate')"]
        summary["overall_pass"] = overall
    print("\n".join(out), flush=True)

    if sha256_file(Path(C.__file__)) != engine_sha:
        raise SystemExit("STOP: the engine changed during the run; no summary written")
    od = Path(a.out)
    od.mkdir(parents=True, exist_ok=True)
    summary.update({"code_sha256": {"r13_forecast.py": sha256_file(Path(__file__)), "r11_forecast_ucb.py":
                                    sha256_file(Path(F.__file__)), "r12_gtsrb_forecast.py": sha256_file(Path(G.__file__)),
                                    "r9_certify.py": engine_sha},
                    "inputs_sha256": {"stage6_summary.json": STAGE6_SHA256, "r12_summary.json": R12_SUMMARY_SHA256},
                    "config": {"runs": a.runs, "n_pop": a.n_pop, "sets": a.sets, "seed": SEED, "boot_n": BOOT_N,
                               "boot_q": BOOT_Q, "btsc_boot_key": list(BTSC_BOOT_KEY)},
                    "seconds": round(time.time() - t0, 1)})
    path = od / "r13_summary.json"
    path.write_text(json.dumps(S2.jsonable(summary), indent=2))
    (od / "r13_report.md").write_text("\n".join(out) + "\n")
    print(f"\nOutput: {path} SHA-256 {sha256_file(path)} and r13_report.md ({summary['seconds']} s). EXPLORATORY.")
    return summary


if __name__ == "__main__":
    main()
