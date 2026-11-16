"""r11 (exploratory, synthetic): forecast of the frame-level (B1) false-certificate rate from a pilot upper bound.

Definitions: LOG_r11_entry.md Part A (committed in 867db93 before this file existed) and Amendment 1 (rho_I >= 1
uses the rho = 1 limit of the beta-binomial instead of stopping).
  Per run: pilot = the run's 261-sign design sample. U_j = one-sided upper 80% Clopper-Pearson bound of the pilot B1
  risk at threshold j; j* = first j in the engine's walk order (per test) with U_j > alpha. Beta-binomial model with
  the pilot ICC rho_I at j* and the risk set to U_j*; run forecast = P(test certifies at j*), computed exactly.
  Setting forecast = mean over runs. Compared with the observed B1 false-certificate rate (95% CP) and with the
  heuristic Phi(Phi^-1(delta) / sqrt(1 + (m - 1) rho)).
  Sets: development (Stage 10, regenerated and checked), replication (same settings, new seeds), held-out (judged by
  Part A (e), exact binomial only).

Usage:  py -3.12 -m src.r11_forecast_ucb
Output: runs/20261003_stage1/r11/r11_summary.json and r11_report.md
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.stats import beta, betabinom, binom

from src import r8_stage3_engine as E
from src import r8_stage3_synth as R8S
from src import r9_certify as C
from src import r9_stage10 as S10
from src import r9_stage10_popcurves as Q
from src import r9_stage2_popcurves as P
from src import r9_stage2_vchecks as V
from src.r9_frozen import check_frozen
from src.util import sha256_file

RUNS = 2000                        # Part A (c): Stage 10 RUNS
UCB_CONF = 0.80                    # Part A (b) 2: one-sided upper bound
TESTS = ("binom", "hb")            # binom primary and judged; hb reported
SEED_NEW = 20261118                # Part A (c): unused by any earlier script
HELD_RHOS, HELD_MS = (0.3, 0.5), (8, 12)
HELD_PAIRS = [(rho, m) for rho in HELD_RHOS for m in HELD_MS]   # pair index 0..3 as in Part A (c)
HELD_HEURISTIC_PCT = {(0.3, 8): 23.33, (0.3, 12): 26.83, (0.5, 8): 27.29, (0.5, 12): 30.76}
HEUR_TOL_PCT = 0.005
CURVES_PATH = Q.OUT / Q.CURVES_NAME
CURVES_SHA = S10.CURVES_SHA                                   # ffa785ea...119f, Part A (c)
STAGE10_PATH = Q.OUT / "stage10_summary.json"
STAGE10_SHA = "4f7ce88317ce251ef6e0e2aad305cb3d5b46999b04a8927b307b2edbed7d5a15"   # LOG.md, Stage 10 Part B
OUT = Path("runs/20261003_stage1/r11")
# Part A (e), verbatim thresholds
CRIT_MAE_PP, CRIT_CLOSER_MIN, CRIT_UNDER_PP, CRIT_UNDER_MIN = 3.0, 7, 2.0, 7


def rng_new(pair, stream):
    """Replication: pair 0..8; held-out: 100 + pair."""
    return np.random.default_rng([SEED_NEW, 11, pair, stream])


# ------------------------------------------------------------------ forecast pieces (Part A (b))
def cp_upper(W, A, conf=UCB_CONF):
    """One-sided upper Clopper-Pearson bound of W / A; 1 where W = A or A = 0."""
    W, A = np.asarray(W, float), np.asarray(A, float)
    out = np.ones_like(A)
    ok = (A > 0) & (W < A)
    out[ok] = beta.ppf(conf, W[ok] + 1.0, A[ok] - W[ok])
    return out


def icc_anova(groups):
    """One-way ANOVA intraclass correlation for 0/1 values in groups of unequal size (r10 Part 3 formula, copied).
    groups: list of (n_i, k_i)."""
    n = np.array([g[0] for g in groups], float)
    k = np.array([g[1] for g in groups], float)
    N, G = n.sum(), len(n)
    grand = k.sum() / N
    msb = ((k - n * grand) ** 2 / n).sum() / (G - 1)
    msw = (k - k ** 2 / n).sum() / (N - G)
    n0 = (N - (n ** 2).sum() / N) / (G - 1)
    return float((msb - msw) / (msb + (n0 - 1) * msw))


def j_star(U, start, alpha):
    """First index from 'start' down to 0 with U_j > alpha; -1 if start is -1 or none qualifies."""
    if start < 0:
        return -1
    idx = np.flatnonzero(np.asarray(U)[: start + 1] > alpha)
    return int(idx.max()) if len(idx) else -1


def j_true(mu_true, start, alpha):
    """First index in walk order whose population mean loss is above alpha (same rule as E.violations); -1 if none."""
    if start < 0:
        return -1
    idx = np.flatnonzero(np.asarray(mu_true)[: start + 1] > alpha + E.TOL)
    return int(idx.max()) if len(idx) else -1


_KMAX = {}


def k_max(n_c, n, alpha, test):
    """Largest k whose p-value at n_c committed frames is <= delta; -1 if none. p is non-decreasing in k."""
    key = (int(n_c), int(n), float(alpha), test)
    if key not in _KMAX:
        if n_c <= 0:
            _KMAX[key] = -1
        else:
            ks = np.arange(int(n_c) + 1)
            p = C.pvalues(ks, np.full(len(ks), n_c), n, alpha, test)
            ok = np.flatnonzero(p <= C.DELTA)
            _KMAX[key] = int(ok.max()) if len(ok) else -1
    return _KMAX[key]


def clipped_pmf(m, p, rho, n_g):
    """pmf of min(K, n_g), K ~ BetaBinomial(m, p, rho) (binomial if rho <= 0); also the clipped mass P(K > n_g)."""
    ks = np.arange(m + 1)
    if p <= 0:
        full = (ks == 0).astype(float)
    elif p >= 1:
        full = (ks == m).astype(float)
    elif rho >= 1:                                   # Amendment 1: the rho = 1 limit, all m frames wrong or none
        full = np.where(ks == m, p, 0.0) + np.where(ks == 0, 1.0 - p, 0.0)
    elif rho > 0:
        s = 1.0 / rho - 1.0
        full = betabinom.pmf(ks, m, p * s, (1.0 - p) * s)
    else:
        full = binom.pmf(ks, m, p)
    out = full[: n_g + 1].copy()
    clip = float(full[n_g + 1:].sum())
    out[n_g] += clip
    return out, clip


def _conv_trunc(a, b, L):
    return np.convolve(a, b)[:L]


def total_cdf(pmf, n_units, kmax):
    """P(sum of n_units iid draws from pmf <= kmax), exactly, by binary powering truncated at kmax + 1 entries
    (entries 0..kmax of a convolution of non-negative sequences depend only on entries 0..kmax)."""
    if kmax < 0:
        return 0.0
    L = kmax + 1
    base = np.asarray(pmf, float)[:L]
    res = np.zeros(L)
    res[0] = 1.0
    e = n_units
    while e:
        if e & 1:
            res = _conv_trunc(res, base, L)
        e >>= 1
        if e:
            base = _conv_trunc(base, base, L)
    return float(min(res.sum(), 1.0))


def run_forecast(acc_g, wr_g, m, alpha, test, start, n_design, n_cert):
    """One run, one test. acc_g, wr_g: (n_design, G) accepted and accepted-and-wrong frames per pilot sign.
    Returns a dict with the forecast and its inputs (Part A (b) 1-6)."""
    A, W = acc_g.sum(0), wr_g.sum(0)
    U = cp_upper(W, A)
    js = j_star(U, start, alpha)
    rec = {"j_star": js, "forecast": 0.0, "clipped": 0.0, "rho_I": None, "binomial_fallback": None, "N_g": None,
           "case": None}
    if js < 0:
        return rec
    f = float(A[js] / (n_design * m))
    p = float(U[js] * f)
    n_g = int(round(m * f))
    if W[js] > 0:
        rho = icc_anova([(m, int(k)) for k in wr_g[:, js]])
        if not np.isfinite(rho):
            raise SystemExit(f"STOP: rho_I = {rho} at j* = {js}; Part A (b) 4 does not define this case")
        case = "rho_ge_1" if rho >= 1 else ("rho_le_0" if rho <= 0 else "betabinom")
    else:
        rho, case = 0.0, "zero_errors"
    fallback = rho <= 0
    rec.update({"rho_I": rho, "binomial_fallback": bool(fallback), "N_g": n_g, "f": f, "p": p, "U": float(U[js]),
                "case": case})
    if n_g == 0:
        return rec
    pmf, clip = clipped_pmf(m, p, 0.0 if fallback else rho, n_g)
    km = k_max(n_cert * n_g, n_cert * m, alpha, test)
    rec.update({"clipped": clip, "k_max": km, "forecast": total_cdf(pmf, n_cert, km)})
    return rec


# ------------------------------------------------------------------ curves for the held-out pairs (Q.pair_curves, rng injected)
def b1_curves(rho, m, rng_of, n_pop=P.N_POP):
    """Q.grids + the B1 part of Q.pair_curves with a given generator per stream (stream 0 pilot, 1 population)."""
    r, mu = P.solve_r(rho, Q.P_FRAME), P.m_of(Q.P_FRAME)
    Xp = Q.draw_fixed(P.N_PILOT, m, r, mu, rng_of(0))
    bgrid = V.b1_grid(Xp, np.full(P.N_PILOT, m))
    rng = rng_of(1)
    A = W = np.zeros(P.G)
    nf = nw = nw2 = npairs = 0.0
    done = 0
    while done < n_pop:
        b = min(P.CHUNK, n_pop - done)
        X = Q.draw_fixed(b, m, r, mu, rng)
        a_, w_, n_ = V.frame_totals(X, np.full(b, m), bgrid)
        A, W, nf = A + a_, W + w_, nf + n_
        k = (X < 0).sum(1).astype(float)
        nw, nw2, npairs = nw + k.sum(), nw2 + (k * (k - 1)).sum(), npairs + b * m * (m - 1)
        done += b
    pf, p11 = nw / nf, nw2 / npairs
    return {"rho": rho, "m": m, "r_latent": r, "mu": mu, "b1_grid": bgrid, "b1_pop_accept": A / nf,
            "b1_true_risk": np.where(A > 0, W / np.maximum(A, 1), np.nan),
            "calibration": {"view_error": float(pf), "wrong_icc": float((p11 - pf * pf) / (pf * (1 - pf)))}}


# ------------------------------------------------------------------ the run loop
def simulate(rng, m, r, mu, bgrid, nd, nc, runs, alphas, mu_true):
    """Stage 10 draw order (design then Cert per run). B1 certification with the frozen engine for both tests, and the
    forecast per run. mu_true[alpha]: population mean loss per threshold."""
    Nd, Nc = nd * m, nc * m
    cert = {(a, t): np.full(runs, -1) for a in alphas for t in TESTS}
    fc = {(a, t): [] for a in alphas for t in TESTS}
    for i in range(runs):
        Xd = Q.draw_fixed(nd, m, r, mu, rng)
        Xc = Q.draw_fixed(nc, m, r, mu, rng)
        acc_g, wr_g = R8S.frame_counts(Xd, bgrid)
        Ad = acc_g.sum(0)
        Ac, Wc, _ = V.frame_totals(Xc, np.full(nc, m), bgrid)
        for a in alphas:
            for t in TESTS:
                cert[(a, t)][i] = C.certify(Wc, Ac, Nc, a, Ad / Nd, t, n_design=Nd)
                st = C.start_for(t, Ad / Nd, Nc, a, Nd)
                rec = run_forecast(acc_g, wr_g, m, a, t, st, nd, nc)
                rec["j_true"] = j_true(mu_true[a], st, a)
                fc[(a, t)].append(rec)
    return cert, fc


def setting_rows(cert, fc, mu_true, alpha, heuristic):
    rows = {}
    for t in TESTS:
        obs = E.rate_row(cert[(alpha, t)], mu_true, alpha)
        f = np.array([r["forecast"] for r in fc[(alpha, t)]])
        jt = np.array([r["j_true"] for r in fc[(alpha, t)]])
        js = np.array([r["j_star"] for r in fc[(alpha, t)]])
        has_true = jt >= 0
        rhos = [r["rho_I"] for r in fc[(alpha, t)] if r["rho_I"] is not None]
        o, lo, hi = obs["violation_rate"], obs["cp95"][0], obs["cp95"][1]
        fm = float(f.mean())
        rows[t] = {"observed": o, "cp95": obs["cp95"], "violations": obs["violations"],
                   "certified_share": obs["certified_share"], "forecast_mean": fm,
                   "forecast_median": float(np.median(f)), "forecast_p10": float(np.quantile(f, 0.1)),
                   "forecast_p90": float(np.quantile(f, 0.9)), "heuristic": heuristic,
                   "forecast_inside_cp": bool(lo <= fm <= hi), "heuristic_inside_cp": bool(lo <= heuristic <= hi),
                   "gap_forecast_pp": 100 * (fm - o), "gap_heuristic_pp": 100 * (heuristic - o),
                   "share_forecast_zero": float(np.mean(f == 0)),
                   "runs_with_j_true": int(has_true.sum()),
                   "share_jstar_at_or_before_jtrue": float(np.mean((js[has_true] >= jt[has_true]) & (js[has_true] >= 0)))
                   if has_true.any() else None,
                   "clipped_mass_mean": float(np.mean([r["clipped"] for r in fc[(alpha, t)]])),
                   "share_binomial_fallback": float(np.mean([bool(r["binomial_fallback"]) for r in fc[(alpha, t)]
                                                             if r["binomial_fallback"] is not None]))
                   if any(r["binomial_fallback"] is not None for r in fc[(alpha, t)]) else None,
                   "rho_I_median": float(np.median(rhos)) if rhos else None,
                   "n_rho_ge_1": sum(r["case"] == "rho_ge_1" for r in fc[(alpha, t)]),
                   "n_zero_errors": sum(r["case"] == "zero_errors" for r in fc[(alpha, t)]),
                   "n_rho_le_0": sum(r["case"] == "rho_le_0" for r in fc[(alpha, t)])}
    return rows


def set_summary(settings, test):
    rows = [s["rows"][test] for s in settings]
    if not rows:
        return None
    of = np.array([r["observed"] for r in rows]) * 100
    ff = np.array([r["forecast_mean"] for r in rows]) * 100
    hh = np.array([r["heuristic"] for r in rows]) * 100
    closer = np.abs(ff - of) < np.abs(hh - of)
    under_ok = ff >= of - CRIT_UNDER_PP
    return {"n": len(rows), "mae_forecast_pp": float(np.abs(ff - of).mean()),
            "mae_heuristic_pp": float(np.abs(hh - of).mean()),
            "mean_signed_forecast_pp": float((ff - of).mean()), "mean_signed_heuristic_pp": float((hh - of).mean()),
            "n_forecast_closer": int(closer.sum()), "n_forecast_not_below_obs_minus_2": int(under_ok.sum()),
            "n_forecast_inside_cp": int(sum(r["forecast_inside_cp"] for r in rows)),
            "n_heuristic_inside_cp": int(sum(r["heuristic_inside_cp"] for r in rows))}


def criteria(summ):
    """Part A (e), held-out, exact binomial only."""
    a = summ["mae_forecast_pp"] <= CRIT_MAE_PP
    b = summ["n_forecast_closer"] >= CRIT_CLOSER_MIN
    c = summ["n_forecast_not_below_obs_minus_2"] >= CRIT_UNDER_MIN
    return {"a_mae_le_3": bool(a), "b_closer_ge_7_of_8": bool(b), "c_not_below_ge_7_of_8": bool(c),
            "all": bool(a and b and c)}


def check_held_heuristic():
    out = []
    for (rho, m), want in HELD_HEURISTIC_PCT.items():
        got = 100 * Q.delta_eff(rho, m)
        if abs(got - want) > HEUR_TOL_PCT:
            raise SystemExit(f"STOP: heuristic({rho}, {m}) = {got:.4f}% differs from Part A (c) {want}%")
        out.append({"rho": rho, "m": m, "DE": Q.design_effect(rho, m), "delta_eff": got / 100, "table_pct": want})
    return out


def run_pair(set_name, rho, m, cv, rng, runs, alphas):
    r, mu, bgrid = cv["r_latent"], cv["mu"], np.asarray(cv["b1_grid"], float)
    acc, risk = np.asarray(cv["b1_pop_accept"], float), np.array([np.nan if v is None else v for v in cv["b1_true_risk"]])
    mu_true = {a: S10.mean_loss(a, acc, risk) for a in alphas}
    cert, fc = simulate(rng, m, r, mu, bgrid, Q.N_DESIGN, Q.N_CERT, runs, alphas, mu_true)
    heur = Q.delta_eff(rho, m)
    settings = []
    for a in alphas:
        b = Q.boundary(acc, risk, Q.N_CERT * m, a)
        settings.append({"set": set_name, "rho": rho, "m": m, "alpha": a, "DE": Q.design_effect(rho, m),
                         "boundary": b, "rows": setting_rows(cert, fc, mu_true[a], a, heur)})
    return settings


# ------------------------------------------------------------------ printing
def pct(x, nd=2):
    return "    n/a" if x is None else f"{100 * x:6.{nd}f}%"


def table(settings, lines):
    lines.append(f"  {'rho':>4} {'m':>3} {'alpha':>5} {'test':>5} | {'observed':>7} {'CP95':>18} | {'forecast':>8} "
                 f"{'median':>7} {'p10-p90':>17} | {'heur.':>7} | {'gap F':>6} {'gap H':>6} | F in CP | "
                 f"{'zero':>5} {'j*<=jt':>6} {'clip':>8} {'binom fb':>8} {'rho_I med':>9} | rho>=1 W=0 rho<=0")
    for s in settings:
        for t in TESTS:
            r = s["rows"][t]
            jt = "  n/a" if r["share_jstar_at_or_before_jtrue"] is None else f"{100 * r['share_jstar_at_or_before_jtrue']:5.1f}%"
            fb = "   n/a" if r["share_binomial_fallback"] is None else f"{100 * r['share_binomial_fallback']:6.1f}%"
            rh = "      n/a" if r["rho_I_median"] is None else f"{r['rho_I_median']:9.4f}"
            lines.append(
                f"  {s['rho']:4.1f} {s['m']:3d} {100 * s['alpha']:4.1f}% {t:>5} | {pct(r['observed'])} "
                f"[{100 * r['cp95'][0]:5.2f}, {100 * r['cp95'][1]:5.2f}]% | {pct(r['forecast_mean'])} "
                f"{pct(r['forecast_median'])} [{100 * r['forecast_p10']:5.2f}, {100 * r['forecast_p90']:5.2f}]% | "
                f"{pct(r['heuristic'])} | {r['gap_forecast_pp']:+6.2f} {r['gap_heuristic_pp']:+6.2f} | "
                f"{'yes' if r['forecast_inside_cp'] else 'no':>7} | {100 * r['share_forecast_zero']:4.1f}% {jt} "
                f"{r['clipped_mass_mean']:8.2e} {fb} {rh} | {r['n_rho_ge_1']:6d} {r['n_zero_errors']:4d} "
                f"{r['n_rho_le_0']:6d}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=RUNS)
    ap.add_argument("--n-pop", type=int, default=P.N_POP)
    ap.add_argument("--curves", default=str(CURVES_PATH))
    ap.add_argument("--stage10", default=str(STAGE10_PATH))
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args(argv)
    t0 = time.time()
    frozen = check_frozen()
    engine_sha = sha256_file(Path(C.__file__))
    Q.check_predictions()
    held_heur = check_held_heuristic()
    curves, curves_sha = S10.load_checked(a.curves, CURVES_SHA)
    s10, s10_sha = S10.load_checked(a.stage10, STAGE10_SHA)
    alphas = Q.ALPHAS
    s10_by = {(s["pair"], s["alpha"]): s for pr in s10["pairs"] for s in pr["settings"]}

    lines = [f"=== r11: pilot upper-bound forecast of the B1 false-certificate rate (EXPLORATORY, synthetic) ===",
             f"Part A: LOG_r11_entry.md (867db93). delta {C.DELTA}; UCB one-sided {UCB_CONF:.0%}; {a.runs} runs per "
             f"setting; Cert {Q.N_CERT} / design {Q.N_DESIGN} signs; P_FRAME {Q.P_FRAME}.",
             "Seeds: development [20261115, 10, pair, 2] (regenerated); replication [20261118, 11, pair, 2], pair 0..8; "
             "held-out [20261118, 11, 100 + pair, stream], streams 0 pilot grid, 1 population, 2 runs.",
             f"Inputs checked: {Q.CURVES_NAME} {curves_sha}; stage10_summary.json {s10_sha}.", ""]
    print("\n".join(lines), flush=True)

    sets = {"development": [], "replication": [], "held-out": []}
    repro = {}
    for pair, (rho, m) in enumerate(Q.PAIRS):
        cv = curves["curves"][pair]
        sets["development"] += run_pair("development", rho, m, cv, Q.rng_for(pair, 2), a.runs, alphas)
        for s in sets["development"][-len(alphas):]:
            want = s10_by[(pair, s["alpha"])]["rows"]
            for t in TESTS:
                ok = (s["rows"][t]["violations"] == want[f"B1_{t}"]["violations"]
                      and s["rows"][t]["certified_share"] == want[f"B1_{t}"]["certified_share"])
                repro[f"pair{pair}_a{s['alpha']}_{t}"] = bool(ok)
                if a.runs == RUNS and not ok:
                    raise SystemExit(f"STOP: development pair {pair} alpha {s['alpha']} {t} does not reproduce Stage 10")
        sets["replication"] += run_pair("replication", rho, m, cv, rng_new(pair, 2), a.runs, alphas)
        print(f"  pair {pair} rho {rho:.1f} m {m:2d}: development + replication done ({time.time() - t0:.0f} s)", flush=True)
    held_curves = []
    for hp, (rho, m) in enumerate(HELD_PAIRS):
        cv = b1_curves(rho, m, lambda s, hp=hp: rng_new(100 + hp, s), a.n_pop)
        held_curves.append(cv)
        sets["held-out"] += run_pair("held-out", rho, m, cv, rng_new(100 + hp, 2), a.runs, alphas)
        print(f"  held-out pair {hp} rho {rho:.1f} m {m:2d}: done ({time.time() - t0:.0f} s)", flush=True)

    summ = {name: {t: set_summary(st, t) for t in TESTS} for name, st in sets.items()}
    crit = criteria(summ["held-out"]["binom"])

    out = [""]
    out.append(f"Development reproduces Stage 10 (B1 binom and hb violations and certified share, 36 cells): "
               f"{all(repro.values())}" + ("" if a.runs == RUNS else f" (runs {a.runs} != {RUNS}: check not applied)"))
    out.append("\nHeld-out calibration and boundary diagnostic (Stage 10 Part A (h)):")
    for cv in held_curves:
        out.append(f"  rho {cv['rho']:.1f} m {cv['m']:2d}: r_latent {cv['r_latent']:.4f}, view error "
                   f"{100 * cv['calibration']['view_error']:.3f}%, wrong ICC {cv['calibration']['wrong_icc']:.4f}")
    for s in sets["held-out"]:
        b = s["boundary"]
        rb = "n/a" if b["R_b"] is None else f"{100 * b['R_b']:.3f}%"
        out.append(f"  rho {s['rho']:.1f} m {s['m']:2d} alpha {100 * s['alpha']:.0f}%: start {b['start']}, j_b "
                   f"{b['first_above_idx']}, R_b {rb}, boundary {'yes' if b['boundary'] else 'no'}")
    out.append("\nColumns: observed B1 false-certificate rate [95% CP]; forecast mean, median, 10-90% range over runs; "
               "heuristic; gaps (prediction - observed, pp); forecast inside CP; share of runs with forecast 0; share "
               "of runs (with a true risky threshold) where j* is at or before j_true; mean clipped mass per sign; "
               "share of binomial fallback; median rho_I; pilots with rho_I >= 1 (Amendment 1 limit), with zero "
               "errors at j*, with rho_I <= 0.")
    for name, st in sets.items():
        out.append(f"\n--- {name.upper()} ---")
        table(st, out)
        for t in TESTS:
            x = summ[name][t]
            out.append(f"  summary {t:>5}: n {x['n']}; MAE forecast {x['mae_forecast_pp']:.2f} pp, heuristic "
                       f"{x['mae_heuristic_pp']:.2f} pp; mean signed forecast {x['mean_signed_forecast_pp']:+.2f}, "
                       f"heuristic {x['mean_signed_heuristic_pp']:+.2f} pp; forecast closer {x['n_forecast_closer']}/{x['n']}; "
                       f"forecast >= observed - 2 pp {x['n_forecast_not_below_obs_minus_2']}/{x['n']}; inside CP: "
                       f"forecast {x['n_forecast_inside_cp']}, heuristic {x['n_heuristic_inside_cp']}")
    hs = summ["held-out"]["binom"]
    out += ["\nHELD-OUT PASS CRITERIA (Part A (e); exact binomial only; HB and replication not judged):",
            f"  (a) MAE of the exact-binomial forecast vs observed <= 3 points: {hs['mae_forecast_pp']:.2f} -> "
            f"{'PASS' if crit['a_mae_le_3'] else 'FAIL'}",
            f"  (b) forecast closer to observed than the heuristic in >= 7 of 8 settings: {hs['n_forecast_closer']} of "
            f"{hs['n']} -> {'PASS' if crit['b_closer_ge_7_of_8'] else 'FAIL'}",
            f"  (c) forecast >= observed - 2 points in >= 7 of 8 settings: {hs['n_forecast_not_below_obs_minus_2']} of "
            f"{hs['n']} -> {'PASS' if crit['c_not_below_ge_7_of_8'] else 'FAIL'}",
            f"  Held-out check: {'PASS' if crit['all'] else 'FAIL'}"]
    print("\n".join(out), flush=True)

    if sha256_file(Path(C.__file__)) != engine_sha:
        raise SystemExit("STOP: the engine changed during the run; no summary written")
    od = Path(a.out)
    od.mkdir(parents=True, exist_ok=True)
    summary = {"label": "EXPLORATORY (synthetic); LOG_r11_entry.md Part A, commit 867db93",
               "frozen_blobs": frozen,
               "code_sha256": {"r11_forecast_ucb.py": sha256_file(Path(__file__)), "r9_certify.py": engine_sha,
                               "r9_stage10.py": sha256_file(Path(S10.__file__)),
                               "r9_stage10_popcurves.py": sha256_file(Path(Q.__file__)),
                               "r9_stage2_popcurves.py": sha256_file(Path(P.__file__)),
                               "r9_stage2_vchecks.py": sha256_file(Path(V.__file__)),
                               "r8_stage3_synth.py": sha256_file(Path(R8S.__file__)),
                               "r8_stage3_engine.py": sha256_file(Path(E.__file__))},
               "inputs_sha256": {Q.CURVES_NAME: curves_sha, "stage10_summary.json": s10_sha},
               "config": {"delta": C.DELTA, "ucb_conf": UCB_CONF, "runs": a.runs, "n_pop": a.n_pop,
                          "n_cert": Q.N_CERT, "n_design": Q.N_DESIGN, "p_frame": Q.P_FRAME, "alphas": alphas,
                          "tests": TESTS, "development_pairs": Q.PAIRS, "held_out_pairs": HELD_PAIRS,
                          "seeds": {"development": "[20261115, 10, pair, 2]", "replication": "[20261118, 11, pair, 2]",
                                    "held-out": "[20261118, 11, 100 + pair, stream]"}},
               "held_out_heuristic": held_heur, "development_reproduces_stage10": repro,
               "held_out_curves": held_curves, "settings": sets, "summary": summ, "held_out_criteria": crit,
               "seconds": round(time.time() - t0, 1)}
    path = od / "r11_summary.json"
    path.write_text(json.dumps(P.jsonable(summary), indent=2))
    (od / "r11_report.md").write_text("\n".join(lines + out) + "\n")
    print(f"\nOutput: {path} SHA-256 {sha256_file(path)} and r11_report.md ({summary['seconds']} s). EXPLORATORY.")
    return summary


if __name__ == "__main__":
    main()
