"""r12b: POST HOC DIAGNOSTIC of the r12 result (GTSRB development data). Not a test, no criteria, not a revised method.

Written after the r12 result (Part B, 18d1f29) was seen. At the true first unsafe threshold jt (first position in the
POC-2 B1 order 19..0 whose pool mean loss is above alpha), on the 1,046 pool tracks:
  1. Oracle forecast: r11 (b) 4-6 with pool inputs (pool risk instead of U, pool f, pool rho_I, jt instead of j*).
  2. Fit check: per-track accepted-and-wrong counts against the clipped beta-binomial(30, pool rate, pool rho_I).
  3. Split at jt (the author's choice of reading): (i) errors given the real acceptance: per-track wrong counts against
     BB(A_g, pool risk, rho) with each track's own accepted count, and an oracle forecast in which Cert tracks keep
     real accepted counts (Monte Carlo); (ii) acceptance: the distribution of A_g against the fixed N_g.
Observed rates and the pilot forecast are read from the hash-checked r12_summary.json.

Usage:  py -3.12 -m src.r12b_diagnostic --run runs/20261003_stage1
Output: runs/20261003_stage1/r12b/r12b_summary.json and r12b_report.md
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.stats import betabinom, binom, chi2

from src import r11_forecast_ucb as F
from src import r12_gtsrb_forecast as G
from src import r8_stage2 as S2
from src import r8_stage3_engine as E
from src import r8_stage3_resample as RS
from src import r9_certify as C
from src.r8_frozen import load_frozen
from src.r9_frozen import check_frozen
from src.util import sha256_file

LABEL = "POST HOC DIAGNOSTIC (GTSRB development data; written after the r12 result; no criteria; not a revised method)"
ALPHAS = (0.005, 0.01)
TESTS = ("hb", "binom")
M, N_CERT, START = 30, 784, 19
R12_SHA256 = "3d559b2fa5ec9de5d25b2d4e01079a083003ff45bee0c2f0b4c5f4a06d18488a"   # LOG_r12_entry.md Part B
MC_KEY, MC_N = (20261119, 13), 20_000
MIN_EXPECTED = 5.0
OUT = Path("runs/20261003_stage1/r12b")


# ------------------------------------------------------------------ pieces (tested on synthetic data)
def oracle_forecast(risk, f, rho, alpha, test, m=M, n_cert=N_CERT):
    """r11 (b) 4-6 with given inputs: p = risk x f, N_g = round(m f), clipped BB (rho <= 0 binomial, rho >= 1 limit)."""
    p = float(risk * f)
    n_g = int(round(m * f))
    if n_g == 0:
        return {"p": p, "N_g": 0, "k_max": -1, "clipped": 0.0, "forecast": 0.0}
    pmf, clip = F.clipped_pmf(m, p, max(rho, 0.0) if rho > 0 else 0.0, n_g)
    km = F.k_max(n_cert * n_g, n_cert * m, alpha, test)
    return {"p": p, "N_g": n_g, "k_max": km, "clipped": clip, "forecast": F.total_cdf(pmf, n_cert, km), "pmf": pmf}


def bb_pmf(n, p, rho):
    """pmf of BetaBinomial(n, p, rho) on 0..n (binomial if rho <= 0, all-or-none if rho >= 1)."""
    ks = np.arange(n + 1)
    if n == 0:
        return np.ones(1)
    if rho <= 0:
        return binom.pmf(ks, n, p)
    if rho >= 1:
        return np.where(ks == n, p, 0.0) + np.where(ks == 0, 1.0 - p, 0.0)
    s = 1.0 / rho - 1.0
    return betabinom.pmf(ks, n, p * s, (1.0 - p) * s)


def mixture_expected(ns, p, rho, kmax):
    """Expected number of tracks with k = 0..kmax when track g has ns[g] trials (one BB per track, summed)."""
    out = np.zeros(kmax + 1)
    for n, c in zip(*np.unique(ns, return_counts=True)):
        q = bb_pmf(int(n), p, rho)
        out[: len(q)] += c * q
    return out


def chi_square(obs_hist, exp_hist, n_params, min_expected=MIN_EXPECTED):
    """Pearson chi-square after merging bins from the top until every expected count is >= min_expected.
    The last bin is open ('>= its lower edge'). df = bins - 1 - n_params. Returns stat, df, p, the merged bins."""
    obs, exp = list(map(float, obs_hist)), list(map(float, exp_hist))
    edges = list(range(len(exp)))
    while len(exp) > 1 and exp[-1] < min_expected:
        e, o = exp.pop(), obs.pop()                         # pop first, then add (a[-2] += a.pop() hits the wrong bin)
        exp[-1] += e
        obs[-1] += o
        edges.pop()
    while len(exp) > 1 and exp[0] < min_expected:          # merge a sparse first bin upwards (rare)
        e, o = exp.pop(0), obs.pop(0)
        exp[0] += e
        obs[0] += o
        edges.pop(1)
    obs, exp = np.array(obs), np.array(exp)
    stat = float(((obs - exp) ** 2 / exp).sum())
    df = len(exp) - 1 - n_params
    p = float(chi2.sf(stat, df)) if df > 0 else None
    return {"stat": stat, "df": int(df), "p": p, "bins_lower_edges": edges, "obs": obs.tolist(), "exp": exp.tolist()}


def categories(k_obs, exp_hist, n_g):
    """Observed vs expected tracks with k = 0, k >= 1, k >= N_g / 2, k = N_g (observed k > N_g shown separately).
    exp_hist covers k = 0..len-1; for the clipped model its last entry (k = N_g) holds the clipped mass."""
    k_obs = np.asarray(k_obs)
    ks = np.arange(len(exp_hist))
    half = n_g / 2.0
    return {"k=0": (int((k_obs == 0).sum()), float(exp_hist[0])),
            "k>=1": (int((k_obs >= 1).sum()), float(exp_hist[1:].sum())),
            "k>=N_g/2": (int((k_obs >= half).sum()), float(exp_hist[ks >= half].sum())),
            "k=N_g": (int((k_obs == n_g).sum()), float(exp_hist[n_g]) if n_g < len(exp_hist) else 0.0),
            "k>N_g (observed only; the clipped model puts this mass at N_g)": (int((k_obs > n_g).sum()),
                                                                              float(exp_hist[ks > n_g].sum()))}


def mc_forecast_real_acceptance(A_pool, risk, rho, alpha, test, rng, n_sets=MC_N, n_cert=N_CERT, m=M, chunk=2000):
    """Cert tracks draw their accepted count from the pool's A_g (with replacement); wrong ~ BB(A, risk, rho).
    Returns P(test p-value at jt <= delta), with HB over n = n_cert x m frames (as r11)."""
    A_pool = np.asarray(A_pool, int)
    hits = 0
    s = None if rho <= 0 or rho >= 1 else 1.0 / rho - 1.0
    for c in range(0, n_sets, chunk):
        b = min(chunk, n_sets - c)
        A = A_pool[rng.integers(0, len(A_pool), (b, n_cert))]
        if rho >= 1:
            K = np.where(rng.random((b, n_cert)) < risk, A, 0)
        elif s is None:
            K = rng.binomial(A, risk)
        else:
            K = rng.binomial(A, rng.beta(risk * s, (1 - risk) * s, (b, n_cert)))
        k, n_c = K.sum(1), A.sum(1)
        p = C.pvalues(k, n_c, n_cert * m, alpha, test)
        hits += int((p <= C.DELTA).sum())
    return hits / n_sets


def load_r12(path, want=R12_SHA256):
    sha = sha256_file(path)
    if sha != want:
        raise SystemExit(f"STOP: {path} SHA-256 {sha} != recorded {want}")
    return json.loads(Path(path).read_text())


def fmt_p(p):
    return "n/a (df <= 0)" if p is None else f"{p:.3g}"


def pct(x, nd=2):
    return "    n/a" if x is None else f"{100 * x:6.{nd}f}%"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/20261003_stage1")
    ap.add_argument("--r12", default=None)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args(argv)
    t0 = time.time()
    run = Path(a.run)
    cfg = load_frozen()
    frozen = check_frozen()
    r12 = load_r12(Path(a.r12) if a.r12 else run / "r12" / "r12_summary.json")
    s3b = G.load_stage3b(run / "poc2" / "stage3" / "stage3b_summary.json")
    d = RS.load_pool(run, cfg)
    P, checks = RS.pool_tables(d, cfg)
    assert d["b1"]["order_positions"] == list(range(START, -1, -1))
    if not all(checks.values()):
        raise SystemExit(f"STOP: POC-2 pool checks failed: {[k for k, v in checks.items() if not v]}")
    A, W, NF = P["A"].astype(int), P["W"].astype(int), P["NF"].astype(int)
    T = len(NF)
    pop = np.ones((1, T))
    rng = np.random.default_rng(list(MC_KEY))

    out = [f"=== r12b: {LABEL} ===",
           f"Pool 1,046 tracks (frames per track: {dict(zip(*map(lambda v: v.tolist(), np.unique(NF, return_counts=True))))}); "
           f"model m = {M}; Cert {N_CERT}; delta {C.DELTA}. Observed rates and pilot forecasts from r12_summary.json "
           f"(SHA-256 checked). Monte Carlo (item 3) key {list(MC_KEY)}, {MC_N:,} Cert sets per cell.",
           f"POC-2 pool checks: {all(checks.values())}", ""]
    res = {}
    for al in ALPHAS:
        mu_true = E.frame_mu(P["A"], P["W"], P["NF"], pop, al)[0][0]
        jt = F.j_true(mu_true, START, al)
        assert jt == r12["rows"][f"{al}_hb"]["j_true"], "jt differs from r12"
        Ag, Wg = A[:, jt], W[:, jt]
        risk, f = Wg.sum() / Ag.sum(), Ag.sum() / NF.sum()
        p_frame = Wg.sum() / NF.sum()
        rho_I = F.icc_anova([(int(n), int(k)) for n, k in zip(NF, Wg)])
        keep = Ag > 0
        rho_acc = F.icc_anova([(int(n), int(k)) for n, k in zip(Ag[keep], Wg[keep])])
        n_g = int(round(M * f))
        r = {"j_true": jt, "pool_risk": float(risk), "pool_f": float(f), "pool_p_frame": float(p_frame),
             "pool_rho_I": rho_I, "pool_rho_accepted": rho_acc, "N_g": n_g}

        # 1. oracle forecast (fixed N_g, clipped BB, as r11)
        r["oracle"] = {}
        for t in TESTS:
            o = oracle_forecast(risk, f, rho_I, al, t)
            o.pop("pmf")
            r["oracle"][t] = o

        # 2. fit check at jt, clipped BB(30, p_frame, rho_I) at N_g
        pmf, _ = F.clipped_pmf(M, p_frame, max(rho_I, 0.0), n_g)
        exp2 = T * pmf
        kmax_obs = int(Wg.max())
        obs_hist2 = np.bincount(np.minimum(Wg, n_g), minlength=n_g + 1)        # observed above N_g go to the top bin
        r["fit_clipped"] = {"categories": categories(Wg, np.r_[exp2, np.zeros(max(0, kmax_obs - n_g))], n_g),
                            "chi_square": chi_square(obs_hist2, exp2, 2), "k_max_observed": kmax_obs}

        # 3 (i) errors given the real acceptance: BB(A_g, risk, rho), two rho readings
        r["fit_conditional"] = {}
        for lab, rho in (("rho_I (all-frame ICC, as in the model)", rho_I), ("rho among accepted frames", rho_acc)):
            exp3 = mixture_expected(Ag, risk, rho, int(Ag.max()))
            obs3 = np.bincount(Wg, minlength=len(exp3))
            r["fit_conditional"][lab] = {"rho": rho, "categories": categories(Wg, exp3, n_g),
                                         "chi_square": chi_square(obs3, exp3, 2)}
        r["oracle_real_acceptance"] = {}
        for lab, rho in (("rho_I", rho_I), ("rho_accepted", rho_acc)):
            r["oracle_real_acceptance"][lab] = {t: mc_forecast_real_acceptance(Ag, risk, rho, al, t, rng) for t in TESTS}

        # 3 (ii) acceptance against the fixed N_g
        r["acceptance"] = {"mean": float(Ag.mean()), "sd": float(Ag.std(ddof=1)),
                           "quantiles_10_25_50_75_90": np.quantile(Ag, [0.1, 0.25, 0.5, 0.75, 0.9]).tolist(),
                           "share_A_eq_0": float(np.mean(Ag == 0)), "share_A_lt_N_g_half": float(np.mean(Ag < n_g / 2)),
                           "share_A_lt_N_g": float(np.mean(Ag < n_g)), "share_A_eq_30": float(np.mean(Ag == M)),
                           "wrong_frames_in_tracks_with_A_lt_N_g": int(Wg[Ag < n_g].sum()), "wrong_frames_total": int(Wg.sum())}
        r["observed"] = {t: r12["rows"][f"{al}_{t}"]["observed"] for t in TESTS}
        r["pilot_forecast"] = {t: r12["rows"][f"{al}_{t}"]["forecast"] for t in TESTS}
        res[al] = r

        out.append(f"--- alpha {100 * al:.1f}%: jt = {jt} (threshold {d['b1']['work_thresholds'][jt]:.2f}) ---")
        out.append(f"  Pool at jt: risk {pct(risk, 3)}, f {pct(f, 1)}, per-frame accepted-wrong {pct(p_frame, 3)}, "
                   f"rho_I {rho_I:.4f}, rho among accepted frames {rho_acc:.4f}, N_g {n_g}")
        out.append("  1. Forecasts and observed rates:")
        out.append(f"     {'test':>5} | {'observed [95% CP]':>26} | {'pilot (r12)':>11} | {'oracle fixed N_g':>16} | "
                   f"{'oracle real acc. rho_I':>22} | {'oracle real acc. rho_acc':>24}")
        for t in TESTS:
            ob = r["observed"][t]
            out.append(f"     {t:>5} | {pct(ob['violation_rate'])} [{100 * ob['cp95'][0]:5.2f}, {100 * ob['cp95'][1]:5.2f}]% | "
                       f"{pct(r['pilot_forecast'][t]):>11} | {pct(r['oracle'][t]['forecast']):>16} | "
                       f"{pct(r['oracle_real_acceptance']['rho_I'][t]):>22} | {pct(r['oracle_real_acceptance']['rho_accepted'][t]):>24}")
        out.append(f"     (oracle fixed N_g: k_max hb {r['oracle']['hb']['k_max']}, binom {r['oracle']['binom']['k_max']}; "
                   f"clipped mass per track {r['oracle']['hb']['clipped']:.2e})")
        out.append(f"  2. Fit at jt: per-track accepted-and-wrong count k against clipped BB(30, {pct(p_frame, 3).strip()}, "
                   f"rho_I {rho_I:.4f}) at N_g {n_g} (largest observed k: {kmax_obs})")
        for name, (ob, ex) in r["fit_clipped"]["categories"].items():
            out.append(f"     {name:>62}: observed {ob:5d}, expected {ex:9.2f}")
        cs = r["fit_clipped"]["chi_square"]
        out.append(f"     chi-square {cs['stat']:.2f}, df {cs['df']}, p {fmt_p(cs['p'])}; "
                   f"bins (lower edges, last open) {cs['bins_lower_edges']}; obs {[int(v) for v in cs['obs']]}; "
                   f"exp {[round(v, 1) for v in cs['exp']]}")
        out.append("  3(i). Errors given the real acceptance: k against BB(A_g, pool risk, rho), summed over tracks")
        for lab, x in r["fit_conditional"].items():
            out.append(f"     {lab} = {x['rho']:.4f}:")
            for name, (ob, ex) in x["categories"].items():
                out.append(f"       {name:>62}: observed {ob:5d}, expected {ex:9.2f}")
            cs = x["chi_square"]
            out.append(f"       chi-square {cs['stat']:.2f}, df {cs['df']}, p {fmt_p(cs['p'])}; "
                       f"bins {cs['bins_lower_edges']}; obs {[int(v) for v in cs['obs']]}; exp {[round(v, 1) for v in cs['exp']]}")
        ac = r["acceptance"]
        q = ac["quantiles_10_25_50_75_90"]
        out.append(f"  3(ii). Acceptance at jt: A_g mean {ac['mean']:.2f}, sd {ac['sd']:.2f}, quantiles 10/25/50/75/90 "
                   f"{'/'.join(f'{v:g}' for v in q)} against the fixed N_g {n_g}; A_g = 0 in {pct(ac['share_A_eq_0'], 1)}, "
                   f"< N_g/2 in {pct(ac['share_A_lt_N_g_half'], 1)}, < N_g in {pct(ac['share_A_lt_N_g'], 1)}, = 30 in "
                   f"{pct(ac['share_A_eq_30'], 1)}; wrong frames in tracks with A_g < N_g: "
                   f"{ac['wrong_frames_in_tracks_with_A_lt_N_g']} of {ac['wrong_frames_total']}")
        out.append("")
    out.append("Notes: chi-square bins merged from the top (and bottom if needed) until every expected count >= 5; df = bins "
               "- 1 - 2 (rate and rho taken from the same pool tracks; rho by ANOVA, not maximum likelihood, so the df is "
               "approximate). The model gives every track 30 frames; the pool has one 29-frame track. Item 3 Monte Carlo "
               "uses HB over 784 x 30 frames, as r11. This is a POST HOC DIAGNOSTIC: no criterion, no revised method.")
    print("\n".join(out), flush=True)

    od = Path(a.out)
    od.mkdir(parents=True, exist_ok=True)
    summary = {"label": LABEL, "frozen_blobs": frozen,
               "code_sha256": {"r12b_diagnostic.py": sha256_file(Path(__file__)),
                               "r11_forecast_ucb.py": sha256_file(Path(F.__file__)),
                               "r12_gtsrb_forecast.py": sha256_file(Path(G.__file__)),
                               "r9_certify.py": sha256_file(Path(C.__file__))},
               "inputs_sha256": {**d["inputs_sha256"], "r12_summary.json": R12_SHA256,
                                 "stage3b_summary.json": G.STAGE3B_SHA256},
               "config": {"alphas": ALPHAS, "tests": TESTS, "m": M, "n_cert": N_CERT, "mc_key": list(MC_KEY),
                          "mc_n": MC_N, "min_expected": MIN_EXPECTED},
               "pool_checks": checks, "results": {str(k): v for k, v in res.items()},
               "seconds": round(time.time() - t0, 1)}
    path = od / "r12b_summary.json"
    path.write_text(json.dumps(S2.jsonable(summary), indent=2))
    (od / "r12b_report.md").write_text("\n".join(out) + "\n")
    print(f"\nOutput: {path} SHA-256 {sha256_file(path)} and r12b_report.md ({summary['seconds']} s). POST HOC DIAGNOSTIC.")
    return summary


if __name__ == "__main__":
    main()
