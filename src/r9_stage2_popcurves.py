"""POC-3 (R9) Stage 2, design values: population curves of the synthetic commit model. No Cert run, no V-check.

LOG_r9_stage2_entry.md Part A (b), (c), (e), (f). Synthetic data only; no dataset file is opened.
Model: POC-2 Stage 3a (src/r8_stage3_synth.py), x_it = a_i + e_it, a_i ~ N(m, r), e_it ~ N(0, 1 - r), wrong iff x < 0,
extended to T_i views per unit. Commit at the first t <= T_i with |S_t| >= lambda (r8 track_table, reused as is: views
past T_i are zero, so S and its running max stay flat and nothing new can commit there).
Per (rho, cluster) pair: a 30-point grid from a 20,000-unit pilot (stream 0), then commit fraction and true risk
R = P(wrong | committed) per grid point from a 1,000,000-unit population (stream 1).
Step 2 of the OPEN-1 decision: per setting (pair x alpha), do the start-eligible grid points (population commit
fraction >= 4 / (n * alpha)) include R above and below alpha?

Usage:  py -3.12 -m src.r9_stage2_popcurves [--p-frame 0.074 --level 0]
Output: runs/20261003_stage1/poc3/stage2/population_curves_L<level>.json
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.optimize import brentq
from scipy.stats import multivariate_normal, norm

from src import r8_stage3_synth as R8S
from src.util import sha256_file

SEED = 20261114                          # base seed; key [SEED, 9, level, pair, stream] (9 = plan R9)
RHOS = (0.0, 0.3, 0.45)
CLUSTERS = ("fixed-3", "view-mix", "fixed-30")
MIX_T = np.array([1, 2, 3, 4])           # Stage 1 testing views per unit, before removal, duplicates included; 4+ as 4
MIX_COUNTS = np.array([14, 8, 816, 10])
N_CERT = {"fixed-3": 470, "view-mix": 470, "fixed-30": 784}
ALPHAS = (0.01, 0.02)
G, F_MAX, F_MIN = 30, 1.00, 0.30
N_PILOT, N_POP, CHUNK = 20_000, 1_000_000, 50_000
OUT = Path("runs/20261003_stage1/poc3/stage2")


# ------------------------------------------------------------------ model, parameterised by the frame error
def m_of(p):
    return float(-norm.ppf(p))


def icc_of_wrong(r, p):
    """Same formula as r8_stage3_synth.icc_of_wrong, with the frame error as an argument (needed for an added level)."""
    if r <= 0:
        return 0.0
    m = m_of(p)
    p11 = multivariate_normal(mean=[0, 0], cov=[[1, r], [r, 1]]).cdf([-m, -m])
    return float((p11 - p * p) / (p * (1 - p)))


def solve_r(rho, p):
    return 0.0 if rho <= 0 else float(brentq(lambda r: icc_of_wrong(r, p) - rho, 1e-6, 0.999, xtol=1e-10))


def tmax(cluster):
    return {"fixed-3": 3, "view-mix": 4, "fixed-30": 30}[cluster]


def draw(n, cluster, r, m, rng):
    """X (n, Tmax) with zeros past each unit's T_i, and T (n,). For fixed T the draw order equals r8 sample()."""
    if cluster == "view-mix":
        T = rng.choice(MIX_T, size=n, p=MIX_COUNTS / MIX_COUNTS.sum())
    else:
        T = np.full(n, tmax(cluster))
    a = rng.normal(m, np.sqrt(r), n) if r > 0 else np.full(n, m)
    X = a[:, None] + rng.normal(0.0, np.sqrt(1.0 - r), (n, tmax(cluster)))
    mask = np.arange(tmax(cluster))[None, :] < T[:, None]
    return np.where(mask, X, 0.0), T


def commit_score(X):
    """C_i = max_t |S_it|; the unit commits at lambda iff C_i >= lambda."""
    return np.abs(np.cumsum(X, axis=1)).max(1)


def make_grid(C):
    """30 thresholds, loosest first: lambda_0 = 0 (all commit at view 1), then pilot commit fractions down to 30%."""
    f = F_MAX - (F_MAX - F_MIN) * np.arange(G) / (G - 1)
    lam = np.quantile(C, 1.0 - f)
    lam[0] = 0.0
    return lam, f


def rng_for(level, pair, stream):
    return np.random.default_rng([SEED, 9, level, pair, stream])


# ------------------------------------------------------------------ curves
def pair_curves(pair, rho, cluster, p, level, n_pop=N_POP):
    r, m = solve_r(rho, p), m_of(p)
    Xp, _ = draw(N_PILOT, cluster, r, m, rng_for(level, pair, 0))
    grid, f_target = make_grid(commit_score(Xp))
    rng = rng_for(level, pair, 1)
    com = wr = np.zeros(G)
    nv = nw = nw2 = npairs = 0.0
    done = 0
    while done < n_pop:
        b = min(CHUNK, n_pop - done)
        X, T = draw(b, cluster, r, m, rng)
        c, w = R8S.track_table(X, grid)
        com, wr = com + c.sum(0), wr + w.sum(0)
        mask = np.arange(X.shape[1])[None, :] < T[:, None]
        k = ((X < 0) & mask).sum(1).astype(float)
        nv, nw = nv + T.sum(), nw + k.sum()
        nw2, npairs = nw2 + (k * (k - 1)).sum(), npairs + (T * (T - 1)).sum()
        done += b
    pf = nw / nv
    p11 = nw2 / npairs
    return {"pair": pair, "rho": rho, "cluster": cluster, "r_latent": r, "m": m, "grid": grid,
            "pilot_commit_fraction": f_target, "pop_commit_fraction": com / n_pop,
            "true_risk": np.where(com > 0, wr / np.maximum(com, 1), np.nan),
            "calibration": {"view_error": float(pf), "wrong_icc": float((p11 - pf * pf) / (pf * (1 - pf))),
                            "mean_views": float(nv / n_pop)}}


def eligibility(cv, n, alpha):
    """Start-eligible grid points: population commit fraction * n * alpha >= 4 (plan 5.3, applied to the population)."""
    elig = np.flatnonzero(cv["pop_commit_fraction"] * n * alpha >= 4)
    R = cv["true_risk"][elig]
    return {"alpha": alpha, "n_cert": n, "floor": 4 / (n * alpha), "eligible_idx": elig.tolist(),
            "n_eligible": int(len(elig)), "n_above": int((R > alpha).sum()), "n_below": int((R < alpha).sum()),
            "min_R_eligible": float(R.min()) if len(R) else None,
            "nonvacuous": bool((R > alpha).any() and (R < alpha).any())}


def jsonable(x):
    if isinstance(x, dict):
        return {k: jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return [None if (isinstance(v, float) and np.isnan(v)) else v for v in x.tolist()]
    if isinstance(x, np.generic):
        return x.item()
    return x


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--p-frame", type=float, default=R8S.P_FRAME)
    ap.add_argument("--level", type=int, default=0)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args(argv)
    t0 = time.time()
    curves, settings = [], []
    pair = 0
    for rho in RHOS:
        for cl in CLUSTERS:
            cv = pair_curves(pair, rho, cl, a.p_frame, a.level)
            curves.append(cv)
            for al in ALPHAS:
                settings.append({"rho": rho, "cluster": cl, **eligibility(cv, N_CERT[cl], al)})
            pair += 1

    print(f"=== R9 Stage 2 design values: population curves (level {a.level}, P_FRAME {a.p_frame}, "
          f"pilot {N_PILOT:,}, population {N_POP:,}; no Cert run) ===")
    for cv in curves:
        cal = cv["calibration"]
        print(f"\nrho {cv['rho']:.2f} | {cv['cluster']:8} | r {cv['r_latent']:.4f} | view error {100 * cal['view_error']:.3f}% "
              f"| wrong ICC {cal['wrong_icc']:.3f} | mean views {cal['mean_views']:.3f}")
        print(f"  {'j':>2} {'lambda':>9} {'pilot f':>8} {'pop f':>8} {'R':>9}")
        for j in range(G):
            print(f"  {j:2d} {cv['grid'][j]:9.4f} {100 * cv['pilot_commit_fraction'][j]:7.2f}% "
                  f"{100 * cv['pop_commit_fraction'][j]:7.2f}% {100 * cv['true_risk'][j]:8.4f}%")
    print("\n=== Step 2: start-eligible thresholds (population commit fraction >= 4 / (n * alpha)) ===")
    print(f"{'rho':>5} {'cluster':8} {'alpha':>6} {'n':>4} {'floor':>7} {'elig':>5} {'R>a':>4} {'R<a':>4} {'min R elig':>11}  non-vacuous")
    for s in settings:
        mr = "n/a" if s["min_R_eligible"] is None else f"{100 * s['min_R_eligible']:.4f}%"
        print(f"{s['rho']:5.2f} {s['cluster']:8} {100 * s['alpha']:5.1f}% {s['n_cert']:4d} {100 * s['floor']:6.2f}% "
              f"{s['n_eligible']:5d} {s['n_above']:4d} {s['n_below']:4d} {mr:>11}  {'yes' if s['nonvacuous'] else 'NO'}")

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = {"plan": "poc3-implementation-plan-r9 Stage 2 (design values, population curves only)",
               "code_sha256": {"r9_stage2_popcurves.py": sha256_file(Path(__file__)),
                               "r8_stage3_synth.py": sha256_file(Path(R8S.__file__))},
               "config": {"level": a.level, "p_frame": a.p_frame, "seed": SEED, "rhos": RHOS, "clusters": CLUSTERS,
                          "mix_T": MIX_T, "mix_counts": MIX_COUNTS, "n_cert": N_CERT, "alphas": ALPHAS, "G": G,
                          "n_pilot": N_PILOT, "n_pop": N_POP},
               "curves": curves, "settings": settings, "seconds": round(time.time() - t0, 1)}
    path = out / f"population_curves_L{a.level}.json"
    path.write_text(json.dumps(jsonable(summary), indent=2))
    print(f"\nOutput: {path} ({summary['seconds']} s)")
    return summary


if __name__ == "__main__":
    main()
