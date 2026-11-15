"""POC-3 (R9) Stage 10, step 2: population curves and the boundary diagnostic. Synthetic data only.

LOG_r9_stage10_entry.md Part A (b), (c), (h). Draws the pilot sample (stream 0) and the population (stream 1) for the
9 (rho, m) pairs; no Cert sample is drawn here (stream 2 is used only by src/r9_stage10.py).
  - predicted delta_eff for every setting, checked against the table committed in Part A (c);
  - sign-level and B1 grids (Stage 2 rule) and their population curves;
  - the boundary diagnostic: true risk of the first start-eligible B1 grid point above alpha, in walk order.

Usage:  py -3.12 -m src.r9_stage10_popcurves
Output: runs/20261003_stage1/poc3/stage10/population_curves_stage10.json
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
from scipy.stats import norm

from src import r8_stage3_synth as R8S
from src import r9_certify as C
from src import r9_stage2_popcurves as P
from src import r9_stage2_vchecks as V
from src.r9_frozen import check_frozen
from src.util import sha256_file

SEED = 20261115                    # key [SEED, 10, pair, stream]; Stage 2 used [20261114, 9, level, pair, stream]
RHOS = (0.1, 0.2, 0.6)
MS = (5, 10, 15)
PAIRS = [(rho, m) for rho in RHOS for m in MS]     # pair index = position in this list (Part A (b))
ALPHAS = (0.01, 0.02)
P_FRAME = R8S.P_FRAME              # 7.4%, Stage 2 level 0
N_CERT, N_DESIGN = 784, 261        # Part A (b), Q1: the GTSRB sizes for every m
BOUNDARY_FACTOR = 2.0              # Part A (h): boundary subset when R_b <= 2 x alpha
OUT = Path("runs/20261003_stage1/poc3/stage10")
CURVES_NAME = "population_curves_stage10.json"

# Part A (c), committed before any run (percent, 2 decimals as printed there); recomputed values must agree to 0.005 pp
PREDICTED_PCT = {(0.1, 5): 13.94, (0.1, 10): 17.63, (0.1, 15): 20.41,
                 (0.2, 5): 16.97, (0.2, 10): 22.19, (0.2, 15): 25.55,
                 (0.6, 5): 24.35, (0.6, 10): 30.62, (0.6, 15): 33.80}
PREDICTED_TOL_PCT = 0.005          # Part A (c); the table's own rounding to 0.01 pp stays inside this


def design_effect(rho, m):
    return 1.0 + (m - 1.0) * rho


def delta_eff(rho, m, delta=C.DELTA):
    """Part A (a): Phi(Phi^-1(delta) / sqrt(DE))."""
    return float(norm.cdf(norm.ppf(delta) / math.sqrt(design_effect(rho, m))))


def check_predictions():
    """Stop if the recomputed delta_eff differs from the committed table (c)."""
    rows = []
    for (rho, m), want in PREDICTED_PCT.items():
        got = 100 * delta_eff(rho, m)
        ok = abs(got - want) <= PREDICTED_TOL_PCT
        rows.append({"rho": rho, "m": m, "DE": design_effect(rho, m), "delta_eff": got / 100, "table_pct": want,
                     "ok": ok})
        if not ok:
            raise SystemExit(f"STOP: delta_eff({rho}, {m}) = {got:.4f}% differs from the Part A table {want}%")
    return rows


def rng_for(pair, stream):
    return np.random.default_rng([SEED, 10, pair, stream])


def draw_fixed(n, T, r, mu, rng):
    """Fixed-T version of r9_stage2_popcurves.draw, same draw order (a_i first, then the n x T noise)."""
    a = rng.normal(mu, np.sqrt(r), n) if r > 0 else np.full(n, mu)
    return a[:, None] + rng.normal(0.0, np.sqrt(1.0 - r), (n, T))


def grids(pair, rho, m):
    """Pilot (stream 0) sign-level and B1 grids by the Stage 2 rule."""
    r, mu = P.solve_r(rho, P_FRAME), P.m_of(P_FRAME)
    Xp = draw_fixed(P.N_PILOT, m, r, mu, rng_for(pair, 0))
    sgrid, _ = P.make_grid(P.commit_score(Xp))
    bgrid = V.b1_grid(Xp, np.full(P.N_PILOT, m))
    return r, mu, sgrid, bgrid


def pair_curves(pair, rho, m, n_pop=P.N_POP):
    """Population curves (stream 1) for one (rho, m), chunked as Stage 2."""
    r, mu, sgrid, bgrid = grids(pair, rho, m)
    rng = rng_for(pair, 1)
    com = wr = A = W = np.zeros(P.G)
    nf = nw = nw2 = npairs = 0.0
    done = 0
    while done < n_pop:
        b = min(P.CHUNK, n_pop - done)
        X = draw_fixed(b, m, r, mu, rng)
        c, w = R8S.track_table(X, sgrid)
        com, wr = com + c.sum(0), wr + w.sum(0)
        a_, w_, n_ = V.frame_totals(X, np.full(b, m), bgrid)
        A, W, nf = A + a_, W + w_, nf + n_
        k = (X < 0).sum(1).astype(float)
        nw, nw2, npairs = nw + k.sum(), nw2 + (k * (k - 1)).sum(), npairs + b * m * (m - 1)
        done += b
    pf, p11 = nw / nf, nw2 / npairs
    return {"pair": pair, "rho": rho, "m": m, "r_latent": r, "mu": mu, "sign_grid": sgrid, "b1_grid": bgrid,
            "sign_pop_commit": com / n_pop, "sign_true_risk": np.where(com > 0, wr / np.maximum(com, 1), np.nan),
            "b1_pop_accept": A / nf, "b1_true_risk": np.where(A > 0, W / np.maximum(A, 1), np.nan),
            "calibration": {"view_error": float(pf), "wrong_icc": float((p11 - pf * pf) / (pf * (1 - pf)))}}


def boundary(accept, risk, n_frames, alpha, factor=BOUNDARY_FACTOR):
    """Part A (h): binomial start from population fractions; first grid point with R > alpha in walk order."""
    js = C.power_start(accept, n_frames, alpha)
    jb = Rb = None
    for j in range(js, -1, -1):
        if risk[j] > alpha:
            jb, Rb = j, float(risk[j])
            break
    return {"alpha": alpha, "n_frames": n_frames, "start": js, "first_above_idx": jb, "R_b": Rb,
            "boundary": Rb is not None and Rb <= factor * alpha}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-pop", type=int, default=P.N_POP)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args(argv)
    t0 = time.time()
    frozen = check_frozen()
    preds = check_predictions()
    curves, settings = [], []
    for pair, (rho, m) in enumerate(PAIRS):
        cv = pair_curves(pair, rho, m, a.n_pop)
        curves.append(cv)
        for al in ALPHAS:
            b = boundary(cv["b1_pop_accept"], cv["b1_true_risk"], N_CERT * m, al)
            settings.append({"pair": pair, "rho": rho, "m": m, "delta_eff": delta_eff(rho, m), **b})
        print(f"  pair {pair} rho {rho:.1f} m {m:2d}: done ({time.time() - t0:.0f} s)", flush=True)

    print(f"\n=== R9 Stage 10 population curves (synthetic; P_FRAME {P_FRAME}; population {a.n_pop:,}; "
          f"no Cert sample) ===")
    print("\nCalibration (population):")
    for cv in curves:
        print(f"  pair {cv['pair']} rho {cv['rho']:.1f} m {cv['m']:2d}: r_latent {cv['r_latent']:.4f}, view error "
              f"{100 * cv['calibration']['view_error']:.3f}%, wrong ICC {cv['calibration']['wrong_icc']:.4f}")
    print("\nB1 population curves (j: accept fraction / true risk), loosest first:")
    for cv in curves:
        print(f"  pair {cv['pair']} rho {cv['rho']:.1f} m {cv['m']:2d}: " + " ".join(
            f"{j}:{100 * f:.0f}/{100 * R:.2f}" for j, (f, R) in enumerate(zip(cv["b1_pop_accept"], cv["b1_true_risk"]))))
    print(f"\nBoundary diagnostic (Part A (h); boundary when R_b <= {BOUNDARY_FACTOR:g} x alpha):")
    print(f"  {'pair':>4} {'rho':>4} {'m':>3} {'alpha':>5} | {'delta_eff':>9} {'start':>5} {'j_b':>4} {'R_b':>8} | boundary")
    for s in settings:
        rb = "     n/a" if s["R_b"] is None else f"{100 * s['R_b']:7.3f}%"
        jb = " n/a" if s["first_above_idx"] is None else f"{s['first_above_idx']:4d}"
        print(f"  {s['pair']:4d} {s['rho']:4.1f} {s['m']:3d} {100 * s['alpha']:4.1f}% | {100 * s['delta_eff']:8.2f}% "
              f"{s['start']:5d} {jb} {rb} | {'yes' if s['boundary'] else 'no'}")
    subset = [(s["pair"], s["alpha"]) for s in settings if s["boundary"]]
    print(f"\nBoundary subset: {len(subset)} of {len(settings)} settings")

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = {"plan": "poc3-implementation-plan-r9, Stage 10 (declared addition), step 2",
               "frozen_blobs": frozen,
               "code_sha256": {"r9_stage10_popcurves.py": sha256_file(Path(__file__)),
                               "r9_certify.py": sha256_file(Path(C.__file__)),
                               "r9_stage2_popcurves.py": sha256_file(Path(P.__file__)),
                               "r9_stage2_vchecks.py": sha256_file(Path(V.__file__)),
                               "r8_stage3_synth.py": sha256_file(Path(R8S.__file__))},
               "config": {"delta": C.DELTA, "p_frame": P_FRAME, "pairs": PAIRS, "alphas": ALPHAS, "n_cert": N_CERT,
                          "n_design": N_DESIGN, "n_pilot": P.N_PILOT, "n_pop": a.n_pop,
                          "boundary_factor": BOUNDARY_FACTOR, "seed_key": "[20261115, 10, pair, stream]"},
               "predictions": preds, "curves": curves, "settings": settings, "boundary_subset": subset}
    path = out / CURVES_NAME
    path.write_text(json.dumps(P.jsonable(summary), indent=2))
    print(f"\nOutput: {path} SHA-256 {sha256_file(path)} ({time.time() - t0:.1f} s)")
    return summary


if __name__ == "__main__":
    main()
