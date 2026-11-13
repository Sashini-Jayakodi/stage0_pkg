"""POC-2 (R8) Stage 3a: synthetic validity test of the track-level commit-rule certifier against B1.
LOG_r8_stage3_entry.md Part A (e), (d), (g), (h). New file; src/certify_synth.py (POC-1) is not changed.

Model (Part A (e)): frame evidence x_it = a_i + e_it, a_i ~ N(m, r), e_it ~ N(0, 1 - r), 30 frames per track.
  frame wrong iff x_it < 0; m fixes the frame error at 7.4%; r is solved so the within-track correlation of the
  frame-wrong indicator equals rho_sim. A strongly negative a_i gives a confidently wrong whole track.
  Commit rule: S_t = x_i1 + ... + x_it, commit at the first t with |S_t| >= lambda, wrong iff S_t < 0 there.
  B1: frame margin |x_it|, accepted iff |x_it| >= threshold, wrong iff x_it < 0. No gate.
Per rho: fixed candidate grids from a pilot 261-track sample (Stage 1 make_grid rule); each run draws a fresh
261-track design sample (working grid and order by Stage 2 rules (h), (c) at the primary alpha) and a fresh
784-track Cert sample. True risk from a 200,000-track population.

Usage:  py -3.12 -m src.r8_stage3_synth --run runs/20261003_stage1
Output: runs/20261003_stage1/poc2/stage3/stage3a_summary.json
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.optimize import brentq
from scipy.stats import multivariate_normal, norm

from src import r8_stage1 as R1
from src import r8_stage2 as S2
from src import r8_stage3_engine as E
from src import r8_frozen as FZ
from src.r8_frozen import load_frozen
from src.util import sha256_file

FROZEN_ARGS = {}                       # tests point load_frozen at a fake root; production uses the F1 hashes
P_FRAME = 0.074                        # plan's single-frame error
FRAMES, N_DESIGN, N_CERT = 30, 261, 784
RUNS, N_POP = 1000, 200_000
RHOS = (0.0, 0.3, 0.45)
SEED = 20261114
CHUNK = 5000                           # population tracks per block (memory: CHUNK x 30 x 200 booleans)
M = float(-norm.ppf(P_FRAME))          # mean of x so that P(x < 0) = 7.4% (total variance 1)


# ------------------------------------------------------------------ calibration of r
def icc_of_wrong(r):
    """Within-track correlation of the frame-wrong indicator when corr(x_s, x_t) = r."""
    if r <= 0:
        return 0.0
    p = P_FRAME
    p11 = multivariate_normal(mean=[0, 0], cov=[[1, r], [r, 1]]).cdf([-M, -M])
    return float((p11 - p * p) / (p * (1 - p)))


def solve_r(rho):
    return 0.0 if rho <= 0 else float(brentq(lambda r: icc_of_wrong(r) - rho, 1e-6, 0.999, xtol=1e-10))


def sample(n, r, rng):
    a = rng.normal(M, np.sqrt(r), n) if r > 0 else np.full(n, M)
    return a[:, None] + rng.normal(0.0, np.sqrt(1.0 - r), (n, FRAMES))


# ------------------------------------------------------------------ outcomes per threshold
def track_table(X, grid):
    """committed, wrong (n, G) for the commit rule. The running max of |S| is non-decreasing, so the commit frame for
    lambda is the number of frames whose running max is still below lambda."""
    S = np.cumsum(X, axis=1)
    runmax = np.maximum.accumulate(np.abs(S), axis=1)
    idx = (runmax[:, :, None] < grid[None, None, :]).sum(1)
    committed = idx < X.shape[1]
    sgn = np.take_along_axis(S, np.minimum(idx, X.shape[1] - 1), axis=1)
    return committed, committed & (sgn < 0)


def frame_counts(X, grid):
    """Accepted and accepted-and-wrong frame counts per track (n, G) for B1."""
    acc = np.abs(X)[:, :, None] >= grid[None, None, :]
    return acc.sum(1), (acc & (X < 0)[:, :, None]).sum(1)


def candidate_grids(X):
    """Stage 1 make_grid rule on a pilot design sample: 0 plus 199 log-spaced points."""
    absS = np.abs(np.cumsum(X, axis=1))
    g_track = R1.make_grid(absS.max(1), absS[absS > 0])
    ax = np.abs(X).ravel()
    return g_track, R1.make_grid(ax, ax[ax > 0])


def design_order(committed, wrong, alpha):
    """Stage 2 rules (h) and (c) on a design table (units x thresholds). Returns the test order as grid columns, or None."""
    z = np.zeros(committed.shape)
    st = S2.grid_stats(committed, wrong, z, z)
    work = S2.working_grid(st, alpha)
    if not len(work):
        return None
    _, order = S2.start_and_order(S2.mean_L(st, alpha)[work])
    return [int(work[o]) for o in order]


def population_risk(r, g_track, g_b1, alpha, rng, n_pop=N_POP):
    """True mean loss per threshold for both methods, from n_pop tracks in blocks. Also the realised frame error and
    within-track wrong-indicator correlation, as a check on the calibration."""
    com = wr = A = W = 0.0
    nw = nw2 = 0.0
    done = 0
    while done < n_pop:
        b = min(CHUNK, n_pop - done)
        X = sample(b, r, rng)
        c, w = track_table(X, g_track)
        com, wr = com + c.sum(0), wr + w.sum(0)
        a_, w_ = frame_counts(X, g_b1)
        A, W = A + a_.sum(0), W + w_.sum(0)
        k = (X < 0).sum(1).astype(float)                              # wrong frames per track
        nw, nw2 = nw + k.sum(), nw2 + (k * (k - 1)).sum()
        done += b
    p = nw / (n_pop * FRAMES)
    p11 = nw2 / (n_pop * FRAMES * (FRAMES - 1))                       # P(two distinct frames of a track both wrong)
    mu_track = alpha * (1 - com / n_pop) + wr / n_pop
    Nf = n_pop * FRAMES
    mu_b1 = (W + alpha * (Nf - A)) / Nf
    return mu_track, mu_b1, {"frame_error": float(p), "wrong_icc": float((p11 - p * p) / (p * (1 - p)))}


def run_rho(rho, ri, alpha, delta, runs=RUNS, n_pop=N_POP):
    r = solve_r(rho)
    g_track, g_b1 = candidate_grids(sample(N_DESIGN, r, np.random.default_rng([SEED, ri, 0])))
    mu_t, mu_b, calib = population_risk(r, g_track, g_b1, alpha, np.random.default_rng([SEED, ri, 1]), n_pop)
    rng = np.random.default_rng([SEED, ri, 2])
    cert_t, cert_b = np.full(runs, -1), np.full(runs, -1)
    empty = {"track": 0, "B1": 0}
    for k in range(runs):
        Xd = sample(N_DESIGN, r, rng)
        o_t = design_order(*track_table(Xd, g_track), alpha)
        flat = Xd.reshape(-1, 1)
        acc = np.abs(flat) >= g_b1[None, :]
        o_b = design_order(acc, acc & (flat < 0), alpha)
        Xc = sample(N_CERT, r, rng)
        if o_t is None:
            empty["track"] += 1
        else:
            c, w = track_table(Xc, g_track)
            mu, n = E.track_mu(c, w, np.ones((1, N_CERT)), alpha)
            cert_t[k] = E.certify(mu, n, alpha, o_t, delta)[0]
        if o_b is None:
            empty["B1"] += 1
        else:
            Af, Wf = frame_counts(Xc, g_b1)
            mu, n = E.frame_mu(Af, Wf, np.full(N_CERT, FRAMES), np.ones((1, N_CERT)), alpha)
            cert_b[k] = E.certify(mu, n, alpha, o_b, delta)[0]
    return {"rho_sim": rho, "r_latent": r, "population": calib, "runs_with_empty_working_grid": empty,
            "track": E.rate_row(cert_t, mu_t, alpha), "B1": E.rate_row(cert_b, mu_b, alpha)}


def p1_parts_3a(rows, delta):
    by = {r["rho_sim"]: r for r in rows}
    track_ok = all(by[x]["track"]["cp95"][1] <= delta + 0.02 for x in RHOS)
    frame_ok = by[0.45]["B1"]["cp95"][0] > delta
    return {"track_part_3a": bool(track_ok), "frame_part_3a": bool(frame_ok)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    a = ap.parse_args(argv)
    cfg = load_frozen(**FROZEN_ARGS)
    alpha, delta = float(cfg["certificate_A"]["primary_alpha"]), float(cfg["fixed_in_every_phase"]["delta"])
    t0 = time.time()
    hand, cb = E.check_hand(), E.check_cert_b(alpha=alpha, delta=delta, gamma=float(cfg["certificate_B"]["gamma"]))
    rows = []
    for ri, rho in enumerate(RHOS):
        rows.append(run_rho(rho, ri, alpha, delta))
        print(f"  rho_sim {rho}: done ({time.time() - t0:.0f} s)", flush=True)
    p1 = p1_parts_3a(rows, delta)

    print(f"\n=== R8 Stage 3a: synthetic validity (alpha {100 * alpha:.1f}%, delta {delta}, {RUNS} runs, "
          f"{N_CERT} Cert tracks x {FRAMES} frames, population {N_POP:,}) ===")
    print(f"{'rho':>5} {'r':>6} {'fr.err':>7} {'icc':>6} | {'method':6} {'cert':>6} {'viol':>5} {'rate':>7} {'CP95':>17} {'mean R*':>8}")
    for r in rows:
        for m in ("track", "B1"):
            x = r[m]
            print(f"{r['rho_sim']:5.2f} {r['r_latent']:6.3f} {100 * r['population']['frame_error']:6.2f}% "
                  f"{r['population']['wrong_icc']:6.3f} | {m:6} {100 * x['certified_share']:5.1f}% {x['violations']:5d} "
                  f"{100 * x['violation_rate']:6.2f}% [{100 * x['cp95'][0]:6.2f}, {100 * x['cp95'][1]:6.2f}]% "
                  + ("    n/a" if x["mean_true_risk_certified"] is None else f"{100 * x['mean_true_risk_certified']:7.3f}%"))
        if any(r["runs_with_empty_working_grid"].values()):
            print(f"      runs with empty working grid: {r['runs_with_empty_working_grid']}")
    print("(mean R* = mean true loss of the certified threshold over certifying runs; violation if R* > alpha)")
    print(f"\nP1 parts from 3a: track part (CP upper <= {delta + 0.02:.2f} at rho 0, 0.3, 0.45): "
          f"{'PASS' if p1['track_part_3a'] else 'FAIL'}; frame part (B1 CP lower > {delta} at rho 0.45): "
          f"{'PASS' if p1['frame_part_3a'] else 'FAIL'}")
    print(f"\nHand p-values (exact arithmetic): {'PASS' if hand['pass'] else 'FAIL'}")
    for c in hand["cases"]:
        print(f"  mu {c['mu_hat']}, n {c['n']}, alpha {c['alpha']}: exact {c['exact']:.12g}, engine {c['engine']:.12g}, "
              f"rel.err {c['rel_err']:.1e}, min term {c['min_term']}")
    print(f"Certificate B max-p (lateness broken only): {'PASS' if cb['pass'] else 'FAIL'}  p_error {cb['p_error']:.3g}, "
          f"p(lateness broken) {cb['p_max_lateness_broken']:.3g}, p(both kept) {cb['p_max_both_kept']:.3g}")

    out = Path(a.run) / "poc2" / "stage3"
    out.mkdir(parents=True, exist_ok=True)
    summary = {"plan": "poc2-implementation-plan-r8 Stage 3a",
               "frozen_r8_sha256": sha256_file(Path(FROZEN_ARGS.get("root", FZ.ROOT)) / FZ.FROZEN),
               "code_sha256": {"r8_stage3_synth.py": sha256_file(Path(__file__)),
                               "r8_stage3_engine.py": sha256_file(Path(E.__file__))},
               "config": {"alpha": alpha, "delta": delta, "p_frame": P_FRAME, "frames": FRAMES, "n_design": N_DESIGN,
                          "n_cert": N_CERT, "runs": RUNS, "n_pop": N_POP, "rhos": RHOS, "seed": SEED},
               "rows": rows, "p1_parts": p1, "hand_pvalues": hand, "cert_b_maxp": cb,
               "seconds": round(time.time() - t0, 1)}
    (out / "stage3a_summary.json").write_text(json.dumps(S2.jsonable(summary), indent=2))
    print(f"\nOutput: {out / 'stage3a_summary.json'} ({summary['seconds']} s)")
    return summary


if __name__ == "__main__":
    main()
