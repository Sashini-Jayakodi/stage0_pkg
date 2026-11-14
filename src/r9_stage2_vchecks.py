"""POC-3 (R9) Stage 2: V-checks V1 to V4 of the certification engine v2 (src/r9_certify.py). Synthetic data only.

LOG_r9_stage2_entry.md Part A (d) to (i).
  V1: plan section 6 table (largest tolerated k per n_c and alpha; smallest zero-error n_c) reproduced exactly.
  V2: 36 settings (2 error levels x 3 rho x 3 cluster settings x 2 alpha), 2,000 runs each; sign-level exact-binomial
      violation rate, upper 95% Clopper-Pearson bound <= delta + 0.02.
  V3: engine p-values against exact rational arithmetic, 1e-9 relative.
  V4: a threshold with 0 committed units is never certified (direct, inside a walk, and in every V2 run).
Reported, not scored: Hoeffding-Bentkus and frame-level B1 (same engine, unit = view/frame) in the same runs.

Usage:  py -3.12 -m src.r9_stage2_vchecks
Output: runs/20261003_stage1/poc3/stage2/stage2_summary.json
"""
import argparse
import json
import math
import time
from fractions import Fraction
from pathlib import Path

import numpy as np

from src import r8_stage3_engine as E
from src import r8_stage3_synth as R8S
from src import r9_certify as C
from src import r9_stage2_popcurves as P
from src.util import sha256_file

LEVELS = {0: R8S.P_FRAME, 1: 0.05}     # main level (POC-2 Stage 3a) and the declared added level (Part A (f))
CURVES_SHA = {0: "fd7fdb44acbf6b0809258969996a5a919e185a503d40a222236a3a9d3881b019",
              1: "5cd5c3aa553f48fe9bd82e2b29066f29a59f9e0591d0143af35af0ee17986260"}
N_DESIGN = {"fixed-3": 157, "view-mix": 157, "fixed-30": 261}
RUNS = 2000
V2_MAX = C.DELTA + 0.02
TESTS = ("binom", "hb")
UNITS = ("sign", "B1")

# ------------------------------------------------------------------ V1: plan section 6 (None = "none")
PLAN_ALPHAS = (0.005, 0.01, 0.02, 0.03, 0.05)
PLAN_TABLE = {784: (1, 3, 10, 17, 30), 700: (0, 3, 8, 14, 27), 560: (0, 2, 6, 11, 20), 470: (0, 1, 5, 8, 17),
              420: (None, 1, 4, 7, 14), 300: (None, 0, 2, 4, 9), 250: (None, 0, 1, 3, 7)}
PLAN_MIN_N = (460, 230, 114, 76, 45)


def v1():
    rows, ok = [], True
    for n_c, plan in PLAN_TABLE.items():
        got = tuple(C.max_errors(n_c, a) for a in PLAN_ALPHAS)
        rows.append({"n_c": n_c, "plan": plan, "engine": got, "match": got == plan})
        ok &= got == plan
    mins = tuple(C.min_n_zero_errors(a) for a in PLAN_ALPHAS)
    ok &= mins == PLAN_MIN_N
    return {"pass": bool(ok), "rows": rows, "min_n_plan": PLAN_MIN_N, "min_n_engine": mins,
            "n_values": 5 * len(PLAN_TABLE) + len(PLAN_MIN_N),
            "n_match": sum(sum(g == p for g, p in zip(r["engine"], r["plan"])) for r in rows)
            + sum(g == p for g, p in zip(mins, PLAN_MIN_N))}


# ------------------------------------------------------------------ V3: exact arithmetic
V3_CASES = [(230, 0, Fraction(1, 100)), (229, 0, Fraction(1, 100)), (114, 0, Fraction(2, 100)),
            (113, 0, Fraction(2, 100)), (45, 0, Fraction(5, 100)), (44, 0, Fraction(5, 100)),
            (460, 0, Fraction(5, 1000)), (459, 0, Fraction(5, 1000)), (784, 3, Fraction(1, 100)),
            (784, 4, Fraction(1, 100)), (300, 2, Fraction(2, 100)), (300, 3, Fraction(2, 100)),
            (1, 1, Fraction(1, 100))]


def exact_binom(n, k, a):
    """P(Binomial(n, a) <= k) as an exact fraction (no scipy)."""
    return sum(Fraction(math.comb(n, i)) * a ** i * (1 - a) ** (n - i) for i in range(k + 1))


def v3():
    rows = []
    for n, k, a in V3_CASES:
        ref = exact_binom(n, k, a)
        got = float(C.binom_pvalue(k, n, float(a)))
        rel = abs(Fraction(got) - ref) / ref
        rows.append({"n_c": n, "k": k, "alpha": float(a), "exact": float(ref), "engine": got, "rel_err": float(rel),
                     "certified": got <= C.DELTA})
    return {"pass": bool(all(r["rel_err"] < 1e-9 for r in rows)), "cases": rows}


# ------------------------------------------------------------------ V4: n_c = 0 never certified (static parts)
def v4_static():
    direct = []
    for n in (1, 45, 470, 784, 23_520):
        for a in (0.005, 0.01, 0.02, 0.05):
            for t in TESTS:
                p = float(C.pvalues(0, 0, n, a, t))
                direct.append({"n": n, "alpha": a, "test": t, "p": p, "ok": p == 1.0})
    # Walk: design commits everything (start = strictest), Cert commits nothing at the two strictest thresholds.
    # The looser thresholds would pass on their own (control), so a -1 shows the n_c = 0 points stopped the walk.
    n, a = 470, 0.02
    n_c, k = np.array([470, 460, 450, 440, 0, 0]), np.zeros(6)       # high enough that HB also passes the control
    walk = {}
    for t in TESTS:
        full = C.certify(k, n_c, n, a, np.ones(6), t)
        control = C.walk(C.pvalues(k, n_c, n, a, t), 3)
        walk[t] = {"certified_from_strictest": full, "control_from_index_3": control, "ok": full == -1 and control == 0}
    ok = all(d["ok"] for d in direct) and all(w["ok"] for w in walk.values())
    return {"pass": bool(ok), "direct_cases": len(direct), "direct_all_p_equal_1": all(d["ok"] for d in direct),
            "walk": walk}


# ------------------------------------------------------------------ simulation pieces
def b1_grid(X, T):
    """30 thresholds on the per-view margin |x| over valid views: accept fractions 100% to 30%, lambda_0 = 0."""
    mask = np.arange(X.shape[1])[None, :] < T[:, None]
    lam, f = P.make_grid(np.abs(X[mask]))
    return lam


def frame_totals(X, T, grid):
    """Accepted and accepted-and-wrong views per threshold over valid views only (padded views are not frames)."""
    mask = np.arange(X.shape[1])[None, :] < T[:, None]
    acc = (np.abs(X)[:, :, None] >= grid[None, None, :]) & mask[:, :, None]
    return acc.sum((0, 1)), (acc & (X < 0)[:, :, None]).sum((0, 1)), int(T.sum())


def load_curves(level, curves_dir):
    """Committed design values for this level; refused if the file's SHA-256 differs from the recorded one."""
    path = Path(curves_dir) / f"population_curves_L{level}.json"
    sha = sha256_file(path)
    if sha != CURVES_SHA[level]:
        raise SystemExit(f"STOP: {path} SHA-256 {sha} != recorded {CURVES_SHA[level]}")
    return json.loads(path.read_text()), sha


def population(level, pair, cluster, r, m, sgrid, bgrid, n_pop):
    """Same draws as r9_stage2_popcurves.pair_curves (stream 1, same chunks), plus the B1 counts."""
    rng = P.rng_for(level, pair, 1)
    com = wr = A = W = np.zeros(P.G)
    nf = 0
    done = 0
    while done < n_pop:
        b = min(P.CHUNK, n_pop - done)
        X, T = P.draw(b, cluster, r, m, rng)
        c, w = R8S.track_table(X, sgrid)
        com, wr = com + c.sum(0), wr + w.sum(0)
        a_, w_, n_ = frame_totals(X, T, bgrid)
        A, W, nf = A + a_, W + w_, nf + n_
        done += b
    return {"sign_c": com / n_pop, "sign_w": wr / n_pop, "sign_R": np.where(com > 0, wr / np.maximum(com, 1), np.nan),
            "b1_c": A / nf, "b1_w": W / nf, "b1_R": np.where(A > 0, W / np.maximum(A, 1), np.nan)}


def run_pair(level, pair, rho, cluster, p_frame, runs, n_pop, curve):
    r, m = P.solve_r(rho, p_frame), P.m_of(p_frame)
    Xp, Tp = P.draw(P.N_PILOT, cluster, r, m, P.rng_for(level, pair, 0))
    sgrid, _ = P.make_grid(P.commit_score(Xp))
    bgrid = b1_grid(Xp, Tp)
    grid_match = bool(np.array_equal(sgrid, np.asarray(curve["grid"], float)))
    pop = population(level, pair, cluster, r, m, sgrid, bgrid, n_pop)
    curve_R = np.array([np.nan if v is None else v for v in curve["true_risk"]], float)
    risk_diff = float(np.nanmax(np.abs(pop["sign_R"] - curve_R))) if n_pop == P.N_POP else None

    nd, nc = N_DESIGN[cluster], P.N_CERT[cluster]
    cert = {(a, u, t): np.full(runs, -1) for a in P.ALPHAS for u in UNITS for t in TESTS}
    no_start = {(a, u, t): 0 for a in P.ALPHAS for u in UNITS for t in TESTS}
    zero_nc_certified = 0
    rng = P.rng_for(level, pair, 2)
    for i in range(runs):
        Xd, Td = P.draw(nd, cluster, r, m, rng)
        Xc, Tc = P.draw(nc, cluster, r, m, rng)
        fd = R8S.track_table(Xd, sgrid)[0].mean(0)
        c, w = R8S.track_table(Xc, sgrid)
        n_c, k = c.sum(0), w.sum(0)
        Ad, _, Nd = frame_totals(Xd, Td, bgrid)
        Ac, Wc, Nc = frame_totals(Xc, Tc, bgrid)
        for a in P.ALPHAS:
            for t in TESTS:                                          # each test has its own start (Amendment 1)
                no_start[(a, "sign", t)] += C.start_for(t, fd, nc, a) < 0
                no_start[(a, "B1", t)] += C.start_for(t, Ad / Nd, Nc, a) < 0
                j = C.certify(k, n_c, nc, a, fd, t)
                jb = C.certify(Wc, Ac, Nc, a, Ad / Nd, t)
                cert[(a, "sign", t)][i], cert[(a, "B1", t)][i] = j, jb
                zero_nc_certified += (j >= 0 and n_c[j] == 0) + (jb >= 0 and Ac[jb] == 0)

    settings = []
    for a in P.ALPHAS:
        mu_sign = a * (1 - pop["sign_c"]) + pop["sign_w"]          # POC-2 mean-loss form; > a  <=>  R > a
        mu_b1 = a * (1 - pop["b1_c"]) + pop["b1_w"]
        rows = {}
        for u, mu in (("sign", mu_sign), ("B1", mu_b1)):
            for t in TESTS:
                cv = cert[(a, u, t)]
                row = E.rate_row(cv, mu, a)
                row["median_certified_index"] = float(np.median(cv[cv >= 0])) if (cv >= 0).any() else None
                row["runs_no_eligible_start"] = int(no_start[(a, u, t)])
                rows[f"{u}_{t}"] = row
        settings.append({"level": level, "p_frame": p_frame, "rho": rho, "cluster": cluster, "alpha": a,
                         "n_cert": nc, "n_design": nd, "rows": rows,
                         "v2_pass": rows["sign_binom"]["cp95"][1] <= V2_MAX})
    return {"level": level, "pair": pair, "rho": rho, "cluster": cluster, "r_latent": r,
            "grid_matches_curve_file": grid_match, "max_abs_risk_diff_vs_curve_file": risk_diff,
            "sign_grid": sgrid, "b1_grid": bgrid, "sign_true_risk": pop["sign_R"], "b1_true_risk": pop["b1_R"],
            "sign_pop_commit": pop["sign_c"], "b1_pop_accept": pop["b1_c"],
            "zero_nc_certified": int(zero_nc_certified), "settings": settings}


def pct(x):
    return "   n/a" if x is None else f"{100 * x:6.2f}%"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=RUNS)
    ap.add_argument("--n-pop", type=int, default=P.N_POP)
    ap.add_argument("--curves-dir", default=str(P.OUT))
    ap.add_argument("--out", default=str(P.OUT))
    a = ap.parse_args(argv)
    t0 = time.time()
    engine_path = Path(C.__file__)
    engine_sha = sha256_file(engine_path)
    curves = {lv: load_curves(lv, a.curves_dir) for lv in LEVELS}

    r1, r3, r4 = v1(), v3(), v4_static()
    pairs = []
    for lv, pf in LEVELS.items():
        pair = 0
        for rho in P.RHOS:
            for cl in P.CLUSTERS:
                pairs.append(run_pair(lv, pair, rho, cl, pf, a.runs, a.n_pop, curves[lv][0]["curves"][pair]))
                print(f"  level {lv} rho {rho:.2f} {cl:8}: done ({time.time() - t0:.0f} s)", flush=True)
                pair += 1
    settings = [s for pr in pairs for s in pr["settings"]]
    zero_total = sum(pr["zero_nc_certified"] for pr in pairs)
    v4_pass = r4["pass"] and zero_total == 0
    v2_pass = all(s["v2_pass"] for s in settings)

    print(f"\n=== R9 Stage 2 V-checks (delta {C.DELTA}; synthetic only; {a.runs} runs per setting; "
          f"population {a.n_pop:,}) ===")
    print(f"\nV1 plan section 6 table: {'PASS' if r1['pass'] else 'FAIL'} ({r1['n_match']} / {r1['n_values']} values)")
    print(f"  {'n_c':>4} | " + " ".join(f"{100 * x:>9.1f}%" for x in PLAN_ALPHAS) + " | match")
    for row in r1["rows"]:
        print(f"  {row['n_c']:4d} | " + " ".join(f"{'none' if g is None else g:>10}" for g in row["engine"])
              + f" | {'yes' if row['match'] else 'NO (plan ' + str(row['plan']) + ')'}")
    print(f"  smallest zero-error n_c: engine {r1['min_n_engine']}, plan {r1['min_n_plan']}")

    print(f"\nV3 exact arithmetic: {'PASS' if r3['pass'] else 'FAIL'}")
    for c in r3["cases"]:
        print(f"  n_c {c['n_c']:4d} k {c['k']} alpha {100 * c['alpha']:.1f}%: exact {c['exact']:.12g} engine "
              f"{c['engine']:.12g} rel.err {c['rel_err']:.1e} {'certified' if c['certified'] else 'not certified'}")

    print(f"\nV4 n_c = 0 never certified: {'PASS' if v4_pass else 'FAIL'}")
    print(f"  direct: {r4['direct_cases']} cases, all p = 1: {r4['direct_all_p_equal_1']}")
    for t, w in r4["walk"].items():
        print(f"  walk ({t}): from strictest {w['certified_from_strictest']} (must be -1); "
              f"control from index 3 {w['control_from_index_3']} (must be 0)")
    print(f"  V2 runs: certified thresholds with n_c = 0 in the Cert sample: {zero_total} (must be 0)")

    print("\nCurve-file cross-check (sign-level grid and true risk recomputed from the same seeds):")
    for pr in pairs:
        d = pr["max_abs_risk_diff_vs_curve_file"]
        print(f"  level {pr['level']} rho {pr['rho']:.2f} {pr['cluster']:8}: grid equal {pr['grid_matches_curve_file']}, "
              f"max |R diff| {'not compared (population size differs)' if d is None else f'{d:.2e}'}")

    print(f"\nV2 sign-level exact binomial, CP95 upper <= {V2_MAX:.2f}: {'PASS' if v2_pass else 'FAIL'} "
          f"({sum(s['v2_pass'] for s in settings)} / {len(settings)} settings)")
    hdr = (f"{'lvl':>3} {'rho':>4} {'cluster':8} {'alpha':>5} | {'unit/test':12} {'cert':>7} {'viol':>5} {'rate':>7} "
           f"{'CP95':>17} {'mean R*':>8} {'med j':>5} {'nostart':>7}")
    print(hdr)
    for s in settings:
        for key in ("sign_binom", "sign_hb", "B1_binom", "B1_hb"):
            x = s["rows"][key]
            mj = "  n/a" if x["median_certified_index"] is None else f"{x['median_certified_index']:5.1f}"
            mr = "     n/a" if x["mean_true_risk_certified"] is None else f"{100 * x['mean_true_risk_certified']:7.3f}%"
            flag = ("  V2 " + ("PASS" if s["v2_pass"] else "FAIL")) if key == "sign_binom" else ""
            print(f"{s['level']:3d} {s['rho']:4.2f} {s['cluster']:8} {100 * s['alpha']:4.1f}% | {key:12} "
                  f"{100 * x['certified_share']:6.1f}% {x['violations']:5d} {100 * x['violation_rate']:6.2f}% "
                  f"[{100 * x['cp95'][0]:6.2f}, {100 * x['cp95'][1]:6.2f}]% {mr} {mj} {x['runs_no_eligible_start']:7d}{flag}")
        print()
    print("(cert = share of runs certifying a threshold; viol = certified and true risk R > alpha; mean R* = mean true "
          "risk of the certified threshold over certifying runs; med j = median certified grid index, 0 = loosest, "
          "29 = strictest; nostart = runs with no eligible start)")
    all_pass = r1["pass"] and v2_pass and r3["pass"] and v4_pass
    print(f"\nV1 {'PASS' if r1['pass'] else 'FAIL'} | V2 {'PASS' if v2_pass else 'FAIL'} | V3 "
          f"{'PASS' if r3['pass'] else 'FAIL'} | V4 {'PASS' if v4_pass else 'FAIL'} | ALL PASS: {all_pass}")

    if sha256_file(engine_path) != engine_sha:          # tamper guard: the engine must not change during the run
        raise SystemExit(f"STOP: {engine_path} changed during the run; no summary written")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = {"plan": "poc3-implementation-plan-r9 Stage 2 (V-checks)",
               "code_sha256": {"r9_certify.py": engine_sha, "r9_stage2_vchecks.py": sha256_file(Path(__file__)),
                               "r9_stage2_popcurves.py": sha256_file(Path(P.__file__)),
                               "r8_stage3_synth.py": sha256_file(Path(R8S.__file__)),
                               "r8_stage3_engine.py": sha256_file(Path(E.__file__))},
               "inputs_sha256": {f"population_curves_L{lv}.json": curves[lv][1] for lv in LEVELS},
               "config": {"delta": C.DELTA, "levels": LEVELS, "runs": a.runs, "n_pop": a.n_pop, "n_design": N_DESIGN,
                          "n_cert": P.N_CERT, "alphas": P.ALPHAS, "seed_key": "[20261114, 9, level, pair, stream]"},
               "V1": r1, "V2": {"pass": v2_pass, "max_cp_upper": V2_MAX}, "V3": r3,
               "V4": {**r4, "v2_runs_zero_nc_certified": zero_total, "pass": v4_pass},
               "all_pass": all_pass, "pairs": pairs, "seconds": round(time.time() - t0, 1)}
    path = out / "stage2_summary.json"
    path.write_text(json.dumps(P.jsonable(summary), indent=2))
    print(f"\nOutput: {path} ({summary['seconds']} s)")
    return summary


if __name__ == "__main__":
    main()
