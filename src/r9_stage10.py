"""POC-3 (R9) Stage 10, step 4: prospective check of the effective-delta formula. Synthetic data only.

LOG_r9_stage10_entry.md Part A (a) to (h).
  Prospective: 18 settings (rho 0.1/0.2/0.6 x m 5/10/15 x alpha 1%/2%), 2,000 runs each, Cert 784 / design 261 signs.
  Frame-level (B1) and sign-level violation rates with the frozen engine, against the predicted
  delta_eff = Phi(Phi^-1(delta) / sqrt(1 + (m - 1) rho)) committed in Part A (c). Reported, not scored.
  Retrospective (the formula was written after these): Stage 2 (36 settings), GTSRB POC-2 Stage 3b, BelgiumTSC Stage 6.

Usage:  py -3.12 -m src.r9_stage10
Output: runs/20261003_stage1/poc3/stage10/stage10_summary.json
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.stats import norm

from src import r8_stage3_engine as E
from src import r8_stage3_synth as R8S
from src import r9_certify as C
from src import r9_stage10_popcurves as Q
from src import r9_stage2_popcurves as P
from src import r9_stage2_vchecks as V
from src.r9_frozen import check_frozen
from src.util import sha256_file

RUNS = 2000
TESTS = C.RULES
UNITS = ("sign", "B1")
CURVES_PATH = Q.OUT / Q.CURVES_NAME
CURVES_SHA = "ffa785eafc545a45b4a2d364b072111f5f8dd9440d4b0f308753e2dcce0b119f"   # Part A (h) values

# Retrospective sources, SHA-256 as recorded in LOG.md (Stage 2 entry, POC-2 Stage 3 entry, Stage 6 entry)
RETRO = {"stage2": (Path("runs/20261003_stage1/poc3/stage2/stage2_summary.json"),
                    "8c152579609bb972b394d8c145746eae2975f7e7c79fae640ba829f46ee3729f"),
         "gtsrb_3b": (Path("runs/20261003_stage1/poc2/stage3/stage3b_summary.json"),
                      "7591a438b569e7231f3a40d8da6420a81b6f65791e707df4c0101338a0bf7367"),
         "btsc_6": (Path("runs/btsc/stage6/stage6_summary.json"),
                    "71be795779554a62546b1e03089222b53c35bc5a8ffac4bb2406c470b39e185f")}
# Part A (d): cluster sizes of the Stage 2 settings (view-mix: m* = sum T^2 / sum T over the mix), recorded DE values
STAGE2_M = {"fixed-3": 3.0, "view-mix": float((P.MIX_T ** 2 * P.MIX_COUNTS).sum() / (P.MIX_T * P.MIX_COUNTS).sum()),
            "fixed-30": 30.0}
GTSRB_RHO, GTSRB_DE = 0.446, 13.9
BTSC_RHO, BTSC_DE, BTSC_DE_CI = 0.631, 2.24, (1.25, 2.74)


def de_to_delta_eff(de, delta=C.DELTA):
    """delta_eff from a design effect directly (the retrospective rows give DE, not rho and m)."""
    return float(norm.cdf(norm.ppf(delta) / np.sqrt(de)))


def load_checked(path, want):
    """Hard rule 2: refuse a file whose SHA-256 differs from the recorded value."""
    sha = sha256_file(path)
    if sha != want:
        raise SystemExit(f"STOP: {path} SHA-256 {sha} != recorded {want}")
    return json.loads(Path(path).read_text()), sha


def simulate(rng, m, r, mu, sgrid, bgrid, nd, nc, runs, alphas):
    """The Stage 2 run loop (V.run_pair) for fixed T = m: fresh design then fresh Cert sample per run.
    Returns the certified index per (alpha, unit, rule) and run, the B1 binomial start per (alpha) and run, and the
    count of certified thresholds with no committed unit (V4 condition)."""
    cert = {(a, u, t): np.full(runs, -1) for a in alphas for u in UNITS for t in TESTS}
    starts = {a: np.full(runs, -1) for a in alphas}
    no_start = {(a, u, t): 0 for a in alphas for u in UNITS for t in TESTS}
    zero_nc = 0
    Td, Tc = np.full(nd, m), np.full(nc, m)
    for i in range(runs):
        Xd = Q.draw_fixed(nd, m, r, mu, rng)
        Xc = Q.draw_fixed(nc, m, r, mu, rng)
        fd = R8S.track_table(Xd, sgrid)[0].mean(0)
        c, w = R8S.track_table(Xc, sgrid)
        n_c, k = c.sum(0), w.sum(0)
        Ad, _, Nd = V.frame_totals(Xd, Td, bgrid)
        Ac, Wc, Nc = V.frame_totals(Xc, Tc, bgrid)
        for a in alphas:
            starts[a][i] = C.start_for("binom", Ad / Nd, Nc, a, Nd)
            for t in TESTS:
                no_start[(a, "sign", t)] += C.start_for(t, fd, nc, a, nd) < 0
                no_start[(a, "B1", t)] += C.start_for(t, Ad / Nd, Nc, a, Nd) < 0
                j = C.certify(k, n_c, nc, a, fd, t, n_design=nd)
                jb = C.certify(Wc, Ac, Nc, a, Ad / Nd, t, n_design=Nd)
                cert[(a, "sign", t)][i], cert[(a, "B1", t)][i] = j, jb
                zero_nc += (j >= 0 and n_c[j] == 0) + (jb >= 0 and Ac[jb] == 0)
    return cert, starts, no_start, zero_nc


def mean_loss(alpha, commit, risk):
    """POC-2 mean-loss form alpha (1 - c) + c R; > alpha <=> R > alpha (Stage 2 Part A (e))."""
    commit, risk = np.asarray(commit, float), np.asarray(risk, float)
    return alpha * (1 - commit) + commit * risk


def agree(cp_lo, pred):
    """Part A (e): agreement when the observed rate does not exceed delta_eff beyond its CP interval."""
    return bool(cp_lo <= pred)


def summarise(rows):
    """Count of agreeing settings, mean absolute and mean signed gap (percentage points) over the given rows."""
    if not rows:
        return {"n": 0, "n_agree": 0, "mean_abs_gap_pp": None, "mean_signed_gap_pp": None}
    gaps = np.array([100 * (r["observed"] - r["predicted"]) for r in rows])
    return {"n": len(rows), "n_agree": int(sum(r["agree"] for r in rows)),
            "mean_abs_gap_pp": float(np.abs(gaps).mean()), "mean_signed_gap_pp": float(gaps.mean())}


def curves_equal(a, b):
    a = np.array([np.nan if v is None else v for v in a], float)
    b = np.array([np.nan if v is None else v for v in b], float)
    return bool(a.shape == b.shape and np.array_equal(np.isnan(a), np.isnan(b))
                and np.array_equal(a[~np.isnan(a)], b[~np.isnan(b)]))


def run_pair(pair, rho, m, runs, n_pop, file_cv, boundary_by_alpha):
    cv = Q.pair_curves(pair, rho, m, n_pop)
    check = {"sign_grid": curves_equal(cv["sign_grid"], file_cv["sign_grid"]),
             "b1_grid": curves_equal(cv["b1_grid"], file_cv["b1_grid"])}
    if n_pop == P.N_POP:
        for key in ("sign_pop_commit", "sign_true_risk", "b1_pop_accept", "b1_true_risk"):
            check[key] = curves_equal(cv[key], file_cv[key])
    if not all(check.values()):
        raise SystemExit(f"STOP: pair {pair} curves differ from the committed curve file: {check}")
    cert, starts, no_start, zero_nc = simulate(Q.rng_for(pair, 2), m, cv["r_latent"], cv["mu"], cv["sign_grid"],
                                               cv["b1_grid"], Q.N_DESIGN, Q.N_CERT, runs, Q.ALPHAS)
    pred = Q.delta_eff(rho, m)
    settings = []
    for a in Q.ALPHAS:
        mus = {"sign": mean_loss(a, cv["sign_pop_commit"], cv["sign_true_risk"]),
               "B1": mean_loss(a, cv["b1_pop_accept"], cv["b1_true_risk"])}
        rows = {}
        for u in UNITS:
            for t in TESTS:
                cvec = cert[(a, u, t)]
                row = E.rate_row(cvec, mus[u], a)
                row["median_certified_index"] = float(np.median(cvec[cvec >= 0])) if (cvec >= 0).any() else None
                row["runs_no_eligible_start"] = int(no_start[(a, u, t)])
                rows[f"{u}_{t}"] = row
        b1 = rows["B1_binom"]
        st = starts[a]
        settings.append({"pair": pair, "rho": rho, "m": m, "alpha": a, "DE": Q.design_effect(rho, m),
                         "predicted": pred, "boundary": bool(boundary_by_alpha[a]["boundary"]),
                         "R_b": boundary_by_alpha[a]["R_b"], "observed": b1["violation_rate"], "cp95": b1["cp95"],
                         "agree": agree(b1["cp95"][0], pred),
                         "b1_start_used": {"median": float(np.median(st)), "min": int(st.min()), "max": int(st.max()),
                                           "share_at_diagnostic_start": float(np.mean(st == boundary_by_alpha[a]["start"]))},
                         "rows": rows})
    return {"pair": pair, "rho": rho, "m": m, "curve_check": check, "zero_nc_certified": int(zero_nc),
            "calibration": cv["calibration"], "settings": settings}


# ------------------------------------------------------------------ retrospective table (Part A (d))
def retrospective():
    s2, sha2 = load_checked(*RETRO["stage2"])
    g3, sha3 = load_checked(*RETRO["gtsrb_3b"])
    b6, sha6 = load_checked(*RETRO["btsc_6"])
    rows = []
    for pr in s2["pairs"]:
        for s in pr["settings"]:
            m = STAGE2_M[s["cluster"]]
            de = 1 + (m - 1) * s["rho"]
            pred = de_to_delta_eff(de)
            for rule in ("binom", "hb"):
                x = s["rows"][f"B1_{rule}"]
                rows.append({"source": "Stage 2", "level": s["level"], "p_frame": s["p_frame"], "rho": s["rho"],
                             "cluster": s["cluster"], "m": m, "alpha": s["alpha"], "rule": rule, "DE": de,
                             "predicted": pred, "predicted_ci": None, "observed": x["violation_rate"],
                             "cp95": x["cp95"], "certified_share": x["certified_share"],
                             "agree": agree(x["cp95"][0], pred)})
    pred_g = de_to_delta_eff(GTSRB_DE)
    for al, res in g3["results"].items():
        x = res["B1"]
        rows.append({"source": "GTSRB POC-2 Stage 3b", "rho": GTSRB_RHO, "m": 30, "alpha": float(al), "rule": "hb (POC-2)",
                     "DE": GTSRB_DE, "predicted": pred_g, "predicted_ci": None, "observed": x["violation_rate"],
                     "cp95": x["cp95"], "certified_share": x["certified_share"], "agree": agree(x["cp95"][0], pred_g)})
    pred_b = de_to_delta_eff(BTSC_DE)
    ci_b = [de_to_delta_eff(BTSC_DE_CI[0]), de_to_delta_eff(BTSC_DE_CI[1])]
    for al, cell in b6["rq1"]["B1"]["cells"].items():
        for rule in ("binom", "hb"):
            x = cell[rule]
            rows.append({"source": "BelgiumTSC Stage 6", "rho": BTSC_RHO, "m": 2.96, "alpha": float(al), "rule": rule,
                         "DE": BTSC_DE, "predicted": pred_b, "predicted_ci": ci_b, "observed": x["rate"],
                         "cp95": x["cp95"], "certified_share": x["certified"] / x["draws"],
                         "agree": agree(x["cp95"][0], pred_b)})
    return rows, {"stage2_summary.json": sha2, "stage3b_summary.json": sha3, "stage6_summary.json": sha6}


def pct(x):
    return "   n/a" if x is None else f"{100 * x:6.2f}%"


def cp(c):
    return f"[{100 * c[0]:5.2f}, {100 * c[1]:5.2f}]%"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=RUNS)
    ap.add_argument("--n-pop", type=int, default=P.N_POP)
    ap.add_argument("--curves", default=str(CURVES_PATH))
    ap.add_argument("--out", default=str(Q.OUT))
    a = ap.parse_args(argv)
    t0 = time.time()
    frozen = check_frozen()
    engine_path = Path(C.__file__)
    engine_sha = sha256_file(engine_path)
    preds = Q.check_predictions()
    curves, curves_sha = load_checked(a.curves, CURVES_SHA)
    retro, retro_sha = retrospective()

    by = {(s["pair"], s["alpha"]): s for s in curves["settings"]}
    pairs = []
    for pair, (rho, m) in enumerate(Q.PAIRS):
        pairs.append(run_pair(pair, rho, m, a.runs, a.n_pop, curves["curves"][pair],
                              {al: by[(pair, al)] for al in Q.ALPHAS}))
        print(f"  pair {pair} rho {rho:.1f} m {m:2d}: done ({time.time() - t0:.0f} s)", flush=True)
    settings = [s for pr in pairs for s in pr["settings"]]
    summ = {"all": summarise(settings), "boundary": summarise([s for s in settings if s["boundary"]])}
    zero_total = sum(pr["zero_nc_certified"] for pr in pairs)

    print(f"\n=== R9 Stage 10: prospective check of delta_eff (delta {C.DELTA}; synthetic; {a.runs} runs per setting; "
          f"Cert {Q.N_CERT} / design {Q.N_DESIGN} signs; P_FRAME {Q.P_FRAME}) ===")
    print(f"Curve file {a.curves}: SHA-256 checked; curves recomputed and equal: "
          f"{all(all(pr['curve_check'].values()) for pr in pairs)}")
    print(f"Certified thresholds with no committed unit (all runs, both units, all rules): {zero_total}")

    print("\nFrame-level B1, exact binomial: predicted delta_eff against observed violation rate (95% CP)")
    print(f"  {'pair':>4} {'rho':>4} {'m':>3} {'alpha':>5} {'bnd':>4} | {'DE':>5} {'pred':>7} | {'observed':>7} "
          f"{'CP95':>18} {'cert':>6} | agree {'gap pp':>7} | start used (med, min-max; share = diag.)")
    for s in settings:
        r = s["rows"]["B1_binom"]
        su = s["b1_start_used"]
        print(f"  {s['pair']:4d} {s['rho']:4.1f} {s['m']:3d} {100 * s['alpha']:4.1f}% {'yes' if s['boundary'] else 'no':>4} | "
              f"{s['DE']:5.2f} {pct(s['predicted'])} | {pct(s['observed'])} {cp(s['cp95']):>18} "
              f"{100 * r['certified_share']:5.1f}% | {'yes' if s['agree'] else 'NO ':>5} "
              f"{100 * (s['observed'] - s['predicted']):+7.2f} | {su['median']:.0f}, {su['min']}-{su['max']}; "
              f"{100 * su['share_at_diagnostic_start']:.1f}%")

    print("\nSummary (B1 exact binomial against delta_eff):")
    for name, x in summ.items():
        lab = "all 18 settings" if name == "all" else f"boundary subset ({x['n']} settings)"
        if x["n"] == 0:
            print(f"  {lab}: empty")
            continue
        print(f"  {lab}: {x['n_agree']} of {x['n']} within the bound; mean |gap| {x['mean_abs_gap_pp']:.2f} pp; "
              f"mean signed gap {x['mean_signed_gap_pp']:+.2f} pp")

    print("\nOther rules and the sign level (violation rate [95% CP], certified share):")
    print(f"  {'pair':>4} {'rho':>4} {'m':>3} {'alpha':>5} | {'B1 hb (primary start)':>34} | {'sign binom':>34} | "
          f"{'sign hb (primary start)':>34}")
    for s in settings:
        cells = []
        for key in ("B1_hb", "sign_binom", "sign_hb"):
            r = s["rows"][key]
            cells.append(f"{pct(r['violation_rate'])} {cp(r['cp95'])} {100 * r['certified_share']:5.1f}%")
        print(f"  {s['pair']:4d} {s['rho']:4.1f} {s['m']:3d} {100 * s['alpha']:4.1f}% | " + " | ".join(f"{c:>34}" for c in cells))
    print("  Sign-level CP95 upper > 0.12 (would be at odds with Stage 2 V2): "
          f"{[(s['pair'], s['alpha']) for s in settings if s['rows']['sign_binom']['cp95'][1] > V.V2_MAX] or 'none'}")
    print("\nAll rules, all units (cert = share certifying; viol; rate; CP95; mean R* of certified; median index; no start):")
    for s in settings:
        for key in [f"{u}_{t}" for u in UNITS for t in TESTS]:
            x = s["rows"][key]
            mr = "     n/a" if x["mean_true_risk_certified"] is None else f"{100 * x['mean_true_risk_certified']:7.3f}%"
            mj = "  n/a" if x["median_certified_index"] is None else f"{x['median_certified_index']:5.1f}"
            print(f"  {s['pair']} {s['rho']:.1f} m{s['m']:2d} {100 * s['alpha']:.0f}% {key:12} {100 * x['certified_share']:6.1f}% "
                  f"{x['violations']:5d} {pct(x['violation_rate'])} {cp(x['cp95'])} {mr} {mj} {x['runs_no_eligible_start']:5d}")

    print("\nRETROSPECTIVE (the formula was written after these results; not a test of it)")
    print(f"  {'source':22} {'lvl':>3} {'rho':>5} {'cluster/m':>9} {'alpha':>5} {'rule':>10} | {'DE':>5} "
          f"{'pred':>7} {'pred range':>17} | {'observed':>7} {'CP95':>18} {'cert':>6} | agree")
    for r in retro:
        lvl = f"{r['level']:3d}" if "level" in r else "  -"
        cl = r.get("cluster", f"{r['m']:g}")
        rng_ = "" if r["predicted_ci"] is None else f"[{100 * r['predicted_ci'][0]:.2f}, {100 * r['predicted_ci'][1]:.2f}]%"
        print(f"  {r['source']:22} {lvl} {r['rho']:5.3f} {cl:>9} {100 * r['alpha']:4.1f}% {r['rule']:>10} | {r['DE']:5.2f} "
              f"{pct(r['predicted'])} {rng_:>17} | {pct(r['observed'])} {cp(r['cp95']):>18} {100 * r['certified_share']:5.1f}% | "
              f"{'yes' if r['agree'] else 'NO'}")
    print("  (Stage 2 view-mix m* = %.4f; GTSRB 3b used the POC-2 HB test; BelgiumTSC pred range from DE %.2f-%.2f)"
          % (STAGE2_M["view-mix"], *BTSC_DE_CI))

    if sha256_file(engine_path) != engine_sha:
        raise SystemExit(f"STOP: {engine_path} changed during the run; no summary written")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = {"plan": "poc3-implementation-plan-r9, Stage 10 (declared addition)",
               "frozen_blobs": frozen,
               "code_sha256": {"r9_stage10.py": sha256_file(Path(__file__)), "r9_certify.py": engine_sha,
                               "r9_stage10_popcurves.py": sha256_file(Path(Q.__file__)),
                               "r9_stage2_popcurves.py": sha256_file(Path(P.__file__)),
                               "r9_stage2_vchecks.py": sha256_file(Path(V.__file__)),
                               "r8_stage3_synth.py": sha256_file(Path(R8S.__file__)),
                               "r8_stage3_engine.py": sha256_file(Path(E.__file__))},
               "inputs_sha256": {Q.CURVES_NAME: curves_sha, **retro_sha},
               "config": {"delta": C.DELTA, "runs": a.runs, "n_pop": a.n_pop, "n_cert": Q.N_CERT,
                          "n_design": Q.N_DESIGN, "p_frame": Q.P_FRAME, "alphas": Q.ALPHAS, "pairs": Q.PAIRS,
                          "seed_key": "[20261115, 10, pair, stream]"},
               "predictions": preds, "zero_nc_certified": zero_total, "summary": summ, "pairs": pairs,
               "retrospective": retro, "seconds": round(time.time() - t0, 1)}
    path = out / "stage10_summary.json"
    path.write_text(json.dumps(P.jsonable(summary), indent=2))
    print(f"\nOutput: {path} SHA-256 {sha256_file(path)} ({summary['seconds']} s)")
    return summary


if __name__ == "__main__":
    main()
