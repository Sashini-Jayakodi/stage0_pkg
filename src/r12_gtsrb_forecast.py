"""r12 (EXPLORATORY, GTSRB development data): the r11 forecast applied to the POC-2 Stage 3b frame-level audit.

Definitions: LOG_r12_entry.md Part A (commit 6491dc8). The forecast is r11's (`src.r11_forecast_ucb.run_forecast`,
imported unchanged: Part A 867db93 and Amendment 1 109ec7b) with GTSRB inputs: pilot = the 261 design tracks
(30 frames each) on the POC-2 B1 grid; Cert 784 tracks x 30 frames; walk from position 19 down to 0.
Observed: Stage 3b B1 HB rows (reproduced exactly) and a new exact-binomial B1 rate on the same 2,000 draws.
Pilot bootstrap: 1,000 resamples of the 261 tracks, 10-90%. Criteria (HB, alpha 0.5% and 1%): Part A (f).

Usage:  py -3.12 -m src.r12_gtsrb_forecast --run runs/20261003_stage1
Output: runs/20261003_stage1/r12/r12_summary.json and r12_report.md
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from src import r11_forecast_ucb as F
from src import r8_stage2 as S2
from src import r8_stage3_engine as E
from src import r8_stage3_resample as RS
from src import r9_certify as C
from src import r9_stage10 as S10
from src.r8_frozen import load_frozen
from src.r8_stage3_b1design import frame_margins
from src.r9_frozen import check_frozen
from src.util import sha256_file

LABEL = "EXPLORATORY (GTSRB development data); LOG_r12_entry.md Part A, commit 6491dc8"
ALPHAS = (0.005, 0.01, 0.02)
JUDGED = (0.005, 0.01)             # Part A (f); 2% reported only
TESTS = ("hb", "binom")            # HB primary (judged), exact binomial secondary
M, N_DESIGN, N_CERT = 30, 261, 784
START = 19                         # Part A (b): the r11 start rule gives the POC-2 B1 order start; checked at run time
BOOT_N, BOOT_KEY = 1000, (20261119, 12)
DE_GTSRB = 13.9                    # recorded POC-1 design effect (Stage 10 retrospective)
STAGE3B_SHA256 = "7591a438b569e7231f3a40d8da6420a81b6f65791e707df4c0101338a0bf7367"   # POC-2 Stage 3 Part B
CRIT_ABS_PP, CRIT_UNDER_PP = 3.0, 2.0
OUT = Path("runs/20261003_stage1/r12")


# ------------------------------------------------------------------ pieces (tested on synthetic data)
def per_track(inv, acc, wr, n_tracks):
    """Accepted and accepted-and-wrong frame counts per track and threshold: (n_tracks, G) each, plus frames per track."""
    G = acc.shape[1]
    A = np.stack([np.bincount(inv, weights=acc[:, j], minlength=n_tracks) for j in range(G)], 1)
    W = np.stack([np.bincount(inv, weights=wr[:, j], minlength=n_tracks) for j in range(G)], 1)
    return A.astype(int), W.astype(int), np.bincount(inv, minlength=n_tracks)


def pilot_forecast(acc_g, wr_g, alpha, test, m=M, n_cert=N_CERT):
    """r11 (b) 1-6 for one pilot: start by the r11 rule from the pilot's own accept fractions, then run_forecast."""
    nd = acc_g.shape[0]
    f_design = acc_g.sum(0) / (nd * m)
    start = C.start_for(test, f_design, n_cert * m, alpha, nd * m)
    rec = F.run_forecast(acc_g, wr_g, m, alpha, test, start, nd, n_cert)
    rec["start"] = int(start)
    return rec


def observed_binom(A, W, Wd, alpha, start=START, delta=C.DELTA):
    """B1 with the exact binomial p-value on each draw, walked from 'start' down to 0. Returns certified index per draw."""
    Wd = np.atleast_2d(Wd)
    k, n_c = Wd @ W.astype(float), Wd @ A.astype(float)
    p = C.binom_pvalue(k, n_c, alpha)
    return E.walk_rows(p, list(range(start, -1, -1)), delta)


def bootstrap(acc_g, wr_g, idx, alpha, test):
    """Every step of r11 (b) recomputed on each resample of tracks; idx (B, n_design)."""
    return [pilot_forecast(acc_g[r], wr_g[r], alpha, test) for r in idx]


def criteria(forecast, observed):
    """Part A (f) for one alpha: (a) |F - O| <= 3 pp, (b) F >= O - 2 pp."""
    d = 100 * forecast - 100 * observed
    return {"a_abs_le_3": bool(abs(d) <= CRIT_ABS_PP), "b_not_below_2": bool(d >= -CRIT_UNDER_PP)}


def load_stage3b(path, want=STAGE3B_SHA256):
    sha = sha256_file(path)
    if sha != want:
        raise SystemExit(f"STOP: {path} SHA-256 {sha} != recorded {want}")
    return json.loads(Path(path).read_text())


# ------------------------------------------------------------------ inputs
def load_design(run, d):
    """The 261 seed-0 Fit tracks on the B1 working grid (as r10 Part 3); files were hash-checked by RS.load_pool."""
    run = Path(run)
    split = np.asarray(np.load(run / "splits" / "seed0.npz")["track_split"])
    E_ = np.load(run / "embeddings" / "train_release.npz")
    track_all = E_["track"].astype(int)
    rows = np.flatnonzero(split[track_all] == 0)
    ev = np.load(run / "poc2" / "stage1" / RS.EVID_FILE)
    mg, top = frame_margins(d["head"], E_["emb"][rows], ev["passed"][rows])
    y = E_["y"][rows].astype(int)
    codes, inv = np.unique(track_all[rows], return_inverse=True)
    thr = np.asarray(d["b1"]["work_thresholds"], float)
    acc = mg[:, None] >= thr[None, :]
    wr = acc & (top != y)[:, None]
    A, W, NF = per_track(inv, acc, wr, len(codes))
    return A, W, NF


def pct(x, nd=2):
    return "    n/a" if x is None else f"{100 * x:6.{nd}f}%"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/20261003_stage1")
    ap.add_argument("--stage3b", default=None)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args(argv)
    t0 = time.time()
    run = Path(a.run)
    cfg = load_frozen()
    frozen = check_frozen()
    delta = float(cfg["fixed_in_every_phase"]["delta"])
    assert delta == C.DELTA == 0.1
    s3b_path = Path(a.stage3b) if a.stage3b else run / "poc2" / "stage3" / "stage3b_summary.json"
    s3b = load_stage3b(s3b_path)
    engine_sha = sha256_file(Path(C.__file__))

    d = RS.load_pool(run, cfg)
    P, checks = RS.pool_tables(d, cfg)
    b1 = d["b1"]
    assert b1["order_positions"] == list(range(START, -1, -1)) and b1["start_position"] == START
    Ad, Wdes, NFd = load_design(run, d)
    checks["design_tracks_261"] = Ad.shape[0] == N_DESIGN
    checks["design_frames_30_each"] = bool((NFd == M).all())
    checks["design_work_c_reproduces"] = bool(np.allclose(Ad.sum(0) / NFd.sum(), b1["work_c"], rtol=0, atol=1e-12))
    checks["design_work_e_reproduces"] = bool(np.allclose(Wdes.sum(0) / Ad.sum(0), b1["work_e"], rtol=0, atol=1e-12))
    starts = {(al, t): C.start_for(t, Ad.sum(0) / (N_DESIGN * M), N_CERT * M, al, N_DESIGN * M) for al in ALPHAS for t in TESTS}
    checks["start_19_every_alpha_and_test"] = all(v == START for v in starts.values())

    T = P["n_tracks"]
    cfg3 = s3b["config"]
    Wd = RS.draw_weights(T, int(cfg3["n_cert"]), int(cfg3["draws"]), int(cfg3["seed"]))
    assert int(cfg3["n_cert"]) == N_CERT and int(cfg3["draws"]) == 2000
    res = RS.resample(P, cfg, b1, Wd, list(ALPHAS), delta)
    same = True
    for al in ALPHAS:
        st, mine = s3b["results"][str(al)]["B1"], res[al]["B1"]
        same &= mine["violations"] == st["violations"] and abs(mine["certified_share"] - st["certified_share"]) < 1e-12
    checks["stage3b_b1_hb_rows_reproduce"] = bool(same)
    if not all(checks.values()):
        raise SystemExit(f"STOP: a Part A (h) check failed: {[k for k, v in checks.items() if not v]}")

    pop = np.ones((1, T))
    pool_risk = P["W"].sum(0) / P["A"].sum(0)
    idx = np.random.default_rng(list(BOOT_KEY)).integers(0, N_DESIGN, (BOOT_N, N_DESIGN))
    heur = S10.de_to_delta_eff(DE_GTSRB)
    rows = {}
    for al in ALPHAS:
        mu_true = E.frame_mu(P["A"], P["W"], P["NF"], pop, al)[0][0]
        jt = F.j_true(mu_true, START, al)
        obs = {"hb": {k: res[al]["B1"][k] for k in ("violations", "violation_rate", "cp95", "certified_share")}}
        cb = observed_binom(P["A"], P["W"], Wd, al)
        rb = E.rate_row(cb, mu_true, al)
        obs["binom"] = {k: rb[k] for k in ("violations", "violation_rate", "cp95", "certified_share")}
        for t in TESTS:
            rec = pilot_forecast(Ad, Wdes, al, t)
            bs = bootstrap(Ad, Wdes, idx, al, t)
            fb = np.array([r["forecast"] for r in bs])
            cases = [r["case"] for r in bs]
            row = {"observed": obs[t], "forecast": rec["forecast"], "pilot": rec, "j_true": jt,
                   "jstar_at_or_before_jtrue": None if jt < 0 else bool(rec["j_star"] >= jt),
                   "boot_p10": float(np.quantile(fb, 0.1)), "boot_p90": float(np.quantile(fb, 0.9)),
                   "boot_median": float(np.median(fb)), "boot_share_zero": float(np.mean(fb == 0)),
                   "boot_cases": {c: int(sum(x == c for x in cases)) for c in ("betabinom", "rho_le_0", "zero_errors", "rho_ge_1", None)},
                   "boot_start_not_19": int(sum(r["start"] != START for r in bs)),
                   "heuristic": heur, "pool_risk_at_jstar": None if rec["j_star"] < 0 else float(pool_risk[rec["j_star"]])}
            if t == "hb" and al in JUDGED:
                row["criteria"] = criteria(rec["forecast"], obs[t]["violation_rate"])
            rows[(al, t)] = row
    judged = {f"{al}": rows[(al, "hb")]["criteria"] for al in JUDGED}
    overall = all(v for c in judged.values() for v in c.values())

    out = [f"=== r12: r11 forecast on the GTSRB frame-level audit ({LABEL}) ===",
           f"delta {delta}; pilot 261 design tracks x 30 frames; audit = POC-2 Stage 3b draws (2,000 x 784 of 1,046 pool "
           f"tracks, seed {cfg3['seed']}); bootstrap {BOOT_N} resamples, key {list(BOOT_KEY)}; heuristic DE {DE_GTSRB}.",
           "Checks (Part A (h)): " + "; ".join(f"{k} {'PASS' if v else 'FAIL'}" for k, v in checks.items()), ""]
    out.append(f"  {'alpha':>5} {'test':>5} | {'observed':>7} {'CP95':>18} {'cert':>6} | {'forecast':>8} "
               f"{'boot p10-p90':>18} {'boot med':>8} | {'heur.':>7} | {'gap':>6} | j* {'U_j*':>7} {'f':>6} "
               f"{'rho_I':>7} {'case':>11} N_g k_max {'clip':>8} | jt j*<=jt {'pool R(j*)':>9}")
    for al in ALPHAS:
        for t in TESTS:
            r = rows[(al, t)]
            o, p = r["observed"], r["pilot"]
            js = p["j_star"]
            u = "    n/a" if js < 0 else pct(p["U"], 3)
            ff = "   n/a" if js < 0 else f"{100 * p['f']:5.1f}%"
            rh = "    n/a" if p["rho_I"] is None else f"{p['rho_I']:7.4f}"
            km = "  n/a" if p.get("k_max") is None else f"{p['k_max']:5d}"
            ng = " n/a" if p["N_g"] is None else f"{p['N_g']:4d}"
            jb = "n/a" if r["jstar_at_or_before_jtrue"] is None else ("yes" if r["jstar_at_or_before_jtrue"] else "no")
            out.append(
                f"  {100 * al:4.1f}% {t:>5} | {pct(o['violation_rate'])} [{100 * o['cp95'][0]:5.2f}, {100 * o['cp95'][1]:5.2f}]% "
                f"{100 * o['certified_share']:5.1f}% | {pct(r['forecast'])} [{100 * r['boot_p10']:6.2f}, {100 * r['boot_p90']:6.2f}]% "
                f"{pct(r['boot_median'])} | {pct(r['heuristic'])} | {100 * (r['forecast'] - o['violation_rate']):+6.2f} | "
                f"{js:2d} {u} {ff} {rh} {str(p['case']):>11} {ng} {km} {p['clipped']:8.2e} | {r['j_true']:2d} {jb:>6} "
                f"{pct(r['pool_risk_at_jstar'], 3):>9}")
    out.append("\nBootstrap detail (counts of 1,000): forecast zero share; rho_I case counts; resamples whose start is not 19")
    for al in ALPHAS:
        for t in TESTS:
            r = rows[(al, t)]
            c = r["boot_cases"]
            out.append(f"  {100 * al:4.1f}% {t:>5}: zero {100 * r['boot_share_zero']:5.1f}%; betabinom {c['betabinom']}, "
                       f"rho_I<=0 {c['rho_le_0']}, zero errors {c['zero_errors']}, rho_I>=1 {c['rho_ge_1']}, no j* {c[None]}; "
                       f"start != 19: {r['boot_start_not_19']}")
    out += ["\nCRITERIA (Part A (f); HB only; consistency check on development data; 2% and exact binomial reported only):"]
    for al in JUDGED:
        r = rows[(al, "hb")]
        cr = r["criteria"]
        g = 100 * (r["forecast"] - r["observed"]["violation_rate"])
        out.append(f"  alpha {100 * al:.1f}%: forecast {100 * r['forecast']:.2f}%, observed "
                   f"{100 * r['observed']['violation_rate']:.2f}%, gap {g:+.2f} pp")
        out.append(f"    (a) |forecast - observed| <= 3 points: {'PASS' if cr['a_abs_le_3'] else 'FAIL'}")
        out.append(f"    (b) forecast >= observed - 2 points: {'PASS' if cr['b_not_below_2'] else 'FAIL'}")
    out.append(f"  Overall (all four must pass): {'PASS' if overall else 'FAIL'}")
    print("\n".join(out), flush=True)

    if sha256_file(Path(C.__file__)) != engine_sha:
        raise SystemExit("STOP: the engine changed during the run; no summary written")
    od = Path(a.out)
    od.mkdir(parents=True, exist_ok=True)
    summary = {"label": LABEL, "frozen_blobs": frozen,
               "code_sha256": {"r12_gtsrb_forecast.py": sha256_file(Path(__file__)),
                               "r11_forecast_ucb.py": sha256_file(Path(F.__file__)), "r9_certify.py": engine_sha,
                               "r8_stage3_resample.py": sha256_file(Path(RS.__file__)),
                               "r8_stage3_engine.py": sha256_file(Path(E.__file__))},
               "inputs_sha256": {**d["inputs_sha256"], "stage3b_summary.json": STAGE3B_SHA256},
               "config": {"delta": delta, "alphas": ALPHAS, "judged_alphas": JUDGED, "tests": TESTS, "m": M,
                          "n_design": N_DESIGN, "n_cert": N_CERT, "start": START, "boot_n": BOOT_N,
                          "boot_key": list(BOOT_KEY), "de_gtsrb": DE_GTSRB, "stage3b_seed": cfg3["seed"]},
               "checks": checks, "starts": {f"{al}_{t}": v for (al, t), v in starts.items()},
               "pool_micro_risk": pool_risk, "rows": {f"{al}_{t}": r for (al, t), r in rows.items()},
               "criteria": judged, "overall_pass": overall, "seconds": round(time.time() - t0, 1)}
    path = od / "r12_summary.json"
    path.write_text(json.dumps(S2.jsonable(summary), indent=2))
    (od / "r12_report.md").write_text("\n".join(out) + "\n")
    print(f"\nOutput: {path} SHA-256 {sha256_file(path)} and r12_report.md ({summary['seconds']} s). EXPLORATORY.")
    return summary


if __name__ == "__main__":
    main()
