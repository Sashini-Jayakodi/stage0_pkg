"""Stage 6, step 1: prove the certification engine on SYNTHETIC data before any real data is touched (plan, Stage 6 table).

Usage:  py -3.12 -m src.certify_synth --run runs/20261003_stage1            (about 2-3 minutes; writes <run>/stage6/synthetic_validation.json)

Six checks, all must pass (the real-data step src.stage6 refuses to run otherwise, and also refuses if src/certify.py changed afterwards):
  1. null          : data with E[L] = alpha EXACTLY -> P(p <= 0.05) <= 0.05 (several distributions, sizes and alphas)
  2. hand values   : p-values equal exact-arithmetic values for 5 small cases (n = 10, 50, 392 with mu_hat = 0, plus two with mu_hat > 0,
                     one where the Hoeffding term is the minimum and one where the Bentkus term is)
  3. floor         : zero errors and everything accepted at n = 392 tracks -> smallest certifiable alpha ~ 0.76%
  4. track validity: 1000 runs per design row (beta rho_sim 0, 0.1, 0.3 and a confident-cluster design), 784 Cert tracks x 30 frames: share of runs whose certified threshold has TRUE risk > alpha
                     must be <= 0.05 (Clopper-Pearson upper bound <= 0.07). The certifier must also actually certify (not vacuous).
  5. naive breaks  : the same runs with every frame its own 'track': on the cluster design the violation rate must clearly exceed 0.05 (CP lower bound > 0.05; negative control)
  6. ordering fix  : a low-risk stratum is certified with the plan-6.3 Fit-ordered sequence and NOT with strict-to-loose ordering

Synthetic designs (choices, stated here and logged). Each track has 30 frames, margins are exponential, alpha = 2%, level delta_k = 0.05
(one stratum of K = 2 at delta = 0.1). Because every track has the same number of frames, the TRUE track-averaged risk at threshold lam is exact
(true_risk) and 'violated' means R(lam_hat) > alpha (equivalently E[L] > alpha). A 200,000-track population cross-checks the formulas.
  beta    : track error probability q_t ~ Beta(mean 8%, rho_sim = 1/(a+b+1)), frames independent given q_t; wrong frames have margin mean 1,
            correct frames mean 20. rho_sim in {0, 0.1, 0.3}.
  cluster : 99.5% of tracks are 'normal' (error probability 2%, wrong-frame margin mean 1); 0.5% are 'confusable' tracks in which EVERY frame is wrong
            and margins look like correct ones (mean 20): confident, whole-track errors. This is the project's premise (confusable pairs -> confident
            errors that arrive in whole tracks) and it is the negative control that matters (see the design note below).
Design note (first draft rejected, 2026-10-11, before any real data was touched): the first draft used correct-frame margin mean 4 and only the beta
design. Two things were wrong with the DESIGN, not the engine (the null, hand-value, floor, ordering and population checks passed): (1) with mean 4
no threshold reaches the ~0.5% accepted risk at high coverage that the Hoeffding-Bentkus test needs at n = 784 and alpha = 2% (it needs a mean shifted
loss of about 0.0095, i.e. below half of alpha), so the track-level certifier certified nothing and the non-vacuity check had no content; (2) in the beta
design the naive certifier never broke (violation <= 2%): accepting only confident frames thins the errors, which dilutes their intra-track
correlation (an error ICC of 0.3 becomes about 0.07 among accepted errors), and the Hoeffding-Bentkus bound is conservative enough to absorb that.
Confident whole-track errors keep the clustering after the thinning, which is why the cluster design carries the negative control. The engine
(src/certify.py) was not changed.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import beta as beta_dist

from src import certify as C
from src import splits as S
from src.util import sha256_file

CFG = {"frames": 30, "n_fit": 261, "n_cert": 784, "mean_err": 0.08, "alpha": 0.02, "delta_k": 0.05, "mu_wrong": 1.0, "mu_correct": 20.0,
       "points": 50, "top_q": 0.98, "runs": 1000, "rhos": [0.0, 0.1, 0.3], "null_runs": 5000, "low_risk_err": 0.005, "ordering_runs": 200,
       "cluster": {"q_normal": 0.02, "pi": 0.005, "q_conf": 1.0}, "population_tracks": 200_000, "seed": 20261011}

# Exact-arithmetic reference values (fractions / 40-digit decimals, independent of scipy): (mu_hat, n, alpha, p)
HAND = [(0.0, 10, 0.1, 0.3486784401),
        (0.0, 50, 0.05, 0.07694497527671333),
        (0.0, 392, 0.0077, 0.048311732434177586),
        (0.05, 20, 0.1, 0.7159606844069272),           # Hoeffding term is the minimum (Bentkus term is 1.0649)
        (0.02, 100, 0.1, 0.005286744607654344)]        # Bentkus term is the minimum (Hoeffding term is 0.005937)


def cp_bounds(k, n, conf=0.95):
    """Clopper-Pearson interval for k events in n trials."""
    a = (1 - conf) / 2
    lo = 0.0 if k == 0 else float(beta_dist.ppf(a, k, n - k + 1))
    hi = 1.0 if k == n else float(beta_dist.ppf(1 - a, k + 1, n - k))
    return lo, hi


# ------------------------------------------------------------------ synthetic data
def beta_design(mean_err, rho):
    return {"kind": "beta", "mean_err": mean_err, "rho": rho}


def cluster_design(cl):
    return {"kind": "cluster", **cl}


def make_tracks(n, frames, design, rng, mu_w, mu_c):
    """margin and wrong, both (n, frames). Frames are independent given the track."""
    if design["kind"] == "beta":
        if design["rho"] > 0:
            s = 1.0 / design["rho"] - 1.0                          # a + b
            q = rng.beta(design["mean_err"] * s, (1 - design["mean_err"]) * s, n)
        else:
            q = np.full(n, design["mean_err"])
        conf = np.zeros(n, bool)
    else:
        conf = rng.random(n) < design["pi"]                        # confusable tracks
        q = np.where(conf, design["q_conf"], design["q_normal"])
    wrong = rng.random((n, frames)) < q[:, None]
    scale_w = np.where(conf, mu_c, mu_w)[:, None]                  # wrong frames of a confusable track look as confident as correct ones
    margin = np.where(wrong, rng.exponential(1.0, (n, frames)) * scale_w, rng.exponential(mu_c, (n, frames)))
    return margin, wrong


def true_risk(lam, design, mu_w, mu_c):
    """Exact track-averaged selective risk at threshold lam (equal-size tracks, so the track average equals the frame average)."""
    lam = np.asarray(lam, dtype=float)
    if design["kind"] == "beta":
        e = design["mean_err"]
        w, c = e * np.exp(-lam / mu_w), (1 - e) * np.exp(-lam / mu_c)
        return w / (w + c)
    pi, qa, qb = design["pi"], design["q_normal"], design["q_conf"]
    wa, ca = qa * np.exp(-lam / mu_w), (1 - qa) * np.exp(-lam / mu_c)
    wb, cb = qb * np.exp(-lam / mu_c), (1 - qb) * np.exp(-lam / mu_c)
    return ((1 - pi) * wa + pi * wb) / ((1 - pi) * (wa + ca) + pi * (wb + cb))


def fit_calibration(cfg, design, rng, alpha):
    """Grid and test order from an independent synthetic Fit set, using the same Stage 4 functions as the real pipeline."""
    m, w = make_tracks(cfg["n_fit"], cfg["frames"], design, rng, cfg["mu_wrong"], cfg["mu_correct"])
    track = np.repeat(np.arange(cfg["n_fit"]), cfg["frames"])
    m, w = m.ravel(), w.ravel()
    grid = S.margin_grid(m, cfg["points"], cfg["top_q"])
    el = S.el_curve(track, np.arange(len(m)), m, w, grid, alpha, cfg["n_fit"])
    start, order = S.order_from_curve(el)
    return grid, order


def certify_cert_set(margin, wrong, grid, order, alpha, delta_k, naive):
    n, f = margin.shape
    track = np.arange(n * f) if naive else np.repeat(np.arange(n), f)         # naive: every frame is its own 'track'
    return C.certify_sequence(track, margin.ravel(), wrong.ravel(), grid, order, alpha, delta_k)


# ------------------------------------------------------------------ the six checks
def check_null(cfg):
    """P(p <= delta_k) when E[L] = alpha exactly, for several bounded distributions (Bernoulli is the extreme case for Bentkus)."""
    rng = np.random.default_rng([cfg["seed"], 1])
    worst, rows = 0.0, []
    for alpha in (0.01, 0.02, 0.05):
        for n in (50, 392, 784):
            for name, draw in (
                    ("bernoulli(alpha)", lambda sh: (rng.random(sh) < alpha).astype(float)),
                    ("two-point {0, 2 alpha}", lambda sh: 2 * alpha * (rng.random(sh) < 0.5)),
                    ("three-point {0, 0.3}", lambda sh: 0.3 * (rng.random(sh) < alpha / 0.3)),
                    ("beta(mean alpha, rho .3)", lambda sh: rng.beta(alpha * (1 / 0.3 - 1), (1 - alpha) * (1 / 0.3 - 1), sh))):
                L = draw((cfg["null_runs"], n))
                p = C.hb_pvalue(L.mean(1), n, alpha)
                rate = float((p <= cfg["delta_k"]).mean())
                worst = max(worst, rate)
                rows.append({"dist": name, "alpha": alpha, "n": n, "reject_rate": rate})
    return {"pass": bool(worst <= cfg["delta_k"]), "worst_reject_rate": worst, "limit": cfg["delta_k"], "cases": len(rows),
            "runs_per_case": cfg["null_runs"], "top5_cases": sorted(rows, key=lambda r: -r["reject_rate"])[:5]}


def check_hand():
    rows = []
    for mu, n, alpha, ref in HAND:
        got = float(C.hb_pvalue(mu, n, alpha))
        rows.append({"mu_hat": mu, "n": n, "alpha": alpha, "reference": ref, "engine": got, "rel_err": abs(got - ref) / ref})
    return {"pass": bool(all(r["rel_err"] < 1e-9 for r in rows)), "cases": rows}


def check_floor(cfg):
    closed = float(C.floor_alpha(392, cfg["delta_k"]))
    # engine route: 392 tracks, no errors, everything accepted -> bisect the smallest alpha the certifier accepts
    track, margin, wrong = np.repeat(np.arange(392), 30), np.full(392 * 30, 5.0), np.zeros(392 * 30, bool)
    lo, hi = 1e-4, 0.05
    for _ in range(60):
        mid = (lo + hi) / 2
        if C.certify_sequence(track, margin, wrong, np.array([0.0]), [0], mid, cfg["delta_k"])["certified"]:
            hi = mid
        else:
            lo = mid
    return {"pass": bool(abs(hi - closed) < 1e-6 and 0.0075 < closed < 0.0077), "closed_form_n392": closed, "engine_bisection_n392": hi,
            "n450_closed_form": float(C.floor_alpha(450, cfg["delta_k"]))}


def validity_rows(cfg):
    rows = [(f"beta rho_sim={r}", beta_design(cfg["mean_err"], r), r) for r in cfg["rhos"]]
    rows.append(("cluster (confident whole-track errors)", cluster_design(cfg["cluster"]), None))
    return rows


def check_validity(cfg):
    """Track-level vs naive frame-level certification, repeated on fresh synthetic Fit/Cert draws, for every design row."""
    alpha, dk = cfg["alpha"], cfg["delta_k"]
    out = {"per_design": []}
    for ri, (name, design, rho) in enumerate(validity_rows(cfg)):
        rng = np.random.default_rng([cfg["seed"], 4, ri])
        res = {"track": {"cert": 0, "viol": 0}, "naive": {"cert": 0, "viol": 0}}
        thr = {"track": [], "naive": []}
        for _ in range(cfg["runs"]):
            grid, order = fit_calibration(cfg, design, rng, alpha)
            margin, wrong = make_tracks(cfg["n_cert"], cfg["frames"], design, rng, cfg["mu_wrong"], cfg["mu_correct"])
            for kind in ("track", "naive"):
                r = certify_cert_set(margin, wrong, grid, order, alpha, dk, naive=(kind == "naive"))
                if r["certified"]:
                    res[kind]["cert"] += 1
                    res[kind]["viol"] += int(true_risk(r["threshold"], design, cfg["mu_wrong"], cfg["mu_correct"]) > alpha)
                    thr[kind].append(r["threshold"])
        row = {"design": name, "kind": design["kind"], "rho_sim": rho, "runs": cfg["runs"]}
        for kind in ("track", "naive"):
            k, n = res[kind]["viol"], cfg["runs"]
            lo, hi = cp_bounds(k, n)
            row[kind] = {"certified_share": res[kind]["cert"] / n, "violations": k, "violation_rate": k / n, "cp95": [lo, hi],
                         "mean_certified_threshold": float(np.mean(thr[kind])) if thr[kind] else None}
        out["per_design"].append(row)
    rows = out["per_design"]
    track_ok = all(r["track"]["violation_rate"] <= 0.05 and r["track"]["cp95"][1] <= 0.07 for r in rows)
    nonvacuous = all(r["track"]["certified_share"] >= 0.5 for r in rows if r["rho_sim"] is None or r["rho_sim"] <= 0.1)
    cl = [r for r in rows if r["kind"] == "cluster"][0]
    naive_ok = cl["naive"]["cp95"][0] > 0.05                           # lower confidence bound above 5%: clearly broken
    out.update(track_level_valid=bool(track_ok), track_level_non_vacuous=bool(nonvacuous), naive_clearly_broken_on_cluster_design=bool(naive_ok),
               pass_track=bool(track_ok and nonvacuous), pass_naive=bool(naive_ok))
    return out


def check_ordering(cfg):
    alpha, dk = cfg["alpha"], cfg["delta_k"]
    rng = np.random.default_rng([cfg["seed"], 6])
    fit_ok = strict_ok = 0
    for _ in range(cfg["ordering_runs"]):
        design = beta_design(cfg["low_risk_err"], 0.1)
        grid, order = fit_calibration(cfg, design, rng, alpha)
        margin, wrong = make_tracks(cfg["n_cert"], cfg["frames"], design, rng, cfg["mu_wrong"], cfg["mu_correct"])
        fit_ok += certify_cert_set(margin, wrong, grid, order, alpha, dk, False)["certified"]
        strict_ok += certify_cert_set(margin, wrong, grid, np.arange(len(grid) - 1, -1, -1), alpha, dk, False)["certified"]
    n = cfg["ordering_runs"]
    return {"pass": bool(fit_ok / n >= 0.9 and strict_ok / n <= 0.05), "low_risk_error_rate": cfg["low_risk_err"],
            "certified_share_fit_ordered": fit_ok / n, "certified_share_strict_to_loose": strict_ok / n, "runs": n}


def check_population(cfg):
    """Cross-check of the exact risk formulas against a simulated 200,000-track population (beta rho_sim = 0 and 0.3, and the cluster design)."""
    rng = np.random.default_rng([cfg["seed"], 7])
    grid = np.quantile(np.concatenate([rng.exponential(cfg["mu_wrong"], 8000), rng.exponential(cfg["mu_correct"], 92000)]), np.linspace(0, 0.9, 25))
    out = {}
    for name, design in (("beta_rho_0", beta_design(cfg["mean_err"], 0.0)), ("beta_rho_0.3", beta_design(cfg["mean_err"], 0.3)),
                         ("cluster", cluster_design(cfg["cluster"]))):
        m, w = make_tracks(cfg["population_tracks"], cfg["frames"], design, rng, cfg["mu_wrong"], cfg["mu_correct"])
        diffs = []
        for lam in grid:
            acc = m >= lam
            e_share, a_share = (acc & w).mean(), acc.mean()                # equal-size tracks: track average = frame average
            diffs.append(abs(e_share / a_share - true_risk(lam, design, cfg["mu_wrong"], cfg["mu_correct"])))
        out[name] = float(max(diffs))
    return {"pass": bool(max(out.values()) < 0.003), "max_abs_diff": out, "population_tracks": cfg["population_tracks"]}


def run_all(cfg):
    res = {"constants": cfg}
    res["1_null"] = check_null(cfg)
    res["2_hand_values"] = check_hand()
    res["3_floor"] = check_floor(cfg)
    v = check_validity(cfg)
    res["4_track_level_validity"] = {k: v[k] for k in v if k != "pass_naive"}
    res["4_track_level_validity"]["pass"] = v["pass_track"]
    res["5_naive_breaks"] = {"pass": v["pass_naive"], "per_design": v["per_design"]}
    res["6_ordering_fix"] = check_ordering(cfg)
    res["population_cross_check"] = check_population(cfg)
    res["all_pass"] = bool(all(res[k]["pass"] for k in ("1_null", "2_hand_values", "3_floor", "4_track_level_validity", "5_naive_breaks",
                                                         "6_ordering_fix", "population_cross_check")))
    return res


def report(res):
    print("=== Stage 6 step 1: synthetic validation of the certification engine ===")
    n = res["1_null"]
    print(f"\n1. null (E[L] = alpha exactly): worst P(p <= {n['limit']}) over {n['cases']} cases = {n['worst_reject_rate']:.4f}  -> {'PASS' if n['pass'] else 'FAIL'}")
    print("\n2. hand-computed p-values (exact arithmetic vs engine):")
    for r in res["2_hand_values"]["cases"]:
        print(f"   n={r['n']:4d} alpha={r['alpha']:<6} mu_hat={r['mu_hat']:<5} reference {r['reference']:.10f} engine {r['engine']:.10f} rel.err {r['rel_err']:.1e}")
    print(f"   -> {'PASS' if res['2_hand_values']['pass'] else 'FAIL'}")
    f = res["3_floor"]
    print(f"\n3. floor at n = 392: closed form {100*f['closed_form_n392']:.3f}%, engine bisection {100*f['engine_bisection_n392']:.3f}% (n = 450: {100*f['n450_closed_form']:.3f}%) -> {'PASS' if f['pass'] else 'FAIL'}")
    v = res["4_track_level_validity"]
    print(f"\n4/5. {res['constants']['runs']} runs per row; violation = certified threshold whose TRUE risk > alpha = {res['constants']['alpha']:.0%}")
    print("   design                                   | track-level: certified  violations  rate   CP95 upper | naive frame-level: certified  violations  rate   CP95 [lower, upper]")
    for r in v["per_design"]:
        t, nv = r["track"], r["naive"]
        print(f"   {r['design']:<40} |              {t['certified_share']:6.1%}   {t['violations']:5d}   {t['violation_rate']:6.3f}   {t['cp95'][1]:.3f}     |"
              f"                   {nv['certified_share']:6.1%}   {nv['violations']:5d}   {nv['violation_rate']:6.3f}   [{nv['cp95'][0]:.3f}, {nv['cp95'][1]:.3f}]")
    print(f"   4. track-level valid on every row and non-vacuous -> {'PASS' if v['pass'] else 'FAIL'} | 5. naive clearly broken on the cluster design -> {'PASS' if res['5_naive_breaks']['pass'] else 'FAIL'}")
    print("      (naive on the beta rows is informational: thinning to confident frames dilutes the error correlation and the HB bound absorbs it; see the design note in the module docstring)")
    o = res["6_ordering_fix"]
    print(f"\n6. ordering (true error {o['low_risk_error_rate']:.1%} < alpha): certified with the Fit-ordered sequence {o['certified_share_fit_ordered']:.0%}, "
          f"with strict-to-loose {o['certified_share_strict_to_loose']:.0%} -> {'PASS' if o['pass'] else 'FAIL'}")
    p = res["population_cross_check"]
    print(f"\nCross-check of the exact risk formula against a {p['population_tracks']:,}-track population: max abs difference {p['max_abs_diff']} -> {'PASS' if p['pass'] else 'FAIL'}")
    print(f"\nALL PASS: {res['all_pass']}")
    print("If anything failed: it is a bug, not a result. Do not touch real data. Check (plan): per-track averaging, the sign of the shift, the clipping in h1, ceil(n mu_hat), tracks with n_tk >= 1 only, the stopping rule.")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--runs", type=int, help="validity runs per rho (default 1000; fewer only for smoke tests)")
    ap.add_argument("--out", help="default <run>/stage6")
    a = ap.parse_args(argv)
    cfg = dict(CFG)
    if a.runs:
        cfg["runs"] = a.runs
    out = Path(a.out) if a.out else Path(a.run) / "stage6"
    out.mkdir(parents=True, exist_ok=True)
    res = run_all(cfg)
    here = Path(__file__).resolve().parent
    res["certify_sha256"] = sha256_file(here / "certify.py")
    res["certify_synth_sha256"] = sha256_file(here / "certify_synth.py")
    (out / "synthetic_validation.json").write_text(json.dumps(res, indent=2, default=float))
    report(res)
    return res


if __name__ == "__main__":
    main()
