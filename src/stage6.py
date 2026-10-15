"""Stage 6, step 2: certification on the real data (plan section 6, Stage 6 "Then on real data"). CPU only, numpy/scipy, a few minutes.

Order of work (it matters, and the code is built to enforce it):
  0. REFUSES TO RUN unless <run>/stage6/synthetic_validation.json says all_pass and was made with this exact src/certify.py.
  1. Verifies the Stage 4 files, recomputes each seed's split, gate tau and cut-point c* and asserts they equal the Stage 4 values; the Fit-side
     margin grids and E[L] curves for the two eps schemes are recomputed here by a generic routine and asserted equal to the saved Stage 4 arrays.
  2. DECIDES: for every scheme x alpha x group, the certified margin threshold comes from Fit (grid and order) and Cert (the test). The Test split's
     labels are never read here: the label array handed to the decision code has Test entries forced to False.
  3. EVALUATES: only after all decisions are fixed, the Test labels are read to measure coverage and the realised track-averaged selective risk.

Schemes (all share src/certify.py):
  eps2        K = 2 strata from the pre-registered eps rule (PRIMARY)             delta_k = delta / 2
  eps3        K = 2 strata from the top-3 eps (first fallback, side by side)     delta_k = delta / 2
  marginal    K = 1, no strata                                                    delta_k = delta
  stallkamp6  K = 6 Stallkamp subsets of the PREDICTED class                      delta_k = delta / 6
  mondrian43  K = 43 groups = predicted class                                     delta_k = delta / 43
  alpha: absolute {1, 2, 3, 5}% for every scheme; 0.5 x Fit base error (per-track mean = primary, pooled = sensitivity) for eps2, eps3, marginal.
Gate-rejected frames are escalated in every scheme. An uncertifiable group escalates all of its frames.

Also: G1(iii) (some absolute alpha in {1, 2, 3}% certifiable in both eps strata in >= 4 of 5 reshuffles), the floor table, an empirical
smallest-certifiable-alpha scan (DESCRIPTIVE: picking the smallest alpha after the fact is not itself a certificate), and the H1 preview
(marginal threshold applied to the hard stratum, Test, track bootstrap).

Usage:  py -3.12 -m src.certify_synth --run runs/20261003_stage1      (once; must say ALL PASS: True)
        py -3.12 -m src.stage6 --run runs/20261003_stage1
"""
import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from src import certify as C
from src import splits as S
from src.util import sha256_file

DEFAULTS6 = {"scan_step": 0.005, "scan_max": 0.10, "boot": 2000, "pass_seeds": 4, "g1iii_alphas": [0.01, 0.02, 0.03], "h1_alpha": 0.02}

# Stallkamp et al., IJCNN 2011, Fig. 6 (plan section 1)
STALLKAMP = {"speed_limits": [0, 1, 2, 3, 4, 5, 7, 8], "other_prohibitory": [9, 10, 15, 16], "derestriction": [6, 32, 41, 42],
             "mandatory": list(range(33, 41)), "danger": [11] + list(range(18, 32)), "unique": [12, 13, 14, 17]}
SCHEME_ORDER = ["eps2", "eps3", "marginal", "stallkamp6", "mondrian43"]


def stallkamp_map():
    m = np.full(S.K, -1)
    for i, (_, cls) in enumerate(STALLKAMP.items()):
        m[cls] = i
    assert (m >= 0).all() and sum(len(v) for v in STALLKAMP.values()) == S.K, "Stallkamp subsets must partition the 43 classes"
    return m


def strip_test(o):
    """Remove every Test-derived key ('test', or ending in '_test' such as 'hard_test') recursively. Used by the tests: decisions must not depend on Test."""
    if isinstance(o, dict):
        return {k: strip_test(v) for k, v in o.items() if k != "test" and not k.endswith("_test")}
    if isinstance(o, list):
        return [strip_test(v) for v in o]
    return o


# ------------------------------------------------------------------ guard
def require_synthetic(run, certify_path):
    f = Path(run) / "stage6" / "synthetic_validation.json"
    assert f.exists(), f"{f} not found: run `py -3.12 -m src.certify_synth --run {run}` first (plan: no real data until the synthetic checks pass)"
    v = json.loads(f.read_text())
    assert v.get("all_pass") is True, "synthetic validation did not pass: do not touch real data (plan, Stage 6 failure modes)"
    assert v["certify_sha256"] == sha256_file(certify_path), "src/certify.py changed after the synthetic validation: rerun src.certify_synth"
    return v, sha256_file(f)


# ------------------------------------------------------------------ per-seed context
def seed_context(seed, data, cfg):
    track, y, top3 = data["track"], data["y"], data["top3"]
    n_all = int(track.max()) + 1
    n_t = np.bincount(track, minlength=n_all).astype(float)
    track_class = np.zeros(n_all, dtype=int)
    track_class[track] = y
    split, _ = S.make_split(track_class, cfg["sizes"], cfg["fractions"], seed)
    tau, _, _ = S.fit_gate(data["s"], track, split, seed, cfg["beta"])
    gated = data["s"] <= tau
    fs = split[track]
    # Labels handed to the decision code: Fit and Cert only. Test entries stay False (Test labels are read in evaluate_seed, after the decisions).
    wrong_dec = np.zeros(len(track), dtype=bool)
    lab = np.flatnonzero(fs < 2)
    wrong_dec[lab] = top3[lab, 0] != y[lab]
    return {"seed": seed, "track": track, "m": data["margin"], "pred": top3[:, 0], "n_all": n_all, "n_t": n_t, "split": split, "fs": fs, "tau": tau,
            "gated": gated, "fit_f": (fs == 0) & gated, "cert_f": (fs == 1) & gated, "wrong_dec": wrong_dec}


def build_schemes(data, ctx, delta):
    track, n_t = ctx["track"], ctx["n_t"]
    w = 1.0 / n_t[track]
    out = {}
    for name, score in (("eps2", data["log_eps12"]), ("eps3", data["log_eps3"])):
        c_star, _ = S.choose_cut(score[ctx["fit_f"]], w[ctx["fit_f"]])              # Fit only, no labels
        out[name] = {"group": (score > c_star).astype(int), "labels": ["easy", "hard"], "K": 2, "rel": True, "c_star": float(c_star)}
    out["marginal"] = {"group": np.zeros(len(track), int), "labels": ["all"], "K": 1, "rel": True}
    out["stallkamp6"] = {"group": stallkamp_map()[ctx["pred"]], "labels": list(STALLKAMP), "K": 6, "rel": False}
    out["mondrian43"] = {"group": ctx["pred"].astype(int), "labels": [f"class{i:02d}" for i in range(S.K)], "K": S.K, "rel": False}
    for s in out.values():
        s["delta_k"] = delta / s["K"]
    return out


# ------------------------------------------------------------------ Fit-side calibration (same maths as Stage 4, written generically)
def calibrate(scheme, ctx, alphas, cfg, with_rel):
    out = {}
    for g in range(scheme["K"]):
        rows = np.flatnonzero(ctx["fit_f"] & (scheme["group"] == g))
        if len(rows) == 0:
            out[g] = None
            continue
        grid = S.margin_grid(ctx["m"][rows], cfg["margin_points"], cfg["margin_top_quantile"])
        pooled, ptm = S.track_weighted_error(ctx["track"], rows, ctx["wrong_dec"], ctx["n_t"])
        al = dict(alphas)
        if with_rel and scheme["rel"]:
            if ptm > 0:
                al["rel_0.5base"] = cfg["alpha_rel_base"] * ptm
            if pooled > 0:
                al["rel_0.5base_pooled"] = cfg["alpha_rel_base"] * pooled
        orders, els = {}, {}
        for an, a in al.items():
            el = S.el_curve(ctx["track"], rows, ctx["m"], ctx["wrong_dec"], grid, a, ctx["n_all"])
            start, order = S.order_from_curve(el)
            orders[an], els[an] = (a, order), el
        out[g] = {"grid": grid, "orders": orders, "els": els, "base_ptm": ptm, "base_pooled": pooled, "fit_frames": int(len(rows))}
    return out


# ------------------------------------------------------------------ certification on Cert
def certify_scheme(scheme, cal, ctx, names):
    """res[an][g] = certify_sequence output (plus alpha, delta_k, floor) for every group g and alpha name an."""
    ct, dk = ctx, scheme["delta_k"]
    res = {an: {} for an in names}
    for g in range(scheme["K"]):
        rows = np.flatnonzero(ct["cert_f"] & (scheme["group"] == g))
        n_tracks = int(len(np.unique(ct["track"][rows])))
        for an in names:
            if cal[g] is None or an not in cal[g]["orders"]:
                res[an][g] = {"certified": False, "threshold": None, "grid_index": None, "n_tracks": n_tracks, "n_frames": int(len(rows)), "alpha": None,
                              "delta_k": dk, "floor_alpha": float(C.floor_alpha(n_tracks, dk)), "p_at_certified": None, "n_tests": 0, "note": "no Fit frames or alpha undefined"}
                continue
            a, order = cal[g]["orders"][an]
            r = C.certify_sequence(ct["track"][rows], ct["m"][rows], ct["wrong_dec"][rows], cal[g]["grid"], order, a, dk)
            tests = r.pop("tests")
            r.update(alpha=float(a), delta_k=dk, floor_alpha=float(C.floor_alpha(r["n_tracks"], dk)), n_tests=len(tests),
                     last_p=tests[-1]["p"] if tests else None, grid_points=int(len(cal[g]["grid"])), start_threshold=float(cal[g]["grid"][order[0]]))
            res[an][g] = r
    return res


# ------------------------------------------------------------------ evaluation (Cert and Test)
def evaluate_split(scheme, res_an, accept, ctx, wrong_eval, k):
    """Per-group and overall stats of one decision on split k (0 fit, 1 cert, 2 test). Labels in wrong_eval."""
    out = {"groups": {}}
    for g, lab in enumerate(scheme["labels"]):
        rows = np.flatnonzero((ctx["fs"] == k) & ctx["gated"] & (scheme["group"] == g))
        a = res_an[g]["alpha"]
        out["groups"][lab] = C.risk_stats(ctx["track"][rows], accept[rows], wrong_eval[rows], a)
    rows = np.flatnonzero(ctx["fs"] == k)
    out["overall"] = C.risk_stats(ctx["track"][rows], accept[rows], wrong_eval[rows], None)
    out["overall"]["gate_escalated_share"] = float((~ctx["gated"][rows]).mean())
    return out


def decide_seed(seed, data, cfg, c6):
    """Everything that needs only Fit and Cert labels. Returns the JSON-able decisions and the in-memory accept masks."""
    ctx = seed_context(seed, data, cfg)
    schemes = build_schemes(data, ctx, cfg["delta"])
    abs_names = {f"abs_{a}": a for a in cfg["alpha_grid"]}
    n_scan = int(round(c6["scan_max"] / c6["scan_step"]))
    scan_names = {f"scan_{(i + 1) * c6['scan_step']:.4f}": (i + 1) * c6["scan_step"] for i in range(n_scan)}
    out = {"seed": seed, "tau": ctx["tau"], "tracks": {n: int((ctx["split"] == k).sum()) for k, n in enumerate(S.SPLIT_NAMES)}, "schemes": {}}
    masks, cals = {}, {}
    for sname in SCHEME_ORDER:
        sc = schemes[sname]
        cal = calibrate(sc, ctx, abs_names, cfg, with_rel=True)
        names = sorted({an for g in cal.values() if g for an in g["orders"]}, key=lambda s: (not s.startswith("abs"), s))
        res = certify_scheme(sc, cal, ctx, names)
        cal_scan = calibrate(sc, ctx, scan_names, cfg, with_rel=False)
        scan = certify_scheme(sc, cal_scan, ctx, list(scan_names))
        entry = {"K": sc["K"], "delta_k": sc["delta_k"], "labels": sc["labels"], "c_star": sc.get("c_star"),
                 "fit_base_error": {sc["labels"][g]: ({"per_track_mean": cal[g]["base_ptm"], "pooled": cal[g]["base_pooled"], "fit_frames": cal[g]["fit_frames"]} if cal[g] else None)
                                    for g in range(sc["K"])},
                 "alphas": {}, "scan": {}}
        for an in names:
            accept = C.accept_mask(sc["group"], ctx["m"], ctx["gated"], res[an])
            masks[(sname, an)] = accept
            groups = {sc["labels"][g]: res[an][g] for g in range(sc["K"])}
            ev = evaluate_split(sc, res[an], accept, ctx, ctx["wrong_dec"], 1)
            for lab in sc["labels"]:
                groups[lab]["cert"] = ev["groups"][lab]                              # Cert-side stats (optimistic: the threshold was chosen on Cert)
            entry["alphas"][an] = {"groups": groups, "cert": ev}
        for an, a in scan_names.items():
            entry["scan"][an] = {sc["labels"][g]: {"alpha": a, "certified": scan[an][g]["certified"], "n_tracks": scan[an][g]["n_tracks"]} for g in range(sc["K"])}
        out["schemes"][sname] = entry
        cals[sname] = cal
    return out, ctx, schemes, masks, cals


def boot_group_risk(track, acc, wrong, rows, alpha, B, rng):
    """Track-bootstrap of the track-averaged selective risk (ratio of track-mean error share to track-mean accepted share) on `rows`."""
    tr = track[rows]
    if len(tr) == 0:
        return None
    _, inv = np.unique(tr, return_inverse=True)
    n = np.bincount(inv).astype(float)
    a = np.bincount(inv, weights=acc[rows].astype(float))
    e = np.bincount(inv, weights=(acc[rows] & wrong[rows]).astype(float))
    T = len(n)
    idx = rng.integers(0, T, (B, T))
    es, as_ = (e / n)[idx].mean(1), (a / n)[idx].mean(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = es / as_
    ok = np.isfinite(r)
    point = float((e / n).mean() / (a / n).mean()) if a.sum() > 0 else float("nan")
    if ok.sum() == 0:
        return {"tracks": int(T), "point": point, "ci95": [float("nan")] * 2, "share_draws_above_alpha": float("nan"), "valid_draws": 0}
    return {"tracks": int(T), "point": point, "ci95": [float(np.percentile(r[ok], 2.5)), float(np.percentile(r[ok], 97.5))],
            "share_draws_above_alpha": float((r[ok] > alpha).mean()), "valid_draws": int(ok.sum())}


def evaluate_seed(out, ctx, schemes, masks, data, c6, B):
    """Reads the Test labels. Called only after decide_seed has fixed every decision."""
    wrong_all = data["top3"][:, 0] != data["y"]
    wrong_test = np.zeros(len(wrong_all), dtype=bool)
    tmask = ctx["fs"] == 2
    wrong_test[tmask] = wrong_all[tmask]
    for sname, entry in out["schemes"].items():
        sc = schemes[sname]
        for an, ae in entry["alphas"].items():
            res_an = {g: ae["groups"][sc["labels"][g]] for g in range(sc["K"])}
            ev = evaluate_split(sc, res_an, masks[(sname, an)], ctx, wrong_test, 2)
            ae["test"] = ev
            for lab in sc["labels"]:                                                 # also copy into the group record, for reading
                ae["groups"][lab]["test"] = ev["groups"][lab]
    # H1 preview: marginal threshold, hard/easy eps2 strata, Test (primary) and Cert (descriptive)
    rng = np.random.default_rng([ctx["seed"], 6, 1])
    h1 = {}
    eps = schemes["eps2"]
    for an, ae in out["schemes"]["marginal"]["alphas"].items():
        if not an.startswith("abs_"):
            continue
        g0 = ae["groups"]["all"]
        a = g0["alpha"]
        item = {"alpha": a, "marginal_certified": g0["certified"], "threshold": g0["threshold"]}
        if g0["certified"]:
            acc = masks[("marginal", an)]
            for lab, gi in (("easy", 0), ("hard", 1)):
                for sp, k, wr in (("test", 2, wrong_test), ("cert", 1, ctx["wrong_dec"])):
                    rows = np.flatnonzero((ctx["fs"] == k) & ctx["gated"] & (eps["group"] == gi))
                    item[f"{lab}_{sp}"] = boot_group_risk(ctx["track"], acc, wr, rows, a, B, rng)
        h1[an] = item
    out["h1_preview"] = h1
    return out


# ------------------------------------------------------------------ aggregation
def aggregate(results, c6):
    agg = {"schemes": {}}
    for sname in SCHEME_ORDER:
        sa = {}
        names = sorted({an for r in results for an in r["schemes"][sname]["alphas"]}, key=lambda s: (not s.startswith("abs"), s))
        for an in names:
            tot = cert = viol = 0
            cov_t, risk_t, cov_c, per_group = [], [], [], {}
            for r in results:
                ae = r["schemes"][sname]["alphas"].get(an)
                if ae is None:
                    continue
                for lab, g in ae["groups"].items():
                    pg = per_group.setdefault(lab, {"seeds_certified": 0, "test_violations": 0, "n_tracks": [], "floor": []})
                    tot += 1
                    pg["n_tracks"].append(g["n_tracks"]); pg["floor"].append(g["floor_alpha"])
                    if g["certified"]:
                        cert += 1; pg["seeds_certified"] += 1
                        v = g["test"]["violates"]
                        viol += int(bool(v)); pg["test_violations"] += int(bool(v))
                cov_t.append(ae["test"]["overall"]["coverage"]); cov_c.append(ae["cert"]["overall"]["coverage"]); risk_t.append(ae["test"]["overall"]["cert_risk"])
            sa[an] = {"group_seed_total": tot, "group_seed_certified": cert, "certified_with_test_violation": viol,
                      "overall_cert_coverage_median": float(np.nanmedian(cov_c)), "overall_test_coverage_median": float(np.nanmedian(cov_t)),
                      "overall_test_risk_median": float(np.nanmedian(risk_t)) if not np.all(np.isnan(risk_t)) else float("nan"),
                      "per_group": per_group if len(per_group) <= 6 else {"n_groups": len(per_group)}}
        # empirical smallest certifiable alpha (descriptive)
        smallest = {}
        for lab in results[0]["schemes"][sname]["labels"]:
            vals = []
            for r in results:
                ok = [v[lab]["alpha"] for v in r["schemes"][sname]["scan"].values() if v[lab]["certified"]]
                vals.append(min(ok) if ok else None)
            smallest[lab] = vals
        agg["schemes"][sname] = {"alphas": sa, "smallest_certifiable_alpha_scan_per_seed": smallest if len(smallest) <= 6 else
                                 {"groups": len(smallest), "share_groups_with_any_certifiable_alpha": float(np.mean([any(v is not None for v in vs) for vs in smallest.values()]))}}
    # G1(iii)
    g3 = {}
    for sname in ("eps2", "eps3"):
        per_seed = []
        for r in results:
            al = r["schemes"][sname]["alphas"]
            ok = [a for a in c6["g1iii_alphas"] if f"abs_{a}" in al and all(g["certified"] for g in al[f"abs_{a}"]["groups"].values())]
            per_seed.append({"seed": r["seed"], "alphas_certifiable_in_both_strata": ok, "pass": bool(ok)})
        n = sum(x["pass"] for x in per_seed)
        g3[sname] = {"per_seed": per_seed, "seeds_passing": n, "passes": bool(n >= c6["pass_seeds"])}
    agg["g1iii"] = g3
    # H1 preview
    h1 = {}
    for an in results[0]["h1_preview"]:
        rows = [r["h1_preview"][an] for r in results]
        cert = [x for x in rows if x["marginal_certified"]]
        a = rows[0]["alpha"]
        item = {"alpha": a, "seeds_marginal_certified": len(cert)}
        for sp in ("test", "cert"):
            hard = [x[f"hard_{sp}"] for x in cert if x.get(f"hard_{sp}")]
            easy = [x[f"easy_{sp}"] for x in cert if x.get(f"easy_{sp}")]
            item[sp] = {"hard_point_above_alpha": int(sum(h["point"] > a for h in hard)), "hard_ci_lower_above_alpha": int(sum(h["ci95"][0] > a for h in hard)),
                        "easy_point_above_alpha": int(sum(e["point"] > a for e in easy)), "hard_points": [h["point"] for h in hard], "easy_points": [e["point"] for e in easy]}
        h1[an] = item
    agg["h1_preview"] = h1
    return agg


def mu_star(n, alpha, delta_k):
    """Largest mean shifted loss mu_hat that still passes the HB test at n tracks (None if even mu_hat = 0 fails, i.e. alpha is below the floor).
    Shows what the test demands: mu_hat = a R + alpha (1 - a) for coverage a and accepted risk R, so R <= (mu_star - alpha (1 - a)) / a."""
    if float(C.hb_pvalue(0.0, n, alpha)) > delta_k:
        return None
    lo, hi = 0.0, alpha
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if float(C.hb_pvalue(mid, n, alpha)) <= delta_k else (lo, mid)
    return lo


def implied_max_risk(mustar, alpha, coverage):
    if mustar is None:
        return None
    r = (mustar - alpha * (1 - coverage)) / coverage
    return r if r >= 0 else None


def needs_table(results, c6, cfg_alphas):
    """For the eps2 strata: median tracks n per stratum, and at each absolute alpha the mean shifted loss the test needs and the accepted risk that implies."""
    out = {}
    for lab in ("easy", "hard"):
        n = int(np.median([r["schemes"]["eps2"]["alphas"][f"abs_{cfg_alphas[0]}"]["groups"][lab]["n_tracks"] for r in results]))
        rows = {}
        for a in cfg_alphas:
            ms = mu_star(n, a, 0.05)
            rows[f"abs_{a}"] = {"alpha": a, "n_tracks": n, "mu_star": ms, "max_accepted_risk_at_coverage": {str(c): implied_max_risk(ms, a, c) for c in (1.0, 0.9, 0.8, 0.6)}}
        out[lab] = rows
    return out


def check_certificates(results):
    """Every certificate has p <= delta_k at the certified threshold, and any recorded failure has p > delta_k."""
    bad = []
    for r in results:
        for sname, e in r["schemes"].items():
            for an, ae in e["alphas"].items():
                for lab, g in ae["groups"].items():
                    if g["certified"] and not (g["p_at_certified"] <= g["delta_k"]):
                        bad.append((r["seed"], sname, an, lab, "p at certified threshold above delta_k"))
                    if g["certified"] and g["n_tracks"] < 1:
                        bad.append((r["seed"], sname, an, lab, "certified a group with no tracks"))
                    if g["certified"] and g["alpha"] < g["floor_alpha"] - 1e-12:
                        bad.append((r["seed"], sname, an, lab, "certified below the zero-error floor"))
                    if (not g["certified"]) and g.get("last_p") is not None and not (g["last_p"] > g["delta_k"]):
                        bad.append((r["seed"], sname, an, lab, "uncertified but last p <= delta_k"))
    return bad


# ------------------------------------------------------------------ main
def _js(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--config", default="configs/poc.yaml")
    ap.add_argument("--boot", type=int, help="override the bootstrap size (default 2000; fewer only for smoke tests)")
    ap.add_argument("--out", help="default <run>/stage6")
    a = ap.parse_args(argv)
    cfg_all = yaml.safe_load(open(a.config))
    cfg = {**S.DEFAULTS, **cfg_all.get("splits", {})}
    c6 = {**DEFAULTS6, **cfg_all.get("stage6", {})}
    if a.boot:
        c6["boot"] = a.boot
    run = Path(a.run)
    out_dir = Path(a.out) if a.out else run / "stage6"
    here = Path(__file__).resolve().parent
    synth, synth_hash = require_synthetic(run, here / "certify.py")                       # step 0: refuse unless the engine was proved
    splits_dir = run / "splits"
    summ = json.loads((splits_dir / "stage4_summary.json").read_text())
    for f, h in summ["files_sha256"].items():
        assert sha256_file(splits_dir / f) == h, f"{f} changed since Stage 4"
    data = S.load_inputs(run, cfg_all)

    results = []
    for seed in cfg["seeds"]:
        out, ctx, schemes, masks, cals = decide_seed(seed, data, cfg, c6)
        # --- reproduce Stage 4 exactly (split, tau, c*, grids, E[L] curves)
        saved = np.load(splits_dir / f"seed{seed}.npz")
        assert (saved["track_split"] == ctx["split"]).all(), f"seed {seed}: recomputed split differs from the Stage 4 file"
        sres = next(r for r in summ["seeds"] if r["seed"] == seed)
        assert abs(ctx["tau"] - sres["gate"]["tau"]) < 1e-9, "tau differs from Stage 4"
        for nm in ("eps2", "eps3"):
            assert abs(schemes[nm]["c_star"] - sres[nm]["c_star"]) < 1e-9, f"c* ({nm}) differs from Stage 4"
            for gi, lab in enumerate(("easy", "hard")):
                assert np.allclose(cals[nm][gi]["grid"], saved[f"{nm}_{lab}_margin_grid"], rtol=0, atol=1e-12), f"{nm} {lab} grid differs from Stage 4"
                for an, el in cals[nm][gi]["els"].items():
                    assert np.allclose(el, saved[f"{nm}_{lab}_EL_{an}"], rtol=0, atol=1e-12), f"{nm} {lab} E[L] ({an}) differs from Stage 4"
        out = evaluate_seed(out, ctx, schemes, masks, data, c6, c6["boot"])               # Test labels are read from here on
        results.append(out)

    bad = check_certificates(results)
    assert not bad, f"certificate consistency check failed: {bad[:5]}"
    agg = aggregate(results, c6)
    agg["needs_table_eps2"] = needs_table(results, c6, cfg["alpha_grid"])
    summary = {"constants_used": {**cfg, "stage6": c6}, "synthetic_validation_sha256": synth_hash, "certify_sha256": synth["certify_sha256"],
               "checks": {"stage4_split_tau_cstar_grids_el_reproduced": True, "certificates_consistent": True, "synthetic_all_pass": True},
               "aggregate": agg, "seeds": results,
               "notes": ["Test labels are read only after every decision is fixed (evaluate_seed).",
                         "Cert-side coverage/risk are optimistic (the threshold was chosen on Cert); only the Test columns are out-of-sample.",
                         "A Test 'violation' (mean_L > alpha) is one noisy draw on ~130 tracks per stratum, not a refutation or a confirmation of the certificate.",
                         "The smallest-certifiable-alpha scan is descriptive; each alpha's certificate is valid by itself, the smallest one picked after looking is not.",
                         "Mondrian-43 and Stallkamp-6 groups are the PREDICTED class / predicted-class subset (label-free, deployable)."]}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "stage6_results.json").write_text(json.dumps(summary, indent=2, default=_js))
    summary["files_sha256"] = {"stage6_results.json": sha256_file(out_dir / "stage6_results.json")}
    report(summary, c6)
    return summary


# ------------------------------------------------------------------ printing
def _pct(x, d=2):
    return "  n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:.{d}f}%"


def report(S6, c6):
    agg, seeds = S6["aggregate"], S6["seeds"]
    print("=== Stage 6: certification on real data (synthetic validation verified; Stage 4 reproduced) ===")
    print(f"\nG1(iii): some absolute alpha in {c6['g1iii_alphas']} certifiable in BOTH strata, in >= {c6['pass_seeds']} of {len(seeds)} reshuffles")
    for sname in ("eps2", "eps3"):
        g = agg["g1iii"][sname]
        print(f"  {sname}: {g['seeds_passing']} of {len(seeds)} reshuffles -> {'PASS' if g['passes'] else 'FAIL'}   per seed: " +
              "; ".join(f"seed {x['seed']}: {x['alphas_certifiable_in_both_strata'] or 'none'}" for x in g["per_seed"]))
    print("\nPer scheme and alpha, over the 5 reshuffles (group-seeds certified / total; Test: how many certified groups had mean_L > alpha on Test)")
    print("  scheme       alpha            certified  Test viol.  Cert cov (median)  Test cov (median)  Test risk overall (median)")
    for sname in SCHEME_ORDER:
        for an, x in agg["schemes"][sname]["alphas"].items():
            print(f"  {sname:<11}  {an:<18} {x['group_seed_certified']:>4}/{x['group_seed_total']:<5} {x['certified_with_test_violation']:>6}      "
                  f"{_pct(x['overall_cert_coverage_median']):>9}          {_pct(x['overall_test_coverage_median']):>9}        {_pct(x['overall_test_risk_median']):>9}")
    print("\nPer reshuffle, eps2 (PRIMARY), absolute alpha:  group: n tracks | floor | certified threshold | Cert cov | Test cov | Test risk (mean_L > alpha?)")
    for r in seeds:
        for an, ae in r["schemes"]["eps2"]["alphas"].items():
            if not an.startswith("abs_"):
                continue
            parts = []
            for lab, g in ae["groups"].items():
                t = g["test"]
                tag = f"thr {g['threshold']:.2f} cov {_pct(g['cert']['coverage'] if 'cert' in g else float('nan'),1)}" if g["certified"] else "UNCERTIFIABLE"
                parts.append(f"{lab}: n={g['n_tracks']} floor {_pct(g['floor_alpha'])} | {tag} | Test cov {_pct(t['coverage'],1)} risk {_pct(t['cert_risk'])}" + (" VIOL" if t["violates"] else ""))
            print(f"  seed {r['seed']} {an:<10} " + "  ||  ".join(parts))
    print("\nFloor table (zero errors, everything accepted): smallest certifiable alpha = 1 - delta_k^(1/n)")
    ref = f"abs_{c6['h1_alpha']}" if f"abs_{c6['h1_alpha']}" in seeds[0]["schemes"]["eps2"]["alphas"] else next(iter(seeds[0]["schemes"]["eps2"]["alphas"]))   # n and floor do not depend on alpha
    for r in seeds:
        e2 = r["schemes"]["eps2"]["alphas"][ref]["groups"]
        e3 = r["schemes"]["eps3"]["alphas"][ref]["groups"]
        print(f"  seed {r['seed']}: eps2 " + ", ".join(f"{lab} n={g['n_tracks']} floor {_pct(g['floor_alpha'])}" for lab, g in e2.items()) + " | eps3 " +
              ", ".join(f"{lab} n={g['n_tracks']} floor {_pct(g['floor_alpha'])}" for lab, g in e3.items()))
    for sname in ("marginal", "stallkamp6", "mondrian43"):
        gg = [g for r in seeds for g in r["schemes"][sname]["alphas"][ref]["groups"].values()]
        fl = [g["floor_alpha"] for g in gg if g["n_tracks"] > 0]
        print(f"  {sname}: groups x seeds {len(gg)}, tracks per group median {int(np.median([g['n_tracks'] for g in gg]))}, floor median {_pct(float(np.median(fl)))} (min {_pct(min(fl))}, max {_pct(max(fl))})")
    if "needs_table_eps2" in agg:
        print("\nWhat the test demands (eps2 strata, median n): the largest mean shifted loss mu* that passes, and the accepted risk R that allows at coverage a  [mu = a R + alpha (1 - a)]")
        for lab, rows in agg["needs_table_eps2"].items():
            for an, x in rows.items():
                cov = x["max_accepted_risk_at_coverage"]
                print(f"  {lab:<5} n={x['n_tracks']:<4} alpha {_pct(x['alpha'],0):>3}: mu* = {'below floor' if x['mu_star'] is None else _pct(x['mu_star'],3)} | max accepted risk at coverage 100%: {_pct(cov['1.0'])}, 90%: {_pct(cov['0.9'])}, 80%: {_pct(cov['0.8'])}, 60%: {_pct(cov['0.6'])}")
    print("\nSmallest certifiable alpha in the descriptive scan (step 0.5%; None = nothing up to 10%), per reshuffle")
    for sname in SCHEME_ORDER:
        sm = agg["schemes"][sname]["smallest_certifiable_alpha_scan_per_seed"]
        if "groups" in sm:
            print(f"  {sname}: {sm['groups']} groups; share of groups with any certifiable alpha <= 10% in some reshuffle: {_pct(sm['share_groups_with_any_certifiable_alpha'],0)}")
        else:
            print(f"  {sname}: " + "; ".join(f"{lab}: {[None if v is None else round(100 * v, 1) for v in vs]}" for lab, vs in sm.items()))
    print(f"\nH1 preview: marginal threshold (K = 1), applied to the eps2 strata; Test, track-cluster bootstrap B = {c6['boot']}")
    for an, h in agg["h1_preview"].items():
        t = h["test"]
        print(f"  {an}: marginal certified in {h['seeds_marginal_certified']} of {len(seeds)}; hard-stratum Test risk > alpha in {t['hard_point_above_alpha']} of them "
              f"(95% lower bound > alpha in {t['hard_ci_lower_above_alpha']}); easy-stratum above alpha in {t['easy_point_above_alpha']}; "
              f"hard points {[round(100 * p, 2) for p in t['hard_points']]}%, easy points {[round(100 * p, 2) for p in t['easy_points']]}%")
    print("\nNotes: " + " | ".join(S6["notes"]))


if __name__ == "__main__":
    main()
