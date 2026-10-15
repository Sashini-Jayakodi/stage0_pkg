"""Stage 5: G1(i), G1(ii), rho (ICC) and design effect, rank-of-error, eps3 ablation, fragment counts (plan section 6, Stage 5).

This is the FIRST stage that reads Cert labels. Everything frozen in the LOG.md pre-registration block (and Addenda A, B) is used as written:
  * strata = c* from the seed's Fit split (recomputed here and asserted equal to the Stage 4 value), gate tau from Fit;
  * labels = the Gaussian head's top-1 on gated-in frames, no margin threshold;
  * G1(i): hard-minus-easy gap, track-cluster bootstrap over CERT tracks, B = 2000, lower bound = 2.5th percentile > 0,
           passes if that holds in >= 4 of 5 reshuffles. PRIMARY = per-track mean (decision 2026-10-10); pooled = sensitivity.
           If the two definitions disagree on the verdict, both are reported with a caveat (Addendum A), per-track stays primary.
  * G1(ii): hard-group frame errors >= 30 AND distinct tracks with a hard error >= 10; passes if the MEDIAN over the 5 reshuffles meets both.
  * eps3 (top-3) is run side by side as the pre-registered first fallback.
Test is never touched: every label read below is a Cert label, except the one dataset-level rho over all 39,209 frames
(descriptive, split-free, like the overall accuracy that is already reported).

Also produced: rho (one-way ANOVA ICC, unequal cluster sizes) with a track bootstrap CI, design effect and effective sample size;
rank of the true class among the wrong frames (2nd / 3rd / 4th+); and, with NO errors, how many gated-in frames each track
contributes to each stratum on Fit and Cert (fragment counts).

Usage:  py -3.12 -m src.stage5 --run runs/20261003_stage1            (needs src.splits to have been run)
"""
import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from src import fitdiag as F
from src import splits as S
from src.util import sha256_file

# Plan constants (section 2 / pre-registration). `stage5:` block of configs/poc.yaml may override, but should not.
DEFAULTS5 = {"boot": 2000, "min_hard_errors": 30, "min_hard_tracks": 10, "pass_seeds": 4, "frames_per_track": 30,
             "icc_low": -0.05}
FRAG_BINS = [(1, 1), (2, 2), (3, 5), (6, 10), (11, 20), (21, 30)]


# ------------------------------------------------------------------ rho: one-way ANOVA ICC
def icc_anova(W, n, e):
    """ANOVA estimator of the intra-class correlation of a 0/1 indicator, clusters = tracks, unequal sizes.
    n, e: per-track frame count and error count (tracks with n > 0 only). W: (R, T) integer multiplicities, one row per
    (bootstrap) sample; a row of ones is the plain estimate. A track drawn twice is two clusters.
        MSB = sum n_i (ybar_i - ybar)^2 / (k-1)      MSW = sum_ij (y_ij - ybar_i)^2 / (N-k)
        n0  = (N - sum n_i^2 / N) / (k-1)            rho = (MSB - MSW) / (MSB + (n0-1) MSW)"""
    W = np.atleast_2d(W).astype(float)
    k, N, S_ = W.sum(1), (W * n).sum(1), (W * e).sum(1)
    between = (W * e ** 2 / n).sum(1) - S_ ** 2 / N
    within = (W * (e - e ** 2 / n)).sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        msb, msw = between / (k - 1), within / (N - k)
        n0 = (N - (W * n ** 2).sum(1) / N) / (k - 1)
        return (msb - msw) / (msb + (n0 - 1) * msw)


def rho_block(track, wrong, sel, n_all, B, rng, m_ref=30):
    """rho with a track-cluster bootstrap CI for the frames in `sel`, plus design effect and effective sample size.
    design effect (plan) = 1 + (m_ref - 1) rho with m_ref = 30; design_effect_actual uses the mean frames per track in this
    selection (a stratum holds fewer frames per track, so the plan's m = 30 would overstate it)."""
    n, e = F.track_stats(track, n_all, sel, wrong)
    has = n > 0
    n, e = n[has], e[has]
    T, N = len(n), float(n.sum())
    out = {"frames": int(N), "tracks": int(T), "errors": int(e.sum()), "mean_frames_per_track": N / T}
    if T < 3 or e.sum() == 0 or e.sum() == N:
        out.update(rho=float("nan"), rho_ci95=[float("nan")] * 2, design_effect_m30=float("nan"),
                   design_effect_actual=float("nan"), effective_sample_size=float("nan")); return out
    W = np.vstack([np.ones(T), rng.multinomial(T, np.full(T, 1 / T), size=B)])
    r = icc_anova(W, n, e)
    rho, rb = float(r[0]), r[1:][np.isfinite(r[1:])]
    de30, dea = 1 + (m_ref - 1) * rho, 1 + (N / T - 1) * rho
    out.update(rho=rho, rho_ci95=[float(np.percentile(rb, 2.5)), float(np.percentile(rb, 97.5))],
               design_effect_m30=de30, design_effect_actual=dea, effective_sample_size=N / dea if dea > 0 else float("nan"))
    return out


# ------------------------------------------------------------------ fragment counts (no labels)
def fragment_counts(track, sel, n_all, tracks):
    """Gated-in frames per track in one stratum, over the given tracks. Counts only, no errors, no labels."""
    cnt = np.bincount(track[sel], minlength=n_all)[tracks]
    has = cnt > 0
    c = cnt[has]
    out = {"tracks_with_frames": int(has.sum()), "frames": int(c.sum()),
           "mean": float(c.mean()) if len(c) else float("nan"), "median": float(np.median(c)) if len(c) else float("nan"),
           "share_tracks_le2_frames": float((c <= 2).mean()) if len(c) else float("nan"),
           "bins": {f"{a}" if a == b else f"{a}-{b}": int(((c >= a) & (c <= b)).sum()) for a, b in FRAG_BINS}}
    return out, cnt


# ------------------------------------------------------------------ rank of the true class
def rank_of_error(top3, y, rows):
    """For the (wrong) frames in `rows`: where is the true class in the posterior ranking? Only the top 3 are stored, so
    rank 2, rank 3, or '4+' (true class outside the top 3)."""
    t, yy = top3[rows], y[rows]
    r2, r3 = t[:, 1] == yy, t[:, 2] == yy
    n = len(rows)
    out = {"errors": int(n), "rank2": int(r2.sum()), "rank3": int((r3 & ~r2).sum()), "rank4plus": int((~r2 & ~r3).sum())}
    out["share_rank2"] = out["rank2"] / n if n else float("nan")
    return out


# ------------------------------------------------------------------ per seed
def analyse_seed(seed, data, cfg, c5):
    track, y, top3 = data["track"], data["y"], data["top3"]
    n_all = int(track.max()) + 1
    n_t = np.bincount(track, minlength=n_all).astype(float)
    track_class = np.zeros(n_all, dtype=int); track_class[track] = y
    split, _ = S.make_split(track_class, cfg["sizes"], cfg["fractions"], seed)
    tau, _, _ = S.fit_gate(data["s"], track, split, seed, cfg["beta"])
    gated = data["s"] <= tau
    fit_tracks, cert_tracks = np.flatnonzero(split == 0), np.flatnonzero(split == 1)
    fit_f, cert_f = (split[track] == 0) & gated, (split[track] == 1) & gated
    # Labels are read for CERT frames only. Every other frame keeps wrong = False and is never used below.
    wrong = np.zeros(len(track), dtype=bool)
    cr = np.flatnonzero(cert_f)
    wrong[cr] = top3[cr, 0] != y[cr]
    B = c5["boot"]
    res = {"seed": seed, "tau": tau, "cert_tracks": int(len(cert_tracks)), "cert_frames_gated_in": int(cert_f.sum()),
           "cert_errors_all_gated_in": int(wrong.sum()), "cert_error_rate_frame": float(wrong[cr].mean()),
           "rank_of_error_cert_all": rank_of_error(top3, y, cr[wrong[cr]]),
           "rho_cert_all": rho_block(track, wrong, cert_f, n_all, B, np.random.default_rng([seed, 4, 9]))}
    for vi, (name, score) in enumerate((("eps2", data["log_eps12"]), ("eps3", data["log_eps3"]))):
        c_star, fit_easy = S.choose_cut(score[fit_f], (1.0 / n_t[track])[fit_f])      # from Fit, no labels
        easy, hard = score <= c_star, score > c_star
        g = F.bootstrap_gap(track, wrong, gated, score, c_star, cert_tracks, n_t, B, np.random.default_rng([seed, 4, vi, 0]))
        g2 = F.bootstrap_gap(track, wrong, gated, score, c_star, cert_tracks, n_t, 2 * B, np.random.default_rng([seed, 4, vi, 1]))
        gap = {}
        for k in ("per_track", "pooled"):
            lb, lb2 = g[k]["ci95"][0], g2[k]["ci95"][0]
            gap[k] = {**g[k], "lower_bound": lb, "passes": bool(lb > 0), "lower_bound_2B": lb2,
                      "passes_2B": bool(lb2 > 0), "stable_when_B_doubled": bool((lb > 0) == (lb2 > 0))}
        strata, frag = {}, {}
        for lab, sel in (("easy", easy), ("hard", hard)):
            cs = cert_f & sel
            rows = np.flatnonzero(cs)
            pooled, ptm = S.track_weighted_error(track, rows, wrong, n_t)
            err_rows = rows[wrong[rows]]
            strata[lab] = {"cert_frames": int(len(rows)), "cert_tracks": int(len(np.unique(track[rows]))),
                           "errors": int(len(err_rows)), "tracks_with_error": int(len(np.unique(track[err_rows]))),
                           "base_error_per_track_mean": ptm, "base_error_pooled": pooled,
                           "rank_of_error": rank_of_error(top3, y, err_rows),
                           "rho": rho_block(track, wrong, cs, n_all, B, np.random.default_rng([seed, 4, vi, 2 if lab == "easy" else 3]))}
            frag[f"cert_{lab}"], _ = fragment_counts(track, cs, n_all, cert_tracks)
            frag[f"fit_{lab}"], _ = fragment_counts(track, fit_f & sel, n_all, fit_tracks)     # counts only, no labels
        both = ((np.bincount(track[cert_f & easy], minlength=n_all) > 0) & (np.bincount(track[cert_f & hard], minlength=n_all) > 0))[cert_tracks]
        frag["cert_tracks_in_both_strata"] = int(both.sum())
        h = strata["hard"]
        res[name] = {"c_star": c_star, "fit_easy_mass_fraction": fit_easy, "gap": gap, "strata": strata, "fragments": frag,
                     "g1ii": {"hard_errors": h["errors"], "hard_tracks_with_error": h["tracks_with_error"],
                              "meets_both_this_seed": bool(h["errors"] >= c5["min_hard_errors"] and h["tracks_with_error"] >= c5["min_hard_tracks"])}}
    return res, split


# ------------------------------------------------------------------ verdicts
def aggregate(results, c5):
    """Apply the pre-registered aggregation rules. Nothing here is tunable."""
    need, out = c5["pass_seeds"], {}
    for name in ("eps2", "eps3"):
        n_pt = sum(r[name]["gap"]["per_track"]["passes"] for r in results)
        n_po = sum(r[name]["gap"]["pooled"]["passes"] for r in results)
        v_pt, v_po = n_pt >= need, n_po >= need
        he = [r[name]["g1ii"]["hard_errors"] for r in results]
        ht = [r[name]["g1ii"]["hard_tracks_with_error"] for r in results]
        med_e, med_t = float(np.median(he)), float(np.median(ht))
        out[name] = {
            "g1i_per_track_seeds_passing": int(n_pt), "g1i_per_track_pass": bool(v_pt),
            "g1i_pooled_seeds_passing": int(n_po), "g1i_pooled_pass": bool(v_po),
            "g1i_definitions_disagree": bool(v_pt != v_po),                       # Addendum A: report both with a caveat
            "g1i_mean_gap_per_track": float(np.mean([r[name]["gap"]["per_track"]["gap"] for r in results])),
            "g1i_mean_gap_pooled": float(np.mean([r[name]["gap"]["pooled"]["gap"] for r in results])),
            "g1i_stable_when_B_doubled_all_seeds": bool(all(r[name]["gap"][k]["stable_when_B_doubled"] for r in results for k in ("per_track", "pooled"))),
            "g1ii_median_hard_errors": med_e, "g1ii_median_hard_tracks": med_t,
            "g1ii_pass": bool(med_e >= c5["min_hard_errors"] and med_t >= c5["min_hard_tracks"])}
    return out


# ------------------------------------------------------------------ main
def _js(o):
    """json fallback for numpy scalars (keeps booleans as booleans, not 0/1 floats)."""
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    raise TypeError(type(o))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--config", default="configs/poc.yaml")
    ap.add_argument("--boot", type=int, help="override B (default 2000, the pre-registered value; use fewer only for smoke tests)")
    ap.add_argument("--out", help="default <run>/splits")
    a = ap.parse_args(argv)
    cfg_all = yaml.safe_load(open(a.config))
    cfg = {**S.DEFAULTS, **cfg_all.get("splits", {})}
    c5 = {**DEFAULTS5, **cfg_all.get("stage5", {})}
    if a.boot:
        c5["boot"] = a.boot
    run = Path(a.run)
    out = Path(a.out) if a.out else run / "splits"
    summ = json.loads((out / "stage4_summary.json").read_text())
    for f, h in summ["files_sha256"].items():
        assert sha256_file(out / f) == h, f"{f} changed since Stage 4"
    data = S.load_inputs(run, cfg_all)
    n_all = int(data["track"].max()) + 1
    n_t = np.bincount(data["track"], minlength=n_all).astype(float)

    # --- dataset-level checks that use all frames (descriptive, split-free)
    wrong_all = data["top3"][:, 0] != data["y"]
    checks = {"head_error_all_frames": float(wrong_all.mean())}
    hs = json.loads((run / "head_emb" / "head_summary.json").read_text())
    if "train_release" in hs:
        checks["reconciles_with_stage3_head_accuracy"] = bool(abs((1 - wrong_all.mean()) - hs["train_release"]["head_accuracy"]) < 1e-9)
    rho_all = rho_block(data["track"], wrong_all, np.ones(len(wrong_all), bool), n_all, c5["boot"], np.random.default_rng([99, 4]))
    checks["rho_all_in_range"] = bool(c5["icc_low"] <= rho_all["rho"] <= 1) if np.isfinite(rho_all["rho"]) else None

    results = []
    for seed in cfg["seeds"]:
        res, split = analyse_seed(seed, data, cfg, c5)
        saved = np.load(out / f"seed{seed}.npz")["track_split"]
        assert (saved == split).all(), f"seed {seed}: recomputed split differs from the Stage 4 file"
        sres = next(r for r in summ["seeds"] if r["seed"] == seed)
        for nm in ("eps2", "eps3"):
            assert abs(res[nm]["c_star"] - sres[nm]["c_star"]) < 1e-9, f"c* ({nm}) differs from Stage 4"
        assert abs(res["tau"] - sres["gate"]["tau"]) < 1e-9, "tau differs from Stage 4"
        results.append(res)
    agg = aggregate(results, c5)
    rhos = [x for r in results for x in (r["rho_cert_all"]["rho"], r["eps2"]["strata"]["easy"]["rho"]["rho"], r["eps2"]["strata"]["hard"]["rho"]["rho"])]
    checks["rho_cert_all_in_range"] = bool(all(c5["icc_low"] <= v <= 1 for v in rhos if np.isfinite(v)))
    summary = {"bootstrap_draws": c5["boot"], "constants_used": {**cfg, **{"stage5": c5}}, "dataset_rho_all_frames": rho_all,
               "checks": checks, "aggregate": agg, "seeds": results,
               "note": "CERT labels only (plus the dataset-level rho over all frames). Test untouched. Per-track mean is the PRIMARY G1(i) definition."}
    (out / "stage5_results.json").write_text(json.dumps(summary, indent=2, default=_js))
    summary["files_sha256"] = {"stage5_results.json": sha256_file(out / "stage5_results.json")}
    report(summary, c5)
    return summary


def report(S5, c5):
    res, agg = S5["seeds"], S5["aggregate"]
    print(f"=== Stage 5 (Cert labels; B = {S5['bootstrap_draws']} track-cluster bootstrap draws) ===")
    for name, title in (("eps2", "eps (PRIMARY)"), ("eps3", "eps3 (pre-registered first fallback)")):
        print(f"\n--- G1(i) hard-minus-easy gap on Cert, {title}  [percentage points] ---")
        print("seed | per-track: easy   hard   gap    SE   95% lower  pass | pooled: gap    95% lower  pass | B->2B stable")
        for r in res:
            p, q = r[name]["gap"]["per_track"], r[name]["gap"]["pooled"]
            print(f"  {r['seed']}  |   {100*p['easy']:5.2f}  {100*p['hard']:5.2f}  {100*p['gap']:+5.2f}  {100*p['boot_se']:4.2f}  {100*p['lower_bound']:+6.2f}    "
                  f"{'YES' if p['passes'] else 'no ':3s} |         {100*q['gap']:+5.2f}  {100*q['lower_bound']:+6.2f}    {'YES' if q['passes'] else 'no ':3s} |"
                  f"   {p['stable_when_B_doubled'] and q['stable_when_B_doubled']}")
        a = agg[name]
        print(f"G1(i) per-track (primary): {a['g1i_per_track_seeds_passing']} of {len(res)} reshuffles pass -> {'PASS' if a['g1i_per_track_pass'] else 'FAIL'}"
              f" (rule: >= {c5['pass_seeds']}) | pooled (sensitivity): {a['g1i_pooled_seeds_passing']} of {len(res)} -> {'PASS' if a['g1i_pooled_pass'] else 'FAIL'}"
              f" | mean gap {100*a['g1i_mean_gap_per_track']:+.2f} pp (pooled {100*a['g1i_mean_gap_pooled']:+.2f})")
        if a["g1i_definitions_disagree"]:
            print("  CAVEAT (Addendum A): the two definitions disagree on the G1(i) verdict. Report both; per-track stays primary.")
        print(f"\n--- G1(ii) hard group on Cert, {title} ---")
        for r in res:
            g = r[name]["g1ii"]
            print(f"  seed {r['seed']}: hard-group frame errors {g['hard_errors']}, distinct tracks with a hard error {g['hard_tracks_with_error']}"
                  f" | meets both: {g['meets_both_this_seed']}")
        print(f"G1(ii): median errors {a['g1ii_median_hard_errors']:.0f} (need >= {c5['min_hard_errors']}), median tracks {a['g1ii_median_hard_tracks']:.0f}"
              f" (need >= {c5['min_hard_tracks']}) -> {'PASS' if a['g1ii_pass'] else 'FAIL'}")

    ra = S5["dataset_rho_all_frames"]
    print("\n--- rho (ICC of the frame-error indicator within tracks) and design effect ---")
    print(f"all {ra['frames']} frames in {ra['tracks']} tracks: rho {ra['rho']:.3f} [{ra['rho_ci95'][0]:.3f}, {ra['rho_ci95'][1]:.3f}] | design effect (m=30) "
          f"{ra['design_effect_m30']:.1f} | effective sample size {ra['effective_sample_size']:.0f} frames")
    print("per reshuffle, gated-in Cert frames:  seed | all: rho [CI]  DE(actual)  ESS | eps easy rho | eps hard rho")
    for r in res:
        c, e, h = r["rho_cert_all"], r["eps2"]["strata"]["easy"]["rho"], r["eps2"]["strata"]["hard"]["rho"]
        print(f"  {r['seed']} | {c['rho']:.3f} [{c['rho_ci95'][0]:.3f},{c['rho_ci95'][1]:.3f}]  {c['design_effect_actual']:.1f}  {c['effective_sample_size']:.0f}"
              f" | {e['rho']:.3f} | {h['rho']:.3f}")

    print("\n--- rank of the true class among wrong gated-in Cert frames (mean over seeds) ---")
    tot = {k: sum(r["rank_of_error_cert_all"][k] for r in res) for k in ("errors", "rank2", "rank3", "rank4plus")}
    print(f"  2nd {tot['rank2']/tot['errors']:.1%} | 3rd {tot['rank3']/tot['errors']:.1%} | 4th or lower {tot['rank4plus']/tot['errors']:.1%} (of {tot['errors']} errors, 5 reshuffles overlap)")

    print("\n--- fragment counts: gated-in frames per track per stratum, eps, seed 0 (counts only, no errors) ---")
    fr = res[0]["eps2"]["fragments"]
    for k in ("fit_easy", "fit_hard", "cert_easy", "cert_hard"):
        f = fr[k]
        print(f"  {k:10s}: {f['tracks_with_frames']:4d} tracks, mean {f['mean']:5.1f} frames/track, median {f['median']:.0f}, "
              f"{f['share_tracks_le2_frames']:.1%} of tracks have <= 2 frames | bins {f['bins']}")
    print(f"  Cert tracks with frames in BOTH strata: {fr['cert_tracks_in_both_strata']} of {res[0]['cert_tracks']}")

    print("\n--- checks ---")
    for k, v in S5["checks"].items():
        print(f"  {k}: {v}")
    print("\nThese are the pre-registered rules applied as written. Nothing was tuned. Log the numbers in LOG.md exactly as printed.")


if __name__ == "__main__":
    main()
