"""Stage 4 add-on: Fit-only diagnostic (information only; changes nothing that is frozen).

Two questions, answered on the FIT split of each reshuffle only. No Cert or Test error is computed here:
every error number comes from frames of Fit tracks (the code selects Fit rows before it touches the labels,
and a unit test flips all non-Fit labels to prove the output does not move).

A. How noisy is the hard-minus-easy gap?
   Track-cluster bootstrap (resample Fit tracks, B = 2000) of the gap in base error, for eps and eps3,
   with the per-track mean (primary) and the pooled rate (sensitivity). Then a rough read-across to Cert:
   Cert has 784 tracks vs 261, so its standard error is about SE_fit * sqrt(261/784) IF the same noise level applies.
B. Where do the errors sit along the eps axis?
   Gated-in Fit frames are cut into 8 slices of equal track-weighted mass by log eps (whole class pairs stay
   together). For each slice: error rates and share of all Fit errors. Shows whether the median cut c* is too coarse.

Rule: this is a rehearsal, not a tuning step. c*, delta, beta, strata, seeds and the base-error definition stay as
committed. If the numbers make you want a different design, that is a separate, declared revision (R7) committed BEFORE
Stage 5, never a quiet change.

Usage:  py -3.12 -m src.fitdiag --run runs/20261003_stage1            (needs src.splits to have been run)
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import yaml
from scipy.stats import binom, norm, spearmanr

from src import splits as S
from src.util import sha256_file

N_SLICES = 8


# ------------------------------------------------------------------ helpers
def track_stats(track, n_tracks, sel, wrong):
    """Per track: number of selected frames and number of wrong ones among them."""
    n = np.bincount(track[sel], minlength=n_tracks).astype(float)
    e = np.bincount(track[sel], weights=wrong[sel].astype(float), minlength=n_tracks)
    return n, e


def stratum_errors(W, n, e, n_t):
    """Base error for each row of the weight matrix W (rows = bootstrap draws, columns = tracks; a row of ones is the
    plain estimate). per_track = weighted mean over tracks with >= 1 frame in the stratum of that track's error rate
    (PRIMARY, as committed). pooled = frames weighted 1/n_t (sensitivity)."""
    has = n > 0
    rate = np.where(has, e / np.where(has, n, 1.0), 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        per_track = (W * rate).sum(1) / (W * has).sum(1)
        pooled = (W * (e / n_t)).sum(1) / (W * (n / n_t)).sum(1)
    return per_track, pooled


def bootstrap_gap(track, wrong, gated, score, c_star, fit_tracks, n_t, B, rng):
    """Hard-minus-easy gap for the Fit tracks, point estimate and track-cluster bootstrap."""
    n_all = len(n_t)
    pos = np.full(n_all, -1); pos[fit_tracks] = np.arange(len(fit_tracks))
    fit_f = (pos[track] >= 0) & gated                     # FIT rows only; labels of other rows are never read
    stats = {}
    for lab, sel in (("easy", score <= c_star), ("hard", score > c_star)):
        n, e = track_stats(track, n_all, fit_f & sel, wrong)
        stats[lab] = (n[fit_tracks], e[fit_tracks])
    nt_fit = n_t[fit_tracks]
    W = np.vstack([np.ones(len(fit_tracks)),
                   rng.multinomial(len(fit_tracks), np.full(len(fit_tracks), 1 / len(fit_tracks)), size=B).astype(float)])
    out = {}
    for name, idx in (("per_track", 0), ("pooled", 1)):
        easy = stratum_errors(W, *stats["easy"], nt_fit)[idx]
        hard = stratum_errors(W, *stats["hard"], nt_fit)[idx]
        gap = hard - easy
        g0, gb = float(gap[0]), gap[1:][np.isfinite(gap[1:])]
        out[name] = {"easy": float(easy[0]), "hard": float(hard[0]), "gap": g0, "boot_se": float(gb.std(ddof=1)),
                     "ci95": [float(np.percentile(gb, 2.5)), float(np.percentile(gb, 97.5))],
                     "frac_boot_gap_positive": float((gb > 0).mean())}
    return out


def cert_readacross(gap, se_fit, n_fit=261, n_cert=784):
    """ROUGH heuristic. Assumes Cert noise = Fit noise * sqrt(n_fit/n_cert) and that the true gap equals the Fit gap
    (which is itself a noisy, possibly optimistic estimate). The G1(i) test needs the 2.5th-percentile bound > 0,
    i.e. roughly gap > 1.96 SE."""
    se_c = se_fit * math.sqrt(n_fit / n_cert)
    return {"se_cert_rough": se_c, "gap_needed_rough": 1.96 * se_c,
            "gap_over_se_cert": gap / se_c if se_c > 0 else float("nan"),
            "rough_pass_probability": float(norm.cdf(gap / se_c - 1.96)) if se_c > 0 else float("nan")}


def rough_g1i(results, name, n_seeds=5, need=4):
    """Very rough chance that G1(i) passes in >= 4 of 5 reshuffles: uses the MEAN Fit gap and MEAN Fit SE over seeds (more
    stable than per-seed values), assumes the true gap equals the Fit gap and that Cert noise scales as 1/sqrt(tracks), and
    treats the reshuffles as independent (they are not; real outcomes are more correlated, i.e. more all-or-nothing)."""
    g = np.mean([r[name]["gap"]["per_track"]["gap"] for r in results])
    se_c = np.mean([r[name]["gap"]["per_track"]["se_cert_rough"] for r in results])
    p = float(norm.cdf(g / se_c - 1.96))
    return {"mean_gap": float(g), "mean_se_cert_rough": float(se_c), "p_single_reshuffle": p,
            "p_at_least_4_of_5": float(binom.sf(need - 1, n_seeds, p))}


def slice_by_mass(score, weight, n_slices=N_SLICES):
    """Slice index 0..n_slices-1 by cumulative weighted mass of `score` (ascending). Frames with an identical score
    (same class pair) always share a slice, so slices are only approximately equal in mass."""
    vals, inv = np.unique(score, return_inverse=True)
    mass = np.bincount(inv, weights=weight)
    cum = np.cumsum(mass)
    mid = (cum - mass / 2) / cum[-1]
    return np.minimum((mid * n_slices).astype(int), n_slices - 1)[inv]


def slice_table(track, wrong, gated, score, split, n_t, n_slices=N_SLICES):
    """Error by confusability slice on gated-in Fit frames."""
    n_all = len(n_t)
    fit_f = (split[track] == 0) & gated
    rows = np.flatnonzero(fit_f)
    w = 1.0 / n_t[track[rows]]
    sl = slice_by_mass(score[rows], w, n_slices)
    w_err_total = wrong[rows].sum()
    out = []
    for k in range(n_slices):
        r = rows[sl == k]
        if len(r) == 0:
            out.append({"slice": k, "frames": 0}); continue
        n, e = track_stats(track, n_all, np.isin(np.arange(len(track)), r), wrong)
        per_track, pooled = stratum_errors(np.ones((1, n_all)), n, e, n_t)
        out.append({"slice": k, "frames": int(len(r)), "tracks": int((n > 0).sum()),
                    "mass_share": float((1.0 / n_t[track[r]]).sum() / w.sum()),
                    "log_eps_min": float(score[r].min()), "log_eps_max": float(score[r].max()),
                    "err_per_track": float(per_track[0]), "err_pooled": float(pooled[0]),
                    "share_of_fit_errors": float(wrong[r].sum() / w_err_total) if w_err_total else float("nan")})
    return out


def diagnose_seed(seed, data, cfg, B):
    track, y = data["track"], data["y"]
    n_tracks = int(track.max()) + 1
    n_t = np.bincount(track, minlength=n_tracks).astype(float)
    track_class = np.zeros(n_tracks, dtype=int); track_class[track] = y
    split, _ = S.make_split(track_class, cfg["sizes"], cfg["fractions"], seed)
    tau, _, _ = S.fit_gate(data["s"], track, split, seed, cfg["beta"])
    gated = data["s"] <= tau
    wrong = data["top3"][:, 0] != y
    fit_tracks = np.flatnonzero(split == 0)
    fit_f = (split[track] == 0) & gated
    rng = np.random.default_rng([seed, 3])
    res = {"seed": seed, "tau": tau}
    for name, score in (("eps2", data["log_eps12"]), ("eps3", data["log_eps3"])):
        c_star, easy_frac = S.choose_cut(score[fit_f], (1.0 / n_t[track])[fit_f])
        g = bootstrap_gap(track, wrong, gated, score, c_star, fit_tracks, n_t, B, rng)
        for k in g:
            g[k].update(cert_readacross(g[k]["gap"], g[k]["boot_se"], len(fit_tracks), cfg["sizes"][1]))
        res[name] = {"c_star": c_star, "fit_easy_mass_fraction": easy_frac, "gap": g,
                     "slices": slice_table(track, wrong, gated, score, split, n_t)}
        sl = [s for s in res[name]["slices"] if s["frames"] > 0]
        if len(sl) >= 3 and np.ptp([s["err_per_track"] for s in sl]) > 0:
            res[name]["slice_spearman_err_vs_slice"] = float(spearmanr([s["slice"] for s in sl], [s["err_per_track"] for s in sl])[0])
    return res, split


# ------------------------------------------------------------------ main
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--config", default="configs/poc.yaml")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--out", help="default <run>/splits")
    a = ap.parse_args(argv)
    cfg_all = yaml.safe_load(open(a.config))
    cfg = {**S.DEFAULTS, **cfg_all.get("splits", {})}
    run = Path(a.run)
    out = Path(a.out) if a.out else run / "splits"
    summ = json.loads((out / "stage4_summary.json").read_text())
    for f, h in summ["files_sha256"].items():
        assert sha256_file(out / f) == h, f"{f} changed since Stage 4"
    data = S.load_inputs(run, cfg_all)
    results = []
    for seed in cfg["seeds"]:
        res, split = diagnose_seed(seed, data, cfg, a.boot)
        saved = np.load(out / f"seed{seed}.npz")["track_split"]
        assert (saved == split).all(), f"seed {seed}: recomputed split differs from the Stage 4 file"
        sres = next(r for r in summ["seeds"] if r["seed"] == seed)
        assert abs(res["eps2"]["c_star"] - sres["eps2"]["c_star"]) < 1e-9, "c* differs from Stage 4"
        assert abs(res["eps3"]["c_star"] - sres["eps3"]["c_star"]) < 1e-9, "c* (eps3) differs from Stage 4"
        results.append(res)
    rough = {n: rough_g1i(results, n, len(results)) for n in ("eps2", "eps3")}
    summary = {"bootstrap_draws": a.boot, "note": "FIT split only. Information, not tuning. Nothing frozen was changed.",
               "rough_g1i": rough, "seeds": results}
    (out / "fit_diagnostic.json").write_text(json.dumps(summary, indent=2))

    print(f"=== Fit-only diagnostic (B = {a.boot} track-cluster bootstrap draws; Fit split only) ===")
    for name, title in (("eps2", "eps (primary)"), ("eps3", "eps3 (first fallback)")):
        print(f"\n--- A. Hard-minus-easy gap, {title}  [percentage points] ---")
        print("seed | per-track gap  SE   95% CI            P(boot>0) | pooled gap  SE   95% CI            | rough Cert: need>  gap/SE  P(pass)")
        for r in results:
            p, q = r[name]["gap"]["per_track"], r[name]["gap"]["pooled"]
            print(f"  {r['seed']}  | {100*p['gap']:+6.2f}  {100*p['boot_se']:5.2f}  [{100*p['ci95'][0]:+5.2f},{100*p['ci95'][1]:+5.2f}]  {p['frac_boot_gap_positive']:6.1%}"
                  f" | {100*q['gap']:+6.2f}  {100*q['boot_se']:5.2f}  [{100*q['ci95'][0]:+5.2f},{100*q['ci95'][1]:+5.2f}] |"
                  f"   {100*p['gap_needed_rough']:4.2f}   {p['gap_over_se_cert']:5.2f}   {p['rough_pass_probability']:5.1%}")
        rg = rough[name]
        print(f"mean per-track gap {100*rg['mean_gap']:+.2f} pp | mean Fit SE {100*np.mean([r[name]['gap']['per_track']['boot_se'] for r in results]):.2f} pp | "
              f"rough Cert SE {100*rg['mean_se_cert_rough']:.2f} pp")
        print(f"ROUGH chance that G1(i) passes in a single reshuffle {rg['p_single_reshuffle']:.1%}; in >= 4 of 5: {rg['p_at_least_4_of_5']:.1%}")
        print(f"\n--- B. Error by confusability slice (8 slices of equal mass, low eps -> high eps), {title}, mean over seeds ---")
        print("slice | mass  | per-track err (min..max over seeds) | pooled err | share of Fit errors")
        for k in range(N_SLICES):
            rows = [r[name]["slices"][k] for r in results if r[name]["slices"][k]["frames"] > 0]
            if not rows:
                continue
            pt = [x["err_per_track"] for x in rows]
            print(f"  {k}   | {np.mean([x['mass_share'] for x in rows]):5.1%} | {100*np.mean(pt):5.2f}%  ({100*min(pt):4.2f}..{100*max(pt):4.2f}) |"
                  f" {100*np.mean([x['err_pooled'] for x in rows]):5.2f}%    | {np.mean([x['share_of_fit_errors'] for x in rows]):6.1%}")
        print("(slices 0-3 are the 'easy' half and 4-7 the 'hard' half, roughly; c* is the median)")
    print("\nREAD THIS: information only. 'rough pass probability' assumes Cert noise scales as 1/sqrt(tracks) and that the true gap equals "
          "the Fit gap; both are uncertain. Do not change c*, strata or definitions because of these numbers. A new design needs a "
          "declared revision (R7), committed before Stage 5.")
    return summary


if __name__ == "__main__":
    main()
