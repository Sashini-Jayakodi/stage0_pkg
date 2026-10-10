"""Stage 4: reshuffles and Fit-side calibration (plan section 6, Stage 4 + the 6.3 ordering fix).

For each seed 0-4 this script:
  1. splits the 1,307 TRACKS (never frames) into Fit / Cert / Test, stratified by class;
  2. fits the open-set gate threshold tau on ONE random frame per Fit track (split-conformal, beta = 5%);
  3. fits the cut-point c* that splits the gated-in Fit mass 50/50 into an easy and a hard stratum;
  4. computes Fit base errors, the Fit margin grid per stratum, and the Fit-based testing ORDER of the grid
     (start at the Fit-argmin of estimated E[L], then relax looser; plan 6.3).
It does this for eps (top-2 pair, the primary) and eps3 (top-3, the pre-registered first fallback).

DELIBERATELY NOT DONE HERE: no error rate of any kind is computed on Cert or Test. Cert gets only counts,
masses and gate-rejection rates, as the plan says. Cert errors are first looked at in Stage 5, after the
pre-registration block in LOG.md has been committed.

Usage:  py -3.12 -m src.splits --run runs/20261003_stage1
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import yaml

from src.train import sha256_file

K = 43
SPLIT_NAMES = ["fit", "cert", "test"]
# Plan constants (section 1). If configs/poc.yaml has a `splits:` block it overrides these.
DEFAULTS = {"sizes": [261, 784, 262], "fractions": [0.2, 0.6, 0.2], "seeds": [0, 1, 2, 3, 4],
            "beta": 0.05, "delta": 0.1, "alpha_grid": [0.01, 0.02, 0.03, 0.05], "alpha_rel_base": 0.5,
            "margin_points": 50, "margin_top_quantile": 0.98}


# ------------------------------------------------------------------ 1. splits
def allocate_counts(T, fractions, targets, rng):
    """How many tracks of each class go to Fit/Cert/Test: 20/60/20 per class, at least 1 per split, then
    adjusted by largest remainder so the three totals hit the targets exactly (261/784/262)."""
    T = np.asarray(T)
    ideal = T[:, None] * np.asarray(fractions)[None, :]
    base = np.maximum(1, np.floor(ideal + 1e-9)).astype(int)
    assert (base.sum(1) <= T).all(), "a class has fewer than 3 tracks"
    cap = np.asarray(targets) - base.sum(0)
    assert (cap >= 0).all(), "targets are too small for one-track-per-split-per-class"
    rem = T - base.sum(1)
    resid = ideal - base
    jitter = rng.random(resid.shape) * 1e-6              # random tie-break only
    while rem.sum() > 0:
        sc = resid + jitter
        sc[rem == 0, :] = -np.inf
        sc[:, cap == 0] = -np.inf
        c, k = np.unravel_index(np.argmax(sc), sc.shape)
        assert np.isfinite(sc[c, k]), "cannot reach the target split sizes"
        base[c, k] += 1; rem[c] -= 1; cap[k] -= 1; resid[c, k] -= 1
    assert (base.sum(0) == np.asarray(targets)).all() and (base.sum(1) == T).all() and (base >= 1).all()
    return base


def make_split(track_class, sizes, fractions, seed):
    """0 = Fit, 1 = Cert, 2 = Test, one entry per track. Which tracks of a class go where is a seeded shuffle."""
    track_class = np.asarray(track_class)
    assert sum(sizes) == len(track_class), "split sizes must add up to the number of tracks"
    rng = np.random.default_rng([seed, 0])
    T = np.bincount(track_class, minlength=K)
    counts = allocate_counts(T, fractions, sizes, rng)
    split = np.full(len(track_class), -1, dtype=np.int8)
    for c in range(K):
        idx = np.flatnonzero(track_class == c)
        rng.shuffle(idx)
        a, b = counts[c, 0], counts[c, 0] + counts[c, 1]
        split[idx[:a]], split[idx[a:b]], split[idx[b:]] = 0, 1, 2
    assert (split >= 0).all()
    return split, counts


# ------------------------------------------------------------------ 2. gate
def conformal_rank(n, beta):
    """k = ceil((n+1)(1-beta)); n=261, beta=5% gives 249. The rounding guards against float noise
    (e.g. 20 * 0.95 = 19.000000000000004 would otherwise ceil to 20)."""
    return math.ceil(round((n + 1) * (1 - beta), 9))


def one_frame_per_track(track, tracks, rng):
    """Row index of one random frame for each track in `tracks`."""
    order = np.argsort(track, kind="stable")
    n = np.bincount(track, minlength=int(track.max()) + 1)
    start = np.concatenate([[0], np.cumsum(n)[:-1]])
    return order[start[tracks] + rng.integers(0, n[tracks])]


def fit_gate(s, track, split, seed, beta):
    fit_tracks = np.flatnonzero(split == 0)
    rows = one_frame_per_track(track, fit_tracks, np.random.default_rng([seed, 1]))
    sc = np.sort(s[rows])
    k = conformal_rank(len(sc), beta)
    assert 1 <= k <= len(sc), "too few Fit tracks for this beta"
    return float(sc[k - 1]), rows, k


# ------------------------------------------------------------------ 3. cut-point
def choose_cut(score, weight):
    """Cut-point c* among the distinct score values: easy side = score <= c*. Pick the c* whose easy side
    holds track-weighted mass closest to 50% of the total. Whole class pairs stay on one side because the
    score depends only on the pair."""
    vals = np.unique(score)
    order = np.argsort(score, kind="stable")
    s_sorted, w_sorted = score[order], weight[order]
    cum = np.cumsum(w_sorted)
    last = np.searchsorted(s_sorted, vals, side="right") - 1          # cumulative mass up to each value
    frac = cum[last] / cum[-1]
    j = int(np.argmin(np.abs(frac - 0.5)))                            # first (lowest) value on ties
    return float(vals[j]), float(frac[j])


# ------------------------------------------------------------------ 4. Fit-side calibration
def margin_grid(m, points, top_q):
    """~50 Fit quantiles of the margin, from the minimum (accept-all) to the 98th percentile.
    Ascending order: index 0 = loosest (accept every frame with m >= min), last = strictest."""
    return np.unique(np.quantile(m, np.linspace(0.0, top_q, points)))


def el_curve(track, rows, m, wrong, grid, alpha, n_tracks):
    """Fit-estimated E[L] at each grid threshold for one stratum, with the shifted loss of the plan:
    l_t = (1/n_tk) * sum_i ( 1[accepted & wrong] - alpha * 1[accepted] ),  L_t = l_t + alpha,
    averaged over the tracks that have at least one frame in the stratum. Accept rule: margin >= threshold."""
    tr = track[rows]
    n_tk = np.bincount(tr, minlength=n_tracks)
    has = n_tk > 0
    out = np.empty(len(grid))
    for g, t in enumerate(grid):
        acc = m[rows] >= t
        err = np.bincount(tr, weights=(acc & wrong[rows]).astype(float), minlength=n_tracks)
        na = np.bincount(tr, weights=acc.astype(float), minlength=n_tracks)
        out[g] = ((err - alpha * na)[has] / n_tk[has]).mean() + alpha
    return out


def order_from_curve(el):
    """Plan 6.3: start at the Fit-argmin of estimated E[L], then step looser one grid point at a time.
    Grid index 0 is the loosest, so the order is start, start-1, ..., 0. Ties -> the loosest minimiser."""
    start = int(np.argmin(el))
    return start, np.arange(start, -1, -1)


def track_weighted_error(track, rows, wrong, n_t):
    """Base error of the gated-in frames of one stratum, no margin threshold, in two readings.
    Returns (pooled, per_track_mean):
      per_track_mean = mean over tracks that have >= 1 frame in the stratum of that track's in-stratum error rate.
                       PRIMARY (decision 2026-10-10): it is what the Stage 6 certificate averages, and what G1(i) uses.
      pooled         = frames weighted 1/n_t (with n_t = 30 almost the plain frame error rate). Sensitivity only."""
    w = 1.0 / n_t[track[rows]]
    pooled = float((w * wrong[rows]).sum() / w.sum())
    tr = track[rows]
    cnt = np.bincount(tr, minlength=len(n_t)); err = np.bincount(tr, weights=wrong[rows].astype(float), minlength=len(n_t))
    has = cnt > 0
    return pooled, float((err[has] / cnt[has]).mean())


def calibrate_variant(name, score, ctx, cfg):
    """Everything Fit-side for one stratifier (eps or eps3). Cert is touched only to count/weigh frames."""
    track, split, gated, m, wrong, n_t = ctx["track"], ctx["split"], ctx["gated"], ctx["m"], ctx["wrong"], ctx["n_t"]
    fit_f = (split[track] == 0) & gated
    cert_f = (split[track] == 1) & gated
    w = 1.0 / n_t[track]
    c_star, easy_frac_fit = choose_cut(score[fit_f], w[fit_f])
    res = {"c_star": c_star, "fit_easy_mass_fraction": easy_frac_fit}
    arrays = {}
    for label, sel in (("easy", lambda f: score <= c_star), ("hard", lambda f: score > c_star)):
        fit_rows = np.flatnonzero(fit_f & sel(0))
        cert_rows = np.flatnonzero(cert_f & sel(0))
        pooled, ptm = track_weighted_error(track, fit_rows, wrong, n_t)
        grid = margin_grid(m[fit_rows], cfg["margin_points"], cfg["margin_top_quantile"])
        alphas = {f"abs_{a}": a for a in cfg["alpha_grid"]}
        if ptm > 0:                                       # primary: 0.5 x per-track-mean base error
            alphas["rel_0.5base"] = cfg["alpha_rel_base"] * ptm
        if pooled > 0:                                    # sensitivity: 0.5 x pooled base error
            alphas["rel_0.5base_pooled"] = cfg["alpha_rel_base"] * pooled
        orders = {}
        for an, a in alphas.items():
            el = el_curve(track, fit_rows, m, wrong, grid, a, len(n_t))
            start, order = order_from_curve(el)
            arrays[f"{name}_{label}_EL_{an}"] = el
            orders[an] = {"alpha": a, "start_index": start, "n_tested_at_most": int(len(order)),
                          "start_threshold": float(grid[start]), "EL_at_start": float(el[start])}
        arrays[f"{name}_{label}_margin_grid"] = grid
        res[label] = {
            "fit_frames_gated_in": int(len(fit_rows)), "fit_tracks": int(len(np.unique(track[fit_rows]))),
            "cert_frames_gated_in": int(len(cert_rows)), "cert_tracks": int(len(np.unique(track[cert_rows]))),
            "fit_base_error_primary": ptm,                # = per-track mean (decision 2026-10-10)
            "fit_base_error_per_track_mean": ptm, "fit_base_error_frame_weighted": pooled,
            "grid_points": int(len(grid)), "orders": orders}
    wc = w[cert_f]
    res["cert_easy_mass_fraction"] = float(wc[score[cert_f] <= c_star].sum() / wc.sum())
    res["fit_split_within_45_55"] = bool(0.45 <= easy_frac_fit <= 0.55)
    res["fit_split_worse_than_65_35"] = bool(max(easy_frac_fit, 1 - easy_frac_fit) > 0.65)
    return res, arrays


def run_seed(seed, data, cfg):
    track, s, m, y = data["track"], data["s"], data["margin"], data["y"]
    n_tracks = int(track.max()) + 1
    n_t = np.bincount(track, minlength=n_tracks).astype(float)
    track_class = np.zeros(n_tracks, dtype=int)
    track_class[track] = y
    split, counts = make_split(track_class, cfg["sizes"], cfg["fractions"], seed)
    # --- checks the plan asks for
    assert (np.bincount(split, minlength=3) == cfg["sizes"]).all()
    for k in range(3):
        assert set(track_class[split == k]) == set(range(K)), f"a class is missing from split {SPLIT_NAMES[k]}"
    tau, gate_rows, rank = fit_gate(s, track, split, seed, cfg["beta"])
    gated = s <= tau
    cert_frames = split[track] == 1
    cert_tracks = np.flatnonzero(split == 1)
    cert_rows = one_frame_per_track(track, cert_tracks, np.random.default_rng([seed, 2]))
    wrong = data["top3"][:, 0] != y
    ctx = {"track": track, "split": split, "gated": gated, "m": m, "wrong": wrong, "n_t": n_t}
    res = {"seed": seed, "tracks": {SPLIT_NAMES[k]: int((split == k).sum()) for k in range(3)},
           "frames": {SPLIT_NAMES[k]: int((split[track] == k).sum()) for k in range(3)},
           "per_class_counts_min_per_split": counts.min(0).tolist(),
           "gate": {"n_fit_tracks": int((split == 0).sum()), "conformal_rank": rank, "tau": tau,
                    "cert_rejection_frame_level": float((~gated[cert_frames]).mean()),
                    "cert_rejection_one_frame_per_track": float((~gated[cert_rows]).mean())}}
    arrays = {"track_split": split, "gate_rows_fit": gate_rows, "class_counts": counts}
    res["eps2"], a2 = calibrate_variant("eps2", data["log_eps12"], ctx, cfg)
    res["eps3"], a3 = calibrate_variant("eps3", data["log_eps3"], ctx, cfg)
    arrays.update(a2); arrays.update(a3)
    return res, arrays


# ------------------------------------------------------------------ main
def load_inputs(run, cfg_all):
    emb_dir, head_dir = run / "embeddings", run / "head_emb"
    es = json.loads((emb_dir / "embeddings_summary.json").read_text())
    assert es["checkpoint_sha256"] == cfg_all["embed"]["expected_checkpoint_sha256"], "not the frozen checkpoint"
    for f, h in es["files_sha256"].items():
        assert sha256_file(emb_dir / f) == h, f"{f} changed since Stage 2"
    hs = json.loads((head_dir / "head_summary.json").read_text())
    for f in ("head.npz", "scores_train_release.npz"):
        assert sha256_file(head_dir / f) == hs["files_sha256"][f], f"{f} changed since Stage 3"
    sc, tab = np.load(head_dir / "scores_train_release.npz"), np.load(head_dir / "head.npz")
    emb = np.load(emb_dir / "train_release.npz")
    assert (emb["y"] == sc["y"]).all() and (emb["track"] == sc["track"]).all(), "Stage 2 and Stage 3 files disagree"
    log_eps, top3 = tab["log_eps"], sc["top3"].astype(int)
    log_eps3 = np.logaddexp(sc["log_eps12"].astype(np.float64), log_eps[top3[:, 0], top3[:, 2]])  # eps3 = eps12 + eps13
    assert np.isfinite(log_eps3).all() and np.isfinite(sc["log_eps12"]).all(), "non-finite log eps (top-3 classes not distinct?)"
    data = {"track": sc["track"].astype(int), "y": sc["y"].astype(int), "s": sc["s"].astype(float),
            "margin": sc["margin"].astype(float), "top3": top3,
            "log_eps12": sc["log_eps12"].astype(float), "log_eps3": log_eps3,
            "track_names": np.asarray(emb["track_names"])}
    return data


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--config", default="configs/poc.yaml")
    ap.add_argument("--out", help="default <run>/splits")
    a = ap.parse_args(argv)
    cfg_all = yaml.safe_load(open(a.config))
    cfg = {**DEFAULTS, **cfg_all.get("splits", {})}
    run = Path(a.run)
    out = Path(a.out) if a.out else run / "splits"
    out.mkdir(parents=True, exist_ok=True)
    data = load_inputs(run, cfg_all)
    n_tracks = int(data["track"].max()) + 1
    assert n_tracks == sum(cfg["sizes"]), f"{n_tracks} tracks but split sizes add to {sum(cfg['sizes'])}"
    results = []
    for seed in cfg["seeds"]:
        res, arrays = run_seed(seed, data, cfg)
        np.savez(out / f"seed{seed}.npz", track_names=data["track_names"], **arrays)
        results.append(res)
    summary = {"constants_used": cfg, "n_tracks": n_tracks, "seeds": results,
               "files_sha256": {f"seed{s}.npz": sha256_file(out / f"seed{s}.npz") for s in cfg["seeds"]},
               "interpretation_notes": [
                   "base error: PRIMARY = per-track mean of in-stratum error rates (decided 2026-10-10, what the certificate averages); the pooled 1/n_t-weighted rate is stored as a sensitivity. c* mass still uses weight 1/n_t per frame (plan wording)",
                   "alpha 'rel_0.5base' = 0.5 x per-track-mean base error (primary); 'rel_0.5base_pooled' = 0.5 x pooled base error (sensitivity)",
                   "margin accept rule: accept when margin >= threshold; grid index 0 = loosest",
                   "eps3 = eps(y1,y2) + eps(y1,y3), added in the log domain",
                   "no Cert or Test error is computed in this stage"]}
    (out / "stage4_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"=== Stage 4 summary (tracks split {cfg['sizes']}, beta {cfg['beta']}) ===")
    for r in results:
        g, e = r["gate"], r["eps2"]
        print(f"\nseed {r['seed']}: tracks {r['tracks']} | frames {r['frames']}")
        print(f"  gate: tau {g['tau']:.4f} (rank {g['conformal_rank']} of {g['n_fit_tracks']}) | Cert rejection: "
              f"frame-level {g['cert_rejection_frame_level']:.1%}, one-frame-per-track {g['cert_rejection_one_frame_per_track']:.1%}")
        print(f"  eps  c* {e['c_star']:.2f} | easy mass Fit {e['fit_easy_mass_fraction']:.1%}, Cert {e['cert_easy_mass_fraction']:.1%}"
              f" | within 45/55 on Fit: {e['fit_split_within_45_55']}")
        for lab in ("easy", "hard"):
            x = e[lab]
            print(f"    {lab}: Fit frames {x['fit_frames_gated_in']} in {x['fit_tracks']} tracks | Cert frames {x['cert_frames_gated_in']} "
                  f"in {x['cert_tracks']} tracks | Fit base error {x['fit_base_error_primary']:.3%} per-track mean (pooled "
                  f"{x['fit_base_error_frame_weighted']:.3%}) | grid {x['grid_points']} pts")
        e3 = r["eps3"]
        print(f"  eps3 c* {e3['c_star']:.2f} | easy mass Fit {e3['fit_easy_mass_fraction']:.1%}, Cert {e3['cert_easy_mass_fraction']:.1%}")
        if e["fit_split_worse_than_65_35"] or e["cert_easy_mass_fraction"] < .35 or e["cert_easy_mass_fraction"] > .65:
            print("  NOTE: lopsided split. Plan: keep c* as defined, report the realised split, never split a pair.")
        if g["cert_rejection_frame_level"] > 0.10:
            print("  NOTE: frame-level Cert rejection above 10%. Plan: expected to some degree; report both rates; never retune on Cert.")
    print("\nNo Cert or Test error was computed. Commit the PRE-REGISTRATION block of LOG.md before Stage 5.")
    return summary


if __name__ == "__main__":
    main()
