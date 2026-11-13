"""POC-2 (R8) Stage 4: Certificate A on five Cert/Test re-splits of the 1,046 non-design tracks, with B2, B3, B4.
LOG_r8_stage4_entry.md Part A (a)-(h), Amendment 1 (B2 among passing frames), Amendment 2 (descriptive P3/P4 at the
smallest alpha certified by both the commit rule and B2). Confirm phase: F1 is closed.

Usage:  py -3.12 -m src.r8_stage4 --run runs/20261003_stage1 --config configs/poc.yaml
Output: runs/20261003_stage1/poc2/stage4/stage4_summary.json
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src import r8_frozen as FZ
from src import r8_stage1 as R1
from src import r8_stage2 as S2
from src import r8_stage3_engine as E
from src import r8_stage3_resample as RS
from src import r8_stage4_baselines_design as BD
from src.r8_frozen import load_frozen
from src.r8_stage3_b1design import frame_margins
from src.util import sha256_file

FROZEN_ARGS = {}                       # tests point load_frozen at a fake root; production uses the F1 hashes
BASELINES_SHA256 = "80fef0be5ac04f0fb911f20752b8efebdeffc7254d7c3e0e47a9c23ca99ad580"   # Stage 4 Part A values
SEEDS = (0, 1, 2, 3, 4)
SPLIT_SALT, CURVE_SALT = 20261114, 20261115
CURVE_N = (100, 200, 400, 784)
METHODS = ("commit", "B2", "B3", "B4")


# ------------------------------------------------------------------ re-splits (Part A (b))
def test_counts(n_k, n_test, n_pool):
    """Largest-remainder Test quota per class, capped at n_k - 1 so every class keeps a Cert track.
    n_k: dict class -> pool tracks. Ties in the fractional part go to the smaller class id."""
    q = {k: n * n_test / n_pool for k, n in n_k.items()}
    t = {k: min(int(np.floor(q[k])), n_k[k] - 1) for k in n_k}
    order = sorted(n_k, key=lambda k: (-(q[k] - np.floor(q[k])), k))
    left = n_test - sum(t.values())
    while left > 0:
        moved = False
        for k in order:
            if left and t[k] < n_k[k] - 1:
                t[k] += 1
                left -= 1
                moved = True
        assert moved, "not enough pool tracks to fill the Test set while keeping every class in Cert"
    return t


def make_split(y_track, seed, n_test):
    """Returns a boolean Test mask over pool tracks (pool order = track-code order)."""
    classes = np.unique(y_track)
    n_k = {int(k): int((y_track == k).sum()) for k in classes}
    t = test_counts(n_k, n_test, len(y_track))
    test = np.zeros(len(y_track), bool)
    for k in classes:
        idx = np.flatnonzero(y_track == k)                            # already in track-code order
        idx = idx[np.random.default_rng([SPLIT_SALT, seed, int(k)]).permutation(len(idx))]
        test[idx[: t[int(k)]]] = True
    return test


# ------------------------------------------------------------------ per-method tables on the pool
def roi_sizes(run, cfg_poc, d):
    """ROI shorter side per pool frame from train_meta.csv (Stage 1 convention), after the Stage 1 alignment checks."""
    meta = pd.read_csv(Path(cfg_poc["data"]["cache_dir"]) / "train_meta.csv")
    E_ = np.load(Path(run) / "embeddings" / "train_release.npz")
    names = np.asarray(E_["track_names"]).astype(str)
    y, track, frame = E_["y"].astype(int), E_["track"].astype(int), E_["frame"].astype(int)
    assert len(meta) == len(y) and (meta["class_id"].to_numpy() == y).all() and (meta["frame_idx"].to_numpy() == frame).all() \
        and (meta["track_id"].astype(str).to_numpy() == names[track].astype(str)).all(), "train_meta.csv rows do not line up with the embeddings"
    roi = np.minimum(meta["roi_x2"] - meta["roi_x1"], meta["roi_y2"] - meta["roi_y1"]).to_numpy().astype(float)
    pool_rows = np.flatnonzero(np.isin(track, d["pool_codes"]))
    return roi[pool_rows]


def method_tables(d, cfg, bdes, roi):
    """committed, wrong, pos, size (T x G) for the commit rule and B2-B4, plus orders and checks."""
    tb, checks = d["table"], {}
    work = np.asarray(cfg["lambda_grid"]["stage1_indices"], int)
    rows = R1.track_rows(d["track"], d["frame"])
    codes = np.array([d["track"][r[0]] for r in rows])
    assert (codes == d["pool_codes"]).all(), "pool track order differs"
    y = np.array([d["y"][r[0]] for r in rows])
    pos_c, size_c = tb["V1_pos"][:, work].astype(int), tb["V1_size"][:, work].astype(float)
    com_c = tb["V1_committed"][:, work]
    # meta cross-check: ROI at the commit frame must equal the size stored by Stage 1 (hash-checked)
    roi_at = np.full(com_c.shape, np.nan)
    for i, r in enumerate(rows):
        c = com_c[i]
        roi_at[i, c] = roi[r[pos_c[i, c] - 1]]
    checks["meta_roi_matches_stage1_sizes"] = bool(np.array_equal(np.isnan(roi_at), np.isnan(size_c))
                                                   and np.allclose(roi_at[com_c], size_c[com_c]))
    T = {"commit": {"committed": com_c, "wrong": com_c & (tb["V1_pred"][:, work] != y[:, None]), "pos": pos_c,
                    "size": size_c, "order": cfg["certificate_A"]["threshold_order"]["order_positions"]}}
    m, top = frame_margins(d["head"], d["emb"], d["passed"])
    n_all = int(bdes["n_all_tracks"])
    pos_b2 = BD.b2_positions(n_all, rows, d["passed"], codes)
    checks["b2_frames_all_pass_gate"] = bool(all(p < 0 or d["passed"][r[p]] for p, r in zip(pos_b2, rows)))
    B = BD.baseline_scores(rows, m, top, d["passed"], pos_b2)
    for b in BD.BASELINES:
        thr = np.asarray(bdes["baselines"][b]["work_thresholds"], float)
        com = B[b]["score"][:, None] >= thr[None, :]
        size = np.array([roi[r[p - 1]] if p > 0 else np.nan for r, p in zip(rows, B[b]["pos"])])
        T[b] = {"committed": com, "wrong": com & (B[b]["label"] != y)[:, None],
                "pos": np.repeat(B[b]["pos"][:, None], len(thr), 1), "size": np.repeat(size[:, None], len(thr), 1),
                "order": bdes["baselines"][b]["order_positions"]}
    nopass = tb["n_passing"] == 0
    return T, y, nopass, checks


# ------------------------------------------------------------------ certification and Test metrics (Part A (c), (d))
def certify_on(tab, cert_mask, alpha, delta):
    mu, n = E.track_mu(tab["committed"], tab["wrong"], cert_mask[None, :].astype(float), alpha)
    return int(E.certify(mu, n, alpha, tab["order"], delta)[0])


def test_metrics(tab, j, test_mask):
    out = {"certified": j >= 0, "grid_position": None if j < 0 else j, "coverage": 0.0, "error_committed": None,
           "median_commit_frame": None, "median_size": None, "committed": 0}
    if j < 0:
        return out
    c = tab["committed"][test_mask, j]
    out.update(coverage=float(c.sum() / test_mask.sum()), committed=int(c.sum()), grid_position=int(j))
    if c.any():
        out["error_committed"] = float(tab["wrong"][test_mask, j][c].sum() / c.sum())
        out["median_commit_frame"] = float(np.median(tab["pos"][test_mask, j][c]))
        out["median_size"] = float(np.median(tab["size"][test_mask, j][c]))
    return out


def smallest_certified(tab, cert_mask, alphas, delta):
    for a in sorted(alphas):
        if certify_on(tab, cert_mask, a, delta) >= 0:
            return a
    return None


def curve_vs_n(tab, cert_idx, seed, alphas, delta, n_tracks):
    perm = cert_idx[np.random.default_rng([CURVE_SALT, seed]).permutation(len(cert_idx))]
    out = {}
    for n in CURVE_N:
        mask = np.zeros(n_tracks, bool)
        mask[perm[:n]] = True
        out[n] = smallest_certified(tab, mask, alphas, delta)
    return out


def p_checks(per_seed, primary, c_min):
    """Part A (f), unchanged by Amendment 2. Returns per-seed booleans."""
    out = {"P2": [], "P3": [], "P4": []}
    for s in per_seed:
        c, b2 = s["alphas"][primary]["commit"], s["alphas"][primary]["B2"]
        out["P2"].append(bool(c["certified"] and c["coverage"] >= c_min))
        out["P3"].append(bool(c["coverage"] > b2["coverage"]))
        out["P4"].append(bool(c["committed"] > 0 and c["median_commit_frame"] < 30))
    return out


def descriptive_common(s, alphas):
    """Amendment 2: P3/P4 comparisons at the smallest alpha certified by both the commit rule and B2, else 'none'."""
    for a in sorted(alphas):
        c, b2 = s["alphas"][a]["commit"], s["alphas"][a]["B2"]
        if c["certified"] and b2["certified"]:
            return {"alpha": a, "commit_coverage": c["coverage"], "b2_coverage": b2["coverage"],
                    "p3_holds": bool(c["coverage"] > b2["coverage"]), "median_commit_frame": c["median_commit_frame"],
                    "p4_holds": bool(c["committed"] > 0 and c["median_commit_frame"] < 30)}
    return "none"


# ------------------------------------------------------------------ main
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--config", default="configs/poc.yaml")
    a = ap.parse_args(argv)
    cfg = load_frozen(**FROZEN_ARGS)
    t0 = time.time()
    alphas = [float(x) for x in cfg["certificate_A"]["alpha_grid"]]
    primary, delta = float(cfg["certificate_A"]["primary_alpha"]), float(cfg["fixed_in_every_phase"]["delta"])
    n_cert, n_test = int(cfg["splits"]["cert_tracks"]), int(cfg["splits"]["test_tracks"])
    lv = cfg["pass_levels"]
    c_min = float(lv["P2"]["c_min"])
    s4 = Path(a.run) / "poc2" / "stage4"
    assert sha256_file(s4 / "baselines_design.json") == BASELINES_SHA256, "baselines_design.json differs from the hash in Stage 4 Part A"
    bdes = json.loads((s4 / "baselines_design.json").read_text())

    d = RS.load_pool(a.run, cfg)                                     # hash-checked, pool rows only
    _, checks = RS.pool_tables(d, cfg)                               # pool table reproduction + frozen grid checks
    bdes["n_all_tracks"] = len(np.load(Path(a.run) / "splits" / "seed0.npz")["track_names"])
    roi = roi_sizes(a.run, yaml.safe_load(Path(a.config).read_text()), d)
    T, y, nopass, c2 = method_tables(d, cfg, bdes, roi)
    checks.update(c2)
    n_pool = len(y)
    checks["pool_size_matches_frozen"] = n_pool == int(cfg["splits"]["pool_tracks"])

    per_seed = []
    split_ok = True
    for s in SEEDS:
        test = make_split(y, s, n_test)
        cert = ~test
        split_ok &= bool(test.sum() == n_test and cert.sum() == n_cert and set(np.unique(y[cert])) == set(np.unique(y)))
        row = {"seed": s, "n_cert": int(cert.sum()), "n_test": int(test.sum()),
               "false_rejection_rate": float(nopass[test].mean()), "alphas": {}}
        for al in alphas:
            row["alphas"][al] = {m: test_metrics(T[m], certify_on(T[m], cert, al, delta), test) for m in METHODS}
        row["smallest_certified"] = {m: smallest_certified(T[m], cert, alphas, delta) for m in METHODS}
        row["curve_vs_n"] = curve_vs_n(T["commit"], np.flatnonzero(cert), s, alphas, delta, n_pool)
        row["descriptive_common_alpha"] = descriptive_common(row, alphas)
        per_seed.append(row)
    checks["splits_valid"] = split_ok                                # sizes, every class in Cert (disjoint by construction)
    P = p_checks(per_seed, primary, c_min)
    verdict = {k: {"per_seed": v, "count": int(sum(v)), "required": int(lv[k]["resplits_required"]),
                   "pass": bool(sum(v) >= int(lv[k]["resplits_required"]))} for k, v in P.items()}

    # ---- print
    print(f"\n=== R8 Stage 4: Certificate A on {len(SEEDS)} re-splits of the {n_pool} non-design tracks "
          f"(Cert {n_cert}, Test {n_test}, delta {delta}) ===")
    f = lambda x, p=2: "  n/a " if x is None else f"{100 * x:6.{p}f}%"
    g = lambda x: " n/a" if x is None else f"{x:4.1f}"
    for al in alphas:
        print(f"\n--- alpha {100 * al:.1f}%" + ("  (primary)" if al == primary else ""))
        print(f"{'seed':>4} | " + " | ".join(f"{m:^31}" for m in METHODS))
        print(f"{'':>4} | " + " | ".join(f"{'cert':>4} {'cov':>7} {'err':>7} {'fr':>4} {'px':>5}" for _ in METHODS))
        for s in per_seed:
            cells = []
            for m in METHODS:
                r = s["alphas"][al][m]
                px = "  n/a" if r["median_size"] is None else f"{r['median_size']:5.1f}"
                cells.append(f"{'yes' if r['certified'] else 'no':>4} {f(r['coverage'])} {f(r['error_committed'])} "
                             f"{g(r['median_commit_frame'])} {px}")
            print(f"{s['seed']:>4} | " + " | ".join(cells))
    print("(cov = Test coverage over 262 tracks; err = error among committed Test signs; fr = median commit frame; "
          "px = median ROI size at commit)")
    print("\n--- Track-level false-rejection rate (Test, no passing frame): "
          + ", ".join(f"seed {s['seed']} {100 * s['false_rejection_rate']:.2f}%" for s in per_seed))
    print("\n--- Certifiability curve (smallest certified alpha)")
    for s in per_seed:
        sc = s["smallest_certified"]
        print(f"  seed {s['seed']}: n=784 " + ", ".join(f"{m} {f(sc[m], 1)}" for m in METHODS)
              + " | commit rule vs n: " + ", ".join(f"n={n} {f(v, 1)}" for n, v in s["curve_vs_n"].items()))
    print(f"\n--- Pass levels at alpha {100 * primary:.0f}% (Part A (f); need >= 4 of 5)")
    for k, v in verdict.items():
        print(f"  {k}: per re-split {['PASS' if x else 'fail' for x in v['per_seed']]} -> {v['count']}/5 -> "
              f"{'PASS' if v['pass'] else 'FAIL'}")
    print("\n--- Descriptive only (Amendment 2): P3/P4 at the smallest alpha certified by both the commit rule and B2")
    for s in per_seed:
        dc = s["descriptive_common_alpha"]
        if dc == "none":
            print(f"  seed {s['seed']}: none")
        else:
            print(f"  seed {s['seed']}: alpha {100 * dc['alpha']:.1f}%: commit {f(dc['commit_coverage'])} vs B2 "
                  f"{f(dc['b2_coverage'])} (P3 {'holds' if dc['p3_holds'] else 'does not hold'}); median commit frame "
                  f"{g(dc['median_commit_frame'])} (P4 {'holds' if dc['p4_holds'] else 'does not hold'})")
    print("\nChecks:")
    for k, v in checks.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    all_pass = all(checks.values())
    print(f"\nALL PASS (checks): {all_pass}")

    summary = {"plan": "poc2-implementation-plan-r8 Stage 4",
               "frozen_r8_sha256": sha256_file(Path(FROZEN_ARGS.get("root", FZ.ROOT)) / FZ.FROZEN),
               "code_sha256": {"r8_stage4.py": sha256_file(Path(__file__))},
               "inputs_sha256": {**d["inputs_sha256"], "baselines_design.json": BASELINES_SHA256,
                                 "train_meta.csv": sha256_file(Path(yaml.safe_load(Path(a.config).read_text())["data"]["cache_dir"]) / "train_meta.csv")},
               "config": {"seeds": SEEDS, "split_salt": SPLIT_SALT, "curve_salt": CURVE_SALT, "curve_n": CURVE_N,
                          "alphas": alphas, "primary": primary, "delta": delta, "c_min": c_min},
               "per_seed": per_seed, "pass_levels": verdict, "checks": checks, "all_pass": all_pass,
               "seconds": round(time.time() - t0, 1)}
    (s4 / "stage4_summary.json").write_text(json.dumps(S2.jsonable(summary), indent=2))
    print(f"Output: {s4 / 'stage4_summary.json'} ({summary['seconds']} s)")
    print("Stage 5 is not started.")
    return summary


if __name__ == "__main__":
    main()
