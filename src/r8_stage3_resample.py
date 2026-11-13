"""POC-2 (R8) Stage 3b: resampling validity test on real data (track-level commit-rule certifier vs B1).
LOG_r8_stage3_entry.md Part A (a)-(d), (f), (g), (i). Confirm phase: F1 is closed, so pool outcomes may be computed.

Population = the 1,046 non-design tracks. The true mean loss of every threshold is computed exactly on them (B1: on
all their frames). 2,000 draws of 784 tracks with replacement; each draw is certified by the track-level rule
(frozen grid and order, n = 784) and by B1 (design-set grid and order from b1_design.json, n = frames in the draw).
Violation = something certified and its true mean loss > alpha. Every alpha of the frozen grid; 1% is the headline.

Usage:  py -3.12 -m src.r8_stage3_resample --run runs/20261003_stage1
Output: runs/20261003_stage1/poc2/stage3/stage3b_summary.json
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from src import r8_stage1 as R1
from src import r8_stage2 as S2
from src import r8_stage3_engine as E
from src import r8_frozen as FZ
from src.r8_frozen import load_frozen
from src.r8_stage3_b1design import frame_margins
from src.util import sha256_file

FROZEN_ARGS = {}                       # tests point load_frozen at a fake root; production uses the F1 hashes
B1_DESIGN_SHA256 = "8fb7e1c534ce1d923595e18794327afa70d654f033d4b55196a4dc2a169d22c9"   # LOG_r8_stage3_entry.md Part A
N_DRAWS = 2000
SEED = 20261115
POOL_FILE, EVID_FILE = "track_table_pool.npz", "frame_evidence.npz"


# ------------------------------------------------------------------ inputs
def load_pool(run, cfg):
    """Hash-check everything (hard rule 2), keep pool rows only. The design set is not used here at all."""
    run = Path(run)
    s1dir, s3dir = run / "poc2" / "stage1", run / "poc2" / "stage3"
    s1 = json.loads((s1dir / "stage1_summary.json").read_text())
    assert s1.get("all_pass") is True, "Stage 1 did not pass"
    for f in (POOL_FILE, EVID_FILE):
        assert sha256_file(s1dir / f) == s1["files_sha256"][f], f"{f} changed since Stage 1"
    ins = s1["inputs_sha256"]
    for rel, key in (("embeddings/train_release.npz", "train_release.npz"), ("head_emb/head.npz", "head.npz"),
                     ("splits/seed0.npz", "seed0.npz")):
        assert sha256_file(run / rel) == ins[key], f"{key} changed since Stage 1"
    hs = json.loads((run / "head_emb" / "head_summary.json").read_text())
    sc = run / "head_emb" / "scores_train_release.npz"
    assert sha256_file(sc) == hs["files_sha256"]["scores_train_release.npz"], "scores_train_release.npz changed since Stage 3"
    assert sha256_file(s3dir / "b1_design.json") == B1_DESIGN_SHA256, "b1_design.json differs from the hash in Stage 3 Part A"

    Sp = np.load(run / "splits" / "seed0.npz")
    names = np.asarray(Sp["track_names"]).astype(str)
    pool_mask = np.asarray(Sp["track_split"]) != 0                  # the 1,046 non-design tracks
    pool_codes = np.flatnonzero(pool_mask)
    assert len(pool_codes) == int(cfg["splits"]["pool_tracks"]), "pool size differs from frozen_r8.yaml"
    E_ = np.load(run / "embeddings" / "train_release.npz")
    track_all = E_["track"].astype(int)
    rows = np.flatnonzero(pool_mask[track_all])
    d = {"emb": E_["emb"][rows], "y": E_["y"][rows].astype(int), "track": track_all[rows], "frame": E_["frame"][rows].astype(int)}
    d["top1"] = np.load(sc)["top3"][rows, 0].astype(int)
    ev = np.load(s1dir / EVID_FILE)
    d["passed"] = ev["passed"][rows]
    H = np.load(run / "head_emb" / "head.npz")
    d["head"] = {k: H[k] for k in ("mu", "W", "priors")}
    T = np.load(s1dir / POOL_FILE)
    d["table"] = {k: T[k] for k in T.files}
    assert (np.asarray(d["table"]["track_names"]).astype(str) == names[pool_codes]).all(), "pool table order differs from seed0"
    assert pool_mask[d["track"]].all() and set(np.unique(d["track"])) == set(pool_codes), "a design track reached Stage 3b"
    d["b1"] = json.loads((s3dir / "b1_design.json").read_text())
    d.update(pool_codes=pool_codes, tau=float(s1["results"]["tau"]),
             inputs_sha256={**ins, POOL_FILE: s1["files_sha256"][POOL_FILE], EVID_FILE: s1["files_sha256"][EVID_FILE],
                            "scores_train_release.npz": hs["files_sha256"]["scores_train_release.npz"],
                            "b1_design.json": B1_DESIGN_SHA256})
    return d


# ------------------------------------------------------------------ per-track outcome tables
def pool_tables(d, cfg):
    """Track-level committed/wrong on the frozen working grid, B1 frame counts per track, plus reproduction checks."""
    tb, checks = d["table"], {}
    work = np.asarray(cfg["lambda_grid"]["stage1_indices"], int)
    checks["frozen_grid_matches_stage1_grid"] = bool(np.array_equal(tb["V1_grid"][work], np.asarray(cfg["lambda_grid"]["values"])))
    rows = R1.track_rows(d["track"], d["frame"])                      # ordered by track code = pool table order
    y_track = np.array([d["y"][r[0]] for r in rows])
    # recompute V1 for the pool tracks with the Stage 1 code; it must equal the stored (hash-checked) table
    ll, s = R1.frame_evidence(d["head"], d["emb"])
    checks["gate_flags_reproduce"] = bool(((s <= d["tau"]) == d["passed"]).all())
    lp, grid = np.log(d["head"]["priors"]), tb["V1_grid"]
    same = True
    for i, r in enumerate(rows):
        M, top = R1.margins(R1.running_scores(ll[r], d["passed"][r], lp, "V1")[0])
        idx = R1.first_commit(M, grid)
        c = idx < len(r)
        pred = np.full(len(grid), -1)
        pred[c] = top[idx[c]]
        same &= bool((c == tb["V1_committed"][i]).all() and (pred == tb["V1_pred"][i]).all())
    checks["pool_commit_table_reproduces"] = bool(same)
    committed = tb["V1_committed"][:, work]
    wrong = committed & (tb["V1_pred"][:, work] != y_track[:, None])

    m, top = frame_margins(d["head"], d["emb"], d["passed"])
    checks["b1_top_equals_stored_top1"] = bool((top == d["top1"]).all())
    thr = np.asarray(d["b1"]["work_thresholds"], float)
    acc = m[:, None] >= thr[None, :]
    wr = acc & (top != d["y"])[:, None]
    inv = np.searchsorted(d["pool_codes"], d["track"])                 # pool track position of each frame
    A = np.stack([np.bincount(inv, weights=acc[:, j], minlength=len(rows)) for j in range(len(thr))], 1)
    W = np.stack([np.bincount(inv, weights=wr[:, j], minlength=len(rows)) for j in range(len(thr))], 1)
    NF = np.bincount(inv, minlength=len(rows)).astype(float)
    return {"committed": committed, "wrong": wrong, "A": A, "W": W, "NF": NF, "n_tracks": len(rows)}, checks


# ------------------------------------------------------------------ the resampling test
def draw_weights(n_tracks, n_cert, draws, seed):
    rng = np.random.default_rng(seed)
    return np.stack([np.bincount(rng.integers(0, n_tracks, n_cert), minlength=n_tracks) for _ in range(draws)]).astype(float)


def resample(P, cfg, b1, Wd, alphas, delta):
    """Violation rows per alpha for both methods. P: pool tables. Wd: (draws, tracks) weights."""
    order_t = cfg["certificate_A"]["threshold_order"]["order_positions"]
    order_b = b1["order_positions"]
    pop = np.ones((1, P["n_tracks"]))
    out = {}
    for a in alphas:
        mu_true_t = E.track_mu(P["committed"], P["wrong"], pop, a)[0][0]
        mu_true_b = E.frame_mu(P["A"], P["W"], P["NF"], pop, a)[0][0]
        mu_t, n_t = E.track_mu(P["committed"], P["wrong"], Wd, a)
        mu_b, n_b = E.frame_mu(P["A"], P["W"], P["NF"], Wd, a)
        ct, cb = E.certify(mu_t, n_t, a, order_t, delta), E.certify(mu_b, n_b, a, order_b, delta)
        out[a] = {"track": {**E.rate_row(ct, mu_true_t, a), "true_mean_L": mu_true_t},
                  "B1": {**E.rate_row(cb, mu_true_b, a), "true_mean_L": mu_true_b}}
    return out


def p1_overall(res3b, primary, delta, s3a):
    """Part A (g). 3b parts here; 3a parts from stage3a_summary.json if it exists."""
    r = res3b[primary]
    parts = {"track_part_3b": r["track"]["cp95"][1] <= delta + 0.02, "frame_part_3b": r["B1"]["cp95"][0] > delta}
    if s3a is not None:
        parts.update(s3a["p1_parts"])
    complete = len(parts) == 4
    return {"parts": {k: bool(v) for k, v in parts.items()}, "complete": complete,
            "pass": bool(complete and all(parts.values()))}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    a = ap.parse_args(argv)
    cfg = load_frozen(**FROZEN_ARGS)
    t0 = time.time()
    alphas = [float(x) for x in cfg["certificate_A"]["alpha_grid"]]
    primary, delta = float(cfg["certificate_A"]["primary_alpha"]), float(cfg["fixed_in_every_phase"]["delta"])
    n_cert = int(cfg["splits"]["cert_tracks"])
    d = load_pool(a.run, cfg)
    P, checks = pool_tables(d, cfg)
    checks["pool_size_matches_frozen"] = P["n_tracks"] == int(cfg["splits"]["pool_tracks"])
    Wd = draw_weights(P["n_tracks"], n_cert, N_DRAWS, SEED)
    res = resample(P, cfg, d["b1"], Wd, alphas, delta)
    s3a_path = Path(a.run) / "poc2" / "stage3" / "stage3a_summary.json"
    s3a = json.loads(s3a_path.read_text()) if s3a_path.exists() else None
    p1 = p1_overall(res, primary, delta, s3a)

    print(f"\n=== R8 Stage 3b: resampling validity on the {P['n_tracks']} non-design tracks "
          f"({N_DRAWS} draws of {n_cert} with replacement, delta {delta}) ===")
    print(f"{'alpha':>6} | {'method':6} {'true L(start)':>13} {'cert':>6} {'viol':>5} {'rate':>7} {'CP95':>17} {'mean R*':>8}")
    for x in alphas:
        for m, start in (("track", cfg["certificate_A"]["threshold_order"]["start_position"]),
                         ("B1", d["b1"]["start_position"])):
            r = res[x][m]
            print(f"{100 * x:5.1f}% | {m:6} {100 * r['true_mean_L'][start]:12.3f}% {100 * r['certified_share']:5.1f}% "
                  f"{r['violations']:5d} {100 * r['violation_rate']:6.2f}% [{100 * r['cp95'][0]:6.2f}, {100 * r['cp95'][1]:6.2f}]% "
                  + ("    n/a" if r["mean_true_risk_certified"] is None else f"{100 * r['mean_true_risk_certified']:7.3f}%")
                  + ("   <- primary" if x == primary and m == "track" else ""))
    print("(true L(start) = population mean loss at the first threshold tested; mean R* = mean true loss of the certified"
          " threshold over certifying runs; violation if R* > alpha)")
    print(f"\nP1 at alpha {100 * primary:.0f}% (two-sided 95% CP):")
    for k, v in p1["parts"].items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    print(f"P1: {'PASS' if p1['pass'] else ('FAIL' if p1['complete'] else 'INCOMPLETE (run 3a first)')}")
    print("\nChecks:")
    for k, v in checks.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    all_pass = all(checks.values())
    print(f"\nALL PASS (checks): {all_pass}")

    out = Path(a.run) / "poc2" / "stage3"
    summary = {"plan": "poc2-implementation-plan-r8 Stage 3b",
               "frozen_r8_sha256": sha256_file(Path(FROZEN_ARGS.get("root", FZ.ROOT)) / FZ.FROZEN),
               "code_sha256": {"r8_stage3_resample.py": sha256_file(Path(__file__)),
                               "r8_stage3_engine.py": sha256_file(Path(E.__file__))},
               "inputs_sha256": d["inputs_sha256"],
               "stage3a_summary_sha256": sha256_file(s3a_path) if s3a is not None else None,
               "config": {"draws": N_DRAWS, "n_cert": n_cert, "seed": SEED, "alphas": alphas, "delta": delta},
               "results": {str(x): res[x] for x in alphas}, "p1": p1, "checks": checks, "all_pass": all_pass,
               "seconds": round(time.time() - t0, 1)}
    (out / "stage3b_summary.json").write_text(json.dumps(S2.jsonable(summary), indent=2))
    print(f"Output: {out / 'stage3b_summary.json'} ({summary['seconds']} s)")
    print("Stage 4 is not started.")
    return summary


if __name__ == "__main__":
    main()
