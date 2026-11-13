"""POC-2 (R8) Stage 1: evidence accumulation and the tracks-by-lambdas commit table.
Plan: claude/poc2-implementation-plan-r8.md, sections 5 (commit rule) and 10 (Stage 1).

What it does
  1. Per-frame class log-likelihoods for all 39,209 training-release frames from the cached embeddings and the frozen
     tied-Gaussian head (same mahal_sq as Stage 3, so the numbers are the POC-1 numbers).
  2. Gate: a frame passes if its gate score s = sqrt(min_k D2) <= tau. tau is the POC-1 seed-0 value, which was fitted on
     the seed-0 Fit split = the R8 design set (plan section 3). Failing frames add no evidence.
  3. Running scores S_t(k) for the three variants (V1 sum, V2 mean, V3 last-5 window), margin M_t, top class per frame.
  4. Commit table for a wide lambda grid: commit = first frame with M_t >= lambda. Built PER LAMBDA (risk R3: the committed
     label can change with lambda).

Deviation from the plan, on purpose (plan section 2): the plan asks for 'committed, wrong' for every track. Computing
'wrong' on the 1,046 non-design tracks before the freeze would mean commit-rule errors on the future Cert/Test pool are
known before F1. So:
  - track_table_design.npz  : the 261 design tracks, WITH labels and 'wrong'.
  - track_table_pool.npz    : the 1,046 pool tracks, label-free (no y, no wrong). Errors on these are first computed in
                              Stage 4, after frozen_r8.yaml is committed.
Nothing about pool outcomes is printed except pass/fail of label-free checks.

Usage:  py -3.12 -m src.r8_stage1 --run runs/20261003_stage1 --config configs/poc.yaml
Output: runs/20261003_stage1/poc2/stage1/
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr

from src.head import mahal_sq          # the Stage 3 function, reused so frame scores match POC-1 exactly
from src.train import sha256_file

K = 43
VARIANTS = ("V1", "V2", "V3")
WINDOW = 5                             # V3 window length (plan section 5)
N_GRID = 200                           # wide grid; Stage 2 picks the ~30-point working grid from this range
DESIGN_SEED = 0                        # design set = seed-0 Fit split of POC-1 (plan section 3)
CKPT_HEAD, CKPT_TAIL = "8652735", "0bcc"   # frozen checkpoint hash as written in the plan
DESIGN_KEYS_WITH_LABELS = {"y_track", "wrong", "wrong_last"}


# ------------------------------------------------------------------ inputs
def load_inputs(run, cfg_all):
    """Read the POC-1 artefacts and refuse to go on if any of them changed since it was written."""
    run = Path(run)
    es = json.loads((run / "embeddings" / "embeddings_summary.json").read_text())
    ck = es["checkpoint_sha256"].lower()
    assert ck == (cfg_all["embed"].get("expected_checkpoint_sha256") or "").lower(), "embeddings are not from the frozen checkpoint"
    assert ck.startswith(CKPT_HEAD) and ck.endswith(CKPT_TAIL), "checkpoint hash is not the one named in the R8 plan"
    assert sha256_file(run / "embeddings" / "train_release.npz") == es["files_sha256"]["train_release.npz"], "train_release.npz changed since Stage 2"
    hs = json.loads((run / "head_emb" / "head_summary.json").read_text())
    for f in ("head.npz", "scores_train_release.npz"):
        assert sha256_file(run / "head_emb" / f) == hs["files_sha256"][f], f"{f} changed since Stage 3"
    summ = json.loads((run / "splits" / "stage4_summary.json").read_text())
    sf = f"seed{DESIGN_SEED}.npz"
    assert sha256_file(run / "splits" / sf) == summ["files_sha256"][sf], f"{sf} changed since Stage 4"

    E = np.load(run / "embeddings" / "train_release.npz")
    H = np.load(run / "head_emb" / "head.npz")
    Sc = np.load(run / "head_emb" / "scores_train_release.npz")
    Sp = np.load(run / "splits" / sf)
    names = np.asarray(E["track_names"])
    assert (np.asarray(Sp["track_names"]) == names).all(), "track order in seed0.npz differs from the embeddings"
    design = np.asarray(Sp["track_split"]) == 0                       # 0 = Fit in POC-1 = the R8 design set
    tau = float(next(r for r in summ["seeds"] if r["seed"] == DESIGN_SEED)["gate"]["tau"])

    meta = pd.read_csv(Path(cfg_all["data"]["cache_dir"]) / "train_meta.csv")
    y, track, frame = E["y"].astype(int), E["track"].astype(int), E["frame"].astype(int)
    assert len(meta) == len(y) and (meta["class_id"].to_numpy() == y).all() and (meta["frame_idx"].to_numpy() == frame).all() \
        and (meta["track_id"].astype(str).to_numpy() == names[track].astype(str)).all(), "train_meta.csv rows do not line up with the embeddings"
    # ROI size = shorter side of the sign box in ORIGINAL pixels (x2 - x1 convention, used everywhere, frozen with s at F1)
    roi = np.minimum(meta["roi_x2"] - meta["roi_x1"], meta["roi_y2"] - meta["roi_y1"]).to_numpy().astype(float)

    return {"emb": E["emb"], "y": y, "track": track, "frame": frame, "names": names, "design": design, "tau": tau, "roi": roi,
            "head": {k: H[k] for k in ("mu", "W", "priors")}, "stored_top1": Sc["top3"][:, 0].astype(int),
            "stored_s": Sc["s"] if "s" in Sc.files else None,
            "head_acc_poc1": (hs.get("train_release") or {}).get("head_accuracy"),
            "input_sha256": {"checkpoint": ck, "train_release.npz": es["files_sha256"]["train_release.npz"],
                             "head.npz": hs["files_sha256"]["head.npz"], sf: summ["files_sha256"][sf]}}


# ------------------------------------------------------------------ per-frame evidence
def frame_evidence(head, emb):
    """ll[i, k] = -0.5 * squared Mahalanobis distance to class mean k; gate score s = sqrt(min_k D2) as in POC-1."""
    D2 = mahal_sq(head, emb)
    return -0.5 * D2, np.sqrt(D2.min(1))


def track_rows(track, frame):
    """Row indices of each track, ordered by frame index. Works for any track length (the 29-frame track included)."""
    order = np.lexsort((frame, track))
    cuts = np.flatnonzero(np.diff(track[order])) + 1
    return np.split(order, cuts)


# ------------------------------------------------------------------ the commit rule (plan section 5)
def running_scores(ll, passed, log_prior, variant):
    """S_t(k) for one track (rows already in frame order). Returns S (n, K) and the number of passing frames so far.
    Before the first passing frame there is no evidence, so S is NaN there (no commit can happen on the prior alone)."""
    contrib = np.where(passed[:, None], ll, 0.0)
    npass = np.cumsum(passed)
    if variant in ("V1", "V2"):
        ev = np.cumsum(contrib, axis=0)
    elif variant == "V3":
        # sum over the last WINDOW passing frames: prefix sums over passing frames only, then a difference
        C = np.vstack([np.zeros((1, ll.shape[1])), np.cumsum(ll[passed], axis=0)])
        ev = C[npass] - C[np.maximum(npass - WINDOW, 0)]
    else:
        raise ValueError(variant)
    S = log_prior[None, :] + ev
    if variant == "V2":
        S = S / np.maximum(npass, 1)[:, None]                         # mean per passing frame (prior included, as defined)
    S[npass == 0] = np.nan
    return S, npass


def margins(S):
    """M_t = top score minus second score, and the top class. NaN rows (no evidence yet) give M = -inf, top = -1."""
    ok = ~np.isnan(S[:, 0])
    M = np.full(len(S), -np.inf)
    top = np.full(len(S), -1, dtype=int)
    if ok.any():
        part = np.partition(S[ok], -2, axis=1)
        M[ok] = part[:, -1] - part[:, -2]
        top[ok] = np.argmax(S[ok], axis=1)
    return M, top


def first_commit(M, grid):
    """For every lambda in the ascending grid: index of the first frame with M_t >= lambda, or len(M) if never.
    Uses the running maximum of M, which is non-decreasing, so one searchsorted does the whole grid."""
    return np.searchsorted(np.maximum.accumulate(M), grid, side="left")


def make_grid(max_margins, positive_margins):
    """Wide grid per variant from DESIGN-SET margins only: 0 (= commit at the first passing frame) plus a log-spaced range."""
    lo = max(float(np.quantile(positive_margins, 0.005)), 1e-3)
    hi = float(np.max(max_margins)) * 1.01
    return np.concatenate([[0.0], np.geomspace(lo, hi, N_GRID - 1)])


# ------------------------------------------------------------------ tables
def build(inp):
    head = inp["head"]
    ll_all, s = frame_evidence(head, inp["emb"])
    passed_all = s <= inp["tau"]
    log_prior = np.log(head["priors"])
    rows = track_rows(inp["track"], inp["frame"])
    tcode = np.array([inp["track"][r[0]] for r in rows])             # track code of each entry in rows
    design = inp["design"][tcode]

    per_variant = {}
    for v in VARIANTS:
        Ms, tops = [], []
        for r in rows:
            S, _ = running_scores(ll_all[r], passed_all[r], log_prior, v)
            M, top = margins(S)
            Ms.append(M); tops.append(top)
        dmax = np.array([M.max() for M, d in zip(Ms, design) if d and np.isfinite(M.max())])
        dpos = np.concatenate([M[np.isfinite(M) & (M > 0)] for M, d in zip(Ms, design) if d])
        grid = make_grid(dmax, dpos)
        T, G = len(rows), len(grid)
        committed = np.zeros((T, G), bool)
        pred = np.full((T, G), -1, np.int16)
        pos = np.full((T, G), -1, np.int16)                           # 1-based position of the commit frame in the track
        fidx = np.full((T, G), -1, np.int16)                          # GTSRB frame index at commit
        size = np.full((T, G), np.nan, np.float32)                    # ROI shorter side at commit
        for i, (r, M, top) in enumerate(zip(rows, Ms, tops)):
            idx = first_commit(M, grid)
            c = idx < len(r)
            committed[i] = c
            pred[i, c] = top[idx[c]]
            pos[i, c] = idx[c] + 1
            fidx[i, c] = inp["frame"][r[idx[c]]]
            size[i, c] = inp["roi"][r[idx[c]]]
        per_variant[v] = {"grid": grid, "committed": committed, "pred": pred, "pos": pos, "frame_idx": fidx, "size": size,
                          "top_last": np.array([t[-1] for t in tops]), "M_last": np.array([M[-1] for M in Ms])}
    return {"ll": ll_all, "s": s, "passed": passed_all, "rows": rows, "tcode": tcode, "design": design, "V": per_variant}


def split_tables(inp, B):
    """Design file gets labels and 'wrong'; pool file gets label-free columns only (plan section 2)."""
    d, tcode = B["design"], B["tcode"]
    n_frames = np.array([len(r) for r in B["rows"]])
    n_pass = np.array([int(B["passed"][r].sum()) for r in B["rows"]])
    y_track = np.array([inp["y"][r[0]] for r in B["rows"]])
    common = lambda m: {"track_names": inp["names"][tcode[m]], "n_frames": n_frames[m], "n_passing": n_pass[m]}
    des, pool = common(d), common(~d)
    des["y_track"] = y_track[d]
    for v, T in B["V"].items():
        for name in ("grid",):
            des[f"{v}_{name}"] = pool[f"{v}_{name}"] = T[name]
        for name in ("committed", "pred", "pos", "frame_idx", "size"):
            des[f"{v}_{name}"], pool[f"{v}_{name}"] = T[name][d], T[name][~d]
        des[f"{v}_wrong"] = T["committed"][d] & (T["pred"][d] != y_track[d][:, None])
        des[f"{v}_top_last"] = T["top_last"][d]
        des[f"{v}_wrong_last"] = (T["top_last"][d] != y_track[d]) & (n_pass[d] > 0)
        pool[f"{v}_top_last"] = T["top_last"][~d]
    assert not any(k.split("_", 1)[-1] in DESIGN_KEYS_WITH_LABELS or k in DESIGN_KEYS_WITH_LABELS for k in pool), "label column in the pool file"
    return des, pool


# ------------------------------------------------------------------ checks (plan Stage 1)
def independent_last_argmax(inp, B):
    """Second route to the V1 last-frame class: sum of squared distances expanded as sum||xw||^2 - 2 m.sum(xw) + n||m||^2,
    so only the SUM of whitened passing embeddings is needed (no per-frame D2, no cumsum)."""
    W, mu = inp["head"]["W"].astype(np.float64), inp["head"]["mu"].astype(np.float64)
    Mw = mu @ W
    out = np.full(len(B["rows"]), -1)
    for i, r in enumerate(B["rows"]):
        p = r[B["passed"][r]]
        if len(p) == 0:
            continue
        xs = (inp["emb"][p].astype(np.float64) @ W).sum(0)
        score = np.log(inp["head"]["priors"]) - 0.5 * (-2 * Mw @ xs + len(p) * (Mw ** 2).sum(1))   # the sum||xw||^2 term is the same for all k
        out[i] = int(np.argmax(score))
    return out


def run_checks(inp, B):
    res, ok = {}, {}
    lp = np.log(inp["head"]["priors"])[None, :] + B["ll"]
    top1 = lp.argmax(1)
    acc = float((top1 == inp["y"]).mean())
    res["frame_head_accuracy"] = acc
    ok["C1a_top1_equals_stored_stage3"] = bool((top1 == inp["stored_top1"]).all())
    ok["C1b_accuracy_equals_poc1"] = bool(inp["head_acc_poc1"] is None or abs(acc - float(inp["head_acc_poc1"])) < 1e-9)
    if inp["stored_s"] is not None:
        ok["C1c_gate_score_equals_stored"] = bool(np.allclose(B["s"], inp["stored_s"], rtol=1e-4, atol=1e-3))

    has = np.array([B["passed"][r].any() for r in B["rows"]])
    alt = independent_last_argmax(inp, B)
    ok["C2_V1_last_top_equals_independent_argmax"] = bool((B["V"]["V1"]["top_last"][has] == alt[has]).all())

    rhos = []
    for r in B["rows"]:
        if np.ptp(inp["roi"][r]) > 0:
            rhos.append(spearmanr(inp["frame"][r], inp["roi"][r])[0])
    res["roi_vs_frame_spearman_median"] = float(np.median(rhos))
    res["roi_vs_frame_spearman_frac_positive"] = float(np.mean(np.array(rhos) > 0))
    ok["C3_roi_grows_with_frame_index"] = res["roi_vs_frame_spearman_median"] >= 0.5      # sanity level, not a research number

    lens = np.array([len(r) for r in B["rows"]])
    res["track_lengths"] = {int(k): int(v) for k, v in zip(*np.unique(lens, return_counts=True))}
    short = np.flatnonzero(lens == lens.min())
    ok["C4_short_track_handled"] = bool(all((B["V"][v]["pos"][short] <= lens.min()).all() for v in VARIANTS))

    ok["C5_commit_fraction_never_rises_with_lambda"] = bool(all(
        (np.diff(B["V"][v]["committed"].mean(0)) <= 0).all() and (np.diff(B["V"][v]["committed"].astype(int), axis=1) <= 0).all()
        for v in VARIANTS))
    ok["C6_design_set_is_261_tracks"] = int(B["design"].sum()) == 261

    res["frames_failing_gate_frac"] = float((~B["passed"]).mean())
    res["tracks_with_no_passing_frame"] = int((~has).sum())
    res["tau"] = inp["tau"]
    return res, ok


# ------------------------------------------------------------------ main
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--config", default="configs/poc.yaml")
    a = ap.parse_args(argv)
    cfg_all = yaml.safe_load(Path(a.config).read_text())
    inp = load_inputs(a.run, cfg_all)
    B = build(inp)
    res, ok = run_checks(inp, B)
    des, pool = split_tables(inp, B)

    out = Path(a.run) / "poc2" / "stage1"
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "track_table_design.npz", **des)
    np.savez_compressed(out / "track_table_pool.npz", **pool)
    np.savez_compressed(out / "frame_evidence.npz", s=B["s"].astype(np.float32), passed=B["passed"])
    summary = {"plan": "poc2-implementation-plan-r8 Stage 1", "inputs_sha256": inp["input_sha256"], "results": res, "checks": ok,
               "all_pass": all(ok.values()), "window": WINDOW, "n_grid": N_GRID,
               "grid_range": {v: [float(B["V"][v]["grid"][1]), float(B["V"][v]["grid"][-1])] for v in VARIANTS},
               "files_sha256": {f: sha256_file(out / f) for f in ("track_table_design.npz", "track_table_pool.npz", "frame_evidence.npz")}}
    (out / "stage1_summary.json").write_text(json.dumps(summary, indent=2))

    print("\n=== R8 Stage 1 summary ===")
    print(f"tau (seed-0 Fit = design set) {res['tau']:.4f} | frames failing gate {100 * res['frames_failing_gate_frac']:.2f}% | "
          f"tracks with no passing frame {res['tracks_with_no_passing_frame']}")
    print(f"frame head accuracy {100 * res['frame_head_accuracy']:.3f}% (POC-1: {inp['head_acc_poc1']})")
    print(f"ROI vs frame index: median per-track Spearman {res['roi_vs_frame_spearman_median']:.3f}, "
          f"{100 * res['roi_vs_frame_spearman_frac_positive']:.1f}% of tracks positive")
    print(f"track lengths {res['track_lengths']}")
    for v in VARIANTS:
        g = B["V"][v]["grid"]
        print(f"{v}: lambda grid 0 + [{g[1]:.4g} .. {g[-1]:.4g}] ({len(g)} points, from design-set margins only)")
    print("\nChecks:")
    for k, v in ok.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    print(f"\nALL PASS: {summary['all_pass']}")
    print(f"Output: {out}")
    print("No error number on the 1,046 pool tracks was computed or printed (plan section 2).")
    return summary


if __name__ == "__main__":
    main()
