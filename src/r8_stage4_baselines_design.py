"""POC-2 (R8) Stage 4, step 1: B2, B3, B4 grids and test orders from the 261 DESIGN tracks only.
LOG_r8_stage4_entry.md Part A (e). Same rules as Stage 2 Part A (h) and (c); one score and one label per track.

  B2 : one random gate-passing frame per track (Amendment 1; u_i from default_rng(20261114) over all 1,307 tracks in
       track-code order; no passing frame = rejected as unknown)
  B3 : the last frame (largest frame index)
  B4 : majority vote over passing frames; score = votes(label) - votes(runner-up); ties to the smaller class id
Gate-failing frames never give a label (B2/B3: track not committed; B4: frame does not vote).

Usage:  py -3.12 -m src.r8_stage4_baselines_design --run runs/20261003_stage1
Output: runs/20261003_stage1/poc2/stage4/baselines_design.json
"""
import argparse
import json
from pathlib import Path

import numpy as np

from src import r8_stage1 as R1
from src import r8_stage2 as S2
from src.r8_frozen import load_frozen
from src.r8_stage3_b1design import frame_margins
from src.util import sha256_file

B2_SEED = 20261114
BASELINES = ("B2", "B3", "B4")


def b2_positions(n_all, rows, passed, codes):
    """Amendment 1 (1): one frame chosen uniformly among the track's gate-passing frames. u is drawn once for ALL
    1,307 tracks in track-code order; the chosen frame is passing frame number floor(u * n_passing) in frame order.
    Returns the 0-based position in the track, or -1 for a track with no passing frame (rejected as unknown)."""
    u = np.random.default_rng(B2_SEED).random(n_all)
    pos = np.full(len(rows), -1)
    for i, r in enumerate(rows):
        ok = np.flatnonzero(passed[r])
        if len(ok):
            pos[i] = ok[int(np.floor(u[codes[i]] * len(ok)))]
    return pos


def baseline_scores(rows, m, top, passed, pos_b2, K=R1.K):
    """Per track: score (-inf = not committed), label, 1-based commit position for B2, B3, B4."""
    out = {b: {"score": np.full(len(rows), -np.inf), "label": np.full(len(rows), -1), "pos": np.zeros(len(rows), int)}
           for b in BASELINES}
    for i, r in enumerate(rows):
        for b, f in (("B2", pos_b2[i]), ("B3", len(r) - 1)):
            if f < 0:                                                   # B2: no passing frame -> rejected as unknown
                continue
            out[b]["pos"][i] = f + 1
            if passed[r[f]]:                                            # B3: a gated-out last frame gives no label
                out[b]["score"][i], out[b]["label"][i] = m[r[f]], top[r[f]]
        out["B4"]["pos"][i] = len(r)
        p = r[passed[r]]
        if len(p):
            votes = np.bincount(top[p], minlength=K)
            lab = int(np.argmax(votes))                                # first maximum = smallest class id on ties
            second = np.partition(votes, -2)[-2]
            out["B4"]["score"][i], out["B4"]["label"][i] = float(votes[lab] - second), lab
    return out


def design_rule(score, label, y, alpha):
    """Stage 1 make_grid on design scores, then Stage 2 rules (h) and (c)."""
    finite = score[np.isfinite(score)]
    grid = R1.make_grid(finite, finite[finite > 0])
    committed = score[:, None] >= grid[None, :]
    wrong = committed & (label != y)[:, None]
    z = np.zeros(committed.shape)
    st = S2.grid_stats(committed, wrong, z, z)
    work = S2.working_grid(st, alpha)
    start, order = (S2.start_and_order(S2.mean_L(st, alpha)[work]) if len(work) else (None, []))
    return {"grid": grid, "stats": st, "work": work, "start": start, "order": order}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    a = ap.parse_args(argv)
    cfg = load_frozen()
    alpha = float(cfg["certificate_A"]["primary_alpha"])
    S2.OPENED.clear()
    d = S2.load_inputs(a.run)                                        # design rows only; refuses the pool file
    # own design-only assertions (hard rule 3)
    assert len(d["design_codes"]) == S2.N_DESIGN == 261, "design set is not the 261 seed-0 Fit tracks"
    assert np.isin(d["track"], d["design_codes"]).all(), "a frame from a non-design track reached the baseline design step"
    n_all = len(np.load(Path(a.run) / "splits" / "seed0.npz")["track_names"])   # 1,307 track codes, label-free
    rows = R1.track_rows(d["track"], d["frame"])
    codes = np.array([d["track"][r[0]] for r in rows])
    assert (codes == d["design_codes"]).all(), "design track order differs"
    m, top = frame_margins(d["head"], d["emb"], d["passed"])
    assert (top == d["top1"]).all(), "single-frame top class differs from the stored POC-1 top-1"
    y = np.array([d["y"][r[0]] for r in rows])
    B = baseline_scores(rows, m, top, d["passed"], b2_positions(n_all, rows, d["passed"], codes))
    assert not any(S2.POOL_FILE in p for p in S2.OPENED), "track_table_pool.npz was opened"

    print(f"\n=== R8 Stage 4 step 1: B2, B3, B4 grids and orders (design set only: {len(rows)} tracks) ===")
    res = {}
    for b in BASELINES:
        r = design_rule(B[b]["score"], B[b]["label"], y, alpha)
        st, w = r["stats"], r["work"]
        nc = int(np.isfinite(B[b]["score"]).sum())
        print(f"\n{b}: tracks with a label {nc}/261; design error at lambda 0 (commit every labelled track) "
              f"{100 * st['e'][0]:.2f}%")
        if not len(w):
            print(f"  no working grid at {100 * alpha:.0f}% (rule (h)): {b} certifies nothing at any alpha")
        else:
            print(f"  working grid {len(w)} points, candidate indices {int(w[0])}..{int(w[-1])}, "
                  f"threshold {r['grid'][w[0]]:.6g} .. {r['grid'][w[-1]]:.6g}; start position {r['start']} "
                  f"(threshold {r['grid'][w[r['start']]]:.6g}, c {100 * st['c'][w[r['start']]]:.2f}%, "
                  f"e {100 * st['e'][w[r['start']]]:.3f}%), relax to 0")
        res[b] = {"candidate_grid": r["grid"], "work_indices": w, "work_thresholds": r["grid"][w] if len(w) else [],
                  "start_position": r["start"], "order_positions": r["order"],
                  "work_c": st["c"][w] if len(w) else [], "work_e": st["e"][w] if len(w) else [],
                  "design_labelled_tracks": nc}
    out = Path(a.run) / "poc2" / "stage4"
    out.mkdir(parents=True, exist_ok=True)
    summary = {"plan": "poc2-implementation-plan-r8 Stage 4 step 1 (B2, B3, B4 grids and orders, design set only)",
               "frozen_r8_sha256": sha256_file(Path("configs/frozen_r8.yaml")), "code_sha256": sha256_file(Path(__file__)),
               "inputs_sha256": d["inputs_sha256"], "files_opened": [Path(p).name for p in S2.OPENED],
               "alpha": alpha, "b2_seed": B2_SEED, "baselines": res}
    (out / "baselines_design.json").write_text(json.dumps(S2.jsonable(summary), indent=2))
    print(f"\nOutput: {out / 'baselines_design.json'}")
    print("Design set only. No pool track was read; no Stage 4 number exists yet.")
    return summary


if __name__ == "__main__":
    main()
