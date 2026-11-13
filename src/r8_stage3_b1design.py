"""POC-2 (R8) Stage 3, step 1: B1 (frame-level Learn-then-Test) grid and test order from the 261 DESIGN tracks only.

B1 is named in frozen_r8.yaml only by plan section 9, so its grid and order are specified here, after F1, from design
data only and before any Stage 3 number (LOG_r8_stage3_entry.md Part A (c)). Same rules as Stage 2 Part A (h) and (c),
applied to frames: every frame is its own unit.

  per-frame margin : top minus second of (log prior + frame log-likelihood) = the V1 score of a single frame
  accepted         : frame passes the gate AND margin >= threshold (gate-failing frames are never accepted)
  commit fraction c: accepted frames / all design frames
  candidate grid   : Stage 1 make_grid rule on design per-frame margins (0 plus 199 log-spaced points)
  working grid     : Stage 2 rule (h) at the primary alpha; order: Stage 2 rule (c)

Usage:  py -3.12 -m src.r8_stage3_b1design --run runs/20261003_stage1
Output: runs/20261003_stage1/poc2/stage3/b1_design.json
"""
import argparse
import json
from pathlib import Path

import numpy as np

from src import r8_stage1 as R1
from src import r8_stage2 as S2
from src.r8_frozen import load_frozen
from src.util import sha256_file


def frame_margins(head, emb, passed):
    """Single-frame V1 margin and top class; gate-failing frames get margin -inf (never accepted)."""
    ll, _ = R1.frame_evidence(head, emb)
    lp = np.log(head["priors"])[None, :] + ll
    part = np.partition(lp, -2, axis=1)
    m = part[:, -1] - part[:, -2]
    return np.where(passed, m, -np.inf), lp.argmax(1)


def candidate_grid(m):
    finite = m[np.isfinite(m)]
    return R1.make_grid(finite, finite[finite > 0])


def b1_design(m, wrong, alpha):
    """Rules (h) and (c) on frames. Returns the candidate grid, stats, working indices, start and order."""
    grid = candidate_grid(m)
    acc = m[:, None] >= grid[None, :]                                # frames x thresholds
    st = S2.grid_stats(acc, acc & wrong[:, None], np.zeros(acc.shape), np.zeros(acc.shape))
    work = S2.working_grid(st, alpha)
    out = {"grid": grid, "stats": st, "work": work, "start": None, "order": []}
    if len(work):
        out["start"], out["order"] = S2.start_and_order(S2.mean_L(st, alpha)[work])
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    a = ap.parse_args(argv)
    cfg = load_frozen()
    alpha = float(cfg["certificate_A"]["primary_alpha"])
    S2.OPENED.clear()
    d = S2.load_inputs(a.run)                                        # design rows only; refuses the pool file
    # own design-only assertions (hard rule 3), on top of the ones inside S2.load_inputs
    assert len(d["design_codes"]) == S2.N_DESIGN == 261, "design set is not the 261 seed-0 Fit tracks"
    assert np.isin(d["track"], d["design_codes"]).all(), "a frame from a non-design track reached the B1 design step"
    assert len(np.unique(d["track"])) == 261 and len(d["track"]) == int(d["table"]["n_frames"].sum()), "frame count is not the design set's"
    m, top = frame_margins(d["head"], d["emb"], d["passed"])
    assert (top == d["top1"]).all(), "single-frame top class differs from the stored POC-1 top-1"
    wrong = top != d["y"]
    r = b1_design(m, wrong, alpha)
    st, w = r["stats"], r["work"]

    print(f"\n=== R8 Stage 3 step 1: B1 grid and order (design set only: {len(m)} frames of {len(d['design_codes'])} tracks) ===")
    print(f"primary alpha {100 * alpha:.1f}% | gate-failing design frames {int((~d['passed']).sum())} (never accepted)")
    if not len(w):
        print("No frame threshold has c >= 70% and e <= alpha: B1 has no working grid at the primary alpha.")
    else:
        print(f"working grid: {len(w)} points, candidate indices {int(w[0])}..{int(w[-1])}, "
              f"threshold {r['grid'][w[0]]:.6g} .. {r['grid'][w[-1]]:.6g}")
        print(f"start: working position {r['start']} (threshold {r['grid'][w[r['start']]]:.6g}), relax to 0")
        print(f"{'pos':>3} {'idx':>4} {'threshold':>12} {'c':>8} {'e':>8} {'mean L':>8}")
        for i, j in enumerate(w):
            print(f"{i:3d} {int(j):4d} {r['grid'][j]:12.6g} {100 * st['c'][j]:7.2f}% {100 * st['e'][j]:7.3f}% "
                  f"{100 * S2.mean_L(st, alpha)[j]:7.3f}%" + ("  <- start" if i == r["start"] else ""))
    assert not any(S2.POOL_FILE in p for p in S2.OPENED), "track_table_pool.npz was opened"
    out = Path(a.run) / "poc2" / "stage3"
    out.mkdir(parents=True, exist_ok=True)
    summary = {"plan": "poc2-implementation-plan-r8 Stage 3 step 1 (B1 grid and order, design set only)",
               "frozen_r8_sha256": sha256_file(Path("configs/frozen_r8.yaml")), "code_sha256": sha256_file(Path(__file__)),
               "inputs_sha256": d["inputs_sha256"], "files_opened": [Path(p).name for p in S2.OPENED],
               "alpha": alpha, "n_frames": int(len(m)), "candidate_grid": r["grid"], "work_indices": w,
               "work_thresholds": r["grid"][w] if len(w) else [], "start_position": r["start"], "order_positions": r["order"],
               "work_c": st["c"][w] if len(w) else [], "work_e": st["e"][w] if len(w) else []}
    (out / "b1_design.json").write_text(json.dumps(S2.jsonable(summary), indent=2))
    print(f"Output: {out / 'b1_design.json'}")
    print("Design set only. No pool track was read; no Stage 3 validity number exists yet.")
    return summary


if __name__ == "__main__":
    main()
