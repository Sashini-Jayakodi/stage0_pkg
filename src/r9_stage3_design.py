"""POC-3 (R9) Stage 3, design phase (exploratory, post-hoc): S2 calibration on two design halves, grids, design
statistics, starts and the uncertified reference, from the 261 design tracks only (LOG_r9_stage3_entry.md Part A).

No Cert or Test track is read: the design loader of POC-2 Stage 2 refuses the pool table and keeps design rows only.

Usage:  py -3.12 -m src.r9_stage3_design --run runs/20261003_stage1
Output: runs/20261003_stage1/poc3/stage3/stage3_design.json
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from src import r8_stage2 as S2
from src import r9_frozen as FZ
from src import r9_stage3_scores as SC
from src.util import sha256_file

FORBIDDEN = ("btsc",)                       # hard rule 3: no BelgiumTSC file in a GTSRB design step


def pct(x):
    return "   n/a" if x is None or not np.isfinite(x) else f"{100 * x:6.2f}"


def run_design(d):
    """Everything Part A (e) to (i) fixes on the design set. d = output of S2.load_inputs (design rows only)."""
    rc = S2.recompute(d)
    ok = S2.reproduces(d, rc)
    assert all(ok.values()), f"design tracks do not reproduce Stage 1: {ok}"
    rows = rc["rows"]
    y_track = np.array([d["y"][r[0]] for r in rows])
    F = SC.frame_inputs(d["head"], d["emb"], d["passed"], d["track"], d["frame"])
    assert all((a == b).all() for a, b in zip(F["rows"], rows)), "track order differs from the Stage 2 recomputation"
    A, B = SC.halves(y_track)
    models = {s: SC.fit_s2(F, y_track, A, B, SC.S2_COMBO[s]) for s in ("S2-sum", "S2-min")}
    seqs = SC.sequences(F, models)
    out = {}
    for s in SC.SCORES:
        cmax = SC.max_conf(seqs[s])
        grid = SC.make_grid(cmax)
        tab = SC.track_table(seqs[s], y_track, grid)
        st = SC.design_stats(tab)
        out[s] = {"grid": grid, "stats": st, "starts": SC.starts(st["f_d"], len(rows)), "reference": SC.reference(st),
                  "tracks_without_value": int((~np.isfinite(cmax)).sum())}
    return {"y_track": y_track, "A": A, "B": B, "models": models, "scores": out}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    a = ap.parse_args(argv)
    t0 = time.time()
    blobs = FZ.check_frozen()                                          # Part A (b): first, before any data
    S2.OPENED.clear()
    d = S2.load_inputs(a.run)                                          # design rows only; refuses the pool table
    n = len(d["design_codes"])
    assert n == S2.N_DESIGN, f"design set has {n} tracks, expected {S2.N_DESIGN}"
    R = run_design(d)
    # hard rule 3 assertions: only design files were opened, no pool table, no BelgiumTSC path
    assert not any(S2.POOL_FILE in p for p in S2.OPENED), "the pool table was opened"
    assert not any(f in p.lower() for p in S2.OPENED for f in FORBIDDEN), "a BelgiumTSC path was opened"

    print(f"\n=== R9 Stage 3 design phase (exploratory, post-hoc): {n} design tracks only ===")
    print(f"S2 halves: A (ICP) {len(R['A'])} tracks, B (ECDF and a, b) {len(R['B'])} tracks; K = {SC.K_WINDOW}")
    for s, m in R["models"].items():
        tag = "primary" if s == SC.PRIMARY_S2 else "secondary"
        print(f"  {s} ({tag}): a = {m['a']:+.1f}, b = {m['b']:+.1f}, AURC {m['aurc']:.5f}, "
              f"ICP scores {len(m['cal'])}, ECDF values {len(m['ecdf'])}")
    for s in SC.SCORES:
        S = R["scores"][s]
        g, st = S["grid"], S["stats"]
        print(f"\n{s}: {len(g)} distinct grid points (of {SC.N_POINTS}); tracks with no value {S['tracks_without_value']}")
        print(f"  lambda_1 .. lambda_max: {g[1]:.6g} .. {g[-1]:.6g}" if len(g) > 1 else "  only lambda_0")
        print(f"  {'j':>3} {'lambda':>12} {'f_d %':>7} {'e_d %':>7} {'wrong':>5} {'med pos':>7}")
        for j in range(len(g)):
            lam = "-inf" if not np.isfinite(g[j]) else f"{g[j]:.6g}"
            print(f"  {j:>3} {lam:>12} {pct(st['f_d'][j])} {pct(st['e_d'][j])} {int(st['wrong'][j]):>5} "
                  f"{st['median_pos'][j]:7.1f}")
        print(f"  starts (grid index; -1 = no start possible) and reference (loosest j with e_d <= alpha):")
        print(f"  {'alpha':>6} " + " ".join(f"{r:>6}" for r in SC.RULES) + f" {'ref':>5}")
        for al in SC.ALPHAS:
            print(f"  {100 * al:5.1f}% " + " ".join(f"{S['starts'][al][r]:>6}" for r in SC.RULES)
                  + f" {S['reference'][al]:>5}")
    print("\nalpha 0.5%: no start possible at n = 784 for the exact binomial (needs design commit fraction >= 102%).")

    outdir = Path(a.run) / "poc3" / "stage3"
    outdir.mkdir(parents=True, exist_ok=True)
    code = {f"{m}.py": sha256_file(Path(__file__).parent / f"{m}.py") for m in ("r9_stage3_design", "r9_stage3_scores")}
    summary = {
        "plan": "poc3-implementation-plan-r9 Stage 3, design phase (exploratory, post-hoc); LOG_r9_stage3_entry.md Part A",
        "frozen_blobs": blobs, "code_sha256": code, "inputs_sha256": d["inputs_sha256"],
        "files_opened": [Path(p).name for p in S2.OPENED], "n_design": n, "n_cert": SC.N_CERT,
        "alphas": list(SC.ALPHAS), "rules": list(SC.RULES), "primary_s2": SC.PRIMARY_S2, "k_window": SC.K_WINDOW,
        "halves": {"salt": list(SC.HALF_SALT), "A_codes": d["design_codes"][R["A"]], "B_codes": d["design_codes"][R["B"]]},
        "s2_models": {s: {"a": m["a"], "b": m["b"], "aurc": m["aurc"], "combo": m["combo"], "K": m["K"],
                          "icp_cal": m["cal"], "ecdf": m["ecdf"]} for s, m in R["models"].items()},
        "scores": {s: {"grid_note": "grid[0] = -inf (stored as null): commit at the first frame with a value",
                       "grid": S["grid"], "f_d": S["stats"]["f_d"], "e_d": S["stats"]["e_d"],
                       "committed": S["stats"]["committed"], "wrong": S["stats"]["wrong"],
                       "median_pos": S["stats"]["median_pos"], "tracks_without_value": S["tracks_without_value"],
                       "starts": {str(al): S["starts"][al] for al in SC.ALPHAS},
                       "reference": {str(al): S["reference"][al] for al in SC.ALPHAS}}
                   for s, S in R["scores"].items()},
        "note_alpha_0.5": "no start possible at n = 784 for the exact binomial (needs design commit fraction >= 102%)",
        "seconds": round(time.time() - t0, 1)}
    (outdir / "stage3_design.json").write_text(json.dumps(S2.jsonable(summary), indent=1))
    print(f"\nOutput: {outdir / 'stage3_design.json'} ({summary['seconds']} s)")
    print("Design tracks only. No Cert or Test track was read; no Cert number exists.")
    return summary


if __name__ == "__main__":
    main()
