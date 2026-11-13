"""POC-2 (R8) Stage 5: Certificate B status (as frozen) and the descriptive commit-by-size curve on Test.
LOG_r8_stage5_entry.md Part A (a)-(e). Confirm phase. No certification is run: every certified threshold is read from
the hash-checked stage4_summary.json, and the Stage 4 splits are regenerated with the hash-checked Stage 4 code.

  committed by size s = share of Test tracks with committed = 1 AND ROI size at commit <= s  (= 1 - late(s), frozen)
  D90 size            = smallest s (multiple of 5 px) with committed share >= 90%; "not reached" otherwise.
                        A Test description, not a certificate.

Usage:  py -3.12 -m src.r8_stage5 --run runs/20261003_stage1 --config configs/poc.yaml
Output: runs/20261003_stage1/poc2/stage5/stage5_summary.json
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import yaml

from src import r8_frozen as FZ
from src import r8_stage2 as S2
from src import r8_stage3_resample as RS
from src import r8_stage4 as S4
from src.r8_frozen import load_frozen
from src.util import sha256_file

FROZEN_ARGS = {}                       # tests point load_frozen at a fake root; production uses the F1 hashes
STAGE4_SUMMARY_SHA256 = "e66e45f210ebae5e6d21a4d8247584e1affc2d9691a760427814c821b164d2cc"   # Stage 4 Part B
STAGE4_CODE_SHA256 = "fae67390a488b44c57f1386c9b256cb0fbbcf2444282f5724c2580aa3e6445c2"      # Stage 5 Part A (c)
SHARE_AT = (25, 50, 75)
D90_LEVEL = 0.90
NOT_REACHED = "not reached"


# ------------------------------------------------------------------ curve and summary numbers (Part A (b), (d))
def committed_by_size(committed, size, s_values):
    """Share of tracks with committed AND size at commit <= s, for each s (= 1 - mean late(s), frozen definition)."""
    sz = np.where(committed, size, np.inf)
    return np.array([float((sz <= s).mean()) for s in s_values])


def s_range(max_size):
    return list(range(5, 5 * math.ceil(max_size / 5.0) + 5, 5))


def d90(curve, s_values):
    for s, v in zip(s_values, curve):
        if v >= D90_LEVEL - 1e-12:
            return int(s)
    return NOT_REACHED


def summary_numbers(curve, s_values):
    out = {f"share_at_{p}px": float(curve[s_values.index(p)]) if p in s_values else float(curve[-1]) for p in SHARE_AT}
    out["d90_size"] = d90(curve, s_values)
    return out


def report_baseline(seed_alpha, method):
    """Part A (d): a baseline's summary numbers are reported only where both it and the commit rule were certified."""
    return bool(seed_alpha["commit"]["certified"] and seed_alpha[method]["certified"])


# ------------------------------------------------------------------ main
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--config", default="configs/poc.yaml")
    a = ap.parse_args(argv)
    cfg = load_frozen(**FROZEN_ARGS)
    t0 = time.time()
    s4dir, out = Path(a.run) / "poc2" / "stage4", Path(a.run) / "poc2" / "stage5"
    assert sha256_file(s4dir / "stage4_summary.json") == STAGE4_SUMMARY_SHA256, "stage4_summary.json differs from the hash in Stage 4 Part B"
    s4 = json.loads((s4dir / "stage4_summary.json").read_text())
    code_now = sha256_file(Path(S4.__file__))
    assert code_now == s4["code_sha256"]["r8_stage4.py"] == STAGE4_CODE_SHA256, "src/r8_stage4.py differs from the code that wrote Stage 4"
    assert sha256_file(s4dir / "baselines_design.json") == S4.BASELINES_SHA256, "baselines_design.json differs from the hash in Stage 4 Part A"
    bdes = json.loads((s4dir / "baselines_design.json").read_text())
    n_test = int(cfg["splits"]["test_tracks"])

    d = RS.load_pool(a.run, cfg)                                     # hash-checked, pool rows only
    _, checks = RS.pool_tables(d, cfg)
    bdes["n_all_tracks"] = len(np.load(Path(a.run) / "splits" / "seed0.npz")["track_names"])
    roi = S4.roi_sizes(a.run, yaml.safe_load(Path(a.config).read_text()), d)
    T, y, _, c2 = S4.method_tables(d, cfg, bdes, roi)
    checks.update(c2)

    # regenerate the Stage 4 splits and collect every certified (method, seed, alpha) where the commit rule is certified
    cells, reproduce = [], True
    for row in s4["per_seed"]:
        seed = int(row["seed"])
        test = S4.make_split(y, seed, n_test)
        reproduce &= int(test.sum()) == int(row["n_test"])
        for al_key, by_m in row["alphas"].items():
            if not by_m["commit"]["certified"]:
                continue                                             # Part A (b): only alphas where the commit rule is certified
            for m in S4.METHODS:
                r4 = by_m[m]
                cell = {"seed": seed, "alpha": float(al_key), "method": m, "certified": bool(r4["certified"])}
                if r4["certified"]:
                    j = int(r4["grid_position"])
                    com, size = T[m]["committed"][test, j], T[m]["size"][test, j]
                    cov = float(com.sum() / test.sum())
                    reproduce &= cov == float(r4["coverage"])        # exact: same tables, same split, same position
                    cell.update(grid_position=j, coverage=cov, _com=com, _size=size,
                                report=(m == "commit") or report_baseline(by_m, m))
                cells.append(cell)
    checks["stage4_coverages_reproduce"] = bool(reproduce)
    assert reproduce, "the regenerated splits do not reproduce Stage 4's Test coverages; stop (Part A (c))"

    max_size = max(float(np.max(c["_size"][c["_com"]])) for c in cells if c["certified"] and c["_com"].any())
    S = s_range(max_size)
    right_end = True
    for c in cells:
        if c["certified"]:
            c["curve"] = committed_by_size(c.pop("_com"), c.pop("_size"), S)
            c["never_committed"] = 1.0 - c["coverage"]
            right_end &= abs(c["curve"][-1] - c["coverage"]) < 1e-12
            if c["report"]:
                c.update(summary_numbers(c["curve"], S))
    checks["curve_right_end_equals_coverage"] = bool(right_end)

    # ---- print
    gamma = float(cfg["certificate_B"]["gamma"])
    print("\n=== R8 Stage 5: Certificate B status and the commit-by-size curve (Test, Stage 4 splits) ===")
    print(f"Certificate B (frozen): s = {cfg['certificate_B']['s']}; gamma = {100 * gamma:.0f}%. "
          "No joint certificate is run; s and gamma are not moved.")
    print(f"Curve: share of the {n_test} Test tracks committed with ROI size at commit <= s, s = 5..{S[-1]} px "
          "(descriptive, not scored, not a certificate)")
    f = lambda x: f"{100 * x:6.2f}%"
    for al in sorted({c["alpha"] for c in cells}):
        print(f"\n--- alpha {100 * al:.1f}%: summary numbers (D90 size = smallest s with >= 90% of Test signs committed; "
              "Test description, not a certificate)")
        print(f"{'seed':>4} {'method':6} {'cov':>8} {'never':>8} {'@25px':>8} {'@50px':>8} {'@75px':>8} {'D90 size':>12}")
        for c in sorted((c for c in cells if c["alpha"] == al), key=lambda c: (c["seed"], S4.METHODS.index(c["method"]))):
            if not c["certified"]:
                print(f"{c['seed']:>4} {c['method']:6} {'not certified':>36}")
            elif not c["report"]:
                print(f"{c['seed']:>4} {c['method']:6} {f(c['coverage'])} {f(c['never_committed'])}   (commit rule not certified; no summary)")
            else:
                dv = c["d90_size"] if c["d90_size"] == NOT_REACHED else f"{c['d90_size']} px"
                print(f"{c['seed']:>4} {c['method']:6} {f(c['coverage'])} {f(c['never_committed'])} {f(c['share_at_25px'])} "
                      f"{f(c['share_at_50px'])} {f(c['share_at_75px'])} {dv:>12}")
    marks = [s for s in S if s % 10 == 0 and s <= 120]
    print(f"\n--- Curves at every 10 px (committed share; full 5 px curves in the summary JSON)")
    print(f"{'alpha':>6} {'seed':>4} {'method':6} " + " ".join(f"{s:>6}" for s in marks))
    for c in sorted((c for c in cells if c["certified"]), key=lambda c: (c["alpha"], c["seed"], S4.METHODS.index(c["method"]))):
        print(f"{100 * c['alpha']:5.1f}% {c['seed']:>4} {c['method']:6} "
              + " ".join(f"{100 * c['curve'][S.index(s)]:5.1f}%" for s in marks))
    print("\nChecks:")
    for k, v in checks.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    all_pass = all(checks.values())
    print(f"\nALL PASS (checks): {all_pass}")

    out.mkdir(parents=True, exist_ok=True)
    summary = {"plan": "poc2-implementation-plan-r8 Stage 5",
               "frozen_r8_sha256": sha256_file(Path(FROZEN_ARGS.get("root", FZ.ROOT)) / FZ.FROZEN),
               "code_sha256": {"r8_stage5.py": sha256_file(Path(__file__)), "r8_stage4.py": code_now},
               "inputs_sha256": {**d["inputs_sha256"], "stage4_summary.json": STAGE4_SUMMARY_SHA256,
                                 "baselines_design.json": S4.BASELINES_SHA256},
               "certificate_B": {"s": cfg["certificate_B"]["s"], "gamma": gamma, "status": "as frozen; not run"},
               "s_values": S, "cells": cells, "checks": checks, "all_pass": all_pass, "seconds": round(time.time() - t0, 1)}
    (out / "stage5_summary.json").write_text(json.dumps(S2.jsonable(summary), indent=2))
    print(f"Output: {out / 'stage5_summary.json'} ({summary['seconds']} s)")
    print("Stage 6 is not started.")
    return summary


if __name__ == "__main__":
    main()
