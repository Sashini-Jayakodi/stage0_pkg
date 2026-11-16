"""EXPLORATORY (post hoc; no certificate, no criteria). Transfer bound from the pooled committed error to each class
group, Grouping 2 of c0 (r1 right-of-way {14, 13, 17, 12}, r2 speed limits {0-5, 7, 8}, r3 others incl. class 6).

Every GTSRB track has been seen before, so nothing here is confirmatory. Inputs: the hash-checked c0_summary.json
(counts at the full-pool certified thresholds, n = 1,046); nothing is recomputed from the pool. Output:
runs/20261003_stage1/c1/.

Per group g, score and alpha, with T = 1,046 pool tracks, n_c committed, k committed errors, n_g committed in g:
  E[c] = n_c / T; w_g = n_g / T (group commit mass); E_pool = k / T (pooled error mass).
  Share w_g / E[c] = n_g / n_c; pooled committed error E_pool / E[c] = k / n_c.
  Scaled to N = 784 (round half up): k_N = round(E_pool N), w_N = round(w_g N).
  U_g = CP_upper(k_N, N, 0.025) / CP_lower(w_N, N, 0.025), one-sided Clopper-Pearson bounds. The group's error mass is
  at most the pooled error mass, so P(wrong | committed, g) <= E_pool / w_g whatever the split of errors across
  groups; the two bounds at 0.025 each give level delta/2 = 0.05 by a union bound. Transfer certifies if U_g < alpha.
  Largest pooled committed error the transfer would accept (E[c], w_N held fixed): x_max = largest integer error count
  at N with CP_upper(x, N, 0.025) < alpha CP_lower(w_N, N, 0.025), found by search; since k_N = round(e E[c] N), every
  pooled committed error e < e_max = (x_max + 0.5) / (E[c] N) gives x <= x_max (supremum, not attained).
  r2 only: feasibility of the direct test at its observed error, scheme b (c0: E[committed in g per 784 draw] >= m_all).

Usage:  py -3.12 -m src.c1_transfer_check --run runs/20261003_stage1
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from src import a0_silent_error_check as A0
from src import r8_stage2 as S2
from src import r9_certify as C
from src import r9_frozen as FZ
from src.util import sha256_file

LABEL = "EXPLORATORY (post hoc; no certificate, no criteria; GTSRB tracks all seen before)"
C0_SUMMARY_SHA256 = "ade431f022ed0cfe39bf91a2aedf0550098d2f186e286393ffe1a2773ab1ad44"   # LOG.md, c0 entry
GROUPING = "Grouping 2 (merged)"
SCORES = ("S1", "S2-sum", "S2-min", "S3", "S4")
ALPHAS = (0.01, 0.02, 0.03, 0.05)
N = 784
T_POOL = 1046
A_SIDE = 0.025                     # each one-sided bound; union = delta / 2


def transfer_bound(k, n_g, T=T_POOL, n=N, a=A_SIDE):
    """(U_g, k_N, w_N, upper, lower). U_g is inf when the lower bound is 0 (w_N = 0)."""
    kN, wN = A0.round_half_up(k / T * n), A0.round_half_up(n_g / T * n)
    up, lo = A0.cp_upper(kN, n, a), A0.cp_lower(wN, n, a)
    return (up / lo if lo > 0 else float("inf")), kN, wN, up, lo


def max_pooled_error(alpha, wN, n_c, T=T_POOL, n=N, a=A_SIDE):
    """(x_max, e_max): largest error count at N the transfer accepts and the supremum pooled committed error; None if
    even 0 errors fail."""
    lo = A0.cp_lower(wN, n, a)
    xs = np.arange(0, n + 1)
    ok = np.array([A0.cp_upper(int(x), n, a) < alpha * lo for x in xs])
    if not ok.any():
        return None, None
    x_max = int(xs[ok].max())                                       # cp_upper rises with x, so ok is a prefix
    return x_max, (x_max + 0.5) / (n_c / T * n)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    a = ap.parse_args(argv)
    t0 = time.time()
    blobs = FZ.check_frozen()
    assert C.DELTA == 0.1 and 2 * A_SIDE == C.DELTA / 2
    run = Path(a.run)
    cpath = run / "c0" / "c0_summary.json"
    A0.verify_sha(cpath, C0_SUMMARY_SHA256)                          # hard rule 2
    c0 = json.loads(cpath.read_text())
    assert c0["n_pool"] == T_POOL and c0["config"]["n_cal"] == N
    checks = {"c0_all_pass": c0["all_pass"] is True,
              "grouping_2": c0["config"]["groupings"][GROUPING] == {"r1 right-of-way": [14, 13, 17, 12],
                                                                    "r2 speed limits": [0, 1, 2, 3, 4, 5, 7, 8],
                                                                    "r3 others": [6] + list(range(9, 12)) + [15, 16] + list(range(18, 43))}}
    res = []
    for s in SCORES:
        for al in ALPHAS:
            cell = c0["results"][GROUPING][s][str(al)]
            if cell["j"] < 0:
                res.append({"score": s, "alpha": al, "certified": False})
                continue
            nc, k = cell["pooled"]["n_committed"], cell["pooled"]["errors"]
            for r in cell["groups"]:
                ng = r["n_committed"]
                U, kN, wN, up, lo = transfer_bound(k, ng)
                xm, em = max_pooled_error(al, wN, nc)
                row = {"score": s, "alpha": al, "certified": True, "j": cell["j"], "group": r["group"],
                       "n_c": nc, "k": k, "n_g": ng, "share": ng / nc, "pooled_error": k / nc,
                       "k_N": kN, "w_N": wN, "cp_upper_k": up, "cp_lower_w": lo, "U_g": U, "U_lt_alpha": bool(U < al),
                       "x_max": xm, "e_max": em}
                checks[f"{s}_{al}_{r['group']}_xmax_consistent"] = (xm is None and U >= al) or (xm is not None and (kN <= xm) == (U < al))
                if r["group"].startswith("r2"):
                    row.update(exp_committed_784=r["exp_committed_784"], m_all=r["b_m_all"], m_note=r["b_note"],
                               feasible=bool(r["b_feasible_obs"]))
                res.append(row)

    # ---- printout
    pc = lambda x, nd=2: "-" if x is None else ("inf" if x == float("inf") else f"{100 * x:.{nd}f}")
    print(f"\n=== c1 transfer check (Grouping 2): {LABEL} ===")
    print(f"Freeze check passed: {', '.join(f'{k} {v[:8]}' for k, v in blobs.items())}")
    print(f"Input c0_summary.json {C0_SUMMARY_SHA256[:12]} (hash-checked) | pool {T_POOL} tracks, full-pool thresholds | "
          f"N = {N} | one-sided CP at {A_SIDE} each (union {2 * A_SIDE:g} = delta/2)")
    print("Columns: share = w_g/E[c]; e_pool = E_pool/E[c]; k_N, w_N = pooled errors and group commits scaled to 784; "
          "U_g = CP_up(k_N)/CP_lo(w_N); x_max = largest k_N the transfer accepts; e_max = sup pooled committed error "
          "accepted; r2 direct = expected committed per 784 draw vs stable minimum (scheme b, observed error).")
    hdr = ["score", "a%", "group", "n_c", "k", "n_g", "share%", "e_pool%", "k_N", "w_N", "CPup%", "CPlo%", "U_g%",
           "U<a", "x_max", "e_max%", "r2 direct"]
    lines = []
    for r in res:
        if not r["certified"]:
            lines.append([r["score"], f"{100 * r['alpha']:g}", "not certified"] + ["-"] * 14)
            continue
        if "feasible" in r:
            m = r["m_all"] if r["m_all"] is not None else r["m_note"]
            direct = f"{r['exp_committed_784']:.1f} vs {m}: {'FEASIBLE' if r['feasible'] else 'NOT'}"
        else:
            direct = "-"
        lines.append([r["score"], f"{100 * r['alpha']:g}", r["group"], str(r["n_c"]), str(r["k"]), str(r["n_g"]),
                      pc(r["share"], 1), pc(r["pooled_error"], 3), str(r["k_N"]), str(r["w_N"]), pc(r["cp_upper_k"], 3),
                      pc(r["cp_lower_w"], 2), pc(r["U_g"], 3), "yes" if r["U_lt_alpha"] else "no",
                      "none" if r["x_max"] is None else str(r["x_max"]), pc(r["e_max"], 3), direct])
    w = [max(len(h), *(len(x[i]) for x in lines)) for i, h in enumerate(hdr)]
    print("  ".join(h.rjust(w[i]) for i, h in enumerate(hdr)))
    prev = None
    for x in lines:
        if prev is not None and x[:2] != prev:
            print()
        prev = x[:2]
        print("  ".join(c.rjust(w[i]) for i, c in enumerate(x)))
    all_pass = all(checks.values())
    print(f"\nChecks: {sum(checks.values())}/{len(checks)} PASS" + ("" if all_pass else
          " | FAIL: " + ", ".join(k for k, v in checks.items() if not v)))

    od = run / "c1"
    od.mkdir(exist_ok=True)
    summary = {"label": LABEL, "frozen_blobs": blobs,
               "code_sha256": {"c1_transfer_check.py": sha256_file(Path(__file__)),
                               "a0_silent_error_check.py": sha256_file(Path(A0.__file__))},
               "inputs_sha256": {"c0_summary.json": C0_SUMMARY_SHA256},
               "config": {"grouping": GROUPING, "alphas": ALPHAS, "n": N, "t_pool": T_POOL, "cp_side": A_SIDE,
                          "rounding": "half up", "direct_test": "c0 scheme b, observed error, stable minimum"},
               "results": res, "checks": checks, "all_pass": all_pass, "seconds": round(time.time() - t0, 1)}
    (od / "c1_summary.json").write_text(json.dumps(S2.jsonable(summary), indent=1))
    print(f"Output: {od / 'c1_summary.json'} ({summary['seconds']} s). {LABEL}")
    return summary


if __name__ == "__main__":
    main()
