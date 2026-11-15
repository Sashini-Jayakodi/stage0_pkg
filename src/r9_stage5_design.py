"""POC-3 (R9) Stage 5, BelgiumTSC design phase: everything LOG_r9_stage5_entry.md Part A fixes on the 133 design
poles, plus the label-free Cert/Test re-splits of the pool poles. Writes runs/btsc/stage5/stage5_design.json.

Design poles only (hard rule 3): pool score and embedding files are refused, every testing label comes through the
pre-F3 guard, and the script stops if F3 is already closed (the design phase is over then).

Usage:  py -3.12 -m src.r9_stage5_design
"""
import argparse
import datetime
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from src import r9_frozen as FZ
from src import r9_stage5_scores as SC
from src.head import score
from src.r8_stage2 import jsonable
from src.r9_btsc_guard import GuardedTestLabels, f3_closed
from src.r9_stage4_data import read_table, stop
from src.util import sha256_file

STAGE4 = Path("runs/btsc/stage4")
OUT = Path("runs/btsc/stage5")
# Part A (a): Stage 4 Part B value, and git blob IDs of the text inputs as committed in 7a1d985
STAGE4_SUMMARY_SHA256 = "1c66a0119d39473f818f1add56c8a878d32ddfcc91aef4b0f5a9bd76568a3a31"
TEXT_BLOBS = {"stage4_summary.json": "3d0d79c42e3a782e301656acdb0a24515646ebab",
              "split_poles.json": "a7affda109adab1332994484c974c07b5fc3eabe",
              "test_views.csv": "e486d6f1a58d52aed01f5366673b7c07a32a3d5b"}
INPUTS = ("test_views.csv", "split_poles.json", "embeddings/test_design.npz", "head/head.npz",
          "head/scores_test_design.npz")
FORBIDDEN = ("embeddings/test_pool.npz", "head/scores_test_pool.npz", "cache/", "model/")
EXPECTED = {"design_poles": 133, "design_units": 157, "design_views": 465, "pool_poles": 530, "pool_units": 626,
            "views_correct": 451, "units_correct_last_view": 150}
CODE = ("src/r9_stage5_design.py", "src/r9_stage5_scores.py")
OPENED = []


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)

    def flush(self):
        for st in self.streams:
            st.flush()


# ------------------------------------------------------------------ inputs (a)
def check_text(path, sha_want, blob_want):
    """A text input passes on its recorded SHA-256 or on its CRLF-normalised git blob ID; otherwise stop."""
    if sha256_file(path) == sha_want:
        return "sha256"
    if FZ.blob_id(path) == blob_want:
        return "blob"
    stop(f"{path} differs from its recorded SHA-256 and git blob ID")


def open_input(base, rel, summary, blobs=TEXT_BLOBS):
    """Hash-check one Stage 4 file and record that it was opened; pool score/embedding files are refused."""
    if any(rel == f or rel.startswith(f) for f in FORBIDDEN):
        raise AssertionError(f"{rel} is a pool or model file; refused in the design phase (hard rule 3)")
    p = Path(base) / rel
    if not p.is_file():
        stop(f"{p} is missing")
    want = summary["files_sha256"][rel]
    if rel in blobs:
        how = check_text(p, want, blobs[rel])
    elif sha256_file(p) == want:
        how = "sha256"
    else:
        stop(f"{p} changed since stage4_summary.json recorded it")
    OPENED.append(rel)
    return p, how


def load_stage4(base=STAGE4, s4_sha=STAGE4_SUMMARY_SHA256, blobs=TEXT_BLOBS):
    sp = Path(base) / "stage4_summary.json"
    if not sp.is_file():
        stop(f"{sp} is missing")
    checks = {"stage4_summary.json": check_text(sp, s4_sha, blobs["stage4_summary.json"])}
    OPENED.append("stage4_summary.json")
    summary = json.loads(sp.read_text())
    P = {}
    for rel in INPUTS:
        P[rel], checks[rel] = open_input(base, rel, summary, blobs)
    return {"rows": read_table(P["test_views.csv"]), "poles": json.loads(P["split_poles.json"].read_text()),
            "emb": dict(np.load(P["embeddings/test_design.npz"])), "head": dict(np.load(P["head/head.npz"])),
            "scores": dict(np.load(P["head/scores_test_design.npz"])), "checks": checks,
            "inputs_sha256": {"stage4_summary.json": sha256_file(sp),
                              **{rel: sha256_file(P[rel]) for rel in INPUTS}},
            "inputs_blob": {rel: FZ.blob_id(Path(base) / rel) for rel in blobs}}


# ------------------------------------------------------------------ design phase
def run_design(d):
    rows, poles, E, head, S4 = d["rows"], d["poles"], d["emb"], d["head"], d["scores"]
    design_poles = sorted(poles["design_poles"])
    integ = {}
    # design rows of the npz must be exactly the design rows of the view table
    part = np.array([r["part"] for r in rows])
    rr = E["row"]
    integ["npz_rows_are_design_rows"] = bool(np.array_equal(np.sort(rr), np.flatnonzero(part == "design")))
    integ["npz_matches_view_table"] = bool(all(
        (rows[i]["pole"], rows[i]["view"], rows[i]["size"]) == (int(p), int(v), int(s))
        for i, p, v, s in zip(rr, E["pole"], E["view"], E["size"])))
    integ["design_poles_match_split"] = sorted(set(E["pole"].tolist())) == design_poles
    # labels through the guard only; must equal the stored design labels
    guard = GuardedTestLabels([r["pole"] for r in rows], [r["cls"] for r in rows], design_poles)
    y_view = guard.labels(rr)
    integ["guard_labels_equal_stored"] = bool(np.array_equal(y_view, E["y"]))
    # recomputed head output equals the stored Stage 4 scores (stored as float32, so compared exactly in float32)
    sc = score(head, E["emb"])
    integ["head_top3_reproduces"] = bool(np.array_equal(sc["top3"], S4["top3"]))
    integ["head_margin_reproduces"] = bool(np.array_equal(sc["margin"], S4["margin"]))
    V = SC.view_inputs(head, E["emb"])
    integ["top1_equals_head_top1"] = bool(np.array_equal(V["top"], S4["top3"][:, 0].astype(int)))
    integ["margin_float64_vs_stored_max_abs"] = float(np.abs(V["m"] - S4["margin"].astype(float)).max())

    units, urows = SC.unit_rows(E["pole"], y_view, E["view"])
    y_unit = np.array([c for _, c in units])
    nv = np.array([len(r) for r in urows])
    correct_view = int((V["top"] == y_view).sum())
    correct_last = int(sum(V["top"][r[-1]] == y_unit[i] for i, r in enumerate(urows)))
    facts = {"design_poles": len(design_poles), "design_units": len(units), "design_views": int(len(rr)),
             "views_correct": correct_view, "units_correct_last_view": correct_last}
    pool_units = sorted({(r["pole"], r["cls"]) for r in rows if r["part"] == "pool"})
    facts.update({"pool_poles": len({p for p, _ in pool_units}), "pool_units": len(pool_units)})
    integ["stage4_facts_reproduce"] = all(facts[k] == EXPECTED[k] for k in EXPECTED)
    if not all(v for k, v in integ.items() if not k.endswith("max_abs")):
        stop(f"integrity check failed: {integ} {facts}")

    nd = len(units)
    A, B, A_poles, B_poles = SC.halves(units)
    models = {s: SC.fit_s2(V, urows, y_unit, A, B, SC.S2_COMBO[s]) for s in ("S2-sum", "S2-min")}
    seqs = SC.sequences(V, urows, models)

    rs = SC.resplits(pool_units)
    n_by_seed = {s: rs[s]["n_cert_units"] for s in SC.SEEDS}
    n_min = min(n_by_seed.values())

    out = {}
    for s in SC.SCORES:
        cmax = SC.max_conf(seqs[s])
        grid = SC.make_grid(cmax)
        st = SC.design_stats(SC.track_table(seqs[s], y_unit, grid))
        out[s] = {"grid": grid, "stats": st}
    r3 = {s: {a: SC.rule3(out[s]["stats"], n_min, a) for a in SC.ALPHAS} for s in SC.SCORES}
    fd1 = any(r3[s][a]["predicted"] for s in SC.SCORES for a in SC.FD1_ALPHAS)
    alphas = SC.ALPHAS
    if not fd1:
        alphas = SC.ALPHAS + (SC.ALPHA_EXTRA,)
        for s in SC.SCORES:
            r3[s][SC.ALPHA_EXTRA] = SC.rule3(out[s]["stats"], n_min, SC.ALPHA_EXTRA)
    prim = SC.primary_alpha(r3, alphas)
    for s in SC.SCORES:
        out[s]["starts"] = SC.starts(out[s]["stats"]["f_d"], n_by_seed, nd, alphas)
        out[s]["reference"] = SC.reference(out[s]["stats"], alphas)

    # (j) correlation on design poles
    wrong_view = (V["top"] != y_view).astype(int)
    e_unit = np.array([wrong_view[r].sum() for r in urows])
    unit_pole = np.array([p for p, _ in units])
    within_unit = SC.rho_ci(nv, e_unit, unit_pole, SC.BOOT_SALT["unit"])
    last_wrong = np.array([int(V["top"][r[-1]] != y_unit[i]) for i, r in enumerate(urows)])
    up = np.unique(unit_pole)
    n_pole = np.array([(unit_pole == p).sum() for p in up])
    e_pole = np.array([last_wrong[unit_pole == p].sum() for p in up])
    within_pole = SC.rho_ci(n_pole, e_pole, up, SC.BOOT_SALT["pole"])
    m_mean = float(nv.mean())
    corr = {"within_unit": within_unit, "within_pole": within_pole, "m_mean_views_per_unit": m_mean,
            "poles_with_more_than_one_unit": int((n_pole > 1).sum())}
    for m_name, m in (("m_mean", m_mean), ("m_3", 3.0)):
        corr[f"design_effect_{m_name}"] = {"value": SC.design_effect(within_unit["rho"], m),
                                           "ci95": [SC.design_effect(x, m) for x in within_unit["ci95"]]}

    # (k) baselines: B1 view grid, B2 design draw, B5 F_design, B6 theta
    b1_seqs = [(np.array([V["m"][i]]), np.array([V["top"][i]])) for i in range(len(rr))]
    b1_grid = SC.make_grid(SC.max_conf(b1_seqs))
    b1_stats = SC.design_stats(SC.track_table(b1_seqs, y_view, b1_grid))
    k_b2 = SC.b2_draw(nv, np.random.default_rng([*SC.B2_SALT, SC.B2_DESIGN_KEY]))
    b2_seqs = [SC.seq_b2(V, r, k) for r, k in zip(urows, k_b2)]
    b2_grid = SC.make_grid(SC.max_conf(b2_seqs))
    b2_stats = SC.design_stats(SC.track_table(b2_seqs, y_unit, b2_grid))
    b2_starts = SC.starts(b2_stats["f_d"], n_by_seed, nd, alphas)
    f_design_b5 = np.sort(np.concatenate([c for c, _ in seqs["S1"]]))
    b6 = {s: SC.b6_theta([seqs[s][i] for i in B], y_unit[B], alphas) for s in ("S2-sum", "S2-min")}

    return {"integrity": integ, "facts": facts, "units": units, "n_views": nv, "y_unit": y_unit,
            "halves": {"A_poles": A_poles, "B_poles": B_poles, "A_units": len(A), "B_units": len(B),
                       "A_views": int(nv[A].sum()), "B_views": int(nv[B].sum())},
            "models": models, "scores": out, "resplits": rs, "n_by_seed": n_by_seed, "n_min": n_min,
            "rule3": r3, "fd1": fd1, "alphas": alphas, "primary_alpha": prim,
            "fallback2": prim is None, "correlation": corr,
            "b1": {"grid": b1_grid, "stats": b1_stats, "n_design_views": int(len(rr))},
            "b2": {"draw": k_b2, "grid": b2_grid, "stats": b2_stats, "starts": b2_starts},
            "b5": {"f_design": f_design_b5}, "b6": b6}


# ------------------------------------------------------------------ printout
def pct(x):
    return "   n/a" if x is None or not np.isfinite(x) else f"{100 * x:6.2f}"


def lam(x):
    return "-inf" if not np.isfinite(x) else f"{x:.6g}"


def report(R):
    print("\n--- (m) integrity checks ---")
    for k, v in R["integrity"].items():
        print(f"  {k}: {v if isinstance(v, bool) else f'{v:.3g}'}")
    print(f"  facts: {R['facts']}")
    h = R["halves"]
    print(f"\n--- (e) S2 halves (seed {SC.HALF_SEED}) ---")
    print(f"  A (ICP): {len(h['A_poles'])} poles, {h['A_units']} units, {h['A_views']} views | "
          f"B (ECDF, a, b): {len(h['B_poles'])} poles, {h['B_units']} units, {h['B_views']} views")
    for s, m in R["models"].items():
        tag = "primary" if s == SC.PRIMARY_S2 else "secondary"
        sizes = ", ".join(f"L{L} {len(v)}" for L, v in m["ecdf"].items())
        print(f"  {s} ({tag}): a = {m['a']:+.1f}, b = {m['b']:+.1f}, AURC {m['aurc']:.5f}, ICP scores {len(m['cal'])}, "
              f"ECDF sizes {sizes}")
    print("\n--- (g) Cert/Test re-splits (pool structure only) ---")
    for s, r in R["resplits"].items():
        print(f"  seed {s}: Cert {len(r['cert_poles'])} poles / {r['n_cert_units']} units | "
              f"Test {len(r['test_poles'])} poles / {r['n_test_units']} units")
    print(f"  n_min = {R['n_min']}")
    for s in SC.SCORES:
        S = R["scores"][s]
        g, st = S["grid"], S["stats"]
        print(f"\n--- {s}: {len(g)} distinct grid points (of {SC.N_POINTS}) ---")
        print(f"  {'j':>3} {'lambda':>12} {'f_d %':>7} {'e_d %':>7} {'wrong':>5} {'med view':>8}")
        for j in range(len(g)):
            print(f"  {j:>3} {lam(g[j]):>12} {pct(st['f_d'][j])} {pct(st['e_d'][j])} {int(st['wrong'][j]):>5} "
                  f"{st['median_pos'][j]:8.1f}")
        print("  starts per seed (grid index; -1 = no start possible): alpha: binom / hb / hb_a1 / hb_bs")
        for seed in SC.SEEDS:
            cells = "  ".join(f"{100 * a:g}%: " + "/".join(str(S['starts'][seed][a][r]) for r in SC.RULES)
                              for a in R["alphas"])
            print(f"   seed {seed} (n = {R['n_by_seed'][seed]}): {cells}")
        print("  reference (loosest j with e_d <= alpha): "
              + ", ".join(f"{100 * a:g}%: {S['reference'][a]}" for a in R["alphas"]))
    print(f"\n--- (i) Rule 3 (n_min = {R['n_min']}, f_d >= 70%, UCB90 x n_c <= k_max) ---")
    print(f"  {'score':>7} {'alpha':>6} {'pred':>5} {'j':>3} {'f_d %':>7} {'wrong/com':>9} {'UCB90 %':>8} "
          f"{'n_c':>4} {'UCB90xn_c':>9} {'k_max':>5}")
    for s in SC.SCORES:
        for a in R["alphas"]:
            r = R["rule3"][s][a]
            d = r["deciding"]
            if d is None:
                print(f"  {s:>7} {100 * a:5.1f}% {'no':>5}   (no grid point with f_d >= 70%)")
                continue
            km = "none" if d["k_max"] is None else str(d["k_max"])
            print(f"  {s:>7} {100 * a:5.1f}% {'YES' if r['predicted'] else 'no':>5} {d['j']:>3} {pct(d['f_d'])} "
                  f"{d['wrong']:>4}/{d['committed']:<4} {100 * d['ucb90']:8.3f} {d['n_c']:>4} "
                  f"{d['ucb90_x_n_c']:9.3f} {km:>5}")
    pa = R["primary_alpha"]
    print(f"\n  F-D1 (some alpha in 1-5% predicted for some score): {'PASS' if R['fd1'] else 'FAIL'}")
    if not R["fd1"]:
        print(f"  10% added to the alpha grid (Part A (i)); Rule 3 rerun with 10%.")
    print(f"  primary alpha: {'none (Fallback 2)' if pa is None else f'{100 * pa:g}%'}")
    c = R["correlation"]
    print("\n--- (j) within-sign error correlation (design poles; head top-1 per view, no threshold) ---")
    for k in ("within_unit", "within_pole"):
        r = c[k]
        print(f"  {k}: rho {r['rho']:.4f} [{r['ci95'][0]:.4f}, {r['ci95'][1]:.4f}] | clusters {r['clusters']}, "
              f"items {r['items']}, errors {r['errors']}, undefined bootstrap draws {r['draws_undefined']}")
    for k in ("design_effect_m_mean", "design_effect_m_3"):
        print(f"  {k}: {c[k]['value']:.4f} [{c[k]['ci95'][0]:.4f}, {c[k]['ci95'][1]:.4f}]")
    print(f"  mean views per unit {c['m_mean_views_per_unit']:.4f}; poles with more than one unit "
          f"{c['poles_with_more_than_one_unit']}")
    print("\n--- (k) baseline design values ---")
    b1 = R["b1"]
    print(f"  B1 view grid: {len(b1['grid'])} points over {b1['n_design_views']} design views; f_d "
          f"{pct(b1['stats']['f_d'][0])} .. {pct(b1['stats']['f_d'][-1])}%; e_d at lambda_0 {pct(b1['stats']['e_d'][0])}%")
    b2 = R["b2"]
    print(f"  B2 design draw: {len(b2['grid'])} grid points; f_d {pct(b2['stats']['f_d'][0])} .. "
          f"{pct(b2['stats']['f_d'][-1])}%; e_d at lambda_0 {pct(b2['stats']['e_d'][0])}%")
    print(f"  B5 F_design: {len(R['b5']['f_design'])} S1 margins (all design views)")
    for s, th in R["b6"].items():
        print(f"  B6 {s} theta per alpha (half B): "
              + ", ".join(f"{100 * a:g}%: {'none' if t is None else f'{t:.4f}'}" for a, t in th.items()))


# ------------------------------------------------------------------ main
def git(*args):
    return subprocess.run(["git", *args], capture_output=True, text=True).stdout.strip()


def to_json(R, d, blobs, code, t0):
    sc = {}
    for s, S in R["scores"].items():
        st = S["stats"]
        sc[s] = {"grid": S["grid"], "f_d": st["f_d"], "e_d": st["e_d"], "committed": st["committed"],
                 "wrong": st["wrong"], "median_view": st["median_pos"],
                 "starts": {str(seed): {str(a): v for a, v in S["starts"][seed].items()} for seed in SC.SEEDS},
                 "reference": {str(a): v for a, v in S["reference"].items()}}
    return jsonable({
        "plan": "poc3-implementation-plan-r9 Stage 5, design phase; LOG_r9_stage5_entry.md Part A",
        "run_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_head": git("rev-parse", "HEAD"), "worktree_clean": git("status", "--porcelain") == "",
        "versions": {"python": platform.python_version(), "numpy": np.__version__},
        "frozen_blobs": blobs, "code_sha256": code, "inputs_sha256": d["inputs_sha256"],
        "inputs_blob": d["inputs_blob"], "input_check_method": d["checks"], "files_opened": list(OPENED),
        "integrity": R["integrity"], "facts": R["facts"], "delta": 0.1,
        "grid_note": "grid[0] = -inf (stored as null): commit at the first view",
        "units": [list(u) for u in R["units"]], "n_views": R["n_views"],
        "halves": {"seed": SC.HALF_SEED, **R["halves"]},
        "s2_models": {s: {"a": m["a"], "b": m["b"], "aurc": m["aurc"], "combo": m["combo"],
                          "max_window": m["max_window"], "icp_cal": m["cal"],
                          "icp_cal_sha256": hashlib.sha256(np.ascontiguousarray(m["cal"], np.float64).tobytes()).hexdigest(),
                          "ecdf": {str(L): v for L, v in m["ecdf"].items()}} for s, m in R["models"].items()},
        "primary_s2": SC.PRIMARY_S2, "scores": sc,
        "resplits": {str(s): r for s, r in R["resplits"].items()}, "n_min": R["n_min"],
        "alphas": list(R["alphas"]), "rules": list(SC.RULES),
        "rule3": {s: {str(a): r for a, r in v.items()} for s, v in R["rule3"].items()},
        "fd1_pass": R["fd1"], "primary_alpha": R["primary_alpha"], "fallback2": R["fallback2"],
        "correlation": R["correlation"],
        "b1": {"grid": R["b1"]["grid"], "f_d": R["b1"]["stats"]["f_d"], "e_d": R["b1"]["stats"]["e_d"],
               "committed": R["b1"]["stats"]["committed"], "wrong": R["b1"]["stats"]["wrong"],
               "n_design_views": R["b1"]["n_design_views"]},
        "b2": {"design_draw": R["b2"]["draw"], "grid": R["b2"]["grid"], "f_d": R["b2"]["stats"]["f_d"],
               "e_d": R["b2"]["stats"]["e_d"],
               "starts": {str(seed): {str(a): v for a, v in R["b2"]["starts"][seed].items()} for seed in SC.SEEDS}},
        "b5": {"f_design": R["b5"]["f_design"]},
        "b6_theta": {s: {str(a): t for a, t in th.items()} for s, th in R["b6"].items()},
        "settings": {"rq1_draws": SC.RQ1_DRAWS, "rq1_salt": list(SC.RQ1_SALT), "b2_salt": list(SC.B2_SALT),
                     "b2_design_key": SC.B2_DESIGN_KEY, "b5c_salt": SC.B5C_SALT, "b5c_tmax": SC.B5C_TMAX,
                     "boot_draws": SC.BOOT_DRAWS, "boot_salt": {k: list(v) for k, v in SC.BOOT_SALT.items()},
                     "test_fraction": list(SC.TEST_FRACTION), "seeds": list(SC.SEEDS)},
        "seconds": round(time.time() - t0, 1)})


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage4", default=str(STAGE4))
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args(argv)
    t0 = time.time()
    blobs = FZ.check_frozen()                                   # Part A (b): first, before any data
    assert not f3_closed(), "F3 is closed; the design phase is over (plan R9 section 2)"
    out = Path(a.out)
    target = out / "stage5_design.json"
    if target.exists():
        stop(f"{target} exists; refusing to overwrite a recorded run")
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / "stage5_printout.txt", "w", encoding="utf-8")
    old, sys.stdout = sys.stdout, Tee(sys.stdout, log)
    try:
        OPENED.clear()
        print(f"=== R9 Stage 5 design phase (BelgiumTSC design poles only) | HEAD {git('rev-parse', 'HEAD')} ===")
        print(f"freeze check PASS: {len(blobs)} files by git blob ID")
        d = load_stage4(a.stage4)
        print(f"input checks PASS: {d['checks']}")
        R = run_design(d)
        assert not any(f in p for p in OPENED for f in FORBIDDEN), "a pool or model file was opened"
        print(f"files opened: {OPENED}")
        report(R)
        code = {f: sha256_file(f) for f in CODE}
        target.write_text(json.dumps(to_json(R, d, blobs, code, t0), indent=1) + "\n")
        print(f"\nOutput: {target}  SHA-256 {sha256_file(target)}  git blob {FZ.blob_id(target)}")
        print("Design poles only. No pool label, prediction, Cert or Test number was read or computed.")
    finally:
        sys.stdout = old
        log.close()
    return R


if __name__ == "__main__":
    main()
