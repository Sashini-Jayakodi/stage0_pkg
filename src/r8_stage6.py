"""POC-2 (R8) Stage 6: B5 / B5c (Ringel et al. 2024), B6 (Boursinos and Koutsoukos 2022) on the Stage 4 splits, and
the comparison table of plan section 9. LOG_r8_stage6_entry.md Part A (a)-(i), Amendment 1, design-set values.

Usage:  py -3.12 -m src.r8_stage6 --run runs/20261003_stage1 --config configs/poc.yaml
Output: runs/20261003_stage1/poc2/stage6/stage6_summary.json
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import yaml

from src import r8_frozen as FZ
from src import r8_stage1 as R1
from src import r8_stage2 as S2
from src import r8_stage3_resample as RS
from src import r8_stage4 as S4
from src import r8_stage6_design as B5D
from src import r8_stage6_methods as M
from src.head import mahal_sq
from src.r8_frozen import load_frozen
from src.util import sha256_file

FROZEN_ARGS = {}                       # tests point load_frozen at a fake root; production uses the F1 hashes
STAGE4_SUMMARY_SHA256 = "e66e45f210ebae5e6d21a4d8247584e1affc2d9691a760427814c821b164d2cc"
STAGE4_CODE_SHA256 = "fae67390a488b44c57f1386c9b256cb0fbbcf2444282f5724c2580aa3e6445c2"
B5_DESIGN_SHA256 = "39d4e9b6f614f3062ef3e7af98a66ef2b4549a07e80106b827083a9b4c3e1924"
STAGE3A_SHA256 = "b6631a1f617ce619616ee58e76f85bd88fe5338366d3489a1313d85b0b8c420d"
STAGE3B_SHA256 = "7591a438b569e7231f3a40d8da6420a81b6f65791e707df4c0101338a0bf7367"
DCAL_SALT = 20261116
TMAX = 30
B6_VARIANTS = [(K, c) for K in (5, 9) for c in ("sum", "min")]
TABLE_ALPHAS = (0.01, 0.02)
check_papers = B5D.check_papers        # tests may replace it (the PDFs are git-ignored)


# ------------------------------------------------------------------ inputs
def load_all(run, config, cfg):
    run = Path(run)
    s3, s4, s6 = run / "poc2" / "stage3", run / "poc2" / "stage4", run / "poc2" / "stage6"
    for p, h, what in ((s4 / "stage4_summary.json", STAGE4_SUMMARY_SHA256, "Stage 4 Part B"),
                       (s6 / "b5_design.json", B5_DESIGN_SHA256, "Stage 6 Part A"),
                       (s3 / "stage3a_summary.json", STAGE3A_SHA256, "Stage 3 Part B"),
                       (s3 / "stage3b_summary.json", STAGE3B_SHA256, "Stage 3 Part B")):
        assert sha256_file(p) == h, f"{p.name} differs from the hash in {what}"
    s4sum = json.loads((s4 / "stage4_summary.json").read_text())
    code = sha256_file(Path(S4.__file__))
    assert code == s4sum["code_sha256"]["r8_stage4.py"] == STAGE4_CODE_SHA256, "src/r8_stage4.py differs from the code that wrote Stage 4"
    assert sha256_file(s4 / "baselines_design.json") == S4.BASELINES_SHA256, "baselines_design.json differs from the hash in Stage 4 Part A"
    bdes = json.loads((s4 / "baselines_design.json").read_text())
    d = RS.load_pool(run, cfg)
    _, checks = RS.pool_tables(d, cfg)
    bdes["n_all_tracks"] = len(np.load(run / "splits" / "seed0.npz")["track_names"])
    roi = S4.roi_sizes(run, yaml.safe_load(Path(config).read_text()), d)
    T, y, _, c2 = S4.method_tables(d, cfg, bdes, roi)
    checks.update(c2)
    return {"d": d, "T": T, "y": y, "roi": roi, "checks": checks, "s4": s4sum,
            "F": np.asarray(json.loads((s6 / "b5_design.json").read_text())["F_margins"], float),
            "s3a": json.loads((s3 / "stage3a_summary.json").read_text()),
            "s3b": json.loads((s3 / "stage3b_summary.json").read_text())}


def track_data(d, roi, F):
    """Per pool track (track-code order): Ringel pi/pred/sizes and the B6 inputs (nonconformity, passing rows)."""
    rows = R1.track_rows(d["track"], d["frame"])
    D2 = mahal_sq(d["head"], d["emb"])
    lp = np.log(d["head"]["priors"])
    pis, preds, sizes, passed_rows = [], [], [], []
    for r in rows:
        S, _ = R1.running_scores(-0.5 * D2[r], d["passed"][r], lp, "V1")
        Mg, top = R1.margins(S)
        pis.append(B5D.score(Mg, F))
        preds.append(top)
        sizes.append(roi[r])
        passed_rows.append(r[d["passed"][r]])
    return {"rows": rows, "pis": pis, "preds": preds, "sizes": sizes, "passed_rows": passed_rows,
            "yfull": np.array([p[-1] for p in preds]), "nc": M.nonconformity(D2)}


# ------------------------------------------------------------------ per re-split
def run_seed(seed, A, D, alphas, delta, n_test):
    y = A["y"]
    test = S4.make_split(y, seed, n_test)
    cert_idx, test_idx = np.flatnonzero(~test), np.flatnonzero(test)
    sub = lambda idx: ([D["pis"][i] for i in idx], [D["preds"][i] for i in idx], y[idx], D["yfull"][idx])
    out = {"seed": seed, "b5": {}, "b5c": {}, "b6": {}, "ece": {}}
    tsz = [D["sizes"][i] for i in test_idx]
    perm = cert_idx[np.random.default_rng([DCAL_SALT, seed]).permutation(len(cert_idx))]
    d1, d2 = np.sort(perm[: len(perm) // 2]), np.sort(perm[len(perm) // 2:])
    for al in alphas:
        lam = M.marginal_calibrate(*sub(cert_idx), al, delta)
        out["b5"][al] = {"lambda": lam, **M.ringel_test(*sub(test_idx), np.full(TMAX, lam), tsz)}
        lam_vec, cand = M.conditional_calibrate(D["pis"], D["preds"], y, D["yfull"], d1, d2, al, delta, TMAX)
        out["b5c"][al] = {"lambda": lam_vec, "candidate": cand, **M.ringel_test(*sub(test_idx), lam_vec, tsz)}

    parts = M.b6_parts(cert_idx, seed)
    used = np.concatenate(list(parts.values()))
    assert not np.isin(used, test_idx).any() and len(np.unique(used)) == len(cert_idx), "B6 would read a Test track"
    for K, combo in B6_VARIANTS:
        name = f"ecdf_{combo}_K{K}"
        model = M.b6_fit(D["nc"], D["passed_rows"], y, parts, K, combo, alphas)
        seqs = {i: M.b6_sequence(model, D["nc"], D["passed_rows"][i]) for i in test_idx}
        res = {}
        for al in alphas:
            lab_end = [M.b6_decide(seqs[i], model["theta"][al]) for i in test_idx]
            com = np.array([l >= 0 for l, _ in lab_end])
            wrong = np.array([l >= 0 and l != y[i] for (l, _), i in zip(lab_end, test_idx)])
            pos = [int(np.flatnonzero(D["rows"][i] == e)[0]) + 1 for (l, e), i in zip(lab_end, test_idx) if l >= 0]
            szs = [A["roi"][e] for l, e in lab_end if l >= 0]
            res[al] = {"theta": model["theta"][al], "coverage": float(com.mean()), "committed": int(com.sum()),
                       "error_committed": float(wrong.sum() / com.sum()) if com.any() else None,
                       "median_commit_frame": float(np.median(pos)) if pos else None,
                       "median_size": float(np.median(szs)) if szs else None}
        out["b6"][name] = {"a": model["a"], "b": model["b"], "aurc": model["aurc"], "alphas": res}
        # ECE on the Test frames with full windows ([BK] eq. 24). The ICP calibration part does not depend on K or the
        # combination, so single-frame ICP and the arithmetic-mean merge are computed once per K.
        wins = [(i, e, w) for i in test_idx for e, w in M.track_windows(D["passed_rows"][i], K)]
        yy = np.array([y[i] for i, _, _ in wins])
        e = out["ece"].setdefault(K, {"frames": len(yy)})
        if combo == "sum":
            e["single_icp"] = M.ece(np.stack([M.icp_pvalues(D["nc"][en], model["cal"]) for _, en, _ in wins]), yy)
            e["arith_mean_x2"] = M.ece(np.stack([np.minimum(1.0, 2.0 * M.icp_pvalues(D["nc"][w], model["cal"]).mean(0))
                                                 for _, _, w in wins]), yy)
        e[f"ecdf_{combo}"] = M.ece(np.stack([M.ecdf(model["ecdf"], M.combine(M.icp_pvalues(D["nc"][w], model["cal"]), combo))
                                             for _, _, w in wins]), yy)
    return out, test


def reproduce_stage4(A, seeds_rows, n_test):
    ok = True
    for row in seeds_rows:
        test = S4.make_split(A["y"], int(row["seed"]), n_test)
        for al_key, by_m in row["alphas"].items():
            for m in S4.METHODS:
                r4 = by_m[m]
                if r4["certified"]:
                    com = A["T"][m]["committed"][test, int(r4["grid_position"])]
                    ok &= float(com.sum() / test.sum()) == float(r4["coverage"])
    return bool(ok)


# ------------------------------------------------------------------ comparison table (Part A (h))
def med_range(vals, p=1, pct=True):
    v = [x for x in vals if x is not None]
    if not v:
        return "n/a"
    f = (lambda x: f"{100 * x:.{p}f}%") if pct else (lambda x: f"{x:.{p}f}")
    return f"{f(float(np.median(v)))} ({f(min(v))} to {f(max(v))})"


def table(A, S, al, b6_names):
    s4 = A["s4"]["per_seed"]
    key = str(al)
    cols = ["Ours", "B1", "B2", "B3", "B4", "B5", "B5c"] + [f"B6 {n}" for n in b6_names]
    s4m = {"Ours": "commit", "B2": "B2", "B3": "B3", "B4": "B4"}
    rows = {}
    promise = {"Ours": "certified real error", "B1": "certified per-frame error (frames assumed independent)",
               "B2": "certified real error", "B3": "certified real error", "B4": "certified real error",
               "B5": "accuracy gap, marginal", "B5c": "accuracy gap, conditional on halt time"}
    rows["1 Type of promise"] = {c: promise.get(c, "none") for c in cols}
    r3b = A["s3b"]["results"].get(key)
    v = {}
    for c in cols:
        if c in ("Ours", "B1"):
            m = "track" if c == "Ours" else "B1"
            s = f"3b {100 * r3b[m]['violation_rate']:.1f}% [{100 * r3b[m]['cp95'][0]:.1f}, {100 * r3b[m]['cp95'][1]:.1f}]"
            if al == 0.01:
                r = next(x for x in A["s3a"]["rows"] if x["rho_sim"] == 0.45)[m]
                s += f"; 3a rho .45 {100 * r['violation_rate']:.1f}%"
            v[c] = s + f" vs delta 10%"
        elif c in ("B2", "B3", "B4"):
            v[c] = "valid by construction, not measured"
        elif c in ("B5", "B5c"):
            v[c] = "valid for own target (i.i.d. tracks), not measured"
        else:
            v[c] = "no promise"
    rows["2 Valid on linked frames"] = v
    cov, err, fr, px = {}, {}, {}, {}
    for c in cols:
        if c == "B1":
            for dct in (cov, err, fr, px):
                dct[c] = "n/a (frame-level; Stage 3 only)"
            continue
        if c in s4m:
            rs = [r["alphas"][key][s4m[c]] for r in s4]
            ncert = sum(r["certified"] for r in rs)
        elif c in ("B5", "B5c"):
            rs = [x[c.lower()][al] for x in S]
            ncert = None
        else:
            rs = [x["b6"][c[3:]]["alphas"][al] for x in S]
            ncert = sum(r["theta"] is not None for r in rs)
        tag = "" if ncert is None else f" [{ncert}/5 {'certified' if c in s4m else 'with a threshold'}]"
        cov[c] = med_range([r["coverage"] for r in rs]) + tag
        err[c] = med_range([r["error_committed"] for r in rs], 2)
        fr[c] = med_range([r["median_commit_frame"] for r in rs], 1, False)
        px[c] = med_range([r["median_size"] for r in rs], 1, False)
    rows["3 Sign coverage (Test)"] = cov
    rows["4 Error among committed signs (Test)"] = err
    sm = {}
    for c in cols:
        if c in s4m:
            sm[c] = med_range([r["smallest_certified"][s4m[c]] for r in s4], 1)
        elif c == "B1":
            ok = [float(a) for a, r in A["s3b"]["results"].items() if r["B1"]["certified_share"] >= 0.5]
            sm[c] = f"{100 * min(ok):.1f}% (3b, >= half of draws)" if ok else "none"
        else:
            sm[c] = "n/a (no certified real error)"
    rows["5 Smallest certifiable alpha, n = 784"] = sm
    rows["6 Median commit frame (Test)"] = fr
    rows["7 Median sign size at commit, px (Test)"] = px
    rows["8 Can escalate"] = {c: ("no" if c in ("B5", "B5c") else "yes") for c in cols}
    return cols, rows


def print_table(title, cols, rows):
    print(f"\n{title}")
    for r, vals in rows.items():
        print(f"  {r}")
        for c in cols:
            print(f"      {c:16} {vals[c]}")


# ------------------------------------------------------------------ main
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--config", default="configs/poc.yaml")
    a = ap.parse_args(argv)
    cfg = load_frozen(**FROZEN_ARGS)
    check_papers()
    t0 = time.time()
    alphas = [float(x) for x in cfg["certificate_A"]["alpha_grid"]]
    delta, n_test = float(cfg["fixed_in_every_phase"]["delta"]), int(cfg["splits"]["test_tracks"])
    req = 4
    A = load_all(a.run, a.config, cfg)
    A["checks"]["stage4_coverages_reproduce"] = reproduce_stage4(A, A["s4"]["per_seed"], n_test)
    assert A["checks"]["stage4_coverages_reproduce"], "the regenerated splits do not reproduce Stage 4's Test coverages; stop"
    D = track_data(A["d"], A["roi"], A["F"])
    S = []
    for row in A["s4"]["per_seed"]:
        out, _ = run_seed(int(row["seed"]), A, D, alphas, delta, n_test)
        S.append(out)
        print(f"  seed {row['seed']} done ({time.time() - t0:.0f} s)", flush=True)

    prim = float(cfg["certificate_A"]["primary_alpha"])
    chk = {}
    chk["B5_own_target"] = [x["b5"][prim]["gap_marginal"] <= prim + 1e-12 for x in S]
    chk["B5c_own_target"] = [max(x["b5c"][prim]["gap_20pct_earliest"], x["b5c"][prim]["gap_50pct_earliest"],
                                 x["b5c"][prim]["gap_marginal"]) <= prim + 1e-12 for x in S]
    chk["B6_calibration_direction_K5"] = [x["ece"][5]["ecdf_sum"] < x["ece"][5]["single_icp"]
                                          and x["ece"][5]["ecdf_min"] < x["ece"][5]["single_icp"] for x in S]
    stage6_checks = {k: {"per_seed": v, "count": int(sum(v)), "pass": bool(sum(v) >= req)} for k, v in chk.items()}

    # ---- print
    f = lambda x, p=2: "  n/a " if x is None else f"{100 * x:6.{p}f}%"
    g = lambda x: " n/a" if x is None else f"{x:5.1f}"
    print(f"\n=== R8 Stage 6: B5 / B5c (Ringel et al. 2024) and B6 (Boursinos and Koutsoukos 2022) on the Stage 4 splits ===")
    for al in TABLE_ALPHAS:
        print(f"\n--- alpha {100 * al:.0f}%  (B5/B5c: accuracy-gap tolerance; B6: target risk r)")
        print(f"{'seed':>4} {'method':16} {'cov':>8} {'err':>8} {'fr':>5} {'px':>6} {'gap':>8} {'gap20':>8} {'gap50':>8}  param")
        for x in S:
            for nm in ("b5", "b5c"):
                r = x[nm][al]
                prm = f"lambda {r['lambda']}" if nm == "b5" else f"lambda finite at {int(np.isfinite(r['lambda']).sum())}/30 frames"
                print(f"{x['seed']:>4} {nm.upper():16} {f(r['coverage'])} {f(r['error_committed'])} {g(r['median_commit_frame'])} "
                      f"{g(r['median_size'])} {f(r['gap_marginal'])} {f(r['gap_20pct_earliest'])} {f(r['gap_50pct_earliest'])}  {prm}")
            for nm, v in x["b6"].items():
                r = v["alphas"][al]
                prm = "no theta" if r["theta"] is None else f"theta {r['theta']:.4g}, (a,b)=({v['a']:+.1f},{v['b']:+.1f})"
                print(f"{x['seed']:>4} {('B6 ' + nm):16} {f(r['coverage'])} {f(r['error_committed'])} {g(r['median_commit_frame'])} "
                      f"{g(r['median_size'])} {'':>8} {'':>8} {'':>8}  {prm}")
    print("\n--- B6 calibration (ECE, [BK] eq. 24, Test frames with full windows)")
    print(f"{'seed':>4} {'K':>2} {'single ICP':>11} {'arith x2':>9} {'ECDF sum':>9} {'ECDF min':>9} {'frames':>7}")
    for x in S:
        for K in (5, 9):
            e = x["ece"][K]
            print(f"{x['seed']:>4} {K:>2} {e['single_icp']:11.4f} {e['arith_mean_x2']:9.4f} {e['ecdf_sum']:9.4f} {e['ecdf_min']:9.4f} {e['frames']:>7}")
    print(f"\n--- Stage 6 checks (Part A (g), alpha {100 * prim:.0f}%, need >= {req} of 5)")
    for k, v in stage6_checks.items():
        print(f"  {k}: {['PASS' if z else 'fail' for z in v['per_seed']]} -> {v['count']}/5 -> {'PASS' if v['pass'] else 'FAIL'}")
    tables = {}
    for al in TABLE_ALPHAS:
        for names, lab in ((["ecdf_sum_K5", "ecdf_min_K5"], "main"), (["ecdf_sum_K9", "ecdf_min_K9"], "K9")):
            cols, rows = table(A, S, al, names)
            tables[f"{lab}_{al}"] = {"columns": cols, "rows": rows}
            print_table(f"=== Comparison table ({'B6 K = 5' if lab == 'main' else 'B6 K = 9 variants'}), alpha {100 * al:.0f}%"
                        f"{' (primary)' if al == prim else ' (declared extra)'}; median (min to max) over 5 re-splits", cols, rows)
    print("\nChecks:")
    for k, v in A["checks"].items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    all_pass = all(A["checks"].values())
    print(f"\nALL PASS (checks): {all_pass}")

    out = Path(a.run) / "poc2" / "stage6"
    summary = {"plan": "poc2-implementation-plan-r8 Stage 6",
               "frozen_r8_sha256": sha256_file(Path(FROZEN_ARGS.get("root", FZ.ROOT)) / FZ.FROZEN),
               "code_sha256": {"r8_stage6.py": sha256_file(Path(__file__)), "r8_stage6_methods.py": sha256_file(Path(M.__file__))},
               "inputs_sha256": {**A["d"]["inputs_sha256"], "stage4_summary.json": STAGE4_SUMMARY_SHA256,
                                 "b5_design.json": B5_DESIGN_SHA256, "stage3a_summary.json": STAGE3A_SHA256,
                                 "stage3b_summary.json": STAGE3B_SHA256, "papers": B5D.PAPERS},
               "per_seed": S, "stage6_checks": stage6_checks, "tables": tables, "checks": A["checks"], "all_pass": all_pass,
               "seconds": round(time.time() - t0, 1)}
    (out / "stage6_summary.json").write_text(json.dumps(S2.jsonable(summary), indent=1))
    print(f"Output: {out / 'stage6_summary.json'} ({summary['seconds']} s)")
    print("Stage 7 is not started.")
    return summary


if __name__ == "__main__":
    main()
