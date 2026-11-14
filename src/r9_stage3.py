"""POC-3 (R9) Stage 3 runner (exploratory, post-hoc): the layer on S1, S2-sum, S2-min, S3 and S4 over the five
POC-2 Stage 4 re-splits, exact binomial and Hoeffding-Bentkus (three starts), every alpha (LOG_r9_stage3_entry.md
Part A (c), (h), (i), (k) and Amendment 1).

Order: freeze check; hash checks of every file read; POC-2 pool loader and Stage 4 tables (with their reproduction
checks); regenerated splits must reproduce every certified Stage 4 Test coverage; only then are the four scores
computed on the pool and certified.

Usage:  py -3.12 -m src.r9_stage3 --run runs/20261003_stage1 --config configs/poc.yaml
Output: runs/20261003_stage1/poc3/stage3/stage3_summary.json
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import yaml

from src import r8_stage2 as S2
from src import r8_stage3_resample as RS
from src import r8_stage4 as S4
from src import r8_stage5 as S5
from src import r9_certify as C
from src import r9_frozen as FZ
from src import r9_stage3_scores as SC
from src.r8_frozen import load_frozen
from src.util import sha256_file

FROZEN_ARGS = {}                       # POC-2 frozen config; tests point it at a fake root
DESIGN_SHA256 = "798585428c4cc345a760e347ded0c37f16c46f65a2ecf7dfa50473bacfb8e62b"        # Part A design-set values
STAGE4_SUMMARY_SHA256 = "e66e45f210ebae5e6d21a4d8247584e1affc2d9691a760427814c821b164d2cc"  # Part A (c)
STAGE4_CODE_SHA256 = "fae67390a488b44c57f1386c9b256cb0fbbcf2444282f5724c2580aa3e6445c2"     # Part A (c)
TRAIN_META_SHA256 = "4c06c3ad07a3b3b89faabcd49b26e8f6dbfff6460f25bf5b66730140de6a132e"      # Part A (i)


# ------------------------------------------------------------------ design values (hash-checked)
def load_design(path):
    """Grids (null at index 0 = -inf), f_d, starts, reference and the S2 models, from stage3_design.json."""
    js = json.loads(Path(path).read_text())
    out = {}
    for s, v in js["scores"].items():
        g = v["grid"]
        assert g[0] is None and all(x is not None for x in g[1:]), f"{s}: only lambda_0 may be -inf"
        out[s] = {"grid": np.array([-np.inf] + [float(x) for x in g[1:]]), "f_d": np.array(v["f_d"], float),
                  "starts": {float(a): r for a, r in v["starts"].items()},
                  "reference": {float(a): int(j) for a, j in v["reference"].items()}}
    models = {s: {"cal": np.array(m["icp_cal"], float), "ecdf": np.array(m["ecdf"], float), "a": float(m["a"]),
                  "b": float(m["b"]), "K": int(m["K"]), "combo": m["combo"]} for s, m in js["s2_models"].items()}
    return js, out, models


# ------------------------------------------------------------------ per-score tables on the pool
def size_table(tab, rows, roi):
    """ROI size at the commit frame per (track, lambda); NaN where not committed."""
    size = np.full(tab["pos"].shape, np.nan)
    for i, r in enumerate(rows):
        c = tab["committed"][i]
        size[i, c] = roi[r[tab["pos"][i, c] - 1]]
    return size


def cert_counts(tab, cert):
    """k(j) and n_c(j) on the Cert tracks, from the tracks-by-lambdas table (hard rule 4)."""
    com = tab["committed"][cert]
    return (com & tab["wrong"][cert]).sum(0), com.sum(0)


def certify_cell(tab, cert, alpha, f_d, rule, n_design):
    k, n_c = cert_counts(tab, cert)
    return int(C.certify(k, n_c, int(cert.sum()), alpha, f_d, test=rule, n_design=n_design))


def test_metrics(tab, size, j, test, s_values):
    """Plan 5.4 metrics on the Test tracks at grid index j (j < 0: nothing certified, nothing commits)."""
    nt = int(test.sum())
    if j < 0:
        return {"j": None, "coverage": 0.0, "escalation": 1.0, "error": None, "committed": 0, "wrong": 0,
                "median_frame": None, "median_size": None, "d90": S5.NOT_REACHED}
    com, wr = tab["committed"][test, j], tab["wrong"][test, j]
    sz = size[test, j]
    nc = int(com.sum())
    return {"j": int(j), "coverage": nc / nt, "escalation": 1.0 - nc / nt, "error": (int(wr.sum()) / nc) if nc else None,
            "committed": nc, "wrong": int(wr.sum()),
            "median_frame": float(np.median(tab["pos"][test, j][com])) if nc else None,
            "median_size": float(np.median(sz[com])) if nc else None,
            "d90": S5.d90(S5.committed_by_size(com, sz, s_values), s_values)}


def smallest_alpha(cells, alphas):
    """Smallest alpha with a certified threshold; None if none."""
    for a in sorted(alphas):
        if cells[a] >= 0:
            return a
    return None


# ------------------------------------------------------------------ split reproduction (Part A (c))
def reproduce_stage4(s4, T, y, n_test):
    """Regenerate every split; every certified Stage 4 cell's Test coverage must come out exactly as recorded."""
    tests, ok, n_cells = {}, True, 0
    for row in s4["per_seed"]:
        seed = int(row["seed"])
        test = S4.make_split(y, seed, n_test)
        ok &= int(test.sum()) == int(row["n_test"])
        for by_m in row["alphas"].values():
            for m in S4.METHODS:
                r4 = by_m[m]
                if r4["certified"]:
                    j = int(r4["grid_position"])
                    ok &= float(T[m]["committed"][test, j].sum() / test.sum()) == float(r4["coverage"])
                    n_cells += 1
        tests[seed] = test
    return tests, bool(ok), n_cells


# ------------------------------------------------------------------ printing helpers
def f_pct(x):
    return "    -" if x is None else f"{100 * x:5.2f}"


def f_num(x, w=5):
    return f"{'-':>{w}}" if x is None else f"{x:{w}.1f}"


def med_range(vals, pct=True):
    v = [x for x in vals if x is not None and not isinstance(x, str)]
    if not v:
        return "-"
    k = 100 if pct else 1
    return f"{k * np.median(v):.2f} ({k * min(v):.2f}-{k * max(v):.2f})" if pct else \
        f"{np.median(v):.1f} ({min(v):.1f}-{max(v):.1f})"


# ------------------------------------------------------------------ main
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--config", default="configs/poc.yaml")
    a = ap.parse_args(argv)
    t0 = time.time()
    blobs = FZ.check_frozen()                                          # Part A (b): first
    cfg = load_frozen(**FROZEN_ARGS)                                   # POC-2 F1 config, read only (pool size, Stage 4 grid)
    run = Path(a.run)
    dpath = run / "poc3" / "stage3" / "stage3_design.json"
    assert sha256_file(dpath) == DESIGN_SHA256, "stage3_design.json differs from the hash in Part A"
    s4dir = run / "poc2" / "stage4"
    assert sha256_file(s4dir / "stage4_summary.json") == STAGE4_SUMMARY_SHA256, "stage4_summary.json differs from the hash in Stage 4 Part B"
    s4 = json.loads((s4dir / "stage4_summary.json").read_text())
    code_now = sha256_file(Path(S4.__file__))
    assert code_now == s4["code_sha256"]["r8_stage4.py"] == STAGE4_CODE_SHA256, "src/r8_stage4.py differs from the code that wrote Stage 4"
    assert sha256_file(s4dir / "baselines_design.json") == S4.BASELINES_SHA256, "baselines_design.json differs from the hash in Stage 4 Part A"
    poc_cfg = yaml.safe_load(Path(a.config).read_text())
    meta = Path(poc_cfg["data"]["cache_dir"]) / "train_meta.csv"
    assert sha256_file(meta) == TRAIN_META_SHA256, "train_meta.csv differs from the hash in Stage 4 Part B"
    # hard rule 3: a GTSRB stage reads nothing from the BelgiumTSC folders
    assert not any("btsc" in str(p).lower() for p in (run, meta)), "a BelgiumTSC path was given"
    js, des, models = load_design(dpath)
    n_cert, n_design, alphas, rules = int(js["n_cert"]), int(js["n_design"]), [float(x) for x in js["alphas"]], js["rules"]

    # POC-2 pool tables and their reproduction checks (unchanged POC-2 code)
    d = RS.load_pool(run, cfg)
    _, checks = RS.pool_tables(d, cfg)
    bdes = json.loads((s4dir / "baselines_design.json").read_text())
    bdes["n_all_tracks"] = len(np.load(run / "splits" / "seed0.npz")["track_names"])
    roi = S4.roi_sizes(run, poc_cfg, d)
    T, y, nopass, c2 = S4.method_tables(d, cfg, bdes, roi)
    checks.update(c2)
    assert all(checks.values()), f"POC-2 pool reproduction checks failed: {checks}"
    n_test = int(cfg["splits"]["test_tracks"])
    tests, ok, n_cells = reproduce_stage4(s4, T, y, n_test)
    checks["stage4_coverages_reproduce"] = ok
    assert ok, "the regenerated splits do not reproduce Stage 4's Test coverages; stop (Part A (c))"

    # the scores on the pool (design calibration only, from the hash-checked design file)
    F = SC.frame_inputs(d["head"], d["emb"], d["passed"], d["track"], d["frame"])
    codes = np.array([d["track"][r[0]] for r in F["rows"]])
    assert (codes == d["pool_codes"]).all(), "pool track order differs"
    seqs = SC.sequences(F, models)
    tabs, sizes, no_value = {}, {}, {}
    for s in SC.SCORES:
        st = SC.starts(des[s]["f_d"], n_design, n=n_cert, alphas=alphas, rules=rules)
        assert all(st[al][r] == des[s]["starts"][al][r] for al in alphas for r in rules), f"{s}: starts differ from design"
        tabs[s] = SC.track_table(seqs[s], y, des[s]["grid"])
        sizes[s] = size_table(tabs[s], F["rows"], roi)
        no_value[s] = int(sum(not np.isfinite(c).any() for c, _ in seqs[s]))
    checks["design_starts_recompute"] = True
    s_values = S5.s_range(float(np.max(roi)))

    res = {s: {"per_seed": {}} for s in SC.SCORES}
    for seed, test in tests.items():
        cert = ~test
        assert int(cert.sum()) == n_cert and int(test.sum()) == n_test, "split sizes differ from Part A"
        for s in SC.SCORES:
            cell = {"cert": {}, "test": {}, "reference": {}, "smallest_alpha": {}}
            for al in alphas:
                cell["cert"][al] = {r: certify_cell(tabs[s], cert, al, des[s]["f_d"], r, n_design) for r in rules}
                cell["test"][al] = {r: test_metrics(tabs[s], sizes[s], cell["cert"][al][r], test, s_values) for r in rules}
                cell["reference"][al] = test_metrics(tabs[s], sizes[s], des[s]["reference"][al], test, s_values)
            for r in rules:
                cell["smallest_alpha"][r] = smallest_alpha({al: cell["cert"][al][r] for al in alphas}, alphas)
            res[s]["per_seed"][seed] = cell
    seeds = sorted(tests)

    # ---- printout
    print("\n=== R9 Stage 3 (exploratory, post-hoc): the layer on GTSRB, 5 re-splits ===")
    print(f"Freeze check passed: {', '.join(f'{k} {v[:8]}' for k, v in blobs.items())}")
    print(f"Pool {len(F['rows'])} tracks | Cert {n_cert} | Test {n_test} | design n {n_design} | delta {C.DELTA}")
    print(f"Stage 4 splits regenerated; {n_cells} certified Stage 4 cells reproduce exactly. POC-2 checks: "
          + ", ".join(f"{k} {'PASS' if v else 'FAIL'}" for k, v in checks.items()))
    print(f"Pool tracks with no value: " + ", ".join(f"{s} {no_value[s]}" for s in SC.SCORES)
          + f" (tracks with no passing frame: {int(nopass.sum())})")
    print("alpha 0.5%: no exact-binomial start at n = 784 (start rule; Amendment 1). HB at 0.5% is not compared with it.")
    for s in SC.SCORES:
        tag = {"S2-sum": " (primary S2)", "S2-min": " (secondary S2)"}.get(s, "")
        print(f"\n--- {s}{tag}: {len(des[s]['grid'])} grid points ---")
        print("certified grid index per seed 0-4 (- = none):")
        print(f"  {'alpha':>6} " + " ".join(f"{r:>18}" for r in rules))
        for al in alphas:
            row = []
            for r in rules:
                js_ = [res[s]["per_seed"][sd]["cert"][al][r] for sd in seeds]
                row.append(" ".join("-" if j < 0 else str(j) for j in js_))
            print(f"  {100 * al:5.1f}% " + " ".join(f"{x:>18}" for x in row))
        for label, getter in (("exact binomial (certified)", lambda sd, al: res[s]["per_seed"][sd]["test"][al]["binom"]),
                              ("uncertified reference (loosest j with design error <= alpha)",
                               lambda sd, al: res[s]["per_seed"][sd]["reference"][al])):
            print(f"Test metrics, {label}:")
            print(f"  {'alpha':>6} {'seed':>4} {'j':>3} {'lambda':>11} {'cov %':>6} {'err %':>6} {'wrong':>5} "
                  f"{'esc %':>6} {'frame':>5} {'size':>5} {'D90':>11}")
            for al in alphas:
                for sd in seeds:
                    m = getter(sd, al)
                    lam = "-" if m["j"] is None else ("-inf" if not np.isfinite(des[s]["grid"][m["j"]]) else f"{des[s]['grid'][m['j']]:.6g}")
                    print(f"  {100 * al:5.1f}% {sd:>4} {('-' if m['j'] is None else m['j']):>3} {lam:>11} "
                          f"{f_pct(m['coverage']):>6} {f_pct(m['error']):>6} {m['wrong']:>5} {f_pct(m['escalation']):>6} "
                          f"{f_num(m['median_frame'])} {f_num(m['median_size'])} {str(m['d90']):>11}")
        print("Summary over seeds, median (min-max): certified share, Test coverage %, error %, commit frame, size")
        for al in alphas:
            for r in rules + ["reference"]:
                ms = [res[s]["per_seed"][sd]["reference"][al] if r == "reference" else res[s]["per_seed"][sd]["test"][al][r]
                      for sd in seeds]
                ncert = sum(m["j"] is not None for m in ms)
                cm = [m for m in ms if m["j"] is not None]
                print(f"  {100 * al:5.1f}% {r:>9} {ncert}/5  cov {med_range([m['coverage'] for m in cm]):>22}  "
                      f"err {med_range([m['error'] for m in cm]):>22}  frame {med_range([m['median_frame'] for m in cm], False):>18}"
                      f"  size {med_range([m['median_size'] for m in cm], False):>20}")
    print("\nSmallest certifiable alpha per seed 0-4 (- = none in the grid):")
    print(f"  {'score':>7} " + " ".join(f"{r:>30}" for r in rules))
    for s in SC.SCORES:
        row = [" ".join("-" if res[s]["per_seed"][sd]["smallest_alpha"][r] is None
                        else f"{100 * res[s]['per_seed'][sd]['smallest_alpha'][r]:g}%" for sd in seeds) for r in rules]
        print(f"  {s:>7} " + " ".join(f"{x:>30}" for x in row))

    out = run / "poc3" / "stage3"
    code = {f"{m}.py": sha256_file(Path(__file__).parent / f"{m}.py") for m in ("r9_stage3", "r9_stage3_scores")}
    summary = {"plan": "poc3-implementation-plan-r9 Stage 3 (exploratory, post-hoc); LOG_r9_stage3_entry.md Part A",
               "frozen_blobs": blobs, "code_sha256": code,
               "inputs_sha256": {**d["inputs_sha256"], "stage3_design.json": DESIGN_SHA256,
                                 "stage4_summary.json": STAGE4_SUMMARY_SHA256, "r8_stage4.py": STAGE4_CODE_SHA256,
                                 "train_meta.csv": TRAIN_META_SHA256},
               "checks": checks, "n_pool": len(F["rows"]), "n_cert": n_cert, "n_test": n_test, "n_design": n_design,
               "alphas": alphas, "rules": rules, "s_values": s_values, "no_value": no_value,
               "results": {s: {str(sd): {"cert": {str(al): v for al, v in c["cert"].items()},
                                         "test": {str(al): v for al, v in c["test"].items()},
                                         "reference": {str(al): v for al, v in c["reference"].items()},
                                         "smallest_alpha": c["smallest_alpha"]}
                               for sd, c in res[s]["per_seed"].items()} for s in SC.SCORES},
               "seconds": round(time.time() - t0, 1)}
    (out / "stage3_summary.json").write_text(json.dumps(S2.jsonable(summary), indent=1))
    print(f"\nOutput: {out / 'stage3_summary.json'} ({summary['seconds']} s)")
    print("All results above are exploratory (post-hoc).")
    return summary


if __name__ == "__main__":
    main()
