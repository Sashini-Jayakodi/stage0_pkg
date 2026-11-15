"""POC-3 (R9) Stage 5, freeze F3: writes configs/frozen_r9_btsc.yaml (LOG_r9_stage5_entry.md Part A (l)).

The design-values file is hash-checked first (hard rule 2). The design phase is then recomputed from the hash-checked
Stage 4 inputs and must reproduce that file exactly; the yaml is written from the recomputation. Why: the JSON file
stores every non-finite number as null, so a B6 threshold of -inf ("commit at view 1") and None ("commit nothing") look
the same there; the recomputation keeps them apart, and the reproduction check proves it is the same design.

This script only writes the yaml. Adding it to FROZEN_BLOBS, the pre-registration block and the F3 commit are separate
steps that wait for the author's go.

Usage:  py -3.12 -m src.r9_stage5_freeze
"""
import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from src import r9_frozen as FZ
from src import r9_stage5_design as DS
from src import r9_stage5_scores as SC
from src.r9_btsc_guard import f3_closed
from src.r9_stage4_data import stop
from src.util import sha256_file

DESIGN = Path("runs/btsc/stage5/stage5_design.json")
DESIGN_SHA256 = "115f98025f60b14aedad9c0d75df6d5deafc568d77b427e2bf9804be1d854a11"   # Part A, design-pole values
DESIGN_BLOB = "b06a09f8c64bbd789bd02addab0b518ecde66d0e"
YAML_OUT = Path("configs/frozen_r9_btsc.yaml")
RUN_FIELDS = ("run_utc", "git_head", "worktree_clean", "seconds")   # differ between runs by nature


def plain(x):
    """Python types for yaml, keeping inf (unlike the JSON writer, which turns it into null)."""
    if isinstance(x, dict):
        return {str(k): plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [plain(v) for v in x]
    if isinstance(x, np.ndarray):
        return plain(x.tolist())
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, (np.floating, float)):
        return float(x)
    return x


def load_design(path=DESIGN, sha=DESIGN_SHA256, blob=DESIGN_BLOB):
    path = Path(path)
    if not path.is_file():
        stop(f"{path} is missing")
    how = DS.check_text(path, sha, blob)
    return json.loads(path.read_text()), how


def reproduce(R, d, design, blobs):
    """The recomputed design must equal the recorded design file in every field except the run fields."""
    code = {f: sha256_file(f) for f in DS.CODE}
    again = json.loads(json.dumps(DS.to_json(R, d, blobs, code, 0.0)))
    diff = sorted(k for k in set(design) | set(again)
                  if k not in RUN_FIELDS and design.get(k) != again.get(k))
    if diff:
        stop(f"the recomputed design differs from the recorded design file in: {diff}")


def build(R, design, design_sha, design_blob):
    """Everything Part A (l) lists, from the recomputed design R (exact values) and the recorded file's hashes."""
    alphas = list(R["alphas"])
    key = lambda a: f"{a:g}"
    scores = {}
    for s in SC.SCORES:
        S, st = R["scores"][s], R["scores"][s]["stats"]
        scores[s] = {"grid": S["grid"], "f_d": st["f_d"], "e_d": st["e_d"], "committed": st["committed"],
                     "wrong": st["wrong"],
                     "starts": {seed: {key(a): S["starts"][seed][a] for a in alphas} for seed in SC.SEEDS},
                     "reference": {key(a): S["reference"][a] for a in alphas}}
    rule3 = {s: {key(a): {"predicted": R["rule3"][s][a]["predicted"],
                          "deciding": R["rule3"][s][a]["deciding"]} for a in alphas} for s in SC.SCORES}
    out = {
        "frozen": "F3, POC-3 plan R9 Stage 5 (LOG_r9_stage5_entry.md Part A (l)). Nothing here changes after F3.",
        "dataset": "BelgiumTSC testing release, pool poles (Stage 4 split)",
        "unit": "(pole, class)", "view_order": "ascending YYYYY (Stage 1 Amendment 2)", "gate": "none",
        "delta": 0.1, "alphas": [key(a) for a in alphas], "alpha_10pct_added": SC.ALPHA_EXTRA in alphas,
        "fd1_pass": R["fd1"], "primary_alpha": None if R["primary_alpha"] is None else key(R["primary_alpha"]),
        "fallback2": R["fallback2"],
        "rule3": {"n_min": R["n_min"], "f_d_min": 0.70, "ucb_one_sided_level": SC.RULE3_UCB_Q,
                  "n_c": "floor(f_d * n_min)", "table": rule3},
        "rules": list(SC.RULES), "primary_rule": "binom", "primary_hb_start": "hb",
        "scores": scores, "primary_s2": SC.PRIMARY_S2,
        "s2": {s: {"a": m["a"], "b": m["b"], "combo": m["combo"], "max_window": m["max_window"],
                   "icp_cal_count": len(m["cal"]),
                   "icp_cal_sha256": design["s2_models"][s]["icp_cal_sha256"],
                   "ecdf": {L: v for L, v in m["ecdf"].items()}} for s, m in R["models"].items()},
        "s2_halves": {"seed": SC.HALF_SEED, "A_poles": R["halves"]["A_poles"], "B_poles": R["halves"]["B_poles"]},
        "resplits": {"seeds": list(SC.SEEDS), "test_fraction": list(SC.TEST_FRACTION),
                     "split": {s: R["resplits"][s] for s in SC.SEEDS}},
        "rq1": {"draws": SC.RQ1_DRAWS, "salt": list(SC.RQ1_SALT), "poles_per_draw": len(R["resplits"][0]["cert_poles"]),
                "scope": "all score variants and B1, every alpha with a start, exact binomial and HB primary start; "
                         "all reported"},
        "stage6_v_check": {"bound": "upper 95% Clopper-Pearson of the sign-level violation rate <= delta + 0.02",
                           "scored_cells": "every score variant at the primary alpha, exact binomial; "
                                           "all other cells reported, not scored"},
        "b1": {"grid": R["b1"]["grid"], "f_d": R["b1"]["stats"]["f_d"], "e_d": R["b1"]["stats"]["e_d"],
               "n_design_views": R["b1"]["n_design_views"], "score": "S4 single-view margin, view as unit"},
        "b2": {"salt": list(SC.B2_SALT), "design_key": SC.B2_DESIGN_KEY, "design_draw": R["b2"]["draw"],
               "grid": R["b2"]["grid"], "f_d": R["b2"]["stats"]["f_d"], "e_d": R["b2"]["stats"]["e_d"],
               "starts": {seed: {key(a): R["b2"]["starts"][seed][a] for a in alphas} for seed in SC.SEEDS}},
        "b5": {"score": "pi = F_design(S1 margin)", "f_design": R["b5"]["f_design"],
               "lambda_grid": "0, 0.01, ..., 1 and infinity", "b5c_salt": SC.B5C_SALT, "b5c_tmax": SC.B5C_TMAX},
        "b6": {"theta": {s: {key(a): th[a] for a in alphas} for s, th in R["b6"].items()},
               "note": "theta -.inf = commit at view 1; null = no theta meets r (commit nothing)"},
        "correlation_design": R["correlation"],
        "sources": {"design_values": {"path": DESIGN.as_posix(), "sha256": design_sha, "git_blob": design_blob},
                    "stage4_inputs_sha256": design["inputs_sha256"], "stage4_inputs_blob": design["inputs_blob"],
                    "frozen_blobs_at_design": design["frozen_blobs"], "code_sha256": design["code_sha256"]}}
    return plain(out)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage4", default=str(DS.STAGE4))
    ap.add_argument("--design", default=str(DESIGN))
    ap.add_argument("--yaml", default=str(YAML_OUT))
    a = ap.parse_args(argv)
    blobs = FZ.check_frozen()
    assert not f3_closed(), "F3 is already closed; the frozen yaml must not be rewritten"
    target = Path(a.yaml)
    if target.exists():
        stop(f"{target} exists; refusing to overwrite")
    design, how = load_design(a.design)
    DS.OPENED.clear()
    d = DS.load_stage4(a.stage4)
    R = DS.run_design(d)
    assert not any(f in p for p in DS.OPENED for f in DS.FORBIDDEN), "a pool or model file was opened"
    reproduce(R, d, design, blobs)
    y = build(R, design, sha256_file(a.design), FZ.blob_id(a.design))
    target.write_text(yaml.safe_dump(y, sort_keys=False, default_flow_style=None, width=120))
    print(f"design file check: {how}; recomputed design reproduces the recorded file: PASS")
    print(f"wrote {target}  SHA-256 {sha256_file(target)}  git blob {FZ.blob_id(target)}")
    print(f"primary alpha {y['primary_alpha']} | F-D1 {'PASS' if y['fd1_pass'] else 'FAIL'} | alphas {y['alphas']}")
    print("Not frozen yet: FROZEN_BLOBS, the pre-registration block and the F3 commit wait for the author's go.")
    return y


if __name__ == "__main__":
    main()
