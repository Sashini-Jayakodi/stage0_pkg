"""POC-2 (R8) Stage 6, step 1: the only design-set quantity of Stage 6, the B5/B5c score transform.
LOG_r8_stage6_entry.md Part A (b) and Amendment 1. DESIGN SET ONLY (261 tracks); the pool table is never opened.
After Amendment 1, nothing in B6 is fitted on the design set (B6 is calibrated on Cert parts in each re-split).

  pi(X_t) = F_design(M_t): the empirical CDF of the V1 margins over all design frames with evidence.
  Also measured: the share of design frames whose largest V1 posterior is exactly 1.0 in float64, the reason the
  paper's "largest class probability" cannot be used directly on our accumulated score.

Usage:  py -3.12 -m src.r8_stage6_design --run runs/20261003_stage1
Output: runs/20261003_stage1/poc2/stage6/b5_design.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

from src import r8_stage1 as R1
from src import r8_stage2 as S2
from src.r8_frozen import load_frozen
from src.util import sha256_file

PAPERS = {"docs/papers/ringel24a.pdf": "cc1ffe7245b21ab282569a522b3b308493c1b1e982cc20bc06098359614b1039",
          "docs/papers/boursinos_koutsoukos_selective_classification.pdf":
          "af0f13ce31e6c8fadad7e13f04d8445bf811c183569c10b7be2f022a1ba5b557"}


def check_papers(root=Path(".")):
    for rel, h in PAPERS.items():
        assert sha256_file(Path(root) / rel) == h, f"{rel} differs from the hash in Stage 6 Part A"


def design_margins(ll, passed, log_prior, rows):
    """V1 margins of every frame with evidence, and how many of those frames have a saturated largest posterior."""
    margins, sat, nev = [], 0, 0
    for r in rows:
        S, _ = R1.running_scores(ll[r], passed[r], log_prior, "V1")
        M, _ = R1.margins(S)
        ok = np.isfinite(M)
        margins.append(M[ok])
        Sv = S[ok]
        pmax = np.exp(Sv.max(1) - logsumexp(Sv, axis=1))
        sat += int((pmax == 1.0).sum())
        nev += int(ok.sum())
    return np.sort(np.concatenate(margins)), sat, nev


def score(M, F_sorted):
    """pi = share of design margins <= M; -1 where there is no evidence yet (M = -inf), so no stop is possible there."""
    M = np.asarray(M, float)
    return np.where(np.isfinite(M), np.searchsorted(F_sorted, M, side="right") / len(F_sorted), -1.0)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    a = ap.parse_args(argv)
    load_frozen()
    check_papers()
    S2.OPENED.clear()
    d = S2.load_inputs(a.run)                                        # design rows only; refuses the pool file
    assert len(d["design_codes"]) == 261 and np.isin(d["track"], d["design_codes"]).all(), "non-design data reached Stage 6 design"
    rows = R1.track_rows(d["track"], d["frame"])
    ll, _ = R1.frame_evidence(d["head"], d["emb"])
    F, sat, nev = design_margins(ll, d["passed"], np.log(d["head"]["priors"]), rows)
    assert not any(S2.POOL_FILE in p for p in S2.OPENED), "track_table_pool.npz was opened"
    q = np.quantile(F, [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99])

    print(f"\n=== R8 Stage 6 step 1: B5/B5c score transform (design set only: {len(rows)} tracks) ===")
    print("Paper hashes checked: ringel24a.pdf, boursinos_koutsoukos_selective_classification.pdf")
    print(f"Design frames with evidence: {nev}")
    print(f"Largest V1 posterior exactly 1.0 (float64): {sat} frames ({100 * sat / nev:.2f}%)")
    print(f"V1 margin over those frames: min {F[0]:.4g}, max {F[-1]:.4g}; quantiles 1/10/25/50/75/90/99%: "
          + ", ".join(f"{x:.4g}" for x in q))
    print("Score pi = F_design(margin): 101 grid values {0, 0.01, ..., 1} map to design margin quantiles")
    print("B6: nothing fitted on the design set (Amendment 1).")
    out = Path(a.run) / "poc2" / "stage6"
    out.mkdir(parents=True, exist_ok=True)
    summary = {"plan": "poc2-implementation-plan-r8 Stage 6 step 1 (B5/B5c score transform, design set only)",
               "frozen_r8_sha256": sha256_file(Path("configs/frozen_r8.yaml")), "code_sha256": sha256_file(Path(__file__)),
               "papers_sha256": PAPERS, "inputs_sha256": d["inputs_sha256"], "files_opened": [Path(p).name for p in S2.OPENED],
               "frames_with_evidence": nev, "posterior_saturated_frames": sat, "F_margins": F}
    (out / "b5_design.json").write_text(json.dumps(S2.jsonable(summary)))
    print(f"Output: {out / 'b5_design.json'}")
    print("Design set only. No pool track was read; no B5/B6 Test number exists yet.")
    return summary


if __name__ == "__main__":
    main()
