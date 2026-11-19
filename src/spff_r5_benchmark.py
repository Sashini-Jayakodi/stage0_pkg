"""Finite-pilot synthetic check of selection-path excess-severity inference."""
import argparse
import json
from pathlib import Path
import numpy as np

from src.spff_r5_path import exact_path, pilot_confidence_interval


SIGNATURES = np.array([
    [[0, 0], [1, 0], [1, 0]],
    [[1, 1], [1, 1], [1, 0]],
    [[1, 0], [0, 0], [0, 0]],
], dtype=int)


def run(seed=20261119, replicates=30, boots=40, n=8, alpha=0.3, delta=0.3):
    rng = np.random.default_rng(seed)
    q = np.array([0.5, 0.2, 0.3])
    truth = exact_path(SIGNATURES, q, n, alpha, delta)["severity"]
    rows = []
    for m in (200, 1000, 3000):
        estimates = []
        honest_covered = []
        honest_width = []
        percentile_covered = []
        percentile_width = []
        for _ in range(replicates):
            counts = rng.multinomial(m, q)
            honest = pilot_confidence_interval(SIGNATURES, counts, n, alpha, delta)
            estimates.append(honest["estimate"])
            honest_covered.append(honest["lower"] <= truth <= honest["upper"])
            honest_width.append(honest["upper"] - honest["lower"])
            p = counts / m
            boot_vals = [
                exact_path(SIGNATURES, rng.multinomial(m, p) / m, n, alpha, delta)["severity"]
                for _ in range(boots)
            ]
            lo, hi = np.quantile(boot_vals, [0.025, 0.975])
            percentile_covered.append(lo <= truth <= hi)
            percentile_width.append(hi - lo)
        rows.append({
            "pilot_signs": m, "replicates": replicates,
            "absolute_error_mean": float(np.mean(np.abs(np.array(estimates) - truth))),
            "honest_interval_coverage": float(np.mean(honest_covered)),
            "honest_interval_mean_width": float(np.mean(honest_width)),
            "percentile_bootstrap_coverage": float(np.mean(percentile_covered)),
            "percentile_bootstrap_mean_width": float(np.mean(percentile_width))
        })
    return {
        "setting": {
            "seed": seed, "calibration_signs": n, "alpha": alpha,
            "delta": delta, "bootstrap_replicates": boots,
            "signature_probabilities": q.tolist()
        },
        "exact_population_severity": truth, "rows": rows,
        "caveat": "Synthetic fixed three-pattern case. Percentile intervals are not finite-sample guarantees. Broad interval covers by construction but may not resolve risk."
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path)
    p.add_argument("--replicates", type=int, default=30)
    p.add_argument("--boots", type=int, default=40)
    args = p.parse_args()
    data = run(replicates=args.replicates, boots=args.boots)
    content = json.dumps(data, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content + "\n", encoding="utf-8")
    print(content)


if __name__ == "__main__":
    main()
