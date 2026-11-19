"""Reference computation for exact selection paths on physical-sign signatures."""
from collections import defaultdict
from math import comb, log, sqrt
import numpy as np


def validate(signatures, probabilities):
    x = np.asarray(signatures, dtype=int)
    q = np.asarray(probabilities, dtype=float)
    if x.ndim != 3 or x.shape[2] != 2 or x.shape[0] == 0 or x.shape[1] == 0:
        raise ValueError("signatures must have shape (patterns, thresholds, 2)")
    if not np.isin(x, [0, 1]).all() or np.any(x[:, :, 1] > x[:, :, 0]):
        raise ValueError("signature indicators must satisfy wrong <= accepted")
    if q.shape != (x.shape[0],) or not np.isfinite(q).all() or np.any(q < 0):
        raise ValueError("one nonnegative finite probability per signature is required")
    if not np.isclose(q.sum(), 1, atol=1e-12, rtol=0):
        raise ValueError("signature probabilities must sum to one")
    return x, q


def binomial_lower_tail(k, n, alpha):
    if n == 0:
        return 1.0
    return sum(comb(n, t) * alpha**t * (1 - alpha)**(n - t)
               for t in range(k + 1))


def selected_threshold(counts, alpha, delta, start=None):
    c = np.asarray(counts, dtype=int)
    if c.ndim != 2 or c.shape[1] != 2:
        raise ValueError("counts must have shape (thresholds, 2)")
    if start is None:
        start = c.shape[0] - 1
    if not 0 <= start < c.shape[0]:
        raise ValueError("invalid fixed-sequence start")
    best = -1
    for j in range(start, -1, -1):
        if binomial_lower_tail(int(c[j, 1]), int(c[j, 0]), alpha) > delta:
            break
        best = j
    return best


def excess_per_threshold(signatures, probabilities, alpha):
    x, q = validate(signatures, probabilities)
    a = q @ x[:, :, 0]
    w = q @ x[:, :, 1]
    return np.maximum(w - alpha * a, 0.0), a, w


def exact_path(signatures, probabilities, n, alpha, delta=0.1, start=None,
               max_states=250000):
    x, q = validate(signatures, probabilities)
    if int(n) != n or n < 0 or n > 100000:
        raise ValueError("n must be a nonnegative integer")
    if not (0 < alpha < 1 and 0 < delta < 1):
        raise ValueError("alpha and delta must lie between zero and one")
    if max_states < 1:
        raise ValueError("max_states must be positive")
    g = x.shape[1]
    patterns = [tuple(int(v) for v in row.reshape(-1)) for row in x]
    states = {tuple([0] * (2 * g)): 1.0}
    for step in range(n):
        next_states = defaultdict(float)
        for state, mass in states.items():
            for pattern, probability in zip(patterns, q):
                if probability > 0:
                    key = tuple(a + b for a, b in zip(state, pattern))
                    next_states[key] += mass * probability
        if len(next_states) > max_states:
            raise RuntimeError(f"state limit exceeded at sign {step + 1}: {len(next_states)} states")
        states = next_states
    selected = np.zeros(g + 1, dtype=float)
    for state, mass in states.items():
        j = selected_threshold(np.asarray(state).reshape(g, 2), alpha, delta, start)
        selected[j + 1] += mass
    excess, acceptance, wrong = excess_per_threshold(x, q, alpha)
    severity = float(selected[1:] @ excess)
    unsafe = np.logical_and(acceptance > 0, wrong > alpha * acceptance)
    failure = float(selected[1:] @ unsafe.astype(float))
    return {"severity": severity, "failure_probability": failure,
            "selected_probabilities": selected, "states": len(states),
            "acceptance": acceptance, "wrong": wrong, "excess": excess}


def pilot_confidence_interval(signatures, pilot_counts, n, alpha, delta=0.1,
                              miscoverage=0.05, max_states=250000):
    """Conservative simultaneous interval on a fixed, complete signature alphabet."""
    counts = np.asarray(pilot_counts, dtype=int)
    if counts.ndim != 1 or len(counts) != len(signatures) or np.any(counts < 0):
        raise ValueError("pilot_counts must cover the fixed signature alphabet")
    m = int(counts.sum())
    if m <= 0 or not 0 < miscoverage < 1:
        raise ValueError("pilot must be nonempty and miscoverage in (0,1)")
    p = counts / m
    point = exact_path(signatures, p, n, alpha, delta, max_states=max_states)
    eps_coordinate = sqrt(log(2 * len(counts) / miscoverage) / (2 * m))
    tv_upper = min(1.0, len(counts) * eps_coordinate / 2)
    halfwidth = min(1.0, (n + 1) * tv_upper)
    return {"estimate": point["severity"],
            "lower": max(0.0, point["severity"] - halfwidth),
            "upper": min(1.0, point["severity"] + halfwidth),
            "confidence": 1 - miscoverage, "radius": halfwidth,
            "tv_upper": tv_upper, "pilot_signs": m,
            "selected_probabilities": point["selected_probabilities"]}
