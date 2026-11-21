"""Reference computation for exact selection paths on physical-sign signatures."""
from collections import defaultdict
from math import log, sqrt
from numbers import Integral
import numpy as np
from scipy.stats import binom


def _integer_scalar(value, name, minimum=0, maximum=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f"{name} is outside its allowed range")
    return value


def _count_array(values, name):
    """Check raw counts before conversion, including the int64 upper boundary."""
    raw = np.asarray(values)
    if raw.dtype.kind not in "iuf" or not np.isfinite(raw).all():
        raise ValueError(f"{name} must contain finite nonnegative integer counts")
    if np.any(raw < 0):
        raise ValueError(f"{name} must contain finite nonnegative integer counts")
    if raw.dtype.kind == "f":
        if np.any(raw != np.floor(raw)) or np.any(raw >= 2**63):
            raise ValueError(f"{name} must contain integer counts in int64 range")
    elif raw.dtype.kind == "u" and np.any(raw > np.iinfo(np.int64).max):
        raise ValueError(f"{name} must contain integer counts in int64 range")
    return raw.astype(np.int64)


def _fixed_start(start, g):
    if start is None:
        return g - 1
    return _integer_scalar(start, "start", minimum=-1, maximum=g - 1)


def validate(signatures, probabilities):
    x = np.asarray(signatures)
    q = np.asarray(probabilities, dtype=float)
    if x.ndim != 3 or x.shape[2] != 2 or x.shape[0] == 0 or x.shape[1] == 0:
        raise ValueError("signatures must have shape (patterns, thresholds, 2)")
    if (x.dtype.kind not in "biuf" or not np.isfinite(x).all()
            or not np.isin(x, [0, 1]).all()
            or np.any(x[:, :, 1] > x[:, :, 0])):
        raise ValueError("signature indicators must be binary with wrong <= accepted")
    if q.shape != (x.shape[0],) or not np.isfinite(q).all() or np.any(q < 0):
        raise ValueError("one nonnegative finite probability per signature is required")
    if not np.isclose(q.sum(), 1, atol=1e-12, rtol=0):
        raise ValueError("signature probabilities must sum to one")
    return x.astype(np.int64), q


def _binomial_pvalues(k, n, alpha):
    # Match frozen R9's float conversion and explicit zero-acceptance convention.
    k, n = np.asarray(k, dtype=float), np.asarray(n, dtype=float)
    p = binom.cdf(k, np.maximum(n, 1.0), alpha)
    return np.where(n > 0, p, 1.0)


def binomial_lower_tail(k, n, alpha):
    return float(_binomial_pvalues(k, n, alpha))


def selected_threshold(counts, alpha, delta, start=None):
    """Return the final passing threshold; start=-1 means no certification."""
    c = _count_array(counts, "counts")
    if c.ndim != 2 or c.shape[1] != 2 or c.shape[0] == 0:
        raise ValueError("counts must have shape (thresholds, 2)")
    if np.any(c[:, 1] > c[:, 0]):
        raise ValueError("counts must satisfy wrong <= accepted")
    if not (0 < alpha < 1 and 0 < delta < 1):
        raise ValueError("alpha and delta must lie between zero and one")
    start = _fixed_start(start, c.shape[0])
    p = _binomial_pvalues(c[:, 1], c[:, 0], alpha)
    best = -1
    for j in range(start, -1, -1):
        if p[j] > delta:
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
    """Evaluate the full joint count law for an integer number of iid signs."""
    x, q = validate(signatures, probabilities)
    n = _integer_scalar(n, "n", maximum=100000)
    if not (0 < alpha < 1 and 0 < delta < 1):
        raise ValueError("alpha and delta must lie between zero and one")
    max_states = _integer_scalar(max_states, "max_states", minimum=1)
    g = x.shape[1]
    start = _fixed_start(start, g)
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
                              miscoverage=0.05, max_states=250000, start=None):
    """Conservative interval on a fixed, complete alphabet and fixed start.

    Integer-valued real pilot counts are accepted; calibration n must have an
    integer type. The start is held fixed, including the valid no-start value -1.
    """
    counts = _count_array(pilot_counts, "pilot_counts")
    if counts.ndim != 1 or len(counts) != len(signatures):
        raise ValueError("pilot_counts must cover the fixed signature alphabet")
    # Python addition avoids wrapping when several valid int64 counts are summed.
    m = sum(int(count) for count in counts)
    if m <= 0 or not 0 < miscoverage < 1:
        raise ValueError("pilot must be nonempty and miscoverage in (0,1)")
    p = counts / m
    point = exact_path(signatures, p, n, alpha, delta, start=start,
                       max_states=max_states)
    eps_coordinate = sqrt(log(2 * len(counts) / miscoverage) / (2 * m))
    tv_upper = min(1.0, len(counts) * eps_coordinate / 2)
    halfwidth = min(1.0, (n + 1) * tv_upper)
    return {"estimate": point["severity"],
            "lower": max(0.0, point["severity"] - halfwidth),
            "upper": min(1.0, point["severity"] + halfwidth),
            "confidence": 1 - miscoverage, "radius": halfwidth,
            "tv_upper": tv_upper, "pilot_signs": m,
            "selected_probabilities": point["selected_probabilities"]}
