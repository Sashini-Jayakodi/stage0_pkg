"""Exact small-support reference for LTT selection on joint frame-count clusters.

The sampled unit is a physical-sign cluster, not an individual frame. The
reference models a *naive* frame-level binomial certificate acting on pooled
accepted/wrong frame counts; this is not a valid sign-level risk guarantee.
"""
from collections import defaultdict
from math import isfinite
import numpy as np
from scipy.stats import binom


def _integer_array(value, name, ndim, positive=False):
    raw = np.asarray(value)
    if raw.ndim != ndim or raw.size == 0:
        raise ValueError(f"{name} must have {ndim} nonempty dimensions")
    if not np.issubdtype(raw.dtype, np.number) or np.iscomplexobj(raw):
        raise ValueError(f"{name} must contain real integer values")
    if not np.all(np.isfinite(raw)) or not np.all(raw == np.floor(raw)):
        raise ValueError(f"{name} must contain finite integers")
    if np.any(raw < (1 if positive else 0)):
        raise ValueError(f"{name} contains values outside its nonnegative domain")
    # Comparing float64 against int64.max silently rounds that bound up to
    # 2**63, admitting values which overflow during astype(np.int64).
    # Compare floating input with the first excluded integer instead.
    if np.issubdtype(raw.dtype, np.floating):
        if np.any(raw >= float(2**63)):
            raise ValueError(f"{name} contains integers outside int64 range")
    elif np.any(raw > np.iinfo(np.int64).max):
        raise ValueError(f"{name} contains integers outside int64 range")
    return raw.astype(np.int64)


def validate(signatures, probabilities, cluster_sizes, nested=False):
    """Validate jointly observed counts, independent-cluster probabilities and T.

    `nested=True` enforces nested acceptance from loose index 0 toward strict
    G-1, and enforces nesting of correct and wrong accepted frame subsets.
    Nonmonotone wrong decisions remain supported with `nested=False`.
    """
    x = _integer_array(signatures, "signatures", 3)
    if x.shape[-1] != 2 or x.shape[0] < 1 or x.shape[1] < 1:
        raise ValueError("signatures must have shape (patterns, thresholds, 2)")
    t = _integer_array(cluster_sizes, "cluster_sizes", 1, positive=True)
    if t.shape != (x.shape[0],):
        raise ValueError("cluster_sizes must have one entry per pattern")
    if np.any(x[:, :, 1] > x[:, :, 0]) or np.any(x[:, :, 0] > t[:, None]):
        raise ValueError("every wrong count <= accepted count <= cluster size")
    if nested and x.shape[1] > 1:
        # C, W and C-W must all decrease (or tie) as the threshold gets stricter.
        c = x[:, :, 0]
        w = x[:, :, 1]
        if np.any(np.diff(c, axis=1) > 0) or np.any(np.diff(w, axis=1) > 0):
            raise ValueError("nested counts require nonincreasing accepted and wrong")
        if np.any(np.diff(c - w, axis=1) > 0):
            raise ValueError("nested counts require nonincreasing correct accepted")
    q = np.asarray(probabilities)
    if q.shape != (x.shape[0],) or not np.issubdtype(q.dtype, np.number) or np.iscomplexobj(q):
        raise ValueError("probabilities must have shape (patterns,)")
    q = q.astype(float)
    if not np.all(np.isfinite(q)) or np.any(q < 0) or not np.isclose(q.sum(), 1, atol=1e-12, rtol=0):
        raise ValueError("probabilities must be nonnegative, finite and sum to one")
    return x, q, t


def _nonnegative_int(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return int(value)


def selected_threshold(accepted_and_wrong, alpha, delta=0.1, start=None):
    """Frozen R9 walk semantics, applied to pooled frame-count totals."""
    c = _integer_array(accepted_and_wrong, "accepted_and_wrong", 2)
    if c.shape[1] != 2 or np.any(c[:, 1] > c[:, 0]):
        raise ValueError("threshold counts must be (accepted, wrong), wrong<=accepted")
    if not (isfinite(alpha) and 0 < alpha < 1 and isfinite(delta) and 0 < delta < 1):
        raise ValueError("alpha and delta must lie strictly between zero and one")
    g = c.shape[0]
    s = g - 1 if start is None else start
    if isinstance(s, (bool, np.bool_)) or not isinstance(s, (int, np.integer)) or not -1 <= s < g:
        raise ValueError("start must be -1 or a valid threshold index")
    best = -1
    for j in range(int(s), -1, -1):
        accepted, wrong = map(int, c[j])
        pval = float(binom.cdf(wrong, accepted, alpha)) if accepted else 1.0
        if pval > delta:
            break
        best = j
    return best


def excess_per_threshold(signatures, probabilities, cluster_sizes, alpha, nested=False):
    x, q, t = validate(signatures, probabilities, cluster_sizes, nested=nested)
    if not (isfinite(alpha) and 0 < alpha < 1):
        raise ValueError("alpha must lie strictly between zero and one")
    a = q @ x[:, :, 0]
    w = q @ x[:, :, 1]
    mean_frames = float(q @ t)
    e = np.maximum(0.0, w - alpha * a) / mean_frames
    return e, a, w, mean_frames


def exact_cluster_path(signatures, probabilities, cluster_sizes, calibration_clusters,
                       alpha, delta=0.1, start=None, max_states=250000, nested=False):
    """Propagate full joint count states for independent sampled clusters.

    No approximation is performed. Arithmetic is floating-point; the guard
    bounds reachable state count *during* construction, not all memory use.
    """
    x, q, t = validate(signatures, probabilities, cluster_sizes, nested=nested)
    n = _nonnegative_int(calibration_clusters, "calibration_clusters")
    if not (isfinite(alpha) and 0 < alpha < 1 and isfinite(delta) and 0 < delta < 1):
        raise ValueError("alpha and delta must lie strictly between zero and one")
    cap = _nonnegative_int(max_states, "max_states")
    if cap == 0:
        raise ValueError("max_states must be at least one")
    g = x.shape[1]
    if start is not None and (isinstance(start, (bool,np.bool_)) or not isinstance(start,(int,np.integer)) or not -1 <= start < g):
        raise ValueError("invalid start")
    s = g - 1 if start is None else int(start)
    e, a, w, mean_frames = excess_per_threshold(x, q, t, alpha, nested=nested)
    unsafe = (a > 0) & (w > alpha*a)
    if s == -1:
        return {
            "severity_per_frame": 0.0,
            "failure_probability": 0.0,
            "selected_probabilities": np.r_[1.0, np.zeros(g)],
            "accepted_per_cluster": a,
            "wrong_per_cluster": w,
            "mean_frames_per_cluster": mean_frames,
            "excess_per_frame": e,
            "unsafe": unsafe,
            "reachable_states": 1,
            "calibration_clusters": n,
        }
    # Avoid passing totals that exceed int64 to the SciPy selection routine.
    # This is a conservative check: it also rejects unattainable extrema.
    for j in range(g):
        for k in range(2):
            if n * int(x[:, j, k].max()) > np.iinfo(np.int64).max:
                raise ValueError("aggregate count may exceed int64 range")
    patterns = [tuple(map(int, row.reshape(-1))) for row in x]
    states = {(0,) * (2*g): 1.}
    for step in range(n):
        new = defaultdict(float)
        for state, mass in states.items():
            for pattern, prob in zip(patterns, q):
                if prob <= 0:
                    continue
                key = tuple(a+b for a,b in zip(state, pattern))
                new[key] += mass * prob
                if len(new) > cap:
                    raise RuntimeError(f"state limit exceeded at cluster {step+1}: {len(new)}")
        states = new
    selection = np.zeros(g + 1)
    for state, mass in states.items():
        j = selected_threshold(np.asarray(state).reshape(g, 2), alpha, delta, start)
        selection[j + 1] += mass
    return {
        "severity_per_frame": float(selection[1:] @ e),
        "failure_probability": float(selection[1:] @ unsafe.astype(float)),
        "selected_probabilities": selection,
        "accepted_per_cluster": a,
        "wrong_per_cluster": w,
        "mean_frames_per_cluster": mean_frames,
        "excess_per_frame": e,
        "unsafe": unsafe,
        "reachable_states": len(states),
        "calibration_clusters": n,
    }
