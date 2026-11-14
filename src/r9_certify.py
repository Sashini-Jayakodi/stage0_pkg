"""POC-3 (R9) certification engine v2: sign-level Learn-then-Test with an exact binomial test.

LOG_r9_stage2_entry.md Part A (a). Pure numpy/scipy, no file I/O. Frozen after the Stage 2 V-checks pass.

Grid convention: thresholds j = 0..G-1 ordered from the loosest (j = 0) to the strictest. Per threshold:
  n_c = committed Cert units, k = committed and wrong Cert units, n = all Cert units.
  exact binomial : p = P(Binomial(n_c, alpha) <= k)       (H0: P(wrong | committed) >= alpha)
  comparator     : Hoeffding-Bentkus on L = wrong - alpha * committed + alpha (src.certify.hb_pvalue, unchanged)
A threshold with n_c = 0 gets p = 1 under both tests, so it is never certified.
Power-aware start (plan 5.3): the strictest j with f_design(j) * n * alpha >= 4. The walk tests start, start-1, ..., 0
at level delta and stops at the first p > delta.
"""
import numpy as np
from scipy.stats import binom

from src.certify import hb_pvalue

DELTA = 0.1              # fixed in every phase (plan section 2)
MIN_EXPECTED = 4         # power-aware start: expected errors at alpha among committed units (plan 5.3)


def binom_pvalue(k, n_c, alpha):
    """Exact binomial p-value per threshold. n_c = 0 gives 1 by definition (explicit, not left to scipy)."""
    k = np.asarray(k, dtype=float)
    n_c = np.asarray(n_c, dtype=float)
    p = binom.cdf(k, np.maximum(n_c, 1.0), alpha)
    return np.where(n_c > 0, p, 1.0)


def hb_pvalue_counts(k, n_c, n, alpha):
    """Hoeffding-Bentkus comparator from counts. Mean shifted loss mu = alpha * (1 - n_c / n) + k / n over all n Cert
    units, as POC-2 (r8_stage3_engine.track_mu). n_c = 0 gives 1 (the formula gives the same; the guard makes it sure)."""
    k = np.asarray(k, dtype=float)
    n_c = np.asarray(n_c, dtype=float)
    mu = alpha * (1.0 - n_c / n) + k / n
    return np.where(n_c > 0, hb_pvalue(mu, n, alpha), 1.0)


def pvalues(k, n_c, n, alpha, test="binom"):
    if test == "binom":
        return binom_pvalue(k, n_c, alpha)
    if test == "hb":
        return hb_pvalue_counts(k, n_c, n, alpha)
    raise ValueError(f"unknown test {test!r}")


def power_start(f_design, n, alpha, min_expected=MIN_EXPECTED):
    """Strictest grid index whose design commit fraction * n * alpha >= min_expected; -1 if none.
    Uses design fractions, n and alpha only: Cert outcomes cannot move the start."""
    ok = np.flatnonzero(np.asarray(f_design, dtype=float) * n * alpha >= min_expected)
    return int(ok.max()) if len(ok) else -1


def walk(p, start, delta=DELTA):
    """Fixed-sequence test from 'start' down to 0. Returns the last index with p <= delta, or -1."""
    best = -1
    for j in range(start, -1, -1):
        if p[j] > delta:
            break
        best = j
    return best


def certify(k, n_c, n, alpha, f_design, test="binom", delta=DELTA):
    """Certified grid index (or -1) for one Cert sample. k, n_c: per threshold, loosest first."""
    p = pvalues(k, n_c, n, alpha, test)
    return walk(p, power_start(f_design, n, alpha), delta)


def max_errors(n_c, alpha, delta=DELTA):
    """Largest k with p <= delta at this n_c, or None if not certifiable even with 0 errors (plan section 6)."""
    if binom_pvalue(0, n_c, alpha) > delta:
        return None
    k = 0
    while binom_pvalue(k + 1, n_c, alpha) <= delta:
        k += 1
    return k


def min_n_zero_errors(alpha, delta=DELTA, n_max=100_000):
    """Smallest n_c that certifies with 0 errors: (1 - alpha)^n_c <= delta."""
    for n in range(1, n_max + 1):
        if binom_pvalue(0, n, alpha) <= delta:
            return n
    return None
