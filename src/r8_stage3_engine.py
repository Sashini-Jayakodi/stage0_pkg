"""POC-2 (R8) Stage 3 engine: vectorised certification of many samples at once, plus the extra checks.

LOG_r8_stage3_entry.md Part A (b), (c), (d), (g), (h). Pure numpy/scipy; no file reading.
A sample is a weight vector over units (tracks): weight 1 = drawn once, 2 = drawn twice, 0 = not drawn. That covers
both the synthetic Cert sets (all weights 1) and bootstrap draws with replacement (3b) with one code path.

  track-level : unit = track. mean L = alpha * (1 - c) + w / n,          n = sum of weights          (Part A (b))
  B1          : unit = frame. mean L = (W + alpha * (N - A)) / N,        N = frames in the sample     (Part A (c))
The p-value is the POC-1 Hoeffding-Bentkus function, unchanged.
"""
import math
from decimal import Decimal, getcontext
from fractions import Fraction

import numpy as np
from scipy.stats import beta as beta_dist

from src.certify import hb_pvalue

TOL = 1e-12


# ------------------------------------------------------------------ mean losses per sample and threshold
def track_mu(committed, wrong, weights, alpha):
    """committed, wrong: (T, G) per track and threshold. weights: (D, T). Returns mean L (D, G) and n (D, 1)."""
    Wt = np.atleast_2d(np.asarray(weights, float))
    n = Wt.sum(1, keepdims=True)
    c = Wt @ committed.astype(float) / n
    w = Wt @ (wrong & committed).astype(float) / n
    return alpha * (1.0 - c) + w, n


def frame_mu(A, W, NF, weights, alpha):
    """A, W: (T, G) accepted / accepted-and-wrong FRAME counts per track; NF: (T,) frames per track. Every frame is its
    own unit, so n = number of frames in the sample. Returns mean L (D, G) and n (D, 1)."""
    Wt = np.atleast_2d(np.asarray(weights, float))
    N = (Wt @ np.asarray(NF, float))[:, None]
    return (Wt @ W.astype(float) + alpha * (N - Wt @ A.astype(float))) / N, N


# ------------------------------------------------------------------ fixed-sequence test (Part A (b))
def walk_rows(P, order, delta):
    """P: (D, G) p-values. Test the columns in 'order' at level delta, stop at the first p > delta.
    Returns the certified column per row, or -1 when the first test already fails."""
    order = np.asarray(order, int)
    passed = P[:, order] <= delta
    npass = np.cumprod(passed, axis=1).sum(1)
    out = np.full(len(P), -1, int)
    ok = npass > 0
    out[ok] = order[npass[ok] - 1]
    return out


def certify(mu, n, alpha, order, delta):
    return walk_rows(hb_pvalue(mu, n, alpha), order, delta)


def violations(cert, mu_true, alpha):
    """Part A (d): violation = something certified AND its true mean loss > alpha. Nothing certified = no violation."""
    cert = np.asarray(cert, int)
    v = np.zeros(len(cert), bool)
    ok = cert >= 0
    v[ok] = np.asarray(mu_true)[cert[ok]] > alpha + TOL
    return v


def cp_bounds(k, n, conf=0.95):
    """Two-sided Clopper-Pearson interval (Part A (g))."""
    a = (1 - conf) / 2
    lo = 0.0 if k == 0 else float(beta_dist.ppf(a, k, n - k + 1))
    hi = 1.0 if k == n else float(beta_dist.ppf(1 - a, k + 1, n - k))
    return lo, hi


def rate_row(cert, mu_true, alpha):
    v = violations(cert, mu_true, alpha)
    k, n = int(v.sum()), len(v)
    lo, hi = cp_bounds(k, n)
    ok = cert >= 0
    return {"runs": n, "certified_share": float(ok.mean()), "violations": k, "violation_rate": k / n, "cp95": [lo, hi],
            "mean_true_risk_certified": float(np.mean(np.asarray(mu_true)[cert[ok]])) if ok.any() else None}


# ------------------------------------------------------------------ Certificate B max-p rule (plan section 6)
def cert_b_pvalue(mu_err, late, n, alpha, gamma):
    """p = max(p_error, p_late): a valid p-value for 'at least one promise is broken', so delta is not split."""
    return np.maximum(hb_pvalue(mu_err, n, alpha), hb_pvalue(late, n, gamma))


# ------------------------------------------------------------------ hand p-values, exact arithmetic (Part A (h))
HAND_CASES = [  # (mu_hat as a fraction, n, alpha as a fraction, which term is expected to be the minimum)
    (Fraction(0), 784, Fraction(1, 100), "hoeffding"),           # zero errors: (1 - alpha)^n
    (Fraction(6, 1000), 784, Fraction(1, 100), "hoeffding"),
    (Fraction(8, 100), 784, Fraction(1, 10), "bentkus"),
]


def exact_hb(mu, n, alpha, digits=60):
    """Hoeffding-Bentkus p-value without scipy: Decimal for exp/log, Fraction for the binomial tail."""
    getcontext().prec = digits
    m, a = min(mu, alpha), alpha
    D = lambda f: Decimal(f.numerator) / Decimal(f.denominator)
    if m == 0:
        hoeff = D((1 - a) ** n)
    else:
        kl = D(m) * (D(m) / D(a)).ln() + D(1 - m) * (D(1 - m) / D(1 - a)).ln()
        hoeff = (-Decimal(n) * kl).exp()
    k = math.ceil(n * mu)                                               # exact: mu is a Fraction
    tail = sum(Fraction(math.comb(n, i)) * a ** i * (1 - a) ** (n - i) for i in range(0, k + 1))
    bentkus = Decimal(1).exp() * D(tail)
    p = min(Decimal(1), hoeff, bentkus)
    return float(p), ("hoeffding" if hoeff <= bentkus else "bentkus")


def check_hand():
    rows = []
    for mu, n, a, expect in HAND_CASES:
        ref, which = exact_hb(mu, n, a)
        got = float(hb_pvalue(float(mu), n, float(a)))
        rows.append({"mu_hat": float(mu), "n": n, "alpha": float(a), "exact": ref, "engine": got,
                     "rel_err": abs(got - ref) / ref, "min_term": which, "expected_min_term": expect})
    ok = all(r["rel_err"] < 1e-9 and r["min_term"] == r["expected_min_term"] for r in rows)
    return {"pass": bool(ok), "cases": rows}


def check_cert_b(n=784, alpha=0.01, gamma=0.10, delta=0.1):
    """Synthetic case: error promise holds (zero errors, 95% committed) in both cases; lateness broken (late 20%) in one,
    kept (late 3%) in the other. Max-p must refuse the first and certify the second."""
    mu_err = alpha * (1 - 0.95) + 0.0
    broken = float(cert_b_pvalue(mu_err, 0.20, n, alpha, gamma))
    kept = float(cert_b_pvalue(mu_err, 0.03, n, alpha, gamma))
    p_err = float(hb_pvalue(mu_err, n, alpha))
    return {"pass": bool(p_err <= delta and broken > delta and kept <= delta), "p_error": p_err,
            "p_max_lateness_broken": broken, "p_max_both_kept": kept}
