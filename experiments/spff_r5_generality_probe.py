"""Independent finite-support fixed-sequence audit generality probe.

This uses synthetic, physically nested 3-threshold variable-acceptance
signatures and exact integer composition enumeration. It does not use
frozen repository code or claim to reproduce the historical dataset.
"""
from __future__ import annotations
import math
from itertools import product
import numpy as np
from scipy.stats import binom

ALPHA = .30
DELTA = .10
# Three ordered thresholds, each signature = (C1,E1,C2,E2,C3,E3).
SIGNATURES = np.array([
    (1,0,2,0,3,0), (1,1,2,1,3,1), (1,0,2,1,3,1),
    (0,0,1,0,3,1), (0,0,2,1,2,1), (1,0,1,0,2,1),
    (0,0,0,0,1,1), (1,0,2,1,3,2),
], dtype=int)
Q = np.array([.30,.05,.05,.10,.10,.10,.20,.10], dtype=float)


def validate_nested(signatures):
    for x in signatures:
        c, e = x[::2], x[1::2]
        assert all(0 <= ej <= cj for cj, ej in zip(c, e))
        assert np.all(np.diff(c) >= 0)
        assert np.all(np.diff(e) >= 0)
        assert np.all(np.diff(c-e) >= 0)


def compositions(total, d):
    if d == 1:
        yield (total,)
        return
    for v in range(total+1):
        for rest in compositions(total-v, d-1):
            yield (v,) + rest


def selection_index(totals):
    selected = -1
    for j in range(3):
        c, e = int(totals[2*j]), int(totals[2*j+1])
        if c == 0 or binom.cdf(e, c, ALPHA) > DELTA:
            break
        selected = j
    return selected


def exact_path(q, n):
    risks_c = q @ SIGNATURES[:, ::2]
    risks_e = q @ SIGNATURES[:, 1::2]
    unsafe = risks_e > ALPHA * risks_c
    p = np.zeros(4)
    lgfact = math.lgamma(n + 1)
    for z in compositions(n, len(q)):
        if any(c > 0 and qk == 0 for c, qk in zip(z, q)):
            continue
        weight = math.exp(lgfact - sum(math.lgamma(c+1) for c in z)
                          + sum(c * math.log(qk) for c, qk in zip(z, q) if c))
        j = selection_index(np.asarray(z) @ SIGNATURES)
        p[j+1] += weight
    return p, float(p[1:][unsafe].sum()), risks_e / risks_c


def exact_single_cutoff(n):
    k = int(binom.ppf(DELTA, n, ALPHA))
    while binom.cdf(k, n, ALPHA) > DELTA:
        k -= 1
    while binom.cdf(k+1, n, ALPHA) <= DELTA:
        k += 1
    b = float(binom.cdf(k, n, ALPHA))
    return k, b, (1-2*.05)*b

if __name__ == '__main__':
    validate_nested(SIGNATURES)
    assert abs(Q.sum() - 1) < 1e-15
    print('Three-threshold variable-acceptance nested example:')
    for n in [7,10]:
        p, f, risks = exact_path(Q, n)
        print(' n=',n,'risk=',risks, 'no/j1/j2/j3=',p, 'unsafe-selection=', f,
              'probability-sum=',p.sum())
        assert abs(p.sum() - 1) < 3e-12
    print('Uniform 95% CI width lower bound near risk alpha, one test:')
    for n in [7,120,784]:
        k,b,lb = exact_single_cutoff(n)
        print(' n=',n,'k=',k,'boundary selection probability=',b,
              'minimum expected length lower bound=',lb)
