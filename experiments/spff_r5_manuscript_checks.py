"""New verification of existing manuscript claims, not historical execution evidence.

The coverage grid is explicitly reconstructed as N=0..261 and t=0,.01,..,1.
Exact integer multinomial sums and ordered paths check the restricted model.
No experiment population or original implementation is changed.
"""
import argparse
from fractions import Fraction
from itertools import product
import hashlib
import json
from math import factorial, prod
from pathlib import Path
import platform
import sys
from time import perf_counter

import numpy as np
import scipy
from scipy.stats import beta, binom, norm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import r9_certify as frozen
from src.spff_r5_cluster_path import exact_cluster_path
from spff_r5_conditional_pilot_prototype import (
    PATTERNS, boundaries, ci_t_conditional, full_path, weights,
)


def compositions(n, d):
    if d == 1:
        yield (n,)
    else:
        for k in range(n + 1):
            for tail in compositions(n-k, d-1):
                yield (k,) + tail


def rational_coefficient(q, n, predicate):
    """Multinomial composition sums with integer masses and denominator 100**n."""
    numer = 0
    for counts in compositions(n, len(q)):
        loose = sum(c*int(p[0]) for c, p in zip(counts, PATTERNS))
        strict = sum(c*int(p[1]) for c, p in zip(counts, PATTERNS))
        if predicate(loose, strict):
            numer += (factorial(n) // prod(factorial(c) for c in counts)
                      * prod(w**c for w, c in zip(q, counts)))
    return Fraction(numer, 100**n)


def small_n():
    rows = []
    a, b = boundaries(7)
    for t in (0, .25, .5, .75, 1):
        q = [44, int(2+24*t), int(24*(1-t)), int(25-24*t), int(24*t), 5]
        f = rational_coefficient(q, 7, lambda l,s: l <= a and s <= b)
        d = Fraction(24*7, 100)*rational_coefficient(
            q, 6, lambda l,s: l == a-1 and s == b)
        fft = full_path(t, 7, return_derivative=True)
        row = dict(t=t, F_exact=str(f), F=float(f), derivative_exact=str(d),
                   derivative=float(d), F_fft_difference=abs(float(f)-fft['loose_unsafe']),
                   derivative_fft_difference=abs(float(d)-fft['analytic_derivative']))
        assert max(row['F_fft_difference'], row['derivative_fft_difference']) < 1e-14
        rows.append(row)
    endpoint_rows = []
    signatures = np.array([[[3,int(l)],[1,int(s)]] for l,s in PATTERNS])
    for t in (0, 1):
        q = [44, 2+24*t, 24*(1-t), 25-24*t, 24*t, 5]
        active = [i for i,w in enumerate(q) if w]
        masses = [0, 0, 0]
        decision_cache = {}
        count = mismatches = 0
        for sequence in product(active, repeat=7):
            l = sum(int(PATTERNS[i,0]) for i in sequence)
            s = sum(int(PATTERNS[i,1]) for i in sequence)
            if (l,s) not in decision_cache:
                p = frozen.binom_pvalue([l,s], [21,7], .3)
                decision_cache[l,s] = frozen.walk(p, 1, .1)
            chosen = decision_cache[l,s]
            independent = -1 if s > b else (0 if l <= a else 1)
            mismatches += chosen != independent
            masses[chosen+1] += prod(q[i] for i in sequence)
            count += 1
        assert count == 78125 and mismatches == 0 and sum(masses) == 100**7
        exact = [float(Fraction(m,100**7)) for m in masses]
        cluster = exact_cluster_path(signatures, np.array(q)/100, [3]*6,
                                     7, .3, .1, nested=True)
        fft = full_path(t,7)
        differences = np.abs(np.array(exact)-[fft['no_certificate'],fft['loose_unsafe'],fft['strict_safe']])
        assert max(differences) < 1e-14
        assert np.max(np.abs(np.array(exact)-cluster['selected_probabilities'])) < 1e-13
        endpoint_rows.append(dict(t=t, ordered_paths=count, selector_mismatches=mismatches,
                                  exact_selection=exact, fft_max_difference=float(max(differences)),
                                  cluster_max_difference=float(np.max(np.abs(np.array(exact)-cluster['selected_probabilities'])))))
    return dict(rational_grid=rows, ordered_endpoints=endpoint_rows,
                max_rational_fft_difference=max(max(r['F_fft_difference'],r['derivative_fft_difference']) for r in rows))


def coverage_grid():
    minimum = dict(coverage=1., N=None, t=None)
    cases = endpoint_checks = 0
    max_tail_error = 0.
    for n in range(262):
        k = np.arange(n+1)
        low, high = np.zeros(n+1), np.ones(n+1)
        if n:
            low[1:] = beta.ppf(.025, k[1:], n-k[1:]+1)
            high[:-1] = beta.ppf(.975, k[:-1]+1, n-k[:-1])
            max_tail_error = max(max_tail_error,
                float(np.max(np.abs(binom.sf(k[1:]-1,n,low[1:])-.025))),
                float(np.max(np.abs(binom.cdf(k[:-1],n,high[:-1])-.025))))
        for j in k:
            counts = np.zeros(6, dtype=int); counts[4]=j; counts[2]=n-j
            actual = ci_t_conditional(counts)
            assert max(abs(actual[0]-low[j]),abs(actual[1]-high[j])) < 1e-14
            endpoint_checks += 1
        for t in np.linspace(0,1,101):
            covered = (low <= t) & (t <= high)
            coverage = float(binom.pmf(k,n,t)[covered].sum())
            assert coverage >= .95-1e-12, (n,t,coverage)
            if coverage < minimum['coverage']:
                minimum = dict(coverage=coverage,N=n,t=float(t))
            cases += 1
    assert cases == 26462 and max_tail_error < 1e-12
    return dict(provenance='Reconstructed grid; original grid specification/log not recovered',
                N_range=[0,261],t_grid='numpy.linspace(0,1,101)',cases=cases,
                minimum=minimum,interval_endpoint_checks=endpoint_checks,
                max_tail_inversion_residual=max_tail_error)


def monte_carlo():
    rng = np.random.default_rng(20261119)
    rows = []
    draws = 350000
    z95 = norm.ppf(.975)
    for t in (0, 1):
        fft = full_path(t)
        counts = rng.multinomial(784,weights(t),size=draws)
        total = counts @ PATTERNS
        strict,loose = total[:,1] <= 218,total[:,0] <= 676
        observed = np.array([np.mean(~strict),np.mean(strict&loose),np.mean(strict&~loose)])
        expected = np.array([fft['no_certificate'],fft['loose_unsafe'],fft['strict_safe']])
        center = (observed+z95**2/(2*draws))/(1+z95**2/draws)
        half = z95*np.sqrt(observed*(1-observed)/draws+z95**2/(4*draws**2))/(1+z95**2/draws)
        assert np.all((center-half <= expected)&(expected <= center+half))
        z = (observed-expected)/np.sqrt(expected*(1-expected)/draws)
        rows.append(dict(t=t,draws=draws,observed=observed.tolist(),fft=expected.tolist(),
                         wilson_low=(center-half).tolist(),wilson_high=(center+half).tolist(),
                         standardized_discrepancy=z.tolist()))
    return dict(seed=20261119,order='t=0 then t=1, same RNG',rows=rows,
                max_abs_z=max(abs(z) for row in rows for z in row['standardized_discrepancy']))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Refusing to overwrite a prior verification record')
    start = perf_counter()
    results = dict(provenance='New manuscript verification; not recovered historical logs',
                   python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,
                   source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in [Path(__file__),ROOT/'src/r9_certify.py',ROOT/'src/certify.py',
                                            ROOT/'src/spff_r5_cluster_path.py',ROOT/'experiments/spff_r5_conditional_pilot_prototype.py']})
    results['small_n'] = small_n()
    results['conditional_binomial'] = coverage_grid()
    results['monte_carlo'] = monte_carlo()
    results['seconds'] = perf_counter()-start
    with args.output.open('x',encoding='utf-8') as f:
        json.dump(results,f,indent=2); f.write('\n')
    print(json.dumps(results,indent=2))


if __name__ == '__main__':
    main()
