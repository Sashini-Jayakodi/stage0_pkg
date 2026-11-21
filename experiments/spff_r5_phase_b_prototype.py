"""SPFF-R5 Phase B exploratory prototype; independent of historical repository files.

Requires numpy and scipy. Produces deterministic synthetic results; it is not a
finite-sample certificate for calibration sizes exceeding the independent pilot.
"""
from itertools import product
from math import sqrt, log
import numpy as np
from scipy.special import gammaln
from scipy.optimize import linprog
from scipy.stats import binom

SIGNATURES = np.array([
    [[0, 0], [1, 0], [1, 0]],
    [[1, 1], [1, 1], [1, 0]],
    [[1, 0], [0, 0], [0, 0]],
], dtype=int)
Q = np.array([.5, .2, .3])


def select_from_counts(counts, alpha, delta, start=None):
    """Counts: thresholds x (accepted, wrong), ordered loose-to-strict."""
    if start is None:
        start = len(counts) - 1
    j = -1
    for k in range(start, -1, -1):
        c, w = map(int, counts[k])
        p = float(binom.cdf(w, c, alpha)) if c else 1.
        if p > delta:
            break
        j = k
    return j


def compositions(n, k):
    if k == 1:
        yield (n,)
    else:
        for v in range(n + 1):
            for tail in compositions(n - v, k - 1):
                yield (v,) + tail


def compile_kernel(x, n, alpha, delta, start=None):
    comps = np.array(list(compositions(n, len(x))), dtype=int)
    chosen = np.array([select_from_counts(np.tensordot(v, x, axes=(0, 0)),
                                          alpha, delta, start) for v in comps])
    return comps, chosen


def pi_from_q(q, n, comps, chosen, g):
    logp = gammaln(n + 1) - np.sum(gammaln(comps + 1), axis=1)
    for h in range(len(q)):
        if q[h] == 0:
            logp = np.where(comps[:, h] == 0, logp, -np.inf)
        else:
            logp += comps[:, h] * np.log(q[h])
    weights = np.exp(logp)
    return np.bincount(chosen + 1, weights=weights, minlength=g + 1)


def pi_u_stat(pilot_counts, n, comps, chosen, g):
    """Complete U-statistic via the multivariate hypergeometric formula."""
    m = int(np.sum(pilot_counts))
    if m < n:
        raise ValueError('Pilot size must be at least calibration size')
    viable = np.all(comps <= pilot_counts, axis=1)
    b = comps[viable]
    logden = gammaln(m + 1) - gammaln(n + 1) - gammaln(m - n + 1)
    lognum = np.sum(gammaln(pilot_counts + 1) - gammaln(b + 1)
                    - gammaln(pilot_counts - b + 1), axis=1)
    pi = np.bincount(chosen[viable] + 1, weights=np.exp(lognum - logden),
                     minlength=g + 1)
    return pi / pi.sum()


def honest_u_stat_interval(pilot_counts, n, x, alpha, delta, comps, chosen,
                           miscoverage=.05):
    """Pilot-honest for fixed path and iid clusters, n <= m, no support assumption.

    Simultaneous Hoeffding U-statistic bounds for selection probabilities and
    Hoeffding mean bounds for excess contributions; independent box relaxation
    optimized using linear programming. For sign-level valid LTT only, intersect
    with the known global bound S <= delta*(1-alpha).
    """
    m = int(np.sum(pilot_counts)); g = x.shape[1]
    phat = pi_u_stat(pilot_counts, n, comps, chosen, g)
    qhat = pilot_counts / m
    d = x[:, :, 1] - alpha * x[:, :, 0]
    ghat = qhat @ d
    estimate = float(phat[1:] @ np.maximum(ghat, 0))
    eta_pi = eta_g = miscoverage / 2
    epsilon_pi = sqrt(log(2 * (g + 1) / eta_pi) / (2 * (m // n)))
    epsilon_g = sqrt(log(2 * g / eta_g) / (2 * m))
    pi_lo = np.maximum(0, phat - epsilon_pi)
    pi_hi = np.minimum(1, phat + epsilon_pi)
    sev_lo = np.r_[0, np.maximum(0, ghat - epsilon_g)]
    sev_hi = np.r_[0, np.minimum(1-alpha, np.maximum(0, ghat + epsilon_g))]
    bounds = list(zip(pi_lo, pi_hi))
    lo = linprog(sev_lo, A_eq=np.ones((1, g + 1)), b_eq=[1],
                 bounds=bounds, method='highs')
    hi = linprog(-sev_hi, A_eq=np.ones((1, g + 1)), b_eq=[1],
                 bounds=bounds, method='highs')
    if not (lo.success and hi.success):
        raise RuntimeError('Infeasible confidence-region optimization')
    return max(0., float(lo.fun)), min((1-alpha)*delta, -float(hi.fun)), estimate


def original_interval(pilot_counts, n, x, alpha, delta, comps, chosen,
                      miscoverage=.05):
    m = int(sum(pilot_counts)); k = len(x)
    qhat = pilot_counts / m
    phat = pi_from_q(qhat, n, comps, chosen, x.shape[1])
    d = x[:, :, 1] - alpha * x[:, :, 0]
    shat = float(phat[1:] @ np.maximum(qhat @ d, 0))
    eps = sqrt(log(2 * k / miscoverage) / (2 * m))
    half = min(1., (n + 1) * min(1., k * eps / 2))
    return max(0, shat-half), min(1., shat+half)


def u_stat_experiment(reps=100):
    alpha, delta, n = .3, .3, 8
    comps, chosen = compile_kernel(SIGNATURES, n, alpha, delta)
    pi = pi_from_q(Q, n, comps, chosen, SIGNATURES.shape[1])
    true_excess = np.maximum(Q @ (SIGNATURES[:,:,1] - alpha*SIGNATURES[:,:,0]), 0)
    truth = float(pi[1:] @ true_excess)
    rng = np.random.default_rng(20261119)
    rows = []
    for m in (200, 1000, 3000):
        novel_widths, old_widths, covered, errors = [], [], [], []
        for _ in range(reps):
            counts = rng.multinomial(m, Q)
            lo, hi, point = honest_u_stat_interval(counts, n, SIGNATURES,
                                                    alpha, delta, comps, chosen)
            o_lo, o_hi = original_interval(counts, n, SIGNATURES,
                                           alpha, delta, comps, chosen)
            novel_widths.append(hi-lo)
            old_widths.append(o_hi-o_lo)
            covered.append(lo <= truth <= hi)
            errors.append(abs(point-truth))
        rows.append({'pilot':m, 'new_mean_width':float(np.mean(novel_widths)),
                     'original_mean_width':float(np.mean(old_widths)),
                     'empirical_coverage':sum(covered), 'reps':reps,
                     'u_stat_point_mae':float(np.mean(errors))})
    return truth, rows


def exact_counterexample():
    """Same per-threshold count marginals, different cross-threshold couplings."""
    A = {(0,0):.44, (1,0):.02, (1,1):.24, (2,0):.25, (3,1):.05}
    B = {(0,0):.44, (1,0):.26, (2,0):.01, (2,1):.24, (3,1):.05}
    def path(d):
        pi = np.zeros(3)
        keys = list(d)
        for seq in product(keys, repeat=7):
            prob = float(np.prod([d[k] for k in seq]))
            counts = np.array([[21, sum(z[0] for z in seq)],
                               [7, sum(z[1] for z in seq)]])
            j = select_from_counts(counts, .3, .1)
            pi[j+1] += prob
        return pi
    pA, pB = path(A), path(B)
    # The loose threshold passes if total wrong <=3 (21 accepted frames).
    def formula(p0, r1, r2):
        return (p0**7 + 7*p0**6*(r1+r2) + 21*p0**5*r1**2
                + 42*p0**5*r1*r2 + 35*p0**4*r1**3)
    expectedA = formula(.44,.02,.25)
    expectedB = formula(.44,.26,.01)
    assert abs(pA[1]-expectedA)<1e-11 and abs(pB[1]-expectedB)<1e-11
    return pA, pB, ((.91-.3*3)/3) * pA[1], ((.91-.3*3)/3) * pB[1]


FRAME_PATTERNS = np.array([
    [[8, 0], [6, 0], [4, 0]],
    [[8, 1], [6, 0], [4, 0]],
    [[8, 1], [5, 1], [3, 0]],
    [[7, 2], [5, 1], [3, 0]],
    [[6, 2], [4, 2], [2, 1]],
    [[8, 3], [6, 1], [4, 1]],
], dtype=int)
FRAME_QS = np.array([
    [.20,.30,.22,.14,.08,.06],
    [.15,.25,.25,.17,.10,.08],
    [.25,.31,.20,.12,.07,.05],
    [.20,.24,.25,.16,.10,.05],
    [.15,.28,.22,.20,.08,.07],
    [.25,.28,.21,.10,.10,.06],
])
FRAME_ALPHAS = [.12,.12,.12,.16,.16,.16]


def select_many(total, alpha, delta=.1):
    a, w = total[:,:,0], total[:,:,1]
    pv = np.where(a > 0, binom.cdf(w, np.maximum(a, 1), alpha), 1.)
    passed = pv <= delta
    j = np.full(len(total), -1, dtype=int)
    alive = np.ones(len(total), dtype=bool)
    for k in range(a.shape[1]-1, -1, -1):
        alive &= passed[:,k]
        j[alive] = k
    return np.bincount(j+1, minlength=a.shape[1]+1)/len(j)


def forecast_path(q, n, alpha, draws, rng, kind='oracle'):
    if kind == 'oracle':
        clusters = rng.multinomial(n, q, size=draws)
        sums = np.einsum('bk,kij->bij', clusters, FRAME_PATTERNS)
    else:
        flat = FRAME_PATTERNS.reshape(len(q), -1)
        mu = q @ flat
        d = flat - mu
        covariance = (q[:,None] * d).T @ d
        gaussian = rng.multivariate_normal(n*mu, n*covariance, size=draws,
                                            check_valid='ignore', method='svd')
        sums = np.rint(gaussian).astype(int).reshape(draws, 3, 2)
        sums[:,:,0] = np.maximum(0, sums[:,:,0])
        sums[:,:,1] = np.clip(sums[:,:,1], 0, sums[:,:,0])
    return select_many(sums, alpha)


def gaussian_experiment():
    rows=[]
    for i,(q,alpha) in enumerate(zip(FRAME_QS, FRAME_ALPHAS)):
        truth = forecast_path(q, 120, alpha, 40000, np.random.default_rng(123+i))
        normal = forecast_path(q, 120, alpha, 40000, np.random.default_rng(456+i), 'gaussian')
        # Pilot-estimated results measure sensitivity, not finite-sample CI coverage.
        rng = np.random.default_rng(900+i)
        empirical_errors=[]
        for _ in range(30):
            pilot = rng.multinomial(261, q)/261
            pred = forecast_path(pilot, 120, alpha, 3000, rng, 'gaussian')
            empirical_errors.append(float(np.max(np.abs(truth-pred))))
        rows.append({'case':i+1,'alpha':alpha,
                     'max_oracle_gaussian_gap':float(np.max(np.abs(truth-normal))),
                     'mean_max_pilot_gaussian_gap':float(np.mean(empirical_errors)),
                     'worst_max_pilot_gaussian_gap':float(np.max(empirical_errors))})
    return rows

if __name__ == '__main__':
    truth, rows = u_stat_experiment()
    print('U-STAT SYNTHETIC TRUE SEVERITY',truth)
    for row in rows:print(row)
    a,b,sa,sb=exact_counterexample()
    print('COUNTEREXAMPLE A,B',a,b,'F:',a[1],b[1],'S:',sa,sb)
    print('GAUSSIAN EXPLORATION')
    for row in gaussian_experiment():print(row)
