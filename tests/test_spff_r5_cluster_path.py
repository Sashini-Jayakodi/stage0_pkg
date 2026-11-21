"""Independent enumeration tests for the exact joint cluster-count reference."""
from itertools import product
import numpy as np
import pytest
from scipy.stats import binom

from src.spff_r5_cluster_path import (
    exact_cluster_path, selected_threshold, validate, excess_per_threshold,
)


def oracle_select(counts, alpha, delta, start):
    """Independent final selection using vectorized SciPy CDF, no implementation imports."""
    counts = np.asarray(counts)
    p = np.where(counts[:, 0] > 0,
                 binom.cdf(counts[:, 1], np.maximum(counts[:, 0], 1), alpha), 1)
    chosen = -1
    for j in range(start, -1, -1):
        if p[j] > delta:
            break
        chosen = j
    return chosen


def oracle_distribution(x, q, t, n, alpha, delta, start):
    g = x.shape[1]
    selection = np.zeros(g+1)
    for sequence in product(range(len(q)), repeat=n):
        counts = x[list(sequence)].sum(axis=0) if n else np.zeros((g,2),dtype=int)
        j = oracle_select(counts, alpha, delta, start)
        selection[j+1] += float(np.prod([q[h] for h in sequence]))
    w, a = q@x[:,:,1], q@x[:,:,0]
    ex = np.maximum(w-alpha*a,0)/(q@t)
    sev = float(selection[1:]@ex)
    failure = float(selection[1:]@((a>0)&(w>alpha*a)))
    return selection,sev,failure


@pytest.mark.parametrize("n", [0,1,2,3,4])
@pytest.mark.parametrize("start", [-1,0,1,2])
def test_general_cluster_count_matches_independent_bruteforce(n,start):
    # Nonmonotone wrong trajectory permitted when nested=False.
    x = np.array([
        [[3,0],[2,0],[1,0]],
        [[2,2],[1,1],[1,0]],
        [[4,1],[2,0],[1,1]],
        [[3,2],[2,1],[0,0]],
    ],dtype=int)
    q=np.array([.2,.3,.1,.4]); t=np.array([4,4,5,4])
    got=exact_cluster_path(x,q,t,n,.30,.30,start=start)
    p,s,f=oracle_distribution(x,q,t,n,.30,.30,start)
    assert np.allclose(got['selected_probabilities'],p,atol=2e-13)
    assert abs(got['severity_per_frame']-s)<2e-13
    assert abs(got['failure_probability']-f)<2e-13
    assert abs(sum(p)-1)<2e-13


def test_nested_frame_count_nonidentifiability_counterexample():
    patterns=[(0,0),(1,0),(1,1),(2,0),(2,1),(3,1)]
    x=np.array([[[3,w0],[1,w1]] for w0,w1 in patterns])
    t=np.full(len(x),3)
    qa=np.array([.44,.02,.24,.25,0,.05])
    qb=np.array([.44,.26,0,.01,.24,.05])
    a=exact_cluster_path(x,qa,t,7,.3,.1,nested=True)
    b=exact_cluster_path(x,qb,t,7,.3,.1,nested=True)
    assert np.allclose(a['accepted_per_cluster'], b['accepted_per_cluster'])
    assert np.allclose(a['wrong_per_cluster'], b['wrong_per_cluster'])
    assert np.allclose(a['failure_probability'], .02051947628544, atol=1e-14)
    assert np.allclose(b['failure_probability'], .06517639096320, atol=1e-14)
    assert np.allclose(a['severity_per_frame'], .0000683982542848, atol=1e-14)
    assert np.allclose(b['severity_per_frame'], .000217254636544, atol=1e-14)
    assert a['failure_probability'] != b['failure_probability']


def test_variable_cluster_size_normalization_and_zero_accepted():
    x=np.array([[[3,1],[0,0]],[[2,0],[1,0]],[[0,0],[0,0]]])
    q=np.array([.3,.5,.2]); t=np.array([4,2,3])
    got=exact_cluster_path(x,q,t,2,.2,.2,nested=True)
    _,a,w,ef=excess_per_threshold(x,q,t,.2,nested=True)
    assert np.allclose(got['accepted_per_cluster'],a)
    assert np.allclose(got['wrong_per_cluster'],w)
    assert np.isclose(got['mean_frames_per_cluster'],ef)
    zero=exact_cluster_path(np.zeros((1,2,2),dtype=int),[1.],[3],4,.3,.1)
    assert np.allclose(zero['selected_probabilities'], [1,0,0])
    assert zero['failure_probability']==zero['severity_per_frame']==0


def test_reject_invalid_and_unrealizable_nested_counts():
    # Loose (3 accepted, 2 wrong) => strict (1 accepted, 0 wrong)
    # is impossible because the two removed frames cannot contain 2 wrong
    # AND 1 additional correct relative to the loose count.
    with pytest.raises(ValueError,match='nonincreasing correct'):
        validate(np.array([[[3,2],[2,0]]]), [1.], [3], nested=True)
    validate(np.array([[[3,2],[2,1]]]), [1.], [3], nested=True)


def test_reject_fractions_negatives_and_counts_over_T():
    with pytest.raises(ValueError): validate([[[2.9,1.0]]],[1.],[3])
    with pytest.raises(ValueError): validate([[[-1,0]]],[1.],[3])
    with pytest.raises(ValueError): validate([[[4,1]]],[1.],[3])
    with pytest.raises(ValueError): validate([[[2,3]]],[1.],[3])
    with pytest.raises(ValueError): validate([[[2,1]]],[1.],[2.5])
    with pytest.raises(ValueError): validate([[[2,1]]],[.9],[3])
    with pytest.raises(ValueError): exact_cluster_path([[[2,1]]],[1.],[3],2.0,.3)
    with pytest.raises(ValueError): exact_cluster_path([[[2,1]]],[1.],[3],2,.3,max_states=0)
    with pytest.raises(RuntimeError,match='state limit'):
        exact_cluster_path([[[2,0]],[[2,1]]],[.5,.5],[2,2],5,.3,max_states=1)


def test_stable_tail_and_no_start():
    assert selected_threshold([[1100,400]],.3,.999998)==0
    assert selected_threshold([[1100,400]],.3,.999998,start=-1)==-1
    assert selected_threshold([[0,0]],.3,.1)==-1


def test_reject_float_integer_at_int64_boundary_without_wrapping():
    with pytest.raises(ValueError, match="outside int64"):
        validate([[[float(2**63), 0.0]]], [1.], [float(2**63)])
    with pytest.raises(ValueError, match="outside int64"):
        selected_threshold([[float(2**63), 0.0]], .3)


def test_detect_aggregate_int64_overflow_before_convolution():
    # Each cluster fits int64, but n clusters cannot fit into SciPy's
    # int64 count interface. Reject instead of silently wrapping totals.
    high = 2**62
    with pytest.raises(ValueError, match="aggregate count"):
        exact_cluster_path([[[high, 0]]], [1.], [high], 3, .3, .1)


def test_start_minus_one_short_circuits_even_with_tiny_state_budget():
    x = [[[3, 0]], [[3, 1]]]
    got = exact_cluster_path(x, [.5, .5], [3,3], 200, .3, .1,
                             start=-1, max_states=1)
    assert got['selected_probabilities'].tolist() == [1., 0.]
    assert got['severity_per_frame'] == got['failure_probability'] == 0.
    assert got['reachable_states'] == 1
