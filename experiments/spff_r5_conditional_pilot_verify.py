"""Independent analytic checks for the restricted conditional-pilot prototype."""
import numpy as np
from scipy.stats import binom
from spff_r5_conditional_pilot_prototype import full_path,ci_t_conditional,weights


def check_derivative():
    for n in (7,120,784):
        for t in (.1,.4,.7,.9):
            h=1e-4
            analytic=full_path(t,n,return_derivative=True)['analytic_derivative']
            numerical=(full_path(t+h,n)['loose_unsafe']-full_path(t-h,n)['loose_unsafe'])/(2*h)
            assert abs(analytic-numerical)<1e-7,(n,t,analytic,numerical)
    print('PASS derivative finite-difference comparisons (12 settings)')


def check_conditional_coverage():
    # Exact unconditional CP coverage evaluated via binomial sum conditional on N;
    # for any N the coverage is >= 1 - eta, including at endpoints t=0,1.
    for n in (1,2,5,15,63,261):
        for t in np.linspace(0,1,21):
            cover=0.
            for k in range(n+1):
                counts=np.zeros(6,dtype=int);counts[4]=k;counts[2]=n-k
                low,high=ci_t_conditional(counts)
                if low-1e-12<=t<=high+1e-12:
                    cover+=binom.pmf(k,n,t)
            assert cover>=.95-1e-12,(n,t,cover)
    print('PASS exact Clopper-Pearson conditional coverage (126 settings)')


def check_fixed_marginals():
    marginal_lo=[];marginal_st=[]
    for t in np.linspace(0,1,11):
        w=weights(float(t))
        assert np.isclose(w.sum(),1)
        marginal_lo.append(np.bincount([0,1,1,2,2,3],weights=w,minlength=4))
        marginal_st.append(np.bincount([0,0,1,0,1,1],weights=w,minlength=2))
    assert np.max(np.abs(np.array(marginal_lo)-marginal_lo[0])) < 1e-12
    assert np.max(np.abs(np.array(marginal_st)-marginal_st[0])) < 1e-12
    print('PASS both threshold-wise marginal distributions constant')


if __name__=='__main__':
    check_fixed_marginals()
    check_derivative()
    check_conditional_coverage()
