"""Exploratory finite-pilot conditional inference for a FIXED-MARGINAL 2x2 switch family.

Requires scipy and numpy. This is not general SPFF-R5 inference.
"""
import numpy as np
from scipy.stats import beta, binom
from scipy.fft import rfft2, irfft2, next_fast_len
from time import perf_counter

PATTERNS=np.array([[0,0],[1,0],[1,1],[2,0],[2,1],[3,1]],dtype=int)

def weights(t):
    if not 0 <= t <= 1: raise ValueError('t must be in [0,1]')
    return np.array([.44, .02+.24*t, .24*(1-t), .25-.24*t, .24*t, .05])

def boundaries(n=784, alpha=.30, delta=.10):
    # Fixed n and accepted frames (3,1). Strict tested first.
    l=np.flatnonzero(binom.cdf(np.arange(3*n+1),3*n,alpha)<=delta)
    s=np.flatnonzero(binom.cdf(np.arange(n+1),n,alpha)<=delta)
    return (int(l[-1]) if len(l) else -1, int(s[-1]) if len(s) else -1)

def full_path(t,n=784,alpha=.30,delta=.10, return_derivative=False):
    q=weights(t)
    a,b=boundaries(n,alpha,delta)
    shape=(next_fast_len(3*n+1), next_fast_len(n+1))
    x=np.zeros(shape)
    for (u,v),w in zip(PATTERNS,q): x[u,v]+=w
    raw=irfft2(rfft2(x)**n,s=shape)[:3*n+1,:n+1]
    negmass=-float(raw[raw<0].sum())
    arr=np.maximum(raw,0)
    arr/=arr.sum()
    strict=float(arr[:,:b+1].sum()) if b>=0 else 0.
    loose=float(arr[:a+1,:b+1].sum()) if a>=0 and b>=0 else 0.
    output={'t':float(t), 'n':n,'loose_unsafe':loose,'strict_safe':strict-loose,'no_certificate':1-strict,'severity':loose/300, 'negative_mass':negmass,'loose_cutoff':a,'strict_cutoff':b}
    if return_derivative:
        if a<1 or b<0: d=0.
        else:
            y=np.zeros(shape)
            for (u,v),w in zip(PATTERNS,q): y[u,v]+=w
            z=irfft2(rfft2(y)**(n-1),s=shape)[:3*(n-1)+1,:n]
            d=.24*n*max(0,float(z[a-1,b]))
        output['analytic_derivative']=d
    return output

def ci_t_conditional(counts,eta=.05):
    """Exact conditional two-sided Clopper-Pearson interval for t.

    Conditional on N_2+N_4, count of pattern (2,1) has Bin(N,t) law.
    """
    counts=np.asarray(counts,dtype=int)
    assert counts.shape==(6,) and np.all(counts>=0)
    k=int(counts[4]); N=int(counts[2]+counts[4])
    if N==0: return 0.,1.
    low=0. if k==0 else float(beta.ppf(eta/2,k,N-k+1))
    high=1. if k==N else float(beta.ppf(1-eta/2,k+1,N-k))
    return low,high

def run(reps=40,m=261,n=784,seed=20261119):
    rng=np.random.default_rng(seed)
    truth={t:full_path(t,n) for t in (0.,.5,1.)}
    rows=[]
    start=perf_counter()
    for t in (0.,.5,1.):
        widths=[]; covers=[]; t_width=[]; min_counts=[]
        for _ in range(reps):
            c=rng.multinomial(m,weights(t))
            tl,tu=ci_t_conditional(c)
            lo=full_path(tl,n)['loose_unsafe'];hi=full_path(tu,n)['loose_unsafe']
            widths.append(max(0.,hi-lo)); covers.append(lo-1e-11<=truth[t]['loose_unsafe']<=hi+1e-11)
            t_width.append(tu-tl); min_counts.append(int(c[2]+c[4]))
        rows.append({'t':t,'m':m,'n':n,'reps':reps,'true_F':truth[t]['loose_unsafe'],'coverage_count':sum(covers),'mean_F_width':float(np.mean(widths)),'max_F_width':float(np.max(widths)),'mean_t_width':float(np.mean(t_width)),'mean_conditional_N':float(np.mean(min_counts))})
    return rows,perf_counter()-start

if __name__=='__main__':
    for t in [0,.25,.5,.75,1]:
        print('ORACLE',full_path(t,784,return_derivative=True))
    rows,secs=run()
    for row in rows: print('PILOT',row)
    print('SECONDS',secs)
