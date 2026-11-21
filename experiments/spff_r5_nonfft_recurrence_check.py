"""New independent reproduction check; not the original archived recurrence.
Truncated nonnegative convolution computes only the joint-pass rectangle.
No FFT is used by the recurrence; the archived FFT is a separate comparator.
Floating-point agreement is not a rigorous error enclosure.
"""
import json
import math
import platform
import time
from pathlib import Path
import numpy as np
import scipy
from scipy.stats import binom
from spff_r5_conditional_pilot_prototype import full_path

PATTERNS = ((0,0),(1,0),(1,1),(2,0),(2,1),(3,1))

def recurrence(n, t, a, b):
    q = (.44,.02+.24*t,.24*(1-t),.25-.24*t,.24*t,.05)
    if a < 0 or b < 0:
        return np.zeros((0,0))
    p = np.zeros((a+1,b+1)); p[0,0]=1
    for k in range(n):
        r = np.zeros_like(p)
        for (l,s), w in zip(PATTERNS,q):
            if w and l <= a and s <= b:
                r[l:,s:] += w*p[:a+1-l,:b+1-s]
        if k == n-2: boundary = float(r[a-1,b]) if a>=1 else 0.
        p=r
    return float(p.sum()), .24*n*boundary

def cutoff(N):
    eligible = np.flatnonzero(binom.cdf(np.arange(N+1),N,.30)<=.10)
    return int(eligible[-1]) if len(eligible) else -1

def main():
    n=784; a,b=cutoff(3*n),cutoff(n)
    assert (a,b)==(676,218)
    rows=[]
    for t in (0.,.25,.5,.75,1.):
        start=time.perf_counter()
        f,d=recurrence(n,t,a,b)
        fft=full_path(t,n,return_derivative=True)
        row=dict(t=t,F_nonfft=f,F_fft=fft['loose_unsafe'],F_abs_difference=abs(f-fft['loose_unsafe']),derivative_nonfft=d,derivative_fft=fft['analytic_derivative'],derivative_abs_difference=abs(d-fft['analytic_derivative']),seconds=time.perf_counter()-start)
        assert row['F_abs_difference']<1e-11 and row['derivative_abs_difference']<1e-11, row
        rows.append(row); print(json.dumps(row),flush=True)
    width=rows[-1]['F_nonfft']-rows[0]['F_nonfft']
    report=dict(provenance='New reproduction check in assistant container; not original historical run',python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,n=n,alpha=.30,delta=.10,a=a,b=b,rows=rows,width=width,minimax_error=width/2,max_F_difference=max(r['F_abs_difference'] for r in rows),max_derivative_difference=max(r['derivative_abs_difference'] for r in rows))
    Path(__file__).with_name('nonfft_reproduction_results.json').write_text(json.dumps(report,indent=2)+'\n')
    print('PASS: independent recurrence agrees with archived FFT at all five coupling settings')

if __name__=='__main__': main()
