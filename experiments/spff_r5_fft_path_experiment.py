import numpy as np
from scipy.stats import binom
from scipy.fft import rfft2,irfft2,next_fast_len
from time import perf_counter
patterns=np.array([[0,0],[1,0],[1,1],[2,0],[2,1],[3,1]])
qa=np.array([.44,.02,.24,.25,0,.05]); qb=np.array([.44,.26,0,.01,.24,.05])
def oracle(q,n,alpha=.3,delta=.1):
    # P(z,w)^n coefficient calculation in padded Fourier domain: no aliasing
    shape=(next_fast_len(3*n+1),next_fast_len(n+1))
    p=np.zeros(shape)
    for (a,b),prob in zip(patterns,q):p[a,b]+=prob
    t=perf_counter()
    f=rfft2(p); out=irfft2(f**n,s=shape)[:3*n+1,:n+1]
    negative=float(-out[out<0].sum());out=np.maximum(out,0);out/=out.sum()
    kl=np.where(binom.cdf(np.arange(3*n+1),3*n,alpha)<=delta)[0]
    ks=np.where(binom.cdf(np.arange(n+1),n,alpha)<=delta)[0]
    a=int(kl[-1]) if len(kl) else -1
    b=int(ks[-1]) if len(ks) else -1
    ps=float(out[:,:b+1].sum()) if b>=0 else 0.
    plo=float(out[:a+1,:b+1].sum()) if a>=0 and b>=0 else 0.
    return {'n':n,'selection':(1-ps,plo,ps-plo),'negative_mass':negative,'normalization_error':float(abs(out.sum()-1)),'time_s':perf_counter()-t,'ks':b,'kl':a,'shape':shape}
if __name__=='__main__':
 for n in [7,20,120,784]:
  for name,q in [('A',qa),('B',qb)]: print(name,oracle(q,n))
