from spff_r5_fft_path_experiment import oracle,qa,qb,patterns
import numpy as np
from scipy.stats import binom
rng=np.random.default_rng(20261119)
n=784;rep=350000
for name,q in [('A',qa),('B',qb)]:
    r=oracle(q,n)
    counts=rng.multinomial(n,q,size=rep)
    ww=counts@patterns
    strict=ww[:,1]<=r['ks'];loose=ww[:,0]<=r['kl']
    obs=np.array([np.mean(~strict),np.mean(strict&loose),np.mean(strict&~loose)])
    ideal=np.array(r['selection']); sd=np.sqrt(ideal*(1-ideal)/rep)
    print(name,'fft',ideal,'MC',obs,'z',np.round((obs-ideal)/sd,3),'maxabs',np.max(np.abs(obs-ideal)))
