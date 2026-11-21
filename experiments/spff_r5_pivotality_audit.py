import numpy as np
from scipy.stats import binom
from scipy.special import gammaln
from itertools import combinations_with_replacement
from math import factorial

# Ordered (W_loose, W_strict). All paired counts nested, C=(3,1), T=3.
patterns=np.array([[0,0],[1,0],[1,1],[2,0],[2,1],[3,1]],dtype=np.int64)
qA=np.array([.44,.02,.24,.25,0.,.05]);qB=np.array([.44,.26,0.,.01,.24,.05])

def comp(n,k):
 if k==1:
  yield (n,);return
 for v in range(n+1):
  for rest in comp(n-v,k-1):yield (v,)+rest

def select(ww,n):
 # fixed acceptance n at strict, 3*n at loose
 if binom.cdf(ww[1],n,.3)>.1: return -1
 if binom.cdf(ww[0],3*n,.3)>.1: return 1
 return 0

def audit(q,n):
 comps=np.array(list(comp(n-1,len(q))), dtype=int)
 logp=gammaln(n)-np.sum(gammaln(comps+1),axis=1)
 for h in range(len(q)):
  if q[h]==0: logp=np.where(comps[:,h]==0,logp,-np.inf)
  else:logp=logp+comps[:,h]*np.log(q[h])
 weights=np.exp(logp)
 pivotal_any=0.;pivot_pair=0.; total=0.; per=[]
 for v,w in zip(comps,weights):
  if w<1e-18:continue
  base=v@patterns
  selections=np.array([select(base+x,n) for x in patterns])
  pr=np.bincount(selections+1,weights=q,minlength=3)
  disagreement=1-np.sum(pr**2)
  pivot_pair+=w*disagreement
  pivotal_any+=w*(np.max(selections)!=np.min(selections))
  total+=w
 return {'n':n,'average_replacement_selection_change':pivot_pair,'worst_case_possible_pivotal_context_probability':pivotal_any,'total_probability':total}
if __name__=='__main__':
 for n in [7,12,20]:
  for name,q in [('A',qA),('B',qB)]:
   print(name,audit(q,n))
