"""Answer-equivalence groups: pooled evidence with one representative per answer."""
import numpy as np
from common import parse_answer
from features_v2 import features_v2
def group_features(bank,embeddings):
    f,valid,_=features_v2(bank,embeddings); out=[]; reps=[]
    for row,e,ff in zip(bank,embeddings,f):
        aa=[parse_answer(c['text']) for c in row['candidates']]
        means=[]; pooled=[]; mask=[]
        for i,a in enumerate(aa):
            js=[j for j,b in enumerate(aa) if b==a] if a is not None else [i]
            means.append(e[js].mean(0))
            pooled.append(np.concatenate([ff[js].mean(0),ff[js].min(0),ff[js].max(0)]))
            mask.append(a is not None and aa.index(a)==i)
        means=np.array(means); pooled=np.array(pooled); mask=np.array(mask)
        context=means[mask].mean(0) if mask.any() else means.mean(0)
        out.append(np.concatenate([means,means-context,pooled],1)); reps.append(mask)
    return np.asarray(out,np.float32),np.asarray(reps),valid
