import pickle
import numpy as np
from common import parse_answer
from features import choose
from features_v2 import features_v2
from selector_v2 import neural_scores
def meta_features(bank,features,s):
    p=np.exp(s-s.max(1,keepdims=True)); p/=p.sum(1,keepdims=True)
    evidence=np.zeros_like(s)
    for i,b in enumerate(bank):
        aa=[parse_answer(c['text']) for c in b['candidates']]
        for j,a in enumerate(aa): evidence[i,j]=sum(p[i,k] for k in range(4) if a is not None and aa[k]==a)
    return np.concatenate([features,s[:,:,None],(s-s.mean(1,keepdims=True))[:,:,None],(s-s.max(1,keepdims=True))[:,:,None],p[:,:,None],evidence[:,:,None]],2)
def scores(bank,arrays,b):
    e=arrays['embeddings']; f,v,_=features_v2(bank,e); x=np.concatenate([e,e-e.mean(1,keepdims=True),f],2)
    s=sum(neural_scores(x,m) for m in b['models'])/len(b['models'])
    if b['meta'] is not None:
        ft=meta_features(bank,f,s); ms=b['meta'].predict(ft.reshape(-1,ft.shape[-1]),prediction_type='RawFormulaVal',thread_count=4).reshape(-1,4)
        s=(1-b['weight'])*s+b['weight']*ms
    return s,v
def predict(bank,arrays,checkpoint):
    b=pickle.loads((checkpoint/'model.pkl').read_bytes()); s,v=scores(bank,arrays,b); return choose(s,v).tolist()
