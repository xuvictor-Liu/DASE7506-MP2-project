"""Learned low-rank question-context/candidate interaction, NumPy inference."""
import pickle
import numpy as np
from features_v2 import features_v2
def build(bank,e):
    f,v,_=features_v2(bank,e); d=e-e.mean(1,keepdims=True)
    return np.concatenate([e,d,f],2),e.mean(1),d,v
def infer(raw,q,d,b):
    x=np.clip((raw-b['mean'])/b['scale'],-10,10).astype(np.float32); q=np.clip((q-b['qmean'])/b['qscale'],-10,10).astype(np.float32); d=np.clip(d/b['dscale'],-10,10).astype(np.float32); scores=np.zeros(raw.shape[:2])
    for m in b['models']:
        a=np.maximum(x@m['hidden.weight'].T+m['hidden.bias'],0)@m['out.weight'].T+m['out.bias']
        iq=np.tanh(q@m['query.weight'].T+m['query.bias'])[:,None,:]; ic=np.tanh(d@m['candidate.weight'].T+m['candidate.bias'])
        s=a+((iq*ic)@m['interaction.weight'].T); scores+=s.squeeze(-1)/len(b['models'])
    return scores
def scores(bank,arrays,b):
    raw,q,d,v=build(bank,arrays['embeddings']); return infer(raw,q,d,b),v
def predict(bank,arrays,checkpoint):
    b=pickle.loads((checkpoint/'model.pkl').read_bytes()); s,v=scores(bank,arrays,b); return np.where(v,s,-1e20).argmax(1).tolist()
