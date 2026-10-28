"""Inference does not read references or retain training-example labels."""
import pickle
import numpy as np
from selector_context import build
def infer(raw,q,d,b):
    x=np.clip((raw-b['mean'])/b['scale'],-10,10).astype(np.float32); q=np.clip((q-b['qmean'])/b['qscale'],-10,10).astype(np.float32); d=np.clip(d/b['dscale'],-10,10).astype(np.float32); result=np.zeros(raw.shape[:2])
    for m in b['models']:
        h=np.maximum(x@m['hidden.weight'].T+m['hidden.bias'],0); context=np.maximum(q@m['query.weight'].T+m['query.bias'],0)[:,None,:]; candidate=np.tanh(d@m['candidate.weight'].T+m['candidate.bias']); z=np.concatenate([h,context*candidate],2); result+=(z@m['out.weight'].T+m['out.bias']).squeeze(-1)/len(b['models'])
    return result
def scores(bank,arrays,b):
    raw,q,d,v=build(bank,arrays['embeddings']); return infer(raw,q,d,b),v
def predict(bank,arrays,checkpoint):
    b=pickle.loads((checkpoint/'model.pkl').read_bytes()); s,v=scores(bank,arrays,b); return np.where(v,s,-1e20).argmax(1).tolist()
