"""Second-round model interface, isolated from the original frozen version."""
import pickle
import numpy as np
from features_v2 import features_v2
from features import choose

def neural_scores(x,b):
    x=np.clip((x-b['mean'])/b['scale'],-10,10).astype(np.float32)
    return sum((np.maximum(x@m['w1']+m['b1'],0)@m['w2']+m['b2']).reshape(-1,4) for m in b['models'])/len(b['models'])

def scores(bank,arrays,b):
    e=arrays['embeddings']; f,valid,votes=features_v2(bank,e)
    if b['kind']=='neural_v2':
        p=b.get('projection')
        if p is None: x=np.concatenate([e,e-e.mean(1,keepdims=True),f],2)
        else: x=np.concatenate([(e-p['mean1'])@p['components1'].T,(e-e.mean(1,keepdims=True)-p['mean2'])@p['components2'].T,f],2)
        return neural_scores(x,b),valid
    if b['kind']=='tree_v2':
        ep=(e-b['mean'])@b['components'].T
        x=np.concatenate([ep,ep-ep.mean(1,keepdims=True),f],2)
        return b['model'].predict(x.reshape(-1,x.shape[-1]),prediction_type='RawFormulaVal',thread_count=4).reshape(-1,4),valid
    if b['kind']=='blend_v2':
        ss=[scores(bank,arrays,m)[0] for m in b['members']]
        return sum(w*s for w,s in zip(b['weights'],ss)),valid
    raise ValueError(b['kind'])

def predict(bank,arrays,checkpoint):
    b=pickle.loads((checkpoint/'model.pkl').read_bytes()); s,v=scores(bank,arrays,b)
    return choose(s,v).tolist()
