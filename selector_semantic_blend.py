"""Experimental offline blend of role text, semantic features and frozen V2."""
import pickle
import numpy as np
from scipy.sparse import hstack,csr_matrix
from local_text import tokens
from features_v2 import features_v2
from selector_refined import scores as base_scores
from selector_v2 import neural_scores
def documents(bank):
    result=[]
    for row in bank:
        for c in row['candidates']:
            q,t=tokens(row['question'],c['text']); qs=set(q); shared=['shared_'+w for w in t[-100:] if w in qs]
            result.append(' '.join(q+['solutionstart']+t+['overlapstart']+shared))
    return result
def norm(s,v):
    mean=np.where(v,s,0).sum(1,keepdims=True)/v.sum(1,keepdims=True); z=s-mean; std=np.sqrt(np.where(v,z*z,0).sum(1,keepdims=True)/v.sum(1,keepdims=True))
    return z/np.maximum(std,.5)
def scores(bank,arrays,b):
    role=b['role']; semantic=b['semantic']; s0,v=base_scores(bank,arrays,role['base']); docs=documents(bank)
    xr=hstack([role['vector'].transform(docs),csr_matrix(s0.reshape(-1,1))],format='csr'); sr=role['model'].decision_function(xr).reshape(-1,4)
    e=arrays['embeddings']; f,_,_=features_v2(bank,e); p=semantic['svd'].transform(semantic['vector'].transform(docs)).reshape(-1,4,64).astype(np.float32); x=np.concatenate([e,e-e.mean(1,keepdims=True),f,p,p-p.mean(1,keepdims=True)],2); ss=neural_scores(x,semantic['net']); w=b['semantic_weight']
    return (1-w)*norm(sr,v)+w*norm(ss,v),v
def predict(bank,arrays,checkpoint):
    b=pickle.loads((checkpoint/'model.pkl').read_bytes()); s,v=scores(bank,arrays,b)
    return np.where(v,s,-1e20).argmax(1).tolist()
