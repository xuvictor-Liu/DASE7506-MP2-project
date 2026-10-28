import pickle
import numpy as np
from features_v2 import features_v2
from selector_v2 import neural_scores
from selector_semantic_blend import norm,scores as development_scores
from selector_refined import scores as v2_scores
def scores(bank,arrays,b):
    e=arrays['embeddings']; f,v,_=features_v2(bank,e); raw=np.concatenate([e,e-e.mean(1,keepdims=True),f],2)
    s=sum(neural_scores(raw,m) for m in b['members'])/len(b['members'])
    if b.get('base') is not None:
        bs,_=(development_scores if b['base_kind']=='development' else v2_scores)(bank,arrays,b['base']); w=b['weight']; s=(1-w)*norm(bs,v)+w*norm(s,v)
    return s,v
def predict(bank,arrays,checkpoint):
    b=pickle.loads((checkpoint/'model.pkl').read_bytes()); s,v=scores(bank,arrays,b)
    return np.where(v,s,-1e20).argmax(1).tolist()
