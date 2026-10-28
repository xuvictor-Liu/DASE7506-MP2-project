"""NumPy CPU inference for the seventh-round development comparisons."""
import pickle
import numpy as np
from selector_semantic_blend import scores as development_scores,norm,documents
from selector_refined import scores as v2_scores
from selector_v2 import scores as neural_scores_full,neural_scores
from features_v2 import features_v2
def auxiliary_scores(bank,arrays,model,kind):
    if kind=='robust': return neural_scores_full(bank,arrays,model)
    e=arrays['embeddings']; f,v,_=features_v2(bank,e); p=model['svd'].transform(model['vector'].transform(documents(bank))).reshape(-1,4,64).astype(np.float32)
    x=np.concatenate([e,e-e.mean(1,keepdims=True),f,p,p-p.mean(1,keepdims=True)],2)
    return neural_scores(x,model['net']),v
def scores(bank,arrays,b):
    a,v=auxiliary_scores(bank,arrays,b['auxiliary'],b['auxiliary_kind'])
    if b['weight']==1: return a,v
    base,_=(development_scores if b['base_kind']=='development' else v2_scores)(bank,arrays,b['base']); w=b['weight']
    return (1-w)*norm(base,v)+w*norm(a,v),v
def predict(bank,arrays,checkpoint):
    b=pickle.loads((checkpoint/'model.pkl').read_bytes()); s,v=scores(bank,arrays,b)
    return np.where(v,s,-1e20).argmax(1).tolist()
