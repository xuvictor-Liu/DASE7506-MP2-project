import pickle
import numpy as np
from scipy.sparse import hstack,csr_matrix
from features_v2 import features_v2
from selector_v2 import neural_scores
from selector_semantic_blend import norm,documents
def scores(bank,arrays,b):
    e=arrays['embeddings']; f,v,_=features_v2(bank,e); raw=np.concatenate([e,e-e.mean(1,keepdims=True),f],2)
    prior=sum(neural_scores(raw,m) for m in b['teachers'])/len(b['teachers']); docs=documents(bank); x=hstack([b['vector'].transform(docs),csr_matrix(norm(prior,v).reshape(-1,1))],format='csr'); s=b['model'].decision_function(x).reshape(-1,4)
    if b.get('semantic') is not None:
        m=b['semantic']; p=m['svd'].transform(m['vector'].transform(docs)).reshape(-1,4,64).astype(np.float32); xx=np.concatenate([raw,p,p-p.mean(1,keepdims=True)],2); other=neural_scores(xx,m['net']); w=b['weight']; s=(1-w)*norm(s,v)+w*norm(other,v)
    return s,v
def predict(bank,arrays,checkpoint):
    b=pickle.loads((checkpoint/'model.pkl').read_bytes()); s,v=scores(bank,arrays,b); return np.where(v,s,-1e20).argmax(1).tolist()
