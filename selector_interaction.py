import pickle
import numpy as np
from features_v2 import features_v2
from numeric_features import relations
def build(bank,e,p):
    f,v,_=features_v2(bank,e); nf=relations(bank)
    center=e-e.mean(1,keepdims=True)
    q=(e.mean(1)-p['qmean'])@p['qcomp'].T
    d=(center-p['dmean'])@p['dcomp'].T
    raw=(e-p['rmean'])@p['rcomp'].T
    cross=(q[:,None,:32,None]*d[:,:,:32][:,:,None,:]).reshape(len(e),4,-1)
    return np.concatenate([raw,d,cross,f,nf],2),v
def predict(bank,arrays,checkpoint):
    b=pickle.loads((checkpoint/'model.pkl').read_bytes()); x,v=build(bank,arrays['embeddings'],b['projection']); x=(x-b['mean'])/b['scale']
    s=x@b['coef']+b['intercept']; return np.where(v,s,-1e20).argmax(1).tolist()
