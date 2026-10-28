import pickle
import numpy as np
from features_v2 import features_v2
def pair_inputs(e,f):
    n=len(e); diff=e[:,:,None,:]-e[:,None,:,:]; ctx=np.broadcast_to(e.mean(1)[:,None,None,:],diff.shape)
    fd=f[:,:,None,:]-f[:,None,:,:]; fm=(f[:,:,None,:]+f[:,None,:,:])*.5
    return np.concatenate([diff,ctx,fd,fm],3).astype(np.float32)
def raw_scores(x,b):
    x=np.clip((x-b['mean'])/b['scale'],-10,10).astype(np.float32)
    return sum((np.maximum(x@m['w1']+m['b1'],0)@m['w2']+m['b2'])[...,0] for m in b['models'])/len(b['models'])
def aggregate(raw,valid):
    raw=.5*(raw-raw.transpose(0,2,1)); rivals=valid[:,None,:]*(1-np.eye(4)[None,:,:])
    return (raw*rivals).sum(2)/np.maximum(rivals.sum(2),1)
def scores(bank,arrays,b):
    e=arrays['embeddings']; f,v,_=features_v2(bank,e); return aggregate(raw_scores(pair_inputs(e,f),b),v),v
def predict(bank,arrays,checkpoint):
    b=pickle.loads((checkpoint/'model.pkl').read_bytes()); s,v=scores(bank,arrays,b); return np.where(v,s,-1e20).argmax(1).tolist()
