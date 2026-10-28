import pickle
import numpy as np
from features_v2 import features_v2
def build(bank,e,b):
    f,v,_=features_v2(bank,e); p=(e-b['pca_mean'])@b['pca_components'].T
    return np.concatenate([p,p-p.mean(1,keepdims=True),f],2),v
def kernel_scores(x,b):
    shape=x.shape[:2]; x=((x.reshape(-1,x.shape[-1])-b['mean'])/b['scale']).astype(np.float32); support=b['support']; support_norm=np.square(support).sum(1); out=[]
    for chunk in np.array_split(x,max(1,(len(x)+255)//256)):
        d=np.square(chunk).sum(1)[:,None]+support_norm[None,:]-2*chunk@support.T
        k=np.exp(-b['gamma']*np.maximum(d,0)); out.append(k@b['dual']+b['intercept'])
    return np.concatenate(out).reshape(shape)
def scores(bank,arrays,b):
    x,v=build(bank,arrays['embeddings'],b); return kernel_scores(x,b),v
def predict(bank,arrays,checkpoint):
    b=pickle.loads((checkpoint/'model.pkl').read_bytes()); s,v=scores(bank,arrays,b); return np.where(v,s,-1e20).argmax(1).tolist()
