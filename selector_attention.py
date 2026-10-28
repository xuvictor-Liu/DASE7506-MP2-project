"""Small learned candidate-attention verifier, pure NumPy CPU inference."""
import pickle
import numpy as np
from features_v2 import features_v2
def build(bank,e):
    f,v,_=features_v2(bank,e)
    return np.concatenate([e,e-e.mean(1,keepdims=True),f],2),v
def forward(x,v,m):
    h=np.maximum(x@m['w1']+m['b1'],0)
    z=np.where(v,h@m['a'],-1e20); z=z-z.max(1,keepdims=True); att=np.exp(z); att/=att.sum(1,keepdims=True)
    context=(h*att[:,:,None]).sum(1,keepdims=True)
    return h@m['w2']+(h*context)@m['wc']
def scores(bank,arrays,b):
    x,v=build(bank,arrays['embeddings']); x=np.clip((x-b['mean'])/b['scale'],-10,10).astype(np.float32)
    return sum(forward(x,v,m) for m in b['models'])/len(b['models']),v
def predict(bank,arrays,checkpoint):
    b=pickle.loads((checkpoint/'model.pkl').read_bytes()); s,v=scores(bank,arrays,b)
    return np.where(v,s,-1e20).argmax(1).tolist()
