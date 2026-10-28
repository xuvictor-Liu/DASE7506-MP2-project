import pickle
import numpy as np
from group_features import group_features
def model_scores(x,b):
    x=np.clip((x-b['mean'])/b['scale'],-8,8).astype(np.float32)
    return sum((np.maximum(x@m['w1']+m['b1'],0)@m['w2']).reshape(-1,4) for m in b['models'])/len(b['models'])
def scores(bank,arrays,b):
    x,reps,valid=group_features(bank,arrays['embeddings']); return model_scores(x,b),valid
def predict(bank,arrays,checkpoint):
    b=pickle.loads((checkpoint/'model.pkl').read_bytes()); s,v=scores(bank,arrays,b)
    return np.where(v,s,-1e20).argmax(1).tolist()
