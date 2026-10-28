"""Frozen offline selector; never opens references."""
import pickle
import numpy as np
from features import features,choose
def predict(bank,arrays,checkpoint):
    with (checkpoint/'model.pkl').open('rb') as f: bundle=pickle.load(f)
    ft,valid,_=features(bank)
    def score(b):
        if b['kind']=='blend': return (1-b['weight'])*score(b['base'])+b['weight']*score(b['neural'])
        if b['kind']=='neural':
            e=arrays['embeddings']; x=np.concatenate([e,e-e.mean(1,keepdims=True),ft],2)
            x=np.clip((x-b['mean'])/b['scale'],-10,10).astype(np.float32)
            return sum((np.maximum(x @ m['w1']+m['b1'],0) @ m['w2']+m['b2']).reshape(-1,4) for m in b['models'])/len(b['models'])
        if b['kind']=='ensemble': return b['weight']*score(b['linear'])+(1-b['weight'])*score(b['tree'])
        if b['kind']=='tree': return b['model'].predict(ft.reshape(-1,ft.shape[-1]),prediction_type='RawFormulaVal',thread_count=4).reshape(-1,4)
        e=arrays['embeddings']; x=np.concatenate([e,e-e.mean(1,keepdims=True)],2)
        if b['mode']=='combined': x=np.concatenate([x,ft],2)
        return b['model'].decision_function(b['scaler'].transform(x.reshape(-1,x.shape[-1]))).reshape(-1,4)
    return choose(score(bundle),valid).tolist()
