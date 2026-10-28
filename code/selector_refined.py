import pickle
import numpy as np
from features import features,choose
from features_v2 import equations
from common import parse_answer
from selector_v2 import scores as v2_scores
def scores(bank,arrays,b):
    cfg=b['config']; s=np.zeros((len(bank),4))
    if b['old'] is not None:
        old=b['old']; e=arrays['embeddings']; f,_,_=features(bank)
        x=np.concatenate([e,e-e.mean(1,keepdims=True),f],2); x=np.clip((x-old['mean'])/old['scale'],-10,10)
        s+=(1-cfg['new_weight'])*sum((np.maximum(x@m['w1']+m['b1'],0)@m['w2']+m['b2']).reshape(-1,4) for m in old['models'])/len(old['models'])
    if b['new'] is not None: s+=cfg['new_weight']*v2_scores(bank,arrays,b['new'])[0]
    if b['text'] is not None:
        from scipy.sparse import hstack
        t=[c['text'] for row in bank for c in row['candidates']]; tm=b['text']
        x=hstack([tm['word'].transform(t),tm['char'].transform(t)],format='csr')
        s+=cfg['text_weight']*tm['model'].decision_function(x).reshape(-1,4)
    if cfg['penalty']: s-=cfg['penalty']*np.array([[equations(c['text'])[0][1]>0 for c in row['candidates']] for row in bank])
    if cfg.get('correction',0):
        from correction_evidence import evidence
        s+=cfg['correction']*evidence(bank)
    aa=[[parse_answer(c['text']) for c in row['candidates']] for row in bank]
    if cfg['aggregate']:
        raw=s.copy()
        for i,answers in enumerate(aa):
            for j,a in enumerate(answers):
                if a is not None:
                    v=raw[i,[k for k in range(4) if answers[k]==a]]; m=v.max(); s[i,j]=m+np.log(np.exp(v-m).sum())
    if b.get('stack') is not None:
        from selector_stack import scores as stack_scores
        w=cfg['stack_weight']; s=(1-w)*s+w*stack_scores(bank,arrays,b['stack'])[0]
    return s,np.array([[a is not None for a in ans] for ans in aa])
def predict(bank,arrays,checkpoint):
    b=pickle.loads((checkpoint/'model.pkl').read_bytes()); s,v=scores(bank,arrays,b); return choose(s,v).tolist()
