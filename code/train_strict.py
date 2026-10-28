"""Fit arithmetic calibration on training OOF scores, not in-sample predictions."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']: os.environ[k]='4'
import json,pickle
from pathlib import Path
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from train import load
from strict_arithmetic import features
from features_v2 import features_v2
from selector_v2 import neural_scores
from selector_refined import scores
from common import parse_answer
out=Path('runs_strict'); out.mkdir(exist_ok=True)
b,e,_,v,_,y=load('train'); ft=features(b); oldft,_,_=features_v2(b,e); raw=np.concatenate([e,e-e.mean(1,keepdims=True),oldft],2)
folds=pickle.loads(Path('runs_stack/model_bundle/model.pkl').read_bytes())['models']; oof=np.zeros_like(y,dtype=float)
for (_,ho),model in zip(KFold(5,shuffle=True,random_state=2026).split(b),folds): oof[ho]=neural_scores(raw[ho],model)
x=np.concatenate([oof[:,:,None],(oof-oof.mean(1,keepdims=True))[:,:,None],ft],2)
hard=np.array([len(set(parse_answer(c['text']) for c in row['candidates'])-{None})>=2 for row in b]); ids=np.flatnonzero(hard); np.random.default_rng(731).shuffle(ids); cut=int(.8*len(ids)); tr,ho=ids[:cut],ids[cut:]
records=[]; best=None
for C in [.01,.1,1]:
    use=v[tr].ravel(); scaler=StandardScaler().fit(x[tr].reshape(-1,x.shape[-1])[use]); model=LogisticRegression(C=C,max_iter=300).fit(scaler.transform(x[tr].reshape(-1,x.shape[-1])[use]),y[tr].ravel()[use]); s=model.decision_function(scaler.transform(x[ho].reshape(-1,x.shape[-1]))).reshape(-1,4); pred=np.where(v[ho],s,-1e20).argmax(1); acc=float(y[ho,pred].mean()*100)
    r=dict(C=C,holdout_accuracy=acc); records.append(r); print(json.dumps(r),flush=True)
    if best is None or acc>best['holdout_accuracy']: best=r
use=v[ids].ravel(); scaler=StandardScaler().fit(x[ids].reshape(-1,x.shape[-1])[use]); model=LogisticRegression(C=best['C'],max_iter=300).fit(scaler.transform(x[ids].reshape(-1,x.shape[-1])[use]),y[ids].ravel()[use])
bv,ev,_,vv,_,yv=load('validation'); fv=features(bv); base=pickle.loads(Path('runs_v2/final_bundle/model.pkl').read_bytes()); vs,_=scores(bv,{'embeddings':ev},base); xv=np.concatenate([vs[:,:,None],(vs-vs.mean(1,keepdims=True))[:,:,None],fv],2); s=model.decision_function(scaler.transform(xv.reshape(-1,xv.shape[-1]))).reshape(-1,4); pred=np.where(vv,s,-1e20).argmax(1); c=yv[np.arange(len(yv)),pred]; hv=np.array([len(set(parse_answer(cc['text']) for cc in r['candidates'])-{None})>=2 for r in bv])
result=dict(selected=best,hard_accuracy=float(c[hv].mean()*100),validation_accuracy=float(c.mean()*100)); print(json.dumps(result),flush=True)
bundle=dict(base=base,mean=scaler.mean_,scale=scaler.scale_,coef=model.coef_.ravel(),intercept=float(model.intercept_[0]))
(out/'model_bundle').mkdir(exist_ok=True); (out/'model_bundle/model.pkl').write_bytes(pickle.dumps(bundle,protocol=5)); (out/'selection.json').write_text(json.dumps(result,indent=2)); (out/'holdout_search.json').write_text(json.dumps(records,indent=2))
for label in [0,1]:
    m=(y==label)&v; print('Training correctness',label,'strict arithmetic flag rate',float((ft[:,:,1][m]>0).mean()),flush=True)
