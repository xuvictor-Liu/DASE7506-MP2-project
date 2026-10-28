"""Numerical relation ranker with training-only holdout model selection."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']: os.environ[k]='4'
import json,time,pickle
from pathlib import Path
import numpy as np
from catboost import CatBoostClassifier,CatBoostRanker
from sklearn.decomposition import PCA
from train import load
from numeric_features import relations
from features_v2 import features_v2
from common import parse_answer
out=Path('runs_numeric'); out.mkdir(exist_ok=True); start=time.perf_counter()
b,e,_,valid,_,y=load('train'); f,_,_=features_v2(b,e); nf=relations(b)
hard=np.array([len(set(parse_answer(c['text']) for c in row['candidates'])-{None})>=2 for row in b]); ids=np.flatnonzero(hard); np.random.default_rng(731).shuffle(ids); cut=int(.8*len(ids)); tr,ho=ids[:cut],ids[cut:]
pca=PCA(n_components=64,svd_solver='randomized',random_state=42).fit(e[tr].reshape(-1,1024)); pe=pca.transform(e.reshape(-1,1024)).reshape(-1,4,64)
x=np.concatenate([f,nf,pe,pe-pe.mean(1,keepdims=True)],2)
records=[]; best=None
for depth in [4,6]:
    model=CatBoostRanker(iterations=1600,depth=depth,learning_rate=.04,loss_function='YetiRank',thread_count=4,random_seed=42,verbose=False,allow_writing_files=False,l2_leaf_reg=10)
    model.fit(x[tr].reshape(-1,x.shape[-1]),y[tr].ravel(),group_id=np.repeat(np.arange(len(tr)),4))
    for nt in [400,800,1600]:
        s=model.predict(x[ho].reshape(-1,x.shape[-1]),ntree_end=nt,thread_count=4).reshape(-1,4); pred=np.where(valid[ho],s,-1e20).argmax(1); acc=float(y[ho,pred].mean()*100)
        r=dict(depth=depth,iterations=nt,holdout_accuracy=acc,seconds=time.perf_counter()-start); records.append(r); print(json.dumps(r),flush=True)
        if best is None or acc>best['holdout_accuracy']: best=r
(out/'holdout_search.json').write_text(json.dumps(records,indent=2))
# PCA may use all training inputs for the final fit, never validation.
pca=PCA(n_components=64,svd_solver='randomized',random_state=42).fit(e.reshape(-1,1024)); pe=pca.transform(e.reshape(-1,1024)).reshape(-1,4,64); x=np.concatenate([f,nf,pe,pe-pe.mean(1,keepdims=True)],2)
model=CatBoostRanker(iterations=best['iterations'],depth=best['depth'],learning_rate=.04,loss_function='YetiRank',thread_count=4,random_seed=42,verbose=False,allow_writing_files=False,l2_leaf_reg=10)
model.fit(x[ids].reshape(-1,x.shape[-1]),y[ids].ravel(),group_id=np.repeat(np.arange(len(ids)),4))
bv,ev,_,vv,_,yv=load('validation'); fv,_,_=features_v2(bv,ev); nv=relations(bv); pv=pca.transform(ev.reshape(-1,1024)).reshape(-1,4,64); xv=np.concatenate([fv,nv,pv,pv-pv.mean(1,keepdims=True)],2)
s=model.predict(xv.reshape(-1,xv.shape[-1]),thread_count=4).reshape(-1,4); pred=np.where(vv,s,-1e20).argmax(1); c=yv[np.arange(len(yv)),pred]; hv=np.array([len(set(parse_answer(cc['text']) for cc in r['candidates'])-{None})>=2 for r in bv])
result=dict(selected=best,hard_accuracy=float(c[hv].mean()*100),validation_accuracy=float(c.mean()*100),seconds=time.perf_counter()-start)
(out/'model_bundle').mkdir(exist_ok=True); (out/'model_bundle/model.pkl').write_bytes(pickle.dumps(dict(mean=pca.mean_,components=pca.components_,model=model),protocol=5)); (out/'selection.json').write_text(json.dumps(result,indent=2)); print(json.dumps(result),flush=True)
