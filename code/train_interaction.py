"""Linear bilinear-feature verifier; C selected on training-only held-out questions."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']: os.environ[k]='4'
import pickle,json,time
from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from train import load
from common import parse_answer
from selector_interaction import build
out=Path('runs_interaction'); out.mkdir(exist_ok=True); start=time.perf_counter()
b,e,_,v,_,y=load('train'); hard=np.array([len(set(parse_answer(c['text']) for c in row['candidates'])-{None})>=2 for row in b]); ids=np.flatnonzero(hard); np.random.default_rng(731).shuffle(ids); cut=int(.8*len(ids)); tr,ho=ids[:cut],ids[cut:]
def projection(e):
    q=PCA(32,svd_solver='randomized',random_state=42).fit(e.mean(1)); d=PCA(96,svd_solver='randomized',random_state=42).fit((e-e.mean(1,keepdims=True)).reshape(-1,1024)); r=PCA(96,svd_solver='randomized',random_state=42).fit(e.reshape(-1,1024))
    return dict(qmean=q.mean_,qcomp=q.components_,dmean=d.mean_,dcomp=d.components_,rmean=r.mean_,rcomp=r.components_)
p=projection(e[tr]); x,_=build(b,e,p); trainmask=v[tr].ravel(); scaler=StandardScaler().fit(x[tr].reshape(-1,x.shape[-1])[trainmask]); xx=scaler.transform(x[tr].reshape(-1,x.shape[-1])[trainmask]); z=scaler.transform(x[ho].reshape(-1,x.shape[-1])); records=[]; best=None
for C in [.0001,.001,.01,.1]:
    m=LogisticRegression(C=C,max_iter=600).fit(xx,y[tr].ravel()[trainmask]); s=m.decision_function(z).reshape(-1,4); pred=np.where(v[ho],s,-1e20).argmax(1); acc=float(y[ho,pred].mean()*100)
    r=dict(C=C,holdout_accuracy=acc,seconds=time.perf_counter()-start); records.append(r); print(json.dumps(r),flush=True)
    if best is None or acc>best['holdout_accuracy']: best=r
p=projection(e); x,_=build(b,e,p); trainmask=v[ids].ravel(); scaler=StandardScaler().fit(x[ids].reshape(-1,x.shape[-1])[trainmask]); m=LogisticRegression(C=best['C'],max_iter=600).fit(scaler.transform(x[ids].reshape(-1,x.shape[-1])[trainmask]),y[ids].ravel()[trainmask])
bv,ev,_,vv,_,yv=load('validation'); xv,_=build(bv,ev,p); s=m.decision_function(scaler.transform(xv.reshape(-1,xv.shape[-1]))).reshape(-1,4); pred=np.where(vv,s,-1e20).argmax(1); c=yv[np.arange(len(yv)),pred]; hv=np.array([len(set(parse_answer(cc['text']) for cc in r['candidates'])-{None})>=2 for r in bv])
result=dict(selected=best,hard_accuracy=float(c[hv].mean()*100),validation_accuracy=float(c.mean()*100),seconds=time.perf_counter()-start)
(out/'model_bundle').mkdir(exist_ok=True); (out/'model_bundle/model.pkl').write_bytes(pickle.dumps(dict(projection=p,mean=scaler.mean_,scale=scaler.scale_,coef=m.coef_.ravel(),intercept=float(m.intercept_[0])),protocol=5)); (out/'holdout_search.json').write_text(json.dumps(records,indent=2)); (out/'selection.json').write_text(json.dumps(result,indent=2)); print(json.dumps(result),flush=True)
