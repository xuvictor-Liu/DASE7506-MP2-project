"""Compact RBF support-vector verifier, selected on a train-only question holdout."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']: os.environ[k]='4'
import json,time,pickle
from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from train import load
from features_v2 import features_v2
from selector_kernel import kernel_scores
from common import parse_answer
out=Path('runs_kernel'); out.mkdir(exist_ok=True); start=time.perf_counter()
b,e,_,v,_,y=load('train'); f,_,_=features_v2(b,e); hard=np.array([len(set(parse_answer(c['text']) for c in row['candidates'])-{None})>=2 for row in b]); ids=np.flatnonzero(hard); np.random.default_rng(731).shuffle(ids); cut=int(.8*len(ids)); tr,ho=ids[:cut],ids[cut:]
pca=PCA(96,svd_solver='randomized',random_state=42).fit(e[tr].reshape(-1,1024)); pe=pca.transform(e.reshape(-1,1024)).reshape(-1,4,96); x=np.concatenate([pe,pe-pe.mean(1,keepdims=True),f],2)
use=v[tr].ravel(); scaler=StandardScaler().fit(x[tr].reshape(-1,x.shape[-1])[use]); xt=scaler.transform(x[tr].reshape(-1,x.shape[-1])[use]); xh=scaler.transform(x[ho].reshape(-1,x.shape[-1])); records=[]; best=None
for C in [.3,3,30]:
  for bandwidth in [.25,1,4]:
    model=SVC(C=C,gamma=bandwidth/x.shape[-1],cache_size=512).fit(xt,y[tr].ravel()[use]); s=model.decision_function(xh).reshape(-1,4); p=np.where(v[ho],s,-1e20).argmax(1); acc=float(y[ho,p].mean()*100)
    r=dict(C=C,bandwidth=bandwidth,holdout_accuracy=acc,seconds=time.perf_counter()-start); records.append(r); print(json.dumps(r),flush=True)
    if best is None or acc>best['holdout_accuracy']: best=r
(out/'holdout_search.json').write_text(json.dumps(records,indent=2))
pca=PCA(96,svd_solver='randomized',random_state=42).fit(e.reshape(-1,1024)); pe=pca.transform(e.reshape(-1,1024)).reshape(-1,4,96); x=np.concatenate([pe,pe-pe.mean(1,keepdims=True),f],2); use=v[ids].ravel(); scaler=StandardScaler().fit(x[ids].reshape(-1,x.shape[-1])[use]); model=SVC(C=best['C'],gamma=best['bandwidth']/x.shape[-1],cache_size=512).fit(scaler.transform(x[ids].reshape(-1,x.shape[-1])[use]),y[ids].ravel()[use])
bundle=dict(pca_mean=pca.mean_,pca_components=pca.components_,mean=scaler.mean_.astype(np.float32),scale=scaler.scale_.astype(np.float32),support=model.support_vectors_.astype(np.float32),dual=model.dual_coef_.ravel().astype(np.float32),gamma=model._gamma,intercept=float(model.intercept_[0]))
bv,ev,_,vv,_,yv=load('validation'); fv,_,_=features_v2(bv,ev); pv=pca.transform(ev.reshape(-1,1024)).reshape(-1,4,96); xv=np.concatenate([pv,pv-pv.mean(1,keepdims=True),fv],2); s=kernel_scores(xv,bundle); p=np.where(vv,s,-1e20).argmax(1); c=yv[np.arange(len(yv)),p]; hv=np.array([len(set(parse_answer(cc['text']) for cc in row['candidates'])-{None})>=2 for row in bv]); result=dict(selected=best,hard_accuracy=float(c[hv].mean()*100),validation_accuracy=float(c.mean()*100),seconds=time.perf_counter()-start)
(out/'model_bundle').mkdir(exist_ok=True); path=out/'model_bundle/model.pkl'; path.write_bytes(pickle.dumps(bundle,protocol=5)); result['asset_bytes']=path.stat().st_size; (out/'selection.json').write_text(json.dumps(result,indent=2)); print(json.dumps(result),flush=True)
