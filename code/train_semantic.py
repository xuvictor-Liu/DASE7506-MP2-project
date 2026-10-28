"""Nonlinear fusion of role-conditioned local text and supplied embeddings."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']: os.environ[k]='4'
import json,pickle,time
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from train import load
from train_role_text import documents
from train_multitask import fit
from features_v2 import features_v2
from selector_v2 import neural_scores
from common import parse_answer
def project_fit(docs):
    vector=TfidfVectorizer(ngram_range=(1,3),max_features=12000,min_df=3,sublinear_tf=True,dtype=np.float32)
    a=vector.fit_transform(docs); svd=TruncatedSVD(n_components=64,n_iter=5,random_state=41).fit(a)
    return vector,svd
def build(bank,e,vector,svd):
    f,v,_=features_v2(bank,e); p=svd.transform(vector.transform(documents(bank))).reshape(-1,4,64).astype(np.float32)
    return np.concatenate([e,e-e.mean(1,keepdims=True),f,p,p-p.mean(1,keepdims=True)],2),v
def main():
    start=time.perf_counter(); out=Path('runs_semantic'); out.mkdir(exist_ok=True)
    b,e,_,v,_,y=load('train'); hard=np.array([len(set(parse_answer(c['text']) for c in r['candidates'])-{None})>=2 for r in b]); ids=np.flatnonzero(hard); np.random.default_rng(731).shuffle(ids); ho=ids[int(.8*len(ids)):]; tr=np.setdiff1d(np.arange(len(b)),ho)
    vector,svd=project_fit(documents([b[i] for i in tr])); x,v=build(b,e,vector,svd); records=[]; best=None
    for alpha in [.1,1]:
        for ep,m in fit(x[tr],y[tr],v[tr],hard[tr],alpha,41,100):
            s=neural_scores(x[ho],m); p=np.where(v[ho],s,-1e20).argmax(1); r=dict(alpha=alpha,epoch=ep,holdout_accuracy=float(y[ho,p].mean()*100)); records.append(r); print(json.dumps(r),flush=True)
            if best is None or r['holdout_accuracy']>best['holdout_accuracy']: best=r
    vector,svd=project_fit(documents(b)); x,v=build(b,e,vector,svd)
    members=[fit(x,y,v,hard,best['alpha'],seed,best['epoch'])[-1][1] for seed in [41,42,43]]
    net=dict(kind='neural_v2',mean=members[0]['mean'],scale=members[0]['scale'],models=[m['models'][0] for m in members]); bundle=dict(vector=vector,svd=svd,net=net)
    bv,ev,_,vv,_,yv=load('validation'); xv,vv=build(bv,ev,vector,svd); s=neural_scores(xv,net); p=np.where(vv,s,-1e20).argmax(1); c=yv[np.arange(len(yv)),p]; hv=np.array([len(set(parse_answer(cc['text']) for cc in r['candidates'])-{None})>=2 for r in bv]); result=dict(selected=best,hard_accuracy=float(c[hv].mean()*100),validation_accuracy=float(c.mean()*100),seconds=time.perf_counter()-start)
    (out/'model_bundle').mkdir(exist_ok=True); path=out/'model_bundle/model.pkl'; path.write_bytes(pickle.dumps(bundle,protocol=5)); result['asset_bytes']=path.stat().st_size; (out/'selection.json').write_text(json.dumps(result,indent=2)); (out/'holdout_search.json').write_text(json.dumps(records,indent=2)); print(json.dumps(result),flush=True)
if __name__=='__main__': main()
