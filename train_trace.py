"""OOF-trained local-evidence calibrator; fitting never sees validation references."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']: os.environ[k]='4'
import pickle,json
from pathlib import Path
import numpy as np
from sklearn.model_selection import KFold
from catboost import CatBoostClassifier
from train import load
from trace_evidence import features
from features_v2 import features_v2
from selector_v2 import neural_scores
from selector_refined import scores
from common import parse_answer
out=Path('runs_trace'); out.mkdir(exist_ok=True)
b,e,_,v,_,y=load('train'); ft=features(b); old,_,_=features_v2(b,e); raw=np.concatenate([e,e-e.mean(1,keepdims=True),old],2); folds=pickle.loads(Path('runs_stack/model_bundle/model.pkl').read_bytes())['models']; oof=np.zeros_like(y,dtype=float)
for (_,ho),m in zip(KFold(5,shuffle=True,random_state=2026).split(b),folds): oof[ho]=neural_scores(raw[ho],m)
x=np.concatenate([oof[:,:,None],(oof-oof.mean(1,keepdims=True))[:,:,None],ft],2); hard=np.array([len(set(parse_answer(c['text']) for c in row['candidates'])-{None})>=2 for row in b]); ids=np.flatnonzero(hard); np.random.default_rng(731).shuffle(ids); cut=int(.8*len(ids)); tr,ho=ids[:cut],ids[cut:]; records=[]; best=None
for depth in [2,4]:
    m=CatBoostClassifier(iterations=800,depth=depth,learning_rate=.03,l2_leaf_reg=10,thread_count=4,random_seed=42,verbose=False,allow_writing_files=False)
    use=v[tr].ravel(); m.fit(x[tr].reshape(-1,x.shape[-1])[use],y[tr].ravel()[use])
    for n in [200,800]:
        s=m.predict(x[ho].reshape(-1,x.shape[-1]),prediction_type='RawFormulaVal',ntree_end=n,thread_count=4).reshape(-1,4); p=np.where(v[ho],s,-1e20).argmax(1); acc=float(y[ho,p].mean()*100); r=dict(depth=depth,iterations=n,holdout_accuracy=acc); records.append(r); print(json.dumps(r),flush=True)
        if best is None or acc>best['holdout_accuracy']: best=r
m=CatBoostClassifier(iterations=best['iterations'],depth=best['depth'],learning_rate=.03,l2_leaf_reg=10,thread_count=4,random_seed=42,verbose=False,allow_writing_files=False); use=v[ids].ravel(); m.fit(x[ids].reshape(-1,x.shape[-1])[use],y[ids].ravel()[use])
bv,ev,_,vv,_,yv=load('validation'); base=pickle.loads(Path('runs_v2/final_bundle/model.pkl').read_bytes()); s0,_=scores(bv,{'embeddings':ev},base); fv=features(bv); xv=np.concatenate([s0[:,:,None],(s0-s0.mean(1,keepdims=True))[:,:,None],fv],2); s=m.predict(xv.reshape(-1,xv.shape[-1]),prediction_type='RawFormulaVal',thread_count=4).reshape(-1,4); p=np.where(vv,s,-1e20).argmax(1); c=yv[np.arange(len(yv)),p]; hv=np.array([len(set(parse_answer(cc['text']) for cc in row['candidates'])-{None})>=2 for row in bv]); result=dict(selected=best,hard_accuracy=float(c[hv].mean()*100),validation_accuracy=float(c.mean()*100)); print(json.dumps(result),flush=True)
(out/'model_bundle').mkdir(exist_ok=True); (out/'model_bundle/model.pkl').write_bytes(pickle.dumps(dict(base=base,model=m),protocol=5)); (out/'holdout_search.json').write_text(json.dumps(records,indent=2)); (out/'selection.json').write_text(json.dumps(result,indent=2))
