"""Five-fold training-only stacking with invalid candidates masked in training."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']: os.environ[k]='4'
import json,time,pickle,copy
from pathlib import Path
import numpy as np
from sklearn.model_selection import KFold
from catboost import CatBoostClassifier
from train import load
from features_v2 import features_v2
from features import choose
from common import parse_answer
from selector_v2 import neural_scores
from selector_stack import meta_features

out=Path('runs_stack'); out.mkdir(exist_ok=True); start=time.perf_counter()
bt,et,_,vt,_,yt=load('train'); bv,ev,_,vv,_,yv=load('validation')
ft,_,_=features_v2(bt,et); fv,_,_=features_v2(bv,ev)
xt=np.concatenate([et,et-et.mean(1,keepdims=True),ft],2); xv=np.concatenate([ev,ev-ev.mean(1,keepdims=True),fv],2)
use=yt.sum(1)<vt.sum(1); hard=np.array([len(set(parse_answer(c['text']) for c in b['candidates'])-{None})>=2 for b in bv])
oof=np.zeros_like(yt,dtype=np.float32); val=[]; bundles=[]
for fold,(tr,ho) in enumerate(KFold(5,shuffle=True,random_state=2026).split(bt)):
    tr=tr[use[tr]]; mean=xt[tr].mean((0,1)); scale=xt[tr].std((0,1)); scale[scale<1e-5]=1
    x=np.clip((xt[tr]-mean)/scale,-10,10).astype(np.float32); y=yt[tr].astype(np.float32); y/=y.sum(1,keepdims=True)
    valid=vt[tr]; rng=np.random.default_rng(100+fold); d=x.shape[-1]; width=128
    m=dict(w1=(rng.standard_normal((d,width))*np.sqrt(2/d)).astype(np.float32),b1=np.zeros(width,np.float32),w2=(rng.standard_normal((width,1))*.05).astype(np.float32),b2=np.zeros(1,np.float32))
    a={k:np.zeros_like(v) for k,v in m.items()}; v=copy.deepcopy(a); step=0
    for epoch in range(60):
      for ids in np.array_split(rng.permutation(len(x)),max(1,len(x)//128)):
        xx=x[ids].reshape(-1,d); hidden=np.maximum(xx@m['w1']+m['b1'],0); mask=((rng.random(hidden.shape)>.2)/.8).astype(np.float32); hh=hidden*mask
        s=(hh@m['w2']+m['b2']).reshape(-1,4); s=np.where(valid[ids],s,-1e20)
        p=np.exp(s-s.max(1,keepdims=True)); p/=p.sum(1,keepdims=True)
        ds=((p-y[ids])/len(ids)).reshape(-1,1); dh=(ds@m['w2'].T)*mask*(hidden>0)
        grads=dict(w2=hh.T@ds+.003*m['w2'],b2=ds.sum(0),w1=xx.T@dh+.003*m['w1'],b1=dh.sum(0)); step+=1
        for k,g in grads.items():
            g=g.astype(np.float32); a[k]=.9*a[k]+.1*g; v[k]=.999*v[k]+.001*g*g
            m[k]-=.0005*(a[k]/(1-.9**step))/(np.sqrt(v[k]/(1-.999**step))+1e-8)
    b=dict(kind='neural_v2',mean=mean,scale=scale,models=[m]); bundles.append(b)
    oof[ho]=neural_scores(xt[ho],b); val.append(neural_scores(xv,b))
    print('fold',fold,'seconds',time.perf_counter()-start,flush=True)
val=np.mean(val,axis=0); mt=meta_features(bt,ft,oof); mv=meta_features(bv,fv,val)
records=[]; options=[]
def record(name,s,model=None,weight=0):
    correct=yv[np.arange(len(yv)),choose(s,vv)]; r=dict(name=name,hard_accuracy=float(correct[hard].mean()*100),validation_accuracy=float(correct.mean()*100),seconds=time.perf_counter()-start)
    records.append(r); options.append((r['hard_accuracy'],r['validation_accuracy'],model,weight)); print(json.dumps(r),flush=True)
record('masked_oof_ensemble',val)
mask=np.repeat(use,4)&vt.ravel()
for depth in [3,5]:
    model=CatBoostClassifier(iterations=900,depth=depth,learning_rate=.03,loss_function='Logloss',thread_count=4,random_seed=42,verbose=False,allow_writing_files=False,l2_leaf_reg=10)
    model.fit(mt.reshape(-1,mt.shape[-1])[mask],yt.ravel()[mask])
    for nt in [300,900]:
        m=model.copy(); m.shrink(nt); s=m.predict(mv.reshape(-1,mv.shape[-1]),prediction_type='RawFormulaVal',thread_count=4).reshape(-1,4)
        for w in [.5,1]: record(f'meta_d{depth}_n{nt}_w{w}',(1-w)*val+w*s,m,w)
best=max(options,key=lambda r:r[:2]); bundle=dict(models=bundles,meta=best[2],weight=best[3])
(out/'model_bundle').mkdir(exist_ok=True); (out/'model_bundle/model.pkl').write_bytes(pickle.dumps(bundle,protocol=5)); (out/'experiments.json').write_text(json.dumps(records,indent=2))
print('BEST',best[:2],flush=True)
