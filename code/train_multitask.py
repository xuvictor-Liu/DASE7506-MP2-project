"""Use easy training examples as correctness supervision plus hard-group ranking."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']: os.environ[k]='4'
import time,json,pickle,copy
from pathlib import Path
import numpy as np
from train import load
from features_v2 import features_v2
from selector_v2 import neural_scores
from common import parse_answer

def fit(x,y,valid,hard,alpha,seed,epochs):
    mean=x.mean((0,1)); scale=x.std((0,1)); scale[scale<1e-5]=1
    x=np.clip((x-mean)/scale,-10,10).astype(np.float32); target=y/y.sum(1,keepdims=True)
    rng=np.random.default_rng(seed); d=x.shape[-1]; width=96
    m=dict(w1=(rng.standard_normal((d,width))*np.sqrt(2/d)).astype(np.float32),b1=np.zeros(width,np.float32),w2=(rng.standard_normal((width,1))*.05).astype(np.float32),b2=np.zeros(1,np.float32))
    a={k:np.zeros_like(v) for k,v in m.items()}; v=copy.deepcopy(a); step=0; snaps=[]
    for ep in range(1,epochs+1):
      for ids in np.array_split(rng.permutation(len(x)),max(1,len(x)//128)):
        xx=x[ids].reshape(-1,d); h=np.maximum(xx@m['w1']+m['b1'],0); drop=((rng.random(h.shape)>.2)/.8).astype(np.float32); hd=h*drop
        s=(hd@m['w2']+m['b2']).reshape(-1,4); sm=np.where(valid[ids],s,-1e20); p=np.exp(sm-sm.max(1,keepdims=True)); p/=p.sum(1,keepdims=True)
        sigmoid=1/(1+np.exp(-np.clip(s,-40,40)))
        ds=(p-target[ids])*hard[ids,None]+alpha*(sigmoid-y[ids])*valid[ids]/valid[ids].sum(1,keepdims=True)
        ds=(ds/len(ids)).reshape(-1,1); dh=(ds@m['w2'].T)*drop*(h>0)
        grads=dict(w2=hd.T@ds+.003*m['w2'],w1=xx.T@dh+.003*m['w1'],b1=dh.sum(0),b2=ds.sum(0)); step+=1
        for k,g in grads.items():
          g=g.astype(np.float32); a[k]=.9*a[k]+.1*g; v[k]=.999*v[k]+.001*g*g
          m[k]-=.0005*(a[k]/(1-.9**step))/(np.sqrt(v[k]/(1-.999**step))+1e-8)
      if ep in [20,50,100]: snaps.append((ep,dict(kind='neural_v2',mean=mean,scale=scale,models=[copy.deepcopy(m)])))
    return snaps
def main():
    out=Path('runs_multitask'); out.mkdir(exist_ok=True); start=time.perf_counter()
    b,e,_,v,_,y=load('train'); f,_,_=features_v2(b,e); x=np.concatenate([e,e-e.mean(1,keepdims=True),f],2)
    hard=np.array([len(set(parse_answer(c['text']) for c in row['candidates'])-{None})>=2 for row in b]); ids=np.flatnonzero(hard); np.random.default_rng(731).shuffle(ids); ho=ids[int(.8*len(ids)):]; tr=np.setdiff1d(np.arange(len(b)),ho)
    records=[]; best=None
    for alpha in [.25,1,4]:
      for ep,bundle in fit(x[tr],y[tr],v[tr],hard[tr],alpha,41,100):
        s=neural_scores(x[ho],bundle); p=np.where(v[ho],s,-1e20).argmax(1); acc=float(y[ho,p].mean()*100)
        r=dict(alpha=alpha,epoch=ep,holdout_accuracy=acc,seconds=time.perf_counter()-start); records.append(r); print(json.dumps(r),flush=True)
        if best is None or acc>best['holdout_accuracy']: best=r
    (out/'holdout_search.json').write_text(json.dumps(records,indent=2))
    members=[fit(x,y,v,hard,best['alpha'],seed,best['epoch'])[-1][1] for seed in [41,42,43]]
    bundle=dict(kind='neural_v2',mean=members[0]['mean'],scale=members[0]['scale'],models=[m['models'][0] for m in members])
    bv,ev,_,vv,_,yv=load('validation'); fv,_,_=features_v2(bv,ev); xv=np.concatenate([ev,ev-ev.mean(1,keepdims=True),fv],2); s=neural_scores(xv,bundle); p=np.where(vv,s,-1e20).argmax(1); c=yv[np.arange(len(yv)),p]
    hv=np.array([len(set(parse_answer(cc['text']) for cc in r['candidates'])-{None})>=2 for r in bv]); result=dict(selected=best,hard_accuracy=float(c[hv].mean()*100),validation_accuracy=float(c.mean()*100),seconds=time.perf_counter()-start)
    (out/'model_bundle').mkdir(exist_ok=True); (out/'model_bundle/model.pkl').write_bytes(pickle.dumps(bundle,protocol=5)); (out/'selection.json').write_text(json.dumps(result,indent=2)); print(json.dumps(result),flush=True)
if __name__=='__main__': main()
