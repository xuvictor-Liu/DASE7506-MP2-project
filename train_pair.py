"""Nonseparable pair comparator; independent train holdout for regularization/epochs."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']: os.environ[k]='4'
import json,time,pickle,copy
from pathlib import Path
import numpy as np
from train import load
from common import parse_answer
from features_v2 import features_v2
from selector_pair import pair_inputs,raw_scores,aggregate
def fit(x,y,seed,reg,epochs):
    mean=x.mean(0); scale=x.std(0); scale[scale<1e-5]=1; x=np.clip((x-mean)/scale,-10,10).astype(np.float32)
    rng=np.random.default_rng(seed); d=x.shape[-1]; width=64
    m=dict(w1=(rng.standard_normal((d,width))*np.sqrt(2/d)).astype(np.float32),b1=np.zeros(width,np.float32),w2=(rng.standard_normal((width,1))*.05).astype(np.float32),b2=np.zeros(1,np.float32))
    a={k:np.zeros_like(v) for k,v in m.items()}; v=copy.deepcopy(a); step=0; snaps=[]
    for ep in range(1,epochs+1):
      for ids in np.array_split(rng.permutation(len(x)),max(1,len(x)//512)):
        xx=x[ids]; h=np.maximum(xx@m['w1']+m['b1'],0); drop=((rng.random(h.shape)>.2)/.8).astype(np.float32); hd=h*drop
        s=hd@m['w2']+m['b2']; prob=1/(1+np.exp(-np.clip(s,-40,40))); ds=(prob-y[ids,None])/len(ids); dh=(ds@m['w2'].T)*drop*(h>0)
        grads=dict(w2=hd.T@ds+reg*m['w2'],w1=xx.T@dh+reg*m['w1'],b1=dh.sum(0),b2=ds.sum(0)); step+=1
        for k,g in grads.items():
          g=g.astype(np.float32); a[k]=.9*a[k]+.1*g; v[k]=.999*v[k]+.001*g*g
          m[k]-=.0005*(a[k]/(1-.9**step))/(np.sqrt(v[k]/(1-.999**step))+1e-8)
      if ep in [20,50,100]: snaps.append((ep,dict(mean=mean,scale=scale,models=[copy.deepcopy(m)])))
    return snaps
def main():
    out=Path('runs_pair'); out.mkdir(exist_ok=True); start=time.perf_counter()
    b,e,_,v,_,y=load('train'); f,_,_=features_v2(b,e); x=pair_inputs(e,f)
    hard=np.array([len(set(parse_answer(c['text']) for c in r['candidates'])-{None})>=2 for r in b]); ids=np.flatnonzero(hard); np.random.default_rng(731).shuffle(ids); cut=int(.8*len(ids)); tr,ho=ids[:cut],ids[cut:]
    eligible=(y[:,:,None]!=y[:,None,:])&v[:,:,None]&v[:,None,:]; target=np.broadcast_to(y[:,:,None],eligible.shape)
    records=[]; best=None
    for reg in [.003,.03]:
      for ep,bundle in fit(x[tr][eligible[tr]],target[tr][eligible[tr]],41,reg,100):
        s=aggregate(raw_scores(x[ho],bundle),v[ho]); p=np.where(v[ho],s,-1e20).argmax(1); acc=float(y[ho,p].mean()*100)
        r=dict(reg=reg,epoch=ep,holdout_accuracy=acc,seconds=time.perf_counter()-start); records.append(r); print(json.dumps(r),flush=True)
        if best is None or acc>best['holdout_accuracy']: best=r
    members=[fit(x[ids][eligible[ids]],target[ids][eligible[ids]],seed,best['reg'],best['epoch'])[-1][1] for seed in [41,42,43]]
    bundle=dict(mean=members[0]['mean'],scale=members[0]['scale'],models=[m['models'][0] for m in members])
    del x
    bv,ev,_,vv,_,yv=load('validation'); fv,_,_=features_v2(bv,ev); s=aggregate(raw_scores(pair_inputs(ev,fv),bundle),vv); p=np.where(vv,s,-1e20).argmax(1); c=yv[np.arange(len(yv)),p]; hv=np.array([len(set(parse_answer(cc['text']) for cc in r['candidates'])-{None})>=2 for r in bv])
    result=dict(selected=best,hard_accuracy=float(c[hv].mean()*100),validation_accuracy=float(c.mean()*100),seconds=time.perf_counter()-start)
    (out/'model_bundle').mkdir(exist_ok=True); (out/'model_bundle/model.pkl').write_bytes(pickle.dumps(bundle,protocol=5)); (out/'holdout_search.json').write_text(json.dumps(records,indent=2)); (out/'selection.json').write_text(json.dumps(result,indent=2)); print(json.dumps(result),flush=True)
if __name__=='__main__': main()
