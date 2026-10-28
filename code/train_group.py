"""Select group ranker architecture on training-only holdout, then inspect validation."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']: os.environ[k]='4'
import time,json,pickle,copy
from pathlib import Path
import numpy as np
from train import load
from group_features import group_features
from selector_group import model_scores

def fit(x,y,mask,seed,width,reg,epochs):
    mean=x.mean((0,1)); scale=x.std((0,1)); scale[scale<1e-5]=1
    x=np.clip((x-mean)/scale,-8,8).astype(np.float32); y=y*mask; y=y/y.sum(1,keepdims=True)
    rng=np.random.default_rng(seed); d=x.shape[-1]
    m=dict(w1=(rng.standard_normal((d,width))*np.sqrt(2/d)).astype(np.float32),b1=np.zeros(width,np.float32),w2=(rng.standard_normal((width,1))*.05).astype(np.float32))
    a={k:np.zeros_like(v) for k,v in m.items()}; v=copy.deepcopy(a); step=0; snapshots=[]
    for epoch in range(1,epochs+1):
      for ids in np.array_split(rng.permutation(len(x)),max(1,len(x)//96)):
        xx=x[ids].reshape(-1,d); h=np.maximum(xx@m['w1']+m['b1'],0); drop=((rng.random(h.shape)>.2)/.8).astype(np.float32); hd=h*drop
        s=(hd@m['w2']).reshape(-1,4); s=np.where(mask[ids],s,-1e20); p=np.exp(s-s.max(1,keepdims=True)); p/=p.sum(1,keepdims=True)
        ds=((p-y[ids])/len(ids)).reshape(-1,1); dh=(ds@m['w2'].T)*drop*(h>0)
        grads=dict(w2=hd.T@ds+reg*m['w2'],w1=xx.T@dh+reg*m['w1'],b1=dh.sum(0)); step+=1
        for k,g in grads.items():
          a[k]=.9*a[k]+.1*g; v[k]=.999*v[k]+.001*g*g
          m[k]-=.0005*(a[k]/(1-.9**step))/(np.sqrt(v[k]/(1-.999**step))+1e-8)
      if epoch in [20,50,100]: snapshots.append((epoch,dict(mean=mean,scale=scale,models=[copy.deepcopy(m)])))
    return snapshots

def main():
    out=Path('runs_group'); out.mkdir(exist_ok=True); start=time.perf_counter()
    bt,et,_,_,_,yt=load('train'); x,mask,valid=group_features(bt,et)
    hard=mask.sum(1)>=2; ids=np.flatnonzero(hard); rng=np.random.default_rng(731); rng.shuffle(ids); cut=int(.8*len(ids)); tr,ho=ids[:cut],ids[cut:]
    records=[]; best=None
    print('Hard training groups',len(ids),'holdout',len(ho),flush=True)
    for width in [48,128]:
      for reg in [.003,.03]:
        for epoch,b in fit(x[tr],yt[tr],mask[tr],41,width,reg,100):
          s=model_scores(x[ho],b); p=np.where(mask[ho],s,-1e20).argmax(1); acc=float(yt[ho,p].mean()*100)
          r=dict(width=width,reg=reg,epoch=epoch,holdout_accuracy=acc,seconds=time.perf_counter()-start); records.append(r); print(json.dumps(r),flush=True)
          if best is None or acc>best['holdout_accuracy']: best=r
    (out/'holdout_search.json').write_text(json.dumps(records,indent=2))
    # Architecture/epoch fixed before reading any official validation references here.
    members=[]
    for seed in [41,42,43]:
        members.append(fit(x[ids],yt[ids],mask[ids],seed,best['width'],best['reg'],best['epoch'])[-1][1])
    bundle=dict(mean=members[0]['mean'],scale=members[0]['scale'],models=[m['models'][0] for m in members])
    bv,ev,_,_,_,yv=load('validation'); xv,mv,vv=group_features(bv,ev); s=model_scores(xv,bundle); p=np.where(vv,s,-1e20).argmax(1); correct=yv[np.arange(len(yv)),p]; hv=mv.sum(1)>=2
    result=dict(selected=best,hard_accuracy=float(correct[hv].mean()*100),validation_accuracy=float(correct.mean()*100),seconds=time.perf_counter()-start)
    (out/'model_bundle').mkdir(exist_ok=True); (out/'model_bundle/model.pkl').write_bytes(pickle.dumps(bundle,protocol=5)); (out/'selection.json').write_text(json.dumps(result,indent=2)); print(json.dumps(result),flush=True)
if __name__=='__main__': main()
