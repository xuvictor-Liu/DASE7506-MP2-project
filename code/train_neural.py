"""Small listwise ranker trained from scratch with NumPy/Adam."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']: os.environ[k]='4'
import time,json,pickle,copy
from pathlib import Path
import numpy as np
from train import load
from features import choose
from common import parse_answer
def main():
    start=time.perf_counter(); out=Path('runs')
    bt,et,ft,vt,at,yt=load('train'); bv,ev,fv,vv,av,yv=load('validation')
    hard=np.array([len(set(parse_answer(c['text']) for c in b['candidates'])-{None})>=2 for b in bv])
    use=(yt.min(1)==0)&vt.any(1)
    x=np.concatenate([et,et-et.mean(1,keepdims=True),ft],2)[use]
    z=np.concatenate([ev,ev-ev.mean(1,keepdims=True),fv],2)
    mean=x.mean((0,1)); scale=x.std((0,1)); scale[scale<1e-5]=1
    x=np.clip((x-mean)/scale,-10,10).astype(np.float32); z=np.clip((z-mean)/scale,-10,10).astype(np.float32)
    y=yt[use].astype(np.float32); y/=y.sum(1,keepdims=True)
    records=[]; options=[]; saved={epoch:[] for epoch in [10,20,40,60]}
    def record(name,s,b):
        p=choose(s,vv); c=yv[np.arange(len(yv)),p]
        r=dict(name=name,validation_accuracy=float(c.mean()*100),hard_accuracy=float(c[hard].mean()*100),elapsed_seconds=time.perf_counter()-start)
        records.append(r); print(json.dumps(r),flush=True); options.append((r['hard_accuracy'],r['validation_accuracy'],b,s))
        (out/'neural_experiments.json').write_text(json.dumps(records,indent=2))
    for seed in [41,42,43]:
        rng=np.random.default_rng(seed); d=x.shape[-1]; h=64
        m=dict(w1=(rng.standard_normal((d,h))*np.sqrt(2/d)).astype(np.float32),b1=np.zeros(h,np.float32),w2=(rng.standard_normal((h,1))*.05).astype(np.float32),b2=np.zeros(1,np.float32))
        a={k:np.zeros_like(v) for k,v in m.items()}; v=copy.deepcopy(a); step=0
        for epoch in range(1,61):
            for ids in np.array_split(rng.permutation(len(x)),max(1,len(x)//128)):
                xx=x[ids].reshape(-1,d); hidden=np.maximum(xx@m['w1']+m['b1'],0)
                mask=(rng.random(hidden.shape)>.15)/.85; hh=hidden*mask
                s=(hh@m['w2']+m['b2']).reshape(-1,4); s-=s.max(1,keepdims=True)
                p=np.exp(s); p/=p.sum(1,keepdims=True)
                ds=((p-y[ids])/len(ids)).reshape(-1,1)
                dh=(ds@m['w2'].T)*mask*(hidden>0)
                grads=dict(w2=hh.T@ds+.002*m['w2'],b2=ds.sum(0),w1=xx.T@dh+.002*m['w1'],b1=dh.sum(0))
                step+=1
                for k,g in grads.items():
                    g=g.astype(np.float32); a[k]=.9*a[k]+.1*g; v[k]=.999*v[k]+.001*g*g
                    m[k]-=.0005*(a[k]/(1-.9**step))/(np.sqrt(v[k]/(1-.999**step))+1e-8)
            if epoch in saved:
                saved[epoch].append(copy.deepcopy(m))
                s=(np.maximum(z@m['w1']+m['b1'],0)@m['w2']+m['b2']).reshape(-1,4)
                record(f'mlp_seed{seed}_epoch{epoch}',s,dict(kind='neural',mean=mean,scale=scale,models=[copy.deepcopy(m)]))
    for epoch,models in saved.items():
        s=sum((np.maximum(z@m['w1']+m['b1'],0)@m['w2']+m['b2']).reshape(-1,4) for m in models)/len(models)
        record(f'mlp_ensemble_epoch{epoch}',s,dict(kind='neural',mean=mean,scale=scale,models=models))
    from selector import predict
    path=out/'model_bundle/model.pkl'; base=pickle.loads(path.read_bytes())
    # First-stage winner is the linear model; retain it unchanged for comparisons.
    assert base['kind']=='linear'
    raw=np.concatenate([ev,ev-ev.mean(1,keepdims=True),fv],2)
    bs=base['model'].decision_function(base['scaler'].transform(raw.reshape(-1,raw.shape[-1]))).reshape(-1,4)
    best_neural=max(options,key=lambda t:t[:2]); record('stage1_best',bs,base)
    for weight in [.25,.5,.75]: record(f'linear_neural_{weight}',(1-weight)*bs+weight*best_neural[3],dict(kind='blend',weight=weight,base=base,neural=best_neural[2]))
    best=max(options,key=lambda t:t[:2]); path.write_bytes(pickle.dumps(best[2],protocol=5))
    (out/'neural_selection.json').write_text(json.dumps(dict(hard_accuracy=best[0],validation_accuracy=best[1],training_seconds=time.perf_counter()-start),indent=2))
if __name__=='__main__': main()
