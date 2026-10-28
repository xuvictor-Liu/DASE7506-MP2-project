"""Second-round development: train/validation only; never scores the test bank."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']: os.environ[k]='4'
import time,json,pickle,copy
from pathlib import Path
import numpy as np
from train import load
from features_v2 import features_v2
from selector_v2 import neural_scores
from common import parse_answer
from features import choose
from sklearn.decomposition import PCA
from catboost import CatBoostClassifier

def main():
    import argparse
    parser=argparse.ArgumentParser(); parser.add_argument('--projected',action='store_true'); args=parser.parse_args()
    out=Path('runs_projected' if args.projected else 'runs_v2'); out.mkdir(exist_ok=True); start=time.perf_counter()
    bt,et,_,vt,at,yt=load('train'); bv,ev,_,vv,av,yv=load('validation')
    ft,_,_=features_v2(bt,et); fv,_,_=features_v2(bv,ev)
    hard=np.array([len(set(parse_answer(c['text']) for c in b['candidates'])-{None})>=2 for b in bv])
    use=yt.min(1)==0
    projection=None
    if args.projected:
        center=et-et.mean(1,keepdims=True)
        p1=PCA(n_components=96,svd_solver='randomized',random_state=42).fit(et.reshape(-1,1024))
        p2=PCA(n_components=96,svd_solver='randomized',random_state=42).fit(center.reshape(-1,1024))
        projection=dict(mean1=p1.mean_,components1=p1.components_,mean2=p2.mean_,components2=p2.components_)
        def project(e): return np.concatenate([(e-p1.mean_)@p1.components_.T,(e-e.mean(1,keepdims=True)-p2.mean_)@p2.components_.T],2)
        raw=np.concatenate([project(et),ft],2)[use]; rawv=np.concatenate([project(ev),fv],2)
    else:
        raw=np.concatenate([et,et-et.mean(1,keepdims=True),ft],2)[use]
        rawv=np.concatenate([ev,ev-ev.mean(1,keepdims=True),fv],2)
    mean=raw.mean((0,1)); scale=raw.std((0,1)); scale[scale<1e-5]=1
    x=np.clip((raw-mean)/scale,-10,10).astype(np.float32); z=np.clip((rawv-mean)/scale,-10,10).astype(np.float32)
    y=yt[use].astype(np.float32); target=y/y.sum(1,keepdims=True)
    records=[]; choices=[]
    def record(name,s,b=None):
        correct=yv[np.arange(len(yv)),choose(s,vv)]
        r=dict(name=name,validation_accuracy=float(correct.mean()*100),hard_accuracy=float(correct[hard].mean()*100),seconds=time.perf_counter()-start)
        records.append(r); print(json.dumps(r),flush=True)
        if b is not None: choices.append((r['hard_accuracy'],r['validation_accuracy'],b,s,name))
        (out/'experiments.json').write_text(json.dumps(records,indent=2))
    for objective in (['soft','pair'] if args.projected else ['soft','pair','marginal']):
      for width in [64,192]:
        snapshots={ep:[] for ep in [20,40,60,100]}
        for seed in [41,42,43]:
          rng=np.random.default_rng(seed); d=x.shape[-1]
          m=dict(w1=(rng.standard_normal((d,width))*np.sqrt(2/d)).astype(np.float32),b1=np.zeros(width,np.float32),w2=(rng.standard_normal((width,1))*.05).astype(np.float32),b2=np.zeros(1,np.float32))
          a={k:np.zeros_like(v) for k,v in m.items()}; v=copy.deepcopy(a); step=0
          for epoch in range(1,101):
            for ids in np.array_split(rng.permutation(len(x)),max(1,len(x)//128)):
              xx=x[ids].reshape(-1,d); hidden=np.maximum(xx@m['w1']+m['b1'],0)
              mask=((rng.random(hidden.shape)>.2)/.8).astype(np.float32); hh=hidden*mask
              s=(hh@m['w2']+m['b2']).reshape(-1,4)
              p=np.exp(s-s.max(1,keepdims=True)); p/=p.sum(1,keepdims=True)
              if objective=='soft': ds=p-target[ids]
              elif objective=='marginal':
                pos=p*y[ids]; pos/=pos.sum(1,keepdims=True)+1e-12; ds=p-pos
              else:
                diff=np.clip(s[:,:,None]-s[:,None,:],-40,40)
                pairs=y[ids,:,None]*(1-y[ids,None,:])
                grad=-pairs/(1+np.exp(diff)); grad/=np.maximum(pairs.sum((1,2)),1)[:,None,None]
                ds=grad.sum(2)-grad.sum(1)
              ds=(ds/len(ids)).reshape(-1,1)
              dh=(ds@m['w2'].T)*mask*(hidden>0)
              grads=dict(w2=hh.T@ds+.002*m['w2'],b2=ds.sum(0),w1=xx.T@dh+.002*m['w1'],b1=dh.sum(0))
              step+=1
              for k,g in grads.items():
                g=g.astype(np.float32); a[k]=.9*a[k]+.1*g; v[k]=.999*v[k]+.001*g*g
                m[k]-=.0005*(a[k]/(1-.9**step))/(np.sqrt(v[k]/(1-.999**step))+1e-8)
            if epoch in snapshots:
              snapshots[epoch].append(copy.deepcopy(m))
              s=(np.maximum(z@m['w1']+m['b1'],0)@m['w2']+m['b2']).reshape(-1,4)
              record(f'{objective}_h{width}_seed{seed}_ep{epoch}',s)
        for ep,models in snapshots.items():
          b=dict(kind='neural_v2',mean=mean,scale=scale,models=models,projection=projection)
          record(f'{objective}_h{width}_ensemble_ep{ep}',neural_scores(rawv,b),b)
    # A different inductive bias: PCA embeddings and structural gradient boosting.
    pca=PCA(n_components=96,svd_solver='randomized',random_state=42).fit(et.reshape(-1,1024))
    pt=pca.transform(et.reshape(-1,1024)).reshape(-1,4,96); pv=pca.transform(ev.reshape(-1,1024)).reshape(-1,4,96)
    tx=np.concatenate([pt,pt-pt.mean(1,keepdims=True),ft],2); vx=np.concatenate([pv,pv-pv.mean(1,keepdims=True),fv],2)
    for depth in ([] if args.projected else [4,6]):
      model=CatBoostClassifier(iterations=1800,depth=depth,learning_rate=.035,loss_function='Logloss',thread_count=4,random_seed=42,verbose=False,allow_writing_files=False,l2_leaf_reg=5)
      model.fit(tx[use].reshape(-1,tx.shape[-1]),yt[use].ravel())
      for nt in [600,1200,1800]:
        m=model.copy(); m.shrink(nt); s=m.predict(vx.reshape(-1,vx.shape[-1]),prediction_type='RawFormulaVal',thread_count=4).reshape(-1,4)
        record(f'tree_depth{depth}_{nt}',s,dict(kind='tree_v2',mean=pca.mean_,components=pca.components_,model=m))
    if not args.projected:
        top=max((c for c in choices if c[2]['kind']=='neural_v2'),key=lambda c:c[:2]); tree=max((c for c in choices if c[2]['kind']=='tree_v2'),key=lambda c:c[:2])
        for w in [.25,.5,.75]: record(f'neural_tree_{w}',(1-w)*top[3]+w*tree[3],dict(kind='blend_v2',members=[top[2],tree[2]],weights=[1-w,w]))
    best=max(choices,key=lambda c:c[:2]); path=out/'model_bundle'; path.mkdir(exist_ok=True)
    (path/'model.pkl').write_bytes(pickle.dumps(best[2],protocol=5))
    (out/'selection.json').write_text(json.dumps(dict(name=best[4],hard_accuracy=best[0],validation_accuracy=best[1],seconds=time.perf_counter()-start,asset_bytes=(path/'model.pkl').stat().st_size),indent=2))
    print('SELECTED',best[4],best[:2],flush=True)
if __name__=='__main__': main()
