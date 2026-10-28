"""Question-level subbagging; selection on training-only holdout."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']: os.environ[k]='4'
import pickle,json,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from train_robust import loss_fn,build
from train import load
from selector_v2 import neural_scores
from selector_semantic_blend import scores as development_scores,norm
from selector_refined import scores as v2_scores
from common import parse_answer
def fit(raw,y,v,seed,fraction,epochs):
    rng=np.random.default_rng(seed); indices=rng.choice(len(raw),int(len(raw)*fraction),replace=False); raw=raw[indices]; y=y[indices]; v=v[indices]; mean=raw.mean((0,1)); scale=raw.std((0,1)); scale[scale<1e-5]=1
    x=torch.from_numpy(np.clip((raw-mean)/scale,-10,10).astype(np.float32)); y=torch.tensor(y,dtype=torch.float32); v=torch.from_numpy(v); torch.manual_seed(seed); model=nn.Sequential(nn.Linear(x.shape[-1],64),nn.ReLU(),nn.Dropout(.2),nn.Linear(64,1)); opt=torch.optim.AdamW(model.parameters(),lr=.0005,weight_decay=.01); snaps={}
    for ep in range(1,epochs+1):
        for ix in np.array_split(rng.permutation(len(x)),max(1,len(x)//128)):
            opt.zero_grad(set_to_none=True); loss_fn(model(x[ix]).squeeze(-1),y[ix],v[ix]).backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5); opt.step()
        if ep in [20,50,100]:
            m=dict(w1=model[0].weight.detach().numpy().T.copy(),b1=model[0].bias.detach().numpy().copy(),w2=model[3].weight.detach().numpy().T.copy(),b2=model[3].bias.detach().numpy().copy()); snaps[ep]=dict(kind='neural_v2',mean=mean,scale=scale,models=[m])
    return snaps
def main():
    start=time.perf_counter(); out=Path('runs_bagged'); out.mkdir(exist_ok=True); protocol=dict(fractions=[.9,1.],epochs=[20,50,100],selection_seeds=[41,42,43,44],final_seeds=list(range(51,59)),holdout_seed=731,width=64,test_scored=False)
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)); b,e,_,_,_,y=load('train'); raw,v=build(b,e); hard=np.array([len(set(parse_answer(c['text']) for c in r['candidates'])-{None})>=2 for r in b]); ids=np.flatnonzero(hard); np.random.default_rng(731).shuffle(ids); cut=int(.8*len(ids)); tr,ho=ids[:cut],ids[cut:]; records=[]
    for fraction in [.9,1.]:
        members=[]
        for seed in [41,42,43,44]:
            members.append(fit(raw[tr],y[tr],v[tr],seed,fraction,100)); print('HOLDOUT FIT',fraction,seed,flush=True)
        for ep in [20,50,100]:
            s=sum(neural_scores(raw[ho],m[ep]) for m in members)/len(members); p=np.where(v[ho],s,-1e20).argmax(1); r=dict(fraction=fraction,epoch=ep,holdout_accuracy=float(y[ho,p].mean()*100)); records.append(r); print(json.dumps(r),flush=True)
    best=max(records,key=lambda r:r['holdout_accuracy']); members=[]
    for seed in range(51,59):
        members.append(fit(raw[ids],y[ids],v[ids],seed,best['fraction'],best['epoch'])[best['epoch']]); print('FINAL FIT',seed,flush=True)
    bundle=dict(members=members); (out/'model_bundle').mkdir(exist_ok=True); (out/'model_bundle/model.pkl').write_bytes(pickle.dumps(bundle,protocol=5)); (out/'holdout_search.json').write_text(json.dumps(records,indent=2))
    bv,ev,_,vv,_,yv=load('validation'); rv,vv=build(bv,ev); sv=sum(neural_scores(rv,m) for m in members)/len(members); hv=np.array([len(set(parse_answer(c['text']) for c in r['candidates'])-{None})>=2 for r in bv]); read=lambda p:pickle.loads(Path(p).read_bytes()); dev=read('runs_round5/development_bundle/model.pkl'); v2=read('runs_v2/final_bundle/model.pkl'); ds,_=development_scores(bv,{'embeddings':ev},dev); vs,_=v2_scores(bv,{'embeddings':ev},v2); bc=yv[np.arange(len(yv)),np.where(vv,ds,-1e20).argmax(1)].astype(bool); comparisons=[]; chosen=None
    def record(name,s,m=None):
        nonlocal chosen
        c=yv[np.arange(len(yv)),np.where(vv,s,-1e20).argmax(1)].astype(bool); r=dict(name=name,hard_correct=int(c[hv].sum()),hard_accuracy=float(c[hv].mean()*100),validation_accuracy=float(c.mean()*100),fixed=int((hv&c&~bc).sum()),regressed=int((hv&~c&bc).sum()),asset_bytes=len(pickle.dumps(m,protocol=5)) if m else 9724630); comparisons.append(r); print(json.dumps(r),flush=True)
        if r['asset_bytes']<=16*1024**2 and (chosen is None or r['hard_correct']>chosen[0]['hard_correct']): chosen=(r,m)
    record('previous_development',ds); record('bagged',sv,bundle)
    for kind,base,s in [('v2',v2,vs),('development',dev,ds)]:
        for w in [.25,.5]: record(kind+'_'+str(w),(1-w)*norm(s,vv)+w*norm(sv,vv),dict(members=members,base=base,base_kind=kind,weight=w))
    if chosen[1] is not None:
        (out/'selected_bundle').mkdir(exist_ok=True); (out/'selected_bundle/model.pkl').write_bytes(pickle.dumps(chosen[1],protocol=5))
    result=dict(training_selection=best,comparison=comparisons,selected=chosen[0],previous_development_retained=chosen[1] is None,seconds=time.perf_counter()-start,test_scored=False)
    (out/'result.json').write_text(json.dumps(result,indent=2))
if __name__=='__main__': main()
