"""Training-only selection of perturbation-robust pairwise ranking."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']: os.environ[k]='4'
import json,pickle,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from train import load
from features_v2 import features_v2
from selector_v2 import neural_scores
from common import parse_answer
torch.set_num_threads(4); torch.set_num_interop_threads(1)
def build(b,e):
    f,v,_=features_v2(b,e); return np.concatenate([e,e-e.mean(1,keepdims=True),f],2),v
def loss_fn(scores,y,valid):
    pairs=y[:,:,None]*(1-y[:,None,:])*valid[:,:,None]*valid[:,None,:]
    losses=F.softplus(scores[:,None,:]-scores[:,:,None])*pairs
    return (losses.sum((1,2))/pairs.sum((1,2)).clamp_min(1)).mean()
def fit(raw,y,valid,rho,seed,epochs):
    torch.manual_seed(seed); rng=np.random.default_rng(seed); mean=raw.mean((0,1)); scale=raw.std((0,1)); scale[scale<1e-5]=1
    x=torch.from_numpy(np.clip((raw-mean)/scale,-10,10).astype(np.float32)); y=torch.as_tensor(y,dtype=torch.float32); valid=torch.from_numpy(valid)
    model=nn.Sequential(nn.Linear(x.shape[-1],96),nn.ReLU(),nn.Dropout(.2),nn.Linear(96,1)); optimizer=torch.optim.AdamW(model.parameters(),lr=.0005,weight_decay=.01); snapshots=[]
    for ep in range(1,epochs+1):
        model.train()
        for ix in np.array_split(rng.permutation(len(x)),max(1,len(x)//128)):
            optimizer.zero_grad(set_to_none=True); state=torch.get_rng_state(); loss=loss_fn(model(x[ix]).squeeze(-1),y[ix],valid[ix]); loss.backward()
            if rho:
                norm=torch.sqrt(sum(p.grad.square().sum() for p in model.parameters())); perturb=[]
                with torch.no_grad():
                    for p in model.parameters():
                        d=p.grad*(rho/(norm+1e-12)); perturb.append(d); p.add_(d)
                optimizer.zero_grad(set_to_none=True); torch.set_rng_state(state); loss=loss_fn(model(x[ix]).squeeze(-1),y[ix],valid[ix]); loss.backward()
                with torch.no_grad():
                    for p,d in zip(model.parameters(),perturb): p.sub_(d)
            torch.nn.utils.clip_grad_norm_(model.parameters(),5); optimizer.step()
        if ep in [20,50,100]:
            weights=dict(w1=model[0].weight.detach().numpy().T.copy(),b1=model[0].bias.detach().numpy().copy(),w2=model[3].weight.detach().numpy().T.copy(),b2=model[3].bias.detach().numpy().copy())
            snapshots.append((ep,dict(kind='neural_v2',mean=mean,scale=scale,models=[weights]))); print('FIT',rho,seed,ep,flush=True)
    return snapshots
def main():
    start=time.perf_counter(); out=Path('runs_robust'); out.mkdir(exist_ok=True); (out/'protocol.json').write_text(json.dumps(dict(rhos=[0,.05,.15],epochs=[20,50,100],holdout_seed=731,final_seeds=[41,42,43],test_scored=False),indent=2))
    b,e,_,_,_,y=load('train'); raw,v=build(b,e); hard=np.array([len(set(parse_answer(c['text']) for c in r['candidates'])-{None})>=2 for r in b]); ids=np.flatnonzero(hard); np.random.default_rng(731).shuffle(ids); cut=int(.8*len(ids)); tr,ho=ids[:cut],ids[cut:]; records=[]
    for rho in [0,.05,.15]:
        for ep,m in fit(raw[tr],y[tr],v[tr],rho,41,100):
            s=neural_scores(raw[ho],m); p=np.where(v[ho],s,-1e20).argmax(1); r=dict(rho=rho,epoch=ep,holdout_accuracy=float(y[ho,p].mean()*100)); records.append(r); print(json.dumps(r),flush=True)
    best=max(records,key=lambda r:r['holdout_accuracy']); members=[fit(raw[ids],y[ids],v[ids],best['rho'],seed,best['epoch'])[-1][1] for seed in [41,42,43]]; bundle=dict(kind='neural_v2',mean=members[0]['mean'],scale=members[0]['scale'],models=[m['models'][0] for m in members])
    (out/'model_bundle').mkdir(exist_ok=True); path=out/'model_bundle/model.pkl'; path.write_bytes(pickle.dumps(bundle,protocol=5))
    bv,ev,_,vv,_,yv=load('validation'); xv,vv=build(bv,ev); s=neural_scores(xv,bundle); p=np.where(vv,s,-1e20).argmax(1); c=yv[np.arange(len(yv)),p]; hv=np.array([len(set(parse_answer(cc['text']) for cc in r['candidates'])-{None})>=2 for r in bv]); result=dict(selected=best,hard_correct=int(c[hv].sum()),hard_accuracy=float(c[hv].mean()*100),validation_accuracy=float(c.mean()*100),seconds=time.perf_counter()-start,asset_bytes=path.stat().st_size,test_scored=False)
    (out/'selection.json').write_text(json.dumps(result,indent=2)); (out/'holdout_search.json').write_text(json.dumps(records,indent=2)); print(json.dumps(result),flush=True)
if __name__=='__main__': main()
