"""Last bounded experiment: emphasize currently difficult training groups."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']: os.environ[k]='4'
import json,pickle,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from sklearn.decomposition import PCA
from train import load
from features_v2 import features_v2
from selector_v2 import neural_scores,scores as projected_scores
from selector_refined import scores as v2_scores
from selector_semantic_blend import scores as development_scores,norm
from common import parse_answer
torch.set_num_threads(4); torch.set_num_interop_threads(1)
def projection(e):
    a=PCA(n_components=96,svd_solver='randomized',random_state=42).fit(e.reshape(-1,1024)); c=e-e.mean(1,keepdims=True); b=PCA(n_components=96,svd_solver='randomized',random_state=42).fit(c.reshape(-1,1024))
    return dict(mean1=a.mean_,components1=a.components_,mean2=b.mean_,components2=b.components_)
def build(e,f,p):
    return np.concatenate([(e-p['mean1'])@p['components1'].T,(e-e.mean(1,keepdims=True)-p['mean2'])@p['components2'].T,f],2)
def focal_loss(s,y,v,gamma):
    logits=s.masked_fill(~v,-1e4); logp=torch.logsumexp(logits.masked_fill(y==0,-1e4),1)-torch.logsumexp(logits,1); probability=logp.exp().clamp(max=1)
    return (-logp*(1-probability).pow(gamma)).mean()
def fit(raw,y,valid,gamma,seed,epochs):
    torch.manual_seed(seed); rng=np.random.default_rng(seed); mean=raw.mean((0,1)); scale=raw.std((0,1)); scale[scale<1e-5]=1; x=torch.from_numpy(np.clip((raw-mean)/scale,-10,10).astype(np.float32)); y=torch.tensor(y,dtype=torch.float32); valid=torch.from_numpy(valid)
    net=nn.Sequential(nn.Linear(x.shape[-1],64),nn.ReLU(),nn.Dropout(.2),nn.Linear(64,1)); opt=torch.optim.AdamW(net.parameters(),lr=.0005,weight_decay=.01); snapshots=[]
    for ep in range(1,epochs+1):
        for ix in np.array_split(rng.permutation(len(x)),max(1,len(x)//128)):
            opt.zero_grad(set_to_none=True); focal_loss(net(x[ix]).squeeze(-1),y[ix],valid[ix],gamma).backward(); torch.nn.utils.clip_grad_norm_(net.parameters(),5); opt.step()
        if ep in [20,50,100]:
            weights=dict(w1=net[0].weight.detach().numpy().T.copy(),b1=net[0].bias.detach().numpy().copy(),w2=net[3].weight.detach().numpy().T.copy(),b2=net[3].bias.detach().numpy().copy()); snapshots.append((ep,dict(kind='neural_v2',mean=mean,scale=scale,models=[weights]))); print('FIT',gamma,seed,ep,flush=True)
    return snapshots
def main():
    start=time.perf_counter(); out=Path('runs_final_focal'); out.mkdir(exist_ok=True); protocol=dict(gamma=[0,1,2],epochs=[20,50,100],holdout_seed=731,seeds=[41,42,43],stop_after_this_family=True,test_scored=False); (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    b,e,_,v,_,y=load('train'); f,v,_=features_v2(b,e); hard=np.array([len(set(parse_answer(c['text']) for c in row['candidates'])-{None})>=2 for row in b]); ids=np.flatnonzero(hard); np.random.default_rng(731).shuffle(ids); ho=ids[int(.8*len(ids)):]; tr=np.setdiff1d(np.arange(len(b)),ho); use=tr[hard[tr]]; p=projection(e[tr]); raw=build(e,f,p); records=[]
    for gamma in [0,1,2]:
        for ep,m in fit(raw[use],y[use],v[use],gamma,41,100):
            s=neural_scores(raw[ho],m); choices=np.where(v[ho],s,-1e20).argmax(1); r=dict(gamma=gamma,epoch=ep,holdout_accuracy=float(y[ho,choices].mean()*100)); records.append(r); print(json.dumps(r),flush=True)
    best=max(records,key=lambda r:r['holdout_accuracy']); p=projection(e); raw=build(e,f,p); members=[fit(raw[ids],y[ids],v[ids],best['gamma'],seed,best['epoch'])[-1][1] for seed in [41,42,43]]; model=dict(kind='neural_v2',mean=members[0]['mean'],scale=members[0]['scale'],projection=p,models=[m['models'][0] for m in members]); (out/'model_bundle').mkdir(exist_ok=True); (out/'model_bundle/model.pkl').write_bytes(pickle.dumps(model,protocol=5))
    bv,ev,_,vv,_,yv=load('validation'); arrays={'embeddings':ev}; s,_=projected_scores(bv,arrays,model); read=lambda path:pickle.loads(Path(path).read_bytes()); dev=read('runs_round5/development_bundle/model.pkl'); v2=read('runs_v2/final_bundle/model.pkl'); sd,_=development_scores(bv,arrays,dev); s2,_=v2_scores(bv,arrays,v2); hv=np.array([len(set(parse_answer(c['text']) for c in row['candidates'])-{None})>=2 for row in bv]); bc=yv[np.arange(len(yv)),np.where(vv,sd,-1e20).argmax(1)].astype(bool); results=[]
    def record(name,z):
        c=yv[np.arange(len(yv)),np.where(vv,z,-1e20).argmax(1)].astype(bool); r=dict(name=name,hard_correct=int(c[hv].sum()),hard_accuracy=float(c[hv].mean()*100),validation_accuracy=float(c.mean()*100),fixed=int((hv&c&~bc).sum()),regressed=int((hv&~c&bc).sum())); results.append(r); print(json.dumps(r),flush=True)
    record('previous_development',sd); record('focal_selected',s)
    for name,z in [('v2',s2),('development',sd)]:
        for w in [.25,.5]: record(name+'_'+str(w),(1-w)*norm(z,vv)+w*norm(s,vv))
    result=dict(selected_training=best,comparison=results,seconds=time.perf_counter()-start,test_scored=False,stop_after_this_family=True); (out/'holdout_search.json').write_text(json.dumps(records,indent=2)); (out/'result.json').write_text(json.dumps(result,indent=2))
if __name__=='__main__': main()
