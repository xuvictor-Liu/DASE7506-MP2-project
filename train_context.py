import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']: os.environ[k]='4'
import json,pickle,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from train_robust import loss_fn
from train import load
from common import parse_answer
from selector_context import build,infer
class Network(nn.Module):
    def __init__(self,dim,rank):
        super().__init__(); self.hidden=nn.Linear(dim,96); self.out=nn.Linear(96,1); self.query=nn.Linear(1024,rank); self.candidate=nn.Linear(1024,rank); self.interaction=nn.Linear(rank,1,bias=False); self.drop=nn.Dropout(.2)
    def forward(self,x,q,d):
        a=self.out(self.drop(torch.relu(self.hidden(x)))); product=torch.tanh(self.query(q))[:,None,:]*torch.tanh(self.candidate(d)); return (a+self.interaction(self.drop(product))).squeeze(-1)
def fit(raw,q,d,y,v,rank,seed,epochs):
    torch.manual_seed(seed); rng=np.random.default_rng(seed); b=dict(mean=raw.mean((0,1)),scale=raw.std((0,1)),qmean=q.mean(0),qscale=q.std(0),dscale=d.std((0,1)))
    for k in ['scale','qscale','dscale']: b[k][b[k]<1e-5]=1
    x=torch.tensor(np.clip((raw-b['mean'])/b['scale'],-10,10),dtype=torch.float32); q=torch.tensor(np.clip((q-b['qmean'])/b['qscale'],-10,10),dtype=torch.float32); d=torch.tensor(np.clip(d/b['dscale'],-10,10),dtype=torch.float32); y=torch.tensor(y,dtype=torch.float32); v=torch.from_numpy(v); net=Network(raw.shape[-1],rank); opt=torch.optim.AdamW(net.parameters(),lr=.0005,weight_decay=.05); snapshots=[]
    for ep in range(1,epochs+1):
        net.train()
        for ix in np.array_split(rng.permutation(len(x)),max(1,len(x)//128)):
            opt.zero_grad(set_to_none=True); loss_fn(net(x[ix],q[ix],d[ix]),y[ix],v[ix]).backward(); torch.nn.utils.clip_grad_norm_(net.parameters(),5); opt.step()
        if ep in [20,50,100]:
            m={k:t.detach().numpy().copy() for k,t in net.state_dict().items()}; snapshots.append((ep,dict(**b,models=[m]))); print('FIT',rank,seed,ep,flush=True)
    # Confirm NumPy export is numerically equivalent before returning it.
    net.eval()
    with torch.no_grad(): expected=net(x[:3],q[:3],d[:3]).numpy()
    last=snapshots[-1][1]; original_q=q[:3].numpy()*b['qscale']+b['qmean']; original_d=d[:3].numpy()*b['dscale']; original_raw=x[:3].numpy()*b['scale']+b['mean']; actual=infer(original_raw,original_q,original_d,last); assert np.allclose(expected,actual,atol=3e-5)
    return snapshots
def main():
    start=time.perf_counter(); out=Path('runs_context'); out.mkdir(exist_ok=True); (out/'protocol.json').write_text(json.dumps(dict(ranks=[16,64],epochs=[20,50,100],holdout_seed=731,final_seeds=[41,42,43],test_scored=False),indent=2))
    bank,e,_,_,_,y=load('train'); raw,q,d,v=build(bank,e); hard=np.array([len(set(parse_answer(c['text']) for c in r['candidates'])-{None})>=2 for r in bank]); ids=np.flatnonzero(hard); np.random.default_rng(731).shuffle(ids); cut=int(.8*len(ids)); tr,ho=ids[:cut],ids[cut:]; records=[]
    for rank in [16,64]:
        for ep,b in fit(raw[tr],q[tr],d[tr],y[tr],v[tr],rank,41,100):
            s=infer(raw[ho],q[ho],d[ho],b); p=np.where(v[ho],s,-1e20).argmax(1); r=dict(rank=rank,epoch=ep,holdout_accuracy=float(y[ho,p].mean()*100)); records.append(r); print(json.dumps(r),flush=True)
    best=max(records,key=lambda r:r['holdout_accuracy']); members=[fit(raw[ids],q[ids],d[ids],y[ids],v[ids],best['rank'],seed,best['epoch'])[-1][1] for seed in [41,42,43]]; b=members[0]; b['models']=[m['models'][0] for m in members]
    (out/'model_bundle').mkdir(exist_ok=True); path=out/'model_bundle/model.pkl'; path.write_bytes(pickle.dumps(b,protocol=5)); bv,ev,_,_,_,yv=load('validation'); rv,qv,dv,vv=build(bv,ev); s=infer(rv,qv,dv,b); p=np.where(vv,s,-1e20).argmax(1); c=yv[np.arange(len(yv)),p]; hv=np.array([len(set(parse_answer(cc['text']) for cc in r['candidates'])-{None})>=2 for r in bv]); result=dict(selected=best,hard_correct=int(c[hv].sum()),hard_accuracy=float(c[hv].mean()*100),validation_accuracy=float(c.mean()*100),seconds=time.perf_counter()-start,asset_bytes=path.stat().st_size,test_scored=False)
    (out/'holdout_search.json').write_text(json.dumps(records,indent=2)); (out/'selection.json').write_text(json.dumps(result,indent=2)); print(json.dumps(result),flush=True)
if __name__=='__main__': main()
