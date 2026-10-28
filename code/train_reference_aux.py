"""Gold-reference operation and answer-scale supervision, training split only."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']: os.environ[k]='4'
import ast,re,pickle,json,time
from fractions import Fraction
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from train_robust import loss_fn
from train import load
from common import parse_answer,read_jsonl,DATA
from selector_context import build
from selector_reference_aux import infer
from selector_semantic_blend import norm,scores as previous_scores
def signature(answer):
    equations=re.findall(r'<<([^<>]+)>>',answer); counts=np.zeros(4)
    for eq in equations:
        left=eq.rsplit('=',1)[0].replace(',','')
        try:
            tree=ast.parse(left,mode='eval')
            for node in ast.walk(tree):
                for i,op in enumerate([ast.Add,ast.Sub,ast.Mult,ast.Div]): counts[i]+=isinstance(node,op)
        except (SyntaxError,ValueError,RecursionError): pass
    value=float(Fraction(parse_answer(answer)))
    return np.r_[np.log1p(counts),np.log1p(len(equations)),value<0,np.log1p(abs(value)),value==int(value)].astype(np.float32)
class Net(nn.Module):
    def __init__(self,dim):
        super().__init__(); self.hidden=nn.Linear(dim,96); self.query=nn.Linear(1024,64); self.candidate=nn.Linear(1024,64); self.out=nn.Linear(160,1); self.aux=nn.Linear(64,8); self.drop=nn.Dropout(.2)
    def forward(self,x,q,d):
        context=torch.relu(self.query(q)); candidate=torch.tanh(self.candidate(d)); h=torch.relu(self.hidden(x)); z=torch.cat([h,context[:,None,:]*candidate],2); return self.out(self.drop(z)).squeeze(-1),self.aux(context)
def fit(raw,q,d,y,valid,targets,hard,alpha,seed,epochs):
    torch.manual_seed(seed); rng=np.random.default_rng(seed); b=dict(mean=raw.mean((0,1)),scale=raw.std((0,1)),qmean=q.mean(0),qscale=q.std(0),dscale=d.std((0,1)))
    for k in ['scale','qscale','dscale']: b[k][b[k]<1e-5]=1
    target_mean=targets.mean(0); target_scale=targets.std(0); target_scale[target_scale<1e-5]=1
    arrays=[np.clip((raw-b['mean'])/b['scale'],-10,10),np.clip((q-b['qmean'])/b['qscale'],-10,10),np.clip(d/b['dscale'],-10,10),y,(targets-target_mean)/target_scale]
    x,q,d,y,t=[torch.tensor(a,dtype=torch.float32) for a in arrays]; v=torch.from_numpy(valid); hard=torch.from_numpy(hard); net=Net(raw.shape[-1]); opt=torch.optim.AdamW(net.parameters(),lr=.0005,weight_decay=.05); snapshots=[]
    for ep in range(1,epochs+1):
        net.train()
        for ix in np.array_split(rng.permutation(len(x)),max(1,len(x)//128)):
            opt.zero_grad(set_to_none=True); s,aux=net(x[ix],q[ix],d[ix]); use=hard[ix]; ranking=loss_fn(s[use],y[ix][use],v[ix][use]) if use.any() else s.sum()*0; loss=ranking+alpha*F.smooth_l1_loss(aux,t[ix]); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(),5); opt.step()
        if ep in [20,50,100]:
            weights={k:z.detach().numpy().copy() for k,z in net.state_dict().items() if not k.startswith('aux.')}; snapshots.append((ep,dict(**b,models=[weights]))); print('FIT',alpha,seed,ep,flush=True)
    net.eval()
    with torch.no_grad(): expected=net(x[:3],q[:3],d[:3])[0].numpy()
    actual=infer(raw[:3],arrays[1][:3]*b['qscale']+b['qmean'],arrays[2][:3]*b['dscale'],snapshots[-1][1]); assert np.allclose(expected,actual,atol=3e-5)
    return snapshots
def main():
    start=time.perf_counter(); out=Path('runs_reference_aux'); out.mkdir(exist_ok=True); protocol=dict(alpha=[0,.1,1],epochs=[20,50,100],holdout_seed=731,final_seeds=[41,42,43],references_used='train only',test_scored=False); (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    bank,e,_,_,_,y=load('train'); refs=read_jsonl(DATA/'gsm8k_train.jsonl'); assert all(a['id']==b['id'] for a,b in zip(bank,refs)); targets=np.array([signature(r['answer']) for r in refs]); raw,q,d,v=build(bank,e); hard=np.array([len(set(parse_answer(c['text']) for c in r['candidates'])-{None})>=2 for r in bank]); ids=np.flatnonzero(hard); np.random.default_rng(731).shuffle(ids); ho=ids[int(.8*len(ids)):]; tr=np.setdiff1d(np.arange(len(bank)),ho); records=[]
    for alpha in [0,.1,1]:
        for ep,b in fit(raw[tr],q[tr],d[tr],y[tr],v[tr],targets[tr],hard[tr],alpha,41,100):
            s=infer(raw[ho],q[ho],d[ho],b); p=np.where(v[ho],s,-1e20).argmax(1); r=dict(alpha=alpha,epoch=ep,holdout_accuracy=float(y[ho,p].mean()*100)); records.append(r); print(json.dumps(r),flush=True)
    best=max(records,key=lambda r:r['holdout_accuracy']); members=[fit(raw,q,d,y,v,targets,hard,best['alpha'],seed,best['epoch'])[-1][1] for seed in [41,42,43]]; b=members[0]; b['models']=[m['models'][0] for m in members]; (out/'model_bundle').mkdir(exist_ok=True); path=out/'model_bundle/model.pkl'; path.write_bytes(pickle.dumps(b,protocol=5))
    bv,ev,_,vv,_,yv=load('validation'); rv,qv,dv,vv=build(bv,ev); s=infer(rv,qv,dv,b); previous=pickle.loads(Path('runs_round5/development_bundle/model.pkl').read_bytes()); bs,_=previous_scores(bv,{'embeddings':ev},previous); hv=np.array([len(set(parse_answer(c['text']) for c in r['candidates'])-{None})>=2 for r in bv]); bc=yv[np.arange(len(yv)),np.where(vv,bs,-1e20).argmax(1)].astype(bool); results=[]
    for name,z in [('previous',bs),('reference_aux',s),('quarter_blend',.75*norm(bs,vv)+.25*norm(s,vv)),('half_blend',.5*norm(bs,vv)+.5*norm(s,vv))]:
        c=yv[np.arange(len(yv)),np.where(vv,z,-1e20).argmax(1)].astype(bool); r=dict(name=name,hard_correct=int(c[hv].sum()),hard_accuracy=float(c[hv].mean()*100),validation_accuracy=float(c.mean()*100),fixed=int((hv&c&~bc).sum()),regressed=int((hv&~c&bc).sum())); results.append(r); print(json.dumps(r),flush=True)
    (out/'holdout_search.json').write_text(json.dumps(records,indent=2)); (out/'result.json').write_text(json.dumps(dict(selected=best,comparison=results,seconds=time.perf_counter()-start,asset_bytes=path.stat().st_size,test_scored=False),indent=2))
if __name__=='__main__': main()
