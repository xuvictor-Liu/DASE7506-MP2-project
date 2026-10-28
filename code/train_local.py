"""GPU training from scratch; architecture selection on training-only holdout."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']: os.environ[k]='4'
import json,time,copy,random,argparse
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from neural_local import LocalVerifier
from local_text import vocabulary,encode
from train import load
from features_v2 import features_v2
from strict_arithmetic import features as arithmetic_features
from common import parse_answer
torch.set_num_threads(4); torch.set_num_interop_threads(1)
def make_x(bank,e):
    f,v,_=features_v2(bank,e); af=arithmetic_features(bank)
    return np.concatenate([e,e-e.mean(1,keepdims=True),f,af],2),v
def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
def fit(x,t,q,valid,y,hard,config,seed,epochs,device):
    seed_all(seed); model=LocalVerifier(x.shape[-1],config['vocab_size'],config['kind'],config['dropout']).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=.0007,weight_decay=.01)
    tensors=[torch.as_tensor(a,device=device) for a in [x,t,q,valid,y.astype(np.float32),hard]]
    x,t,q,valid,y,hard=tensors; target=y/y.sum(1,keepdim=True); snapshots=[]; rng=np.random.default_rng(seed)
    for ep in range(1,epochs+1):
        model.train(); losses=[]
        for ids in np.array_split(rng.permutation(len(x)),max(1,len(x)//32)):
            ix=torch.as_tensor(ids,device=device); opt.zero_grad(set_to_none=True)
            scores=model(x[ix],t[ix],q[ix],valid[ix]); masked=scores.masked_fill(~valid[ix],-1e4)
            ranking=-(target[ix]*F.log_softmax(masked,1)).sum(1)
            bce=(F.binary_cross_entropy_with_logits(scores,y[ix],reduction='none')*valid[ix]).sum(1)/valid[ix].sum(1)
            loss=(ranking*hard[ix]).mean()+.1*bce.mean(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),2); opt.step(); losses.append(loss.item())
        if ep in [5,15,30]:
            snapshots.append((ep,{k:v.detach().cpu().clone() for k,v in model.state_dict().items()}))
            print('TRAIN',config['kind'],'seed',seed,'epoch',ep,'loss',float(np.mean(losses)),flush=True)
    return snapshots
def infer(x,t,q,valid,config,states,device):
    model=LocalVerifier(x.shape[-1],config['vocab_size'],config['kind'],config['dropout']).to(device); result=np.zeros((len(x),4))
    with torch.no_grad():
      for state in states:
        model.load_state_dict(state); model.eval(); outputs=[]
        for start in range(0,len(x),32):
            args=[torch.as_tensor(a[start:start+32],device=device) for a in [x,t,q,valid]]; outputs.append(model(*args).cpu().numpy())
        result+=np.concatenate(outputs)/len(states)
    return result
def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--kind',choices=['cnn','set'],required=True); args=parser.parse_args()
    out=Path('runs_local_'+args.kind); out.mkdir(exist_ok=True); start=time.perf_counter(); device='cuda' if torch.cuda.is_available() else 'cpu'
    print('DEVICE',device,torch.cuda.get_device_name(0) if device=='cuda' else '',flush=True)
    b,e,_,_,_,y=load('train'); raw,valid=make_x(b,e); hard=np.array([len(set(parse_answer(c['text']) for c in row['candidates'])-{None})>=2 for row in b]); ids=np.flatnonzero(hard); np.random.default_rng(731).shuffle(ids); ho=ids[int(.8*len(ids)):]; tr=np.setdiff1d(np.arange(len(b)),ho)
    vocab=vocabulary([b[i] for i in tr]); text,qs=encode(b,vocab); mean=raw[tr].mean((0,1)); scale=raw[tr].std((0,1)); scale[scale<1e-5]=1; x=np.clip((raw-mean)/scale,-10,10).astype(np.float32)
    records=[]; best=None
    for dropout in [.15,.4]:
        config=dict(kind=args.kind,dropout=dropout,vocab_size=max(vocab.values())+1)
        for ep,state in fit(x[tr],text[tr],qs[tr],valid[tr],y[tr],hard[tr],config,41,30,device):
            s=infer(x[ho],text[ho],qs[ho],valid[ho],config,[state],device); p=np.where(valid[ho],s,-1e20).argmax(1); acc=float(y[ho,p].mean()*100)
            r=dict(dropout=dropout,epoch=ep,holdout_accuracy=acc,seconds=time.perf_counter()-start); records.append(r); print(json.dumps(r),flush=True)
            if best is None or acc>best['holdout_accuracy']: best=r
    (out/'holdout_search.json').write_text(json.dumps(records,indent=2))
    vocab=vocabulary(b); text,qs=encode(b,vocab); mean=raw.mean((0,1)); scale=raw.std((0,1)); scale[scale<1e-5]=1; x=np.clip((raw-mean)/scale,-10,10).astype(np.float32); config=dict(kind=args.kind,dropout=best['dropout'],vocab_size=max(vocab.values())+1)
    states=[fit(x,text,qs,valid,y,hard,config,seed,best['epoch'],device)[-1][1] for seed in [41,42]]
    bv,ev,_,_,_,yv=load('validation'); rawv,vv=make_x(bv,ev); xv=np.clip((rawv-mean)/scale,-10,10).astype(np.float32); tv,qv=encode(bv,vocab); s=infer(xv,tv,qv,vv,config,states,device); p=np.where(vv,s,-1e20).argmax(1); c=yv[np.arange(len(yv)),p]; hv=np.array([len(set(parse_answer(cc['text']) for cc in row['candidates'])-{None})>=2 for row in bv])
    result=dict(selected=best,hard_accuracy=float(c[hv].mean()*100),validation_accuracy=float(c.mean()*100),seconds=time.perf_counter()-start,device=device)
    (out/'model_bundle').mkdir(exist_ok=True); torch.save(dict(config=config,vocab=vocab,mean=mean,scale=scale,states=states),out/'model_bundle/model.pt'); result['asset_bytes']=(out/'model_bundle/model.pt').stat().st_size
    (out/'selection.json').write_text(json.dumps(result,indent=2)); print(json.dumps(result),flush=True)
if __name__=='__main__': main()
