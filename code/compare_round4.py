"""Development comparison only: never loads test inputs or references."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']: os.environ[k]='4'
import json,pickle
from pathlib import Path
import numpy as np
from train import load
from selector_refined import scores
from common import parse_answer
from trace_evidence import features
from train_role_text import documents,build
b,e,_,v,_,y=load('validation'); hard=np.array([len(set(parse_answer(c['text']) for c in r['candidates'])-{None})>=2 for r in b]); read=lambda p:pickle.loads(Path(p).read_bytes())
s0,_=scores(b,{'embeddings':e},read('runs_v2/final_bundle/model.pkl')); base=y[np.arange(len(y)),np.where(v,s0,-1e20).argmax(1)].astype(bool)
m=read('runs_trace/model_bundle/model.pkl'); f=features(b); x=np.concatenate([s0[:,:,None],(s0-s0.mean(1,keepdims=True))[:,:,None],f],2); trace=m['model'].predict(x.reshape(-1,x.shape[-1]),prediction_type='RawFormulaVal',thread_count=4).reshape(-1,4)
m=read('runs_role_text/model_bundle/model.pkl'); role=m['model'].decision_function(build(m['vector'],documents(b),s0)).reshape(-1,4); alternatives={'trace':trace,'role_text':role}
for kind in ['cnn','set']:
    path=Path('runs_local_'+kind+'/model_bundle/model.pt')
    if path.exists():
        import torch
        from train_local import make_x,infer
        from local_text import encode
        model=torch.load(path,map_location='cpu',weights_only=False); raw,valid=make_x(b,e); x=np.clip((raw-model['mean'])/model['scale'],-10,10).astype(np.float32); t,q=encode(b,model['vocab']); alternatives[kind]=infer(x,t,q,valid,model['config'],model['states'],'cpu')
def norm(s):
    mean=np.where(v,s,0).sum(1,keepdims=True)/v.sum(1,keepdims=True); z=s-mean; std=np.sqrt(np.where(v,z*z,0).sum(1,keepdims=True)/v.sum(1,keepdims=True)); return z/np.maximum(std,.5)
records=[]
def record(name,s):
    p=np.where(v,s,-1e20).argmax(1); c=y[np.arange(len(y)),p].astype(bool); d=c[hard].astype(float)-base[hard]; rng=np.random.default_rng(731); boot=np.array([rng.choice(d,len(d),replace=True).mean()*100 for _ in range(3000)])
    result=dict(name=name,correct_hard=int(c[hard].sum()),hard_n=int(hard.sum()),hard_accuracy=float(c[hard].mean()*100),validation_accuracy=float(c.mean()*100),fixed=int((hard&c&~base).sum()),regressed=int((hard&~c&base).sum()),paired_bootstrap_gain_ci95=np.quantile(boot,[.025,.975]).tolist()); records.append(result); print(json.dumps(result),flush=True)
record('frozen_v2',s0)
for name,s in alternatives.items():
    record(name,s)
    for w in [.25,.5]: record(name+'_blend_'+str(w),(1-w)*norm(s0)+w*norm(s))
out=Path('runs_round4'); out.mkdir(exist_ok=True); (out/'comparison.json').write_text(json.dumps(records,indent=2)); (out/'status.json').write_text(json.dumps(dict(test_scored=False,frozen_submission='V2',best_development=max(records,key=lambda r:r['hard_accuracy']),note='Repeated development evaluation; small gains are not independent evidence. OOF prior calibration holdout is not fully nested.'),indent=2))
