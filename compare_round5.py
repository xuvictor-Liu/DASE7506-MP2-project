"""Validation-only comparison of new models and prespecified simple blends."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']: os.environ[k]='4'
import json,pickle
from pathlib import Path
import numpy as np
from train import load
from selector_refined import scores
from selector_attention import scores as attention_scores
from selector_v2 import neural_scores
from train_semantic import build
from train_role_text import documents,build as role_build
from common import parse_answer
def main():
    b,e,_,v,_,y=load('validation'); hard=np.array([len(set(parse_answer(c['text']) for c in r['candidates'])-{None})>=2 for r in b]); read=lambda p:pickle.loads(Path(p).read_bytes()); base=read('runs_v2/final_bundle/model.pkl'); s0,_=scores(b,{'embeddings':e},base)
    m=read('runs_semantic/model_bundle/model.pkl'); x,_=build(b,e,m['vector'],m['svd']); semantic=neural_scores(x,m['net']); m=read('runs_attention/model_bundle/model.pkl'); attention,_=attention_scores(b,{'embeddings':e},m); m=read('runs_role_text/model_bundle/model.pkl'); role=m['model'].decision_function(role_build(m['vector'],documents(b),s0)).reshape(-1,4)
    alternatives={'semantic':semantic,'attention':attention}; bases={'v2':s0,'role_text':role}; bc=y[np.arange(len(y)),np.where(v,s0,-1e20).argmax(1)].astype(bool); records=[]
    def norm(s):
        mean=np.where(v,s,0).sum(1,keepdims=True)/v.sum(1,keepdims=True); z=s-mean; std=np.sqrt(np.where(v,z*z,0).sum(1,keepdims=True)/v.sum(1,keepdims=True)); return z/np.maximum(std,.5)
    def record(name,s):
        p=np.where(v,s,-1e20).argmax(1); c=y[np.arange(len(y)),p].astype(bool); d=c[hard].astype(float)-bc[hard]; rng=np.random.default_rng(731); boot=np.array([rng.choice(d,len(d),replace=True).mean()*100 for _ in range(3000)]); result=dict(name=name,correct_hard=int(c[hard].sum()),hard_accuracy=float(c[hard].mean()*100),validation_accuracy=float(c.mean()*100),fixed=int((hard&c&~bc).sum()),regressed=int((hard&~c&bc).sum()),paired_bootstrap_gain_ci95=np.quantile(boot,[.025,.975]).tolist()); records.append(result); print(json.dumps(result),flush=True)
    for name,s in bases.items(): record(name,s)
    for name,s in alternatives.items():
        record(name,s)
        for bn,bs in bases.items():
            for w in [.25,.5]: record(bn+'_'+name+'_'+str(w),(1-w)*norm(bs)+w*norm(s))
    for kind in ['set','cnn']:
        path=Path('runs_local_'+kind+'/model_bundle/model.pt')
        if path.exists():
            import torch
            from train_local import make_x,infer
            from local_text import encode
            m=torch.load(path,map_location='cpu',weights_only=False); raw,vv=make_x(b,e); x=np.clip((raw-m['mean'])/m['scale'],-10,10).astype(np.float32); t,q=encode(b,m['vocab']); s=infer(x,t,q,vv,m['config'],m['states'],'cpu'); record(kind,s)
            for bn,bs in bases.items():
                for w in [.25,.5]: record(bn+'_'+kind+'_'+str(w),(1-w)*norm(bs)+w*norm(s))
    out=Path('runs_round5'); out.mkdir(exist_ok=True); (out/'comparison.json').write_text(json.dumps(records,indent=2)); (out/'status.json').write_text(json.dumps(dict(test_scored=False,best_development=max(records,key=lambda r:r['hard_accuracy']),frozen_submission='V2',note='Repeated validation development, not an unbiased estimate. No official test rerun.'),indent=2))
if __name__=='__main__': main()
