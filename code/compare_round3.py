"""Validation-only comparison of independently developed third-round methods."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']: os.environ[k]='4'
import json,pickle,time
from pathlib import Path
import numpy as np
from train import load
from common import parse_answer
from selector_refined import scores as base_scores
from selector_group import scores as group_scores
from selector_pair import scores as pair_scores
from selector_v2 import scores as v2_scores
from selector_kernel import scores as kernel_scores
from selector_interaction import build as interaction_build
from features_v2 import features_v2
from numeric_features import relations
from strict_arithmetic import features as strict_features
out=Path('runs_round3'); out.mkdir(exist_ok=True); start=time.perf_counter()
b,e,_,v,_,y=load('validation'); arrays={'embeddings':e}; hard=np.array([len(set(parse_answer(c['text']) for c in row['candidates'])-{None})>=2 for row in b])
read=lambda p:pickle.loads(Path(p).read_bytes())
base=read('runs_v2/final_bundle/model.pkl'); s0,_=base_scores(b,arrays,base)
models={}; scores={}
for name,folder,fn in [('group','runs_group',group_scores),('pair','runs_pair',pair_scores),('multitask','runs_multitask',v2_scores),('kernel','runs_kernel',kernel_scores)]:
    model=read(folder+'/model_bundle/model.pkl'); s,_=fn(b,arrays,model); models[name]=model; scores[name]=s
model=read('runs_interaction/model_bundle/model.pkl'); x,_=interaction_build(b,e,model['projection']); scores['interaction']=((x-model['mean'])/model['scale'])@model['coef']+model['intercept']; models['interaction']=model
model=read('runs_numeric/model_bundle/model.pkl'); f,_,_=features_v2(b,e); nf=relations(b); pe=(e-model['mean'])@model['components'].T; x=np.concatenate([f,nf,pe,pe-pe.mean(1,keepdims=True)],2); scores['numeric']=model['model'].predict(x.reshape(-1,x.shape[-1]),thread_count=4).reshape(-1,4); models['numeric']=model
model=read('runs_strict/model_bundle/model.pkl'); f=strict_features(b); x=np.concatenate([s0[:,:,None],(s0-s0.mean(1,keepdims=True))[:,:,None],f],2); scores['strict']=((x-model['mean'])/model['scale'])@model['coef']+model['intercept']; models['strict']=model
def norm(s):
    valid_s=np.where(v,s,0); mean=valid_s.sum(1,keepdims=True)/v.sum(1,keepdims=True); centered=s-mean; std=np.sqrt(np.where(v,centered**2,0).sum(1,keepdims=True)/v.sum(1,keepdims=True)); return centered/np.maximum(std,.5)
base_correct=y[np.arange(len(y)),np.where(v,s0,-1e20).argmax(1)]; records=[]; candidates=[]
def record(name,s,config):
    pred=np.where(v,s,-1e20).argmax(1); correct=y[np.arange(len(y)),pred]
    result=dict(name=name,hard_accuracy=float(correct[hard].mean()*100),validation_accuracy=float(correct.mean()*100),fixed=int((hard&correct.astype(bool)&~base_correct.astype(bool)).sum()),regressed=int((hard&~correct.astype(bool)&base_correct.astype(bool)).sum()))
    records.append(result); candidates.append((result['hard_accuracy'],config,s)); print(json.dumps(result),flush=True)
record('v2_frozen',s0,dict(name='base',weight=0))
for name,s in scores.items():
    record(name,s,dict(name=name,weight=1))
    for w in [.25,.5]: record('v2_'+name+'_'+str(w),(1-w)*norm(s0)+w*norm(s),dict(name=name,weight=w))
best=max(candidates,key=lambda t:t[0]); p=np.where(v,best[2],-1e20).argmax(1); d=(y[np.arange(len(y)),p]-base_correct)[hard].astype(float); rng=np.random.default_rng(731); boot=np.array([rng.choice(d,len(d),replace=True).mean()*100 for _ in range(5000)])
result=dict(selected=best[1],hard_accuracy=best[0],paired_bootstrap_gain_ci95=np.quantile(boot,[.025,.975]).tolist(),seconds=time.perf_counter()-start,test_scored=False,test_gate='Only consider a new official test after >=90% on validation disagreement subset; otherwise retain V2 as current frozen submission.')
(out/'comparison.json').write_text(json.dumps(records,indent=2)); (out/'selection.json').write_text(json.dumps(result,indent=2))
if best[1]['name']!='base':
    bundle=dict(base=base,auxiliary=models[best[1]['name']],config=best[1]); (out/'development_bundle').mkdir(exist_ok=True); (out/'development_bundle/model.pkl').write_bytes(pickle.dumps(bundle,protocol=5))
print(json.dumps(result),flush=True)
