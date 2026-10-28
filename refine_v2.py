"""Predefined validation-only fusion and arithmetic-penalty comparison."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']: os.environ[k]='4'
import json,pickle
from pathlib import Path
import numpy as np
from train import load
from common import parse_answer
from features import choose
from features_v2 import equations
from selector_v2 import scores

bt,e,f,valid,votes,y=load('validation')
hard=np.array([len(set(parse_answer(c['text']) for c in b['candidates'])-{None})>=2 for b in bt])
old=pickle.loads(Path('runs/model_bundle/model.pkl').read_bytes())
x=np.concatenate([e,e-e.mean(1,keepdims=True),f],2); x=np.clip((x-old['mean'])/old['scale'],-10,10)
old_s=sum((np.maximum(x@m['w1']+m['b1'],0)@m['w2']+m['b2']).reshape(-1,4) for m in old['models'])/len(old['models'])
new=pickle.loads(Path('runs_v2/model_bundle/model.pkl').read_bytes()); new_s,_=scores(bt,{'embeddings':e},new)
projected=pickle.loads(Path('runs_projected/model_bundle/model.pkl').read_bytes()); projected_s,_=scores(bt,{'embeddings':e},projected)
preliminary=[('full',new,new_s),('projected',projected,projected_s),('full_projected_average',dict(kind='blend_v2',members=[new,projected],weights=[.5,.5]),.5*(new_s+projected_s))]
preliminary_records=[]
for name,b,s in preliminary:
    correct=y[np.arange(len(y)),choose(s,valid)]
    preliminary_records.append(dict(name=name,hard_accuracy=float(correct[hard].mean()*100),validation_accuracy=float(correct.mean()*100)))
index=max(range(len(preliminary)),key=lambda i:preliminary_records[i]['hard_accuracy']); new=preliminary[index][1]; new_s=preliminary[index][2]
Path('runs_v2/preliminary_refinement.json').write_text(json.dumps(preliminary_records,indent=2))
text=np.load('runs_text/validation_scores.npy'); textmodel=pickle.loads(Path('runs_text/model.pkl').read_bytes())
bad=np.array([[equations(c['text'])[0][1]>0 for c in b['candidates']] for b in bt])
answers=[[parse_answer(c['text']) for c in b['candidates']] for b in bt]
def aggregate(s):
    result=s.copy()
    for i,aa in enumerate(answers):
        for j,a in enumerate(aa):
            if a is None: continue
            js=[k for k in range(4) if aa[k]==a]; v=s[i,js]; m=v.max()
            result[i,j]=m+np.log(np.exp(v-m).sum())
    return result
records=[]; options=[]
def record(name,s,cfg):
    p=choose(s,valid); correct=y[np.arange(len(y)),p]
    r=dict(name=name,hard_accuracy=float(correct[hard].mean()*100),validation_accuracy=float(correct.mean()*100))
    records.append(r); options.append((r['hard_accuracy'],r['validation_accuracy'],cfg,s)); print(json.dumps(r),flush=True)
for w in [0,.25,.5,.75,1]:
    record(f'old_new_{w}',(1-w)*old_s+w*new_s,dict(new_weight=w,text_weight=0,penalty=0,aggregate=False))
best=max(options,key=lambda x:x[:2]); base=best[3]; cfg=best[2]
for w in [.1,.25,.5]:
    record(f'text_weight_{w}',base+w*text,dict(cfg,text_weight=w))
best=max(options,key=lambda x:x[:2]); base=best[3]; cfg=best[2]
for penalty in [.25,.5,1,2]:
    record(f'arithmetic_penalty_{penalty}',base-penalty*bad,dict(cfg,penalty=penalty))
    record(f'answer_aggregation_penalty_{penalty}',aggregate(base-penalty*bad),dict(cfg,penalty=penalty,aggregate=True))
record('answer_aggregation',aggregate(base),dict(cfg,aggregate=True))
best=max(options,key=lambda x:x[:2]); cfg=best[2]
bundle=dict(config=cfg,old=old if cfg['new_weight']<1 else None,new=new if cfg['new_weight']>0 else None,text=textmodel if cfg['text_weight']>0 else None)
out=Path('runs_v2/refined_bundle'); out.mkdir(exist_ok=True)
(out/'model.pkl').write_bytes(pickle.dumps(bundle,protocol=5))
Path('runs_v2/refinement.json').write_text(json.dumps(records,indent=2))
Path('runs_v2/refined_selection.json').write_text(json.dumps(dict(config=cfg,hard_accuracy=best[0],validation_accuracy=best[1],asset_bytes=(out/'model.pkl').stat().st_size),indent=2))
