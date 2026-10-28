"""Validation-only descriptive analysis; does not change the frozen model."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']: os.environ[k]='4'
import json,pickle
from pathlib import Path
import numpy as np
from common import load_inputs,read_jsonl,parse_answer,DATA
from features import features,choose
bank,arrays=load_inputs('validation'); refs=read_jsonl(DATA/'gsm8k_validation.jsonl')
f,valid,votes=features(bank); e=arrays['embeddings']; b=pickle.loads(Path('runs/model_bundle/model.pkl').read_bytes())
assert b['kind']=='neural'
x=np.clip((np.concatenate([e,e-e.mean(1,keepdims=True),f],2)-b['mean'])/b['scale'],-10,10)
s=sum((np.maximum(x@m['w1']+m['b1'],0)@m['w2']+m['b2']).reshape(-1,4) for m in b['models'])/len(b['models'])
p=choose(s,valid); base=choose(votes,valid)
y=np.array([[parse_answer(c['text']) is not None and parse_answer(c['text'])==parse_answer(r['answer']) for c in row['candidates']] for row,r in zip(bank,refs)])
hard=np.array([len(set(parse_answer(c['text']) for c in row['candidates'])-{None})>=2 for row in bank])
correct=y[np.arange(len(bank)),p]; bc=y[np.arange(len(bank)),base]
s=np.where(valid,s,-1e20); prob=np.exp(s-s.max(1,keepdims=True)); prob/=prob.sum(1,keepdims=True)
confidence=prob.max(1); bins=[]
for lo,hi in [(0,.5),(.5,.7),(.7,.9),(.9,1.00001)]:
    mask=hard&(confidence>=lo)&(confidence<hi)
    bins.append(dict(lower=lo,upper=min(hi,1),n=int(mask.sum()),accuracy=float(correct[mask].mean()*100) if mask.any() else None))
delta=(correct.astype(float)-bc.astype(float))[hard]; rng=np.random.default_rng(2026)
boot=np.array([rng.choice(delta,len(delta),replace=True).mean()*100 for _ in range(5000)])
errors=[]
for i in np.flatnonzero(hard&~correct)[:5]:
    errors.append(dict(question=bank[i]['question'],selected_text=bank[i]['candidates'][p[i]]['text'],reference=refs[i]['answer']))
result=dict(hard_n=int(hard.sum()),baseline_correct=int(bc[hard].sum()),model_correct=int(correct[hard].sum()),fixed=int((hard&correct&~bc).sum()),regressed=int((hard&~correct&bc).sum()),paired_bootstrap_gain_ci95=np.quantile(boot,[.025,.975]).tolist(),confidence_bins=bins,errors=errors)
Path('runs/validation_analysis.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k!='errors'},indent=2))
