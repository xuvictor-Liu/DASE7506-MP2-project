import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']: os.environ[k]='4'
import json,pickle
from pathlib import Path
import numpy as np
from train import load
from correction_evidence import evidence
from selector_refined import scores
from features import choose
from common import parse_answer
bundle=pickle.loads(Path('runs_v2/refined_bundle/model.pkl').read_bytes())
b,e,_,v,_,y=load('train'); et=evidence(b); mask=et>0
print('Train correction support',int(mask.sum()),'correct',int(y[mask].sum()),flush=True)
b,e,_,v,_,y=load('validation'); ev=evidence(b); s,_=scores(b,{'embeddings':e},bundle)
hard=np.array([len(set(parse_answer(c['text']) for c in row['candidates'])-{None})>=2 for row in b]); records=[]; best=None
for w in [0,.5,1,2,4]:
    p=choose(s+w*ev,v); c=y[np.arange(len(y)),p]
    r=dict(correction=w,hard_accuracy=float(c[hard].mean()*100),validation_accuracy=float(c.mean()*100)); records.append(r); print(json.dumps(r),flush=True)
    if best is None or r['hard_accuracy']>best['hard_accuracy']: best=r
bundle['config']['correction']=best['correction']; out=Path('runs_v2/final_bundle'); out.mkdir(exist_ok=True)
(out/'model.pkl').write_bytes(pickle.dumps(bundle,protocol=5)); Path('runs_v2/correction_experiments.json').write_text(json.dumps(records,indent=2))
