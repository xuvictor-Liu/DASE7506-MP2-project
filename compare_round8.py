import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']: os.environ[k]='4'
import json,pickle
from pathlib import Path
import numpy as np
from train import load
from common import parse_answer
from selector_context import scores as context_scores
from selector_semantic_blend import scores as development_scores,norm
from selector_refined import scores as v2_scores
b,e,_,v,_,y=load('validation'); read=lambda p:pickle.loads(Path(p).read_bytes()); context=read('runs_context/model_bundle/model.pkl'); dev=read('runs_round5/development_bundle/model.pkl'); old=read('runs_v2/final_bundle/model.pkl'); a={'embeddings':e}; sc,_=context_scores(b,a,context); sd,_=development_scores(b,a,dev); so,_=v2_scores(b,a,old); hard=np.array([len(set(parse_answer(c['text']) for c in r['candidates'])-{None})>=2 for r in b]); bc=y[np.arange(len(y)),np.where(v,sd,-1e20).argmax(1)].astype(bool); records=[]
def record(name,s):
    c=y[np.arange(len(y)),np.where(v,s,-1e20).argmax(1)].astype(bool); r=dict(name=name,hard_correct=int(c[hard].sum()),hard_accuracy=float(c[hard].mean()*100),validation_accuracy=float(c.mean()*100),fixed=int((hard&c&~bc).sum()),regressed=int((hard&~c&bc).sum())); records.append(r); print(json.dumps(r),flush=True)
record('previous_development',sd); record('context',sc)
for name,s in [('v2',so),('development',sd)]:
    for w in [.25,.5]: record(name+'_context_'+str(w),(1-w)*norm(s,v)+w*norm(sc,v))
out=Path('runs_round8'); out.mkdir(exist_ok=True); (out/'comparison.json').write_text(json.dumps(records,indent=2)); (out/'status.json').write_text(json.dumps(dict(best=max(records,key=lambda r:r['hard_correct']),test_scored=False,target_test_correct=407,target_test_n=467),indent=2))
