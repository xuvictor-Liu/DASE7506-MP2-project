"""Final validation comparison; no test access."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']: os.environ[k]='4'
import json,pickle,hashlib
from pathlib import Path
import numpy as np
from train import load
from selector_refined import scores
from selector_stack import scores as stack_scores
from features import choose
from common import parse_answer
b,e,_,v,_,y=load('validation'); arrays={'embeddings':e}
path=Path('runs_v2/final_bundle/model.pkl'); bundle=pickle.loads(path.read_bytes()); stack=pickle.loads(Path('runs_stack/model_bundle/model.pkl').read_bytes())
s,_=scores(b,arrays,bundle); ss,_=stack_scores(b,arrays,stack)
hard=np.array([len(set(parse_answer(c['text']) for c in row['candidates'])-{None})>=2 for row in b]); records=[]; best=None
for w in [0,.25,.5,.75,1]:
    p=choose((1-w)*s+w*ss,v); c=y[np.arange(len(y)),p]
    r=dict(stack_weight=w,hard_accuracy=float(c[hard].mean()*100),validation_accuracy=float(c.mean()*100)); records.append(r); print(json.dumps(r))
    if best is None or r['hard_accuracy']>best['hard_accuracy']: best=r
bundle['stack']=stack if best['stack_weight'] else None; bundle['config']['stack_weight']=best['stack_weight']
path.write_bytes(pickle.dumps(bundle,protocol=5)); Path('runs_v2/final_comparison.json').write_text(json.dumps(records,indent=2))
files=['selector_refined.py','selector_stack.py','selector_v2.py','features_v2.py','features.py','arithmetic.py','correction_evidence.py',str(path)]
manifest=dict(selection=best,sha256={f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in files},asset_bytes=path.stat().st_size,protocol='dase7506-mp2-open-hard-verifier-v5',note='Second research round; selection uses train/validation only. V1 aggregate test score was already known; no test-answer inspection or test-driven selection in round two.')
assert path.stat().st_size<=16*2**20
Path('runs_v2/frozen_manifest.json').write_text(json.dumps(manifest,indent=2))
