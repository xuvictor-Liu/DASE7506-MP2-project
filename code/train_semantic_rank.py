"""Hard-question-only pairwise ranking with role-text semantic features."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']: os.environ[k]='4'
import pickle,json,time
from pathlib import Path
import numpy as np
from train import load
from train_semantic import project_fit,build
from train_role_text import documents
from train_robust import fit
from selector_v2 import neural_scores
from common import parse_answer
def main():
    start=time.perf_counter(); out=Path('runs_semantic_rank'); out.mkdir(exist_ok=True)
    cfg=json.loads(Path('runs_robust/selection.json').read_text())['selected']; (out/'protocol.json').write_text(json.dumps(dict(configuration_from_training_holdout=cfg,final_seeds=[41,42,43],extra_hyperparameter_search=False,test_scored=False),indent=2))
    b,e,_,v,_,y=load('train'); hard=np.array([len(set(parse_answer(c['text']) for c in r['candidates'])-{None})>=2 for r in b]); ids=np.flatnonzero(hard); np.random.default_rng(731).shuffle(ids); ho=ids[int(.8*len(ids)):]; tr=np.setdiff1d(np.arange(len(b)),ho); tr_hard=tr[hard[tr]]
    vector,svd=project_fit(documents([b[i] for i in tr])); x,v=build(b,e,vector,svd); m=fit(x[tr_hard],y[tr_hard],v[tr_hard],cfg['rho'],41,cfg['epoch'])[-1][1]; s=neural_scores(x[ho],m); p=np.where(v[ho],s,-1e20).argmax(1); holdout=float(y[ho,p].mean()*100); print('HOLDOUT',holdout,flush=True)
    vector,svd=project_fit(documents(b)); x,v=build(b,e,vector,svd); members=[fit(x[ids],y[ids],v[ids],cfg['rho'],seed,cfg['epoch'])[-1][1] for seed in [41,42,43]]; net=dict(kind='neural_v2',mean=members[0]['mean'],scale=members[0]['scale'],models=[m['models'][0] for m in members]); bundle=dict(vector=vector,svd=svd,net=net)
    (out/'model_bundle').mkdir(exist_ok=True); path=out/'model_bundle/model.pkl'; path.write_bytes(pickle.dumps(bundle,protocol=5))
    bv,ev,_,vv,_,yv=load('validation'); xv,vv=build(bv,ev,vector,svd); s=neural_scores(xv,net); p=np.where(vv,s,-1e20).argmax(1); c=yv[np.arange(len(yv)),p]; hv=np.array([len(set(parse_answer(cc['text']) for cc in r['candidates'])-{None})>=2 for r in bv]); result=dict(config=cfg,holdout_accuracy=holdout,hard_correct=int(c[hv].sum()),hard_accuracy=float(c[hv].mean()*100),validation_accuracy=float(c.mean()*100),seconds=time.perf_counter()-start,asset_bytes=path.stat().st_size,test_scored=False)
    (out/'selection.json').write_text(json.dumps(result,indent=2)); print(json.dumps(result),flush=True)
if __name__=='__main__': main()
