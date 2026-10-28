"""Audit-driven fixed calibration: same teacher family and normalized prior."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']: os.environ[k]='4'
import pickle,json,time
from pathlib import Path
import numpy as np
from scipy.sparse import hstack,csr_matrix
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from train import load
from features_v2 import features_v2
from selector_v2 import neural_scores
from selector_semantic_blend import documents,norm,scores as previous_scores
from selector_aligned import scores
from common import parse_answer
def main():
    start=time.perf_counter(); out=Path('runs_aligned'); out.mkdir(exist_ok=True); read=lambda p:pickle.loads(Path(p).read_bytes()); teachers=read('runs_stack/model_bundle/model.pkl')['models']; old=read('runs_role_text/model_bundle/model.pkl'); b,e,_,v,_,y=load('train'); f,_,_=features_v2(b,e); raw=np.concatenate([e,e-e.mean(1,keepdims=True),f],2); prior=np.zeros_like(y,dtype=float)
    for (_,ho),m in zip(KFold(5,shuffle=True,random_state=2026).split(b),teachers): prior[ho]=neural_scores(raw[ho],m)
    hard=np.array([len(set(parse_answer(c['text']) for c in r['candidates'])-{None})>=2 for r in b]); use=(hard[:,None]&v).ravel(); vector=old['vector']; x=hstack([vector.transform(documents(b)),csr_matrix(norm(prior,v).reshape(-1,1))],format='csr'); model=LogisticRegression(C=1,solver='liblinear',max_iter=500,random_state=41).fit(x[use],y.ravel()[use]); bundle=dict(teachers=teachers,vector=vector,model=model)
    (out/'model_bundle').mkdir(exist_ok=True); (out/'model_bundle/model.pkl').write_bytes(pickle.dumps(bundle,protocol=5))
    bv,ev,_,vv,_,yv=load('validation'); arrays={'embeddings':ev}; previous=read('runs_round5/development_bundle/model.pkl'); s0,_=previous_scores(bv,arrays,previous); hv=np.array([len(set(parse_answer(c['text']) for c in r['candidates'])-{None})>=2 for r in bv]); bc=yv[np.arange(len(yv)),np.where(vv,s0,-1e20).argmax(1)].astype(bool); records=[]
    for name,m in [('previous',None),('aligned',bundle),('aligned_semantic_25',dict(**bundle,semantic=previous['semantic'],weight=.25))]:
        s=s0 if m is None else scores(bv,arrays,m)[0]; c=yv[np.arange(len(yv)),np.where(vv,s,-1e20).argmax(1)].astype(bool); r=dict(name=name,hard_correct=int(c[hv].sum()),hard_accuracy=float(c[hv].mean()*100),validation_accuracy=float(c.mean()*100),fixed=int((hv&c&~bc).sum()),regressed=int((hv&~c&bc).sum()),asset_bytes=len(pickle.dumps(m,protocol=5)) if m else 9724630); records.append(r); print(json.dumps(r),flush=True)
        if m is not None and r['hard_correct']>int(bc[hv].sum()) and r['asset_bytes']<=16*1024**2:
            p=out/name; p.mkdir(exist_ok=True); (p/'model.pkl').write_bytes(pickle.dumps(m,protocol=5))
    result=dict(comparison=records,seconds=time.perf_counter()-start,C=1,hyperparameter_search=False,test_scored=False,note='Fixed prior/text vocabulary inherited from training; single-fold OOF vs averaged deployment teacher variance still differs.')
    (out/'result.json').write_text(json.dumps(result,indent=2))
if __name__=='__main__': main()
