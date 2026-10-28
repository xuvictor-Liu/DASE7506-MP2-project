"""Fixed development comparisons; preserve previous best on ties."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']: os.environ[k]='4'
import pickle,json
from pathlib import Path
import numpy as np
from train import load
from common import parse_answer
from selector_semantic_blend import scores as development_scores,norm
from selector_refined import scores as v2_scores
from selector_round7 import auxiliary_scores
def main():
    read=lambda p:pickle.loads(Path(p).read_bytes()); b,e,_,v,_,y=load('validation'); arrays={'embeddings':e}; hard=np.array([len(set(parse_answer(c['text']) for c in r['candidates'])-{None})>=2 for r in b]); dev=read('runs_round5/development_bundle/model.pkl'); v2=read('runs_v2/final_bundle/model.pkl'); sd,_=development_scores(b,arrays,dev); s2,_=v2_scores(b,arrays,v2); correct_base=y[np.arange(len(y)),np.where(v,sd,-1e20).argmax(1)].astype(bool); records=[]; best=None
    def record(name,s,model=None):
        nonlocal best
        p=np.where(v,s,-1e20).argmax(1); c=y[np.arange(len(y)),p].astype(bool); d=c[hard].astype(float)-correct_base[hard]; rng=np.random.default_rng(731); boot=np.array([rng.choice(d,len(d),replace=True).mean()*100 for _ in range(3000)]); size=len(pickle.dumps(model,protocol=5)) if model is not None else 9724630
        r=dict(name=name,hard_correct=int(c[hard].sum()),hard_accuracy=float(c[hard].mean()*100),validation_accuracy=float(c.mean()*100),fixed=int((hard&c&~correct_base).sum()),regressed=int((hard&~c&correct_base).sum()),gain_ci95=np.quantile(boot,[.025,.975]).tolist(),asset_bytes=size); records.append(r); print(json.dumps(r),flush=True)
        if size<=16*1024**2 and (best is None or r['hard_correct']>best[0]['hard_correct']): best=(r,model)
    record('previous_development',sd)
    for kind,folder in [('robust','runs_robust'),('semantic_rank','runs_semantic_rank')]:
        m=read(folder+'/model_bundle/model.pkl'); s,_=auxiliary_scores(b,arrays,m,kind)
        record(kind,s,dict(auxiliary=m,auxiliary_kind=kind,weight=1))
        for base_kind,base,bs in [('v2',v2,s2),('development',dev,sd)]:
            for w in [.25,.5]: record(base_kind+'_'+kind+'_'+str(w),(1-w)*norm(bs,v)+w*norm(s,v),dict(base=base,base_kind=base_kind,auxiliary=m,auxiliary_kind=kind,weight=w))
    out=Path('runs_round7'); out.mkdir(exist_ok=True); (out/'comparison.json').write_text(json.dumps(records,indent=2)); result=dict(selected=best[0],previous_development_retained=best[1] is None,test_scored=False,official_test_accuracy=84.36830835117773)
    if best[1] is not None:
        (out/'development_bundle').mkdir(exist_ok=True); (out/'development_bundle/model.pkl').write_bytes(pickle.dumps(best[1],protocol=5))
    (out/'selection.json').write_text(json.dumps(result,indent=2)); print(json.dumps(result),flush=True)
if __name__=='__main__': main()
