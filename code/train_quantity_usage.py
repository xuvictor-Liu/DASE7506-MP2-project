"""Fixed regularized calibration using out-of-fold training expert scores."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']: os.environ[k]='4'
import pickle,json,time
from pathlib import Path
import numpy as np
import torch
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from train import load
from quantity_usage import features
from selector_semantic_blend import norm,scores as base_scores
from selector_v2 import neural_scores
from train_semantic import build
from train_local import make_x,infer
from local_text import encode
from common import parse_answer
def combine(bank,ss,sc,v):
    return np.concatenate([norm(ss,v)[:,:,None],norm(sc,v)[:,:,None],features(bank)],2)
def main():
    start=time.perf_counter(); out=Path('runs_quantity_usage'); out.mkdir(exist_ok=True)
    b,e,_,v,_,y=load('train'); oof=np.load('runs_gate/training_oof_scores.npz'); x=combine(b,oof['semantic'],oof['cnn'],v); hard=np.array([len(set(parse_answer(c['text']) for c in r['candidates'])-{None})>=2 for r in b]); use=(hard[:,None]&v).ravel()
    model=make_pipeline(StandardScaler(),LogisticRegression(C=.1,max_iter=1000,random_state=41)); model.fit(x.reshape(-1,x.shape[-1])[use],y.ravel()[use]); (out/'model_bundle').mkdir(exist_ok=True); (out/'model_bundle/calibrator.pkl').write_bytes(pickle.dumps(model,protocol=5))
    bv,ev,_,vv,_,yv=load('validation'); base=pickle.loads(Path('runs_round5/development_bundle/model.pkl').read_bytes()); sem=base['semantic']; xv,_=build(bv,ev,sem['vector'],sem['svd']); ss=neural_scores(xv,sem['net']); cnn=torch.load('runs_local_cnn/model_bundle/model.pt',map_location='cpu',weights_only=False); raw,vv=make_x(bv,ev); xx=np.clip((raw-cnn['mean'])/cnn['scale'],-10,10).astype(np.float32); t,q=encode(bv,cnn['vocab']); sc=infer(xx,t,q,vv,cnn['config'],cnn['states'],'cpu'); x=combine(bv,ss,sc,vv); s=model.decision_function(x.reshape(-1,x.shape[-1])).reshape(-1,4); bs,_=base_scores(bv,{'embeddings':ev},base)
    hv=np.array([len(set(parse_answer(c['text']) for c in r['candidates'])-{None})>=2 for r in bv]); bc=yv[np.arange(len(yv)),np.where(vv,bs,-1e20).argmax(1)].astype(bool); records=[]
    for name,z in [('previous_best',bs),('quantity_calibrator',s),('quarter_blend',.75*norm(bs,vv)+.25*norm(s,vv)),('half_blend',.5*norm(bs,vv)+.5*norm(s,vv))]:
        c=yv[np.arange(len(yv)),np.where(vv,z,-1e20).argmax(1)].astype(bool); r=dict(name=name,hard_correct=int(c[hv].sum()),hard_accuracy=float(c[hv].mean()*100),validation_accuracy=float(c.mean()*100),fixed=int((hv&c&~bc).sum()),regressed=int((hv&~c&bc).sum())); records.append(r); print(json.dumps(r),flush=True)
    result=dict(comparison=records,seconds=time.perf_counter()-start,test_scored=False,configuration='Fixed C=0.1; no hyperparameter search; features designed after inspecting training-only mistakes.')
    (out/'result.json').write_text(json.dumps(result,indent=2))
if __name__=='__main__': main()
