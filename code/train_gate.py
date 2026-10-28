"""One final fixed experiment: two-fold cross-fitted expert reliability gate."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']: os.environ[k]='4'
import json,pickle,time
from pathlib import Path
import numpy as np
import torch
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from train import load
from train_semantic import project_fit,build as semantic_build
from train_role_text import documents
from train_multitask import fit as fit_semantic
from selector_v2 import neural_scores
from train_local import make_x,fit as fit_cnn,infer
from local_text import vocabulary,encode
from selector_semantic_blend import norm,scores as baseline_scores
from gate_features import build as gate_features
from common import parse_answer
def main():
    start=time.perf_counter(); out=Path('runs_gate'); out.mkdir(exist_ok=True)
    protocol=dict(folds=2,split_seed=1861,semantic_epochs=50,semantic_alpha=.1,cnn_epochs=30,cnn_dropout=.15,gate_C=.1,anchored_gate_weight=.25,hyperparameter_search=False,validation_evaluations='one final comparison',test_scored=False)
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    b,e,_,v,_,y=load('train'); hard=np.array([len(set(parse_answer(c['text']) for c in r['candidates'])-{None})>=2 for r in b]); raw,v=make_x(b,e); ss=np.zeros_like(y,dtype=float); sc=np.zeros_like(y,dtype=float)
    for fold,(tr,ho) in enumerate(KFold(2,shuffle=True,random_state=1861).split(b)):
        bt=[b[i] for i in tr]; bh=[b[i] for i in ho]; vector,svd=project_fit(documents(bt)); xt,_=semantic_build(bt,e[tr],vector,svd); xh,_=semantic_build(bh,e[ho],vector,svd)
        net=fit_semantic(xt,y[tr],v[tr],hard[tr],.1,41+fold,50)[-1][1]; ss[ho]=neural_scores(xh,net)
        vocab=vocabulary(bt); t,q=encode(bt,vocab); th,qh=encode(bh,vocab); mean=raw[tr].mean((0,1)); scale=raw[tr].std((0,1)); scale[scale<1e-5]=1
        xt=np.clip((raw[tr]-mean)/scale,-10,10).astype(np.float32); xh=np.clip((raw[ho]-mean)/scale,-10,10).astype(np.float32); cfg=dict(kind='cnn',dropout=.15,vocab_size=max(vocab.values())+1)
        state=fit_cnn(xt,t,q,v[tr],y[tr],hard[tr],cfg,41+fold,30,'cpu')[-1][1]; sc[ho]=infer(xh,th,qh,v[ho],cfg,[state],'cpu')
        print(json.dumps(dict(fold=fold,completed_questions=len(ho),seconds=time.perf_counter()-start)),flush=True)
    np.savez_compressed(out/'training_oof_scores.npz',semantic=ss,cnn=sc)
    ps=np.where(v,ss,-1e20).argmax(1); pc=np.where(v,sc,-1e20).argmax(1); cs=y[np.arange(len(y)),ps].astype(bool); cc=y[np.arange(len(y)),pc].astype(bool); use=hard&(cs!=cc)
    gate=make_pipeline(StandardScaler(),LogisticRegression(C=.1,max_iter=1000,random_state=41)); gate.fit(gate_features(b,ss,sc,v)[use],cc[use].astype(int))
    diagnostics=dict(n_gate_questions=int(use.sum()),cnn_preferred=int(cc[use].sum()),semantic_oof_hard=float(cs[hard].mean()*100),cnn_oof_hard=float(cc[hard].mean()*100),either_expert_oof_hard=float((cs|cc)[hard].mean()*100),note='Gate fitted on cross-fitted training predictions; no gate hyperparameter search. Final expert ensembles were previously fitted on all training data.')
    (out/'training_diagnostics.json').write_text(json.dumps(diagnostics,indent=2)); print(json.dumps(diagnostics),flush=True)
    (out/'model_bundle').mkdir(exist_ok=True); (out/'model_bundle/gate.pkl').write_bytes(pickle.dumps(gate,protocol=5))
    # First and only access to official validation in this script.
    bv,ev,_,vv,_,yv=load('validation'); baseline=pickle.loads(Path('runs_round5/development_bundle/model.pkl').read_bytes()); sem=baseline['semantic']; xv,_=semantic_build(bv,ev,sem['vector'],sem['svd']); sv=neural_scores(xv,sem['net'])
    cnn=torch.load('runs_local_cnn/model_bundle/model.pt',map_location='cpu',weights_only=False); rv,vv=make_x(bv,ev); xv=np.clip((rv-cnn['mean'])/cnn['scale'],-10,10).astype(np.float32); tv,qv=encode(bv,cnn['vocab']); cv=infer(xv,tv,qv,vv,cnn['config'],cnn['states'],'cpu')
    probability=gate.predict_proba(gate_features(bv,sv,cv,vv))[:,1,None]; gated=(1-probability)*norm(sv,vv)+probability*norm(cv,vv); bs,_=baseline_scores(bv,{'embeddings':ev},baseline); anchored=.75*norm(bs,vv)+.25*norm(gated,vv)
    hv=np.array([len(set(parse_answer(c['text']) for c in r['candidates'])-{None})>=2 for r in bv]); bc=yv[np.arange(len(yv)),np.where(vv,bs,-1e20).argmax(1)].astype(bool); records=[]
    for name,s in [('previous_best',bs),('learned_gate',gated),('anchored_gate',anchored)]:
        p=np.where(vv,s,-1e20).argmax(1); correct=yv[np.arange(len(yv)),p].astype(bool); result=dict(name=name,hard_correct=int(correct[hv].sum()),hard_n=int(hv.sum()),hard_accuracy=float(correct[hv].mean()*100),validation_accuracy=float(correct.mean()*100),fixed=int((hv&correct&~bc).sum()),regressed=int((hv&~correct&bc).sum())); records.append(result); print(json.dumps(result),flush=True)
    result=dict(comparison=records,seconds=time.perf_counter()-start,test_scored=False,previous_best_retained=max(r['hard_correct'] for r in records)<=int(bc[hv].sum()),stop_after_this_experiment=True)
    (out/'result.json').write_text(json.dumps(result,indent=2))
if __name__=='__main__': main()
