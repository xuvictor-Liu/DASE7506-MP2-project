"""Train on training references only; select using validation only."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']: os.environ[k]='4'
import json,time,pickle
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostClassifier
from common import load_inputs,read_jsonl,DATA,parse_answer
from features import features,choose
def load(split):
    bank,arrays=load_inputs(split); refs=read_jsonl(DATA/f'gsm8k_{split}.jsonl')
    assert all(b['id']==r['id'] for b,r in zip(bank,refs))
    y=np.array([[parse_answer(c['text']) is not None and parse_answer(c['text'])==parse_answer(r['answer']) for c in b['candidates']] for b,r in zip(bank,refs)],dtype=int)
    f,valid,votes=features(bank)
    return bank,arrays['embeddings'],f,valid,votes,y
def main():
    start=time.perf_counter(); out=Path('runs'); out.mkdir(exist_ok=True)
    bt,et,ft,vt,at,yt=load('train'); bv,ev,fv,vv,av,yv=load('validation')
    hard=np.array([len(set(parse_answer(c['text']) for c in b['candidates'])-{None})>=2 for b in bv])
    records=[]; candidates=[]
    def record(name,scores,bundle=None):
        p=choose(scores,vv); correct=yv[np.arange(len(yv)),p]
        r=dict(name=name,validation_accuracy=float(correct.mean()*100),hard_accuracy=float(correct[hard].mean()*100),hard_n=int(hard.sum()),elapsed_seconds=time.perf_counter()-start)
        records.append(r); print(json.dumps(r),flush=True)
        if bundle is not None: candidates.append((r['hard_accuracy'],r['validation_accuracy'],bundle,scores))
        (out/'experiments.json').write_text(json.dumps(records,indent=2))
    record('first_valid',np.tile([4,3,2,1],(len(bv),1))); record('majority',av)
    train_groups=yt.min(axis=1)==0
    for mode in ['embedding','combined']:
        xt=np.concatenate([et,et-et.mean(1,keepdims=True)],2); xv=np.concatenate([ev,ev-ev.mean(1,keepdims=True)],2)
        if mode=='combined': xt=np.concatenate([xt,ft],2); xv=np.concatenate([xv,fv],2)
        scaler=StandardScaler().fit(xt[train_groups].reshape(-1,xt.shape[-1]))
        x=scaler.transform(xt[train_groups].reshape(-1,xt.shape[-1])); z=scaler.transform(xv.reshape(-1,xv.shape[-1]))
        for C in [.001,.01,.1,1]:
            model=LogisticRegression(C=C,max_iter=500,random_state=42).fit(x,yt[train_groups].ravel())
            scores=model.decision_function(z).reshape(-1,4)
            record(f'linear_{mode}_{C}',scores,dict(kind='linear',mode=mode,scaler=scaler,model=model))
    for depth in [4,6]:
        model=CatBoostClassifier(iterations=1200,depth=depth,learning_rate=.035,loss_function='Logloss',thread_count=4,random_seed=42,verbose=False,allow_writing_files=False,l2_leaf_reg=5)
        model.fit(ft[train_groups].reshape(-1,ft.shape[-1]),yt[train_groups].ravel())
        for nt in [300,600,1200]:
            m=model.copy(); m.shrink(nt)
            scores=m.predict(fv.reshape(-1,fv.shape[-1]),prediction_type='RawFormulaVal',thread_count=4).reshape(-1,4)
            record(f'structure_depth{depth}_{nt}',scores,dict(kind='tree',model=m))
    linear=max((x for x in candidates if x[2]['kind']=='linear'),key=lambda x:x[:2]); tree=max((x for x in candidates if x[2]['kind']=='tree'),key=lambda x:x[:2])
    for w in [.25,.5,.75]: record(f'ensemble_{w}',w*linear[3]+(1-w)*tree[3],dict(kind='ensemble',weight=w,linear=linear[2],tree=tree[2]))
    best=max(candidates,key=lambda x:x[:2]); bundle=out/'model_bundle'; bundle.mkdir(exist_ok=True)
    with (bundle/'model.pkl').open('wb') as f: pickle.dump(best[2],f,protocol=5)
    (out/'selection.json').write_text(json.dumps(dict(selection_metric='validation hard-subset accuracy; overall validation tie-break',hard_accuracy=best[0],validation_accuracy=best[1],training_seconds=time.perf_counter()-start),indent=2))
    print('Saved selected model',flush=True)
if __name__=='__main__': main()
