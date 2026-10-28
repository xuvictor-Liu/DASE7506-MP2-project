"""Arithmetic template classifier with a training-only question holdout."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']: os.environ[k]='4'
import json,pickle,time
from pathlib import Path
from collections import Counter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from common import read_jsonl,DATA,load_inputs,parse_answer
from train import load
from programs import question,extract,program_support
out=Path('runs_programs'); out.mkdir(exist_ok=True); start=time.perf_counter()
b,e,_,v,_,y=load('train'); refs=read_jsonl(DATA/'gsm8k_train.jsonl'); templates=[extract(r['question'],r['answer']) for r in refs]
hard=np.array([len(set(parse_answer(c['text']) for c in row['candidates'])-{None})>=2 for row in b]); ids=np.flatnonzero(hard); np.random.default_rng(731).shuffle(ids); ho=ids[int(.8*len(ids)):]; tr=np.setdiff1d(np.arange(len(b)),ho)
def fit(ids):
    counts=Counter(templates[i] for i in ids if templates[i] is not None); allowed={p for p,n in counts.most_common(96) if n>=3}; good=[i for i in ids if templates[i] in allowed]
    vec=TfidfVectorizer(ngram_range=(1,2),max_features=7000,min_df=2,sublinear_tf=True,dtype=np.float32); x=vec.fit_transform([question(b[i]['question'])[0] for i in good])
    model=LogisticRegression(C=5,max_iter=400).fit(x,[templates[i] for i in good])
    return dict(vectorizer=vec,model=model),len(good),len(allowed)
model,count,classes=fit(tr); hs,hc=program_support([b[i] for i in ho],model); pred=hs.argmax(1); covered=hs.max(1)>0; correct=y[ho,pred]
diagnostics=[]
for threshold in [.1,.25,.5,.75]:
    use=(hc>=threshold)&covered; diagnostics.append(dict(threshold=threshold,n=int(use.sum()),accuracy=float(correct[use].mean()*100) if use.any() else None))
print('extracted',sum(p is not None for p in templates),'fit',count,'classes',classes,'holdout',diagnostics,flush=True)
bundle,count,classes=fit(np.arange(len(b))); bv,ev,_,vv,_,yv=load('validation'); s,c=program_support(bv,bundle); hv=np.array([len(set(parse_answer(cc['text']) for cc in row['candidates'])-{None})>=2 for row in bv]); pred=s.argmax(1); valid=s.max(1)>0; correct=yv[np.arange(len(yv)),pred]
result=dict(extracted=sum(p is not None for p in templates),fit_questions=count,classes=classes,holdout=diagnostics,validation_hard_coverage=int((hv&valid).sum()),validation_hard_covered_accuracy=float(correct[hv&valid].mean()*100),seconds=time.perf_counter()-start)
(out/'model_bundle').mkdir(exist_ok=True); (out/'model_bundle/model.pkl').write_bytes(pickle.dumps(bundle,protocol=5)); (out/'selection.json').write_text(json.dumps(result,indent=2)); print(json.dumps(result),flush=True)
