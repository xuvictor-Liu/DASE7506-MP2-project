"""Train-only answer-conditioned text classifier with an OOF verifier prior."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']: os.environ[k]='4'
import pickle,json
from pathlib import Path
import numpy as np
from scipy.sparse import hstack,csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from train import load
from local_text import tokens
from features_v2 import features_v2
from selector_v2 import neural_scores
from selector_refined import scores
from common import parse_answer
def documents(bank):
    docs=[]
    for row in bank:
        for c in row['candidates']:
            q,t=tokens(row['question'],c['text']); qs=set(q)
            shared=['shared_'+w for w in t[-100:] if w in qs]
            docs.append(' '.join(q+['solutionstart']+t+['overlapstart']+shared))
    return docs
def build(vectorizer,docs,prior):
    return hstack([vectorizer.transform(docs),csr_matrix(prior.reshape(-1,1))],format='csr')
def main():
    out=Path('runs_role_text'); out.mkdir(exist_ok=True)
    b,e,_,v,_,y=load('train'); f,_,_=features_v2(b,e); raw=np.concatenate([e,e-e.mean(1,keepdims=True),f],2)
    folds=pickle.loads(Path('runs_stack/model_bundle/model.pkl').read_bytes())['models']; prior=np.zeros_like(y,dtype=float)
    for (_,ho),m in zip(KFold(5,shuffle=True,random_state=2026).split(b),folds): prior[ho]=neural_scores(raw[ho],m)
    docs=np.array(documents(b)); hard=np.array([len(set(parse_answer(c['text']) for c in r['candidates'])-{None})>=2 for r in b]); ids=np.flatnonzero(hard); np.random.default_rng(731).shuffle(ids); cut=int(.8*len(ids)); tr,ho=ids[:cut],ids[cut:]
    ti=(tr[:,None]*4+np.arange(4)).ravel(); hi=(ho[:,None]*4+np.arange(4)).ravel(); vector=TfidfVectorizer(ngram_range=(1,3),max_features=30000,min_df=3,sublinear_tf=True,dtype=np.float32); vector.fit(docs[ti]); x=build(vector,docs,prior); records=[]
    for c in [.1,1,10,100]:
        model=LogisticRegression(C=c,solver='liblinear',max_iter=500,random_state=41); use=ti[v[tr].ravel()]; model.fit(x[use],y.ravel()[use]); s=model.decision_function(x[hi]).reshape(-1,4); p=np.where(v[ho],s,-1e20).argmax(1); r=dict(C=c,holdout_accuracy=float(y[ho,p].mean()*100)); records.append(r); print(json.dumps(r),flush=True)
    best=max(records,key=lambda r:r['holdout_accuracy']); ti=(ids[:,None]*4+np.arange(4)).ravel(); vector.fit(docs[ti]); x=build(vector,docs,prior); use=ti[v[ids].ravel()]; model=LogisticRegression(C=best['C'],solver='liblinear',max_iter=500,random_state=41).fit(x[use],y.ravel()[use])
    bv,ev,_,vv,_,yv=load('validation'); base=pickle.loads(Path('runs_v2/final_bundle/model.pkl').read_bytes()); s0,_=scores(bv,{'embeddings':ev},base); s=model.decision_function(build(vector,documents(bv),s0)).reshape(-1,4); p=np.where(vv,s,-1e20).argmax(1); correct=yv[np.arange(len(yv)),p]; hv=np.array([len(set(parse_answer(c['text']) for c in r['candidates'])-{None})>=2 for r in bv]); result=dict(selected=best,hard_accuracy=float(correct[hv].mean()*100),validation_accuracy=float(correct.mean()*100))
    (out/'model_bundle').mkdir(exist_ok=True); (out/'model_bundle/model.pkl').write_bytes(pickle.dumps(dict(vector=vector,model=model,base=base),protocol=5)); (out/'selection.json').write_text(json.dumps(result,indent=2)); (out/'holdout_search.json').write_text(json.dumps(records,indent=2)); print(json.dumps(result),flush=True)
if __name__=='__main__': main()
