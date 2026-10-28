"""Train-only sparse text verifier as a complementary development experiment."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']: os.environ[k]='4'
import json,time,pickle
from pathlib import Path
import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from train import load
from common import parse_answer
from features import choose

start=time.perf_counter(); out=Path('runs_text'); out.mkdir(exist_ok=True)
bt,et,ft,vt,at,yt=load('train'); bv,ev,fv,vv,av,yv=load('validation')
hard=np.array([len(set(parse_answer(c['text']) for c in b['candidates'])-{None})>=2 for b in bv])
tt=[c['text'] for b in bt for c in b['candidates']]; tv=[c['text'] for b in bv for c in b['candidates']]
word=TfidfVectorizer(ngram_range=(1,2),min_df=4,max_features=16000,sublinear_tf=True,dtype=np.float32)
char=TfidfVectorizer(analyzer='char',ngram_range=(3,5),min_df=6,max_features=16000,sublinear_tf=True,dtype=np.float32)
x=hstack([word.fit_transform(tt),char.fit_transform(tt)],format='csr'); z=hstack([word.transform(tv),char.transform(tv)],format='csr')
use=np.repeat(yt.min(1)==0,4); records=[]; best=None
for C in [.1,1,10]:
    model=LogisticRegression(C=C,max_iter=400,solver='liblinear',random_state=42).fit(x[use],yt.ravel()[use])
    scores=model.decision_function(z).reshape(-1,4); c=yv[np.arange(len(yv)),choose(scores,vv)]
    r=dict(C=C,hard_accuracy=float(c[hard].mean()*100),validation_accuracy=float(c.mean()*100),seconds=time.perf_counter()-start)
    records.append(r); print(json.dumps(r),flush=True)
    if best is None or r['hard_accuracy']>best[0]: best=(r['hard_accuracy'],model,scores)
(out/'experiments.json').write_text(json.dumps(records,indent=2))
(out/'model.pkl').write_bytes(pickle.dumps(dict(word=word,char=char,model=best[1]),protocol=5))
np.save(out/'validation_scores.npy',best[2])
