"""Reproduce frozen second-round model with fresh official processes."""
import os,sys,json,time,hashlib,subprocess,copy
from pathlib import Path
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']: os.environ[k]='4'
import numpy as np
from common import load_inputs,parse_answer
from selector_refined import predict
manifest=json.loads(Path('runs_v2/frozen_manifest.json').read_text())
for f,h in manifest['sha256'].items(): assert hashlib.sha256(Path(f).read_bytes()).hexdigest()==h
b,a=load_inputs('validation'); b=b[:24]; a={k:v[:24] for k,v in a.items()}; checkpoint=Path('runs_v2/final_bundle')
p=predict(b,a,checkpoint)
assert p==[predict([row],{k:v[i:i+1] for k,v in a.items()},checkpoint)[0] for i,row in enumerate(b)]
perm=[2,0,3,1]; bp=copy.deepcopy(b)
for row in bp:
    row['candidates']=[row['candidates'][i] for i in perm]
    for i,c in enumerate(row['candidates']): c['index']=i
pp=predict(bp,{k:v[:,perm] for k,v in a.items()},checkpoint)
assert all(parse_answer(row['candidates'][i]['text'])==parse_answer(row['candidates'][perm[j]]['text']) for row,i,j in zip(b,p,pp))
result=dict(batch_independence=True,permutation_answer_consistency=True,asset_bytes=(checkpoint/'model.pkl').stat().st_size,cpu='Intel Core Ultra 7 251HX',threads=4)
for split in ['validation','test']:
    start=time.perf_counter()
    subprocess.run([sys.executable,'predict.py','--method','selector_refined.py','--checkpoint',str(checkpoint),'--split',split,'--output',f'runs_v2/{split}-predictions.json'],check=True)
    mid=time.perf_counter()
    subprocess.run([sys.executable,'score_predictions.py','--split',split,'--predictions',f'runs_v2/{split}-predictions.json','--output',f'runs_v2/{split}-score.json'],check=True)
    end=time.perf_counter(); result[split]=dict(prediction_seconds=mid-start,scoring_seconds=end-mid,total_seconds=end-start)
Path('runs_v2/resources.json').write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))
