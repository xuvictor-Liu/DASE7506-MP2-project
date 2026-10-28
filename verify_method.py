"""Check batch independence, candidate permutation, and asset size on validation."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']: os.environ[k]='4'
import copy,json
from pathlib import Path
import numpy as np
from common import load_inputs,parse_answer
from selector import predict

bank,arrays=load_inputs('validation'); bank=bank[:24]; arrays={k:v[:24] for k,v in arrays.items()}
path=Path('runs/model_bundle'); p=predict(bank,arrays,path)
single=[predict([b],{k:v[i:i+1] for k,v in arrays.items()},path)[0] for i,b in enumerate(bank)]
assert p==single,'Predictions depend on batch composition'
perm=[2,0,3,1]; b2=copy.deepcopy(bank)
for b in b2:
    b['candidates']=[b['candidates'][i] for i in perm]
    for i,c in enumerate(b['candidates']): c['index']=i
p2=predict(b2,{k:v[:,perm] for k,v in arrays.items()},path)
# Ties may select different candidates with equivalent answers.
assert all(parse_answer(b['candidates'][i]['text'])==parse_answer(b['candidates'][perm[j]]['text']) for b,i,j in zip(bank,p,p2))
size=sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
assert size<=16*2**20
print(json.dumps(dict(batch_independence=True,permutation_answer_consistency=True,asset_bytes=size,asset_mib=size/2**20)))
