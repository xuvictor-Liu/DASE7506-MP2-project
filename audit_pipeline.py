"""Audit train/validation labels and frozen inference; no test reference access."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']: os.environ[k]='4'
import copy,json,pickle,hashlib
from pathlib import Path
import numpy as np
from train import load
from common import check_data,score_choices
from selector_refined import scores as v2_scores
from selector_semantic_blend import scores as v3_scores
def main():
    out=Path('runs_audit'); out.mkdir(exist_ok=True); result={'test_references_read':False,'splits':{},'inference':{},'hashes':{}}
    for split in ['train','validation']:
        check_data(splits=[split]); b,e,_,v,_,y=load(split); ids=[r['id'] for r in b]
        official=[score_choices(split,ids,[j]*len(b))['correct'] for j in range(4)]; assert official==y.sum(0).tolist(); assert (y.sum(1)>0).all(); assert not ((y>0)&~v).any()
        result['splits'][split]=dict(n=len(b),column_label_counts=official,invalid_candidates=int((~v).sum()),positive_labels_invalid=0,every_question_has_correct_candidate=True)
    bv,ev,_,_,_,_=load('validation'); bank=bv[:48]; e=ev[:48]; order=[2,0,3,1]; permuted=copy.deepcopy(bank)
    for row in permuted:
        row['candidates']=[row['candidates'][j] for j in order]
        for index,c in enumerate(row['candidates']): c['index']=index
    for name,path,fn in [('v2','runs_v2/final_bundle/model.pkl',v2_scores),('v3','runs_v3/final_bundle/model.pkl',v3_scores)]:
        model=pickle.loads(Path(path).read_bytes()); s,v=fn(bank,{'embeddings':e},model); sp,vp=fn(permuted,{'embeddings':e[:,order]},model); chunks=[fn(bank[i:i+7],{'embeddings':e[i:i+7]},model)[0] for i in range(0,len(bank),7)]; batch=np.concatenate(chunks)
        assert np.allclose(sp,s[:,order],atol=3e-5); assert np.array_equal(vp,v[:,order]); assert np.allclose(batch,s,atol=3e-5)
        result['inference'][name]=dict(permutation_max_error=float(np.max(np.abs(sp-s[:,order]))),batch_max_error=float(np.max(np.abs(batch-s))),finite_scores=bool(np.isfinite(s).all()))
        manifest=json.loads(Path('runs_'+name+'/frozen_manifest.json').read_text()); bad=[p for p,h in manifest['sha256'].items() if hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h]; assert not bad,bad; result['hashes'][name]='all frozen hashes match'
    result['findings']=[dict(severity='methodological',finding='V3 role-text calibrator is trained with raw single-fold stack scores but deployed with raw V2 scores. Different model family and score scale; no proof this caused test degradation.'),dict(severity='methodological',finding='Role-text regularization selection uses a non-nested meta holdout over precomputed OOF teacher scores.'),dict(severity='evaluation',finding='Hard validation and full test are distinct metrics. Many rounds reuse the validation set.'),dict(severity='documentation',finding='Historical V2 README/report still say current; a consolidated report must distinguish V2 and V3.')]
    (out/'audit.json').write_text(json.dumps(result,indent=2)); print(json.dumps(result),flush=True)
if __name__=='__main__': main()
