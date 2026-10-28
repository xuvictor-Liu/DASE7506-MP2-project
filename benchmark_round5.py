"""Fresh-process resource check on validation, not official test scoring."""
import os,sys,time,json,subprocess,argparse
from pathlib import Path
def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--method',default='selector_semantic_blend.py'); parser.add_argument('--checkpoint',default='runs_round5/development_bundle'); parser.add_argument('--name',default='semantic_blend'); args=parser.parse_args()
    env=os.environ.copy()
    for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']: env[key]='4'
    out=Path('runs_round5'); out.mkdir(exist_ok=True); prediction=out/(args.name+'_validation_predictions.json'); score=out/(args.name+'_validation_score.json'); start=time.perf_counter()
    subprocess.run([sys.executable,'predict.py','--method',args.method,'--checkpoint',args.checkpoint,'--split','validation','--output',str(prediction)],env=env,check=True)
    subprocess.run([sys.executable,'score_predictions.py','--predictions',str(prediction),'--split','validation','--output',str(score)],env=env,check=True)
    result=dict(fresh_prediction_plus_scoring_seconds=time.perf_counter()-start,split='validation',n=798,asset_bytes=sum(p.stat().st_size for p in Path(args.checkpoint).rglob('*') if p.is_file()),threads=4,method=args.method,checkpoint=args.checkpoint,note='Local validation-bank timing; not an official reference-CPU or test-bank certification.')
    (out/(args.name+'_resources.json')).write_text(json.dumps(result,indent=2)); print(json.dumps(result),flush=True)
if __name__=='__main__': main()
