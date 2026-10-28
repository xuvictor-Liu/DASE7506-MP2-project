"""Freeze the validation-selected model before one independent final test evaluation."""
import os,sys,json,hashlib,shutil,time,subprocess
from pathlib import Path
from datetime import datetime,timezone
def main():
    out=Path('runs_v3'); out.mkdir(exist_ok=False); bundle=out/'final_bundle'; bundle.mkdir()
    source=Path('runs_round5/development_bundle/model.pkl'); shutil.copy2(source,bundle/'model.pkl')
    files=list(Path('.').glob('*.py'))+[bundle/'model.pkl']
    frozen=dict(frozen_at_utc=datetime.now(timezone.utc).isoformat(),method='selector_semantic_blend.py',checkpoint=str(bundle),selection_basis='Highest hard-validation accuracy across completed development rounds; ties retain the earlier model.',hard_validation_correct=309,hard_validation_n=355,blend_weight=.25,asset_bytes=source.stat().st_size,sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},protocol_note='One final test evaluation requested after development. Earlier >=90% validation gate was a self-imposed research stopping rule, not a course requirement; this evaluation replaces that rule before any new test score is observed. Prior aggregate V1/V2 test scores were already known. No test-driven training, selection or error inspection is performed.')
    assert frozen['asset_bytes']<=16*1024**2
    (out/'frozen_manifest.json').write_text(json.dumps(frozen,indent=2),encoding='utf-8')
    env=os.environ.copy()
    for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']: env[k]='4'
    start=time.perf_counter()
    subprocess.run([sys.executable,'predict.py','--method','selector_semantic_blend.py','--checkpoint',str(bundle),'--split','test','--output',str(out/'test-predictions.json')],env=env,check=True)
    subprocess.run([sys.executable,'score_predictions.py','--predictions',str(out/'test-predictions.json'),'--split','test','--output',str(out/'test-score.json')],env=env,check=True)
    elapsed=time.perf_counter()-start
    for p,h in frozen['sha256'].items(): assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
    result=dict(fresh_prediction_and_scoring_seconds=elapsed,threads=4,asset_bytes=frozen['asset_bytes'],local_cpu='Intel Core Ultra 7 251HX',reference_cpu_certified=False,hashes_unchanged=True)
    (out/'resources.json').write_text(json.dumps(result,indent=2),encoding='utf-8'); print(json.dumps(result),flush=True)
if __name__=='__main__': main()
