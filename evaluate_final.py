"""Freeze assets, then time fresh official prediction and scorer processes."""
import os,sys,time,json,hashlib,subprocess,platform
from pathlib import Path
env=os.environ.copy()
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']: env[k]='4'
out=Path('runs')
files=['selector.py','features.py','arithmetic.py','runs/model_bundle/model.pkl']
hashes={f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in files}
(out/'frozen_manifest.json').write_text(json.dumps(dict(sha256=hashes,protocol='dase7506-mp2-open-hard-verifier-v5',selection='validation only; frozen before test scoring'),indent=2))
timings={}
for split in ['validation','test']:
    start=time.perf_counter()
    subprocess.run([sys.executable,'predict.py','--method','selector.py','--checkpoint','runs/model_bundle','--split',split,'--output',f'runs/{split}-predictions.json'],check=True,env=env)
    mid=time.perf_counter()
    subprocess.run([sys.executable,'score_predictions.py','--split',split,'--predictions',f'runs/{split}-predictions.json','--output',f'runs/{split}-score.json'],check=True,env=env)
    end=time.perf_counter()
    timings[split]=dict(prediction_process_seconds=mid-start,scoring_process_seconds=end-mid,total_seconds=end-start)
timings['environment']=dict(platform=platform.platform(),python=platform.python_version(),cpu='Intel Core Ultra 7 251HX',threads=4,gpu_used=False)
timings['asset_bytes']=Path('runs/model_bundle/model.pkl').stat().st_size
(out/'resources.json').write_text(json.dumps(timings,indent=2))
print(json.dumps(timings,indent=2))
