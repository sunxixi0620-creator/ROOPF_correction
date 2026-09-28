"""Dependency-aware continuation of the already launched supplementary jobs."""
import json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
PY=str(ROOT/'.venv/bin/python')
STAGES=[]

def wait_complete(name):
    p=ROOT/'results'/name/'COMPLETE'
    while not p.exists():time.sleep(15)

def launch(name,args,gpu=False,threads=1):
    env=os.environ.copy();env.update(OMP_NUM_THREADS=str(threads),MKL_NUM_THREADS=str(threads),OPENBLAS_NUM_THREADS=str(threads))
    if not gpu:env['CUDA_VISIBLE_DEVICES']=''
    log=(ROOT/'results'/f'{name}.log').open('a')
    p=subprocess.Popen([PY,'-u',*args],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
    return (name,p,time.perf_counter(),log,args)

def finish(job):
    name,p,start,log,args=job
    code=p.wait();log.close();assert code==0,(name,code)
    STAGES.append({'name':name,'seconds':time.perf_counter()-start,'command':[PY,*args]})
    (ROOT/'results/remaining_pipeline_stages_20260928.json').write_text(json.dumps(STAGES,indent=2))

def main():
    wait_complete('retrain_d20_canonical_20260928')
    # One freed GPU training slot; keep the paired10-D reconstruction separate.
    curated=launch('retrain_d10_curated_20260928',['scripts/reconstruct_anchor_training.py','--dim','10','--order','curated','--output','results/retrain_d10_curated_20260928'],True,2)
    labels=launch('native_residual_logs_20260928',['scripts/generate_native_residual_logs.py','--checkpoint','results/retrain_d20_canonical_20260928/anchor.pt','--output','results/native_residual_logs_20260928','--workers','8'])
    finish(labels)
    # Keep residual validation tensors from competing with two full unrolls.
    wait_complete('retrain_d10_original_20260928')
    holdouts=','.join(json.loads((ROOT/'artifacts/training_provenance/residual_training_metrics.json').read_text())['holdout_fids'])
    resdir=ROOT/'results/retrain_residual_d20_20260928';resdir.mkdir(exist_ok=True)
    args=['artifacts/recovered_sources/train_router_earlystop.py','--csv','results/native_residual_logs_20260928/*_pool.csv',
        '--out',str(resdir/'residual.pt'),'--metrics-out',str(resdir/'metrics.json'),'--epochs','400','--patience','50',
        '--seed','20260711','--holdout-fids',holdouts,'--max-rows','1200000']
    (resdir/'protocol.json').write_text(json.dumps({'command':[PY,*args],'holdouts':'residual only; seen by anchor','label':'incumbent improvement; no test labels'},indent=2))
    finish(launch('retrain_residual_d20_20260928',args,True,2));(resdir/'COMPLETE').write_text('Training complete; selected on residual held-out training functions only.')
    native=launch('native20_eval_20260928',['scripts/evaluate_retrained.py','--dim','20','--checkpoint','results/retrain_d20_canonical_20260928/anchor.pt',
        '--residual',str(resdir/'residual.pt'),'--output','results/native20_eval_20260928','--workers','10'])
    wait_complete('retrain_d10_original_20260928')
    original=launch('order_original_eval_20260928',['scripts/evaluate_retrained.py','--dim','10','--checkpoint','results/retrain_d10_original_20260928/anchor.pt','--output','results/order_original_eval_20260928','--workers','6'])
    finish(original);finish(native);finish(curated)
    finish(launch('order_curated_eval_20260928',['scripts/evaluate_retrained.py','--dim','10','--checkpoint','results/retrain_d10_curated_20260928/anchor.pt','--output','results/order_curated_eval_20260928','--workers','12']))
    (ROOT/'results/remaining_training_pipeline_COMPLETE').write_text('All scheduled training and dependent evaluations complete.')

if __name__=='__main__':main()
