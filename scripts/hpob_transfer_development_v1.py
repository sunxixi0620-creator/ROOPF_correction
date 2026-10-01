"""Freeze, fit source bank, cache predictions and run isolated HPO-B development."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import argparse
import hashlib
import json
import multiprocessing as mp
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from roopf.experiment_io import CaseStore,write_json,save_torch,seed_for,sha256,atomic_path
from roopf.hpob_transfer_v1 import standardize,fit,build_gp,Predictor,propose
from roopf.hpob_grouped_oracle import PaidOracle
RUN=ROOT/'results/hpob_transfer_development_v1'
OUT=ROOT/'docs/revision/hpob_transfer_development_v1'
SPLIT=ROOT/'docs/research/hpob_preparation/RAW_GROUPED_SPLIT.json'
DATA=ROOT/'results/hpob_source_preparation/data'
SOURCES=('scripts/hpob_transfer_development_v1.py','roopf/hpob_transfer_v1.py',
 'roopf/hpob_grouped_oracle.py','scripts/check_hpob_transfer_v1.py',
 'roopf/experiment_io.py','docs/experiments/HPOB_TRANSFER_DEVELOPMENT_V1.md')
METHODS=('O','P','A','R')


def tasks(role):
    return [t for t in json.loads(SPLIT.read_text())['tasks'] if t['eligible'] and t['role']==role]


def key(t):return t['space']+'_'+t['task_id']


def verify():
    z=json.loads((RUN/'identity.json').read_text())
    assert z['split']==sha256(SPLIT)
    for p,h in z['sources'].items():assert sha256(ROOT/p)==h,p
    assert z['exports']==sha256(RUN/'EXPORTS.json')
    return z


def prepare():
    RUN.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
    if (RUN/'EXPORTS.json').exists():
        z=json.loads((RUN/'EXPORTS.json').read_text());assert z['split']==sha256(SPLIT)
        for p,h in z['files'].items():assert sha256(RUN/p)==h
        return
    rows=tasks('source')+tasks('development');files={};source_labels=0
    groups={r:{t['group'] for t in tasks(r)} for r in ('source','development','confirmation')}
    assert all(not groups[a]&groups[b] for a,b in (('source','development'),('source','confirmation'),('development','confirmation')))
    for filename in ('meta-train-dataset.json','meta-validation-dataset.json'):
        tables=json.loads((DATA/filename).read_text())
        for t in rows:
            if t['original_file']!=filename:continue
            task=tables[t['space']][t['task_id']];x=np.asarray(task['X'],dtype=np.float64)
            _,indices=np.unique(x,axis=0,return_index=True);indices=np.sort(indices)
            assert hashlib.sha256(indices.astype('<i8').tobytes()).hexdigest()==t['canonical_indices_sha256']
            if t['role']=='source':
                rng=np.random.default_rng(seed_for('hpob_transfer_source_v1',t['space'],t['task_id']))
                chosen=indices[rng.permutation(len(indices))[:256]]
                # Extract only the frozen source subset, not other source responses.
                raw_y=task['y'];y=np.asarray([raw_y[int(i)] for i in chosen],dtype=np.float64).reshape(-1)
                p=RUN/'source_data'/f'{key(t)}.npz'
                with atomic_path(p) as temp:
                    with temp.open('wb') as stream:np.savez(stream,x=x[chosen],y=y,original_indices=chosen)
                files[str(p.relative_to(RUN))]=sha256(p);source_labels+=len(y)
            else:
                # Full target responses belong only to the separate evaluator export.
                y=np.asarray(task['y'],dtype=np.float64).reshape(-1)[indices]
                for sub,value in [('public',x[indices]),('oracle',y)]:
                    p=RUN/sub/f'{key(t)}.npy'
                    with atomic_path(p) as temp:
                        with temp.open('wb') as stream:np.save(stream,value,allow_pickle=False)
                    files[str(p.relative_to(RUN))]=sha256(p)
            assert np.isfinite(y).all()
        del tables
    write_json(RUN/'EXPORTS.json',dict(split=sha256(SPLIT),files=files,source_labels=source_labels,
        source_tasks=344,development_tasks=66,confirmation_exported=False,target_calls=0,
        inputs={p:sha256(DATA/p) for p in ('meta-train-dataset.json','meta-validation-dataset.json')}))
    print('exports',source_labels,'source labels; target calls=0',flush=True)


def freeze():
    import botorch,gpytorch,scipy
    prepare()
    if not (RUN/'identity.json').exists():
        write_json(RUN/'identity.json',dict(split=sha256(SPLIT),exports=sha256(RUN/'EXPORTS.json'),
            sources={p:sha256(ROOT/p) for p in SOURCES},time=time.time(),
            packages=dict(torch=torch.__version__,botorch=botorch.__version__,gpytorch=gpytorch.__version__,scipy=scipy.__version__),
            budget=105,seeds=list(range(5)),methods=METHODS,source_cap=256))
    verify()
    write_json(OUT/'IDENTITY.json',json.loads((RUN/'identity.json').read_text()))


def source_job(t):
    torch.set_num_threads(1);identity=verify();store=CaseStore(RUN/'models',identity)
    name=key(t);case=dict(task=t)
    with store.lock(name):
        if store.load(name,case) is not None:return name
        path=RUN/'source_data'/f'{name}.npz';ex=json.loads((RUN/'EXPORTS.json').read_text())
        assert sha256(path)==ex['files'][str(path.relative_to(RUN))]
        d=np.load(path);x=torch.from_numpy(d['x']);y=torch.from_numpy(d['y'])
        z,mean,scale=standardize(y);model,info=fit(x,z,('source',name))
        store.save(name,case,dict(x=x,z=z,mean=mean,scale=scale,info=info,task=t,
            calls=len(y),input_sha256=sha256(path)))
    return name


def cache_job(t,device='cpu'):
    torch.set_num_threads(1);identity=verify();store=CaseStore(RUN/'caches',identity)
    name=key(t);case=dict(task=t)
    with store.lock(name):
        if store.load(name,case) is not None:return name
        start=time.perf_counter();ex=json.loads((RUN/'EXPORTS.json').read_text())
        path=RUN/'public'/f'{name}.npy';assert sha256(path)==ex['files'][str(path.relative_to(RUN))]
        x=torch.from_numpy(np.load(path)).to(device);means=[];source_keys=[];model_hashes={}
        models=CaseStore(RUN/'models',identity)
        for st in tasks('source'):
            if st['space']!=t['space']:continue
            v=models.load(key(st),dict(task=st));assert v is not None
            model=build_gp(v['x'],v['z'])
            with torch.no_grad():
                for n,p in model.named_parameters():p.copy_(v['info']['parameters'][n])
            model=model.to(device);model.eval();pred=Predictor(model)
            means.append(pred.moments(x,variance=False)[0].cpu());source_keys.append(key(st))
            model_hashes[key(st)]=sha256(RUN/'models'/f'{key(st)}.pt')
        assert len(means)>=5
        store.save(name,case,dict(means=torch.stack(means),sources=source_keys,model_hashes=model_hashes,
            public_sha256=sha256(path),seconds=time.perf_counter()-start,device=device,target_calls=0))
    return name


def run_job(job):
    name,seed,method=job;torch.set_num_threads(1);identity=verify();case=dict(job=job)
    store=CaseStore(RUN/'cases',identity);case_id=f'{name}_{seed}_{method}'
    with store.lock(case_id):
        if store.load(case_id,case) is not None:return case_id
        t=next(t for t in tasks('development') if key(t)==name)
        cache=CaseStore(RUN/'caches',identity).load(name,dict(task=t));assert cache is not None
        path=RUN/'public'/f'{name}.npy';assert sha256(path)==cache['public_sha256']
        x=torch.from_numpy(np.load(path));source=cache['means']
        initialization=np.random.default_rng(seed_for('hpob_transfer_initial_v1',name,seed)).permutation(len(x))[:5].tolist()
        checkpoint=RUN/'progress'/f'{case_id}.pt'
        state=torch.load(checkpoint,weights_only=False) if checkpoint.exists() else dict(trace=[],seconds=0.,pending=None)
        if 'identity' in state:assert state['identity']==sha256(RUN/'identity.json') and state['job']==job
        trace=state['trace'];start=time.perf_counter();elapsed=state['seconds']
        oracle=PaidOracle(RUN/'oracle'/f'{name}.npy',RUN/'journals'/f'{case_id}.json')
        def commit(pending=None):
            save_torch(checkpoint,dict(trace=trace,seconds=elapsed+time.perf_counter()-start,
                pending=pending,identity=sha256(RUN/'identity.json'),job=job))
        try:
            # A crashed call may be present in the journal before the post-call checkpoint.
            pending=state.get('pending')
            if pending:
                n=pending['paid_before']
                assert len(oracle.paid) in (n,n+1)
                if len(oracle.paid)==n:oracle.query(pending['selected'])
                assert oracle.paid[n]['index']==pending['selected']
                trace.append(pending);commit()
            while len(oracle.paid)<5:oracle.query(initialization[len(oracle.paid)])
            assert [r['index'] for r in oracle.paid[:5]]==initialization
            assert len(trace)==len(oracle.paid)-5
            previous=trace[-1]['parameters'] if trace else None
            while len(oracle.paid)<105:
                paid=[r['index'] for r in oracle.paid];values=torch.tensor([r['value'] for r in oracle.paid],dtype=torch.double)
                selected,info=propose(x,paid,values,source,method,(name,seed,method),previous)
                commit(pending=info)  # Decision durable before objective query.
                oracle.query(selected);trace.append(info);previous=info['parameters'];commit()
            metrics=oracle.metrics()
            store.save(case_id,case,dict(task=t,seed=seed,method=method,paid=oracle.paid,
                trace=trace,metrics=metrics,calls=len(oracle.paid),
                seconds=elapsed+time.perf_counter()-start,source_cache_sha256=sha256(RUN/'caches'/f'{name}.pt')))
            (RUN/'errors'/f'{case_id}.json').unlink(missing_ok=True)
        except Exception as exc:
            write_json(RUN/'errors'/f'{case_id}.json',dict(job=job,error=repr(exc),paid=len(oracle.paid),traceback=traceback.format_exc()))
            raise
        finally:oracle.close()
    return case_id


def parallel(phase,workers,device='cpu'):
    verify();assert (OUT/'CONTRACTS.json').exists();start=time.perf_counter();errors=[]
    if phase=='sources':jobs=tasks('source');fn=source_job
    elif phase=='caches':jobs=tasks('development');fn=cache_job
    else:
        assert (RUN/'CACHES_COMPLETE.json').exists()
        jobs=[(key(t),seed,method) for seed in range(5) for t in tasks('development') for method in METHODS];fn=run_job
    with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn')) as pool:
        fs={pool.submit(fn,j,device) if phase=='caches' else pool.submit(fn,j):j for j in jobs}
        for n,f in enumerate(as_completed(fs),1):
            try:name=f.result()
            except Exception as exc:
                name=str(fs[f]);errors.append(dict(job=fs[f],error=repr(exc)))
            write_json(RUN/f'{phase.upper()}_STATUS.json',dict(finished=n,failed=errors,total=len(jobs),seconds=time.perf_counter()-start,workers=workers,device=device))
            print(phase,n,'/',len(jobs),'failed',len(errors),'seconds',round(time.perf_counter()-start,1),name,flush=True)
    if errors:raise RuntimeError(f'{len(errors)} failures; preserve journals and retry exact jobs')
    write_json(RUN/f'{phase.upper()}_COMPLETE.json',dict(jobs=len(jobs),workers=workers,device=device,seconds=time.perf_counter()-start,
        target_calls=138600 if phase=='run' else 0))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','freeze','sources','caches','run'])
    p.add_argument('--workers',type=int,default=16);p.add_argument('--device',choices=['cpu','cuda'],default='cpu');a=p.parse_args()
    if a.phase in ('prepare','freeze'):globals()[a.phase]()
    else:parallel(a.phase,a.workers,a.device)
