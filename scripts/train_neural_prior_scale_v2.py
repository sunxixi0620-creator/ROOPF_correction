"""One-seed, 45-fit bounded scale correction; preserves all v1 evidence."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse,json,multiprocessing as mp,sys,time,traceback
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import train_neural_prior_v1 as parent
from roopf.neural_prior_scale_v2 import StableSourceModel
from roopf.neural_prior_training_v1 import optimizer_for,update,cpu_tree
from roopf.experiment_io import CaseStore,write_json,save_torch,sha256
RUN=ROOT/'results/neural_prior_scale_v2';OUT=ROOT/'docs/revision/neural_prior_scale_v2'
AUDIT=ROOT/'docs/revision/neural_prior_training_v1_audit'
SOURCES=('roopf/neural_prior_scale_v2.py','scripts/check_neural_prior_scale_v2.py',
         'scripts/train_neural_prior_scale_v2.py','scripts/report_neural_prior_scale_v2.py',
         'docs/experiments/NEURAL_PRIOR_SCALE_V2.json')


def freeze():
    torch.set_num_threads(1);parent.verify();RUN.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
    assert json.loads((AUDIT/'RESULTS.json').read_text())['numerical']['same_device_cuda_replay_passed']
    c=json.loads((OUT/'CONTRACTS.json').read_text())
    assert c['passed'] and c['source_sha256']==sha256(ROOT/SOURCES[0]) and c['test_sha256']==sha256(ROOT/SOURCES[1])
    path=RUN/'identity.json'
    if not path.exists():
        spaces=sorted({t['space'] for t in parent.tasks()},key=int);references={};jobs=[]
        for index,space in enumerate(spaces):
            values=[];names=[]
            for t in parent.tasks():
                if t['space']==space and t['source_role']=='train':
                    x,y=parent.data_for(t);values.append(float(y.var(unbiased=True)));names.append(parent.source.parent.key(t))
            value=float(np.median(values));assert value>0 and np.isfinite(value)
            references[space]=dict(variance=value,source_task_names=names,source_task_variances=values)
            jobs.extend(dict(space=space,seed=0,kind=kind,device='cuda' if index%8==0 else 'cpu') for kind in ('N','M','PCA'))
        write_json(path,dict(parent=sha256(parent.RUN/'identity.json'),audit=sha256(AUDIT/'RESULTS.json'),
            contracts=sha256(OUT/'CONTRACTS.json'),sources={p:sha256(ROOT/p) for p in SOURCES},
            references=references,jobs=jobs,time=time.time()))
    verify();write_json(OUT/'IDENTITY.json',json.loads(path.read_text()))


def verify():
    parent.verify();z=json.loads((RUN/'identity.json').read_text())
    assert z['parent']==sha256(parent.RUN/'identity.json') and z['audit']==sha256(AUDIT/'RESULTS.json')
    assert z['contracts']==sha256(OUT/'CONTRACTS.json')
    for p,h in z['sources'].items():assert sha256(ROOT/p)==h,p
    return z


def training_job(job):
    torch.set_num_threads(1);identity=verify();space,seed,kind,device=(job[k] for k in ('space','seed','kind','device'))
    name=f'{space}_{seed}_{kind}';store=CaseStore(RUN/'models',identity)
    with store.lock(name):
        if store.load(name,job) is not None:return name
        train,val=parent.load_episodes(space,device);dim=train.tasks[0]['x'].shape[-1]
        reference=identity['references'][space]['variance']
        model=StableSourceModel(dim,kind,seed,space,reference).to(device);optimizer=optimizer_for(model)
        path=RUN/'progress'/f'{name}.pt';start=time.perf_counter();step=0;stale=0;elapsed=0.;history=[]
        if path.exists():
            p=torch.load(path,map_location='cpu',weights_only=False)
            assert p['identity']==sha256(RUN/'identity.json') and p['job']==job
            model.load_state_dict(p['model']);optimizer.load_state_dict(p['optimizer'])
            step,stale,elapsed,history,best=(p[k] for k in ('step','stale','seconds','history','best'))
            recovery_path=RUN/'recovery'/f'{name}.json';events=json.loads(recovery_path.read_text()) if recovery_path.exists() else []
            events.append(dict(step=step,time=time.time(),uncheckpointed_updates_upper_bound=99,lost_compute_seconds='unknown'))
            write_json(recovery_path,events)
        else:
            initial=val.validation(model);assert np.isfinite(initial['nll'])
            best=dict(step=0,metrics=initial,state=cpu_tree(model.state_dict()))
            history=[dict(step=0,**{k:v for k,v in initial.items() if k!='rows'})]
        def commit():
            seconds=elapsed+time.perf_counter()-start
            save_torch(path,dict(identity=sha256(RUN/'identity.json'),job=job,step=step,stale=stale,seconds=seconds,
                model=cpu_tree(model.state_dict()),optimizer=cpu_tree(optimizer.state_dict()),best=best,history=history))
            write_json(RUN/'progress'/f'{name}.json',dict(job=job,step=step,best_step=best['step'],
                best_validation_nll=best['metrics']['nll'],stale=stale,seconds=seconds))
        commit();losses=[]
        try:
            while step<20000 and stale<15:
                step+=1;loss,norm=update(model,optimizer,train.training(seed,step));losses.append(loss)
                if step%100==0:
                    metrics=val.validation(model);assert np.isfinite(metrics['nll'])
                    history.append(dict(step=step,training_nll=float(np.mean(losses)),**{k:v for k,v in metrics.items() if k!='rows'}));losses=[]
                    if metrics['nll']<best['metrics']['nll']:
                        best=dict(step=step,metrics=metrics,state=cpu_tree(model.state_dict()));stale=0
                    else:stale+=1
                    commit()
                    if step==100 or step%1000==0:print('updates',name,device,step,'best',best['step'],flush=True)
            store.save(name,job,dict(job=job,dim=dim,steps=step,reference_variance=reference,best=best,history=history,
                seconds=elapsed+time.perf_counter()-start,target_calls=0,confirmation_calls=0,
                train_tasks=[t['name'] for t in train.tasks],validation_tasks=[t['name'] for t in val.tasks]))
            (RUN/'errors'/f'{name}.json').unlink(missing_ok=True)
        except Exception as err:
            write_json(RUN/'errors'/f'{name}.json',dict(job=job,step=step,error=repr(err),traceback=traceback.format_exc()));raise
    return name


def run():
    identity=verify();start=time.perf_counter();failed=[]
    # All three methods in a space share the same preassigned device. Separate
    # executors prevent idle GPU workers from waiting for CPU-owned case locks.
    with ProcessPoolExecutor(max_workers=12,mp_context=mp.get_context('spawn')) as cpu,\
         ProcessPoolExecutor(max_workers=3,mp_context=mp.get_context('spawn')) as gpu:
        futures={(gpu if j['device']=='cuda' else cpu).submit(training_job,j):j for j in identity['jobs']}
        for n,f in enumerate(as_completed(futures),1):
            try:name=f.result()
            except Exception as err:name=str(futures[f]);failed.append(dict(job=futures[f],error=repr(err)))
            write_json(RUN/'STATUS.json',dict(finished=n,total=45,failed=failed,seconds=time.perf_counter()-start))
            print('trained',n,'/45','failed',len(failed),name,round(time.perf_counter()-start,1),flush=True)
    if failed:raise RuntimeError(f'{len(failed)} retained fit failures')
    write_json(RUN/'COMPLETE.json',dict(fits=45,seconds=time.perf_counter()-start,target_calls=0,confirmation_calls=0))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['freeze','run']);a=p.parse_args()
    freeze() if a.phase=='freeze' else run()
