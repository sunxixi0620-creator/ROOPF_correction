"""Gated source-only three-way prior training with durable paired checkpoints."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse,json,multiprocessing as mp,sys,time,traceback
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import prepare_neural_prior_source_v1 as source
from roopf.hpob_transfer_v1 import propose
from roopf.neural_prior_training_v1 import SourceModel,Episodes,optimizer_for,update,cpu_tree
from roopf.experiment_io import CaseStore,write_json,save_torch,sha256,seed_for
RUN=ROOT/'results/neural_prior_training_v1';OUT=ROOT/'docs/revision/neural_prior_training_v1'
PROTOCOL=ROOT/'docs/experiments/NEURAL_PRIOR_TRAINING_V1.json'
SOURCES=('scripts/train_neural_prior_v1.py','scripts/check_neural_prior_training_v1.py',
         'roopf/neural_prior_training_v1.py','docs/experiments/NEURAL_PRIOR_TRAINING_V1.json')


def tasks():return json.loads((source.OUT/'SOURCE_SPLIT.json').read_text())['tasks']


def freeze():
    source.verify();source.assess_parent()
    gate=json.loads((source.OUT/'ENTRY_DECISION.json').read_text());assert gate['entry_gate_passed']
    assert json.loads((source.OUT/'PCA_VERIFICATION.json').read_text())['passed']
    contracts=json.loads((OUT/'CONTRACTS.json').read_text())
    for key,path in (('source_sha256','roopf/neural_prior_training_v1.py'),
                     ('runner_sha256','scripts/train_neural_prior_v1.py'),
                     ('test_sha256','scripts/check_neural_prior_training_v1.py')):
        assert contracts[key]==sha256(ROOT/path),(key,'Run the training contract checks again')
    RUN.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
    p=RUN/'identity.json'
    if not p.exists():write_json(p,dict(source=sha256(source.RUN/'identity.json'),gate=sha256(source.OUT/'ENTRY_DECISION.json'),
        sources={q:sha256(ROOT/q) for q in SOURCES},time=time.time(),
        contracts=sha256(OUT/'CONTRACTS.json'),
        pca_files={s:sha256(source.RUN/'pca'/f'{s}.pt') for s in sorted({t['space'] for t in tasks()})}))
    verify();write_json(OUT/'IDENTITY.json',json.loads(p.read_text()))


def verify():
    source.verify();z=json.loads((RUN/'identity.json').read_text())
    assert z['source']==sha256(source.RUN/'identity.json') and z['gate']==sha256(source.OUT/'ENTRY_DECISION.json')
    assert z['contracts']==sha256(OUT/'CONTRACTS.json')
    for p,h in z['sources'].items():assert sha256(ROOT/p)==h,p
    return z


def data_for(t):
    name=source.parent.key(t);path=source.parent.RUN/'source_data'/f'{name}.npz'
    expected=json.loads((source.parent.RUN/'EXPORTS.json').read_text())['files']['source_data/'+name+'.npz']
    assert sha256(path)==expected
    with np.load(path) as d:return torch.from_numpy(d['x'].copy()),torch.from_numpy(d['y'].copy())


def context_job(job):
    torch.set_num_threads(1);t,seed=job;identity=verify();name=source.parent.key(t)+f'_{seed}'
    store=CaseStore(RUN/'contexts',identity);case=dict(task=t,seed=seed)
    with store.lock(name):
        if store.load(name,case) is not None:return name
        start=time.perf_counter();x,y=data_for(t);progress=RUN/'context_progress'/f'{name}.pt'
        if progress.exists():
            v=torch.load(progress,map_location='cpu',weights_only=False)
            assert v['identity']==sha256(RUN/'identity.json') and v['case']==case
            paid,trace,previous=v['paid'],v['trace'],v['previous'];elapsed=v['seconds']
        else:
            paid=np.random.default_rng(seed_for('source_prefix_init_v1',source.parent.key(t),seed)).permutation(len(x))[:5].tolist()
            trace=[];previous=None;elapsed=0.
        while len(paid)<40:
            index,info=propose(x,paid,y[paid],torch.empty(0,len(x),dtype=torch.double),'O',
                ('source_prefix_v1',source.parent.key(t),seed),previous)
            assert index not in paid;paid.append(index);trace.append(info);previous=info['parameters']
            save_torch(progress,dict(identity=sha256(RUN/'identity.json'),case=case,
                paid=paid,trace=trace,previous=previous,seconds=elapsed+time.perf_counter()-start))
        assert len(set(paid))==40
        store.save(name,case,dict(paid=paid,values=y[paid],trace=trace,role=t['source_role'],
            seconds=elapsed+time.perf_counter()-start,source_table_accesses=40,new_labels=0,target_calls=0))
    return name


def contexts(workers):
    verify();assert (OUT/'CONTRACTS.json').exists();start=time.perf_counter();failed=[]
    jobs=[(t,seed) for t in tasks() for seed in (0,1)]
    with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn')) as ex:
        fs={ex.submit(context_job,j):j for j in jobs}
        for n,f in enumerate(as_completed(fs),1):
            try:name=f.result()
            except Exception as err:name=str(fs[f]);failed.append(dict(job=fs[f],error=repr(err)))
            write_json(RUN/'CONTEXT_STATUS.json',dict(finished=n,total=len(jobs),failed=failed,seconds=time.perf_counter()-start))
            print('context',n,'/',len(jobs),'failed',len(failed),round(time.perf_counter()-start,1),name,flush=True)
    if failed:raise RuntimeError(f'{len(failed)} context failures retained')
    write_json(RUN/'CONTEXT_COMPLETE.json',dict(jobs=688,source_table_accesses=27520,
        unique_new_labels=0,target_calls=0,seconds=time.perf_counter()-start,workers=workers))


def load_episodes(space,device):
    identity=verify();store=CaseStore(RUN/'contexts',identity)
    assert sha256(source.RUN/'pca'/f'{space}.pt')==identity['pca_files'][space]
    cache=CaseStore(source.RUN/'pca',source.verify()).load(space,dict(space=space));packed={'train':[],'validation':[]}
    for t in tasks():
        if t['space']!=space:continue
        name=source.parent.key(t);x,y=data_for(t);p=cache['priors'][name]
        assert torch.equal(x,p['x']) and t['source_role']==p['source_role']
        prefixes=[]
        for seed in (0,1):
            v=store.load(name+f'_{seed}',dict(task=t,seed=seed));assert v is not None
            assert v['role']==t['source_role'] and torch.equal(v['values'],y[v['paid']])
            prefixes.append(v['paid'])
        packed[t['source_role']].append(dict(name=name,group=t['group'],role=t['source_role'],
            x=x,y=y,mean=p['mean'],features=p['features'],prefixes=prefixes))
    assert not {t['group'] for t in packed['train']}&{t['group'] for t in packed['validation']}
    return Episodes(packed['train'],space,'train',device),Episodes(packed['validation'],space,'validation',device)


def training_job(job,device):
    torch.set_num_threads(1);space,seed,kind=job;identity=verify();name=f'{space}_{seed}_{kind}'
    store=CaseStore(RUN/'models',identity);case=dict(job=job)
    with store.lock(name):
        if store.load(name,case) is not None:return name
        train,val=load_episodes(space,device);dim=train.tasks[0]['x'].shape[-1]
        model=SourceModel(dim,kind,seed,space).to(device);optimizer=optimizer_for(model)
        path=RUN/'progress'/f'{name}.pt';start=time.perf_counter();elapsed=0.;history=[];step=0;stale=0
        if path.exists():
            prior=torch.load(path,map_location='cpu',weights_only=False)
            assert prior['identity']==sha256(RUN/'identity.json') and prior['job']==job
            model.load_state_dict(prior['model']);optimizer.load_state_dict(prior['optimizer'])
            step=prior['step'];stale=prior['stale'];history=prior['history'];elapsed=prior['seconds']
            best=cpu_tree(prior['best'])
            recovery_path=RUN/'recovery'/f'{name}.json'
            recoveries=json.loads(recovery_path.read_text()) if recovery_path.exists() else []
            recoveries.append(dict(resumed_from_step=step,time=time.time(),
                uncheckpointed_updates_upper_bound=99,
                lost_compute_seconds='unknown; checkpoint accounting excludes interrupted work'))
            write_json(recovery_path,recoveries)
        else:
            initial=val.validation(model)
            assert np.isfinite(initial['nll'])
            best=dict(step=0,metrics=initial,state=cpu_tree(model.state_dict()))
            history.append(dict(step=0,**{k:v for k,v in initial.items() if k!='rows'}))
        def commit():
            save_torch(path,dict(identity=sha256(RUN/'identity.json'),job=job,step=step,stale=stale,
                model=cpu_tree(model.state_dict()),optimizer=cpu_tree(optimizer.state_dict()),
                best=best,history=history,seconds=elapsed+time.perf_counter()-start))
            write_json(RUN/'progress'/f'{name}.json',dict(job=job,step=step,best_step=best['step'],
                validation_nll=history[-1]['nll'],best_validation_nll=best['metrics']['nll'],stale=stale,
                seconds=elapsed+time.perf_counter()-start,device=device))
        commit();losses=[];norms=[]
        try:
            while step<20000 and stale<15:
                step+=1;loss,norm=update(model,optimizer,train.training(seed,step));losses.append(loss);norms.append(norm)
                if step%100==0:
                    metrics=val.validation(model);assert np.isfinite(metrics['nll'])
                    history.append(dict(step=step,training_nll=float(np.mean(losses)),max_gradient_norm=max(norms),
                        **{k:v for k,v in metrics.items() if k!='rows'}));losses=[];norms=[]
                    if metrics['nll']<best['metrics']['nll']:
                        best=dict(step=step,metrics=metrics,state=cpu_tree(model.state_dict()));stale=0
                    else:stale+=1
                    commit()
                    if step==100 or step%1000==0:print('updates',name,step,'best',best['step'],'seconds',round(elapsed+time.perf_counter()-start,1),flush=True)
            store.save(name,case,dict(job=job,dim=dim,kind=kind,steps=step,best=best,history=history,
                state=cpu_tree(model.state_dict()),seconds=elapsed+time.perf_counter()-start,
                network_parameters=sum(p.numel() for p in model.prior.parameters()) if model.prior is not None else 0,
                base_parameters=sum(p.numel() for p in model.base.parameters()),
                train_tasks=[t['name'] for t in train.tasks],validation_tasks=[t['name'] for t in val.tasks],
                device=device,target_calls=0,confirmation_calls=0))
            (RUN/'errors'/f'{name}.json').unlink(missing_ok=True)
        except Exception as exc:
            # Preserve the last valid checkpoint. Do not overwrite it with a
            # failed/nonfinite update or silently skip the responsible episode.
            write_json(RUN/'errors'/f'{name}.json',dict(job=job,attempted_step=step,error=repr(exc),traceback=traceback.format_exc(),
                seconds=elapsed+time.perf_counter()-start));raise
    return name


def training(workers,device):
    verify();assert (RUN/'CONTEXT_COMPLETE.json').exists();start=time.perf_counter();failed=[]
    spaces=sorted({t['space'] for t in tasks()},key=int)
    jobs=[(space,seed,kind) for seed in (0,1,2) for space in spaces for kind in ('N','M','PCA')]
    with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn')) as ex:
        fs={ex.submit(training_job,j,device):j for j in jobs}
        for n,f in enumerate(as_completed(fs),1):
            try:name=f.result()
            except Exception as err:name=str(fs[f]);failed.append(dict(job=fs[f],error=repr(err)))
            write_json(RUN/'TRAIN_STATUS.json',dict(finished=n,total=len(jobs),failed=failed,seconds=time.perf_counter()-start,workers=workers,device=device))
            print('trained',n,'/',len(jobs),'failed',len(failed),name,round(time.perf_counter()-start,1),flush=True)
    if failed:raise RuntimeError(f'{len(failed)} fitting failures retained')
    write_json(RUN/'TRAIN_COMPLETE.json',dict(jobs=135,seconds=time.perf_counter()-start,workers=workers,device=device,target_calls=0,confirmation_calls=0))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['freeze','contexts','training'])
    p.add_argument('--workers',type=int,default=3);p.add_argument('--device',choices=['cpu','cuda'],default='cuda');a=p.parse_args()
    if a.phase=='freeze':freeze()
    elif a.phase=='contexts':contexts(a.workers)
    else:training(a.workers,a.device)
