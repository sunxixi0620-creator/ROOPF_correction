"""Amended baseline audit with exact per-step checkpoints and cold-fit recovery."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import argparse,json,multiprocessing as mp,sys,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import strong_online_baselines as previous_run
from scripts import continuous_prior_replication as parent
from roopf.fitted_online_v2 import TrustRegion,propose,replay_score
from roopf.experiment_io import CaseStore,sha256,write_json,save_torch,seed_for
RUN=ROOT/'results/strong_online_baselines_v2';OUT=ROOT/'docs/revision/strong_online_baselines_v2'
METHODS=previous_run.METHODS
SOURCES=('scripts/strong_online_baselines_v2.py','roopf/fitted_online_v2.py',
         'docs/experiments/STRONG_ONLINE_BASELINES_V2_AMENDMENT.md')


def freeze():
    previous_run.verify();OUT.mkdir(parents=True,exist_ok=True);RUN.mkdir(parents=True,exist_ok=True)
    if not (RUN/'identity.json').exists():
        write_json(RUN/'identity.json',dict(previous=sha256(previous_run.RUN/'identity.json'),
            sources={p:sha256(ROOT/p) for p in SOURCES},time=time.time()))
    return verify()


def verify():
    previous_run.verify();z=json.loads((RUN/'identity.json').read_text())
    assert z['previous']==sha256(previous_run.RUN/'identity.json')
    for p,h in z['sources'].items():assert sha256(ROOT/p)==h,p
    return z


def jobs():return previous_run.jobs()


def rollout(f,i,method,budget=300,role=parent.ROLE,checkpoint=None):
    torch.set_num_threads(1);start=time.time()
    task=parent.parent.ProceduralTask(f,i,role,dim=20)
    if checkpoint and checkpoint.exists():
        v=torch.load(checkpoint,weights_only=False)
        assert v['run_sha256']==sha256(RUN/'identity.json') and v['job']==(f,i,method)
        x,y,state,params,trace=v['x'],v['y'],v['state'],v['parameters'],v['trace']
        task.points=v['calls'];elapsed=v['seconds']
    else:
        x=parent.st.inputs(f,i,role,10);y=task.calfitness(x[None])[0]
        state=TrustRegion(best=float(y.min())) if method=='TuRBO_LogEI' else None
        params=None;trace=[];elapsed=0.
    def commit():
        if checkpoint:
            save_torch(checkpoint,dict(run_sha256=sha256(RUN/'identity.json'),job=(f,i,method),
                x=x,y=y,state=state,parameters=params,trace=trace,calls=task.points,seconds=elapsed+time.time()-start))
            write_json(checkpoint.with_suffix('.json'),dict(job=(f,i,method),paid=task.points,
                seconds=elapsed+time.time()-start,recoveries=sum(r.get('cold_fit_recovery',False) for r in trace)))
    commit()
    while len(x)<budget:
        if state and state.length<.5**7:
            n=min(10,budget-len(x));offset=len(x)
            seed=seed_for('strong_online_restart',f,i,role,state.restarts)%(2**31-1)
            q=torch.quasirandom.SobolEngine(20,scramble=True,seed=seed).draw(n)*10-5
            assert torch.cdist(q.double(),x.double()).min()>1e-6
            v=task.calfitness(q[None])[0];x,y=torch.cat((x,q)),torch.cat((y,v))
            state=TrustRegion(best=float(v.min()),start=offset,restarts=state.restarts+1)
            trace.append(dict(restart=True,paid_before=offset,count=n,points=q));params=None;commit();continue
        before=task.points
        q,info,params=propose(x,y,method,(f,i,role),state,params)
        assert task.points==before and task.diagnostic_points==0
        value=task.calfitness(q[None])[0,0];info.update(paid_before=len(x),restart=False)
        trace.append(info);x,y=torch.cat((x,q)),torch.cat((y,value[None]))
        if state:state.update(float(value))
        commit()
    assert task.points==budget and task.diagnostic_points==0
    return dict(fid=f,instance=i,method=method,x=x,y=y,initial_x=x[:10].clone(),initial_y=y[:10].clone(),
        calls=task.points,teacher_calls=0,trace=trace,seconds=elapsed+time.time()-start,device='cpu')


def contracts():
    verify()
    if (OUT/'CONTRACTS.json').exists():return
    old=torch.load(previous_run.RUN/'contracts.pt',weights_only=False)
    results={}
    for m in METHODS:
        a=rollout(0,0,m,12,'strong_online_contract_v1')
        assert torch.equal(a['x'],old[m][0]['x']) and torch.equal(a['y'],old[m][0]['y'])
        results[m]=a
    # Reuse the exact failure state's paid observations: zero additional calls.
    d=torch.load(previous_run.RUN/'diagnostic/paid.pt',weights_only=False)
    state=TrustRegion(best=float(d['y'][:10].min()))
    for value in d['y'][10:]:state.update(float(value))
    q,info,params=propose(d['x'],d['y'],'TuRBO_LogEI',(0,0,parent.ROLE),state,d['trace'][-1]['parameters'])
    assert info['cold_fit_recovery'] and torch.isfinite(q).all()
    err=abs(replay_score(d['x'],d['y'],'TuRBO_LogEI',info)-info['logei']);assert err<1e-6
    # Exact resume after 11 paid values, compared with the uninterrupted contract.
    p=RUN/'resume_contract.pt'
    rollout(0,0,'GP_LogEI',11,'strong_online_contract_v1',p)
    resumed=rollout(0,0,'GP_LogEI',12,'strong_online_contract_v1',p)
    assert torch.equal(resumed['x'],results['GP_LogEI']['x'])
    assert torch.equal(resumed['y'],results['GP_LogEI']['y'])
    save_torch(RUN/'contracts.pt',dict(small=results,recovered=info,resumed=resumed))
    write_json(OUT/'CONTRACTS.json',dict(calls=36,recovery_replay_extra_calls=0,
        no_failure_path_matches_v1=True,cold_recovery_passed=True,exact_resume=True,replay_error=err))


def collect(j):
    store=CaseStore(RUN/'cases',verify());name='_'.join(map(str,j))
    with store.lock(name):
        if store.load(name,dict(job=j)) is not None:return True
        # V2 differs only on failed fitting; completed V1 cases followed no failed path.
        old=CaseStore(previous_run.RUN/'cases',previous_run.verify()).load(name,dict(job=j))
        if old is not None:
            for info in old['trace']:info['cold_fit_recovery']=False
            old['imported_v1_sha256']=sha256(previous_run.RUN/'cases'/f'{name}.pt')
            store.save(name,dict(job=j),old);return True
        try:
            result=rollout(*j,checkpoint=RUN/'progress'/f'{name}.pt')
            store.save(name,dict(job=j),result)
            (RUN/'errors'/f'{name}.json').unlink(missing_ok=True)
            return True
        except Exception as exc:
            write_json(RUN/'errors'/f'{name}.json',dict(job=j,error=repr(exc)))
            return False


def run(workers):
    verify();assert (OUT/'CONTRACTS.json').exists();start=time.time();failed=[]
    with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn')) as ex:
        fs={ex.submit(collect,j):j for j in jobs()}
        for n,f in enumerate(as_completed(fs),1):
            if not f.result():failed.append(fs[f])
            write_json(RUN/'STATUS.json',dict(finished=n,completed=n-len(failed),failed=failed,total=144,elapsed=time.time()-start))
            print(f'{n}/144 failed={len(failed)} {time.time()-start:.1f}s {fs[f]}',flush=True)
    if failed:raise RuntimeError(f'{len(failed)} failed cases retained with paid checkpoints')
    write_json(RUN/'COMPLETE.json',dict(cases=144,calls=43200,new_calls=41400,
        imported_v1_cases=6,seconds=time.time()-start,workers=workers))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['freeze','contracts','run'])
    p.add_argument('--workers',type=int,default=16);a=p.parse_args()
    run(a.workers) if a.phase=='run' else globals()[a.phase]()
