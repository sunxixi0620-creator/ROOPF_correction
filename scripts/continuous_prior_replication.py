"""Three frozen training seeds with the exact continuous acquisition solver."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse,json,time,multiprocessing as mp
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import acquisition_factorial as parent
from roopf.acquisition_search import SmoothPrior,propose,original_scores
from roopf.experiment_io import CaseStore,sha256,write_json,save_torch
st=parent.st
RUN=ROOT/'results/continuous_prior_replication_v1';OUT=ROOT/'docs/revision/continuous_prior_replication'
ROLE='continuous_prior_replication_v1'
SOURCES=('scripts/continuous_prior_replication.py','docs/experiments/CONTINUOUS_PRIOR_REPLICATION_PROTOCOL.md')

def freeze():
    parent.verify();RUN.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
    if (RUN/'identity.json').exists():return verify()
    write_json(RUN/'identity.json',dict(parent=sha256(parent.RUN/'identity.json'),sources={p:sha256(ROOT/p) for p in SOURCES},time=time.time()))
    return verify()
def verify():
    parent.verify();z=json.loads((RUN/'identity.json').read_text());assert sha256(parent.RUN/'identity.json')==z['parent']
    for p,h in z['sources'].items():assert sha256(ROOT/p)==h,p
    return z

def rollout(f,i,m,seed,budget=300,role=ROLE):
    torch.set_num_threads(1);start=time.perf_counter();task=parent.ProceduralTask(f,i,role,dim=20)
    x=st.inputs(f,i,role,10);y=task.calfitness(x[None])[0];ix=x.clone();iy=y.clone()
    p=st.base.Prior(f,seed,{'O':'zero','WA':'analytic','W':'correct'}[m],x,y)
    sp=SmoothPrior(p);active=m!='O';gp=st.base.GP(x.numpy(),(y.numpy().astype(float)-p.center)/p.scale-(p(x.numpy()) if active else 0))
    trace=[];meta=[];switch=None
    for step in range(budget-10):
        if m in ('W','WA') and len(x)==40:
            active=False;gp=st.base.GP(x.numpy(),(y.numpy().astype(float)-p.center)/p.scale);switch=dict(count=len(gp.x),targets=gp.y.copy())
        q=st.pool(x,y.numpy(),f,i,role,step);raw=st.choose(q,x,y.numpy(),p,gp,active,m)
        logwinner=int(original_scores(q,x,y.numpy(),p,gp,active)[0].argmax());before=task.points
        candidates,new,info=propose(q,x,y.numpy(),p,gp,active,sp)
        info['original_rank_changed']=bool(logwinner!=raw[0]);assert task.points==before
        k,pm,mu,sd,score=new;point=torch.from_numpy(candidates[k:k+1]);value=task.calfitness(point[None])[0,0]
        std=(float(value)-p.center)/p.scale;gp.append(candidates[k],std-pm)
        trace.append([k,pm,mu,sd,score,std,float(active)]);meta.append(info);x=torch.cat((x,point));y=torch.cat((y,value[None]))
    p.verify();assert task.points==budget and task.diagnostic_points==0
    return dict(fid=f,instance=i,seed=seed,method=m,engine='continuous',x=x,y=y,initial_x=ix,initial_y=iy,
                trace=np.array(trace),acquisition=meta,switch=switch,calls=task.points,teacher_calls=0,prior_identity=p.identity,seconds=time.perf_counter()-start)

def jobs():return [(f,i,m,s) for f in range(36) for i in range(4) for m in ('O','WA','W') for s in (range(3) if m=='W' else [0])]
def collect(j):
    store=CaseStore(RUN/'cases',verify());name='_'.join(map(str,j))
    with store.lock(name):
        if store.load(name,dict(job=j)) is None:store.save(name,dict(job=j),rollout(*j))

def contracts():
    verify()
    if (OUT/'CONTRACTS.json').exists():return
    role='continuous_prior_replication_contract_v1';rows={}
    for m in ('O','WA','W'):
        a=rollout(0,0,m,0,45,role);b=parent.rollout(0,0,m,'continuous',45,role)
        assert torch.equal(a['x'],b['x']) and torch.equal(a['y'],b['y']) and np.array_equal(a['trace'],b['trace'])
        rows[m+'_copy']=a;rows[m+'_parent']=b
    for seed in (1,2):rows[f'W_{seed}']=rollout(0,0,'W',seed,45,role)
    assert len({rows[f'W_{seed}']['prior_identity'] for seed in (1,2)}|{rows['W_copy']['prior_identity']})==3
    save_torch(RUN/'contracts.pt',rows);write_json(OUT/'CONTRACTS.json',dict(calls=360,parent_loop_equivalence=True,three_distinct_model_seeds=True))

def run(workers):
    verify();assert (OUT/'CONTRACTS.json').exists();start=time.time()
    with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn')) as ex:
        fs=[ex.submit(collect,j) for j in jobs()]
        for n,f in enumerate(as_completed(fs),1):
            f.result();write_json(RUN/'STATUS.json',dict(completed=n,total=720,elapsed=time.time()-start))
            if n%24==0:print(f'{n}/720 {time.time()-start:.1f}s',flush=True)
    write_json(RUN/'COMPLETE.json',dict(cases=720,calls=216000,seconds=time.time()-start,workers=workers))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['freeze','contracts','run']);p.add_argument('--workers',type=int,default=16);a=p.parse_args()
    run(a.workers) if a.phase=='run' else globals()[a.phase]()
