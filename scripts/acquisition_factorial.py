"""Frozen development factorial for acquisition search, with resumable cases."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing as mp
from pathlib import Path
import sys
import time
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import cold_start_components as parent
from roopf.acquisition_search import SmoothPrior,propose,original_scores
from roopf.experiment_io import CaseStore,sha256,write_json,save_torch
from roopf.revision_tasks import ProceduralTask

st=parent.st
RUN=ROOT/'results/acquisition_factorial_v1'
OUT=ROOT/'docs/revision/acquisition_factorial'
ROLE='acquisition_factorial_v1'
PRIORS=('O','WA','W');ENGINES=('pool','continuous')
SOURCES=('roopf/acquisition_search.py','scripts/acquisition_factorial.py','docs/experiments/ACQUISITION_FACTORIAL_PROTOCOL.md')

def freeze():
    parent.verify();RUN.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
    if (RUN/'identity.json').exists():return verify()
    write_json(RUN/'identity.json',dict(parent=sha256(parent.RUN/'identity.json'),sources={p:sha256(ROOT/p) for p in SOURCES},frozen=time.time()))
    return verify()

def verify():
    parent.verify();z=json.loads((RUN/'identity.json').read_text())
    assert sha256(parent.RUN/'identity.json')==z['parent']
    for p,h in z['sources'].items():assert sha256(ROOT/p)==h,p
    return z

def rollout(f,i,m,engine,budget=300,role=ROLE,zero=False):
    torch.set_num_threads(1);start=time.perf_counter()
    if engine=='pool':
        v=parent.rollout(f,i,0,m,budget,role,zero);v['engine']=engine;return v
    task=ProceduralTask(f,i,role,dim=20)
    x=st.inputs(f,i,role,10);y=task.calfitness(x[None])[0];ix=x.clone();iy=y.clone()
    p=st.base.Prior(f,0,'zero' if zero or m=='O' else ('analytic' if m=='WA' else 'correct'),x,y)
    sp=SmoothPrior(p);active=m!='O'
    gp=st.base.GP(x.numpy(),(y.numpy().astype(float)-p.center)/p.scale-(p(x.numpy()) if active else 0))
    trace=[];meta=[];switch=None
    for step in range(budget-10):
        if m in ('W','WA') and len(x)==40:
            active=False;gp=st.base.GP(x.numpy(),(y.numpy().astype(float)-p.center)/p.scale)
            switch=dict(count=len(gp.x),targets=gp.y.copy())
        q=st.pool(x,y.numpy(),f,i,role,step)
        raw=st.choose(q,x,y.numpy(),p,gp,active,m)
        logwinner=int(original_scores(q,x,y.numpy(),p,gp,active)[0].argmax())
        before=task.points
        candidates,new,info=propose(q,x,y.numpy(),p,gp,active,sp)
        info['original_rank_changed']=bool(logwinner!=raw[0])
        assert task.points==before
        k,pm,mu,sd,score=new
        point=torch.from_numpy(candidates[k:k+1]);value=task.calfitness(point[None])[0,0]
        std=(float(value)-p.center)/p.scale;gp.append(candidates[k],std-pm)
        trace.append([k,pm,mu,sd,score,std,float(active)]);meta.append(info)
        x=torch.cat((x,point));y=torch.cat((y,value[None]))
    p.verify();assert task.points==budget and task.diagnostic_points==0
    return dict(fid=f,instance=i,seed=0,method=m,engine=engine,x=x,y=y,initial_x=ix,initial_y=iy,
                trace=np.array(trace),acquisition=meta,switch=switch,calls=task.points,teacher_calls=0,
                prior_identity=p.identity,seconds=time.perf_counter()-start)

def jobs():return [(f,i,m,e) for f in range(36) for i in range(2) for e in ENGINES for m in PRIORS]

def collect(j):
    store=CaseStore(RUN/'cases',verify());name='_'.join(map(str,j))
    with store.lock(name):
        v=store.load(name,dict(job=j))
        if v is None:
            v=rollout(*j);store.save(name,dict(job=j),v)
    return v['seconds']

def run(workers):
    verify();assert (OUT/'CONTRACTS.json').exists()
    start=time.time()
    serial={}
    for j in jobs():
        if j[0]==j[1]==0:
            serial['_'.join(map(str,j))]=collect(j)
            print('serial',j,serial['_'.join(map(str,j))],flush=True)
    write_json(OUT/'SERIAL_TIMING.json',dict(seconds=serial,scope='fid0 instance0; part of main study, not extra calls'))
    remaining=[j for j in jobs() if not j[0]==j[1]==0]
    with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn')) as pool:
        fs={pool.submit(collect,j):j for j in remaining}
        for n,future in enumerate(as_completed(fs),7):
            future.result()
            write_json(RUN/'STATUS.json',dict(completed=n,total=432,elapsed=time.time()-start,workers=workers))
            if n%12==0:print(f'{n}/432 complete, {time.time()-start:.1f}s',flush=True)
    write_json(RUN/'COMPLETE.json',dict(cases=432,calls=129600,workers=workers,seconds=time.time()-start))

def contracts():
    verify()
    if (OUT/'CONTRACTS.json').exists():return
    role='acquisition_factorial_contract_v1';r={}
    for m in PRIORS:
        a=rollout(0,0,m,'pool',45,role);b=parent.rollout(0,0,0,m,45,role)
        assert torch.equal(a['x'],b['x']) and torch.equal(a['y'],b['y'])
        r[m+'_pool']=a;r[m+'_parent']=b
    a=rollout(0,0,'O','continuous',45,role);b=rollout(0,0,'W','continuous',45,role,True)
    assert torch.equal(a['x'],b['x']) and torch.equal(a['y'],b['y'])
    r['O_continuous']=a;r['W_zero_continuous']=b
    assert b['switch']['count']==40
    save_torch(RUN/'contracts.pt',r)
    write_json(OUT/'CONTRACTS.json',dict(calls=360,pool_equivalence=True,zero_prior_equivalence=True,switch=True,
               monotone_acquisition=all(z['logei']>=z['original_logei'] for v in (a,b) for z in v['acquisition'])))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['freeze','contracts','run']);p.add_argument('--workers',type=int,default=24)
    a=p.parse_args();run(a.workers) if a.phase=='run' else globals()[a.phase]()
