"""Single fixed spatial-coverage candidate; no parameter search."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,time,argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import torch
from scipy.stats import qmc
from scipy.spatial.distance import cdist
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import cold_start as st
from roopf.experiment_io import CaseStore,sha256,write_json,save_torch,seed_for
from roopf.revision_tasks import ProceduralTask
base=st.base;inputs=st.inputs;pool=st.pool;choose=st.choose
RUN=ROOT/'results/cold_start_coverage_v1';OUT=ROOT/'docs/revision/cold_start_coverage'
ROLE='cold_start_coverage_v1'
SOURCES=('scripts/cold_start_coverage.py','docs/experiments/COLD_START_COVERAGE_PROTOCOL.md')
@torch.no_grad()
def rollout(fid,inst,seed,method,budget=600,role='cold_start_coverage_v1',zero=False):
    torch.set_num_threads(1);start=time.perf_counter();t=ProceduralTask(fid,inst,role,dim=20)
    x=inputs(fid,inst,role,10);y=t.calfitness(x[None])[0];ix=x.clone();iy=y.clone()
    p=base.Prior(fid,seed,'zero' if zero or method in ('O','S') else 'correct',x,y)
    active=method in ('P','F','W','H');gp=None
    if method!='P':gp=base.GP(x.numpy(),(y.numpy().astype(float)-p.center)/p.scale-(p(x.numpy()) if active else 0))
    sobol=qmc.Sobol(20,scramble=True,seed=seed_for(role,'sobol',fid,inst)%(2**32)).random_base2(5)[:30].astype('float32')*10-5
    trace=[];switch=None
    for step in range(budget-10):
        if method in ('W','H') and len(x)==40:
            active=False;gp=base.GP(x.numpy(),(y.numpy().astype(float)-p.center)/p.scale)
            switch=dict(count=len(gp.x),targets=gp.y.copy())
        if (method=='S' or (method=='H' and step%3==2)) and len(x)<40:
            index=step//3 if method=='H' else step; q=sobol[index:index+1];assert cdist(q,x.numpy()).min()>1e-6
            i,pm,mu,sd,score=choose(q,x,y,p,gp,active,'O')
        else:q=pool(x,y.numpy(),fid,inst,role,step);i,pm,mu,sd,score=choose(q,x,y.numpy(),p,gp,active,method)
        point=torch.from_numpy(q[i:i+1]);v=t.calfitness(point[None])[0,0];std=(float(v)-p.center)/p.scale
        if gp is not None:gp.append(q[i],std-pm)
        trace.append([i,pm,mu,sd,score,std,float(active)])
        x=torch.cat((x,point));y=torch.cat((y,v[None]))
    p.verify();assert t.points==budget and t.diagnostic_points==0
    return dict(fid=fid,instance=inst,seed=seed,method=method,x=x,y=y,trace=np.array(trace),initial_x=ix,initial_y=iy,switch=switch,calls=t.points,teacher_calls=t.diagnostic_points,seconds=time.perf_counter()-start,prior_identity=p.identity)

def verify():
    st.verify();z=json.loads((RUN/'identity.json').read_text())
    for p,h in z['sources'].items():assert sha256(ROOT/p)==h
    assert sha256(st.RUN/'identity.json')==z['parent'];return z
def freeze():
    st.verify();RUN.mkdir(exist_ok=True)
    if (RUN/'identity.json').exists():return verify()
    write_json(RUN/'identity.json',dict(sources={p:sha256(ROOT/p) for p in SOURCES},parent=sha256(st.RUN/'identity.json'),time=time.time()))
def jobs():return [(f,i,s,m) for f in range(36) for i in range(6) for m in ('H','W','O','S') for s in (range(3) if m in ('H','W') else [0])]
def collect(j):
    store=CaseStore(RUN/'cases',verify());name='_'.join(map(str,j))
    with store.lock(name):
        if store.load(name,dict(job=j)) is None:store.save(name,dict(job=j),rollout(*j,role=ROLE) if j[3]=='H' else st.rollout(*j,role=ROLE))
def run():
    start=time.time()
    with ProcessPoolExecutor(max_workers=24) as ex:
        fs=[ex.submit(collect,j) for j in jobs()]
        for n,f in enumerate(as_completed(fs),1):
            f.result()
            if n%144==0:print(f'{n}/1728, {time.time()-start:.1f}s',flush=True)
    write_json(RUN/'COMPLETE.json',dict(cases=1728,calls=1036800,seconds=time.time()-start))
def report():
    store=CaseStore(RUN/'cases',verify());OUT.mkdir(parents=True,exist_ok=True)
    arrays={m:{'time':np.zeros((3,36,6,3)),'success':np.zeros((3,36,6,3)),'u':np.zeros((3,36,6,591))} for m in ('H','W','O','S')};calls=0
    for j in jobs():
        f,i,s,m=j;v=store.load('_'.join(map(str,j)),dict(job=j));assert v is not None;calls+=v['calls'];y=v['y'].numpy().astype(float);initial=y[:10];gain=np.maximum(0,initial.min()-np.minimum.accumulate(y)[9:])/max(initial.std(ddof=1),1e-8)
        ts=[];ss=[]
        for g in (.25,.5,1.):
            hit=np.flatnonzero(gain[:291]>=g);ts.append(int(hit[0]+10) if len(hit) else 301);ss.append(bool(len(hit)))
        for seed in ([s] if m in ('H','W') else range(3)):
            arrays[m]['time'][seed,f,i]=ts;arrays[m]['success'][seed,f,i]=ss;arrays[m]['u'][seed,f,i]=gain/(1+gain)
    assert calls==1036800;rng=np.random.default_rng(202609304);index=rng.integers(0,12,(10000,12))
    def est(d):
        groups=d.reshape(3,12,3,6).mean((0,2,3));lo,hi=np.quantile(groups[index].mean(1),[.0125,.9875]);return dict(mean=float(d.mean()),lower=float(lo),upper=float(hi),seeds=d.mean((1,2)).tolist())
    contrasts=[]
    for m in ('O','S','W'):
        a=est(arrays[m]['time'][...,1]-arrays['H']['time'][...,1]);b=est(arrays['H']['u'][...,-1]-arrays[m]['u'][...,-1]);c=est(arrays['H']['success'][...,1]-arrays[m]['success'][...,1]);passed=a['mean']>=5 and a['lower']>0 and min(a['seeds'])>0 and b['lower']>-.005 and c['mean']>=0
        contrasts.append(dict(control=m,primary=m in ('O','S'),time_saved=a,terminal_delta=b,attainment_delta=c,passed=bool(passed)))
    result=dict(joint_pass=all(c['passed'] for c in contrasts if c['primary']),contrasts=contrasts,calls=calls,epochs=0,means={m:{'time':a['time'].mean((0,1,2)).tolist(),'success':a['success'].mean((0,1,2)).tolist(),'terminal':float(a['u'][...,-1].mean())} for m,a in arrays.items()})
    write_json(OUT/'RESULTS.json',result);np.savez_compressed(OUT/'arrays.npz',**{m+'_'+k:v for m,a in arrays.items() for k,v in a.items()});print(json.dumps(result,indent=2),flush=True)

def contracts():
    verify()
    if (RUN/'CONTRACTS.json').exists():return
    v=rollout(0,0,0,'H',45,'cold_start_coverage_contract_v1');g=qmc.Sobol(20,scramble=True,seed=seed_for('cold_start_coverage_contract_v1','sobol',0,0)%(2**32)).random_base2(5)[:10].astype('float32')*10-5
    assert np.array_equal(v['x'][12:40:3].numpy(),g)
    assert v['switch']['count']==40 and not v['trace'][30:,6].any()
    save_torch(RUN/'contracts.pt',v);write_json(RUN/'CONTRACTS.json',dict(calls=45,sobol_slots=10,guided_slots=20,switch=True))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['freeze','contracts','run','report']);a=p.parse_args();globals()[a.phase]()
