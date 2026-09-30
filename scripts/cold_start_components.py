"""Frozen seven-group cold-start component study."""
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
RUN=ROOT/'results/cold_start_components_v1';OUT=ROOT/'docs/revision/cold_start_components'
ROLE='cold_start_components_v1'
METHODS=('W','O','WS','WA','P','F','S');SEEDED=('W','WS','P','F')
SOURCES=('scripts/cold_start_components.py','docs/experiments/COLD_START_COMPONENTS_PROTOCOL.md')
@torch.no_grad()
def rollout(fid,inst,seed,method,budget=600,role='cold_start_components_v1',zero=False):
    torch.set_num_threads(1);start=time.perf_counter();t=ProceduralTask(fid,inst,role,dim=20)
    x=inputs(fid,inst,role,10);y=t.calfitness(x[None])[0];ix=x.clone();iy=y.clone()
    p=base.Prior(fid,seed,'zero' if zero or method in ('O','S') else ('shuffled' if method=='WS' else ('analytic' if method=='WA' else 'correct')),x,y)
    active=method in ('P','F','W','WS','WA');gp=None
    if method!='P':gp=base.GP(x.numpy(),(y.numpy().astype(float)-p.center)/p.scale-(p(x.numpy()) if active else 0))
    sobol=qmc.Sobol(20,scramble=True,seed=seed_for(role,'sobol',fid,inst)%(2**32)).random_base2(5)[:30].astype('float32')*10-5
    trace=[];switch=None
    for step in range(budget-10):
        if method in ('W','WS','WA') and len(x)==40:
            active=False;gp=base.GP(x.numpy(),(y.numpy().astype(float)-p.center)/p.scale)
            switch=dict(count=len(gp.x),targets=gp.y.copy())
        if method=='S' and len(x)<40:
            q=sobol[step:step+1];assert cdist(q,x.numpy()).min()>1e-6
            i,pm,mu,sd,score=choose(q,x,y,p,gp,False,'O')
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
def jobs():return [(f,i,s,m) for f in range(36) for i in range(6) for m in METHODS for s in (range(3) if m in SEEDED else [0])]
def collect(j):
    store=CaseStore(RUN/'cases',verify());name='_'.join(map(str,j))
    with store.lock(name):
        if store.load(name,dict(job=j)) is None:store.save(name,dict(job=j),rollout(*j,role=ROLE) if j[3] in ('WS','WA') else st.rollout(*j,role=ROLE))
def run():
    start=time.time()
    with ProcessPoolExecutor(max_workers=24) as ex:
        fs=[ex.submit(collect,j) for j in jobs()]
        for n,f in enumerate(as_completed(fs),1):
            f.result()
            if n%144==0:print(f'{n}/3240, {time.time()-start:.1f}s',flush=True)
    write_json(RUN/'COMPLETE.json',dict(cases=3240,calls=1944000,seconds=time.time()-start))

def contracts():
    verify()
    if (RUN/'CONTRACTS.json').exists():return
    role='cold_start_components_contract_v1';rows={}
    for k,fn,m,z in [('copyW',rollout,'W',False),('originalW',st.rollout,'W',False),('zeroW',rollout,'W',True),('originalO',st.rollout,'O',False)]:rows[k]=fn(0,0,0,m,45,role,z)
    for a,b in [('copyW','originalW'),('zeroW','originalO')]:
        assert torch.equal(rows[a]['x'],rows[b]['x']) and torch.equal(rows[a]['y'],rows[b]['y'])
    assert rows['copyW']['switch']['count']==40
    save_torch(RUN/'contracts.pt',rows);write_json(RUN/'CONTRACTS.json',dict(calls=180,copied_loop_equivalence=True,zero_prior_equivalence=True,switch=True))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['freeze','contracts','run']);a=p.parse_args();globals()[a.phase]()
