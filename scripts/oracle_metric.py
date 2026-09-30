"""Privileged quadratic-metric diagnostic, not a learned method."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,time,argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import cold_start_components as parent
from roopf.experiment_io import CaseStore,write_json,save_torch,sha256,seed_for
from roopf.revision_tasks import ProceduralTask
st=parent.st;RUN=ROOT/'results/oracle_metric_v1';OUT=ROOT/'docs/revision/oracle_metric';ROLE='oracle_metric_v1'
METHODS=('O','WA','T40','T600','R40');SOURCES=('scripts/oracle_metric.py','docs/experiments/ORACLE_METRIC_PROTOCOL.md')
def freeze():
    parent.verify();RUN.mkdir(exist_ok=True)
    if (RUN/'identity.json').exists():return verify()
    write_json(RUN/'identity.json',dict(parent=sha256(parent.RUN/'identity.json'),sources={p:sha256(ROOT/p) for p in SOURCES},time=time.time()))
def verify():
    parent.verify();z=json.loads((RUN/'identity.json').read_text());assert sha256(parent.RUN/'identity.json')==z['parent']
    for p,h in z['sources'].items():assert sha256(ROOT/p)==h
    return z
class MetricGP:
    def __init__(self,x,y,A):self.A=A;self.gp=st.base.GP(np.asarray(x,dtype=float)@A,y)
    def predict(self,q):return self.gp.predict(np.asarray(q,dtype=float)@self.A)
    def append(self,x,y):self.gp.append(np.asarray(x,dtype=float)@self.A,y)
def transform(task,method,role,f,i):
    if method=='O':return np.eye(20)
    R=task.params['rotation'].numpy().astype(float)
    if method=='R40':R=np.linalg.qr(np.random.default_rng(seed_for(role,'random_metric',f,i)).normal(size=(20,20)))[0]
    axis=task.params['axis'].numpy().astype(float);return R@np.diag(np.sqrt(axis/axis.mean()))
@torch.no_grad()
def rollout(f,i,m,budget=600,role=ROLE):
    torch.set_num_threads(1);start=time.perf_counter();task=ProceduralTask(f,i,role,dim=20);x=st.inputs(f,i,role,10);y=task.calfitness(x[None])[0]
    p=st.base.Prior(f,0,'zero',x,y);A=transform(task,m,role,f,i);gp=MetricGP(x.numpy(),(y.numpy().astype(float)-p.center)/p.scale,A);tr=[];switch=None
    for step in range(budget-10):
        if m in ('T40','R40') and len(x)==40:
            gp=MetricGP(x.numpy(),(y.numpy().astype(float)-p.center)/p.scale,np.eye(20));switch=gp.gp.y.copy()
        q=st.pool(x,y.numpy(),f,i,role,step);chosen=st.choose(q,x,y.numpy(),p,gp,False,'O');k=chosen[0];point=torch.from_numpy(q[k:k+1]);v=task.calfitness(point[None])[0,0];std=(float(v)-p.center)/p.scale;gp.append(q[k],std)
        tr.append([*chosen,std]);x=torch.cat((x,point));y=torch.cat((y,v[None]))
    assert task.points==budget and task.diagnostic_points==0;p.verify()
    return dict(fid=f,instance=i,method=m,x=x,y=y,A=A,trace=np.array(tr),switch=switch,calls=task.points,teacher_calls=0,seconds=time.perf_counter()-start)
def contracts():
    verify()
    if (RUN/'CONTRACTS.json').exists():return
    g=np.random.default_rng(202609306);x=g.normal(size=(40,20));q=g.normal(size=(17,20));y=g.normal(size=40);Q=np.linalg.qr(g.normal(size=(20,20)))[0]
    err=float(abs(st.base.kernel(x,q)-st.base.kernel(x@Q,q@Q)).max());assert err<1e-12
    A=Q@np.diag(np.sqrt(np.linspace(.1,3,20)/np.linspace(.1,3,20).mean()));gp=MetricGP(x[:10],y[:10],A);errors=[]
    for n in range(10,40):
        gp.append(x[n],y[n]);dense=MetricGP(x[:n+1],y[:n+1],A);a,b=gp.predict(q);c,d=dense.predict(q);errors.append(float(max(abs(a-c).max(),abs(b-d).max())))
    assert max(errors)<1e-9 and abs(np.trace(A@A.T)-20)<1e-12
    a=rollout(0,0,'O',45,'oracle_metric_contract_v1');b=st.rollout(0,0,0,'O',45,'oracle_metric_contract_v1')
    assert torch.equal(a['x'],b['x']) and torch.equal(a['y'],b['y'])
    save_torch(RUN/'contracts.pt',dict(metric=a,parent=b));write_json(RUN/'CONTRACTS.json',dict(calls=90,rotation_invariance_error=err,incremental_error=max(errors),identity_equivalence=True))
def jobs():return [(f,i,m) for f in range(36) for i in range(4) for m in METHODS]
def collect(j):
    store=CaseStore(RUN/'cases',verify());name='_'.join(map(str,j))
    with store.lock(name):
        if store.load(name,dict(job=j)) is None:
            f,i,m=j;v=parent.rollout(f,i,0,'WA',role=ROLE) if m=='WA' else (st.rollout(f,i,0,'O',role=ROLE) if m=='O' else rollout(*j));store.save(name,dict(job=j),v)
def run():
    start=time.time()
    with ProcessPoolExecutor(max_workers=24) as ex:
        fs=[ex.submit(collect,j) for j in jobs()]
        for n,f in enumerate(as_completed(fs),1):
            f.result()
            if n%72==0:print(f'{n}/720 {time.time()-start:.1f}s',flush=True)
    write_json(RUN/'COMPLETE.json',dict(calls=432000,cases=720,seconds=time.time()-start))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['freeze','contracts','run']);a=p.parse_args();globals()[a.phase]()
