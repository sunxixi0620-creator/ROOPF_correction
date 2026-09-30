"""Frozen cold-start feasibility experiment; independent from released ROOPF."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,time,argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import torch
from scipy.stats import qmc
from scipy.spatial.distance import cdist
from scipy.special import ndtr
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import frozen_prior_correction as base
from roopf.experiment_io import CaseStore,write_json,save_torch,sha256,seed_for
from roopf.revision_tasks import ProceduralTask
RUN=ROOT/'results/cold_start_v1';OUT=ROOT/'docs/revision/cold_start'
SOURCES=('scripts/cold_start.py','docs/experiments/COLD_START_PROTOCOL.md')
METHODS=('O','P','F','W','S')
def freeze():
    base.verify();RUN.mkdir(exist_ok=True)
    if (RUN/'identity.json').exists():return verify()
    write_json(RUN/'identity.json',dict(sources={p:sha256(ROOT/p) for p in SOURCES},parent=sha256(base.RUN/'identity.json'),time=time.time()))
def verify():
    z=json.loads((RUN/'identity.json').read_text());base.verify()
    for p,h in z['sources'].items():assert sha256(ROOT/p)==h
    assert sha256(base.RUN/'identity.json')==z['parent'];return z

def inputs(fid,inst,role,n):
    g=torch.Generator().manual_seed(seed_for(role,'initial',fid,inst));return torch.rand(n,20,generator=g)*10-5

def pool(x,y,fid,inst,role,step):
    g=torch.Generator().manual_seed(seed_for(role,'pool',fid,inst,step))
    return torch.cat((torch.rand(64,20,generator=g)*10-5,(x[int(np.argmin(y))]+10*(.05+.35*(600-len(x))/600)*torch.randn(64,20,generator=g)).clamp(-5,5))).numpy()

def choose(q,x,y,p,gp,active,method):
    pm=p(q) if active else np.zeros(len(q));mu=pm.copy();sd=np.zeros(len(q))
    if gp is not None:gm,sd=gp.predict(q);mu+=gm
    if method=='P':score=-mu
    else:
        delta=(float(min(y))-p.center)/p.scale-mu;z=delta/np.maximum(sd,1e-14)
        score=delta*ndtr(z)+sd*np.exp(-z*z/2)/np.sqrt(2*np.pi)
    score[cdist(q,x.numpy()).min(1)<=1e-6]=-np.inf;i=int(score.argmax());assert np.isfinite(score[i])
    return i,float(pm[i]),float(mu[i]),float(sd[i]),float(score[i])

@torch.no_grad()
def rollout(fid,inst,seed,method,budget=600,role='cold_start_search_v1',zero=False):
    torch.set_num_threads(1);start=time.perf_counter();t=ProceduralTask(fid,inst,role,dim=20)
    x=inputs(fid,inst,role,10);y=t.calfitness(x[None])[0];ix=x.clone();iy=y.clone()
    p=base.Prior(fid,seed,'zero' if zero or method in ('O','S') else 'correct',x,y)
    active=method in ('P','F','W');gp=None
    if method!='P':gp=base.GP(x.numpy(),(y.numpy().astype(float)-p.center)/p.scale-(p(x.numpy()) if active else 0))
    sobol=qmc.Sobol(20,scramble=True,seed=seed_for(role,'sobol',fid,inst)%(2**32)).random_base2(5)[:30].astype('float32')*10-5
    trace=[];switch=None
    for step in range(budget-10):
        if method=='W' and len(x)==40:
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

def contracts():
    if (RUN/'CONTRACTS.json').exists():return
    verify();r={}
    for key,m,z in [('O','O',False),('Wzero','W',True),('F','F',False),('W','W',False)]:r[key]=rollout(0,0,0,m,45,'cold_start_contract_v1',z)
    assert torch.equal(r['O']['x'],r['Wzero']['x']);assert torch.equal(r['F']['x'][:40],r['W']['x'][:40])
    assert r['W']['switch']['count']==40 and not r['W']['trace'][30:,6].any()
    p=base.Prior(0,0,'correct',r['W']['initial_x'],r['W']['initial_y'])
    assert np.array_equal(r['W']['switch']['targets'],(r['W']['y'][:40].numpy().astype(float)-p.center)/p.scale)
    save_torch(RUN/'contracts.pt',r);write_json(RUN/'CONTRACTS.json',dict(calls=180,zero_equivalence=True,shared_prefix=True,all_history_switch=True))

@torch.no_grad()
def diagnostic(fid):
    torch.set_num_threads(1);role='cold_start_diagnostic_v1';t=ProceduralTask(fid,0,role,dim=20)
    x=inputs(fid,0,role,168);y=t.calfitness(x[None])[0];out={}
    for n in (5,10,20,40):
        for seed in range(3):
            for method in ('P','Shuffled','O','F'):
                p=base.Prior(fid,seed,'shuffled' if method=='Shuffled' else ('zero' if method=='O' else 'correct'),x[:n],y[:n])
                q=x[40:].numpy();pm=p(q);mu=pm.copy()
                if method in ('O','F'):
                    gp=base.GP(x[:n].numpy(),(y[:n].numpy().astype(float)-p.center)/p.scale-p(x[:n].numpy()));mu+=gp.predict(q)[0]
                truth=(y[40:].numpy().astype(float)-p.center)/p.scale
                picked=int(mu.argmin());top=set(np.argsort(truth)[:13]);pred=set(np.argsort(mu)[:13])
                out[f'{n}_{seed}_{method}']=dict(mse=float(np.mean((mu-truth)**2)),recall=len(top&pred)/13,regret=float(truth[picked]-truth.min()),improves=bool(y[40+picked]<y[:n].min()),prediction=mu)
    assert t.points==168;return dict(x=x,y=y,metrics=out,calls=168)

def jobs():return [(f,i,s,m) for f in range(36) for i in range(2) for m in METHODS for s in (range(3) if m in ('P','F','W') else [0])]
def collect(job,diag=False):
    store=CaseStore(RUN/('diagnostics' if diag else 'cases'),verify());name='f'+str(job) if diag else '_'.join(map(str,job));spec=dict(job=job)
    with store.lock(name):
        if store.load(name,spec) is None:store.save(name,spec,diagnostic(job) if diag else rollout(*job))
    return name

def run(workers,diag=False):
    verify();todo=list(range(36)) if diag else jobs();start=time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        fs=[ex.submit(collect,j,diag) for j in todo]
        for n,f in enumerate(as_completed(fs),1):
            f.result()
            if n%72==0 or n==len(fs):print(f'{n}/{len(fs)}, {time.time()-start:.1f}s',flush=True)
    write_json(RUN/('DIAGNOSTICS_COMPLETE.json' if diag else 'SEARCH_COMPLETE.json'),dict(cases=len(todo),seconds=time.time()-start,workers=workers))

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('phase',choices=['freeze','contracts','diagnostics','run']);a.add_argument('--workers',type=int,default=24);args=a.parse_args()
    if args.phase=='freeze':freeze()
    elif args.phase=='contracts':contracts()
    else:run(args.workers,args.phase=='diagnostics')
