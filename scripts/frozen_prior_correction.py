"""Fixed conditional prior, exact online GP corrections, matched search controls."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import sys,json,time,argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import torch
from scipy.linalg import cholesky,solve_triangular,cho_solve
from scipy.spatial.distance import cdist
from scipy.special import ndtr
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import fewshot_prior as parent
from roopf.revision_tasks import ProceduralTask
from roopf.experiment_io import sha256,fingerprint,seed_for,write_json,CaseStore,save_torch
RUN=ROOT/'results/frozen_prior_correction_20260930';OUT=ROOT/'docs/revision/frozen_prior_correction'
METHODS=('P','O100','O60','F','Mean','Shuffled','Blend','Analytic')
SEEDED=('P','F','Mean','Shuffled','Blend')
SOURCES=('scripts/frozen_prior_correction.py','docs/experiments/FROZEN_PRIOR_CORRECTION_PROTOCOL.md','roopf/revision_tasks.py','roopf/experiment_io.py')
NOISE=1e-4

def kernel(x,z):
    d=cdist(np.asarray(x)/10,np.asarray(z)/10)/(20**.5*.25);a=5**.5*d
    return (1+a+a*a/3)*np.exp(-a)

class GP:
    def __init__(self,x,y):
        self.x=np.asarray(x,dtype=np.float64).copy();self.y=np.asarray(y,dtype=np.float64).copy()
        self.L=cholesky(kernel(self.x,self.x)+NOISE*np.eye(len(x)),lower=True,check_finite=False)
    def predict(self,q):
        k=kernel(self.x,q);alpha=cho_solve((self.L,True),self.y,check_finite=False)
        v=solve_triangular(self.L,k,lower=True,check_finite=False);var=1-np.sum(v*v,axis=0)
        assert var.min()>-1e-8
        return k.T@alpha,np.sqrt(np.maximum(var,1e-14))
    def append(self,x,y):
        x=np.asarray(x,dtype=np.float64).reshape(1,20);k=kernel(self.x,x)[:,0]
        v=solve_triangular(self.L,k,lower=True,check_finite=False);diag=1+NOISE-v@v;assert diag>0
        n=len(self.x);L=np.zeros((n+1,n+1));L[:n,:n]=self.L;L[n,:n]=v;L[n,n]=np.sqrt(diag)
        self.L=L;self.x=np.vstack((self.x,x));self.y=np.append(self.y,y)

class Prior:
    def __init__(self,fid,seed,kind,cx,cy):
        self.cx=cx[:40].clone();self.cy=cy[:40].clone();self.center=float(self.cy.mean());self.scale=max(float(self.cy.std(unbiased=False)),1e-6)
        self.kind=kind;self.model=None
        if kind in ('correct','shuffled'):
            fold=(fid//3)%3;self.model=parent.net(fold,seed).eval().requires_grad_(False)
            self.model.load_state_dict(torch.load(parent.RUN/f'model_{fold}_{kind}_{seed}.pt',weights_only=False)['state'])
        self.identity=fingerprint(dict(cx=self.cx,cy=self.cy,kind=kind,weights=self.model.state_dict() if self.model else None))
    @torch.no_grad()
    def __call__(self,q):
        q=torch.as_tensor(np.asarray(q),dtype=torch.float32)
        if self.kind=='zero':return np.zeros(len(q))
        if self.kind=='analytic':
            cr=(self.cx/5).square().mean(-1);qr=(q/5).square().mean(-1)
            return ((qr-cr.mean())/cr.std(unbiased=False).clamp_min(.01)).double().numpy()
        return self.model(self.cx[None],((self.cy-self.center)/self.scale)[None],q[None])[0].double().numpy()
    def verify(self):
        assert fingerprint(dict(cx=self.cx,cy=self.cy,kind=self.kind,weights=self.model.state_dict() if self.model else None))==self.identity

def freeze():
    RUN.mkdir(parents=True,exist_ok=True);parent.verify()
    if (RUN/'identity.json').exists():verify();return
    models=json.loads((parent.RUN/'MODELS_FROZEN.json').read_text())
    for p,h in models.items():assert sha256(parent.RUN/p)==h
    write_json(RUN/'identity.json',dict(version='frozen_prior_correction_v1',frozen=time.time(),sources={p:sha256(ROOT/p) for p in SOURCES},
        parent_identity=sha256(parent.RUN/'identity.json'),models=models,original={p:sha256(ROOT/p) for p in ('checkpoints/anchor_policy_d10.pt','checkpoints/residual_selector_generated36_d10.pt')}))
def verify():
    z=json.loads((RUN/'identity.json').read_text());parent.verify()
    for p,h in {**z['sources'],**z['original']}.items():assert sha256(ROOT/p)==h,p
    assert sha256(parent.RUN/'identity.json')==z['parent_identity']
    for p,h in z['models'].items():assert sha256(parent.RUN/p)==h
    return z
def initial(fid,role):
    g=torch.Generator().manual_seed(seed_for(role,'initial',fid));return 10*torch.rand(100,20,generator=g)-5
def pool_for(ax,ay,fid,step,role,budget):
    g=torch.Generator().manual_seed(seed_for(role,'pool',fid,step));remaining=(budget-len(ax))/budget;best=ax[int(np.argmin(ay))]
    return torch.cat((10*torch.rand(64,20,generator=g)-5,(best+10*(.05+.35*remaining)*torch.randn(64,20,generator=g)).clamp(-5,5))).numpy()
def make_state(fid,seed,method,ax,ay,zero=False,disabled=False):
    kind='zero' if method.startswith('O') or zero else ('shuffled' if method=='Shuffled' else ('analytic' if method=='Analytic' else 'correct'))
    p=Prior(fid,seed,kind,ax,ay);start=0 if method=='O100' else 40;gp=None
    if method!='P' and not disabled:
        values=(ay.numpy()-p.center)/p.scale
        residual=values[start:] if method=='Blend' else values[start:]-p(ax[start:].numpy())
        gp=GP(ax[start:].numpy(),residual)
    return p,gp
def decision(method,p,gp,ax,ay,q):
    pm=p(q);mu=pm.copy();sd=np.zeros(len(q))
    if gp is not None:
        gm,sd=gp.predict(q)
        if method=='Blend':mu=.5*(pm+gm);sd=.5*sd
        else:mu=pm+gm
    if method in ('P','Mean') or gp is None:score=-mu
    else:
        delta=(float(np.min(ay))-p.center)/p.scale-mu;z=delta/np.maximum(sd,1e-14)
        score=delta*ndtr(z)+sd*np.exp(-.5*z*z)/np.sqrt(2*np.pi)
    score[cdist(q,ax.numpy()).min(1)<=1e-6]=-np.inf;i=int(np.argmax(score));assert np.isfinite(score[i])
    return i,pm,mu,sd,float(score[i])

@torch.no_grad()
def rollout(fid,seed,method,budget=600,role='frozen_prior_correction_v1',zero=False,disabled=False):
    torch.set_num_threads(1);task=ProceduralTask(fid,0,role,dim=20);ax=initial(fid,role);ay=task.calfitness(ax[None])[0]
    ix=ax.clone();iy=ay.clone();p,gp=make_state(fid,seed,method,ax,ay,zero,disabled);steps=[];point=[];vals=[];trail=[]
    probe=ix[:8].numpy();before=p(probe).copy();start=time.perf_counter()
    for step in range(budget-100):
        q=pool_for(ax,ay.numpy(),fid,step,role,budget);count=task.points;i,pm,mu,sd,score=decision(method,p,gp,ax,ay.numpy(),q)
        assert task.points==count and task.diagnostic_points==0
        chosen=torch.from_numpy(q[i:i+1]);value=task.calfitness(chosen[None])[0,0];v=float(value)
        standardized=(v-p.center)/p.scale
        if gp is not None:gp.append(q[i],standardized if method=='Blend' else standardized-pm[i])
        steps.append([i,pm[i],mu[i],sd[i],score,standardized]);point.append(chosen[0]);vals.append(value)
        ax=torch.cat((ax,chosen));ay=torch.cat((ay,value[None]));trail.append(float(ay.min()))
    p.verify();assert np.array_equal(before,p(probe));assert task.points==budget and task.diagnostic_points==0
    g=max(0,float(iy.min()-ay.min()))/max(float(iy.std()),1e-8)
    return dict(fid=fid,seed=seed,method=method,initial_x=ix,initial_y=iy,points=torch.stack(point),values=torch.stack(vals),trail=np.asarray(trail),
        decisions=np.asarray(steps),utility=g/(1+g),calls=task.points,teacher_calls=task.diagnostic_points,seconds=time.perf_counter()-start,prior_identity=p.identity,
        gp_initial_count=0 if gp is None else (100 if method=='O100' else 60),gp_final_count=0 if gp is None else len(gp.x))

def contracts():
    if (RUN/'CONTRACTS.json').exists():return
    verify();g=np.random.default_rng(280930);x=g.uniform(-5,5,(30,20));y=g.normal(size=30);q=g.uniform(-5,5,(12,20));gp=GP(x[:10],y[:10]);error=[]
    for n in range(10,30):
        gp.append(x[n],y[n]);direct=GP(x[:n+1],y[:n+1]);a,b=gp.predict(q);c,d=direct.predict(q)
        assert np.allclose(a,c,rtol=1e-10,atol=1e-10) and np.allclose(b,d,rtol=1e-10,atol=1e-10);error.append(float(max(abs(a-c).max(),abs(b-d).max())))
    outputs={}
    for key,method,zero,disabled in [('O60','O60',False,False),('Fzero','F',True,False),('P','P',False,False),('Mean_disabled','Mean',False,True)]:
        outputs[key]=rollout(0,0,method,104,'frozen_prior_correction_contract',zero,disabled)
    assert torch.equal(outputs['O60']['points'],outputs['Fzero']['points']) and torch.equal(outputs['P']['points'],outputs['Mean_disabled']['points'])
    assert len({fingerprint(v['initial_x']) for v in outputs.values()})==1
    save_torch(RUN/'contracts.pt',outputs);write_json(RUN/'CONTRACTS.json',dict(calls=416,incremental_dense_max_error=max(error),zero_prior_equivalence=True,disabled_correction_equivalence=True,prior_frozen=True))

@torch.no_grad()
def diagnostic_case(fid):
    torch.set_num_threads(1);role='frozen_prior_correction_diagnostic';task=ProceduralTask(fid,0,role,dim=20);ax=initial(fid,role);ay=task.calfitness(ax[None])[0]
    gen=torch.Generator().manual_seed(seed_for(role,'queries',fid));q=(10*torch.rand(128,20,generator=gen)-5).numpy();truth=task.calfitness(torch.from_numpy(q)[None])[0].numpy();outputs={}
    for method in ('P','O100','O60','F','Shuffled','Blend','Analytic'):
        for seed in (range(3) if method in SEEDED else [0]):
            p,gp=make_state(fid,seed,method,ax,ay);_,pm,mu,sd,_=decision(method,p,gp,ax,ay.numpy(),q)
            outputs[f'{method}_{seed}']=dict(mean=mu,std=sd,prior=pm,target=(truth-p.center)/p.scale)
    assert task.points==228
    return dict(initial_x=ax,initial_y=ay,query_x=q,query_y=truth,outputs=outputs,calls=task.points)
def diagnostics(workers):
    z=verify();store=CaseStore(RUN/'diagnostics',z);start=time.time()
    jobs=[f for f in range(36) if store.load(f'f{f}',dict(fid=f)) is None]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        fs={pool.submit(diagnostic_case,f):f for f in jobs}
        for future in as_completed(fs):
            f=fs[future];out=future.result()
            with store.lock(f'f{f}'):store.save(f'f{f}',dict(fid=f),out)
    write_json(RUN/'DIAGNOSTICS_COMPLETE.json',dict(calls=8208,cases=36,seconds=time.time()-start))
    print('Predictive diagnostic complete; no model/config changes permitted.',flush=True)

def jobs():return [(f,s,m) for f in range(36) for m in METHODS for s in (range(3) if m in SEEDED else [0])]
def collect(job):
    f,s,m=job;store=CaseStore(RUN/'cases',verify());name=f'f{f}_s{s}_{m}';spec=dict(fid=f,seed=s,method=m)
    with store.lock(name):
        if store.load(name,spec) is not None:return name
        store.save(name,spec,rollout(f,s,m))
    return name
def run(workers,one=False):
    verify();assert (RUN/'CONTRACTS.json').exists();start=time.time();todo=jobs() if not one else [(0,0,'F')]
    with ProcessPoolExecutor(max_workers=workers if not one else 1) as pool:
        fs=[pool.submit(collect,j) for j in todo]
        for n,f in enumerate(as_completed(fs),1):
            f.result()
            if n%12==0 or n==len(fs):print(f'{n}/{len(fs)} trajectories, {time.time()-start:.1f}s',flush=True)
    write_json(RUN/('SMOKE_TIMING.json' if one else 'SEARCH_COMPLETE.json'),dict(trajectories=len(todo),seconds=time.time()-start,workers=workers if not one else 1))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['freeze','contracts','diagnostics','run','smoke']);p.add_argument('--workers',type=int,default=24);a=p.parse_args()
    if a.phase=='freeze':freeze()
    elif a.phase=='contracts':contracts()
    elif a.phase=='diagnostics':diagnostics(a.workers)
    else:run(a.workers,a.phase=='smoke')
