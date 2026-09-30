"""Numerical contracts and resource selection without new objective queries."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys,time,json
import multiprocessing as multiprocessing
import numpy as np
import torch
from scipy.optimize._numdiff import approx_derivative
import mpmath as mp
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import acquisition_factorial as s
from roopf.acquisition_search import SmoothPrior,SmoothAcquisition,log_ei_parts,propose
from roopf.experiment_io import write_json,sha256

def fixture(n=150,kind='correct',seed=982):
    torch.set_num_threads(1);g=np.random.default_rng(seed)
    x=torch.tensor(g.uniform(-4,4,(n,20)),dtype=torch.float32);y=torch.tensor(g.normal(size=n),dtype=torch.float32)
    p=s.st.base.Prior(0,0,kind,x[:10],y[:10]);sp=SmoothPrior(p)
    gp=s.st.base.GP(x.numpy(),(y.numpy().astype(float)-p.center)/p.scale-p(x.numpy()))
    q=g.uniform(-3,3,(8,20));return x,y,p,sp,gp,q

def microjob(i):
    n=(20,150,290)[i%3];kind=('correct','analytic','zero')[i%3]
    x,y,p,sp,gp,q=fixture(n,kind,900+i)
    q=s.st.pool(x,y.numpy(),0,i,'acquisition_resource_only',n-10)
    start=time.perf_counter();_,_,meta=propose(q,x,y.numpy(),p,gp,True,sp)
    return time.perf_counter()-start,meta['nfev']

def torch_reference(gp,p,q,device):
    # Independent autograd route through the GP triangular solve and prior.
    tx=torch.as_tensor(gp.x,device=device,dtype=torch.double)
    tq=torch.tensor(q,device=device,dtype=torch.double,requires_grad=True)
    L=torch.as_tensor(gp.L,device=device,dtype=torch.double)
    targets=torch.as_tensor(gp.y,device=device,dtype=torch.double)
    r=((tx[:,None]-tq[None]).square().sum(-1)/125+1e-30).sqrt();a=5**.5*r
    k=(1+a+a.square()/3)*torch.exp(-a)
    alpha=torch.cholesky_solve(targets[:,None],L)[:,0]
    v=torch.linalg.solve_triangular(L,k,upper=False)
    mu=k.T@alpha;sd=(1-v.square().sum(0)).clamp_min(1e-14).sqrt()
    sp=SmoothPrior(p,device)
    if sp.kind=='analytic':mu=mu+((tq.square()/25).mean(-1)-sp.radius_center)/sp.radius_scale
    elif sp.kind!='zero':mu=mu+sp.model(sp.cx[None],sp.cy[None],tq[None])[0]
    dm,=torch.autograd.grad(mu.sum(),tq,retain_graph=True);ds,=torch.autograd.grad(sd.sum(),tq)
    return [a.detach().cpu().numpy() for a in (mu,sd,dm,ds)]

def main():
    s.OUT.mkdir(parents=True,exist_ok=True);s.parent.verify()
    result=dict(objective_calls=0,training_epochs=0,sources={p:sha256(ROOT/p) for p in s.SOURCES},checks=[])
    mp.mp.dps=80;errors=[]
    for u in (-1000,-100,-21,-20,-10,-1,0,10,1000):
        z=mp.mpf(u);h=mp.exp(-z*z/2)/mp.sqrt(2*mp.pi)+z*mp.erfc(-z/mp.sqrt(2))/2
        truth=float(mp.log(h));estimate=float(log_ei_parts(np.array([-u]),np.ones(1),0)[0][0])
        errors.append(abs(truth-estimate));assert abs(truth-estimate)<1e-8
    result['logei_max_error']=max(errors)
    for kind in ('zero','analytic','correct'):
        x,y,p,sp,gp,q=fixture(40,kind);q=q[:3]
        acq=SmoothAcquisition(gp,sp,True,-2);mu,sd,dm,ds=acq.posterior(q)
        om,os=gp.predict(q);mean_error=float(abs(mu-om-p(q)).max());sd_error=float(abs(sd-os).max())
        assert mean_error<2e-6 and sd_error<1e-10
        f,grad=acq.value_gradient(q)
        finite=approx_derivative(lambda z:acq.value_gradient(z.reshape(3,20))[0],q.ravel(),method='3-point')
        analytic=np.zeros_like(finite)
        for i in range(3):analytic[i,20*i:20*(i+1)]=grad[i]
        gradient_error=float(abs(finite-analytic).max());assert gradient_error<2e-6
        result['checks'].append(dict(kind=kind,mean_error=mean_error,sd_error=sd_error,gradient_error=gradient_error))
    # Measure parallel throughput before initializing CUDA.
    cpu={}
    for workers in (8,16,24):
        with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as ex:
            list(ex.map(microjob,range(workers)))
            start=time.perf_counter()
            values=list(ex.map(microjob,range(96)))
            elapsed=time.perf_counter()-start
        cpu[str(workers)]=dict(seconds=elapsed,decisions_per_second=96/elapsed,mean_evaluations=float(np.mean([v[1] for v in values])))
        print('cpu',workers,cpu[str(workers)],flush=True)
    result['cpu_resources']=cpu;result['workers']=int(max(cpu,key=lambda k:cpu[k]['decisions_per_second']))
    result['gpu_checks']=[];timings={}
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32=False
        for kind in ('zero','analytic','correct'):
            x,y,p,sp,gp,q=fixture(150,kind);ref=SmoothAcquisition(gp,sp,True,-2).posterior(q)
            for device in ('cpu','cuda'):
                observed=torch_reference(gp,p,q,device)
                err=max(float(abs(a-b).max()) for a,b in zip(ref,observed));assert err<1e-8
                result['gpu_checks'].append(dict(kind=kind,device=device,posterior_and_derivative_max_error=err))
        x,y,p,sp,gp,q=fixture(150,'correct')
        for count in (8,512):
            query=np.random.default_rng(35).uniform(-4,4,(count,20))
            for device in ('cpu','cuda'):
                # Reused model, actual per-call autograd+host transfer cost.
                model=SmoothPrior(p,device);model.value_gradient(query)
                if device=='cuda':torch.cuda.synchronize()
                begin=time.perf_counter()
                for _ in range(20):model.value_gradient(query)
                if device=='cuda':torch.cuda.synchronize()
                timings[f'{device}_{count}']=(time.perf_counter()-begin)/20
        result['gpu_name']=torch.cuda.get_device_name(0)
        result['neural_gradient_seconds']=timings
    result['main_backend']='CPU analytic GP plus neural autograd; GPU independently checks posterior/derivatives'
    write_json(s.OUT/'NUMERICAL_AND_RESOURCES.json',result)
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
