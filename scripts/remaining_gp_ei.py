"""Sequential GP-EI baseline with versioned sklearn regression, 300 NFE."""
import argparse,json,time,multiprocessing,warnings,hashlib
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import numpy as np
from scipy.stats import norm,qmc
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel,Matern
from sklearn.exceptions import ConvergenceWarning
import sklearn
from independent_benchmarks import ExternalProblem
from supplementary_experiments import write_csv

def optimize_kernel(obj,theta,bounds):
    res=minimize(obj,theta,method='L-BFGS-B',jac=True,bounds=bounds,options={'maxiter':60})
    return res.x,res.fun

def run(suite,fid,instance,seed):
    p=ExternalProblem(suite,fid,instance);lo,hi=p.fun['xlb'],p.fun['xub']
    rng=np.random.default_rng(seed);x=list(qmc.LatinHypercube(10,seed=seed).random(20));y=[]
    start=time.perf_counter()
    for v in x:y.append(p.evaluate(lo+(hi-lo)*v))
    kernel=ConstantKernel(1.,(.01,100.))*Matern(np.ones(10)*.2,(.01,10.),nu=2.5)
    for n in range(20,300):
        gp=GaussianProcessRegressor(kernel=kernel,alpha=1e-6,normalize_y=True,
            optimizer=optimize_kernel if n%20==0 else None,n_restarts_optimizer=0)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',ConvergenceWarning);gp.fit(np.array(x),np.array(y))
        kernel=gp.kernel_
        # Fixed mixed acquisition search: global Sobol + local normal candidates.
        global_x=qmc.Sobol(10,scramble=True,seed=seed+1009*n).random_base2(9)
        local_x=np.clip(np.array(x)[np.argmin(y)]+rng.normal(0,.1,(512,10)),0,1)
        candidates=np.vstack([global_x,local_x]);mu,std=gp.predict(candidates,return_std=True)
        std=np.maximum(std,1e-12);improvement=min(y)-mu;z=improvement/std
        ei=improvement*norm.cdf(z)+std*norm.pdf(z)
        # Repeated observations consume budget but provide no noiseless novelty.
        duplicate=np.min(np.sum((candidates[:,None,:]-np.asarray(x)[None,:,:])**2,axis=2),axis=1)<1e-16
        ei[duplicate]=-np.inf;v=candidates[np.argmax(ei)]
        x.append(v);y.append(p.evaluate(lo+(hi-lo)*v))
    assert p.calls==300 and np.isfinite(p.trace).all()
    return {'suite':suite,'fid':fid,'instance':instance,'seed':seed,'method':'gp_ei',
        'final':p.trace[-1],'actual_nfe':p.calls,'seconds':time.perf_counter()-start},np.array(p.trace)

def worker(job):
    suite,fid,inst,seed,out=job;out=Path(out);stem=f'{suite}_{fid}_{inst}_{seed}'
    p=out/(stem+'.json')
    if p.exists():return json.loads(p.read_text())
    row,trace=run(suite,fid,inst,seed);np.save(out/(stem+'.npy'),trace);p.write_text(json.dumps(row,indent=2));return row

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=8);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    cases=[('coco',f,i) for f in range(1,25) for i in (101,102)]+[('cec2017',f,1) for f in range(10,30)]
    protocol={'cases':cases,'seeds':list(range(20265000,20265010)),'nfe':300,'init':'20 LHS',
        'kernel':'constant * ARD Matern5/2','noise':1e-6,'normalize_y':True,'hyperparameter_refit_every':20,
        'hyperparameter_maxiter':60,'restarts':0,'EI_xi':0,'acquisition_candidates':'512 global Sobol +512 clipped local normal sigma=.1',
        'sklearn':sklearn.__version__,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'reference':'https://scikit-learn.org/stable/modules/gaussian_process.html'}
    protocol=json.loads(json.dumps(protocol));p=a.output/'protocol.json'
    if p.exists():assert json.loads(p.read_text())==protocol
    else:p.write_text(json.dumps(protocol,indent=2))
    rows=[]
    with ProcessPoolExecutor(a.workers,mp_context=multiprocessing.get_context('spawn')) as ex:
        fs=[ex.submit(worker,(*c,s,str(a.output))) for c in cases for s in protocol['seeds']]
        for i,f in enumerate(as_completed(fs)):
            rows.append(f.result())
            if (i+1)%10==0:print(i+1,'/',len(fs),flush=True)
    write_csv(a.output/'raw_results.csv',rows);(a.output/'COMPLETE').write_text(str(len(rows)))

if __name__=='__main__':main()
