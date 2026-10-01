"""Independent conditioning and backward error at all failed GPU audit states."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import json,sys
from pathlib import Path
import numpy as np
import torch
from scipy.linalg import cho_factor,cho_solve
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.report_strong_online_baselines_v2 import load
from scripts import strong_online_baselines_v2 as s
from roopf.fitted_online_v2 import model_for
from roopf.hpob_transfer_v1 import log_ei
from roopf.experiment_io import write_json


def main():
    torch.set_num_threads(1)
    inventory=json.loads((s.OUT/'REPLAY_INVENTORY.json').read_text());rows=[]
    for failure in inventory['summary']['failures']:
        job=failure['job'];v=load(tuple(job));n=failure['paid']
        info=next(r for r in v['trace'] if r['paid_before']==n)
        states={};matrices={}
        for device in ('cpu','cuda'):
            m=model_for(v['x'][info['offset']:n],v['y'][info['offset']:n],job[2],device)
            with torch.no_grad():
                for name,p in m.named_parameters():p.copy_(info['parameters'][name].to(p))
            m.eval();x=m.train_inputs[0];y=m.train_targets;point=((info['point'].to(x)+5)/10)[None]
            with torch.no_grad():
                post=m.posterior(point.reshape(1,1,20).repeat(522,1,1))
                mean=float(post.mean[0,0,0]);var=float(post.variance[0,0,0])
                K=m.covar_module(x).to_dense()+m.likelihood.noise*torch.eye(len(x),dtype=x.dtype,device=device)
                c=m.covar_module(x,point).to_dense();r=(y-m.mean_module(x))[:,None]
                alpha=torch.cholesky_solve(r,torch.linalg.cholesky(K))
                residual=float(torch.linalg.norm(K@alpha-r)/(torch.linalg.norm(K)*torch.linalg.norm(alpha)+torch.linalg.norm(r)))
            states[device]=dict(mean=mean,variance=var,logei=float(log_ei(torch.tensor(mean,dtype=torch.double),torch.tensor(var,dtype=torch.double),(-v['y'][info['offset']:n].double()).max())),backward_error=residual)
            matrices[device]=K.detach().cpu().numpy()
            if device=='cpu':
                xx=x.detach().numpy();qq=point.numpy();yy=y.numpy()
                ls=m.covar_module.base_kernel.lengthscale.detach().numpy().reshape(-1)
                amp=float(m.covar_module.outputscale);noise=float(m.likelihood.noise);const=float(m.mean_module.constant)
                ym=float(m.outcome_transform.means);ys=float(m.outcome_transform.stdvs)
        def kernel(a,b):
            d=np.sqrt((((a[:,None,:]-b[None,:,:])/ls)**2).sum(-1));z=np.sqrt(5)*d
            return amp*(1+z+z*z/3)*np.exp(-z)
        # Pairwise coordinate subtraction avoids squared-distance cancellation.
        K=kernel(xx,xx)+noise*np.eye(len(xx));c=kernel(xx,qq)
        fact=cho_factor(K,lower=True);alpha=cho_solve(fact,yy-const)
        mu=float(const+(c.T@alpha)[0]);var=float(amp-(c.T@cho_solve(fact,c))[0,0])
        raw_mu=ym+ys*mu;raw_var=ys*ys*var
        best=float((-v['y'][info['offset']:n].double()).max())
        rows.append(dict(job=job,paid=n,recorded_logei=info['logei'],states=states,
            noise=noise,outputscale=amp,condition_number=float(np.linalg.cond(K)),
            cpu_gpu_kernel_max_difference=float(np.abs(matrices['cpu']-matrices['cuda']).max()),
            stable_kernel_cpu_max_difference=float(np.abs(K-matrices['cpu']).max()),
            independent=dict(mean=raw_mu,variance=raw_var,logei=float(log_ei(torch.tensor(raw_mu,dtype=torch.double),torch.tensor(raw_var,dtype=torch.double),best))),
            interpretation='Finite precision sensitivity in tightly clustered, low-noise Matérn posterior; original GPU gate remains failed'))
    write_json(s.OUT/'NUMERICAL_DIAGNOSIS.json',dict(rows=rows,additional_objective_calls=0,gate_relaxed=False))
    for r in rows:print(r,flush=True)


if __name__=='__main__':main()
