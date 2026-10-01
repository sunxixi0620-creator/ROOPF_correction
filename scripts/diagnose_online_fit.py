"""Reproduce first failed TuRBO case, recording every paid point and fit failure."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import sys,time,traceback
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import strong_online_baselines as s
from roopf import fitted_online as engine
from roopf.experiment_io import save_torch,write_json
RUN=s.RUN/'diagnostic'


def main():
    torch.set_num_threads(1);RUN.mkdir(exist_ok=True)
    original_fit=engine.fit_gpytorch_mll
    def diagnostic_fit(mll,**kwargs):
        try:return original_fit(mll,**kwargs)
        except Exception:
            model=mll.model
            save_torch(RUN/'failed_model.pt',dict(x=model.train_inputs[0].detach().cpu(),
                y=model.train_targets.detach().cpu(),state=model.state_dict(),
                parameters={n:p.detach().cpu() for n,p in model.named_parameters()}))
            raise
    engine.fit_gpytorch_mll=diagnostic_fit
    f,i=0,0;role=s.parent.ROLE
    task=s.parent.parent.ProceduralTask(f,i,role,dim=20)
    x=s.parent.st.inputs(f,i,role,10);y=task.calfitness(x[None])[0]
    state=engine.TrustRegion(best=float(y.min()));params=None;trace=[];start=time.time()
    try:
        while len(x)<300:
            save_torch(RUN/'paid.pt',dict(x=x,y=y,calls=task.points,trace=trace))
            q,info,params=engine.propose(x,y,'TuRBO_LogEI',(f,i,role),state,params)
            v=task.calfitness(q[None])[0,0];x=torch.cat((x,q));y=torch.cat((y,v[None]));trace.append(info);state.update(float(v))
            if len(x)%10==0:print(len(x),time.time()-start,flush=True)
    except Exception as exc:
        save_torch(RUN/'paid.pt',dict(x=x,y=y,calls=task.points,trace=trace))
        write_json(RUN/'FAILURE.json',dict(error=repr(exc),calls=task.points,seconds=time.time()-start,traceback=traceback.format_exc()))
        print(repr(exc),'at',task.points,flush=True)


if __name__=='__main__':main()
