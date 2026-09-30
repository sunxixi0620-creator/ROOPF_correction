"""Replay replicated decisions using saved paid observations; zero new labels."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import torch
from scipy.stats import qmc
from scipy.spatial.distance import pdist
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import cold_start_replication as study
from roopf.experiment_io import CaseStore,write_json,seed_for,fingerprint
st=study.st

def check(j):
    torch.set_num_threads(1);v=CaseStore(study.RUN/'cases',study.verify()).load('_'.join(map(str,j)),dict(job=j));f,inst,s,m=j;x,y=v['x'],v['y']
    assert v['calls']==600 and v['teacher_calls']==0 and len(y)==600 and torch.isfinite(y).all() and pdist(x.numpy()).min()>1e-6
    assert torch.equal(x[:10],st.inputs(f,inst,study.ROLE,10));p=st.base.Prior(f,s,'correct' if m=='W' else 'zero',x[:10],y[:10]);assert p.identity==v['prior_identity'];error=0
    if m=='W':assert np.array_equal(v['switch']['targets'],(y[:40].numpy().astype(float)-p.center)/p.scale)
    for step in (0,29,30,289,589):
        n=10+step;active=m=='W' and n<40
        if m=='W' and n>=40:targets=(y[:n].numpy().astype(float)-p.center)/p.scale
        else:
            init=(y[:10].numpy().astype(float)-p.center)/p.scale-(p(x[:10].numpy()) if active else 0);trace=v['trace'][:step];targets=np.r_[init,trace[:,5]-trace[:,1]]
        gp=st.base.GP(x[:n].numpy(),targets)
        if m=='S' and n<40:
            sobol=qmc.Sobol(20,scramble=True,seed=seed_for(study.ROLE,'sobol',f,inst)%(2**32)).random_base2(5)[:30].astype('float32')*10-5;q=sobol[step:step+1]
        else:q=st.pool(x[:n],y[:n].numpy(),f,inst,study.ROLE,step)
        new=st.choose(q,x[:n],y[:n].numpy(),p,gp,active,m);old=v['trace'][step,:5]
        assert new[0]==int(old[0]) and np.array_equal(q[new[0]],x[n].numpy());assert np.allclose(new,old,rtol=1e-7,atol=1e-8)
        error=max(error,float(abs(np.array(new)-old).max()))
    p.verify();return ((f,inst),fingerprint(dict(x=x[:10],y=y[:10])),error)
def main():
    initial={};errors=[]
    with ProcessPoolExecutor(max_workers=24) as pool:
        for n,(key,identity,err) in enumerate(pool.map(check,study.jobs()),1):
            initial.setdefault(key,set()).add(identity);errors.append(err)
            if n%288==0:print(f'verified {n}/1440',flush=True)
    assert len(initial)==288 and all(len(v)==1 for v in initial.values());study.OUT.mkdir(parents=True,exist_ok=True)
    write_json(study.OUT/'VERIFICATION.json',dict(cases=1440,decisions=7200,task_instances=288,shared_initializations=True,budget_and_unique=True,max_error=max(errors),extra_calls=0))
if __name__=='__main__':main()
