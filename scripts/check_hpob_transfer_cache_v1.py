"""Validate all source caches against CPU conditioning without target responses."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import hpob_transfer_development_v1 as s
from roopf.hpob_transfer_v1 import build_gp,Predictor
from roopf.experiment_io import CaseStore,write_json,sha256


def main():
    torch.set_num_threads(1);identity=s.verify();models=CaseStore(s.RUN/'models',identity);caches=CaseStore(s.RUN/'caches',identity)
    bank={};maximum=0.;checks=0;rows=[]
    for st in s.tasks('source'):
        v=models.load(s.key(st),dict(task=st));assert v is not None
        model=build_gp(v['x'],v['z'])
        with torch.no_grad():
            for name,p in model.named_parameters():p.copy_(v['info']['parameters'][name])
        model.eval();bank[s.key(st)]=Predictor(model)
    for t in s.tasks('development'):
        name=s.key(t);cache=caches.load(name,dict(task=t));assert cache is not None
        x=torch.from_numpy(np.load(s.RUN/'public'/f'{name}.npy'));indices=np.linspace(0,len(x)-1,11,dtype=int)
        assert cache['means'].shape==(len(t['source_groups']),len(x)) and torch.isfinite(cache['means']).all()
        errors=[]
        for k,src in enumerate(cache['sources']):
            assert cache['model_hashes'][src]==sha256(s.RUN/'models'/f'{src}.pt')
            mu,_=bank[src].moments(x[indices],False)
            err=float((mu-cache['means'][k,indices]).abs().max())
            errors.append(err);checks+=len(indices);maximum=max(maximum,err)
        rows.append(dict(task=name,max_error=max(errors)))
    result=dict(caches=len(rows),source_models=len(bank),point_replays=checks,max_error=maximum,tolerance=1e-7,
                passed=maximum<1e-7,target_calls=0,target_responses_read=False,rows=rows)
    write_json(s.OUT/'CACHE_VERIFICATION.json',result);assert result['passed'],maximum
    print('source caches passed',checks,maximum,flush=True)


if __name__=='__main__':main()
