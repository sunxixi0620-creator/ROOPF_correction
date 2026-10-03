"""Synthetic checks for variance shrinkage, pairing and exact device-local resume."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.check_neural_prior_training_v1 import fake_tasks
from roopf.neural_prior_training_v1 import Episodes,optimizer_for,update,cpu_tree
from roopf.neural_prior_scale_v2 import StableSourceModel,normalize_with_reference
from roopf.experiment_io import write_json,sha256


def main():
    torch.set_num_threads(1)
    yc=torch.tensor([[.889279,.889279,.889279,.8892794376098417,.889279]],dtype=torch.double)
    yq=torch.tensor([[.6274165202108963,.929701]],dtype=torch.double)
    zc,zq,mean,scale=normalize_with_reference(yc,yq,.01)
    expected=np.sqrt((4*.01+4*np.var(yc.numpy(),ddof=1))/8)
    assert abs(float(scale)-expected)<1e-15 and float(zq.abs().max())<4
    assert torch.equal(scale,normalize_with_reference(yc,yq*1e6,.01)[3])
    constant=torch.ones(1,5,dtype=torch.double)
    near=constant.clone();near[0,0]+=1e-7
    assert abs(float(normalize_with_reference(constant,None,.01)[3])-float(normalize_with_reference(near,None,.01)[3]))<1e-12
    errors={};states={}
    for device in ('cpu','cuda'):
        tr=Episodes(fake_tasks('train'),'fake','train',device)
        for kind in ('N','M','PCA'):
            model=StableSourceModel(3,kind,0,'fake',.1).to(device);opt=optimizer_for(model)
            for step in range(1,9):
                update(model,opt,tr.training(0,step))
                if step==4:snap=dict(model=cpu_tree(model.state_dict()),optimizer=cpu_tree(opt.state_dict()))
            resumed=StableSourceModel(3,kind,0,'fake',.1).to(device);resumed.load_state_dict(snap['model'])
            opt2=optimizer_for(resumed);opt2.load_state_dict(snap['optimizer'])
            for step in range(5,9):update(resumed,opt2,tr.training(0,step))
            err=max(float((v-resumed.state_dict()[k]).abs().max()) for k,v in model.state_dict().items())
            assert err<1e-12;errors[device+'_'+kind]=err;states[device,kind]=cpu_tree(model.state_dict())
    a=StableSourceModel(3,'N',0,'fake',.1);b=StableSourceModel(3,'M',0,'fake',.1)
    assert all(torch.equal(a.prior.state_dict()[k],b.prior.state_dict()[k]) for k in a.prior.state_dict())
    a.load_state_dict(states['cpu','N']);b=StableSourceModel(3,'N',0,'fake',.1).cuda();b.load_state_dict(states['cpu','N'])
    with torch.no_grad():
        ac=a(Episodes(fake_tasks('train'),'fake','train','cpu').training(0,9))[0]
        bc=b(Episodes(fake_tasks('train'),'fake','train','cuda').training(0,9))[0].cpu()
    err=float((ac-bc).abs().max());assert err<1e-7
    out=dict(passed=True,fabricated_only=True,query_independent_scale=True,continuous_at_constant_context=True,
        independent_formula_error=abs(float(scale)-expected),max_standardized_synthetic_query=float(zq.abs().max()),
        exact_resume_errors=errors,cpu_cuda_nll_error=err,target_calls=0,real_source_updates=0,
        source_sha256=sha256(ROOT/'roopf/neural_prior_scale_v2.py'),test_sha256=sha256(Path(__file__)))
    write_json(ROOT/'docs/revision/neural_prior_scale_v2/CONTRACTS.json',out);print(out,flush=True)


if __name__=='__main__':main()
