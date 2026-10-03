"""Fabricated-data tests for paired episodes, held-out roles and exact resume."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,tempfile,time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from roopf.neural_prior_training_v1 import SourceModel,Episodes,optimizer_for,update,cpu_tree
from roopf.experiment_io import write_json,sha256


def fake_tasks(role):
    rng=np.random.default_rng(124 if role=='train' else 125);tasks=[]
    for i in range(3):
        x=torch.tensor(rng.uniform(size=(96,3)),dtype=torch.double)
        y=(x-.1*i).square().sum(-1)+torch.sin(9*x[:,0])
        tasks.append(dict(name=f'{role}_{i}',group=f'{role}_{i}',role=role,x=x,y=y,
            mean=torch.zeros(96,dtype=torch.double),features=torch.cat((x,torch.zeros(96,5,dtype=torch.double)),dim=1),
            prefixes=[rng.permutation(96)[:40].tolist() for _ in range(2)]))
    return tasks


def main():
    torch.set_num_threads(1);train=Episodes(fake_tasks('train'),'fake','train','cpu')
    validation=Episodes(fake_tasks('validation'),'fake','validation','cpu')
    try:validation.training(0,1);raise RuntimeError('validation must not feed updates')
    except AssertionError:pass
    assert all(torch.equal(a,b) for a,b in zip(train.training(0,1)[:4],train.training(0,1)[:4]))
    a=SourceModel(3,'N',0,'fake');b=SourceModel(3,'M',0,'fake')
    assert all(torch.equal(a.prior.state_dict()[k],b.prior.state_dict()[k]) for k in a.prior.state_dict())
    resume_errors={};cuda_error=None;benchmark={}
    for device in (['cpu','cuda'] if torch.cuda.is_available() else ['cpu']):
        tr=Episodes(fake_tasks('train'),'fake','train',device)
        for kind in ('N','M','PCA'):
            model=SourceModel(3,kind,0,'fake').to(device);opt=optimizer_for(model)
            start=time.perf_counter();snap=None
            for step in range(1,9):
                update(model,opt,tr.training(0,step))
                if step==4:snap=dict(model=cpu_tree(model.state_dict()),optimizer=cpu_tree(opt.state_dict()))
            benchmark[device+'_'+kind]=time.perf_counter()-start
            restored=SourceModel(3,kind,0,'fake').to(device);restored.load_state_dict(snap['model'])
            opt2=optimizer_for(restored);opt2.load_state_dict(snap['optimizer'])
            for step in range(5,9):update(restored,opt2,tr.training(0,step))
            err=max(float((v-restored.state_dict()[k]).abs().max()) for k,v in model.state_dict().items())
            assert err<1e-12,(device,kind,err);resume_errors[device+'_'+kind]=err
            if kind=='M':
                # Unused feature outputs must not train through covariance.
                last=model.prior.network[-1]
                assert torch.equal(last.weight.grad[1:],torch.zeros_like(last.weight.grad[1:]))
            metrics=Episodes(fake_tasks('validation'),'fake','validation',device).validation(model)
            assert np.isfinite(metrics['nll']) and len(metrics['rows'])==96
            if kind=='N' and device=='cpu':cpu_model=cpu_tree(model.state_dict())
        if device=='cuda':
            c=SourceModel(3,'N',0,'fake');c.load_state_dict(cpu_model)
            gpu=SourceModel(3,'N',0,'fake').cuda();gpu.load_state_dict(cpu_model)
            with torch.no_grad():
                one=c(train.training(0,9))[0];two=gpu(tr.training(0,9))[0].cpu()
            cuda_error=float((one-two).abs().max());assert cuda_error<1e-7,cuda_error
    result=dict(fabricated_only=True,real_source_updates=0,target_calls=0,paired_initialization=True,
        validation_gradient_access_rejected=True,exact_resume_errors=resume_errors,cpu_cuda_nll_error=cuda_error,
        eight_update_benchmark_seconds=benchmark,source_sha256=sha256(ROOT/'roopf/neural_prior_training_v1.py'),
        runner_sha256=sha256(ROOT/'scripts/train_neural_prior_v1.py'),test_sha256=sha256(Path(__file__)))
    write_json(ROOT/'docs/revision/neural_prior_training_v1/CONTRACTS.json',result)
    print(result,flush=True)


if __name__=='__main__':main()
