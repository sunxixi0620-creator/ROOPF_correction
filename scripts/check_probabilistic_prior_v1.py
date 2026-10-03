"""Independent algebra, PSD, gradient and CPU/CUDA tests; fabricated labels only."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import sys
from pathlib import Path
import numpy as np
import torch
from scipy.linalg import cho_factor,cho_solve
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from roopf.probabilistic_prior_v1 import (NeuralPrior,BaseCovariance,condition,
    conditional_nll,normalize_context,pca_directions,pca_prior)
from roopf.experiment_io import write_json,sha256


def main():
    torch.set_num_threads(1);torch.manual_seed(417)
    dim=6;xc=torch.rand(2,7,dim,dtype=torch.double);xq=torch.rand(2,9,dim,dtype=torch.double)
    yc=torch.randn(2,7,dtype=torch.double);yq=torch.randn(2,9,dtype=torch.double)
    zc,zq,shift,scale=normalize_context(yc,yq)
    assert torch.equal(normalize_context(yc,yq*99)[0],zc)
    prior=NeuralPrior(dim);base=BaseCovariance(dim)
    assert sum(p.numel() for p in prior.parameters())==32*dim+1385
    pc,pq=prior(xc),prior(xq)
    mu,cov=condition(base,xc,zc,xq,pc,pq);loss=conditional_nll(mu,cov,zq).mean()
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in list(prior.parameters())+list(base.parameters()))
    assert torch.linalg.eigvalsh(cov).min()>0
    cpu_mean=mu.detach();cpu_cov=cov.detach();noise=float(base.noise.detach())
    # Independent NumPy/SciPy block-Gaussian calculation.
    err=0.;ls=base.lengthscale.detach().numpy()
    for b in range(2):
        xx=np.concatenate((xc[b].numpy(),xq[b].numpy()))
        ff=np.concatenate((pc[1][b].detach().numpy(),pq[1][b].detach().numpy()))
        m=np.concatenate((pc[0][b].detach().numpy(),pq[0][b].detach().numpy()))
        d=np.linalg.norm((xx[:,None]-xx[None,:])/ls,axis=-1);r=np.sqrt(5)*d
        K=(1+r+r*r/3)*np.exp(-r)+ff@ff.T+noise*np.eye(len(xx))
        L=cho_factor(K[:7,:7],lower=True)
        mean=m[7:]+K[7:,:7]@cho_solve(L,zc[b].numpy()-m[:7])
        covariance=K[7:,7:]-K[7:,:7]@cho_solve(L,K[:7,7:])
        err=max(err,float(np.abs(mean-cpu_mean[b].numpy()).max()),float(np.abs(covariance-cpu_cov[b].numpy()).max()))
    assert err<1e-10,err
    # Zero neural covariance/mean reduces to the ordinary base GP.
    zeros_c=(torch.zeros_like(zc),torch.zeros(2,7,8,dtype=torch.double))
    zeros_q=(torch.zeros_like(zq),torch.zeros(2,9,8,dtype=torch.double))
    ordinary=condition(base,xc,zc,xq,zeros_c,zeros_q)
    K=base(xc,xc)+base.noise*torch.eye(7,dtype=torch.double)
    expected=base(xq,xc)@torch.linalg.solve(K,zc.unsqueeze(-1))
    assert torch.allclose(ordinary[0],expected.squeeze(-1),atol=1e-10,rtol=1e-10)
    # Full task rank recovers empirical covariance; low task count pads zeros.
    predictions=torch.randn(5,19,dtype=torch.double)
    directions,diag=pca_directions(predictions,8);mean,features=pca_prior(predictions,directions)
    centered=predictions-predictions.mean(0)
    pca_error=float((features@features.T-centered.T@centered/4).abs().max())
    assert pca_error<1e-10 and diag['active_rank']==4 and torch.equal(features[:,4:],torch.zeros(19,4,dtype=torch.double))
    cuda_error=None
    if torch.cuda.is_available():
        prior=prior.cuda();base=base.cuda()
        gmu,gcov=condition(base,xc.cuda(),zc.cuda(),xq.cuda(),prior(xc.cuda()),prior(xq.cuda()))
        cuda_error=max(float((gmu.detach().cpu()-cpu_mean).abs().max()),float((gcov.detach().cpu()-cpu_cov).abs().max()))
        assert cuda_error<1e-8,cuda_error
    result=dict(fabricated_only=True,real_training_updates=0,target_calls=0,
        gaussian_block_error=err,pca_covariance_error=pca_error,cpu_cuda_error=cuda_error,
        gradients_finite=True,zero_source_reduces_to_gp=True,context_only_normalization=True,
        source_sha256=sha256(ROOT/'roopf/probabilistic_prior_v1.py'),test_sha256=sha256(Path(__file__)))
    write_json(ROOT/'docs/revision/neural_prior_source_v1/CONTRACTS.json',result)
    print(result,flush=True)


if __name__=='__main__':main()
