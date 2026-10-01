"""Paid-data-only GP and RGPE-TAF (2022 bootstrap variant) primitives."""
import time
import warnings
import numpy as np
import torch
from botorch.models import SingleTaskGP
from botorch.models.utils.gpytorch_modules import get_covar_module_with_dim_scaled_prior
from botorch.fit import fit_gpytorch_mll
from botorch.exceptions.errors import ModelFittingError
from botorch.acquisition.analytic import _log_ei_helper
from gpytorch.mlls import ExactMarginalLogLikelihood
from roopf.experiment_io import seed_for


def standardize(y):
    mean=y.mean();scale=y.std(unbiased=True)
    if float(scale)<1e-8:scale=torch.ones_like(scale)
    return (y-mean)/scale,mean,scale


def build_gp(x,z):
    kernel=get_covar_module_with_dim_scaled_prior(x.shape[-1],use_rbf_kernel=False)
    return SingleTaskGP(x,z[:,None],covar_module=kernel,outcome_transform=None).double()


def fit(x,z,identity,previous=None):
    torch.manual_seed(seed_for('hpob_transfer_fit_v1',identity,len(x))%(2**31-1))
    start=time.perf_counter();model=build_gp(x,z)
    if previous:
        with torch.no_grad():
            for name,p in model.named_parameters():p.copy_(previous[name])
    recovered=False
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        try:
            fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood,model),
                optimizer_kwargs={'options':{'maxiter':100,'ftol':1e-9}})
        except ModelFittingError:
            recovered=True;model=build_gp(x,z)
            fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood,model),
                optimizer_kwargs={'options':{'maxiter':100,'ftol':1e-9}})
    model.eval()
    params={name:p.detach().cpu().clone() for name,p in model.named_parameters()}
    return model,dict(parameters=params,recovered=recovered,
        warnings=[str(w.message) for w in caught],seconds=time.perf_counter()-start)


class Predictor:
    """Exact conditioning; avoids constructing a joint candidate covariance."""
    def __init__(self,model):
        self.model=model;self.x=model.train_inputs[0];self.z=model.train_targets
        with torch.no_grad():
            self.k=model.covar_module(self.x).to_dense()
            self.k=self.k+model.likelihood.noise*torch.eye(len(self.x),device=self.x.device,dtype=self.x.dtype)
            self.chol=torch.linalg.cholesky(self.k)
            self.residual=self.z-model.mean_module(self.x)
            self.alpha=torch.cholesky_solve(self.residual[:,None],self.chol)

    @torch.no_grad()
    def moments(self,q,variance=True,batch=2048):
        means=[];vs=[]
        for part in q.split(batch):
            cross=self.model.covar_module(part,self.x).to_dense()
            means.append(self.model.mean_module(part)+(cross@self.alpha).squeeze(-1))
            if variance:
                v=torch.linalg.solve_triangular(self.chol,cross.T,upper=False)
                vs.append((self.model.covar_module(part,diag=True)-v.square().sum(0)).clamp_min(1e-10))
        return torch.cat(means),torch.cat(vs) if variance else None

    @torch.no_grad()
    def loo_mean(self):
        inverse=torch.cholesky_inverse(self.chol)
        return self.z-self.alpha[:,0]/inverse.diag()


def log_ei(mean,variance,best):
    sigma=variance.clamp_min(1e-10).sqrt()
    return sigma.log()+_log_ei_helper((mean-best)/sigma)


def discordances(source_means,target_loo,y):
    """All ordered pairs, including diagonal, exactly as paper Eqs. 3/4."""
    observed=y[:,None]<y[None,:]
    ds=(source_means[:,:,None]<source_means[:,None,:])^observed[None]
    dt=(target_loo[:,None]<y[None,:])^observed
    return np.concatenate((ds,dt[None]),axis=0)


def bootstrap_losses(discord,indices):
    # Resampling pairs is c.T @ discord @ c where c contains bootstrap counts.
    # This retains repeated-index pairs and the target's Eq.4 diagonal terms.
    n=discord.shape[-1]
    counts=np.zeros((len(indices),n),dtype=np.float64)
    np.add.at(counts,(np.repeat(np.arange(len(indices)),n),indices.ravel()),1.)
    return np.einsum('sn,mnk,sk->ms',counts,discord.astype(float),counts,optimize=True)


def rank_weights(source_means,target_loo,y,identity,n,horizon=105):
    rng=np.random.default_rng(seed_for('hpob_rgpe_bootstrap_v1',identity,n))
    indices=rng.integers(0,n,(1000,n))
    losses=bootstrap_losses(discordances(source_means,target_loo,y),indices)
    keep=(1-n/horizon)*(losses[:-1]<losses[-1]).mean(1)
    dropped=rng.random(len(keep))>=keep
    adjusted=losses.copy();adjusted[:-1][dropped]=np.inf
    winners=adjusted==adjusted.min(0,keepdims=True)
    weights=(winners/winners.sum(0,keepdims=True)).mean(1)
    assert np.isfinite(weights).all() and abs(weights.sum()-1)<1e-12
    return weights,dict(drop=dropped.tolist(),keep_probability=keep.tolist(),
                        mean_rank_loss=losses.mean(1).tolist())


def taf_scores(target_logei,source_means,paid,weights):
    """Source-standardized improvements plus target-standardized EI, log space."""
    source_best=source_means[:,paid].max(1,keepdim=True).values
    improvement=(source_means-source_best).clamp_min(0)
    w=torch.as_tensor(weights,dtype=source_means.dtype,device=source_means.device)
    source_part=(w[:-1,None]*improvement).sum(0)
    return torch.logaddexp(target_logei+w[-1].log(),source_part.log())


def propose(x,paid,values,source_means,method,identity,previous=None):
    start=time.perf_counter();n=len(paid)
    common=source_means.mean(0) if method in ('P','A') else None
    if method=='P':
        scores=common.clone();info=dict(parameters=None,warnings=[],seconds=0.,weights=None)
    else:
        z,mu,scale=standardize(values)
        prior=common[paid] if method=='A' else 0
        model,info=fit(x[paid],z-prior,identity,previous)
        predictor=Predictor(model)
        mean,var=predictor.moments(x)
        if method=='A':mean=mean+common
        best=mean[paid].max()
        scores=log_ei(mean,var,best)
        info.update(y_mean=float(mu),y_scale=float(scale),best=float(best),weights=None)
        if method=='R':
            weights,diag=rank_weights(source_means[:,paid].numpy(),
                predictor.loo_mean().numpy(),z.numpy(),identity,n)
            scores=taf_scores(scores,source_means,paid,weights)
            info.update(weights=weights.tolist(),ranking=diag)
    # With zero target weight, every source may have already attained its
    # predicted finite-pool maximum. Then TAF is identically zero, log=-inf.
    # Honor the declared canonical-index tie rule; do not add online fallback.
    available=torch.ones(len(x),dtype=torch.bool);available[paid]=False
    info['score_kind']='prior_mean' if method=='P' else 'log_acquisition'
    if method=='R' and torch.isneginf(scores[available]).all():
        scores=torch.zeros_like(scores);info['score_kind']='raw_zero_acquisition'
    scores[paid]=-float('inf');selected=int(scores.argmax())
    assert torch.isfinite(scores[selected]) and selected not in paid
    info.update(selected=selected,score=float(scores[selected]),paid_before=n,
                total_seconds=time.perf_counter()-start)
    return selected,info
