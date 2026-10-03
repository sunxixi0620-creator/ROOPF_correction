"""Small frozen neural/PCA priors and one joint Gaussian conditioning step.

This is a research prototype, not the original ROOPF residual selector. All
functions consume explicitly supplied observations; none can query an oracle.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


class NeuralPrior(nn.Module):
    def __init__(self,dim,rank=8,width=32):
        super().__init__();self.dim=dim;self.rank=rank
        self.network=nn.Sequential(nn.Linear(dim,width),nn.GELU(),
            nn.Linear(width,width),nn.GELU(),nn.Linear(width,rank+1))
        self.double()

    def forward(self,x):
        value=self.network(x)
        return value[...,0],value[...,1:]/math.sqrt(self.rank)


class BaseCovariance(nn.Module):
    """Positive Matern-5/2 lengthscales and noise for source conditional fitting.

    The initial numerical values match the dimension-scaled prior modes used by
    the reference GP; this module does not claim identical parameter priors or
    optimizer semantics. Online GP integration requires its own frozen contract.
    """
    def __init__(self,dim):
        super().__init__()
        initial=math.exp(math.sqrt(2)+.5*math.log(dim)-3)
        self.raw_lengthscale=nn.Parameter(torch.full((dim,),math.log(math.expm1(initial-.025)),dtype=torch.double))
        self.raw_noise=nn.Parameter(torch.tensor(math.log(math.expm1(math.exp(-5)-1e-4)),dtype=torch.double))

    @property
    def lengthscale(self):return .025+F.softplus(self.raw_lengthscale)

    @property
    def noise(self):return 1e-4+F.softplus(self.raw_noise)

    def forward(self,x1,x2):
        difference=(x1.unsqueeze(-2)-x2.unsqueeze(-3))/self.lengthscale
        distance=difference.square().sum(-1).clamp_min(1e-30).sqrt()
        scaled=math.sqrt(5)*distance
        return (1+scaled+scaled.square()/3)*torch.exp(-scaled)


def normalize_context(yc,yq=None):
    """Never use query responses in the normalization statistics."""
    mean=yc.mean(-1,keepdim=True);scale=yc.std(-1,unbiased=True,keepdim=True)
    scale=torch.where(scale<1e-8,torch.ones_like(scale),scale)
    zc=(yc-mean)/scale
    return zc,((yq-mean)/scale if yq is not None else None),mean,scale


def condition(base,xc,zc,xq,context_prior,query_prior,observation_noise=True):
    """Condition K_base + Phi Phi^T once on paid/context data, in float64."""
    mc,pc=context_prior;mq,pq=query_prior
    kcc=base(xc,xc)+pc@pc.transpose(-1,-2)
    eye=torch.eye(xc.shape[-2],dtype=xc.dtype,device=xc.device)
    kcc=kcc+base.noise*eye
    kqc=base(xq,xc)+pq@pc.transpose(-1,-2)
    kqq=base(xq,xq)+pq@pq.transpose(-1,-2)
    if observation_noise:
        kqq=kqq+base.noise*torch.eye(xq.shape[-2],dtype=xq.dtype,device=xq.device)
    chol=torch.linalg.cholesky(kcc)
    alpha=torch.cholesky_solve((zc-mc).unsqueeze(-1),chol)
    mean=mq+(kqc@alpha).squeeze(-1)
    solve=torch.linalg.solve_triangular(chol,kqc.transpose(-1,-2),upper=False)
    covariance=kqq-solve.transpose(-1,-2)@solve
    covariance=(covariance+covariance.transpose(-1,-2))*.5
    return mean,covariance


def conditional_nll(mean,covariance,query_y):
    """Joint predictive NLL per query observation, before averaging episodes."""
    chol=torch.linalg.cholesky(covariance)
    residual=(query_y-mean).unsqueeze(-1)
    white=torch.linalg.solve_triangular(chol,residual,upper=False).squeeze(-1)
    return (white.square().sum(-1)+2*chol.diagonal(dim1=-2,dim2=-1).log().sum(-1)
            +query_y.shape[-1]*math.log(2*math.pi))/(2*query_y.shape[-1])


def pca_directions(source_predictions,rank=8):
    """Task-space PCA on a training-X-only reference grid.

    Rows are source task functions, columns public reference configurations.
    Eigenvectors project centered source predictions at any future X. Scaling
    by sqrt(number_of_source_tasks-1) preserves empirical function covariance.
    Insufficient task rank is padded with zeros, not fabricated extra variance.
    """
    centered=source_predictions-source_predictions.mean(0,keepdim=True)
    u,s,_=torch.linalg.svd(centered,full_matrices=False)
    active=min(rank,len(source_predictions)-1,len(s))
    directions=torch.zeros(len(source_predictions),rank,dtype=centered.dtype,device=centered.device)
    directions[:,:active]=u[:,:active]/math.sqrt(len(source_predictions)-1)
    # Fix eigenvector signs for deterministic serialization; sign has no effect
    # on Phi Phi.T or posterior predictions.
    for k in range(active):
        pivot=directions[:,k].abs().argmax()
        if directions[pivot,k]<0:directions[:,k]*=-1
    fraction=float(s[:active].square().sum()/s.square().sum()) if float(s.square().sum())>0 else 0.
    return directions,dict(active_rank=active,explained_variance=fraction,
                            singular_values=s.detach().cpu().tolist())


def pca_prior(source_predictions,directions):
    mean=source_predictions.mean(0)
    features=(source_predictions-mean).transpose(-1,-2)@directions
    return mean,features
