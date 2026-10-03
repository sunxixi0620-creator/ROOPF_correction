"""Source-only variance shrinkage for the bounded v2 training diagnostic."""
import torch
from roopf.neural_prior_training_v1 import SourceModel
from roopf.probabilistic_prior_v1 import condition,conditional_nll


def normalize_with_reference(yc,yq,reference_variance):
    """Four pseudo degrees of freedom; no query-dependent scale or label clip."""
    n=yc.shape[-1];assert n>=2
    mean=yc.mean(-1,keepdim=True);variance=yc.var(-1,unbiased=True,keepdim=True)
    reference=torch.as_tensor(reference_variance,dtype=yc.dtype,device=yc.device)
    assert torch.isfinite(reference).all() and (reference>0).all()
    scale=((4*reference+(n-1)*variance)/(n+3)).sqrt()
    return (yc-mean)/scale,((yq-mean)/scale if yq is not None else None),mean,scale


class StableSourceModel(SourceModel):
    def __init__(self,dim,kind,seed,space,reference_variance):
        super().__init__(dim,kind,seed,space)
        self.register_buffer('reference_variance',torch.tensor(reference_variance,dtype=torch.double))

    def forward(self,batch):
        xc,yc,xq,yq,pc,pq=batch
        zc,zq,_,_=normalize_with_reference(yc,yq,self.reference_variance)
        if self.prior is not None:
            pc,pq=self.prior(xc),self.prior(xq)
            if self.kind=='M':pc=(pc[0],pc[1]*0);pq=(pq[0],pq[1]*0)
        mean,cov=condition(self.base,xc,zc,xq,pc,pq)
        return conditional_nll(mean,cov,zq),mean,cov,zq
