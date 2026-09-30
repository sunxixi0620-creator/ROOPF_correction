"""Bounded candidate-score calibration; frozen proposal and online intent."""
import math
import torch
from torch import nn
from .complementary_proposal import ContextPortfolio
from .model import _get_bounds,_normalize_x,EPS

class ScoreCalibration(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(10,64),nn.GELU(),nn.Linear(64,1))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
    def forward(self,features):
        return .5*torch.tanh(self.net(features).squeeze(-1))

class CalibratedPortfolio(ContextPortfolio):
    def __init__(self,proposal,calibrator=None,budget=600,capture=False):
        super().__init__(proposal=proposal,dim=20,budget=budget)
        self.calibrator=calibrator
        self.collect=capture
        self.records=[]
    def scores(self,candidates,prior,ax,ay,problem,remaining,stagnation):
        # This computes unchanged O intent, generates frozen proposals, and ranks
        # the resulting38 exactly as before. Calibrate only this final score.
        base=super().scores(candidates,prior,ax,ay,problem,remaining,stagnation)
        sample=self.collect and (self.evalnum-100)//2 in (0,35,70,105,140,175,210,245)
        if self.calibrator is None and not sample:return base
        mu,sigma=self.surrogate.predict(ax,ay,candidates,problem,distance_scale=.03)
        scale=ay.std(1,keepdim=True,unbiased=False).clamp_min(EPS)
        lb,ub=_get_bounds(problem,candidates)
        cx,hx=_normalize_x(candidates,lb,ub),_normalize_x(ax,lb,ub)
        best=hx[torch.arange(len(hx)),ay.argmin(1)][:,None]
        shape=base.shape
        rank=torch.argsort(torch.argsort(base,dim=1,stable=True),dim=1,stable=True)/37
        source=torch.zeros_like(base);source[:,:2]=1
        features=torch.stack((base,(mu-ay.min(1,keepdim=True).values)/scale,sigma/scale,
            prior.clamp_min(EPS).log(),(cx-best).norm(dim=-1)/math.sqrt(20),
            torch.cdist(cx,hx).amin(2),torch.full_like(base,remaining),
            stagnation[:,None].expand(shape),source,rank),-1).clamp(-10,10)
        if sample:self.records.append(dict(features=features.clone(),scores=base.clone(),
            candidates=candidates.clone(),incumbent=ay.min(1).values.clone(),scale=scale[:,0].clone()))
        return base if self.calibrator is None else base+self.calibrator(features)

def build_calibrated(proposal,calibrator=None,budget=600,capture=False):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(20260630)
        model=CalibratedPortfolio(proposal,calibrator,budget,capture)
    return model.eval().requires_grad_(False)
