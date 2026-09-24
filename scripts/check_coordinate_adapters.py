"""Check adapter identity, translation property, and lack of hidden objective calls."""
import sys
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import torch
import run_roopf as r
from coordinate_experiment import build

class BoundsOnly:
    def __init__(self,lo,hi):self.fun={'xlb':lo,'xub':hi}
    def repaire(self,x):return torch.maximum(torch.minimum(x,self.fun['xub']),self.fun['xlb'])
    def calfitness(self,x):raise AssertionError('Adapter must not query objective')

r.set_seed(123)
x=torch.randn(2,100,10,device=r.DEVICE)
y=torch.arange(100,device=r.DEVICE).float().view(1,-1).expand(2,-1)
lb=torch.full((10,),-100.,device=r.DEVICE);ub=-lb
p=BoundsOnly(lb,ub)
cfg=SimpleNamespace(dimension=10,population_size=100,budget=300)
r.set_seed(20260630);old=r.build_optimizer(cfg)
new=build('legacy')
with torch.no_grad():
    a=old._baseline_candidates(x,y,p,2);b=new._baseline_candidates(x,y,p,2)
assert torch.equal(a,b)
errors={}
delta=torch.linspace(-10,10,10,device=r.DEVICE)
for mode in ['centroid_slot','relative_slot']:
    opt=build(mode)
    a=opt._baseline_candidates(x,y,p,2)
    b=opt._baseline_candidates(x+delta,y,BoundsOnly(lb+delta,ub+delta),2)
    error=float((b[:,1]-a[:,1]-delta).abs().max())
    assert error<1e-4,error
    assert torch.equal(a[:,:1],old._baseline_candidates(x,y,p,2)[:,:1])
    assert torch.equal(opt._baseline_candidates(x,y,p,1),old._baseline_candidates(x,y,p,1))
    errors[mode]=error
print({'legacy_identity':True,'first_slot_preserved':True,'last_slot_preserved':True,'objective_calls':0,'translation_max_abs_errors':errors})
