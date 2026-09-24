"""Read-only algorithm audit: same frozen models, counted objective calls."""
import csv,json,sys,time
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
import run_roopf as r

class Counted(r.CECStyleProblem):
    def __init__(self,f):
        super().__init__(f,10); self.actual=0
    def calfitness(self,x):
        self.actual+=x.shape[1]
        return super().calfitness(x)

out=ROOT/'results/evidence_audit_v1'
out.mkdir(exist_ok=False)
rows=[]
cfg=SimpleNamespace(dimension=10,population_size=100,budget=300)
for fid in range(1,7):
    previous=None
    for variant in ['published','anchor_only']:
        r.set_seed(20260630)
        opt=r.build_optimizer(cfg)
        if variant=='anchor_only': opt.ablation.add('baseline_only')
        opt.log_candidates=True
        problem=Counted(dict(r.CEC_FUNCTIONS[f'cecf{fid}']))
        initial=problem.genRandomPop((1,100,10))
        with torch.no_grad(): x,trail,nfe,_=opt(initial,problem)
        assert nfe==problem.actual==300
        identical=True if previous is None else torch.equal(previous,trail)
        if previous is None: previous=trail.clone()
        row={'case':f'cec{fid}','variant':variant,'seed':20260630,'shift':None,'final':float(trail[0,-1]),'actual_nfe':problem.actual,'portfolio_evals':sum(not s['is_baseline'] for s in opt.candidate_samples),'equal_published_trail':identical}
        rows.append(row); print(json.dumps(row),flush=True)
        torch.save(trail.cpu(),out/f'cec{fid}_{variant}.pt')
for seed in [20260630,20260631,20260632]:
    for shift in [0.0,10.0,-10.0]:
        r.set_seed(seed); opt=r.build_optimizer(cfg)
        f=dict(r.CEC_FUNCTIONS['cecf1'])
        f['bias']=torch.full((10,),shift,device=r.DEVICE)
        problem=Counted(f); initial=problem.genRandomPop((1,100,10))
        with torch.no_grad(): x,trail,nfe,_=opt(initial,problem)
        assert nfe==problem.actual==300
        row={'case':'sphere_shift_audit','variant':'published','seed':seed,'shift':shift,'final':float(trail[0,-1]),'actual_nfe':problem.actual,'portfolio_evals':0,'equal_published_trail':''}
        rows.append(row);print(json.dumps(row),flush=True)
with (out/'summary.csv').open('w',newline='') as s:
    w=csv.DictWriter(s,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
(out/'metadata.json').write_text(json.dumps({'torch':torch.__version__,'device':str(r.DEVICE),'purpose':'Mechanism diagnostic only; shifts chosen to probe coordinate bias, not a formal benchmark','hashes':{str(p.relative_to(ROOT)):r.sha256(p) for p in [Path(__file__),ROOT/'roopf/model.py',ROOT/'run_roopf.py',ROOT/'roopf/benchmarks/cecfunctions.py']}},indent=2))
(out/'COMPLETE').write_text('21 counted trajectories finished.\n')
