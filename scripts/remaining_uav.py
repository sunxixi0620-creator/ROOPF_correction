"""Re-evaluate recovered UAV proxy scenarios; geometry is post-hoc diagnostic."""
import argparse, importlib.util, sys, json, time, hashlib, multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np
import torch
import cma
from scipy.optimize import differential_evolution
from supplementary_experiments import ROOT, build, Counted, write_csv
import run_roopf as r

SOURCE=ROOT/'artifacts/recovered_sources/uav_benchmark.py'
spec=importlib.util.spec_from_file_location('recovered_uav',SOURCE)
uav=importlib.util.module_from_spec(spec);sys.modules[spec.name]=uav;spec.loader.exec_module(uav)
METHODS=['full','anchor_only','no_residual','score_only','no_proxy','cma_es','de']

class Problem:
    dim=10
    def __init__(self,fid):
        self.scenario=uav.SCENARIOS[fid-1];self.fun={'fid':f'uavf{fid}','xlb':-5.,'xub':5.}
        self.calls=0;self.trace=[];self.best=float('inf');self.best_x=None
    def getfunname(self):return self.fun['fid']
    def repaire(self,x):return x.clamp(-5,5)
    def calfitness(self,x):
        y=uav.uav_objective(x,self.scenario)
        for point,value in zip(x.detach().reshape(-1,10).cpu().numpy(),y.detach().reshape(-1).cpu().numpy()):
            self.calls+=1
            if float(value)<self.best:self.best=float(value);self.best_x=point.copy()
            self.trace.append(self.best)
        return y
    def evaluate(self,x):return float(self.calfitness(torch.tensor(x,dtype=torch.float32).view(1,1,10))[0,0])

def components_and_clearance(x,scenario):
    x=torch.tensor(x,dtype=torch.float32).view(1,1,10)
    p=uav._as_points(x);seg=p[...,1:,:]-p[...,:-1,:]
    length=torch.linalg.norm(seg,dim=-1).clamp_min(1e-8);samples=uav._segment_samples(p)
    vals={'length':float(length.sum())};threat=collision=nofly=0.
    def clearance(c,radius):
        a=p[...,:-1,:];v=seg;t=((c-a)*v).sum(-1)/(v*v).sum(-1).clamp_min(1e-12)
        nearest=a+t.clamp(0,1).unsqueeze(-1)*v
        return float((torch.linalg.norm(nearest-c,dim=-1)-radius).min())
    clear=[]
    for cx,cy,radius,weight in scenario.obstacles:
        c=torch.tensor([cx,cy]);dist=torch.linalg.norm(samples-c,dim=-1)
        threat+=float(weight*torch.exp(-2*(dist/radius)**2).mean())
        collision+=float(80*torch.relu(radius*.65-dist).pow(2).mean())
        clear.append(clearance(c,radius*.65))
    for cx,cy,radius,weight in scenario.nofly:
        c=torch.tensor([cx,cy]);dist=torch.linalg.norm(samples-c,dim=-1)
        nofly+=float(weight*torch.relu(radius-dist).pow(2).mean());clear.append(clearance(c,radius))
    unit=seg/length.unsqueeze(-1)
    smooth=float(scenario.smooth_weight*torch.linalg.norm(unit[...,1:,:]-unit[...,:-1,:],dim=-1).pow(2).sum())
    energy=float(scenario.energy_weight*(length*(1+2*torch.relu(-(unit*torch.tensor(scenario.wind)).sum(-1)))).sum())
    direct=torch.linspace(-4.5,4.5,7);corridor=torch.stack([direct,direct],-1)
    corridor=float(scenario.corridor_weight*torch.linalg.norm(p-corridor,dim=-1).mean().pow(2))
    boundary=float(20*torch.relu(p.abs()-4.8).pow(2).sum())
    vals.update(threat=threat,collision=collision,nofly=nofly,smooth=smooth,energy=energy,corridor=corridor,boundary=boundary)
    reconstructed=sum(vals.values());truth=float(uav.uav_objective(x,scenario))
    assert abs(reconstructed-truth)<1e-4*max(1,abs(truth))
    vals.update(min_clearance=min(clear),geometrically_feasible=int(min(clear)>=-1e-6 and bool((p.abs()<=4.8+1e-6).all())))
    return vals

def run(fid,seed,method):
    p=Problem(fid);pop=-5+10*torch.rand((1,100,10),generator=torch.Generator().manual_seed(seed))
    opt=build(method) if method in METHODS[:5] else None;r.set_seed(seed+910000);start=time.perf_counter()
    if opt is not None:
        counted=Counted(p)
        with torch.no_grad(): _,trail,nfe,_=opt(pop,counted)
        assert counted.actual==nfe==300
    elif method=='cma_es':
        es=cma.CMAEvolutionStrategy([0.]*10,2.,{'bounds':[-5,5],'seed':seed,'popsize':10,'verbose':-9,'verb_log':0})
        for _ in range(30):
            pts=es.ask();es.tell(pts,[p.evaluate(v) for v in pts])
    else:
        differential_evolution(p.evaluate,[(-5,5)]*10,init=pop[0].numpy().astype(float),maxiter=2,tol=0,atol=0,
            mutation=(.5,1),recombination=.7,seed=seed,polish=False,updating='immediate',workers=1)
    seconds=time.perf_counter()-start;assert p.calls==300 and np.isfinite(p.trace).all()
    return dict(fid=fid,seed=seed,method=method,final=p.best,actual_nfe=p.calls,seconds=seconds,
        **components_and_clearance(p.best_x,p.scenario)),p.trace,p.best_x

def worker(job):
    fid,seed,out=job;out=Path(out);rows=[]
    for m in METHODS:
        stem=f'uav{fid}_{m}_{seed}';p=out/(stem+'.json')
        if p.exists():rows.append(json.loads(p.read_text()));continue
        row,trace,x=run(fid,seed,m);np.savez_compressed(out/(stem+'.npz'),trace=trace,best_x=x)
        p.write_text(json.dumps(row,indent=2));rows.append(row)
    return rows

def main():
    a=argparse.ArgumentParser();a.add_argument('--output',type=Path,required=True);a.add_argument('--workers',type=int,default=8);a=a.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    protocol={'methods':METHODS,'seeds':list(range(20270000,20270030)),'nfe':300,'scenario_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'posthoc_geometry_queries':1,'timing':'parallel, not controlled latency'}
    p=a.output/'protocol.json'
    if p.exists():assert json.loads(p.read_text())==protocol
    else:p.write_text(json.dumps(protocol,indent=2))
    rows=[]
    with ProcessPoolExecutor(a.workers,mp_context=multiprocessing.get_context('spawn')) as ex:
        fs=[ex.submit(worker,(f,s,str(a.output))) for f in range(1,6) for s in protocol['seeds']]
        for i,f in enumerate(as_completed(fs)):
            rows+=f.result()
            if (i+1)%10==0:print(i+1,'/150',flush=True)
    write_csv(a.output/'raw_results.csv',rows);(a.output/'COMPLETE').write_text(str(len(rows)))

if __name__=='__main__':main()
