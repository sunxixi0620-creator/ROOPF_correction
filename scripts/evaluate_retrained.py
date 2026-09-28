"""Native-dimensional evaluation of explicitly reconstructed anchors."""
import argparse,json,time,multiprocessing,hashlib
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import numpy as np
import torch,cma,cocoex
from scipy.optimize import differential_evolution
from supplementary_experiments import ROOT,Counted,write_csv
from roopf.supplement import SupplementOptimizer
import run_roopf as r

class NativeProblem:
    def __init__(self,dim,fid,instance):
        self.dim=dim;self.suite=cocoex.Suite('bbob',f'instances: {instance}',f'dimensions: {dim} function_indices: {fid}')
        self.obj=self.suite[0];self.fun={'fid':fid,'xlb':-5.,'xub':5.};self.trace=[];self.calls=0
    def evaluate(self,x):
        y=float(self.obj(np.asarray(x,dtype=float)));assert np.isfinite(y)
        self.calls+=1;self.trace.append(min(y,self.trace[-1] if self.trace else float('inf')));return y
    def getfunname(self):return self.fun['fid']
    def repaire(self,x):return x.clamp(-5,5)
    def calfitness(self,x):
        return torch.tensor([self.evaluate(v) for v in x.detach().cpu().numpy().reshape(-1,self.dim)],dtype=x.dtype,device=x.device).reshape(x.shape[:2])

def run(dim,fid,inst,seed,nfe,method,checkpoint,residual):
    p=NativeProblem(dim,fid,inst);r.set_seed(20260630)
    if method in ('full','full_transferred_residual','no_residual','anchor_only'):
        opt=SupplementOptimizer(dim=dim,hidden_dim=200,popSize=100,max_nfe=nfe,k_nums=2,
            pool_per_op=6,surrogate_members=5,ablation='roopf',baseline_ckpt=checkpoint,
            router_ckpt=(str(ROOT/'checkpoints/residual_selector_generated36_d10.pt') if method=='full_transferred_residual' else residual),router_weight=.008).eval().configure('full' if method=='full_transferred_residual' else method)
    pop=-5+10*torch.rand((1,100,dim),generator=torch.Generator().manual_seed(seed));r.set_seed(seed+910000);start=time.perf_counter()
    if method in ('full','full_transferred_residual','no_residual','anchor_only'):
        counted=Counted(p)
        with torch.no_grad():_,trail,used,_=opt(pop,counted)
        assert counted.actual==used==nfe
    elif method=='cma_es':
        es=cma.CMAEvolutionStrategy([0.]*dim,2.,{'bounds':[-5,5],'seed':seed,'popsize':10,'verbose':-9,'verb_log':0})
        for _ in range(nfe//10):
            pts=es.ask();es.tell(pts,[p.evaluate(v) for v in pts])
    else:
        differential_evolution(p.evaluate,[(-5,5)]*dim,init=pop[0].numpy().astype(float),
            maxiter=nfe//100-1,tol=0,atol=0,mutation=(.5,1),recombination=.7,seed=seed,polish=False,updating='immediate',workers=1)
    assert p.calls==nfe
    return {'dimension':dim,'fid':fid,'instance':inst,'seed':seed,'budget':nfe,'method':method,
        'final':p.trace[-1],'actual_nfe':p.calls,'seconds':time.perf_counter()-start},np.array(p.trace)

def worker(job):
    dim,fid,inst,seed,nfe,methods,checkpoint,residual,out=job;out=Path(out);rows=[]
    for m in methods:
        stem=f'd{dim}_f{fid}_i{inst}_s{seed}_b{nfe}_{m}';p=out/(stem+'.json')
        if p.exists():rows.append(json.loads(p.read_text()));continue
        row,trace=run(dim,fid,inst,seed,nfe,m,checkpoint,residual);np.save(out/(stem+'.npy'),trace);p.write_text(json.dumps(row,indent=2));rows.append(row)
    return rows

def main():
    p=argparse.ArgumentParser();p.add_argument('--dim',type=int,required=True);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--residual',type=Path,default=ROOT/'checkpoints/residual_selector_generated36_d10.pt')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=10);a=p.parse_args()
    assert (a.checkpoint.parent/'COMPLETE').exists(),'Training must finish before test evaluation'
    budgets=[300,600] if a.dim==20 else [300]
    methods=['full','full_transferred_residual','no_residual','anchor_only','cma_es','de'] if a.dim==20 else ['full','anchor_only']
    protocol={'dimension':a.dim,'checkpoint_sha256':hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),
        'methods':methods,'budgets':budgets,'seeds':list(range(20265000,20265010)),
        'instances':[101,102],'functions':list(range(1,25)),'residual_sha256':hashlib.sha256(a.residual.read_bytes()).hexdigest(),
        'residual':'native20-D training for full at20D; explicitly separate original10-D transfer comparator',
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    a.output.mkdir(parents=True,exist_ok=True);p=a.output/'protocol.json'
    if p.exists():assert json.loads(p.read_text())==protocol
    else:p.write_text(json.dumps(protocol,indent=2))
    rows=[]
    jobs=[(a.dim,f,i,s,b,methods,str(a.checkpoint.resolve()),str(a.residual.resolve()),str(a.output)) for f in range(1,25) for i in (101,102) for s in protocol['seeds'] for b in budgets]
    with ProcessPoolExecutor(a.workers,mp_context=multiprocessing.get_context('spawn')) as ex:
        fs=[ex.submit(worker,j) for j in jobs]
        for i,f in enumerate(as_completed(fs)):
            rows+=f.result()
            if (i+1)%20==0:print(i+1,'/',len(fs),flush=True)
    write_csv(a.output/'raw_results.csv',rows);(a.output/'COMPLETE').write_text(str(len(rows)))

if __name__=='__main__':main()
