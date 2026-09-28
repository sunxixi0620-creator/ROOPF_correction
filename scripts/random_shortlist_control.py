"""Only replace pool shortlist ranking; retain second-stage score and gate."""
import argparse,json,multiprocessing,hashlib
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import numpy as np
import torch
from supplementary_experiments import build,problem_for,write_csv
from roopf.supplement import SupplementOptimizer
from remaining_mechanisms import CASES
import run_roopf as r

class RandomShortlist(SupplementOptimizer):
    def _select_by_acquisition(self,*args,**kwargs):
        idx=super()._select_by_acquisition(*args,**kwargs)
        if args[2].shape[1]==36:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(self.control_seed+1000*self.evalnum)
                idx=torch.stack([torch.randperm(36)[:idx.shape[1]] for _ in range(idx.shape[0])])
        return idx

def worker(job):
    case,seed,out=job;out=Path(out);stem=f'{case}_{seed}';p=out/(stem+'.json')
    if p.exists():return json.loads(p.read_text())
    task,_=problem_for(case);opt=build();opt.__class__=RandomShortlist;opt.control_seed=seed
    pop=task.fun['xlb']+(task.fun['xub']-task.fun['xlb'])*torch.rand(1,100,10,generator=torch.Generator().manual_seed(seed))
    r.set_seed(seed+770000)
    with torch.no_grad():_,trail,nfe,points=opt(pop,task)
    assert nfe==task.actual==300
    row={'case':case,'seed':seed,'method':'random_pool_shortlist','final':float(trail[0,-1]),'actual_nfe':300}
    np.savez_compressed(out/(stem+'.npz'),trail=trail.numpy(),points=points.numpy());p.write_text(json.dumps(row,indent=2));return row

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=8);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    protocol={'cases':CASES,'seeds':list(range(20269000,20269030)),'nfe':300,
        'replacement':'uniform2-of36 pool shortlist only; second-stage scoring and protected gate unchanged; isolated random stream',
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    p=a.output/'protocol.json'
    if p.exists():assert json.loads(p.read_text())==protocol
    else:p.write_text(json.dumps(protocol,indent=2))
    rows=[]
    with ProcessPoolExecutor(a.workers,mp_context=multiprocessing.get_context('spawn')) as ex:
        fs=[ex.submit(worker,(c,s,str(a.output))) for c in CASES for s in protocol['seeds']]
        for f in as_completed(fs):rows.append(f.result())
    write_csv(a.output/'raw_results.csv',rows);(a.output/'COMPLETE').write_text(str(len(rows)))

if __name__=='__main__':main()
