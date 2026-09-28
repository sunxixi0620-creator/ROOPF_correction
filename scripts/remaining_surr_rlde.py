"""Released Surr-RLDE policy on exactly the Stage3 external objectives."""
import argparse,json,pickle,types,sys,time,hashlib,multiprocessing
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import numpy as np
import torch
from independent_benchmarks import ExternalProblem
from supplementary_experiments import ROOT,write_csv

VENDOR=ROOT/'artifacts/external/surr_rlde'
sys.path.insert(0,str(VENDOR))
for name in ['agent','optimizer','problem']:
    pkg=types.ModuleType(name);pkg.__path__=[str(VENDOR/name)];sys.modules[name]=pkg
from optimizer.Surr_RLDE_Optimizer import Surr_RLDE_Optimizer

class Objective:
    optimum=None
    def __init__(self,p):self.p=p;self.lb=p.fun['xlb'];self.ub=p.fun['xub'];self.dim=10
    def reset(self):pass
    def eval(self,x):
        return np.array([self.p.evaluate(v) for v in np.asarray(x).reshape(-1,10)])

class Env:
    def __init__(self,p,opt):self.problem=p;self.optimizer=opt
    def reset(self):self.problem.reset();return self.optimizer.init_population(self.problem)
    def step(self,a):return self.optimizer.update(a,self.problem)

def run(suite,fid,inst,seed):
    p=ExternalProblem(suite,fid,inst)
    with (VENDOR/'agent_model/test/Surr_RLDE_Agent.pkl').open('rb') as f:agent=pickle.load(f)
    agent.device=torch.device('cpu');agent.pred_Qnet.to('cpu').eval()
    cfg=types.SimpleNamespace(dim=10,device=torch.device('cpu'),is_train=False,NP=100,
        maxFEs=200,upperbound=p.fun['xub'],log_interval=4,n_logpoint=50)
    np.random.seed(seed);torch.manual_seed(seed)
    start=time.perf_counter();opt=Surr_RLDE_Optimizer(cfg);info=agent.rollout_episode(Env(Objective(p),opt))
    assert p.calls==300 and int(info['fes'])==300,(p.calls,info['fes'])
    assert np.isfinite(p.trace).all()
    return {'suite':suite,'fid':fid,'instance':inst,'seed':seed,'method':'surr_rlde','final':p.trace[-1],
        'actual_nfe':p.calls,'reported_fes':int(info['fes']),'seconds':time.perf_counter()-start},np.array(p.trace)

def worker(j):
    suite,fid,inst,seed,out=j;out=Path(out);stem=f'{suite}_{fid}_{inst}_{seed}';p=out/(stem+'.json')
    if p.exists():return json.loads(p.read_text())
    row,trace=run(suite,fid,inst,seed);np.save(out/(stem+'.npy'),trace);p.write_text(json.dumps(row,indent=2));return row

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=8);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    cases=[('coco',f,i) for f in range(1,25) for i in (101,102)]+[('cec2017',f,1) for f in range(10,30)]
    protocol={'cases':cases,'seeds':list(range(20265000,20265010)),'nfe':300,'population':100,
        'internal_maxFEs':200,'budget_note':'upstream checks termination before adding last100; actual counter asserts300',
        'optimum':'not supplied','source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'upstream':json.loads((VENDOR/'PROVENANCE.json').read_text())}
    protocol=json.loads(json.dumps(protocol));p=a.output/'protocol.json'
    if p.exists():assert json.loads(p.read_text())==protocol
    else:p.write_text(json.dumps(protocol,indent=2))
    rows=[]
    with ProcessPoolExecutor(a.workers,mp_context=multiprocessing.get_context('spawn')) as ex:
        fs=[ex.submit(worker,(*c,s,str(a.output))) for c in cases for s in protocol['seeds']]
        for i,f in enumerate(as_completed(fs)):
            rows.append(f.result())
            if (i+1)%50==0:print(i+1,'/',len(fs),flush=True)
    write_csv(a.output/'raw_results.csv',rows);(a.output/'COMPLETE').write_text(str(len(rows)))

if __name__=='__main__':main()
