"""Generate native20-D full-pool labels without benchmark data or residual use."""
import argparse,json,time,hashlib,multiprocessing
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import torch
from supplementary_experiments import ROOT,write_csv
from reconstruct_anchor_training import module,TrainingProblem
from roopf.supplement import SupplementOptimizer
import run_roopf as r

class Task(TrainingProblem):
    dim=20
    def getfunname(self):return self.fun['fid']

def worker(job):
    idx,checkpoint,out=job;out=Path(out);p=out/f'f{idx:02d}.json'
    if p.exists():return json.loads(p.read_text())
    gen=module('generated',ROOT/'artifacts/training_provenance/generated36_original.py')
    fun=dict(sorted(gen.TRAIN_FUNCTIONS,key=lambda f:f['fid'])[idx])
    r.set_seed(20272000+idx);gen.gen_train_offset(20,fun);task=Task(fun,gen)
    r.set_seed(20260630)
    opt=SupplementOptimizer(dim=20,hidden_dim=200,popSize=100,max_nfe=300,k_nums=2,pool_per_op=6,
        surrogate_members=5,ablation='roopf',baseline_ckpt=checkpoint,router_ckpt=None).eval().configure('no_residual')
    opt.log_pool_candidates=True;opt.log_candidates=False
    r.set_seed(20273000+idx);pop=10*torch.rand(10,100,20)-5
    start=time.perf_counter()
    with torch.no_grad():_,trail,nfe,_=opt(pop,task)
    assert nfe==300 and task.points==41000 and len(opt.pool_candidate_samples)==38000
    rows=[{'target':'generated36','fid':fun['fid'],**row} for row in opt.pool_candidate_samples]
    write_csv(out/f'f{idx:02d}_pool.csv',rows)
    row={'fid':fun['fid'],'trajectories':10,'main_points':3000,'teacher_points':38000,
        'pool_rows':len(rows),'portfolio_rows':36000,'seconds':time.perf_counter()-start}
    p.write_text(json.dumps(row,indent=2));return row

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,default=8);a=p.parse_args();assert (a.checkpoint.parent/'COMPLETE').exists()
    a.output.mkdir(parents=True,exist_ok=True)
    protocol={'dim':20,'functions':36,'trajectories_per_function':10,'seed_offset':20272000,'seed_population':20273000,
        'behavior':'final policy without residual; online proxies active','checkpoint_sha256':hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    pp=a.output/'protocol.json'
    if pp.exists():assert json.loads(pp.read_text())==protocol
    else:pp.write_text(json.dumps(protocol,indent=2))
    rows=[]
    with ProcessPoolExecutor(a.workers,mp_context=multiprocessing.get_context('spawn')) as ex:
        fs=[ex.submit(worker,(i,str(a.checkpoint.resolve()),str(a.output))) for i in range(36)]
        for f in as_completed(fs):
            rows.append(f.result());print(len(rows),'/36',flush=True)
    write_csv(a.output/'counts.csv',rows);(a.output/'COMPLETE').write_text(json.dumps({'portfolio_labels':1296000,'teacher_points':1368000,'main_points':108000}))

if __name__=='__main__':main()
