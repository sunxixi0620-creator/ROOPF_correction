"""Paired multi-seed coordinate diagnostic; preserve published source and checkpoints."""
import argparse,csv,hashlib,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
import run_roopf as r
from roopf.coordinate_variants import CoordinateOptimizer


class CountedProblem(r.CECStyleProblem):
    def __init__(self,fun):
        super().__init__(fun,10);self.actual=0
    def calfitness(self,x):
        self.actual+=x.shape[1]
        return super().calfitness(x)


def sync():
    if r.DEVICE.type=='cuda':torch.cuda.synchronize()


def build(mode):
    r.set_seed(20260630)
    opt=CoordinateOptimizer(dim=10,hidden_dim=200,popSize=100,max_nfe=300,k_nums=2,pool_per_op=6,surrogate_members=5,ablation='roopf',baseline_ckpt=str(ROOT/'checkpoints/anchor_policy_d10.pt'),router_ckpt=str(ROOT/'checkpoints/residual_selector_generated36_d10.pt'),router_weight=0.008).to(r.DEVICE).eval()
    opt.mode=mode
    if mode=='unlocked':opt.ablation.discard('structured_system_lock')
    opt.log_candidates=True
    assert not opt.log_pool_candidates
    return opt


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seeds',type=int,default=10)
    p.add_argument('--functions',default='1,2,3,4,5,6')
    p.add_argument('--shift-seed',type=int,default=20260923)
    p.add_argument('--variants',default='legacy,unlocked,centroid_slot,relative_slot')
    args=p.parse_args()
    modes=args.variants.split(',')
    assert set(modes)<={'legacy','unlocked','centroid_slot','relative_slot'}
    assert args.seeds>0
    args.output.mkdir(parents=True,exist_ok=False)
    seeds=list(range(20261000,20261000+args.seeds))
    params={'NFE':300,'initial_population':100,'dimension':10,'seeds':seeds,'model_seed':20260630,'shift_seed':args.shift_seed,'variants':modes,'elite_count':20,'first_anchor_preserved':True,'notes':'One fixed random shifted instance per function; seeds vary initialization. Coordinate adapters are frozen-checkpoint heuristics, not retrained models. All functions have known minimum zero. Batched runs may differ numerically from earlier single-trajectory runs.'}
    params['hashes']={str(f.relative_to(ROOT)):r.sha256(f) for f in [Path(__file__),ROOT/'roopf/coordinate_variants.py',ROOT/'roopf/model.py',ROOT/'roopf/anchor_backbone.py',ROOT/'roopf/benchmarks/cecfunctions.py',*sorted((ROOT/'checkpoints').glob('*.pt'))]}
    params['environment']={'torch':torch.__version__,'numpy':np.__version__,'device':str(r.DEVICE),'gpu':torch.cuda.get_device_name() if torch.cuda.is_available() else ''}
    (args.output/'protocol.json').write_text(json.dumps(params,indent=2))
    rows=[]; instances={}
    for fid in r.parse_function_ids(args.functions,range(1,7)):
        source=dict(r.CEC_FUNCTIONS[f'cecf{fid}'])
        gen=torch.Generator(device='cpu').manual_seed(args.shift_seed+fid)
        shift=source['blb']+(source['bub']-source['blb'])*torch.rand((10,),generator=gen)
        for condition in ['zero','shifted']:
            key=f'f{fid}_{condition}'
            bias=torch.zeros(10) if condition=='zero' else shift
            instances[key]={'fid':fid,'bias':bias.tolist(),'bounds':[source['xlb'],source['xub']]}
            f=dict(source);f['bias']=bias.to(r.DEVICE)
            population=torch.stack([source['xlb']+(source['xub']-source['xlb'])*torch.rand((100,10),generator=torch.Generator().manual_seed(s)) for s in seeds]).to(r.DEVICE)
            pop_hash=hashlib.sha256(population.cpu().numpy().tobytes()).hexdigest()
            for mode in modes:
                opt=build(mode)
                # Algorithm RNG starts identically for every variant and condition.
                r.set_seed(20262000+fid)
                problem=CountedProblem(dict(f))
                sync();start=time.perf_counter()
                with torch.no_grad(): _,trail,nfe,_=opt(population.clone(),problem)
                sync();seconds=time.perf_counter()-start
                assert nfe==problem.actual==300
                assert torch.isfinite(trail).all()
                assert (trail[:,1:]<=trail[:,:-1]+1e-6).all()
                assert len(opt.candidate_samples)==args.seeds*200
                np.savez_compressed(args.output/f'{key}_{mode}.npz',trail=trail.cpu().numpy(),nfe=np.arange(102,301,2),bias=bias.numpy())
                records=opt.candidate_samples
                with (args.output/f'{key}_{mode}_candidates.csv').open('w',newline='') as stream:
                    w=csv.DictWriter(stream,fieldnames=records[0]);w.writeheader();w.writerows(records)
                for bi,seed in enumerate(seeds):
                    own=[row for row in records if row['batch']==bi]
                    rows.append({'case':key,'fid':fid,'condition':condition,'variant':mode,'seed':seed,'final':float(trail[bi,-1]),'actual_nfe':problem.actual,'reported_nfe':nfe,'batch_seconds':seconds,'portfolio_evals':sum(not row['is_baseline'] for row in own),'slot2_best_improvements':sum(row['improved_best'] for row in own if row['candidate_slot']==1),'initial_population_sha256':pop_hash})
                with (args.output/'raw_results.csv').open('w',newline='') as stream:
                    w=csv.DictWriter(stream,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
                a=trail[:,-1].cpu().numpy()
                print(json.dumps({'case':key,'variant':mode,'mean':float(a.mean()),'std':float(a.std(ddof=1)) if len(a)>1 else 0,'batch_seconds':seconds,'completed_trajectories':len(rows)}),flush=True)
            (args.output/'instances.json').write_text(json.dumps(instances,indent=2))
    (args.output/'COMPLETE').write_text(f'{len(rows)} trajectories; all exact-budget/finite/monotonic checks passed.\n')


if __name__=='__main__':main()
