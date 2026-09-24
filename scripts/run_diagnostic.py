"""Paired initial-population pilot; no unbudgeted candidate-pool labels."""
import argparse
import csv
import hashlib
import json
import platform
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
import run_roopf as base
from roopf.model_improved import ROOPFOptimizer as Improved


class CountedProblem(base.BBOBProblem):
    def __init__(self, function, dimension):
        super().__init__(function, dimension)
        self.actual_nfe = 0

    def calfitness(self, x):
        self.actual_nfe += x.shape[1]
        return super().calfitness(x)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--functions', default='1,8,15')
    p.add_argument('--seeds', default='20260630,20260631,20260632')
    p.add_argument('--variants', default='baseline,counter_fix,early_gate')
    p.add_argument('--budget', type=int, default=300)
    p.add_argument('--output', type=Path, default=ROOT/'results'/'diagnostic_pilot')
    args = p.parse_args()
    if args.budget != 300:
        p.error('This early-gate pilot is specified for NFE=300 only.')
    variants = args.variants.split(',')
    if not set(variants) <= {'baseline','counter_fix','early_gate'}:
        p.error('Unknown variant')
    fids = base.parse_function_ids(args.functions, range(1,25))
    seeds = [int(x) for x in args.seeds.split(',')]
    args.output.mkdir(parents=True, exist_ok=False)
    offsets = base.load_bbob_offsets(ROOT/'data/bbob_offsets_d10.pkl')
    hashes = {str(x.relative_to(ROOT)):base.sha256(x) for x in [ROOT/'roopf/model.py',ROOT/'roopf/model_improved.py',Path(__file__),*sorted((ROOT/'checkpoints').glob('*.pt')), ROOT/'data/bbob_offsets_d10.pkl']}
    metadata = {'arguments':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,'device':str(base.DEVICE),'hashes':hashes,'note':'Paired initial populations; later trajectories diverge. No pool labels. Pilot is not confirmatory testing.'}
    (args.output/'metadata.json').write_text(json.dumps(metadata,indent=2))
    rows=[]
    for fid in fids:
        for seed in seeds:
            initial_hash = None
            for variant in variants:
                base.set_seed(seed)
                cfg = SimpleNamespace(dimension=10,population_size=100,budget=args.budget)
                if variant == 'baseline':
                    opt=base.build_optimizer(cfg)
                else:
                    opt=Improved(dim=10,hidden_dim=200,popSize=100,max_nfe=args.budget,k_nums=2,pool_per_op=6,surrogate_members=5,ablation='roopf',baseline_ckpt=str(ROOT/'checkpoints/anchor_policy_d10.pt'),router_ckpt=str(ROOT/'checkpoints/residual_selector_generated36_d10.pt'),router_weight=0.008).to(base.DEVICE).eval()
                    opt.early_gate = variant == 'early_gate'
                opt.log_candidates=True
                assert not opt.log_pool_candidates
                fun=dict(base.BBOB_FUNCTIONS[fid]); fun['xlb']=-5; fun['xub']=5
                base.setOffset(fun,offsets[fid])
                problem=CountedProblem(fun,10)
                population=problem.genRandomPop((1,100,10))
                pop_hash=hashlib.sha256(population.cpu().numpy().tobytes()).hexdigest()
                if initial_hash is None: initial_hash=pop_hash
                assert pop_hash == initial_hash, 'Initial population mismatch'
                if base.DEVICE.type=='cuda': torch.cuda.synchronize()
                start=time.perf_counter()
                with torch.no_grad(): _,trail,nfe,_=opt(population,problem)
                if base.DEVICE.type=='cuda': torch.cuda.synchronize()
                seconds=time.perf_counter()-start
                assert nfe == problem.actual_nfe == args.budget, (nfe,problem.actual_nfe)
                assert torch.isfinite(trail).all()
                assert (trail[:,1:] <= trail[:,:-1]+1e-6).all()
                records=opt.candidate_samples
                assert len(records)==args.budget-100
                portfolio=[r for r in records if not r['is_baseline']]
                row={'fid':fid,'seed':seed,'variant':variant,'final':float(trail[0,-1]),'nfe':int(nfe),'actual_nfe':problem.actual_nfe,'seconds':seconds,'portfolio_evaluations':len(portfolio),'portfolio_best_improvements':sum(r['improved_best'] for r in portfolio),'initial_population_sha256':pop_hash}
                rows.append(row)
                stem=args.output/f'f{fid}_s{seed}_{variant}'
                np.save(stem.with_suffix('.npy'),trail.cpu().numpy())
                with stem.with_suffix('.csv').open('w',newline='') as stream:
                    writer=csv.DictWriter(stream,fieldnames=records[0].keys()); writer.writeheader();writer.writerows(records)
                with (args.output/'summary.csv').open('w',newline='') as stream:
                    writer=csv.DictWriter(stream,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)
                print(json.dumps(row),flush=True)
    (args.output/'COMPLETE').write_text('All runs passed exact-budget and finite/monotonic-trace checks.\n')


if __name__=='__main__': main()
