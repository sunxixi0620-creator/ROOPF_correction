"""Frozen post-development 10-D benchmark expansion using external implementations.

COCO supplies new BBOB instances; opfunu supplies its versioned CEC2017 hybrid /
composition classes. Neither is described as an entirely unseen primitive family.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import csv
import importlib.metadata
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import differential_evolution
import cma
import cocoex
from opfunu.cec_based import cec2017

from supplementary_experiments import build, Counted, sync, write_csv, ROOT
import run_roopf as r


METHODS = ('full', 'anchor_only', 'cma_es', 'de', 'random')


class ExternalProblem:
    def __init__(self, suite, fid, instance):
        self.suite_name = suite
        self.dim = 10
        self.trace = []
        self.calls = 0
        if suite == 'coco':
            self.suite = cocoex.Suite('bbob', f'instances: {instance}', f'dimensions: 10 function_indices: {fid}')
            self.objective = self.suite[0]
            lb, ub = self.objective.lower_bounds, self.objective.upper_bounds
            self.label = self.objective.id
            self.fopt = None
        else:
            self.objective = getattr(cec2017, f'F{fid}2017')(ndim=10)
            lb, ub = self.objective.lb, self.objective.ub
            self.label = self.objective.name
            self.fopt = float(self.objective.f_global)
        assert np.all(lb == lb[0]) and np.all(ub == ub[0])
        # Keep the final method's CEC category for the CEC suite. COCO uses its
        # numerical BBOB identifier, activating the same existing BBOB rules.
        self.fun = {'fid': fid if suite == 'coco' else f'cecf2017_{fid}', 'xlb': float(lb[0]), 'xub': float(ub[0])}

    def evaluate(self, x):
        x = np.clip(np.asarray(x, dtype=float), self.fun['xlb'], self.fun['xub'])
        value = float(self.objective(x) if self.suite_name == 'coco' else self.objective.evaluate(x))
        if not np.isfinite(value):
            raise ValueError(f'Nonfinite objective: {self.label}')
        self.calls += 1
        self.trace.append(min(value, self.trace[-1] if self.trace else float('inf')))
        return value

    def getfunname(self):
        return self.fun['fid']

    def repaire(self, x):
        return x.clamp(self.fun['xlb'], self.fun['xub'])

    def calfitness(self, x):
        # Each point evaluated independently: no batch-coupled objective terms.
        arr = x.detach().cpu().numpy().reshape(-1, 10)
        vals = [self.evaluate(v) for v in arr]
        return torch.tensor(vals, dtype=x.dtype, device=x.device).reshape(x.shape[:2])


def run(suite, fid, instance, method, seed):
    objective = ExternalProblem(suite, fid, instance)
    lo, hi = objective.fun['xlb'], objective.fun['xub']
    opt = build(method) if method in ('full', 'anchor_only') else None
    # Single-trajectory calls make runtime latency and per-evaluation trajectories
    # directly observable. Data are not combined with batched Stage 1 times.
    pop = lo + (hi-lo)*torch.rand((1,100,10), generator=torch.Generator().manual_seed(seed))
    r.set_seed(910000+seed)
    sync(); start = time.perf_counter()
    if method in ('full', 'anchor_only'):
        counted = Counted(objective)
        with torch.no_grad():
            _, trail, nfe, _ = opt(pop.to(r.DEVICE), counted)
        assert counted.actual == nfe == 300
    elif method == 'random':
        rng = np.random.default_rng(seed)
        for point in rng.uniform(lo, hi, (300,10)):
            objective.evaluate(point)
    elif method == 'de':
        # One 100-point initial population and two generations = exactly 300.
        differential_evolution(objective.evaluate, [(lo,hi)]*10, init=pop[0].numpy().astype(float),
            maxiter=2, tol=0, atol=0, mutation=(.5,1.), recombination=.7,
            seed=seed, polish=False, updating='immediate', workers=1)
    elif method == 'cma_es':
        es = cma.CMAEvolutionStrategy([.5*(lo+hi)]*10, .2*(hi-lo),
            {'bounds':[lo,hi], 'seed':seed, 'popsize':10, 'verbose':-9,
             'verb_log':0, 'maxfevals':300})
        for _ in range(30):
            points = es.ask(); values = [objective.evaluate(x) for x in points]
            es.tell(points, values)
    sync(); seconds = time.perf_counter() - start
    assert objective.calls == 300, (method, objective.calls)
    trace = np.array(objective.trace)
    assert np.isfinite(trace).all() and np.all(np.diff(trace) <= 0)
    return {'suite':suite, 'fid':fid, 'instance':instance, 'label':objective.label,
        'method':method, 'seed':seed, 'final':float(trace[-1]), 'fopt':objective.fopt,
        'regret':float(trace[-1])-objective.fopt if objective.fopt is not None else None,
        'actual_nfe':objective.calls, 'seconds':seconds}, trace


def job_worker(job):
    suite,fid,instance,seed,output=job
    output=Path(output);rows=[]
    for method in METHODS:
        name=f'{suite}_f{fid}_i{instance}_{method}_s{seed}'
        record=output/f'{name}.json'
        if record.exists():
            rows.append(json.loads(record.read_text()));continue
        row,trace=run(suite,fid,instance,method,seed)
        np.save(output/f'{name}.npy',trace)
        record.write_text(json.dumps(row,indent=2));rows.append(row)
    return rows


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--smoke',action='store_true');p.add_argument('--resume',action='store_true');p.add_argument('--workers',type=int,default=12)
    args=p.parse_args()
    if args.smoke:
        tasks=[('coco',2,999),('cec2017',10,1)]
        seeds=[20269999]
    else:
        tasks=[('coco',f,i) for f in range(1,25) for i in (101,102)]+[('cec2017',f,1) for f in range(10,30)]
        seeds=list(range(20265000,20265010))
    protocol={'tasks':tasks,'seeds':seeds,'methods':METHODS,'dimension':10,'NFE':300,
        'versions':{pkg:importlib.metadata.version(pkg) for pkg in ('coco-experiment','opfunu','cma','scipy','torch','numpy')},
        'torch_threads':torch.get_num_threads(),'device':str(r.DEVICE),'workers':args.workers,
        'source_hashes':{str(f.relative_to(ROOT)):r.sha256(f) for f in [Path(__file__),ROOT/'roopf/model.py',ROOT/'roopf/supplement.py',ROOT/'scripts/supplementary_experiments.py',*sorted((ROOT/'checkpoints').glob('*.pt'))]},
        'scope':'Frozen post-development instances/compositions. Not a claim all primitive function families were unseen during ELA curation.',
        'cec_implementation':'opfunu 1.0.4 class numbering F10..F29 (hybrids/compositions); not claimed as official competition implementation.',
        'initialization':'ROOPF/anchor/DE share 100 points. CMA-ES starts at domain center with sigma=0.2*width and population10. Random uses 300 uniform points.',
        'runtime':'parallel throughput run: wall times not controlled latency comparisons; ROOPF instrumented; no diagnostic queries; external evaluation float64, ROOPF observations cast float32.',
        'selection':'All cases predeclared, including failures. No hyperparameter tuning on results.'}
    # Normalize tuples for exact resume comparison.
    protocol=json.loads(json.dumps(protocol))
    if args.resume:
        assert json.loads((args.output/'protocol.json').read_text())==protocol
    else:
        args.output.mkdir(parents=True,exist_ok=False)
        (args.output/'protocol.json').write_text(json.dumps(protocol,indent=2))
    rows=[]
    jobs=[(suite,fid,instance,seed,str(args.output)) for suite,fid,instance in tasks for seed in seeds]
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        futures=[pool.submit(job_worker,j) for j in jobs]
        for future in as_completed(futures):
            rows.extend(future.result())
            write_csv(args.output/'raw_results.csv',rows)
            if len(rows)%50==0:
                print(json.dumps({'completed':len(rows),'expected':len(jobs)*len(METHODS)}),flush=True)
    assert len(rows)==len(jobs)*len(METHODS)
    assert len({(x['suite'],x['fid'],x['instance'],x['method'],x['seed']) for x in rows})==len(rows)
    write_csv(args.output/'raw_results.csv',rows)
    (args.output/'COMPLETE').write_text(f'{len(rows)} exact-budget trajectories completed.\n')


if __name__=='__main__':main()
