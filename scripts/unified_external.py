"""Frozen CEC2022 verification; no method selection from external outcomes."""
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
import inspect
import json
import multiprocessing as mp
from pathlib import Path
import platform
import sys
import time
import warnings

import cma
import numpy as np
import opfunu
from opfunu.cec_based import cec2022
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from scipy.stats import norm, qmc
import scipy
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from roopf.experiment_io import CaseStore, canonical, seed_for, sha256, write_json
from roopf.unified import build_unified

CONDITIONS = ((10, 300), (20, 300), (20, 600))
METHODS = ('no_residual', 'anchor_only', 'no_residual_score_only', 'cma_es', 'gp_ei')
# Fixed before seeing external outcomes: (dimension, budget, function, run index).
TIMING_PANEL = ((10, 300, 1, 0), (10, 300, 8, 0), (20, 600, 1, 0), (20, 600, 8, 0))
ARTIFACTS = ROOT/'artifacts/unified_revision_v1'
NATIVE20 = ROOT/'results/unified_native20_20260928'
SOURCES = ('scripts/unified_external.py', 'scripts/unified_revision.py',
    'roopf/unified.py', 'roopf/model.py', 'roopf/anchor_backbone.py',
    'roopf/revision_tasks.py', 'roopf/residual_features.py', 'roopf/experiment_io.py',
    'docs/experiments/UNIFIED_EXTERNAL_PROTOCOL.md')


def implementation_identity():
    package = Path(opfunu.__file__).parent
    files = sorted(set(package.rglob('*.py')) | set((package/'cec_based/data_2022').rglob('*')))
    return {str(p.relative_to(package)): sha256(p) for p in files if p.is_file()}


def identity():
    return canonical(dict(version='unified_external_v1',
        sources={p: sha256(ROOT/p) for p in SOURCES},
        anchor10={s: sha256(ARTIFACTS/f'anchor_{s}.pt') for s in range(3)},
        unused_residual={s: sha256(ARTIFACTS/f'residual_{s}.pt') for s in range(3)},
        native20_training=json.loads((NATIVE20/'identity.json').read_text()),
        functions=list(range(1, 13)), conditions=CONDITIONS, runs=30, methods=METHODS,
        timing_panel=TIMING_PANEL,
        versions=dict(torch=torch.__version__, numpy=np.__version__, scipy=scipy.__version__,
            sklearn=sklearn.__version__, cma=cma.__version__, opfunu=opfunu.__version__,
            python=platform.python_version()), opfunu_files=implementation_identity()))


@lru_cache(maxsize=None)
def verify(run):
    expected = json.loads((Path(run)/'identity.json').read_text())
    if identity() != expected:
        raise ValueError('Frozen external implementation or source data changed')
    return expected


class ExternalTask:
    """Count every actual query; the controller receives no optimum information."""
    def __init__(self, fid, dim, budget):
        self.objective = getattr(cec2022, f'F{fid}2022')(ndim=dim)
        self.dim, self.budget = dim, budget
        self.fun = {'fid': 'external_bounded_objective', 'xlb': -5., 'xub': 5.}
        self.points, self.values = [], []
        self.objective_seconds = 0.

    def getfunname(self):
        return self.fun['fid']

    def repaire(self, x):
        return x.clamp(-5, 5)

    def evaluate(self, x):
        if len(self.values) >= self.budget:
            raise RuntimeError('Objective evaluation budget exceeded')
        x = np.asarray(x, dtype=np.float64)
        assert x.shape == (self.dim,) and np.isfinite(x).all()
        assert np.max(np.abs(x)) <= 5.000002
        x = np.clip(x, -5, 5)
        physical = self.objective.lb + (self.objective.ub-self.objective.lb)*(x+5)/10
        begin = time.perf_counter()
        value = float(self.objective.evaluate(physical))
        self.objective_seconds += time.perf_counter()-begin
        if not np.isfinite(value):
            raise ValueError('Nonfinite external objective')
        self.points.append(x.copy()); self.values.append(value)
        return value

    def calfitness(self, x):
        values = [self.evaluate(p) for p in x.detach().cpu().numpy().reshape(-1, self.dim)]
        return torch.as_tensor(values, device=x.device, dtype=x.dtype).reshape(x.shape[:2])

    def diagnostic_fitness(self, x):
        raise RuntimeError('Teacher calls are forbidden in external performance evaluation')


def optimize_kernel(objective, theta, bounds):
    result = minimize(objective, theta, method='L-BFGS-B', jac=True,
                      bounds=bounds, options={'maxiter': 60})
    return result.x, result.fun


def gp_ei(problem, seed):
    d, budget = problem.dim, problem.budget
    rng = np.random.default_rng(seed)
    x = list(qmc.LatinHypercube(d, seed=seed).random(20))
    y = [problem.evaluate(10*v-5) for v in x]
    kernel = ConstantKernel(1., (.01, 100.))*Matern(np.ones(d)*.2, (.01, 10.), nu=2.5)
    for n in range(20, budget):
        gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True,
            optimizer=optimize_kernel if n%20 == 0 else None, n_restarts_optimizer=0)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', ConvergenceWarning)
            gp.fit(np.asarray(x), np.asarray(y))
        kernel = gp.kernel_
        global_x = qmc.Sobol(d, scramble=True,
            seed=seed_for('external_v1', seed, 'sobol', n)).random_base2(9)
        local_x = np.clip(np.asarray(x)[np.argmin(y)]+rng.normal(0, .1, (512, d)), 0, 1)
        candidates = np.vstack((global_x, local_x))
        mu, std = gp.predict(candidates, return_std=True)
        std = np.maximum(std, 1e-12)
        improvement = min(y)-mu; z = improvement/std
        ei = improvement*norm.cdf(z)+std*norm.pdf(z)
        duplicate = cdist(candidates, np.asarray(x), 'sqeuclidean').min(1) < 1e-16
        ei[duplicate] = -np.inf
        assert np.isfinite(ei).any()
        point = candidates[np.argmax(ei)]
        x.append(point); y.append(problem.evaluate(10*point-5))


def cma_es(problem, seed):
    # pycma uses NumPy's legacy 32-bit seed; record both full and effective seeds.
    strategy = cma.CMAEvolutionStrategy(np.zeros(problem.dim), 2., dict(
        seed=seed, bounds=[-5, 5], popsize=10, verbose=-9, verb_log=0))
    while len(problem.values) < problem.budget:
        points = strategy.ask()
        assert len(points) <= problem.budget-len(problem.values)
        strategy.tell(points, [problem.evaluate(x) for x in points])


def checkpoint(dim, group):
    if dim == 10:
        return ARTIFACTS/f'anchor_{group}.pt'
    path = NATIVE20/f'anchor_{group}/selected.pt'
    complete = json.loads((path.parent/'COMPLETE.json').read_text())
    if sha256(path) != complete['selected_sha256']:
        raise ValueError('Native20 selected checkpoint changed')
    return path


def case_spec(dim, budget, fid, run_index, method):
    group = run_index//10
    pseed = seed_for('external_v1', 'population', dim, budget, fid, run_index)
    policy = seed_for('external_v1', 'policy', dim, budget, fid, run_index)
    optimizer = seed_for('external_v1', 'optimizer', method, dim, budget, fid, run_index)
    result = dict(dimension=dim, budget=budget, fid=fid, run_index=run_index,
        group=group, method=method, population_seed=pseed, policy_seed=policy,
        optimizer_seed=optimizer, effective_optimizer_seed=optimizer%(2**32-1) or 1)
    if method not in ('cma_es', 'gp_ei'):
        result['anchor_sha256'] = sha256(checkpoint(dim, group))
        result['unused_residual_sha256'] = sha256(ARTIFACTS/f'residual_{group}.pt')
    return result


def execute(spec):
    torch.set_num_threads(1)
    d, budget, fid, method = (spec[k] for k in ('dimension', 'budget', 'fid', 'method'))
    problem = ExternalTask(fid, d, budget)
    start = time.perf_counter(); decisions = []
    if method == 'gp_ei':
        gp_ei(problem, spec['optimizer_seed'])
    elif method == 'cma_es':
        cma_es(problem, spec['effective_optimizer_seed'])
    else:
        opt = build_unified(checkpoint(d, spec['group']),
            ARTIFACTS/f"residual_{spec['group']}.pt", method, dim=d, budget=budget)
        assert opt.teacher_observer is None
        g = torch.Generator().manual_seed(spec['population_seed'])
        population = 10*torch.rand((1, 100, d), generator=g)-5
        torch.manual_seed(spec['policy_seed'])
        _, _, nfe, _ = opt(population, problem)
        assert nfe == budget and opt.teacher_points == 0
        decisions = opt.decisions
        assert all(not x['residual_active'] for x in decisions)
    elapsed = time.perf_counter()-start
    points = np.asarray(problem.points); values = np.asarray(problem.values)
    assert points.shape == (budget, d) and len(values) == budget
    trace = np.minimum.accumulate(values)
    initial = values[:100]
    gain = max(0., (initial.min()-values.min())/max(initial.std(ddof=1), 1e-8))
    # Known optimum enters reporting only after optimization has finished.
    error = float(trace[-1]-problem.objective.f_global)
    assert error >= -1e-7
    row = {**spec, 'final': float(trace[-1]), 'error': error,
        'log_error': float(np.log10(max(error, 1e-8))), 'actual_nfe': budget,
        'seconds': elapsed, 'objective_seconds': problem.objective_seconds,
        'overhead_seconds': elapsed-problem.objective_seconds,
        'accepted': sum(x['accepted'] for x in decisions),
        'early_accepted': sum(x['accepted'] and x['eval_before'] < .7*budget for x in decisions),
        'initial_best': float(initial.min()), 'initial_scale': float(initial.std(ddof=1)),
        'utility': gain/(1+gain) if method not in ('cma_es', 'gp_ei') else None}
    return dict(row=row, points=points, values=values, trace=trace, decisions=decisions)


def worker(job):
    run, dim, budget, fid, run_index, method = job
    store = CaseStore(Path(run)/'cases', verify(run))
    spec = case_spec(dim, budget, fid, run_index, method)
    key = f'd{dim}_b{budget}_f{fid:02d}_r{run_index:02d}_{method}'
    with store.lock(key):
        value = store.load(key, spec)
        if value is None:
            value = execute(spec); store.save(key, spec, value)
    return value['row']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('prepare', 'baselines', 'learned10', 'learned20'))
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=12)
    args = parser.parse_args(); run = args.run.resolve()
    if args.mode == 'prepare':
        run.mkdir(parents=True, exist_ok=False)
        write_json(run/'identity.json', identity())
        print(run, flush=True); return
    verify(str(run))
    conditions = CONDITIONS if args.mode == 'baselines' else (
        (CONDITIONS[0],) if args.mode == 'learned10' else CONDITIONS[1:])
    methods = METHODS[-2:] if args.mode == 'baselines' else METHODS[:3]
    jobs = [(str(run), d, b, f, r, m) for d, b in conditions for f in range(1, 13)
            for r in range(30) for m in methods]
    if args.mode == 'learned20':
        write_json(run/'NATIVE20_FROZEN.json', {s: sha256(checkpoint(20, s)) for s in range(3)})
    begin = time.perf_counter()
    with ProcessPoolExecutor(args.workers, mp_context=mp.get_context('spawn')) as pool:
        futures = [pool.submit(worker, j) for j in jobs]
        for i, future in enumerate(as_completed(futures)):
            future.result()
            if (i+1)%30 == 0 or i+1 == len(jobs):
                status = dict(completed=i+1, total=len(jobs), seconds=time.perf_counter()-begin)
                write_json(run/f'{args.mode}_status.json', status)
                print(args.mode, status, flush=True)
    write_json(run/f'{args.mode}_COMPLETE.json', dict(cases=len(jobs),
        all_workers_joined=True, seconds=time.perf_counter()-begin))


if __name__ == '__main__':
    main()
