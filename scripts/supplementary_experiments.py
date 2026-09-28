"""Frozen-protocol supplementary experiments; immutable per-run outputs and resume."""
import argparse
import csv
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import run_roopf as r
from roopf.supplement import SupplementOptimizer, VARIANTS, sync


def build(variant='full', warmup=.70, diagnostic_stride=0):
    r.set_seed(20260630)
    opt = SupplementOptimizer(dim=10, hidden_dim=200, popSize=100, max_nfe=300,
        k_nums=2, pool_per_op=6, surrogate_members=5, ablation='roopf',
        baseline_ckpt=str(ROOT/'checkpoints/anchor_policy_d10.pt'),
        router_ckpt=str(ROOT/'checkpoints/residual_selector_generated36_d10.pt'),
        router_weight=.008).to(r.DEVICE).eval().configure(variant, warmup, diagnostic_stride)
    original_predict = opt.surrogate.predict
    def measured_predict(*args, **kwargs):
        with opt.measured('proxy_seconds'):
            return original_predict(*args, **kwargs)
    opt.surrogate.predict = measured_predict
    return opt


class Counted:
    def __init__(self, base):
        self.base = base
        self.fun = base.fun
        self.actual = 0
        self.diagnostic_actual = 0
        self.initial_best = None
        self.objective_seconds = 0.

    def __getattr__(self, name):
        return getattr(self.base, name)

    def calfitness(self, x):
        sync(); start = time.perf_counter()
        y = self.base.calfitness(x)
        sync(); self.objective_seconds += time.perf_counter() - start
        self.actual += x.shape[1]
        if self.initial_best is None:
            self.initial_best = y.reshape(x.shape[0], -1).min(dim=1).values.detach().cpu().numpy()
        return y

    def diagnostic_fitness(self, x):
        self.diagnostic_actual += x.shape[1]
        devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
        with torch.random.fork_rng(devices=devices):
            return self.base.calfitness(x).reshape(x.shape[0], -1)


def problem_for(case, shift_seed=20260925):
    group, fid_text, condition = case.split('_')
    fid = int(fid_text[1:])
    if group == 'bbob':
        fun = dict(r.BBOB_FUNCTIONS[fid], xlb=-5, xub=5)
        offsets = r.load_bbob_offsets(ROOT/'data/bbob_offsets_d10.pkl')
        r.setOffset(fun, offsets[fid])
        return Counted(r.BBOBProblem(fun, 10)), {'condition': 'fixed', 'fid': fid}
    source = r.CEC_FUNCTIONS[f'cecf{fid}']
    gen = torch.Generator().manual_seed(shift_seed + fid)
    bias = torch.zeros(10) if condition == 'zero' else source['blb'] + (source['bub'] - source['blb']) * torch.rand(10, generator=gen)
    return Counted(r.CECStyleProblem(dict(source, bias=bias.to(r.DEVICE)), 10)), {'condition': condition, 'fid': fid, 'bias': bias.tolist()}


def write_csv(path, rows):
    if not rows:
        path.write_text('')
        return
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def run(case, variant, seeds, algorithm_seed, warmup=.70, diagnostic_stride=0, shift_seed=20260925):
    problem, instance = problem_for(case, shift_seed)
    population = torch.stack([problem.fun['xlb'] + (problem.fun['xub'] - problem.fun['xlb']) * torch.rand((100, 10), generator=torch.Generator().manual_seed(seed)) for seed in seeds]).to(r.DEVICE)
    pop_hash = hashlib.sha256(population.cpu().numpy().tobytes()).hexdigest()
    opt = build(variant, warmup, diagnostic_stride)
    r.set_seed(algorithm_seed)
    sync(); start = time.perf_counter()
    with torch.no_grad():
        _, trail, nfe, points = opt(population.clone(), problem)
    sync(); seconds = time.perf_counter() - start
    assert nfe == problem.actual == 300
    assert torch.isfinite(trail).all() and torch.isfinite(points).all()
    assert (trail[:, 1:] <= trail[:, :-1] + 1e-6).all()
    assert len(opt.candidate_samples) == len(seeds) * 200
    rows = []
    for bi, seed in enumerate(seeds):
        own = [x for x in opt.candidate_samples if x['batch'] == bi]
        decisions = [x for x in opt.decisions if x['batch'] == bi]
        portfolio = [x for x in own if not x['is_baseline']]
        # Attribute actual incumbent decreases sequentially: slot 0 first, then
        # only the extra decrease from slot 1. Never double-count a batch gain.
        incumbent = float(problem.initial_best[bi]); portfolio_gains = []
        for event in own:
            gain = max(0., incumbent - event['candidate_fit'])
            incumbent = min(incumbent, event['candidate_fit'])
            if not event['is_baseline'] and gain > 0:
                portfolio_gains.append(gain)
        sampled = [d for d in decisions if d['diagnostic_sampled']]
        rows.append({'case': case, 'variant': variant, 'warmup': warmup, 'seed': seed,
            'algorithm_seed': algorithm_seed, 'initial_population_sha256': pop_hash,
            'initial_best': float(problem.initial_best[bi]), 'final': float(trail[bi, -1]),
            'actual_nfe': problem.actual, 'diagnostic_nfe': problem.diagnostic_actual,
            'portfolio_evals': len(portfolio), 'portfolio_incumbent_improvements': sum(x['improved_best'] for x in portfolio),
            'portfolio_sequential_gain': sum(portfolio_gains), 'portfolio_top3_gain': sum(sorted(portfolio_gains, reverse=True)[:3]),
            'gate_opportunities': len(decisions), 'sampled_decisions': len(sampled),
            'residual_changes_sampled': sum(int(d['residual_decision_changed']) for d in sampled),
            'batch_size': len(seeds), 'batch_seconds': seconds,
            'observer_seconds': opt.timings['observer_seconds'],
            'online_seconds_excluding_observer': seconds - opt.timings['observer_seconds'],
            'objective_seconds': problem.objective_seconds,
            'candidate_pool_seconds': opt.timings['candidate_pool_seconds'],
            'anchor_seconds': opt.timings['anchor_seconds'], 'ranking_seconds': opt.timings['ranking_seconds'],
            'proxy_seconds': opt.timings['proxy_seconds']})
    return rows, opt, trail, instance


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--cases', default='all')
    p.add_argument('--variants', default=','.join(VARIANTS))
    p.add_argument('--seeds', type=int, default=30)
    p.add_argument('--batch-size', type=int, default=10)
    p.add_argument('--seed-start', type=int, default=20263000)
    p.add_argument('--shift-seed', type=int, default=20260925)
    p.add_argument('--warmups', default='0.70')
    p.add_argument('--diagnostic-stride', type=int, default=0)
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    cases = ([f'bbob_f{i}_fixed' for i in range(1, 25) if i != 20] + [f'cec_f{i}_{c}' for i in range(1, 7) for c in ('zero', 'shifted')]) if args.cases == 'all' else args.cases.split(',')
    variants = args.variants.split(','); warmups = [float(w) for w in args.warmups.split(',')]
    assert args.seeds > 0 and args.batch_size > 0 and set(variants) <= set(VARIANTS)
    assert all(1/3 <= w <= 1 for w in warmups)
    protocol = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items() if k not in ('output', 'resume')}
    protocol.update({'cases_expanded': cases, 'dimension': 10, 'NFE': 300, 'population': 100,
        'purpose': 'Development/diagnostic evidence; not independent confirmation',
        'excluded': {'bbob_f20_fixed': 'Per-call random signs and cross-point reduction in released objective; unsuitable for fixed noiseless paired aggregate.'},
        'runtime_note': 'Synchronized, instrumented batch wall time; observer excluded separately. Component timers are nested, not additive. No inference from these times about simulator cost.',
        'no_proxy_note': 'Structural ranking plus isolated jitter; residual/gate kept with mu=sigma=0. This is an operational ablation, not a calibrated replacement surrogate.',
        'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'python': platform.python_version(), 'torch': torch.__version__, 'numpy': np.__version__,
        'device': str(r.DEVICE), 'torch_threads': torch.get_num_threads(), 'cpu': platform.processor(), 'gpu': torch.cuda.get_device_name() if torch.cuda.is_available() else '',
        'hashes': {str(f.relative_to(ROOT)): r.sha256(f) for f in [Path(__file__), ROOT/'roopf/model.py', ROOT/'roopf/supplement.py', ROOT/'run_roopf.py', ROOT/'roopf/benchmarks/cecfunctions.py', ROOT/'data/bbob_offsets_d10.pkl', *sorted((ROOT/'checkpoints').glob('*.pt'))]}})
    if args.resume:
        saved = json.loads((args.output/'protocol.json').read_text())
        assert saved == protocol, 'Resume protocol/code mismatch'
    else:
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output/'protocol.json').write_text(json.dumps(protocol, indent=2))
    all_rows = []
    for case_index, case in enumerate(cases):
        for start_seed in range(0, args.seeds, args.batch_size):
            seeds = list(range(args.seed_start + start_seed, args.seed_start + min(args.seeds, start_seed + args.batch_size)))
            algorithm_seed = 20264000 + int(case.split('_')[1][1:]) * 100 + start_seed
            # Rotate order to reduce systematic thermal/order bias; no selection
            # depends on wall time. Build resets identical model weights each run.
            order = variants[case_index % len(variants):] + variants[:case_index % len(variants)]
            for warmup in warmups:
                for variant in order:
                    dest = args.output/f'{case}_{variant}_w{warmup:.4f}_s{seeds[0]}'
                    if (dest/'COMPLETE').exists():
                        all_rows.extend(json.loads((dest/'rows.json').read_text())); continue
                    dest.mkdir(exist_ok=True)
                    rows, opt, trail, instance = run(case, variant, seeds, algorithm_seed, warmup, args.diagnostic_stride, args.shift_seed)
                    (dest/'rows.json').write_text(json.dumps(rows, indent=2))
                    (dest/'instance.json').write_text(json.dumps(instance, indent=2))
                    write_csv(dest/'candidates.csv', opt.candidate_samples)
                    write_csv(dest/'decisions.csv', opt.decisions)
                    np.savez_compressed(dest/'trails.npz', trail=trail.cpu().numpy(), nfe=np.arange(102, 301, 2), seeds=seeds)
                    (dest/'COMPLETE').write_text('Exact budget, finite values, monotonic trace and candidate counts passed.\n')
                    all_rows.extend(rows)
                    write_csv(args.output/'raw_results.csv', all_rows)
                    print(json.dumps({'case': case, 'variant': variant, 'warmup': warmup, 'seeds': seeds,
                        'mean': float(np.mean([x['final'] for x in rows])), 'seconds': rows[0]['batch_seconds'],
                        'completed_trajectories': len(all_rows)}), flush=True)
    write_csv(args.output/'raw_results.csv', all_rows)
    (args.output/'COMPLETE').write_text(f'{len(all_rows)} trajectories completed.\n')


if __name__ == '__main__':
    main()
