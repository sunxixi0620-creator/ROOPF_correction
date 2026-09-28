"""Paired exact-budget comparison against the frozen ROOPF release."""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_roopf as r
from coordinate_experiment import build
from roopf.benchmarks.bbobfunctions import FUNCTIONS as BBOB_FUNCTIONS
from roopf.benchmarks.cecfunctions import FUNCTIONS as CEC_FUNCTIONS
from roopf.benchmarks.utils import setOffset


class CountedBBOB(r.BBOBProblem):
    def __init__(self, fun):
        super().__init__(fun, 10)
        self.actual = 0

    def calfitness(self, x):
        self.actual += x.shape[1]
        return super().calfitness(x)


class CountedCEC(r.CECStyleProblem):
    def __init__(self, fun):
        super().__init__(fun, 10)
        self.actual = 0

    def calfitness(self, x):
        self.actual += x.shape[1]
        return super().calfitness(x)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--benchmark', choices=['bbob', 'cec', 'both'], default='both')
    parser.add_argument('--functions', default='all')
    parser.add_argument('--seeds', type=int, default=10)
    parser.add_argument('--shift-seed', type=int, default=20260925)
    args = parser.parse_args()
    if args.seeds < 1 or (args.benchmark == 'both' and args.functions != 'all'):
        parser.error('Positive seeds required; function subsets require one benchmark.')
    args.output.mkdir(parents=True, exist_ok=False)
    modes = ('legacy', 'unlocked', 'adaptive_unlocked', 'unlocked_centroid')
    seeds = list(range(20261000, 20261000 + args.seeds))
    (args.output / 'protocol.json').write_text(json.dumps({
        'dimension': 10, 'NFE': 300, 'initial_population': 100, 'seeds': seeds,
        'shift_seed': args.shift_seed, 'modes': modes,
        'hashes': {str(path): r.sha256(path) for path in (
            Path(__file__), Path(build.__code__.co_filename), r.ROOT / 'roopf/model.py',
            r.ROOT / 'roopf/coordinate_variants.py',
            *sorted((r.ROOT / 'checkpoints').glob('*.pt')))},
    }, indent=2))
    rows = []
    offsets = r.load_bbob_offsets(r.ROOT / 'data/bbob_offsets_d10.pkl') if args.benchmark != 'cec' else None
    for benchmark in (('bbob', 'cec') if args.benchmark == 'both' else (args.benchmark,)):
        ids = r.parse_function_ids(args.functions, range(1, 25 if benchmark == 'bbob' else 7))
        for fid in ids:
            if benchmark == 'bbob':
                fun = dict(BBOB_FUNCTIONS[fid], xlb=-5, xub=5)
                setOffset(fun, offsets[fid])
                conditions = [('fixed', fun)]
            else:
                source = CEC_FUNCTIONS[f'cecf{fid}']
                generator = torch.Generator().manual_seed(args.shift_seed + fid)
                shift = source['blb'] + (source['bub'] - source['blb']) * torch.rand(10, generator=generator)
                conditions = [('zero', dict(source, bias=torch.zeros(10, device=r.DEVICE))),
                              ('shifted', dict(source, bias=shift.to(r.DEVICE)))]
            for condition, fun in conditions:
                case = f'{benchmark}_f{fid}_{condition}'
                population = torch.stack([
                    fun['xlb'] + (fun['xub'] - fun['xlb']) * torch.rand(
                        (100, 10), generator=torch.Generator().manual_seed(seed))
                    for seed in seeds
                ]).to(r.DEVICE)
                for mode in modes:
                    optimizer = build(mode)
                    r.set_seed(20262000 + fid)
                    problem = CountedBBOB(fun) if benchmark == 'bbob' else CountedCEC(fun)
                    with torch.no_grad():
                        _, trail, nfe, _ = optimizer(population.clone(), problem)
                    assert nfe == problem.actual == 300
                    assert torch.isfinite(trail).all()
                    assert (trail[:, 1:] <= trail[:, :-1] + 1e-6).all()
                    for seed, final in zip(seeds, trail[:, -1].tolist()):
                        rows.append({'case': case, 'variant': mode, 'seed': seed,
                                     'final': final, 'actual_nfe': problem.actual})
                    print(f'{case} {mode}: {trail[:, -1].mean().item():.8g}', flush=True)
                with (args.output / 'raw_results.csv').open('w', newline='') as stream:
                    writer = csv.DictWriter(stream, fieldnames=rows[0]); writer.writeheader(); writer.writerows(rows)
    summary = []
    for case in dict.fromkeys(row['case'] for row in rows):
        per_mode = {mode: np.array([row['final'] for row in rows if row['case'] == case and row['variant'] == mode]) for mode in modes}
        baseline = per_mode['legacy']
        for mode in modes[1:]:
            current = per_mode[mode]
            summary.append({'case': case, 'variant': mode, 'legacy_mean': baseline.mean(),
                            'variant_mean': current.mean(), 'wins': int((current < baseline).sum()),
                            'ties': int((current == baseline).sum()), 'losses': int((current > baseline).sum())})
    with (args.output / 'summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=summary[0]); writer.writeheader(); writer.writerows(summary)
    (args.output / 'COMPLETE').write_text('All exact-budget checks passed.\n')


if __name__ == '__main__':
    main()