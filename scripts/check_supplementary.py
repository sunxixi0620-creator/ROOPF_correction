"""Numerical parity against the GitHub-frozen method before costly experiments."""
import json
import subprocess
import sys
import types
from pathlib import Path

import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from supplementary_experiments import build, problem_for
import run_roopf as r


def frozen():
    module = types.ModuleType('frozen_roopf_model')
    source = subprocess.check_output(['git', 'show', 'roopf-final-before-supplement-20260928:roopf/model.py'], cwd=ROOT, text=True)
    exec(compile(source, 'frozen_roopf_model.py', 'exec'), module.__dict__)
    r.set_seed(20260630)
    opt = module.ROOPFOptimizer(dim=10, hidden_dim=200, popSize=100, max_nfe=300,
        k_nums=2, pool_per_op=6, surrogate_members=5, ablation='roopf',
        baseline_ckpt=str(ROOT/'checkpoints/anchor_policy_d10.pt'),
        router_ckpt=str(ROOT/'checkpoints/residual_selector_generated36_d10.pt'),
        router_weight=.008).to(r.DEVICE).eval()
    opt.ablation.discard('structured_system_lock')
    return opt


def execute(opt, case):
    problem, _ = problem_for(case)
    pop = problem.fun['xlb'] + (problem.fun['xub'] - problem.fun['xlb']) * torch.rand((2, 100, 10), generator=torch.Generator().manual_seed(7182))
    r.set_seed(9941)
    with torch.no_grad():
        result = opt(pop.to(r.DEVICE), problem)
    assert result[2] == problem.actual == 300
    rng = torch.cuda.get_rng_state() if r.DEVICE.type == 'cuda' else torch.get_rng_state()
    return result, rng, problem


def main():
    records = []
    for case in ('bbob_f1_fixed', 'bbob_f9_fixed', 'bbob_f20_fixed', 'cec_f1_shifted'):
        expected, expected_rng, _ = execute(frozen(), case)
        for stride in (0, 10):
            actual, rng, problem = execute(build('full', diagnostic_stride=stride), case)
            for index in (0, 1, 3):
                assert torch.equal(expected[index], actual[index]), (case, stride, index)
            assert torch.equal(expected_rng, rng), (case, 'observer changed RNG')
            assert problem.diagnostic_actual == (8 if stride else 0)
        if case == 'bbob_f20_fixed':
            records.append({'case': case, 'observer_parity': True, 'excluded_from_primary': 'stochastic, batch-coupled objective'})
            print(json.dumps(records[-1]), flush=True)
            continue
        anchor = frozen(); anchor.ablation.add('baseline_only')
        expected, _, _ = execute(anchor, case)
        actual, _, _ = execute(build('anchor_only'), case)
        for index in (0, 1, 3):
            assert torch.equal(expected[index], actual[index]), (case, 'anchor', index)
        records.append({'case': case, 'full_trace_points_population_rng_exact': True,
            'observer_queries_separate': True, 'anchor_fast_path_exact': True})
        print(json.dumps(records[-1]), flush=True)
    dest = ROOT/f'results/supplement_parity_{r.DEVICE.type}_20260928.json'
    dest.write_text(json.dumps(records, indent=2)+'\n')

if __name__ == '__main__':
    main()
