import gzip
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import experiment_r02 as legacy

EXP = ROOT / 'experiments/r03_evidence_fusion'


def save(name, value):
    (EXP / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def main():
    contracts = json.loads((EXP / 'CONTRACT_VERIFICATION.json').read_text())
    if contracts['status'] != 'passed':
        raise RuntimeError('Contracts not verified')
    binding = json.loads((EXP / 'EVALUATION_BINDING.json').read_text())
    for name, sha in binding['code'].items():
        if legacy.digest((ROOT / name).read_bytes()) != sha:
            raise RuntimeError(f'Evaluation code changed: {name}')
    memory = json.loads((EXP / 'offline_memory.json').read_text())
    if legacy.digest((EXP / 'offline_memory.json').read_bytes()) != binding['memory']:
        raise RuntimeError('Memory changed')
    if memory['test_data_used'] or len(memory['cases']) != 1305:
        raise RuntimeError('Invalid training memory')
    if {row['function'] for row in memory['cases']} & set(memory['heldout_functions']):
        raise RuntimeError('Heldout leakage')
    if any(any(key not in legacy.HELDOUT for key in value['functions'])
           for context in memory['context_validation'].values() for value in context.values()):
        raise RuntimeError('Invalid calibration sources')
    native = changed = records = requests = failures = 0
    verified = []
    for group, fid in legacy.TASKS:
        folder = EXP / 'results' / f'{group}_f{fid:02d}'
        reference = legacy.original_values(group, fid)[:10]
        initial_hashes = set()
        for mode in ('off', 'retrieval'):
            values = np.load(folder / f'{mode}.npz')['values']
            receipt = json.loads((folder / f'{mode}.json').read_text())
            if values.shape != (10, 300) or receipt['nfe'] != 300:
                raise RuntimeError('Invalid NFE')
            if not np.array_equal(values, reference):
                raise RuntimeError('Unexpected baseline divergence; inspect before publication')
            if receipt['code'] != binding['code'] or receipt['memory'] != binding['memory']:
                raise RuntimeError('Mixed evidence versions')
            initial_hashes.add(receipt['paired_initial_sha256'])
            native += receipt['native_fe']
            changed += receipt['agent_stats']['changed_points']
            requests += receipt['agent_stats']['requests']
            failures += receipt['agent_stats']['failures']
            with gzip.open(folder / f'{mode}_decisions.json.gz', 'rt') as stream:
                decisions = json.load(stream)
            expected = 45 if mode == 'retrieval' and group != 'CEC-style' else 0
            if len(decisions) != expected:
                raise RuntimeError('Decision log incomplete')
            for step, decision in enumerate(decisions):
                if decision['step'] != step or len(decision['current']) != 10:
                    raise RuntimeError('Invalid logged batch')
                if any(name != 'release' for name in decision['configs']):
                    raise RuntimeError('Unexpected unsupported action')
                if any(names != ['release'] for names in decision['eligible_configs']):
                    raise RuntimeError('Unexpected eligible action')
                if step == 0 and any(row['mean_proxy_error'] is not None or row['mean_residual_error'] is not None for row in decision['feedback']):
                    raise RuntimeError('Missing feedback presented as zero')
            records += len(decisions)
        if len(initial_hashes) != 1:
            raise RuntimeError('Unpaired initialization')
        verified.append(f'{group}_f{fid:02d}')
    support = sum(value['supported'] for context in memory['context_validation'].values() for value in context.values())
    if support != 0 or requests or failures or native or changed:
        raise RuntimeError('Unexpected intervention; update analysis rather than reuse no-action claim')
    sources = ['roopf/model.py', 'roopf/fusion_agent.py', 'roopf/fusion_rules.py',
               'scripts/experiment_r03.py', 'scripts/verify_r03.py', 'scripts/finalize_r03.py',
               'scripts/worldcodes_request.ps1']
    save('CODE_BINDING.json', {name: legacy.digest((ROOT / name).read_bytes()) for name in sources})
    result = dict(status='repair_verified_not_an_improvement_claim', functions=35,
        paired_trajectories_per_arm=350, exact_nfe=300, primary_excludes_bbob20=True,
        original_and_repaired_off_bitwise_equal=True,
        guarded_retrieval_bitwise_equal_to_original=True,
        primary_trajectory_wins_ties_losses=[0, 340, 0],
        decision_log_steps=records, trajectory_decisions=records * 10,
        actual_changed_points=changed, supported_context_actions=support,
        training_function_evaluations=memory['training_native_fe'],
        new_benchmark_objective_evaluations=native, live_api_calls=requests,
        live_llm_efficacy_tested=False, api_budget_spent=0,
        conclusion='Shared-rule and evidence plumbing fixed. Current heldout training cases support no alternative action. Do not spend on live benchmark calls or claim optimization gains.',
        next_requirement='Collect independent, more diverse training disagreement cases and assess multi-step utility; freeze protocol before fresh efficacy tests. Do not loosen guards to manufacture intervention.',
        fixed_scope=['OR support semantics and all original veto/lock/penalty guards',
            'persistent semantic decision logs', 'actual residual coefficient bookkeeping',
            'named state, incumbent-relative predictions, per-config candidate action table',
            'conflict-based retrieval across different functions with historical evidence retained',
            'unknown feedback as null; independent proxy/residual counts; competing-slot credit',
            'training-only context calibration and unsupported-state no-API fallback',
            'strict JSON IDs, eligible actions, finish reason, bounded output and paid attempts'],
        unchanged_scope=['frozen anchor/residual weights', 'upstream 36-candidate generation',
            '3-candidate competing shortlist', 'late 30 percent intervention window',
            'one protected Anchor slot', '23-step request cadence', 'CEC structured-system lock'],
        limitations=['Calibration is one-step and heuristic, not a terminal-return guarantee',
            'No stronger-model comparison or actual live model capability claim',
            'Only one seed per training function; anchor pretraining includes all36',
            'Benchmark suite already inspected during development; future paper claims need fresh seeds/tasks'])
    save('FINAL_RESULT.json', result)
    manifest = []
    for path in sorted(EXP.rglob('*')):
        if path.is_file() and path.name not in {'MANIFEST.json', 'COMPLETE.json'}:
            manifest.append(dict(path=str(path.relative_to(EXP)), bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    save('MANIFEST.json', manifest)
    save('COMPLETE.json', dict(status='verified', artifact_files=len(manifest),
        manifest_sha256=legacy.digest((EXP / 'MANIFEST.json').read_bytes()), tasks=verified))
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
