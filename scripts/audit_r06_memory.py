import gzip
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / 'experiments/r06_event_reviewed_fusion'


def load(path):
    with gzip.open(path, 'rb') as stream:
        return pickle.load(stream)


def main():
    memory = json.loads((EXP / 'offline_memory.json').read_text())
    with gzip.open(EXP / 'CALIBRATION_CASES.json.gz', 'rt') as stream:
        validation = json.load(stream)
    train_functions = {row['function'] for row in memory['cases']}
    heldout = set(memory['heldout_functions'])
    if train_functions & heldout or any(row['function'] not in heldout for row in validation):
        raise RuntimeError('Training/validation contamination')
    binding = json.loads((EXP / 'COLLECTION_BINDING.json').read_text())
    for name, expected in binding['code'].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError('Frozen collection source changed')
    checked, native, branches, real_reviews = 0, 0, 0, 0
    for folder in sorted((EXP / 'collection').iterdir()):
        if not folder.is_dir():
            continue
        receipt = json.loads((folder / 'COMPLETE.json').read_text())
        baseline = load(folder / 'BASELINE.pkl.gz')
        values = baseline['values']
        if values.shape != (1, 300) or len(baseline['records']) != 45:
            raise RuntimeError('Incomplete baseline episode')
        local_native = sum(np.prod(call['shape']) // 10 for call in baseline['calls'])
        if local_native != 300:
            raise RuntimeError('Training objective accounting differs')
        with gzip.open(folder / 'CASES.json.gz', 'rt') as stream:
            cases = json.load(stream)
        actual_branches = {}
        for path in folder.glob('d*_i*.pkl.gz'):
            data = load(path)
            branch_native = sum(np.prod(call['shape']) // 10 for call in data['calls'])
            if branch_native != data['native_fe'] or branch_native > 20:
                raise RuntimeError('Counterfactual native count differs')
            prefix = data['prefix_nfe']
            if not np.array_equal(data['values'][:, :prefix], values[:, :prefix]):
                raise RuntimeError('Counterfactual prefix differs')
            step = int(path.stem[1:3])
            selected = int(path.name.split('_i', 1)[1].split('.', 1)[0])
            review = data['records'][step + 5]['feedback'][0]
            if not review['completed_reviews']:
                raise RuntimeError('Real-objective action did not receive5-round review')
            for item in review['completed_reviews']:
                if item['step'] != step or review['adaptation']['config_incumbent_utility'][item['config']]['n'] < 1:
                    raise RuntimeError('Real-objective review lacks prior executed-action utility')
                expected_progress = max(0.0, item['start_best'] - float(data['values'][:, :prefix + 10].min())) / item['scale']
                if not np.isclose(expected_progress, item['observed_best_progress'], rtol=1e-6, atol=1e-8):
                    raise RuntimeError('Real-objective review progress differs')
            real_reviews += 1
            actual_branches[step, selected] = data
            local_native += branch_native
            branches += 1
        if local_native != receipt['native_fe']:
            raise RuntimeError('Episode accounting differs')
        native += local_native
        for case in cases:
            if case['candidate_count'] != 37 or case['policy_id'] != memory['policy_id']:
                raise RuntimeError('Old candidate/action semantics detected')
            current = dict(case['current'])
            identity = current.pop('context_sha256')
            if hashlib.sha256(json.dumps(current, sort_keys=True).encode()).hexdigest() != identity:
                raise RuntimeError('Full-state identity differs')
            step = case['decision_step']
            for name, action in case['current']['actions'].items():
                branch = actual_branches.get((step, action['selected_index']))
                outcomes = values if branch is None else branch['values']
                for horizon in (1, 5, 10):
                    stop = case['current']['nfe_before'] + 2 * horizon
                    expected = (float(values[:, :stop].min()) - float(outcomes[:, :stop].min())) / case['current']['scale']
                    if not np.isclose(expected, case['horizon_advantages'][str(horizon)][name], atol=1e-12, rtol=1e-12):
                        raise RuntimeError('Paired horizon label differs from raw outcomes')
                    checked += 1
    validation_record = json.loads((EXP / 'OFFLINE_VALIDATION.json').read_text())
    if native != validation_record['native_training_fe'] or native > 100000:
        raise RuntimeError('Offline native cap or accounting differs')
    result = dict(status='passed', labels_recalculated=checked, branches=branches, native_fe=int(native),
        training_functions=len(train_functions), heldout_functions=len(heldout),
        real_objective_local_reviews_verified=real_reviews,
        full37_policy_verified=True, exact_prefix_verified=True, no_split_contamination=True,
        code_binding_verified=True, no_test_benchmark_labels=True)
    (EXP / 'LABEL_AUDIT.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
