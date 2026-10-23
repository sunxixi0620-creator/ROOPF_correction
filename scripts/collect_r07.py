import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import argparse
import gzip
import json
import multiprocessing
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT))
import experiment_r04 as generated
import experiment_r07_full as run
from roopf.joint_generation_agent import JointGenerationAgent, FEATURES, POLICY_ID, PROFILES, digest

EXP, legacy = run.EXP, run.legacy


def freeze_collection():
    identity = run.binding(include_memory=False)
    path = EXP / 'COLLECTION_BINDING.json'
    if path.exists() and json.loads(path.read_text()) != identity:
        raise RuntimeError('Collection source changed; preserve previous attempt')
    if not path.exists():
        legacy.write_json(path, identity)
    return identity


def episode(function_index, seed_index, forced, cache=None):
    model, problem, population, rng, seed, function = generated.episode(function_index, seed_index)
    controller = JointGenerationAgent(None, EXP / 'collection_no_api', mode='collect')
    controller.forced = forced
    records = []
    controller.active_record = records.append
    model.fusion_agent = controller
    measured = legacy.Counted(problem, cache, stochastic=True)
    torch.set_rng_state(rng)
    with torch.no_grad():
        _, trail, nfe, _ = model(population.clone(), measured)
    values = torch.cat(measured.values, 1).numpy()
    if nfe != 300 or values.shape != (1, 300) or len(records) != 45:
        raise RuntimeError('Offline episode incomplete')
    if not np.array_equal(values.min(1), trail[:, -1].numpy()):
        raise RuntimeError('Offline best trace mismatch')
    return dict(records=records, values=values, calls=measured.new_calls, seed=seed,
        function=function, native_fe=measured.native_fe, cached_fe=measured.cached_fe,
        initial_population=population.numpy(), initial_rng=rng.numpy()), problem


def collect(task):
    function_index, seed_index = task
    identity = freeze_collection()
    folder = EXP / 'collection' / f'f{function_index:02d}_s{seed_index:02d}'
    folder.mkdir(parents=True, exist_ok=True)
    marker = folder / 'COMPLETE.json'
    if marker.exists():
        receipt = json.loads(marker.read_text())
        if receipt['binding'] != identity:
            raise RuntimeError('Offline receipt binding mismatch')
        return receipt
    names = list(PROFILES)
    prior_profile = names[1 + (function_index + seed_index) % (len(names) - 1)]
    cases, native_total, diagnostic_total = [], 0, 0
    for decision_step in (5, 15):
        prefix_actions = {} if decision_step == 5 else {5: {0: prior_profile}}
        reference, problem = episode(function_index, seed_index, prefix_actions)
        native_total += reference['native_fe']
        run.storage.save_pickle(folder / f'd{decision_step:02d}_reference.pkl.gz', reference)
        cache = run.storage.merged_calls({}, reference['calls'])
        current = reference['records'][decision_step]['current'][0]
        context = current['generation']
        prefix = context['nfe_before']
        scale = float(np.std(reference['values'][0, :prefix]))
        scale = max(scale, 1e-8)
        case = dict(id=f"{reference['function']}:s{seed_index}:d{decision_step}",
            function=reference['function'], seed_index=seed_index, step=decision_step,
            prefix_profile='release' if decision_step == 5 else prior_profile,
            features=context['features'], generation_context_sha256=digest(context),
            scale=scale, nfe_before=prefix, advantages={'5': {}, '10': {}},
            candidate_diagnostics={}, policy_id=POLICY_ID)
        for name in names:
            if name == 'release':
                branch = reference
            else:
                forced = dict(prefix_actions)
                forced[decision_step] = {0: name}
                branch, problem = episode(function_index, seed_index, forced, cache)
                native_total += branch['native_fe']
                cache = run.storage.merged_calls(cache, branch['calls'])
                run.storage.save_pickle(folder / f'd{decision_step:02d}_{name}.pkl.gz', branch)
            if not np.array_equal(branch['values'][:, :prefix], reference['values'][:, :prefix]):
                raise RuntimeError('Counterfactual prefix differs')
            branch_context = branch['records'][decision_step]['current'][0]['generation']
            if digest(branch_context) != case['generation_context_sha256']:
                raise RuntimeError('Counterfactual pre-generation state differs')
            for horizon in (5, 10):
                stop = prefix + 2 * horizon
                reference_best = float(reference['values'][0, :stop].min())
                branch_best = float(branch['values'][0, :stop].min())
                case['advantages'][str(horizon)][name] = (reference_best - branch_best) / scale
            row = branch['records'][decision_step]['current'][0]
            candidates = torch.tensor(row['candidates'], dtype=torch.float32).unsqueeze(0)
            observations = []
            with torch.random.fork_rng(devices=[]), torch.no_grad():
                torch.manual_seed(reference['seed'] + 100000 + decision_step)
                for slot in range(candidates.size(1)):
                    observations.append(float(problem.calfitness(candidates[:, slot:slot + 1]).reshape(-1)[0]))
            diagnostic_total += candidates.size(1)
            selected_value = observations[row['selected_index']]
            case['candidate_diagnostics'][name] = dict(values=observations,
                selected_index=row['selected_index'], sampled_pool_best=min(observations),
                sampled_selection_gap=selected_value - min(observations),
                anchor=row['anchor'], extra_origin=row['extra_origin'],
                never_provided_to_runtime=True, stochastic_snapshot_not_causal=True)
        cases.append(case)
    with gzip.open(folder / 'CASES.json.gz', 'wt') as stream:
        json.dump(cases, stream, allow_nan=False)
    receipt = dict(status='complete', function=reference['function'], seed_index=seed_index,
        cases=len(cases), native_fe=native_total, diagnostic_native_fe=diagnostic_total,
        binding=identity, real_api_calls=0)
    legacy.write_json(marker, receipt)
    return receipt


def memory():
    cases, receipts = [], []
    identity = freeze_collection()
    for function_index in range(36):
        for seed_index in range(4):
            folder = EXP / 'collection' / f'f{function_index:02d}_s{seed_index:02d}'
            receipt = json.loads((folder / 'COMPLETE.json').read_text())
            if receipt['binding'] != identity:
                raise RuntimeError('Mixed-policy collection')
            receipts.append(receipt)
            with gzip.open(folder / 'CASES.json.gz', 'rt') as stream:
                cases.extend(json.load(stream))
    training = [row for row in cases if row['function'] not in legacy.HELDOUT]
    heldout = [row for row in cases if row['function'] in legacy.HELDOUT]
    native_fe = sum(row['native_fe'] + row['diagnostic_native_fe'] for row in receipts)
    if native_fe > run.settings()['offline_native_fe_cap']:
        raise RuntimeError('Offline objective budget exceeded')
    compact = [{key: row[key] for key in ('id', 'function', 'features', 'advantages', 'policy_id', 'prefix_profile')}
        for row in training]
    result = dict(policy_id=POLICY_ID, cases=compact, feature_names=FEATURES,
        allowed_configs=list(PROFILES), heldout_functions=sorted(legacy.HELDOUT),
        action_duration=5, candidate_count=38, no_test_suite_labels=True,
        policy_screening=False, collection_binding=identity)
    legacy.write_json(run.MEMORY, result)
    with gzip.open(EXP / 'HELDOUT_CASES.json.gz', 'wt') as stream:
        json.dump(heldout, stream, allow_nan=False)
    summary = dict(status='collected_not_performance_validated', episodes=len(receipts),
        training_cases=len(training), heldout_cases=len(heldout), native_fe=native_fe,
        policy_screening=False, retained_profiles=list(PROFILES))
    legacy.write_json(EXP / 'COLLECTION_SUMMARY.json', summary)
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['collect', 'memory'])
    arguments = parser.parse_args()
    if arguments.phase == 'memory':
        memory()
    else:
        freeze_collection()
        with multiprocessing.get_context('spawn').Pool(run.settings()['workers']) as pool:
            for receipt in pool.imap_unordered(collect, [(function, seed) for function in range(36) for seed in range(4)]):
                print(json.dumps({key: value for key, value in receipt.items() if key != 'binding'}), flush=True)
