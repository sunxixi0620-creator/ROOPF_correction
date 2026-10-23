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
import experiment_r04 as previous
from roopf.reviewed_fusion_agent import ReviewedFusionAgent, POLICY_ID, CONFIGS

legacy = previous.legacy
EXP = ROOT / 'experiments/r06_event_reviewed_fusion'
BOOTSTRAP = EXP / 'BOOTSTRAP.json'
SOURCES = ('roopf/model.py', 'roopf/adaptive_fusion_agent.py', 'roopf/reviewed_fusion_agent.py',
    'roopf/fusion_agent.py', 'roopf/fusion_rules.py', 'scripts/collect_r06.py',
    'scripts/verify_r06.py', 'scripts/experiment_r02.py', 'scripts/experiment_r04.py',
    'run_roopf.py', 'checkpoints/anchor_policy_d10.pt', 'checkpoints/residual_selector_generated36_d10.pt',
    'experiments/r02_fusion_agent/training_assets/generated36_curated.py')


def freeze():
    binding = dict(code={name: legacy.digest((ROOT / name).read_bytes()) for name in SOURCES},
        protocol_sha256=legacy.digest((EXP / 'PROTOCOL.json').read_bytes()),
        bootstrap_sha256=legacy.digest(BOOTSTRAP.read_bytes()))
    target = EXP / 'COLLECTION_BINDING.json'
    if target.exists() and json.loads(target.read_text()) != binding:
        raise RuntimeError('Collection code/protocol changed; preserve existing attempt')
    if not target.exists():
        legacy.write_json(target, binding)
    return binding


def agent():
    return ReviewedFusionAgent(BOOTSTRAP, EXP / 'offline_no_api', mode='collect')


def calls_from(rows):
    return {index: (dict(shape=row['shape'], x=row['x'], rng=legacy.digest(row['rng_before'])),
        dict(shape=row['shape'][:2], y=row['y'], rng_after=row['rng_after'])) for index, row in enumerate(rows)}


def collect(task):
    function_index, seed_index = task
    folder = EXP / 'collection' / f'f{function_index:02d}_s{seed_index:02d}'
    folder.mkdir(parents=True, exist_ok=True)
    binding = freeze()
    receipt_path = folder / 'COMPLETE.json'
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if receipt['binding'] != binding:
            raise RuntimeError('Completed collection binding differs')
        return receipt
    model, problem, population, rng, seed, function = previous.episode(function_index, seed_index)
    controller = agent()
    records = []
    controller.active_record = records.append
    model.fusion_agent = controller
    measured = legacy.Counted(problem, stochastic=True)
    torch.set_rng_state(rng)
    with torch.no_grad():
        _, trail, nfe, _ = model(population.clone(), measured)
    values = torch.cat(measured.values, 1).numpy()
    if nfe != 300 or measured.native_fe != 300 or len(records) != 45 or not np.array_equal(values.min(1), trail[:, -1].numpy()):
        raise RuntimeError('Invalid full37 baseline episode')
    previous.save_pickle(folder / 'BASELINE.pkl.gz', dict(records=records, values=values,
        calls=measured.new_calls, seed=seed, rng=rng.numpy(), population=population.numpy()))
    calls = calls_from(measured.new_calls)
    chosen = []
    for start, stop in ((5, 12), (13, 20), (21, 28), (29, 34)):
        available = [row for row in records if start <= row['step'] <= stop and len(row['eligible_configs'][0]) > 1]
        if available:
            chosen.append(available[0])
    cases, native, branches = [], 300, 0
    for record in chosen:
        row = record['current'][0]
        step, prefix = record['step'], row['nfe_before']
        stops = {str(horizon): prefix + 2 * horizon for horizon in (1, 5, 10)}
        label = dict(id=f'{function}:s{seed_index}:d{step}', function=function, seed=seed,
            seed_index=seed_index, decision_step=step, policy_id=POLICY_ID, candidate_count=37,
            context_key=row['context_key'], fusion_features=row['fusion_features'],
            context_sha256=row['context_sha256'], current=row,
            action_changes={name: row['actions'][name]['changes_original'] for name in CONFIGS},
            horizon_advantages={horizon: {} for horizon in stops}, horizon_best_values={horizon: {} for horizon in stops})
        evaluated = {}
        original = row['actions']['release']['selected_index']
        for name in CONFIGS:
            selected = row['actions'][name]['selected_index']
            if selected == original or selected in evaluated:
                continue
            branch_model = legacy.build()
            forced = agent()
            forced.forced = {step: {'0': name}}
            branch_records = []
            forced.active_record = branch_records.append
            branch_model.fusion_agent = forced
            objective = previous.PrefixObjective(problem, calls, stops['10'])
            torch.set_rng_state(rng)
            try:
                with torch.no_grad():
                    branch_model(population.clone(), objective)
                raise RuntimeError('Counterfactual did not stop')
            except previous.HorizonReached:
                pass
            branch_values = torch.cat(objective.values, 1).numpy()
            forced_row = branch_records[step]
            if forced_row['current'][0]['context_sha256'] != row['context_sha256']:
                raise RuntimeError('Counterfactual pre-action context differs')
            if not np.array_equal(branch_values[:, :prefix], values[:, :prefix]) or objective.native_fe > 20:
                raise RuntimeError('Counterfactual prefix/NFE differs')
            if sum(item['configs'][0] != 'release' for item in branch_records) != 1:
                raise RuntimeError('Counterfactual is not one-action')
            previous.save_pickle(folder / f'd{step:02d}_i{selected:02d}.pkl.gz',
                dict(values=branch_values, calls=objective.new_calls, records=branch_records,
                    native_fe=objective.native_fe, cached_fe=objective.cached_fe,
                    prefix_nfe=prefix, stop_nfe=stops['10'], context_sha256=row['context_sha256']))
            native += objective.native_fe
            branches += 1
            evaluated[selected] = branch_values
        for name in CONFIGS:
            branch_values = evaluated.get(row['actions'][name]['selected_index'], values)
            for horizon, stop in stops.items():
                reference, outcome = float(values[:, :stop].min()), float(branch_values[:, :stop].min())
                label['horizon_best_values'][horizon][name] = outcome
                label['horizon_advantages'][horizon][name] = (reference - outcome) / row['scale']
        cases.append(label)
    with gzip.open(folder / 'CASES.json.gz', 'wt') as stream:
        json.dump(cases, stream, allow_nan=False)
    receipt = dict(status='complete', function=function, seed_index=seed_index, seed=seed,
        retained_cases=len(cases), counterfactual_branches=branches, native_fe=native, binding=binding)
    legacy.write_json(receipt_path, receipt)
    return receipt


def build_memory():
    receipts, cases = [], []
    for function_index in range(36):
        for seed_index in range(4):
            folder = EXP / 'collection' / f'f{function_index:02d}_s{seed_index:02d}'
            receipt = json.loads((folder / 'COMPLETE.json').read_text())
            if receipt['binding'] != freeze():
                raise RuntimeError('Mixed collection binding')
            receipts.append(receipt)
            with gzip.open(folder / 'CASES.json.gz', 'rt') as stream:
                cases.extend(json.load(stream))
    training = [row for row in cases if row['function'] not in legacy.HELDOUT]
    validation = [row for row in cases if row['function'] in legacy.HELDOUT]
    if not training or not validation:
        raise RuntimeError('Missing training or validation cases')
    memory = dict(version='r06', policy_id=POLICY_ID, candidate_count=37, allowed_configs=list(CONFIGS),
        cases=training, action_duration_rounds=1, primary_horizon_rounds=5,
        heldout_functions=sorted(legacy.HELDOUT), feature_definition='full37 fusion_features',
        validation_case_count=len(validation), test_suite_used_for_labels=False)
    legacy.write_json(EXP / 'offline_memory.json', memory)
    with gzip.open(EXP / 'CALIBRATION_CASES.json.gz', 'wt') as stream:
        json.dump(validation, stream, allow_nan=False)
    audit = dict(status='passed', episodes=len(receipts), train_cases=len(training), validation_cases=len(validation),
        generated_training_functions=36, seeds_per_function=4, candidate_count=37,
        native_training_fe=sum(row['native_fe'] for row in receipts),
        paired_prefix_and_policy_identity_verified=True, no_test_function_labels=True,
        heldout_functions=sorted(legacy.HELDOUT),
        validation_advantage_mean={name: float(np.mean([row['horizon_advantages']['5'][name] for row in validation])) for name in CONFIGS})
    legacy.write_json(EXP / 'OFFLINE_VALIDATION.json', audit)
    print(json.dumps(audit), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['run', 'memory'])
    arguments = parser.parse_args()
    if arguments.phase == 'run':
        freeze()
        with multiprocessing.get_context('spawn').Pool(4) as pool:
            for receipt in pool.imap_unordered(collect, [(function, seed) for function in range(36) for seed in range(4)]):
                print(json.dumps({key: value for key, value in receipt.items() if key != 'binding'}), flush=True)
    else:
        build_memory()
