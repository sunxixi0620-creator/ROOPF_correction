import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import argparse
import csv
import gzip
import json
from pathlib import Path
import pickle
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT))
import experiment_r02 as legacy
from roopf.fusion_agent import CONFIGS, context_rows, fusion_indices
from roopf.multistep_fusion_agent import MultiStepFusionAgent

EXP = ROOT / 'experiments/r04_multiseed_horizon'
SOURCES = ('roopf/model.py', 'roopf/fusion_rules.py', 'roopf/fusion_agent.py',
           'roopf/multistep_fusion_agent.py', 'scripts/experiment_r04.py',
           'scripts/experiment_r02.py', 'run_roopf.py',
           'experiments/r02_fusion_agent/training_assets/generated36_curated.py',
           'checkpoints/anchor_policy_d10.pt', 'checkpoints/residual_selector_generated36_d10.pt')


def write_json(path, value):
    legacy.write_json(path, value)


def save_pickle(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + '.pending')
    with gzip.open(pending, 'wb') as stream:
        pickle.dump(value, stream, protocol=5)
    pending.replace(path)


def load_pickle(path):
    with gzip.open(path, 'rb') as stream:
        return pickle.load(stream)


def protocol():
    value = json.loads((EXP / 'PROTOCOL.json').read_text())
    if set(value['heldout_functions']) != legacy.HELDOUT:
        raise RuntimeError('Calibration split changed')
    return value


def freeze():
    identity = dict(code={name: legacy.digest((ROOT / name).read_bytes()) for name in SOURCES},
        protocol_sha256=legacy.digest((EXP / 'PROTOCOL.json').read_bytes()))
    target = EXP / 'COLLECTION_BINDING.json'
    if target.exists() and json.loads(target.read_text()) != identity:
        raise RuntimeError('Code/protocol changed after collection started; preserve this round and use a new directory')
    if not target.exists():
        write_json(target, identity)
    return identity


def generated():
    return legacy.load_module('r04_generated36', legacy.EXP / 'training_assets/generated36_curated.py')


def episode(function_index, seed_index):
    module = generated()
    function = dict(module.TRAIN_FUNCTIONS[function_index])
    seed = 2026102400 + 1000 * seed_index + function_index
    model = legacy.build()
    legacy.release.set_seed(seed)
    module.gen_train_offset(10, function)
    problem = legacy.release.BBOBProblem(dict(fid='generated36:' + function['fid'], xlb=-5, xub=5), 10)
    problem.calfitness = lambda candidates: module.get_train_fitness(candidates, function)
    population = problem.genRandomPop((1, 100, 10))
    return model, problem, population, torch.get_rng_state().clone(), seed, function['fid']


def context(kwargs):
    scale = kwargs['archive_y'].std(1, keepdim=True, unbiased=False).clamp_min(1e-8)
    choices = fusion_indices(kwargs['score'], kwargs['probability'], scale,
        kwargs['original_idx'], kwargs['residual_weight'], kwargs['selection_rule'])
    row = context_rows(kwargs['state'], kwargs['score'], kwargs['mu'], kwargs['sigma'],
        kwargs['probability'], scale, kwargs['original_idx'], kwargs['residual_weight'],
        fitness=kwargs['fitness'], candidate_ops=kwargs['candidate_ops'], options=choices)[0]
    row['scale'] = float(scale[0, 0])
    row['nfe_before'] = int(round(300 * (1 - kwargs['remaining'])))
    row['context_sha256'] = legacy.digest(json.dumps(row, sort_keys=True).encode())
    original = int(kwargs['original_idx'][0, 0])
    for name, selected in choices.items():
        chosen = int(selected[0, 0])
        row['actions'][name]['changes_point'] = bool((kwargs['candidates'][0, chosen] != kwargs['candidates'][0, original]).any())
        row['actions'][name]['changes_operator'] = int(kwargs['candidate_ops'][0, chosen]) != int(kwargs['candidate_ops'][0, original])
    return row, choices


class Recorder:
    def reset(self, batch):
        if batch != 1:
            raise ValueError('Collection requires scalar episodes')
        self.records = []

    def observe(self, *args):
        pass

    def choose(self, **kwargs):
        row, _ = context(kwargs)
        row['decision_step'] = len(self.records)
        self.records.append(row)
        return kwargs['original_idx']


class ForcedAction:
    def __init__(self, record, config):
        self.record = record
        self.config = config

    def reset(self, batch):
        self.steps = 0
        self.applied = 0

    def observe(self, *args):
        pass

    def choose(self, **kwargs):
        selected = kwargs['original_idx']
        if self.steps == self.record['decision_step']:
            row, choices = context(kwargs)
            if row['context_sha256'] != self.record['context_sha256']:
                raise RuntimeError('Counterfactual prefix state differs')
            selected = choices[self.config]
            if int(selected[0, 0]) != self.record['actions'][self.config]['selected_index']:
                raise RuntimeError('Counterfactual selected a different action')
            self.applied += 1
        self.steps += 1
        return selected


class HorizonReached(Exception):
    pass


class PrefixObjective(legacy.Counted):
    def __init__(self, problem, calls, stop_nfe):
        super().__init__(problem, calls=calls, stochastic=True)
        self.stop_nfe = stop_nfe

    def calfitness(self, candidates):
        values = super().calfitness(candidates)
        total = self.native_fe + self.cached_fe
        if total > self.stop_nfe:
            raise RuntimeError('Continuation exceeded its evaluation horizon')
        if total == self.stop_nfe:
            raise HorizonReached()
        return values


def collect_episode(function_index, seed_index, folder):
    receipt_path = folder / 'COMPLETE.json'
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if receipt['binding'] != freeze():
            raise RuntimeError('Episode binding mismatch')
        return receipt
    folder.mkdir(parents=True, exist_ok=True)
    model, problem, population, rng, seed, function = episode(function_index, seed_index)
    baseline_path = folder / 'BASELINE.pkl.gz'
    if baseline_path.exists():
        baseline = load_pickle(baseline_path)
        if baseline['seed'] != seed or baseline['population_sha256'] != legacy.digest(population.numpy().tobytes()):
            raise RuntimeError('Baseline episode changed')
    else:
        collector = Recorder()
        model.fusion_agent = collector
        measured = legacy.Counted(problem, stochastic=True)
        torch.set_rng_state(rng)
        frozen = {name: value.clone() for name, value in model.state_dict().items()}
        with torch.no_grad():
            _, trail, nfe, _ = model(population.clone(), measured)
        if nfe != 300 or measured.native_fe != 300 or len(collector.records) != 45:
            raise RuntimeError('Baseline budget or decision coverage invalid')
        if not all(torch.equal(value, frozen[name]) for name, value in model.state_dict().items()):
            raise RuntimeError('Frozen weights changed')
        values = torch.cat(measured.values, 1).numpy()
        if not np.array_equal(values.min(1), trail[:, -1].numpy()):
            raise RuntimeError('Invalid baseline best curve')
        baseline = dict(seed=seed, function=function, seed_index=seed_index, values=values,
            records=collector.records, calls=measured.new_calls,
            population_sha256=legacy.digest(population.numpy().tobytes()),
            solver_rng_sha256=legacy.digest(rng.numpy().tobytes()), native_fe=300)
        save_pickle(baseline_path, baseline)
    calls = {}
    for index, call in enumerate(baseline['calls']):
        request = dict(shape=call['shape'], x=call['x'], rng=legacy.digest(call['rng_before']))
        response = dict(shape=call['shape'][:2], y=call['y'], rng_after=call['rng_after'])
        calls[index] = request, response
    cases, new_fe, branch_count = [], 300, 0
    for record in baseline['records']:
        if not any(action['changes_original'] for action in record['actions'].values()):
            continue
        row = dict(record)
        row.update(id=f"{function}:s{seed_index}:d{row['decision_step']}", function=function,
            seed_index=seed_index, seed=seed, horizon_advantages={str(horizon): {} for horizon in (1, 5, 10)},
            horizon_best_values={str(horizon): {} for horizon in (1, 5, 10)},
            horizon_nfe={str(horizon): min(300, row['nfe_before'] + 2 * horizon) for horizon in (1, 5, 10)})
        by_index = {}
        for name, action in row['actions'].items():
            chosen = action['selected_index']
            if chosen == row['original_index']:
                continue
            if chosen in by_index:
                continue
            branch_path = folder / f"d{row['decision_step']:02d}_i{chosen}.pkl.gz"
            if branch_path.exists():
                branch = load_pickle(branch_path)
                if branch['context_sha256'] != row['context_sha256'] or branch['selected_index'] != chosen:
                    raise RuntimeError('Cached branch changed')
            else:
                branch_model = legacy.build()
                forced = ForcedAction(row, name)
                branch_model.fusion_agent = forced
                measured = PrefixObjective(problem, calls, row['horizon_nfe']['10'])
                torch.set_rng_state(rng)
                try:
                    with torch.no_grad():
                        branch_model(population.clone(), measured)
                    raise RuntimeError('Continuation did not stop at its horizon')
                except HorizonReached:
                    pass
                if forced.applied != 1 or branch_model.MaxNFE != 300:
                    raise RuntimeError('Counterfactual intervention duration/budget invalid')
                values = torch.cat(measured.values, 1).numpy()
                prefix = row['nfe_before']
                if not np.array_equal(values[:, :prefix], baseline['values'][:, :prefix]):
                    raise RuntimeError('Counterfactual prefix objectives differ')
                if measured.native_fe > 20 or measured.native_fe + measured.cached_fe != values.shape[1]:
                    raise RuntimeError('Counterfactual accounting invalid')
                branch = dict(context_sha256=row['context_sha256'], selected_index=chosen,
                    config=name, applied=forced.applied, values=values, native_fe=measured.native_fe,
                    cached_fe=measured.cached_fe, calls=measured.new_calls,
                    prefix_nfe=prefix, stop_nfe=row['horizon_nfe']['10'])
                save_pickle(branch_path, branch)
            new_fe += branch['native_fe']
            branch_count += 1
            by_index[chosen] = branch
        for name, action in row['actions'].items():
            branch = by_index.get(action['selected_index'])
            for horizon, stop in row['horizon_nfe'].items():
                reference = float(baseline['values'][:, :stop].min())
                value = reference if branch is None else float(branch['values'][:, :stop].min())
                row['horizon_best_values'][horizon][name] = value
                row['horizon_advantages'][horizon][name] = float(np.clip((reference - value) / row['scale'], -20, 20))
        row['advantages'] = row['horizon_advantages']['5']
        cases.append(row)
    with gzip.open(folder / 'CASES.json.gz', 'wt') as stream:
        json.dump(cases, stream, allow_nan=False)
    receipt = dict(status='complete', function=function, function_index=function_index,
        seed_index=seed_index, seed=seed, baseline_states=len(baseline['records']),
        retained_cases=len(cases), conflict_cases=sum(row['fusion_features'][0] == 1 for row in cases),
        distinct_branches=branch_count, native_fe=new_fe, baseline_native_fe=300,
        continuation_native_fe=new_fe - 300, population_sha256=baseline['population_sha256'], binding=freeze())
    if branch_count > 90 or new_fe > 2100:
        raise RuntimeError('Per-episode resource cap exceeded')
    write_json(receipt_path, receipt)
    return receipt


def collect():
    settings = protocol()
    freeze()
    functions = generated().TRAIN_FUNCTIONS
    total_fe = 0
    for seed_index in settings['training_seed_indices']:
        for function_index, function in enumerate(functions):
            folder = EXP / 'collection' / f"s{seed_index:02d}_{function['fid']}"
            receipt = collect_episode(function_index, seed_index, folder)
            total_fe += receipt['native_fe']
            if total_fe > settings['limits']['hard_stop_new_training_objective_evaluations']:
                raise RuntimeError('Collection resource cap exceeded')
            print(json.dumps(dict(phase='collection', **{key: value for key, value in receipt.items() if key != 'binding'}, cumulative_native_fe=total_fe)), flush=True)


def balanced_mean(cases, name, horizon):
    functions = sorted({row['function'] for row in cases})
    function_means = []
    for function in functions:
        seeds = sorted({row['seed_index'] for row in cases if row['function'] == function})
        seed_means = [np.mean([row['horizon_advantages'][str(horizon)][name] for row in cases
            if row['function'] == function and row['seed_index'] == seed]) for seed in seeds]
        function_means.append(float(np.mean(seed_means)))
    return float(np.mean(function_means)) if function_means else None


def calibrate():
    settings, identity = protocol(), freeze()
    target = EXP / 'offline_memory.json'
    if target.exists():
        raise FileExistsError('Memory is frozen; do not overwrite or retune this round')
    cases, receipts = [], []
    for seed_index in settings['training_seed_indices']:
        for function in generated().TRAIN_FUNCTIONS:
            folder = EXP / 'collection' / f"s{seed_index:02d}_{function['fid']}"
            receipt = json.loads((folder / 'COMPLETE.json').read_text())
            if receipt['binding'] != identity:
                raise RuntimeError('Mixed collection versions')
            receipts.append(receipt)
            with gzip.open(folder / 'CASES.json.gz', 'rt') as stream:
                cases.extend(json.load(stream))
    memory_cases = [row for row in cases if row['function'] not in legacy.HELDOUT]
    heldout = [row for row in cases if row['function'] in legacy.HELDOUT]
    contexts = {}
    for key in ('agree_end', 'agree_late', 'conflict_end', 'conflict_late'):
        contexts[key] = {}
        for name in CONFIGS:
            changed = [row for row in heldout if row['context_key'] == key and row['actions'][name]['changes_original']]
            functions = sorted({row['function'] for row in changed})
            seeds = sorted({row['seed_index'] for row in changed})
            episodes = {(row['function'], row['seed_index']) for row in changed}
            gains = {str(horizon): balanced_mean(changed, name, horizon) for horizon in (1, 5, 10)}
            supported = len(changed) >= 8 and len(episodes) >= 6 and len(functions) >= 2 and len(seeds) >= 3
            supported = bool(supported and gains['5'] > 0 and gains['10'] >= 0)
            contexts[key][name] = dict(changed_cases=len(changed), independent_episodes=len(episodes),
                functions=functions, seed_indices=seeds, balanced_horizon_advantages=gains, supported=supported)
    memory = dict(version='r04', primary_horizon_rounds=5, action_duration_rounds=1,
        test_data_used=False, cases=memory_cases, allowed_configs=list(CONFIGS), configs=CONFIGS,
        context_validation=contexts, heldout_functions=sorted(legacy.HELDOUT),
        training_native_fe=sum(row['native_fe'] for row in receipts), collection_binding=identity,
        scope='One-action deviation with original-policy continuation, paired common-random-number prefixes; horizons truncated at budget300',
        anchor_pretrained_on_all36=True)
    write_json(target, memory)
    with gzip.open(EXP / 'CALIBRATION_CASES.json.gz', 'wt') as stream:
        json.dump(heldout, stream, allow_nan=False)
    summary = dict(status='complete', baseline_episodes=len(receipts),
        baseline_states=sum(row['baseline_states'] for row in receipts), retained_cases=len(cases),
        memory_cases=len(memory_cases), calibration_cases=len(heldout),
        conflict_cases=sum(row['fusion_features'][0] == 1 for row in cases),
        distinct_branches=sum(row['distinct_branches'] for row in receipts),
        native_fe=sum(row['native_fe'] for row in receipts),
        supported_context_actions=sum(value['supported'] for context in contexts.values() for value in context.values()),
        contexts=contexts, live_api_calls=0)
    write_json(EXP / 'OFFLINE_VALIDATION.json', summary)
    print(json.dumps(summary), flush=True)


def confirm():
    settings, identity = protocol(), freeze()
    memory_sha = legacy.digest((EXP / 'offline_memory.json').read_bytes())
    rows = []
    for function_index, function in enumerate(generated().TRAIN_FUNCTIONS):
        if function['fid'] not in legacy.HELDOUT:
            continue
        for seed_index in settings['confirmatory_seed_indices']:
            folder = EXP / 'confirmatory' / f"s{seed_index}_{function['fid']}"
            folder.mkdir(parents=True, exist_ok=True)
            receipt_path = folder / 'COMPLETE.json'
            if receipt_path.exists():
                receipt = json.loads(receipt_path.read_text())
                if receipt['binding'] != identity or receipt['memory_sha256'] != memory_sha:
                    raise RuntimeError('Mixed confirmation versions')
                rows.append(receipt)
                continue
            model, problem, population, rng, seed, name = episode(function_index, seed_index)
            baseline = legacy.Counted(problem, stochastic=True)
            torch.set_rng_state(rng)
            with torch.no_grad():
                _, _, nfe, _ = model(population.clone(), baseline)
            reference = torch.cat(baseline.values, 1).numpy()
            if nfe != 300 or baseline.native_fe != 300:
                raise RuntimeError('Invalid confirmatory baseline')
            calls = {index: (dict(shape=row['shape'], x=row['x'], rng=legacy.digest(row['rng_before'])),
                dict(shape=row['shape'][:2], y=row['y'], rng_after=row['rng_after'])) for index, row in enumerate(baseline.new_calls)}
            modified = legacy.build()
            agent = MultiStepFusionAgent(EXP / 'offline_memory.json', EXP / 'no_api_journal', budget=dict(used=0, limit=0))
            records = []
            agent.active_record = records.append
            modified.fusion_agent = agent
            measured = legacy.Counted(problem, calls, stochastic=True)
            torch.set_rng_state(rng)
            with torch.no_grad():
                _, trail, nfe, _ = modified(population.clone(), measured)
            values = torch.cat(measured.values, 1).numpy()
            if nfe != 300 or measured.native_fe + measured.cached_fe != 300 or len(records) != 45:
                raise RuntimeError('Invalid confirmatory NFE/logs')
            if agent.stats['requests'] or not np.array_equal(values.min(1), trail[:, -1].numpy()):
                raise RuntimeError('Unexpected API or best-curve mismatch')
            receipt = dict(function=name, function_index=function_index, seed=seed, seed_index=seed_index,
                baseline_best=float(reference.min()), retrieval_best=float(values.min()),
                improved=bool(values.min() < reference.min()), worsened=bool(values.min() > reference.min()),
                unchanged=bool(values.min() == reference.min()), native_fe=300 + measured.native_fe,
                baseline_native_fe=300, retrieval_native_fe=measured.native_fe,
                retrieval_cached_fe=measured.cached_fe, agent_stats=agent.stats,
                decision_steps_logged=len(records), binding=identity, memory_sha256=memory_sha)
            np.savez_compressed(folder / 'VALUES.npz', baseline=reference, retrieval=values)
            with gzip.open(folder / 'DECISIONS.json.gz', 'wt') as stream:
                json.dump(records, stream, allow_nan=False)
            save_pickle(folder / 'OBJECTIVE_CALLS.pkl.gz', dict(baseline=baseline.new_calls, retrieval=measured.new_calls))
            write_json(receipt_path, receipt)
            rows.append(receipt)
            print(json.dumps(dict(phase='confirmatory', **receipt)), flush=True)
    write_json(EXP / 'CONFIRMATORY_RESULTS.json', rows)
    with (EXP / 'CONFIRMATORY_RESULTS.csv').open('w', newline='') as stream:
        fields = ['function', 'seed_index', 'baseline_best', 'retrieval_best', 'improved', 'unchanged', 'worsened', 'native_fe']
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['collect', 'calibrate', 'confirm'])
    args = parser.parse_args()
    if args.phase == 'collect':
        collect()
    elif args.phase == 'calibrate':
        calibrate()
    else:
        confirm()
