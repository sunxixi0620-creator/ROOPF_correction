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
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT))
import experiment_r02 as legacy
from roopf.fusion_agent import CONFIGS, FusionAgent, context_rows, fusion_indices

EXP = ROOT / 'experiments/r03_evidence_fusion'


class Collector:
    def __init__(self, function, records):
        self.function = function
        self.records = records

    def reset(self, batch):
        pass

    def observe(self, *args):
        pass

    def choose(self, **kwargs):
        scale = kwargs['archive_y'].std(1, keepdim=True, unbiased=False).clamp_min(1e-8)
        choices = fusion_indices(kwargs['score'], kwargs['probability'], scale,
            kwargs['original_idx'], kwargs['residual_weight'], kwargs['selection_rule'])
        rows = context_rows(kwargs['state'], kwargs['score'], kwargs['mu'], kwargs['sigma'],
            kwargs['probability'], scale, kwargs['original_idx'], kwargs['residual_weight'],
            fitness=kwargs['fitness'], candidate_ops=kwargs['candidate_ops'], options=choices)
        labels = kwargs['problem'].calfitness(kwargs['candidates']).reshape(kwargs['candidates'].shape[:2])
        reference = torch.gather(labels, 1, kwargs['original_idx'])
        for index, row in enumerate(rows):
            row.update(id=f'{self.function}:{len(self.records)}', function=self.function,
                advantages={name: float(((reference[index, 0] - labels[index, selected[index, 0]]) /
                    scale[index, 0]).clamp(-20, 20)) for name, selected in choices.items()})
            self.records.append(row)
        return kwargs['original_idx']


def build_memory():
    target = EXP / 'offline_memory.json'
    if target.exists():
        raise FileExistsError('R03 memory already exists; do not overwrite frozen evidence')
    generated = legacy.load_module('r03_generated36', legacy.EXP / 'training_assets/generated36_curated.py')
    records, native = [], 0
    for index, function in enumerate(generated.TRAIN_FUNCTIONS):
        model = legacy.build()
        model.fusion_agent = Collector(function['fid'], records)
        legacy.release.set_seed(2026102200 + index)
        function = dict(function)
        generated.gen_train_offset(10, function)
        problem = legacy.release.BBOBProblem(dict(fid='generated36:' + function['fid'], xlb=-5, xub=5), 10)
        problem.calfitness = lambda candidates, function=function: generated.get_train_fitness(candidates, function)
        measured = legacy.Counted(problem)
        with torch.no_grad():
            model(problem.genRandomPop((1, 100, 10)), measured)
        native += measured.native_fe
        print(json.dumps(dict(phase='memory', functions=index + 1, cases=len(records), training_fe=native)), flush=True)
    training = [row for row in records if row['function'] not in legacy.HELDOUT]
    heldout = [row for row in records if row['function'] in legacy.HELDOUT]
    validation = {}
    for key in sorted({row['context_key'] for row in records}):
        validation[key] = {}
        for name in CONFIGS:
            changed = [row for row in heldout if row['context_key'] == key and row['actions'][name]['changes_original']]
            functions = sorted({row['function'] for row in changed})
            means = [float(np.mean([row['advantages'][name] for row in changed if row['function'] == function])) for function in functions]
            mean = float(np.mean(means)) if means else None
            validation[key][name] = dict(changed_cases=len(changed), functions=functions,
                function_balanced_mean_advantage=mean,
                supported=bool(len(changed) >= 3 and len(functions) >= 2 and mean > 0))
    memory = dict(version='r03', test_data_used=False, cases=training,
        allowed_configs=list(CONFIGS), configs=CONFIGS, context_validation=validation,
        heldout_functions=sorted(legacy.HELDOUT), training_native_fe=native,
        scope='One-step training-function calibration only; anchor was pretrained on all36; no terminal guarantee',
        support_rule='At least 3 changed heldout cases from 2 functions, positive function-balanced mean; positive retrieved local mean at runtime',
        source_r02_commit='01a9e67f8f7065ca557a5df8a28dcd17862c3966',
        seed_rule='2026102200 + fixed training function index',
        generator_sha256=legacy.digest((legacy.EXP / 'training_assets/generated36_curated.py').read_bytes()))
    legacy.write_json(target, memory)
    with gzip.open(EXP / 'HELDOUT_CASES.json.gz', 'wt') as stream:
        json.dump(heldout, stream)
    legacy.write_json(EXP / 'OFFLINE_VALIDATION.json', dict(contexts=validation,
        training_cases=len(training), heldout_cases=len(heldout), training_native_fe=native,
        supported_context_actions=sum(value['supported'] for context in validation.values() for value in context.values()),
        api_calls=0, benchmark_test_fe=0))


def evaluate(limit, modes, api_limit):
    if 'llm_live' in modes and api_limit <= 0:
        raise ValueError('Live mode requires an explicit positive --api-limit; no implicit paid API calls')
    budget = dict(used=0, limit=api_limit)
    binding = {name: legacy.digest((ROOT / name).read_bytes()) for name in
        ('roopf/model.py', 'roopf/fusion_agent.py', 'roopf/fusion_rules.py', 'scripts/experiment_r03.py')}
    identity = dict(code=binding, memory=legacy.digest((EXP / 'offline_memory.json').read_bytes()))
    protocol_path = EXP / 'EVALUATION_BINDING.json'
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != identity:
        raise RuntimeError('Code/memory changed; use a new experiment, never mix result versions')
    legacy.write_json(protocol_path, identity)
    summary = []
    for group, fid in legacy.TASKS[:limit]:
        folder = EXP / 'results' / f'{group}_f{fid:02d}'
        folder.mkdir(parents=True, exist_ok=True)
        model = legacy.build(original=True)
        if fid > 1:
            torch.set_rng_state(torch.from_numpy(np.frombuffer((legacy.SNAP / f'{group}_f{fid-1:02d}/FULL_RNG.bin').read_bytes(), dtype=np.uint8).copy()))
        problem = legacy.problem_for(group, fid)
        population = problem.genRandomPop((10, 100, 10))
        rng = torch.get_rng_state().clone()
        calls = legacy.archived_calls(group, fid)
        baseline = legacy.original_values(group, fid)[:10]
        for mode in modes:
            receipt_path = folder / f'{mode}.json'
            if receipt_path.exists():
                summary.append(json.loads(receipt_path.read_text()))
                continue
            modified = legacy.build()
            records = []
            agent = FusionAgent(EXP / 'offline_memory.json', EXP / 'api_journal',
                mode='llm' if mode == 'llm_live' else mode, budget=budget)
            if mode != 'off':
                agent.active_record = records.append
                modified.fusion_agent = agent
            frozen = {name: value.clone() for name, value in modified.state_dict().items()}
            measured = legacy.Replay(problem, calls) if mode == 'off' else legacy.Counted(problem, calls, stochastic=(group == 'BBOB' and fid == 20))
            torch.set_rng_state(rng)
            started = time.monotonic()
            with torch.no_grad():
                _, trail, nfe, _ = modified(population.clone(), measured)
            values = torch.cat(measured.values, 1).numpy()
            if not all(torch.equal(value, frozen[name]) for name, value in modified.state_dict().items()):
                raise RuntimeError('Frozen weights changed')
            if nfe != 300 or values.shape != (10, 300) or not np.isfinite(values).all():
                raise RuntimeError('Invalid NFE or output')
            if not np.array_equal(values.min(1), trail[:, -1].numpy()):
                raise RuntimeError('Best-curve mismatch')
            if mode == 'off' and not np.array_equal(values, baseline):
                raise RuntimeError('Original equivalence failed')
            expected_records = 0 if mode == 'off' or group == 'CEC-style' else agent.stats['decisions'] // 10
            if len(records) != expected_records or (mode != 'off' and group != 'CEC-style' and not records):
                raise RuntimeError('Missing semantic decision logs')
            if mode != 'off' and measured.native_fe + measured.cached_fe != 3000:
                raise RuntimeError('Evaluation accounting mismatch')
            np.savez_compressed(folder / f'{mode}.npz', values=values)
            with gzip.open(folder / f'{mode}_decisions.json.gz', 'wt') as stream:
                json.dump(records, stream)
            if measured.new_calls:
                with gzip.open(folder / f'{mode}_NATIVE_CALLS.pkl.gz', 'wb') as stream:
                    pickle.dump(measured.new_calls, stream, protocol=5)
            best, reference = values.min(1).astype(float), baseline.min(1).astype(float)
            receipt = dict(group=group, fid=fid, mode=mode, mean=float(best.mean()), std=float(best.std()),
                baseline_mean=float(reference.mean()), baseline_std=float(reference.std()),
                trajectory_wins=int((best < reference).sum()), trajectory_ties=int((best == reference).sum()),
                trajectory_losses=int((best > reference).sum()), n=10, nfe=300,
                primary=not (group == 'BBOB' and fid == 20), native_fe=measured.native_fe,
                cached_fe=3000 if mode == 'off' else measured.cached_fe,
                decisions_logged=len(records), agent_stats=agent.stats,
                elapsed_seconds=time.monotonic() - started,
                paired_initial_sha256=legacy.digest(population.numpy().tobytes()), **identity)
            legacy.write_json(receipt_path, receipt)
            summary.append(receipt)
            print(json.dumps(receipt), flush=True)
    legacy.write_json(EXP / 'EVALUATION_SUMMARY.json', summary)
    with (EXP / 'FUNCTION_COMPARISON.csv').open('w', newline='') as stream:
        fields = ['group', 'fid', 'mode', 'primary', 'mean', 'std', 'baseline_mean', 'baseline_std',
                  'trajectory_wins', 'trajectory_ties', 'trajectory_losses', 'native_fe', 'decisions_logged']
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(summary)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['memory', 'evaluate'])
    parser.add_argument('--limit', type=int, default=35)
    parser.add_argument('--modes', default='off,retrieval')
    parser.add_argument('--api-limit', type=int, default=0)
    args = parser.parse_args()
    if args.phase == 'memory':
        build_memory()
    else:
        modes = args.modes.split(',')
        if not modes or any(mode not in {'off', 'retrieval', 'llm_live'} for mode in modes):
            parser.error('Invalid modes')
        evaluate(args.limit, modes, args.api_limit)
