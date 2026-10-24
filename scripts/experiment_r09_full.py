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
import pickle
import sys
import time
import traceback

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT))
import experiment_r04_full as storage
from roopf.evidence_fusion_agent import EvidenceFusionAgent

legacy = storage.legacy
EXP = ROOT / 'experiments/r09_evidence_fusion'
MEMORY = EXP / 'offline_memory.json'
SOURCES = tuple(storage.SOURCES) + ('roopf/anchor_backbone.py', 'roopf/joint_generation_agent.py',
    'scripts/experiment_r07_full.py', 'scripts/collect_r07.py', 'scripts/report_r07.py',
    'scripts/experiment_r04.py', 'experiments/r02_fusion_agent/training_assets/generated36_curated.py',
    'roopf/matched38_agent.py', 'roopf/evidence_fusion_agent.py', 'roopf/policy_api.py', 'scripts/worldcodes_request_r09.ps1', 'scripts/collect_r09.py', 'scripts/experiment_r09_full.py',
    'scripts/run_r09_hardware.py', 'scripts/run_r09_hardware.sh', 'scripts/report_r09.py',
    'scripts/audit_r09_records.py')


def settings():
    return json.loads((EXP / 'PROTOCOL.json').read_text())


def binding(include_memory=True):
    identity = dict(code={name: legacy.digest((ROOT / name).read_bytes()) for name in SOURCES},
        protocol_sha256=legacy.digest((EXP / 'PROTOCOL.json').read_bytes()))
    if include_memory:
        identity['memory_sha256'] = legacy.digest(MEMORY.read_bytes())
    return identity


def freeze():
    identity = binding()
    path = EXP / 'CODE_BINDING.json'
    if path.exists() and json.loads(path.read_text()) != identity:
        raise RuntimeError('Frozen full run changed; preserve previous attempt')
    if not path.exists():
        legacy.write_json(path, identity)
    return identity


def agent_for(arm):
    return EvidenceFusionAgent(MEMORY, EXP / ('api_journal' if arm == 'llm_live' else 'no_api'),
        mode='llm' if arm == 'llm_live' else 'retrieval',
        budget=dict(used=0, limit=settings()['max_new_api_attempts']), api_config=settings()['api'],
        arm=arm if arm in ('anchor', 'local', 'matched38') else 'joint')


def run_function(task):
    group, fid = task
    name = f'{group}_f{fid:02d}'
    folder = EXP / 'results' / name
    folder.mkdir(parents=True, exist_ok=True)
    identity = freeze()
    complete = folder / 'COMPLETE.json'
    if complete.exists():
        receipt = json.loads(complete.read_text())
        if receipt['binding'] != identity:
            raise RuntimeError('Completed function binding mismatch')
        return receipt
    arms = settings()['arms']
    original = legacy.build(original=True)
    original_start = torch.get_rng_state().clone()
    models, agents = {}, {}
    for arm in arms:
        models[arm] = legacy.build()
        if arm != 'off':
            agents[arm] = agent_for(arm)
            models[arm].fusion_agent = agents[arm]
    torch.set_rng_state(original_start)
    if fid > 1:
        torch.set_rng_state(torch.from_numpy(np.frombuffer(
            (legacy.SNAP / f'{group}_f{fid-1:02d}/FULL_RNG.bin').read_bytes(), dtype=np.uint8).copy()))
    problem = legacy.problem_for(group, fid)
    all_calls, source = storage.full_calls(group, fid)
    reference = legacy.original_values(group, fid)
    for batch in range(10):
        population = problem.genRandomPop((10, 100, 10))
        start = torch.get_rng_state().clone()
        baseline = legacy.Replay(problem, all_calls[batch])
        with torch.no_grad():
            original(population.clone(), baseline)
        full_after = torch.get_rng_state().clone()
        expected = reference[10 * batch:10 * (batch + 1)]
        if not np.array_equal(torch.cat(baseline.values, 1).numpy(), expected):
            raise RuntimeError('Frozen baseline replay mismatch')
        batch_folder = folder / f'b{batch:02d}'
        batch_folder.mkdir(exist_ok=True)
        cache = all_calls[batch]
        for arm in arms:
            target = batch_folder / f'{arm}.json'
            native_path = batch_folder / f'{arm}_NATIVE_CALLS.pkl.gz'
            if target.exists():
                if json.loads(target.read_text())['binding'] != identity:
                    raise RuntimeError('Partial run binding mismatch')
                if native_path.exists():
                    with gzip.open(native_path, 'rb') as stream:
                        cache = storage.merged_calls(cache, pickle.load(stream))
                continue
            records = []
            if arm != 'off':
                agents[arm].execution_context = dict(group=group, fid=fid, batch=batch, arm=arm,
                    paired_initial_sha256=legacy.digest(population.numpy().tobytes()),
                    paired_solver_rng_sha256=legacy.digest(start.numpy().tobytes()))
                agents[arm].active_record = records.append
            measured = legacy.Replay(problem, all_calls[batch]) if arm == 'off' else legacy.Counted(
                problem, cache, stochastic=(group == 'BBOB' and fid == 20))
            torch.set_rng_state(start)
            began = time.monotonic()
            with torch.no_grad():
                _, trail, nfe, _ = models[arm](population.clone(), measured)
            values = torch.cat(measured.values, 1).numpy()
            if nfe != 300 or values.shape != (10, 300) or not np.isfinite(values).all():
                raise RuntimeError('Incomplete or invalid trajectory')
            if not np.array_equal(values.min(1), trail[:, -1].numpy()):
                raise RuntimeError('Best trace differs from objective record')
            if arm == 'off' and not np.array_equal(values, expected):
                raise RuntimeError('Original-equivalent arm changed')
            if arm != 'off' and agents[arm].resource_exhausted:
                raise RuntimeError('API cap reached; full live run is incomplete')
            if arm != 'off' and agents[arm].uncertain_api_outcome:
                raise RuntimeError('Unknown prior API outcome; preserve journal and resolve before continuing')
            expected_logs = 0 if arm == 'off' or group == 'CEC-style' else 45
            if len(records) != expected_logs:
                raise RuntimeError('Missing active-phase decisions')
            np.savez_compressed(batch_folder / f'{arm}.npz', values=values)
            with gzip.open(batch_folder / f'{arm}_DECISIONS.json.gz', 'wt') as stream:
                json.dump(records, stream, allow_nan=False)
            if measured.new_calls:
                storage.save_pickle(native_path, measured.new_calls)
                cache = storage.merged_calls(cache, measured.new_calls)
            receipt = dict(group=group, fid=fid, batch=batch, arm=arm, n=10, nfe=300,
                native_fe=measured.native_fe, cached_fe=3000 if arm == 'off' else measured.cached_fe,
                elapsed_seconds=time.monotonic() - began,
                agent_stats={} if arm == 'off' else dict(agents[arm].stats),
                paired_initial_sha256=legacy.digest(population.numpy().tobytes()),
                paired_solver_rng_sha256=legacy.digest(start.numpy().tobytes()),
                baseline_payment_archive=source, binding=identity)
            legacy.write_json(target, receipt)
            print(json.dumps(dict(event='batch_complete', group=group, fid=fid, batch=batch, arm=arm)), flush=True)
        storage.save_pickle(batch_folder / 'PAIRED_START.pkl.gz', dict(population=population.numpy(),
            rng=start.numpy(), original_after=full_after.numpy()))
        torch.set_rng_state(full_after)
    if torch.get_rng_state().numpy().tobytes() != (legacy.SNAP / name / 'FULL_RNG.bin').read_bytes():
        raise RuntimeError('Original final random state mismatch')
    for arm in arms:
        values = np.concatenate([np.load(folder / f'b{batch:02d}/{arm}.npz')['values'] for batch in range(10)])
        np.savez_compressed(folder / f'{arm}_FULL100.npz', values=values)
    receipt = dict(status='complete', group=group, fid=fid, n=100, nfe=300, binding=identity)
    legacy.write_json(complete, receipt)
    return receipt


def worker(task):
    try:
        return run_function(task)
    except Exception:
        legacy.write_json(EXP / 'results' / f'{task[0]}_f{task[1]:02d}' / 'ERROR.json',
            dict(status='failed', traceback=traceback.format_exc()))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['run'])
    parser.parse_args()
    freeze()
    with multiprocessing.get_context('spawn').Pool(settings()['workers']) as pool:
        for receipt in pool.imap_unordered(worker, legacy.TASKS, chunksize=1):
            print(json.dumps(dict(event='function_complete', group=receipt['group'], fid=receipt['fid'])), flush=True)
