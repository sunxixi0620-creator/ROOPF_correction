import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import argparse
import fcntl
import gzip
import json
import lzma
import multiprocessing
from pathlib import Path
import pickle
import sqlite3
import sys
import tempfile
import time
import traceback
import zlib

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT))
import experiment_r02 as legacy
from roopf.multistep_fusion_agent import MultiStepFusionAgent

EXP = ROOT / 'experiments/r04_full_llm_benchmark'
MEMORY = ROOT / 'experiments/r04_multiseed_horizon/offline_memory.json'
JOURNAL = EXP / 'api_journal'
ARMS = ('off', 'retrieval', 'llm_live')
SOURCES = ('roopf/model.py', 'roopf/fusion_agent.py', 'roopf/fusion_rules.py',
           'roopf/multistep_fusion_agent.py', 'run_roopf.py', 'scripts/experiment_r02.py',
           'scripts/worldcodes_request.ps1', 'scripts/experiment_r04_full.py',
           'checkpoints/anchor_policy_d10.pt', 'checkpoints/residual_selector_generated36_d10.pt',
           'experiments/r02_fusion_agent/baseline_52ec388/release_model.py')


def settings():
    return json.loads((EXP / 'PROTOCOL.json').read_text())


def freeze():
    identity = dict(code={name: legacy.digest((ROOT / name).read_bytes()) for name in SOURCES},
        memory_sha256=legacy.digest(MEMORY.read_bytes()),
        protocol_sha256=legacy.digest((EXP / 'PROTOCOL.json').read_bytes()))
    target = EXP / 'CODE_BINDING.json'
    if target.exists() and json.loads(target.read_text()) != identity:
        raise RuntimeError('Full experiment code, policy, memory, or protocol changed')
    if not target.exists():
        legacy.write_json(target, identity)
    return identity


def save_pickle(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + '.pending')
    with gzip.open(pending, 'wb') as stream:
        pickle.dump(value, stream, protocol=5)
    pending.replace(path)


def full_calls(group, fid):
    folder = legacy.SOURCE / 'docs/revision/zip_replication820_results' / f'{group}_f{fid:02d}'
    batches = {batch: {} for batch in range(10)}
    wanted = {f'b{batch:02d}:{kind}' for batch in range(10) for kind in ('initial', 'full')}
    def decode(blob):
        value = pickle.loads(blob if blob[:1] == b'\x80' else zlib.decompress(blob))
        if not isinstance(value, dict):
            raise RuntimeError('Unexpected decoded payment record')
        return value
    def accept(job, index, request, response):
        if job in wanted:
            if response is None:
                raise RuntimeError('Original payment response missing')
            batches[int(job[1:3])][index] = (decode(request), decode(response))
    source = folder / 'PAYMENTS.sqlite.gz'
    if source.exists():
        with tempfile.TemporaryDirectory(prefix='r04_full_original_readonly_') as temporary:
            restored = Path(temporary) / 'original.sqlite'
            restored.write_bytes(gzip.decompress(source.read_bytes()))
            connection = sqlite3.connect('file:' + str(restored) + '?mode=ro', uri=True)
            placeholders = ','.join('?' for _ in wanted)
            for row in connection.execute(f'select job,idx,request,response from calls where job in ({placeholders})', sorted(wanted)):
                accept(*row)
            connection.close()
    else:
        source = folder / 'PAYMENTS.rows.xz'
        with lzma.open(source, 'rb') as stream:
            if pickle.load(stream)['format'] != 'roopf_sqlite_rows_v825':
                raise RuntimeError('Unknown payment archive')
            while True:
                try:
                    row = pickle.load(stream)
                except EOFError:
                    break
                if row[0] == 'calls':
                    accept(row[1], row[2], row[3], row[4])
    if any(set(calls) != set(range(201)) for calls in batches.values()):
        raise RuntimeError('Original full100 payment transactions incomplete')
    return batches, dict(path=str(source), sha256=legacy.digest(source.read_bytes()))


class FullLiveAgent(MultiStepFusionAgent):
    def call(self, payload):
        JOURNAL.mkdir(parents=True, exist_ok=True)
        with (JOURNAL / 'API.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            self.budget['used'] = len(list(JOURNAL.glob('*.json'))) + len(list((EXP / 'setup_attempt_v1/api_journal').glob('*.json')))
            self.budget['limit'] = settings()['max_new_api_attempts']
            try:
                return super().call(payload)
            except RuntimeError as error:
                if str(error) == '本轮API次数上限':
                    self.resource_exhausted = True
                raise


def merged_calls(original, new_calls):
    result = dict(original)
    for index, row in enumerate(new_calls):
        result[f'new:{index}'] = (dict(shape=row['shape'], x=row['x'], rng=legacy.digest(row['rng_before'])),
            dict(shape=row['shape'][:2], y=row['y'], rng_after=row['rng_after']))
    return result


def run_function(task, batches_to_run=10):
    group, fid = task
    name = f'{group}_f{fid:02d}'
    folder = EXP / 'results' / name
    folder.mkdir(parents=True, exist_ok=True)
    identity = freeze()
    complete = folder / 'COMPLETE.json'
    if complete.exists():
        receipt = json.loads(complete.read_text())
        if receipt['binding'] != identity:
            raise RuntimeError('Completed function binding differs')
        return receipt
    available = next(int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:'))
    if available < 16 * 1024 ** 2:
        raise RuntimeError('Available memory below16GiB; refusing new function')
    original = legacy.build(original=True)
    original_start = torch.get_rng_state().clone()
    models, agents = {}, {}
    for arm in ARMS:
        models[arm] = legacy.build()
        if arm != 'off':
            agents[arm] = (FullLiveAgent if arm == 'llm_live' else MultiStepFusionAgent)(
                MEMORY, JOURNAL if arm == 'llm_live' else EXP / 'retrieval_no_api',
                mode='llm' if arm == 'llm_live' else 'retrieval', budget=dict(used=0, limit=settings()['max_new_api_attempts']))
            models[arm].fusion_agent = agents[arm]
    torch.set_rng_state(original_start)
    if fid > 1:
        torch.set_rng_state(torch.from_numpy(np.frombuffer((legacy.SNAP / f'{group}_f{fid-1:02d}/FULL_RNG.bin').read_bytes(), dtype=np.uint8).copy()))
    problem = legacy.problem_for(group, fid)
    all_calls, source = full_calls(group, fid)
    reference = legacy.original_values(group, fid)
    if reference.shape != (100, 300):
        raise RuntimeError('Original baseline is not full100')
    frozen = {arm: {key: value.clone() for key, value in model.state_dict().items()} for arm, model in models.items()}
    for batch in range(batches_to_run):
        population = problem.genRandomPop((10, 100, 10))
        start = torch.get_rng_state().clone()
        baseline = legacy.Replay(problem, all_calls[batch])
        with torch.no_grad():
            _, _, nfe, _ = original(population.clone(), baseline)
        full_after = torch.get_rng_state().clone()
        baseline_values = torch.cat(baseline.values, 1).numpy()
        expected = reference[batch * 10:(batch + 1) * 10]
        if nfe != 300 or not np.array_equal(baseline_values, expected):
            raise RuntimeError('Full100 original replay differs from frozen results')
        batch_folder = folder / f'b{batch:02d}'
        batch_folder.mkdir(exist_ok=True)
        cache = all_calls[batch]
        for arm in ARMS:
            receipt_path = batch_folder / f'{arm}.json'
            native_path = batch_folder / f'{arm}_NATIVE_CALLS.pkl.gz'
            if receipt_path.exists():
                receipt = json.loads(receipt_path.read_text())
                if receipt['binding'] != identity:
                    raise RuntimeError('Partial batch binding mismatch')
                if native_path.exists():
                    with gzip.open(native_path, 'rb') as stream:
                        cache = merged_calls(cache, pickle.load(stream))
                continue
            model = models[arm]
            records = []
            if arm != 'off':
                agents[arm].active_record = records.append
            measured = legacy.Replay(problem, all_calls[batch]) if arm == 'off' else legacy.Counted(problem, cache, stochastic=(group == 'BBOB' and fid == 20))
            torch.set_rng_state(start)
            began = time.monotonic()
            with torch.no_grad():
                _, trail, nfe, _ = model(population.clone(), measured)
            elapsed = time.monotonic() - began
            if arm != 'off' and getattr(agents[arm], 'resource_exhausted', False):
                raise RuntimeError('Live API cap reached; do not label fallback as a completed full model')
            values = torch.cat(measured.values, 1).numpy()
            if nfe != 300 or values.shape != (10, 300) or not np.isfinite(values).all():
                raise RuntimeError('Invalid result or NFE')
            if not np.array_equal(values.min(1), trail[:, -1].numpy()):
                raise RuntimeError('Best curve mismatch')
            if not all(torch.equal(value, frozen[arm][key]) for key, value in model.state_dict().items()):
                raise RuntimeError('Frozen model weights changed')
            if arm == 'off' and not np.array_equal(values, expected):
                raise RuntimeError('Agent-off code is not equivalent across all100 original trajectories')
            stats = dict(decisions=0, changed=0, changed_points=0, requests=0, failures=0, nonrelease=0, opportunities=0, no_supported_action=0) if arm == 'off' else dict(agents[arm].stats)
            expected_logs = 0 if arm == 'off' or group == 'CEC-style' else 45
            if len(records) != expected_logs:
                raise RuntimeError('Missing semantic decision logs')
            if arm != 'off' and measured.native_fe + measured.cached_fe != 3000:
                raise RuntimeError('Native plus cached evaluation count differs from virtual budget')
            np.savez_compressed(batch_folder / f'{arm}.npz', values=values)
            with gzip.open(batch_folder / f'{arm}_DECISIONS.json.gz', 'wt') as stream:
                json.dump(records, stream, allow_nan=False)
            if measured.new_calls:
                save_pickle(native_path, measured.new_calls)
                cache = merged_calls(cache, measured.new_calls)
            receipt = dict(group=group, fid=fid, batch=batch, arm=arm, n=10, nfe=300,
                mean=float(values.min(1).astype(float).mean()), std=float(values.min(1).astype(float).std()),
                native_fe=measured.native_fe, cached_fe=3000 if arm == 'off' else measured.cached_fe,
                elapsed_seconds=elapsed, agent_stats=stats, decisions_logged=len(records),
                paired_initial_sha256=legacy.digest(population.numpy().tobytes()),
                paired_solver_rng_sha256=legacy.digest(start.numpy().tobytes()),
                baseline_payment_archive=source, binding=identity)
            legacy.write_json(receipt_path, receipt)
            print(json.dumps(dict(event='batch_complete', **{key: value for key, value in receipt.items() if key not in {'binding', 'baseline_payment_archive'}})), flush=True)
        save_pickle(batch_folder / 'PAIRED_START.pkl.gz', dict(population=population.numpy(), rng=start.numpy(), original_after=full_after.numpy()))
        torch.set_rng_state(full_after)
    if batches_to_run == 10:
        if torch.get_rng_state().numpy().tobytes() != (legacy.SNAP / name / 'FULL_RNG.bin').read_bytes():
            raise RuntimeError('Final original full100 RNG differs from frozen baseline')
        for arm in ARMS:
            values = np.concatenate([np.load(folder / f'b{batch:02d}/{arm}.npz')['values'] for batch in range(10)])
            if values.shape != (100, 300):
                raise RuntimeError('Function missing complete100 trajectories')
            np.savez_compressed(folder / f'{arm}_FULL100.npz', values=values)
        receipt = dict(status='complete', group=group, fid=fid, n=100, nfe=300,
            baseline_all100_bitwise_verified=True, original_final_rng_bitwise_verified=True, binding=identity)
        legacy.write_json(complete, receipt)
        return receipt
    return dict(status='health_batch_complete', group=group, fid=fid, binding=identity)


def worker(task):
    try:
        return run_function(task)
    except Exception:
        folder = EXP / 'results' / f'{task[0]}_f{task[1]:02d}'
        legacy.write_json(folder / 'ERROR.json', dict(status='failed', traceback=traceback.format_exc()))
        raise


def health():
    freeze()
    attempted_tasks = []
    for task in [('BBOB', 17), ('BBOB', 19), ('UAV', 5)]:
        run_function(task, batches_to_run=1)
        attempted_tasks.append(list(task))
        receipts = [json.loads(path.read_text()) for path in JOURNAL.glob('*.json')]
        valid = [row for row in receipts if row['status'] == 'ok']
        if valid:
            break
    record = dict(status='passed' if valid else 'failed', real_api_attempts=len(receipts),
        valid_api_results=len(valid), model_requested='deepseek-v4-flash',
        provider_reported_models=sorted({str(row.get('response', {}).get('model')) for row in receipts}),
        unknown_or_failed=len(receipts) - len(valid), tasks=attempted_tasks, batch=0,
        retained_as_part_of_full100=True)
    legacy.write_json(EXP / 'API_HEALTH.json', record)
    print(json.dumps(record), flush=True)
    if not valid:
        raise RuntimeError('No valid real API result; stop before bulk full-model evaluation')


def run_all():
    freeze()
    health()
    order = [('BBOB', 17)] + [task for task in legacy.TASKS if task != ('BBOB', 17)]
    with multiprocessing.get_context('spawn').Pool(processes=settings()['workers']) as pool:
        for receipt in pool.imap_unordered(worker, order, chunksize=1):
            print(json.dumps(dict(event='function_complete', **{key: value for key, value in receipt.items() if key != 'binding'})), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['health', 'run'])
    args = parser.parse_args()
    if args.phase == 'health':
        health()
    else:
        run_all()
