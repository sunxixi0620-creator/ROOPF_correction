import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import argparse
import csv
import gzip
import hashlib
import importlib.util
import io
import json
import lzma
from pathlib import Path
import pickle
import sqlite3
import sys
import tempfile
import time
from types import SimpleNamespace
import zlib

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import run_roopf as release
from roopf.fusion_agent import CONFIGS, FusionAgent, context_rows, fusion_indices

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
EXP = ROOT / 'experiments/r02_fusion_agent'
OUT = EXP / 'results'
SNAP = EXP / 'baseline_52ec388/docs/revision/zip_replication820_results'
SOURCE = Path('/home/skq/repos/ROOPF_improve/ROOPF')
TASKS = [(group, fid) for group, count in [('BBOB', 24), ('CEC-style', 6), ('UAV', 5)] for fid in range(1, count + 1)]
HELDOUT = {'comp_global', 'comp_rugged', 'dirridge_noisy', 'lowrank_tanh', 'plateau_sharp', 'ring_single', 'wavepacket_local'}


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + '.pending')
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    pending.replace(path)


def build(original=False):
    release.set_seed(20260630)
    if original:
        module = load_module('roopf.release_r02_original', EXP / 'baseline_52ec388/release_model.py')
        return module.ROOPFOptimizer(dim=10, hidden_dim=200, popSize=100, max_nfe=300,
                    k_nums=2, pool_per_op=6, surrogate_members=5, ablation='roopf',
                    baseline_ckpt=str(ROOT / 'checkpoints/anchor_policy_d10.pt'),
                    router_ckpt=str(ROOT / 'checkpoints/residual_selector_generated36_d10.pt'), router_weight=.008).eval()
    return release.build_optimizer(SimpleNamespace(dimension=10, population_size=100, budget=300, agentic_mode='off'))


class Counted:
    def __init__(self, problem, calls=None, stochastic=False):
        self.problem = problem
        self.fun = problem.fun
        self.values = []
        self.native_fe = 0
        self.cached_fe = 0
        self.cache = {}
        self.new_calls = []
        self.stochastic = stochastic
        for request, response in (calls or {}).values():
            identity = (request['shape'], request['x'], request['rng'] if stochastic else None)
            self.cache[identity] = response

    def __getattr__(self, name):
        return getattr(self.problem, name)

    def calfitness(self, candidates):
        raw = candidates.detach().contiguous().numpy().tobytes()
        before = torch.get_rng_state().numpy().tobytes()
        identity = (tuple(candidates.shape), raw, digest(before) if self.stochastic else None)
        if identity in self.cache:
            response = self.cache[identity]
            if response['rng_after'] is not None:
                torch.set_rng_state(torch.from_numpy(np.frombuffer(response['rng_after'], dtype=np.uint8).copy()))
            values = torch.from_numpy(np.frombuffer(response['y'], dtype=np.float32).copy()).reshape(response['shape'])
            self.values.append(values)
            self.cached_fe += candidates.numel() // 10
            return values
        values = self.problem.calfitness(candidates)
        self.native_fe += candidates.numel() // 10
        self.values.append(values.detach().reshape(candidates.shape[:2]).cpu())
        self.new_calls.append(dict(shape=tuple(candidates.shape), x=raw, y=values.detach().contiguous().numpy().tobytes(), rng_before=before, rng_after=torch.get_rng_state().numpy().tobytes()))
        return values


class Collector:
    def __init__(self, function_name, records):
        self.function_name = function_name
        self.records = records

    def reset(self, batch):
        pass

    def observe(self, *args):
        pass

    def choose(self, **kwargs):
        scale = kwargs['archive_y'].std(1, keepdim=True, unbiased=False).clamp_min(1e-8)
        choices = fusion_indices(kwargs['score'], kwargs['probability'], scale, kwargs['original_idx'], kwargs['residual_weight'])
        rows = context_rows(kwargs['state'], kwargs['score'], kwargs['mu'], kwargs['sigma'], kwargs['probability'], scale, kwargs['original_idx'], kwargs['residual_weight'])
        labels = kwargs['problem'].calfitness(kwargs['candidates']).reshape(kwargs['candidates'].shape[:2])
        reference = torch.gather(labels, 1, kwargs['original_idx'])
        for index, row in enumerate(rows):
            row.update(id=f'{self.function_name}:{len(self.records)}', function=self.function_name,
                       advantages={name: float(((reference[index, 0] - labels[index, selected[index, 0]]) / scale[index, 0]).clamp(-20, 20)) for name, selected in choices.items()},
                       candidate_targets=labels[index].tolist())
            self.records.append(row)
        return kwargs['original_idx']


def train_memory():
    path = EXP / 'offline_memory.json'
    if path.exists():
        print('OFFLINE_MEMORY_EXISTS', digest(path.read_bytes()), flush=True)
        return
    generated = load_module('r02_generated36', EXP / 'training_assets/generated36_curated.py')
    all_cases = []
    native = 0
    pool_rows = []
    started = time.monotonic()
    for index, function in enumerate(generated.TRAIN_FUNCTIONS):
        model = build()
        model.fusion_agent = Collector(function['fid'], all_cases)
        release.set_seed(2026102200 + index)
        function = dict(function)
        generated.gen_train_offset(10, function)
        problem = release.BBOBProblem(dict(fid='generated36:' + function['fid'], xlb=-5, xub=5), 10)
        problem.calfitness = lambda candidates, function=function: generated.get_train_fitness(candidates, function)
        counted = Counted(problem)
        original_logger = model._record_pool_candidate_samples
        def late_logger(**kwargs):
            if kwargs['remaining'] <= .30:
                original_logger(**kwargs)
        model._record_pool_candidate_samples = late_logger
        model.log_pool_candidates = True
        with torch.no_grad():
            model(problem.genRandomPop((1, 100, 10)), counted)
        for row in model.pool_candidate_samples:
            row['function'] = function['fid']
        pool_rows.extend(model.pool_candidate_samples)
        native += counted.native_fe
        print(json.dumps(dict(phase='offline', function=index + 1, cases=len(all_cases), native_fe=native)), flush=True)
    validation = {}
    validation_cases = [case for case in all_cases if case['function'] in HELDOUT]
    for name in CONFIGS:
        validation[name] = float(np.mean([case['advantages'][name] for case in validation_cases]))
    allowed = [name for name in CONFIGS if name == 'release' or validation[name] >= 0]
    memory = dict(version='r02', test_data_used=False, validation_scope='memory only; frozen anchor trained on all36',
                  configs=CONFIGS, allowed_configs=allowed, validation_mean_one_step_advantage=validation,
                  cases=[{key: value for key, value in case.items() if key != 'candidate_targets'} for case in all_cases if case['function'] not in HELDOUT],
                  heldout_functions=sorted(HELDOUT), training_native_fe=native,
                  generator_sha256=digest((EXP / 'training_assets/generated36_curated.py').read_bytes()),
                  elapsed_seconds=time.monotonic() - started)
    with gzip.open(EXP / 'OFFLINE_FULL_POOL.jsonl.gz', 'wt') as stream:
        for row in pool_rows:
            stream.write(json.dumps(row) + '\n')
    write_json(EXP / 'OFFLINE_VALIDATION.json', dict(validation=validation, allowed_configs=allowed, heldout_cases=len(validation_cases), full_pool_rows=len(pool_rows), one_step_only=True))
    write_json(path, memory)
    print('OFFLINE_MEMORY_COMPLETE', allowed, flush=True)


def problem_for(group, fid):
    if group == 'BBOB':
        function = dict(release.BBOB_FUNCTIONS[fid], xlb=-5, xub=5)
        release.setOffset(function, release.load_bbob_offsets(ROOT / 'data/bbob_offsets_d10.pkl')[fid])
        return release.BBOBProblem(function, 10)
    if group == 'CEC-style':
        function = dict(release.CEC_FUNCTIONS[f'cecf{fid}'])
        function.update(xlb=function.get('xlb', -100), xub=function.get('xub', 100))
        return release.CECStyleProblem(function, 10)
    module = load_module('r02_uav', EXP / 'training_assets/uav_benchmark.py')
    problem = release.BBOBProblem(dict(fid=f'uav{fid}', xlb=-5, xub=5), 10)
    problem.calfitness = lambda candidates: module.uav_objective(candidates, module.SCENARIOS[fid - 1])
    return problem


def archived_calls(group, fid):
    folder = SOURCE / 'docs/revision/zip_replication820_results' / f'{group}_f{fid:02d}'
    compressed = folder / 'PAYMENTS.sqlite.gz'
    calls = {}
    if compressed.exists():
        with tempfile.TemporaryDirectory(prefix='roopf_r02_replay_') as directory:
            restored = Path(directory) / 'read_only_archive.sqlite'
            restored.write_bytes(gzip.decompress(compressed.read_bytes()))
            connection = sqlite3.connect('file:' + str(restored) + '?mode=ro', uri=True)
            for job, index, request, response in connection.execute('select job,idx,request,response from calls where job in ("b00:full","b00:initial")'):
                calls[index] = (pickle.loads(zlib.decompress(request)), pickle.loads(zlib.decompress(response)))
            connection.close()
    else:
        with lzma.open(folder / 'PAYMENTS.rows.xz', 'rb') as stream:
            assert pickle.load(stream)['format'] == 'roopf_sqlite_rows_v825'
            while True:
                try:
                    row = pickle.load(stream)
                except EOFError:
                    break
                if row[0] == 'calls' and row[1] in ('b00:full', 'b00:initial'):
                    calls[row[2]] = (pickle.loads(row[3]), pickle.loads(row[4]))
                if row[0] == 'calls' and row[1].startswith('b01:'):
                    break
    assert len(calls) == 201
    return calls


class Replay(Counted):
    def __init__(self, problem, calls):
        super().__init__(problem)
        self.calls = calls
        self.index = 0

    def calfitness(self, candidates):
        request, response = self.calls[self.index]
        self.index += 1
        assert tuple(candidates.shape) == request['shape']
        assert candidates.detach().contiguous().numpy().tobytes() == request['x'], f'原版回放候选不一致:{self.index}'
        assert digest(torch.get_rng_state().numpy().tobytes()) == request['rng'], f'原版随机流不一致:{self.index}'
        if response['rng_after'] is not None:
            torch.set_rng_state(torch.from_numpy(np.frombuffer(response['rng_after'], dtype=np.uint8).copy()))
        values = torch.from_numpy(np.frombuffer(response['y'], dtype=np.float32).copy()).reshape(response['shape'])
        self.values.append(values)
        return values


def original_values(group, fid):
    data = pickle.loads(zlib.decompress((SNAP / f'{group}_f{fid:02d}/VALUES.bin').read_bytes()))
    shape, raw = data['full']
    return np.frombuffer(raw, dtype=np.float32).reshape(shape).copy()


def run_task(task, budget, modes):
    available = next(int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:'))
    assert available >= 16 * 1024 ** 2, '可用内存不足16GiB，停止新任务'
    group, fid = task
    folder = OUT / f'{group}_f{fid:02d}'
    if (folder / 'llm_live.json').exists() and (folder / 'COMPLETE.json').exists():
        print('SKIP_COMPLETE', folder.name, flush=True)
        return
    folder.mkdir(parents=True, exist_ok=True)
    model = build(original=True)
    if fid > 1:
        torch.set_rng_state(torch.from_numpy(np.frombuffer((SNAP / f'{group}_f{fid-1:02d}/FULL_RNG.bin').read_bytes(), dtype=np.uint8).copy()))
    problem = problem_for(group, fid)
    population = problem.genRandomPop((10, 100, 10))
    rng = torch.get_rng_state().clone()
    calls = archived_calls(group, fid)
    replay = Replay(problem, calls)
    with torch.no_grad():
        _, baseline_trail, baseline_nfe, _ = model(population.clone(), replay)
    baseline = torch.cat(replay.values, 1).numpy()
    expected = original_values(group, fid)
    assert baseline_nfe == 300 and np.array_equal(baseline, expected[:10])
    arrays = {'baseline': baseline}
    rows = []
    for mode in modes:
        arm_file = folder / f'{mode}.npz'
        receipt = folder / f'{mode}.json'
        if arm_file.exists() and receipt.exists():
            arrays[mode] = np.load(arm_file)['values']
            rows.append(json.loads(receipt.read_text()))
            continue
        modified = build()
        frozen = {name: value.clone() for name, value in modified.state_dict().items()}
        records = []
        agent = FusionAgent(EXP / 'offline_memory.json', OUT / 'api_journal' / 'json_contract_v4', mode='llm' if mode == 'llm_live' else mode, budget=budget)
        if mode != 'off':
            agent.active_record = records.append
            modified.fusion_agent = agent
        torch.set_rng_state(rng)
        measured = Replay(problem, calls) if mode == 'off' else Counted(problem, calls=calls if mode == 'llm_live' else None, stochastic=(group == 'BBOB' and fid == 20))
        started = time.monotonic()
        with torch.no_grad():
            _, trail, nfe, _ = modified(population.clone(), measured)
        values = torch.cat(measured.values, 1).numpy()
        assert all(torch.equal(value, frozen[name]) for name, value in modified.state_dict().items()), '冻结权重发生变化'
        assert nfe == 300 and values.shape == (10, 300)
        assert measured.native_fe == 0 if mode == 'off' else measured.native_fe + measured.cached_fe == 3000
        assert np.array_equal(values.min(1), trail[:, -1].numpy())
        if mode == 'off':
            assert np.array_equal(values, baseline), '关闭Agent时不等价于压缩包'
        arrays[mode] = values
        np.savez_compressed(arm_file, values=values)
        if measured.new_calls:
            with gzip.open(folder / f'{mode}_NATIVE_CALLS.pkl.gz', 'wb') as stream:
                pickle.dump(measured.new_calls, stream, protocol=5)
        row = dict(group=group, fid=fid, mode=mode, n=10, nfe=300, native_fe=measured.native_fe,
                   mean=float(values.min(1).astype(float).mean()), std=float(values.min(1).astype(float).std()),
                   cached_fe=measured.cached_fe, elapsed_seconds=time.monotonic() - started, agent_stats=agent.stats,
                   paired_initial_sha256=digest(population.numpy().tobytes()),
                   memory_sha256=agent.memory_hash)
        write_json(receipt, row)
        write_json(folder / f'{mode}_decisions.json', records)
        rows.append(row)
        print(json.dumps(dict(phase='evaluation', **row)), flush=True)
    np.savez_compressed(folder / 'archived_baseline.npz', values=baseline)
    live_stats = json.loads((folder / 'llm_live.json').read_text())['agent_stats'] if (folder / 'llm_live.json').exists() else None
    valid_llm = live_stats is not None and (live_stats['decisions'] == 0 or live_stats['failures'] < 2)
    marker = 'COMPLETE.json' if valid_llm and all((folder / (mode + '.json')).exists() for mode in ['off', 'retrieval', 'llm_live']) else 'LOCAL_CHECKS_COMPLETE.json'
    write_json(folder / marker, dict(status='complete' if marker == 'COMPLETE.json' else 'api_pending_approval', baseline_replayed_bitwise=True, rows=rows, stochastic_reference=(group == 'BBOB' and fid == 20)))


def summarize():
    rows = []
    trajectories = []
    missing = []
    for group, fid in TASKS:
        folder = OUT / f'{group}_f{fid:02d}'
        if not (folder / 'COMPLETE.json').exists():
            missing.append(folder.name)
        if not (folder / 'archived_baseline.npz').exists():
            continue
        baseline_all = original_values(group, fid).min(1).astype(float)
        baseline = baseline_all[:10]
        for mode in ['off', 'retrieval', 'llm', 'llm_live']:
            if not (folder / f'{mode}.json').exists():
                continue
            receipt = json.loads((folder / f'{mode}.json').read_text())
            values = np.load(folder / f'{mode}.npz')['values']
            final = values.min(1).astype(float)
            delta = final - baseline
            rows.append(dict(group=group, fid=fid, mode=mode, n=10, mean=float(final.mean()), std=float(final.std()),
                             paired_baseline_mean=float(baseline.mean()), paired_baseline_std=float(baseline.std()),
                             historical_100_mean=float(baseline_all.mean()), historical_100_std=float(baseline_all.std()),
                             mean_difference=float(delta.mean()), paired_wins=int((delta < 0).sum()), ties=int((delta == 0).sum()), losses=int((delta > 0).sum()),
                             changed_decisions=receipt['agent_stats']['changed'], decisions=receipt['agent_stats']['decisions'],
                             api_requests=receipt['agent_stats']['requests'], api_failures=receipt['agent_stats']['failures'],
                             elapsed_seconds=receipt['elapsed_seconds']))
            for index in range(10):
                trajectories.append(dict(group=group, fid=fid, mode=mode, trajectory=index, final=float(final[index]), baseline=float(baseline[index]), delta=float(delta[index]), nfe=300))
    for filename, table in [('FUNCTION_COMPARISON.csv', rows), ('PAIRED_TRAJECTORIES.csv', trajectories)]:
        if table:
            with (OUT / filename).open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(table[0]))
                writer.writeheader()
                writer.writerows(table)
    summary = dict(status='complete' if not missing else 'partial', functions=len(TASKS)-len(missing), missing=missing, n_per_function=10,
                   baseline_n=100, paired_reference='first10 exact archived trajectories', groups={},
                   caveats=['BBOB_f20 stochastic RNG coupling; excluded from primary count', 'CEC original structured lock preserved; expected no intervention', 'UAV recovered proxy protocol; not original ZIP identity', 'pilot only; no independent full100 confirmation'])
    for group in ['BBOB', 'CEC-style', 'UAV']:
        selected = [row for row in rows if row['mode'] == 'llm_live' and row['group'] == group and not (group == 'BBOB' and row['fid'] == 20)]
        summary['groups'][group] = dict(functions=len(selected), wins=sum(row['mean_difference'] < 0 for row in selected), ties=sum(row['mean_difference'] == 0 for row in selected), losses=sum(row['mean_difference'] > 0 for row in selected), changed=sum(row['changed_decisions'] for row in selected), decisions=sum(row['decisions'] for row in selected))
    journals = [json.loads(path.read_text()) for path in (OUT / 'api_journal').rglob('*.json')]
    summary['api'] = dict(requests=len(journals), successful=sum(row['status'] == 'ok' for row in journals), failed=sum(row['status'] != 'ok' for row in journals), total_tokens=sum(row.get('response', {}).get('usage', {}).get('total_tokens', 0) for row in journals), elapsed_seconds=sum(row.get('elapsed_seconds', 0) for row in journals), currency_cost='not supplied by provider; token usage recorded')
    write_json(OUT / 'SUMMARY.json', summary)
    if not missing:
        write_json(OUT / 'COMPLETE.json', dict(status='complete', functions=35, trajectories=1050, logical_fe=315000, original_baseline_reused=True))
    print(json.dumps(summary, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['offline', 'evaluate', 'summary'])
    parser.add_argument('--limit', type=int, default=35)
    parser.add_argument('--modes', default='off,retrieval,llm_live')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.phase == 'offline':
        train_memory()
    elif args.phase == 'evaluate':
        if (EXP / 'API_AND_DECISION_JOURNAL.tar.xz').exists():
            from scripts.restore_r02_journal import restore
            restore(EXP)
        existing = len(list((OUT / 'api_journal').rglob('*.json')))
        budget = dict(used=existing, limit=96)
        modes = args.modes.split(',')
        assert modes and all(mode in ['off', 'retrieval', 'llm', 'llm_live'] for mode in modes)
        for task in TASKS[:args.limit]:
            run_task(task, budget, modes)
            receipt = OUT / f'{task[0]}_f{task[1]:02d}/llm_live.json'
            if 'llm_live' in modes and receipt.exists() and json.loads(receipt.read_text())['agent_stats']['failures'] >= 2:
                summarize()
                raise RuntimeError('API未产生有效决策，停止新增API试验；失败结果已保存')
        summarize()
    else:
        summarize()


if __name__ == '__main__':
    main()
