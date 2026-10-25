"""Complete live-LLM no-Anchor ablation paired to frozen R10 Full."""
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import argparse
import fcntl
import gzip
import hashlib
import importlib.metadata
import json
import multiprocessing as mp
from pathlib import Path
import pickle
import secrets
import sys
import tempfile
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'scripts'))
from hardware_backend import serve, write_json
from run_r10_hardware import Progress, configure as configure_r10, automatic_workers

EXP = ROOT / 'experiments/r11_anchor_ablation'
PARENT = ROOT / 'experiments/r10_current_candidate_fusion'
EXTRA = ('roopf/no_anchor_ablation.py', 'scripts/experiment_r11_anchor_ablation.py',
         'scripts/report_r11_anchor_ablation.py', 'scripts/run_r11_anchor_ablation.sh')


def configure(directory, runtime, address=None, authkey=None):
    global EXP, RUN
    EXP = Path(directory)
    RUN, _ = configure_r10(directory, runtime, address, authkey)
    RUN.SOURCES = tuple(dict.fromkeys((*RUN.SOURCES, *EXTRA)))
    return RUN


def binding():
    return RUN.binding()


def run_function(task):
    import numpy as np
    import torch
    from roopf.no_anchor_ablation import install, NoAnchorAgent
    group, fid = task
    name = f'{group}_f{fid:02d}'
    source = PARENT / 'results' / name
    folder = EXP / 'results' / name
    folder.mkdir(parents=True, exist_ok=True)
    identity = binding()
    marker = folder / 'COMPLETE.json'
    if marker.exists():
        receipt = json.loads(marker.read_text()); assert receipt['binding'] == identity
        return receipt
    model = install(RUN.legacy.build())
    agent = NoAnchorAgent(RUN.MEMORY, EXP / 'api_journal', mode='llm', arm='joint',
        budget=dict(used=0, limit=RUN.settings()['max_new_api_attempts']), api_config=RUN.settings()['api'])
    model.fusion_agent = agent
    problem = RUN.legacy.problem_for(group, fid)
    original_calls, _ = RUN.storage.full_calls(group, fid)
    for batch in range(10):
        target = folder / f'b{batch:02d}'
        target.mkdir(exist_ok=True)
        receipt_path = target / 'llm_no_anchor.json'
        if receipt_path.exists():
            assert json.loads(receipt_path.read_text())['binding'] == identity
            continue
        with gzip.open(source / f'b{batch:02d}/PAIRED_START.pkl.gz', 'rb') as stream:
            paired = pickle.load(stream)
        population = torch.from_numpy(paired['population'].copy())
        rng = torch.from_numpy(paired['rng'].copy())
        frozen = json.loads((source / f'b{batch:02d}/llm_live.json').read_text())
        initial_sha = hashlib.sha256(population.numpy().tobytes()).hexdigest()
        rng_sha = hashlib.sha256(rng.numpy().tobytes()).hexdigest()
        assert initial_sha == frozen['paired_initial_sha256'] and rng_sha == frozen['paired_solver_rng_sha256']
        records = []
        agent.active_record = records.append
        agent.execution_context = dict(group=group, fid=fid, batch=batch, arm='llm_no_anchor',
            paired_initial_sha256=initial_sha, paired_solver_rng_sha256=rng_sha,
            parent_full_binding_sha256=hashlib.sha256((PARENT/'CODE_BINDING.json').read_bytes()).hexdigest())
        cache = original_calls[batch]
        # Exact full request/RNG identities; no reuse of LLM decisions from Full.
        for path in sorted((source / f'b{batch:02d}').glob('*_NATIVE_CALLS.pkl.gz')):
            with gzip.open(path, 'rb') as stream:
                cache = RUN.storage.merged_calls(cache, pickle.load(stream))
        # Require RNG identity for EVERY cache hit: legacy cached responses also
        # restore RNG, even when an objective itself is deterministic.
        measured = RUN.legacy.Counted(problem, cache, stochastic=True)
        model.no_anchor_references = []
        torch.set_rng_state(rng)
        began = time.monotonic()
        with torch.no_grad():
            _, trail, nfe, executed = model(population.clone(), measured)
        values = torch.cat(measured.values, 1).numpy()
        assert nfe == 300 and values.shape == (10, 300) and np.isfinite(values).all()
        assert np.array_equal(values.min(1), trail[:, -1].numpy())
        assert model.baseline_model is None and len(model.no_anchor_references) == 100
        assert len(records) == (0 if group == 'CEC-style' else 45)
        assert not agent.resource_exhausted and not agent.uncertain_api_outcome
        references = torch.stack(model.no_anchor_references, 1).numpy()
        executed = executed.numpy().reshape(10, 100, 2, 10)
        assert np.array_equal(executed[:, :, 0], references[:, :, 0])
        if group == 'CEC-style': assert np.array_equal(executed, references)
        for record in records:
            assert all(c['anchor']['enabled'] is False and c['extra_origin'] == 'native_extra' for c in record['current'])
        np.savez_compressed(target / 'llm_no_anchor.npz', values=values,
            executed_points=executed, native_references=references)
        with gzip.open(target / 'llm_no_anchor_DECISIONS.json.gz', 'wt') as stream:
            json.dump(records, stream, allow_nan=False)
        if measured.new_calls:
            RUN.storage.save_pickle(target / 'llm_no_anchor_NATIVE_CALLS.pkl.gz', measured.new_calls)
        receipt = dict(group=group, fid=fid, batch=batch, arm='llm_no_anchor', n=10, nfe=300,
            native_fe=measured.native_fe, cached_fe=measured.cached_fe, elapsed_seconds=time.monotonic()-began,
            agent_stats=dict(agent.stats), paired_initial_sha256=initial_sha, paired_solver_rng_sha256=rng_sha,
            anchor_model_present=False, native_reference_rounds=100, binding=identity)
        write_json(receipt_path, receipt)
        print(json.dumps(dict(event='batch_complete', group=group, fid=fid, batch=batch, arm='llm_no_anchor')), flush=True)
    all_values = np.concatenate([np.load(folder/f'b{b:02d}/llm_no_anchor.npz')['values'] for b in range(10)])
    np.savez_compressed(folder / 'llm_no_anchor_FULL100.npz', values=all_values)
    receipt = dict(status='complete', group=group, fid=fid, n=100, nfe=300, binding=identity)
    write_json(marker, receipt)
    return receipt


def worker(task):
    try:
        return run_function(task)
    except Exception:
        write_json(EXP/'results'/f'{task[0]}_f{task[1]:02d}'/'ERROR.json', dict(traceback=traceback.format_exc()))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', nargs='?', default='all', choices=['all', 'run', 'report', 'verify'])
    parser.add_argument('--workers', type=int, default=24)
    parser.add_argument('--progress-seconds', type=float, default=5)
    args = parser.parse_args()
    assert Path(sys.prefix).name == 'ooneurgo'
    assert 1 <= args.workers <= 24 and args.progress_seconds >= 1
    runtime = json.loads((PARENT/'HARDWARE_RUNTIME.json').read_text())
    runtime['workers'] = args.workers
    with (EXP/'.run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = EXP/'HARDWARE_RUNTIME.json'
        if path.exists(): assert json.loads(path.read_text()) == runtime
        else: write_json(path, runtime)
        stamp = time.strftime('%Y%m%d_%H%M%S')
        stats = EXP/f'GPU_STATS_{stamp}.json'
        progress = Progress(EXP, args.progress_seconds, stats)
        gpu = None
        try:
            with tempfile.TemporaryDirectory(prefix='roopf_r11_gpu_') as temp:
                address, authkey = str(Path(temp)/'cuda.sock'), secrets.token_bytes(32)
                if args.phase in ('all', 'run'):
                    context = mp.get_context('spawn')
                    ready, child = context.Pipe(duplex=False)
                    gpu = context.Process(target=serve, args=(address, authkey, runtime['visible_devices'],child,str(stats),runtime['gpu_concurrency']))
                    gpu.start(); child.close()
                    assert ready.poll(60), 'CUDA initialization timeout'
                    metadata = ready.recv(); ready.close()
                    assert metadata['status']=='ready', metadata
                    progress.event(event='gpu_ready', **metadata)
                    write_json(EXP/f'HARDWARE_{stamp}.json',metadata)
                else: address=authkey=None
                run = configure(str(EXP), runtime)
                identity = binding()
                frozen = EXP/'CODE_BINDING.json'
                if frozen.exists(): assert json.loads(frozen.read_text()) == identity
                else: write_json(frozen,identity)
                phases = ['run','report','verify'] if args.phase=='all' else [args.phase]
                for phase in phases:
                    if phase=='run':
                        progress.begin('run',35)
                        with mp.get_context('spawn').Pool(args.workers,initializer=configure,
                            initargs=(str(EXP),runtime,address,authkey)) as pool:
                            pending=pool.imap_unordered(worker,run.legacy.TASKS,chunksize=1)
                            while progress.done<35:
                                try: receipt=pending.next(timeout=1)
                                except mp.TimeoutError: continue
                                progress.done+=1
                                progress.event(event='function_complete',group=receipt['group'],fid=receipt['fid'],completed=progress.done,total=35)
                    else:
                        progress.begin(phase,1)
                        from report_r11_anchor_ablation import report
                        result=report(run,verify=phase=='verify')
                        progress.done=1
                        progress.event(event='report_result',status=result['status'])
                    progress.event(event='stage_complete',stage=phase)
                progress.event(event='requested_phases_complete',phases=phases)
        except BaseException:
            write_json(EXP/f'FAILURE_{stamp}.json',dict(status='partial_or_failed',traceback=traceback.format_exc()))
            raise
        finally:
            if gpu is not None:
                if gpu.is_alive():
                    from multiprocessing.connection import Client
                    try:
                        with Client(address,family='AF_UNIX',authkey=authkey) as conn: conn.send(dict(op='shutdown'))
                    except OSError: pass
                gpu.join(timeout=5)
                if gpu.is_alive(): gpu.terminate(); gpu.join(timeout=5)
            progress.close()


if __name__=='__main__':
    main()
