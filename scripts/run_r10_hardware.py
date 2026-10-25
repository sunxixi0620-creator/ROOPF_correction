#!/usr/bin/env python3
"""Full R10 matched38-default pipeline with CPU solvers and a shared CUDA ridge service.

Historical files are imported, not rewritten. Scientific parameters stay fixed.
Hardware runs have their own directory and bind the backend and runtime config.
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import gzip
import hashlib
import importlib.metadata
import json
import multiprocessing as mp
import os
from pathlib import Path
import secrets
import subprocess
import sys
import tempfile
import threading
import time
import traceback

os.environ['TZ'] = 'Asia/Shanghai'
time.tzset()
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
from hardware_backend import ProxyClient, serve, write_json

EXTRA_SOURCES = ('scripts/hardware_backend.py', 'scripts/run_r10_hardware.py',
                 'scripts/run_r10_hardware.sh')


def available_memory():
    rows = Path('/proc/meminfo').read_text().splitlines()
    return next(int(row.split()[1]) * 1024 for row in rows if row.startswith('MemAvailable:'))


def automatic_workers():
    cpus = len(os.sched_getaffinity(0))
    # Leave room for the CUDA owner, filesystem cache, desktop and API transport.
    memory_slots = max(1, (available_memory() - 8 * 1024**3) // (2 * 1024**3))
    return max(1, min(cpus - 2, memory_slots, 24))


def configure(directory, runtime, address=None, authkey=None):
    """Runs in each spawned CPU worker before a scientific task is dispatched."""
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    import experiment_r10_full as run
    import collect_r10 as collection
    import torch
    torch.set_num_threads(1)
    run.EXP = collection.EXP = Path(directory)
    run.MEMORY = run.EXP / 'offline_memory.json'
    run.SOURCES = tuple(dict.fromkeys((*run.SOURCES, *EXTRA_SOURCES)))
    original_binding = getattr(run, '_hardware_original_binding', run.binding)
    run._hardware_original_binding = original_binding

    def binding(include_memory=True):
        value = original_binding(include_memory)
        value['hardware_runtime_sha256'] = hashlib.sha256(
            json.dumps(runtime, sort_keys=True).encode()).hexdigest()
        return value

    run.binding = binding
    original_build = getattr(run.legacy, '_hardware_original_build', run.legacy.build)
    run.legacy._hardware_original_build = original_build
    run.legacy.build = original_build
    previous_client = getattr(run, '_hardware_proxy_client', None)
    if previous_client is not None:
        previous_client.close()
        run._hardware_proxy_client = None
    if address is not None:
        client = ProxyClient(address, authkey)
        run._hardware_proxy_client = client

        def build(original=False):
            model = original_build(original=original)
            if original:
                return model
            proxy = model.surrogate
            cpu_predict = proxy.predict

            def predict(archive_x, archive_y, candidates, problem, distance_scale=0.15):
                # Original and Agent-off always retain the exact historical CPU path.
                if getattr(model, 'fusion_agent', None) is None or archive_x.shape[0] < runtime.get('gpu_min_batch', 2):
                    return cpu_predict(archive_x, archive_y, candidates, problem, distance_scale)
                return client.predict(proxy, archive_x, archive_y, candidates, problem, distance_scale)

            proxy.predict = predict
            return model

        run.legacy.build = build
    return run, collection


def audit_execution(run):
    from report_r10 import audit_api
    return audit_api(run, write=True)


def process_tree_usage():
    """Linux process tree accounting without an extra psutil dependency."""
    entries = {}
    for path in Path('/proc').glob('[0-9]*/stat'):
        try:
            raw = path.read_text()
            fields = raw[raw.rfind(')') + 2:].split()
            entries[int(path.parent.name)] = (int(fields[1]), int(fields[11]) + int(fields[12]), int(fields[21]))
        except (OSError, ValueError, IndexError):
            continue
    included = {os.getpid()}
    while True:
        children = {pid for pid, (parent, _, _) in entries.items() if parent in included}
        added = children - included
        if not added:
            break
        included.update(added)
    ticks = sum(entries[pid][1] for pid in included if pid in entries)
    rss = sum(entries[pid][2] for pid in included if pid in entries) * os.sysconf('SC_PAGE_SIZE')
    return ticks / os.sysconf('SC_CLK_TCK'), rss, len(included)


class Progress:
    def __init__(self, directory, interval, gpu_stats):
        self.directory, self.interval, self.gpu_stats = Path(directory), interval, gpu_stats
        self.stage, self.done, self.total = 'setup', 0, 0
        self.began = self.stage_began = time.monotonic()
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self.monitor, daemon=True)
        self.thread.start()

    def event(self, **data):
        data.update(time=time.strftime('%Y-%m-%dT%H:%M:%S%z'), elapsed_seconds=round(time.monotonic() - self.began, 1))
        line = json.dumps(data, ensure_ascii=False)
        with self.lock:
            with (self.directory / 'PROGRESS.jsonl').open('a') as stream:
                stream.write(line + '\n')
            print(line, flush=True)

    def begin(self, stage, total):
        self.stage, self.done, self.total = stage, 0, total
        self.stage_began = time.monotonic()
        self.event(event='stage_start', stage=stage, total=total)

    def monitor(self):
        previous_seconds, _, _ = process_tree_usage()
        previous_time = time.monotonic()
        cpus = len(os.sched_getaffinity(0))
        while not self.stop.wait(self.interval):
            try:
                seconds, rss, processes = process_tree_usage()
                now = time.monotonic()
                cores = max(0, seconds - previous_seconds) / (now - previous_time)
                previous_seconds, previous_time = seconds, now
                data = dict(event='progress', stage=self.stage, completed=self.done, total=self.total,
                            percent=round(100 * self.done / self.total, 1) if self.total else None,
                            cpu_cores_used=round(cores, 2), cpu_capacity_percent=round(100 * cores / cpus, 1),
                            process_tree_rss_gib=round(rss / 1024**3, 2), processes=processes)
                if self.stage == 'run':
                    data['completed_arm_batches'] = sum(1 for _ in self.directory.glob('results/*/b??/*.json'))
                    protocol = json.loads((self.directory / 'PROTOCOL.json').read_text())
                    data['total_arm_batches'] = 35 * 10 * len(protocol['arms'])
                    api_counts = {'ok': 0, 'partial': 0, 'failed': 0, 'pending': 0}
                    for journal_path in (self.directory / 'api_journal').glob('*.json'):
                        entry = json.loads(journal_path.read_text())
                        api_counts[entry['status']] += 1
                    data['api_journal_entries'] = sum(api_counts.values())
                    data['actual_api'] = api_counts
                    done_batches = data['completed_arm_batches']
                    data['arm_batch_percent'] = round(100 * done_batches / data['total_arm_batches'], 1)
                    if done_batches and done_batches < data['total_arm_batches']:
                        data['eta_arm_batch_seconds'] = round((now - self.stage_began) * (data['total_arm_batches'] - done_batches) / done_batches)
                if self.done and self.done < self.total:
                    data['eta_seconds'] = round((now - self.stage_began) * (self.total - self.done) / self.done)
                try:
                    result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total',
                                             '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=3)
                    data['gpu_util_memory_mib'] = result.stdout.strip()
                except (OSError, subprocess.TimeoutExpired):
                    pass
                if self.gpu_stats and Path(self.gpu_stats).exists():
                    stats = json.loads(Path(self.gpu_stats).read_text())
                    data['gpu_requests'] = stats.get('requests', 0)
                    data['gpu_batches'] = stats.get('gpu_batches', 0)
                    data['gpu_failures'] = stats.get('failures', 0)
                self.event(**data)
            except Exception as error:
                self.event(event='monitor_error', message=str(error))

    def close(self):
        self.stop.set()
        self.thread.join(timeout=5)


def run_pool(progress, function, tasks, workers, initializer_args):
    tasks = list(tasks)
    with mp.get_context('spawn').Pool(min(workers, len(tasks)), initializer=configure,
                                      initargs=initializer_args) as pool:
        pending = pool.imap_unordered(function, tasks, chunksize=1)
        while progress.done < len(tasks):
            try:
                receipt = pending.next(timeout=1)
            except mp.TimeoutError:
                continue
            progress.done += 1
            progress.event(event='task_complete', stage=progress.stage, completed=progress.done,
                           total=len(tasks), task={key: receipt[key] for key in ('group', 'fid', 'function', 'seed_index')
                                                   if key in receipt})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', nargs='?', default='all', choices=['all', 'collect', 'memory', 'run', 'report', 'verify'])
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--workers', type=int, default=None, help='default: CPU affinity and available RAM, maximum 24')
    parser.add_argument('--gpu-concurrency', type=int, default=2, help='CUDA streams in the shared GPU process')
    parser.add_argument('--visible-devices', default=os.environ.get('CUDA_VISIBLE_DEVICES', '0'))
    parser.add_argument('--progress-seconds', type=float, default=5.0)
    parser.add_argument('--api-transport', choices=['proxy_http', 'windows'], default='proxy_http')
    parser.add_argument('--api-proxy', default='http://127.0.0.1:7890')
    parser.add_argument('--api-concurrency', type=int, default=2)
    parser.add_argument('--api-timeout', type=int, default=90)
    parser.add_argument('--api-max-tokens', type=int, default=1024)
    parser.add_argument('--experiment-dir', type=Path, default=ROOT / 'experiments/r10_current_candidate_fusion')
    args = parser.parse_args()
    if Path(sys.prefix).name != 'ooneurgo':
        parser.error('Use the ooneurgo conda environment: conda activate ooneurgo')
    args.workers = args.workers if args.workers is not None else automatic_workers()
    if args.workers < 1 or args.progress_seconds < 1 or not 1 <= args.gpu_concurrency <= 8:
        parser.error('Invalid worker/batching/progress setting')
    if not 1 <= args.api_concurrency <= 8 or not 30 <= args.api_timeout <= 300 or not 512 <= args.api_max_tokens <= 32768:
        parser.error('Invalid API concurrency, timeout or output-token budget')
    directory = args.experiment_dir.resolve()
    if not directory.is_relative_to(ROOT / 'experiments'):
        parser.error('Experiment directory must be inside this copy of experiments/')
    if directory == ROOT / 'experiments' or directory.exists() and not (directory / 'HARDWARE_RUNTIME.json').exists() and not (directory / 'PRE_MIGRATION.json').exists():
        parser.error('Use a new hardware experiment directory; historical directories are protected')
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / '.run.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error('Another launch is already using this experiment directory')
        runtime = dict(device=args.device, workers=args.workers, visible_devices=args.visible_devices,
                       batching='fixed per-request trajectories and ensemble members; no cross-request merging',
                       gpu_min_batch=2, gpu_concurrency=args.gpu_concurrency,
                       solver_device='cpu', baseline_device='cpu', cpu_threads_per_worker=1,
                       proxy_dtype='float32', tf32=False, cuda_numerics_may_change_decisions=True,
                       python=sys.version, python_executable=sys.executable,
                       packages={name: importlib.metadata.version(name) for name in ('torch', 'numpy', 'sympy')})
        from roopf.policy_api_r10 import DEFAULT_API
        api_config = dict(DEFAULT_API, transport=args.api_transport, proxy=args.api_proxy,
                          concurrency=args.api_concurrency, timeout_seconds=args.api_timeout, max_tokens=args.api_max_tokens)
        runtime['api'] = api_config
        config_path = directory / 'HARDWARE_RUNTIME.json'
        if config_path.exists() and json.loads(config_path.read_text()) != runtime:
            parser.error('Runtime differs from this directory; use a new --experiment-dir')
        if not config_path.exists():
            write_json(config_path, runtime)
            protocol = json.loads((directory / 'PROTOCOL.json').read_text())
            protocol.update(implementation_status='hardware_backend_requires_full_validation',
                            automatic_execution=True, api_and_full_runs_status='not yet complete',
                            hardware_runtime=runtime, api=api_config)
            write_json(directory / 'PROTOCOL.json', protocol)
            if not (directory / 'PRE_MIGRATION.json').exists():
                write_json(directory / 'PRE_MIGRATION.json', json.loads(
                    (ROOT / 'experiments/r10_current_candidate_fusion/PRE_MIGRATION.json').read_text()))
        launch_id = f'{time.strftime("%Y%m%d_%H%M%S")}_{os.getpid()}'
        stats_path = directory / f'GPU_STATS_{launch_id}.json'
        progress = Progress(directory, args.progress_seconds, stats_path if args.device == 'cuda' else None)
        gpu_process, address, authkey = None, None, None
        gpu_temporary = tempfile.TemporaryDirectory(prefix='roopf_gpu_')
        try:
            with contextlib.nullcontext(gpu_temporary.name) as temporary:
                if args.device == 'cuda' and args.phase in ('all', 'collect', 'run'):
                    context = mp.get_context('spawn')
                    ready, child_ready = context.Pipe(duplex=False)
                    address, authkey = str(Path(temporary) / 'cuda.sock'), secrets.token_bytes(32)
                    gpu_process = context.Process(target=serve, args=(address, authkey, args.visible_devices,
                        child_ready, str(stats_path), args.gpu_concurrency))
                    gpu_process.start()
                    child_ready.close()
                    if not ready.poll(60):
                        raise RuntimeError('CUDA backend did not initialize in 60 seconds')
                    metadata = ready.recv()
                    ready.close()
                    if metadata.get('status') != 'ready':
                        raise RuntimeError(metadata.get('error', str(metadata)))
                    progress.event(event='gpu_ready', **metadata)
                    write_json(directory / f'HARDWARE_{launch_id}.json', dict(
                        **metadata, python=sys.executable, prefix=sys.prefix,
                        cpu_affinity=sorted(os.sched_getaffinity(0)), runtime=runtime))
                run, collection = configure(str(directory), runtime)
                # Freeze sources before starting workers. All workers see the same paths.
                collection.freeze_collection()
                progress.event(event='launch', phase=args.phase, workers=args.workers, device=args.device,
                               python=sys.executable, directory=str(directory))
                phases = ['collect', 'memory', 'run', 'report', 'verify'] if args.phase == 'all' else [args.phase]
                for phase in phases:
                    if phase == 'collect':
                        tasks = [(function, seed) for function in range(36) for seed in range(8)]
                        progress.begin(phase, len(tasks))
                        run_pool(progress, collection.collect, tasks, args.workers, (str(directory), runtime, address, authkey))
                    elif phase == 'memory':
                        progress.begin(phase, 1)
                        collection.memory()
                        progress.done = 1
                    elif phase == 'run':
                        run.freeze()
                        progress.begin(phase, len(run.legacy.TASKS))
                        run_pool(progress, run.worker, run.legacy.TASKS, args.workers, (str(directory), runtime, address, authkey))
                    elif phase == 'report':
                        progress.begin(phase, 1)
                        audit = audit_execution(run)
                        progress.event(event='api_execution_audit', **audit)
                        import report_r10
                        report_r10.main(archive_workers=args.workers, archive_progress=lambda done, total, name:
                            progress.event(event='archive_progress', completed=done, total=total, archive=name))
                        progress.done = 1
                    else:
                        progress.begin(phase, 1)
                        import report_r10
                        report_r10.verify(run)
                        progress.done = 1
                    progress.event(event='stage_complete', stage=phase, completed=progress.done, total=progress.total)
                progress.event(event='requested_phases_complete', phases=phases)
        except BaseException:
            write_json(directory / f'FAILURE_{launch_id}.json', dict(status='partial_or_failed', traceback=traceback.format_exc()))
            raise
        finally:
            if gpu_process is not None:
                if gpu_process.is_alive() and address and Path(address).exists():
                    try:
                        from multiprocessing.connection import Client
                        with Client(address, family='AF_UNIX', authkey=authkey) as connection:
                            connection.send(dict(op='shutdown'))
                    except OSError:
                        pass
                gpu_process.join(timeout=5)
                if gpu_process.is_alive():
                    gpu_process.terminate()
                    gpu_process.join(timeout=5)
            progress.close()
            gpu_temporary.cleanup()


if __name__ == '__main__':
    main()
