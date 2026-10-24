#!/usr/bin/env python3
"""Hardware engineering checks only. No API calls or optimization efficacy claim."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import hashlib
import multiprocessing as mp
import os
from pathlib import Path
import secrets
import sys
import tempfile
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
from hardware_backend import ProxyClient, serve, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'experiments/r07_hardware_checks_v4')
    parser.add_argument('--integration', action='store_true', help='also run a complete paired baseline batch and offline episode branches')
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error('Check directory already contains evidence; use a new --output-dir')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    names = ('scripts/hardware_backend.py', 'scripts/run_r07_hardware.py', 'scripts/run_r07_hardware.sh',
             'scripts/verify_hardware_backend.py')
    write_json(args.output_dir / 'SOURCE_SHA256.json', {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in names})
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    context = mp.get_context('spawn')
    with tempfile.TemporaryDirectory(prefix='roopf_check_') as temporary:
        address, authkey = str(Path(temporary) / 'gpu.sock'), secrets.token_bytes(32)
        ready, child = context.Pipe(duplex=False)
        process = context.Process(target=serve, args=(address, authkey, '0', child,
                                                     str(args.output_dir / 'GPU_STATS.json')))
        process.start()
        child.close()
        clients = []
        try:
            assert ready.poll(60), 'CUDA startup timeout'
            metadata = ready.recv()
            assert metadata['status'] == 'ready', metadata
            import numpy as np
            import torch
            from types import SimpleNamespace
            from roopf.model import RidgeFeatureEnsemble
            torch.set_num_threads(1)
            client = ProxyClient(address, authkey)
            clients.append(client)
            problem = SimpleNamespace(fun=dict(xlb=-5, xub=5))
            cases, records = [], []
            for batch, n, distance_scale, constant in [(1, 100, 0.15, False), (10, 299, 0.15, False),
                                                      (10, 100, 0, False), (1, 200, 0.15, True)]:
                torch.manual_seed(20261023 + len(cases))
                proxy = RidgeFeatureEnsemble(10, 5, 40, .002)
                x = torch.rand(batch, n, 10) * 10 - 5
                y = x.square().sum(-1) if not constant else torch.ones(batch, n)
                candidates = torch.rand(batch, 38, 10) * 10 - 5
                state = torch.get_rng_state().clone()
                expected = proxy.predict(x, y, candidates, problem, distance_scale)
                actual = client.predict(proxy, x, y, candidates, problem, distance_scale)
                assert torch.equal(state, torch.get_rng_state()), 'Backend advanced solver RNG'
                scale = y.std(1, unbiased=False).clamp_min(1e-8)[:, None]
                errors = [float(((a - e).abs() / (scale + 1e-5)).max()) for a, e in zip(actual, expected)]
                # Preregistered engineering tolerance; not an objective/result tolerance.
                for a, e in zip(actual, expected):
                    assert torch.all((a - e).abs() <= 2e-3 * scale + 1e-5), errors
                cases.append((proxy, x, y, candidates, distance_scale, actual))
                records.append(dict(batch=batch, archive=n, distance_scale=distance_scale,
                                    constant_objective=constant, normalized_errors=errors))

            # Scheduler grouping must not change predictions for the same request.
            barrier = threading.Barrier(16)
            def concurrent(index):
                remote = ProxyClient(address, authkey)
                try:
                    proxy, x, y, candidates, distance, expected = cases[index % len(cases)]
                    barrier.wait(timeout=30)
                    actual = remote.predict(proxy, x, y, candidates, problem, distance)
                    return all(torch.equal(a, e) for a, e in zip(actual, expected))
                finally:
                    remote.close()
            with ThreadPoolExecutor(16) as pool:
                assert all(pool.map(concurrent, range(16))), 'Queue-dependent CUDA predictions'
            # Compare steady-state compute including IPC, not startup/import time.
            timings = []
            for proxy, x, y, candidates, distance, _ in cases[:2]:
                functions = dict(cpu=lambda: proxy.predict(x, y, candidates, problem, distance),
                                 gpu_ipc=lambda: client.predict(proxy, x, y, candidates, problem, distance))
                result = dict(batch=len(x), repetitions=20)
                for name, function in functions.items():
                    function()
                    start = time.perf_counter()
                    for _ in range(20):
                        function()
                    result[name + '_seconds_per_request'] = (time.perf_counter() - start) / 20
                result['proxy_speedup'] = result['cpu_seconds_per_request'] / result['gpu_ipc_seconds_per_request']
                timings.append(result)
            summary = dict(status='engineering_checks_passed', hardware=metadata, comparisons=records,
                           concurrent_requests=16, queue_grouping_bitwise_invariant=True,
                           cpu_rng_unchanged=True, timings=timings, actual_api_calls=0,
                           full_live_validation=False, optimization_improvement_claim=False)
            write_json(args.output_dir / 'BACKEND_CHECKS.json', summary)
            print(json.dumps(summary, ensure_ascii=False), flush=True)
            if args.integration:
                from run_r07_hardware import configure
                run, collection = configure(str(args.output_dir), dict(device='cuda', gpu_min_batch=1), address, authkey)
                # Replay one entire 10-trajectory batch with the frozen baseline.
                model = run.legacy.build(original=True)
                objective = run.legacy.problem_for('BBOB', 1)
                calls, _ = run.storage.full_calls('BBOB', 1)
                population = objective.genRandomPop((10, 100, 10))
                replay = run.legacy.Replay(objective, calls[0])
                with torch.no_grad():
                    model(population, replay)
                assert np.array_equal(torch.cat(replay.values, 1).numpy(), run.legacy.original_values('BBOB', 1)[:10])
                print('Frozen CPU baseline: all 300 evaluations of 10 trajectories match exactly', flush=True)
                # Execute same-prefix GPU branches as used by offline collection.
                reference, _ = collection.episode(0, 0, {})
                step = 5
                prefix = reference['records'][step]['current'][0]['generation']['nfe_before']
                reference_context = reference['records'][step]['current'][0]['generation']
                for name in ('joint_sharp_wide', 'joint_soft_fine'):
                    branch, _ = collection.episode(0, 0, {step: {0: name}})
                    assert np.array_equal(branch['values'][:, :prefix], reference['values'][:, :prefix])
                    assert branch['records'][step]['current'][0]['generation'] == reference_context
                    print(f'GPU paired prefix verified: {name}, prefix={prefix}', flush=True)
                write_json(args.output_dir / 'INTEGRATION_CHECKS.json', dict(
                    status='engineering_integration_passed', baseline_trajectories=10, nfe=300,
                    paired_gpu_offline_branches=2, actual_api_calls=0, full_live_validation=False))
                # Real spawn workers, complete collection tasks, immutable bindings,
                # and the default CPU dispatch for small offline episodes.
                from run_r07_hardware import Progress, run_pool
                pool_directory = args.output_dir / 'pool_run'
                pool_directory.mkdir()
                write_json(pool_directory / 'PROTOCOL.json', run.settings() if (run.EXP / 'PROTOCOL.json').exists()
                           else json.loads((ROOT / 'experiments/r07_joint_generation/PROTOCOL.json').read_text()))
                runtime = dict(device='cuda', gpu_min_batch=2, workers=2)
                run, collection = configure(str(pool_directory), runtime)
                collection.freeze_collection()
                progress = Progress(pool_directory, 5, args.output_dir / 'GPU_STATS.json')
                progress.begin('engineering_collect', 2)
                try:
                    run_pool(progress, collection.collect, [(0, 0), (1, 0)], 2,
                             (str(pool_directory), runtime, address, authkey))
                    for index in (0, 1):
                        receipt = json.loads((pool_directory / 'collection' / f'f{index:02d}_s00/COMPLETE.json').read_text())
                        assert receipt['status'] == 'complete' and receipt['real_api_calls'] == 0
                    write_json(args.output_dir / 'SPAWN_COLLECTION_CHECKS.json', dict(
                        status='passed', completed_tasks=2, workers=2, complete_counterfactual_branches=True,
                        cpu_small_task_dispatch=True, real_api_calls=0, full_live_validation=False))
                    print('Spawn pool: two complete offline collection tasks passed', flush=True)
                finally:
                    progress.close()
        except BaseException:
            import traceback
            write_json(args.output_dir / 'FAILED_CHECKS.json', dict(status='failed', traceback=traceback.format_exc()))
            raise
        finally:
            for client in clients:
                client.close()
            if process.is_alive() and Path(address).exists():
                from multiprocessing.connection import Client
                with Client(address, family='AF_UNIX', authkey=authkey) as connection:
                    connection.send(dict(op='shutdown'))
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)


if __name__ == '__main__':
    main()
