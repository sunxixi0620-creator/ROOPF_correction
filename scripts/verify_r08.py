"""R08 engineering contracts and exact matched38 restoration; not efficacy validation."""
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import secrets
import sys
import tempfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
from hardware_backend import serve, write_json


def main():
    directory = ROOT / 'experiments/r08_matched38_agent/engineering_checks'
    directory.mkdir(exist_ok=False)
    raw = directory / 'raw'
    raw.mkdir()
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    context = mp.get_context('spawn')
    checks = dict(status='running', full_live_validation=False, actual_api_calls=0)
    with tempfile.TemporaryDirectory(prefix='roopf_r08_verify_') as temporary:
        address, key = str(Path(temporary) / 'gpu.sock'), secrets.token_bytes(32)
        ready, child = context.Pipe(duplex=False)
        process = context.Process(target=serve, args=(address, key, '0', child, str(directory / 'GPU_STATS.json'), 2))
        process.start()
        child.close()
        try:
            assert ready.poll(60), 'CUDA startup timeout'
            result = ready.recv()
            assert result['status'] == 'ready', result
            import numpy as np
            import torch
            from roopf.joint_generation_agent import JointGenerationAgent, PROFILES
            from roopf.matched38_agent import Matched38Agent, POLICY_ID
            from run_r08_hardware import configure
            run, collection = configure(str(ROOT / 'experiments/r08_matched38_agent'),
                                        dict(device='cuda', gpu_min_batch=2), address, key)
            # Deliberately mocked contract failures, not live API experiments.
            agent = Matched38Agent(None, raw / 'contract', mode='llm')
            payload = dict(trajectories=[dict(trajectory_id=str(i)) for i in range(2)])
            with patch.object(JointGenerationAgent, 'call', side_effect=RuntimeError('mock transport failure')):
                assert agent.call(payload) == ['matched38', 'matched38']
            assert agent.stats['failures'] == 1
            with patch.object(JointGenerationAgent, 'call', return_value=['anchor_sharp', 'release']) as call:
                assert agent.call(payload) == ['anchor_sharp', 'release']
                sent = call.call_args.args[0]
                assert sent['policy_id'] == POLICY_ID and sent['failure_fallback'] == 'matched38'
            assert set(agent.allowed()) == set(PROFILES) and agent.allowed()[0] == 'matched38'
            checks['mocked_contract_checks'] = dict(fallback_matched38=True, explicit_release_retained=True,
                                                    all_eight_actions_retained=True, policy_identity_rewritten=True)
            # Restore the actual frozen matched38 trajectories, including stochastic f20.
            for fid in (9, 20):
                run.legacy.build(original=True)
                torch.set_rng_state(torch.from_numpy(np.frombuffer(
                    (run.legacy.SNAP / f'BBOB_f{fid-1:02d}/FULL_RNG.bin').read_bytes(), dtype=np.uint8).copy()))
                problem = run.legacy.problem_for('BBOB', fid)
                population = problem.genRandomPop((10, 100, 10))
                start = torch.get_rng_state().clone()
                calls, _ = run.storage.full_calls('BBOB', fid)
                values, end_rng = [], []
                for cls in (JointGenerationAgent, Matched38Agent):
                    model = run.legacy.build()
                    controller = cls(None, raw / ('old' if cls is JointGenerationAgent else 'new'), mode='retrieval', arm='matched38')
                    model.fusion_agent = controller
                    objective = run.legacy.Counted(problem, calls[0], stochastic=fid == 20)
                    torch.set_rng_state(start)
                    with torch.no_grad():
                        model(population.clone(), objective)
                    values.append(torch.cat(objective.values, 1).numpy())
                    end_rng.append(torch.get_rng_state().numpy())
                frozen = np.load(ROOT / 'experiments/r07_hardware_accelerated_v1/results' / f'BBOB_f{fid:02d}/b00/matched38.npz')['values']
                assert np.array_equal(values[0], frozen), f'Old matched38 replay differs: f{fid}'
                assert np.array_equal(values[1], frozen), f'R08 changed matched38 base: f{fid}'
                assert np.array_equal(end_rng[0], end_rng[1]), f'RNG differs: f{fid}'
                print(f'Exact GPU matched38 restoration passed: BBOB-f{fid}, 10 trajectories x 300 evaluations', flush=True)
            checks['exact_frozen_matched38_batches'] = ['BBOB_f09_b00', 'BBOB_f20_b00']
            # Offline same-prefix labels cover every retained action, not a screened subset.
            reference, _ = collection.episode(0, 0, {})
            for step in (5, 15):
                assert reference['records'][step]['configs'] == ['matched38']
            prefix = reference['records'][5]['current'][0]['generation']['nfe_before']
            for name in PROFILES:
                branch, _ = collection.episode(0, 0, {5: {0: name}})
                assert branch['records'][5]['configs'] == [name]
                assert branch['records'][5]['policy_id'] == POLICY_ID
                assert np.array_equal(branch['values'][:, :prefix], reference['values'][:, :prefix])
                assert branch['records'][5]['current'][0]['generation'] == reference['records'][5]['current'][0]['generation']
            print('All eight action profiles and paired offline prefixes verified', flush=True)
            checks['offline_actions_verified'] = list(PROFILES)
            checks['status'] = 'engineering_checks_passed'
        except BaseException:
            import traceback
            checks.update(status='failed', traceback=traceback.format_exc())
            raise
        finally:
            names = ('roopf/matched38_agent.py', 'scripts/collect_r08.py', 'scripts/experiment_r08_full.py',
                     'scripts/run_r08_hardware.py', 'scripts/report_r08.py', 'scripts/verify_r08.py')
            checks['source_sha256'] = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in names}
            write_json(directory / 'CHECKS.json', checks)
            if process.is_alive() and Path(address).exists():
                from multiprocessing.connection import Client
                with Client(address, family='AF_UNIX', authkey=key) as connection:
                    connection.send(dict(op='shutdown'))
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)


if __name__ == '__main__':
    main()
