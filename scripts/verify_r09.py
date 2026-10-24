"""R09 engineering contracts and exact matched38 restoration; not efficacy validation."""
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



def contracts(raw):
    from concurrent.futures import ThreadPoolExecutor
    import threading
    import time
    import numpy as np
    import torch
    import roopf.policy_api as api
    from roopf.evidence_fusion_agent import EvidenceFusionAgent as Agent, PROFILES, ACTION_NAMES, ACTION_LOOKUP, digest

    def payload(value=0):
        return dict(request_kind='generation', trajectories=[dict(trajectory_id='0',
            current=dict(features=[value] + [0.0] * 7, feedback_samples=0, proxy_error_samples=0,
                         previous_generation={}, source_statistics={}, reviews=[]),
            offline_cases=[], eligible_configs=list(PROFILES))])

    def response(actions, finish='stop'):
        return dict(choices=[dict(finish_reason=finish, message=dict(content=json.dumps(dict(actions=actions))))])

    agent = Agent(None, raw / 'contract')
    compact = agent.compact_payload(payload())
    agent.execution_context = dict(paired_solver_rng_sha256='first', batch=0)
    first = agent.compact_payload(payload())
    agent.execution_context = dict(paired_solver_rng_sha256='second', batch=1)
    second = agent.compact_payload(payload())
    assert first['trajectories'] == second['trajectories']
    assert first['decision_context_sha256'] != second['decision_context_sha256']
    assert first['execution_context_sha256'] != second['execution_context_sha256']
    agent.execution_context = None
    all_rows = [[str(i), *pair] for i, pair in enumerate(ACTION_LOOKUP)]
    all_payload = dict(compact, trajectories=[dict(compact['trajectories'][0], trajectory_id=str(i)) for i in range(22)])
    assert agent.decode(response(all_rows), all_payload) == list(ACTION_LOOKUP.values())
    bad = [response([['0', 0, 1]], 'length'), response([['0', 0, 1], ['0', 0, 1]]),
           response([['0', 1, 0]]), response([['0', True, 1]]), response([]),
           response([['unknown', 0, 1]]), dict(choices=[dict(finish_reason='stop', message=dict(content='reasoning only'))])]
    for r in bad:
        try:
            agent.decode(r, compact)
        except (ValueError, KeyError, TypeError):
            pass
        else:
            raise AssertionError('Invalid final response was accepted')
    with patch.object(api, 'send_request', return_value=response([['0', 0, 1]])) as send:
        assert agent.call(payload()) == ['matched38']
        assert agent.call(payload()) == ['matched38']
        assert send.call_count == 1 and agent.stats['api_reused'] == 1
    failed = Agent(None, raw / 'failed')
    with patch.object(api, 'send_request', side_effect=RuntimeError('mock http_503')) as send:
        assert failed.call(payload()) == ['matched38']
        assert failed.call(payload()) == ['matched38']
        assert send.call_count == 1
    pending_path = next(failed.journal.glob('*.json'))
    pending = json.loads(pending_path.read_text()); pending['status'] = 'pending'
    pending_path.write_text(json.dumps(pending))
    uncertain = Agent(None, failed.journal)
    with patch.object(api, 'send_request') as send:
        assert uncertain.call(payload()) == ['matched38'] and uncertain.uncertain_api_outcome
        send.assert_not_called()
    truncated = Agent(None, raw / 'truncated')
    with patch.object(api, 'send_request', return_value=response([['0', 0, 1]], 'length')):
        assert truncated.call(payload()) == ['matched38']
        assert json.loads(next(truncated.journal.glob('*.json')).read_text())['error'] == 'output_truncated_or_incomplete'
    active, maximum, calls = 0, 0, 0
    lock = threading.Lock()
    def concurrent_send(*_):
        nonlocal active, maximum, calls
        with lock:
            active += 1; calls += 1; maximum = max(maximum, active)
        time.sleep(0.08)
        with lock:
            active -= 1
        return response([['0', 0, 1]])
    clients = [Agent(None, raw / 'concurrency', budget=dict(limit=4, used=0)) for _ in range(4)]
    with patch.object(api, 'send_request', side_effect=concurrent_send):
        with ThreadPoolExecutor(max_workers=4) as pool:
            assert list(pool.map(lambda i: clients[i].call(payload(i)), range(4))) == [['matched38']] * 4
        assert calls == 4 and maximum == 2
        capped = Agent(None, raw / 'concurrency', budget=dict(limit=4, used=0))
        assert capped.call(payload(99)) == ['matched38'] and capped.resource_exhausted
        assert calls == 4
    # Exercise the actual numerical fusion; the real protected selector is also
    # checked by the frozen GPU trajectory/RNG comparisons below.
    def select(name, residual_slot=1):
        a = Agent(None, raw / 'numerical', mode='collect'); a.reset(1)
        a.ids = [name]; a.width = torch.ones(1, 1, 2); a.scale = torch.ones(1)
        a.pre_best = torch.ones(1); a.protected_point = torch.zeros(1, 1, 2)
        a.protected_mu = torch.zeros(1, 1); a.extra_origin = ['native_extra']
        a.anchor_diagnostics = [{}]; a.generation_context = [{}]
        a.local_scale = torch.tensor([PROFILES[name]['local_scale']]); a.request_event = False
        a.api_trajectories = []
        online = torch.ones(1, 38); online[0, 0] = 0; online[0, 1] = 0.015
        probability = torch.zeros(1, 38); probability[0, residual_slot] = 1
        signal = ((probability - probability.mean(1, keepdim=True)) / probability.std(1, keepdim=True, unbiased=False)).clamp(-2.5, 2.5)
        chosen = a.choose(candidates=torch.arange(76).float().reshape(1, 38, 2) / 100 + 1,
            original_idx=torch.zeros(1, 1, dtype=torch.long), score=online - .004 * signal,
            probability=probability, residual_weight=.004, mu=torch.zeros(1, 38), sigma=torch.ones(1, 38),
            candidate_ops=torch.zeros(1, 38, dtype=torch.long),
            selection_rule=lambda acquisition, proposed, **kw: proposed.clone())
        return int(chosen[0, 0]), a
    assert select('matched38@online')[0] == 0
    assert select('matched38')[0] == 0
    selected, boosted = select('matched38@residual_boost')
    assert selected == 1 and boosted.stats['fusion_changed_points'] == 1
    selected, excluded = select('local_wide@residual_boost', residual_slot=22)
    assert selected == 0 and excluded.last_record['current'][0]['residual_eligible'][22] == 0
    # Reuse old contexts only to measure serialization size, never as R09 training labels.
    sizes, old_sizes = [], []
    for path in (ROOT / 'experiments/r08_matched38_agent/api_journal').glob('*.json'):
        record = json.loads(path.read_text())
        prior = json.JSONDecoder().raw_decode(record['request']['messages'][-1]['content'].split('INPUT:\n', 1)[1])[0]
        for item in prior['trajectories']:
            item['eligible_configs'] = list(PROFILES)
            for case in item['offline_cases']:
                for horizon in ('advantages5', 'advantages10'):
                    case[horizon] = {name: case[horizon][v['generation']] for name, v in PROFILES.items()}
        compact = agent.compact_payload(prior)
        request = dict(model='deepseek-v4-flash', temperature=0, max_tokens=8192,
                       thinking={'type':'disabled'}, response_format={'type':'json_object'},
                       messages=[dict(role='system', content=agent.system_prompt),
                                 dict(role='user', content='INPUT:\n'+json.dumps(compact,separators=(',',':')))])
        sizes.append(len(api.wire_bytes(request))); old_sizes.append(len(json.dumps(record['request']).encode()))
    assert max(sizes) < agent.api_config['max_request_bytes']
    size_report = dict(contexts=len(sizes), old_median_bytes=float(np.median(old_sizes)),
                       new_median_bytes=float(np.median(sizes)), new_max_bytes=max(sizes),
                       synthetic_extra_action_advantages_for_size_only=True)
    print('Mocked transport, full22 schema, concurrency, true fusion effect and input bounds passed: '+json.dumps(size_report), flush=True)
    return dict(all22_actions_decoded=True, strict_final_output=True, no_automatic_retry=True,
                request_identity_includes_paired_rng=True, unknown_outcomes_block_completion=True,
                max_concurrent_requests=maximum, global_api_cap=True, exact_request_reuse=True,
                numerical_fusion_changes_selected_point=True, changed_distribution_residual_mask=True,
                request_size_contract=size_report, actual_api_calls=0)


def main():
    directory = ROOT / 'experiments/r09_evidence_fusion/engineering_checks'
    directory.mkdir(exist_ok=True)
    if (directory / 'CHECKS.json').exists():
        raise RuntimeError('Preserve existing CHECKS.json before rerunning engineering checks')
    raw_root = directory / 'raw'
    raw_root.mkdir(exist_ok=True)
    raw = Path(tempfile.mkdtemp(prefix='attempt_', dir=raw_root))
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    context = mp.get_context('spawn')
    checks = dict(status='running', full_live_validation=False, actual_api_calls=0)
    with tempfile.TemporaryDirectory(prefix='roopf_r09_verify_') as temporary:
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
            from roopf.matched38_agent import Matched38Agent
            from roopf.evidence_fusion_agent import EvidenceFusionAgent, POLICY_ID, PROFILES
            from run_r09_hardware import configure
            run, collection = configure(str(ROOT / 'experiments/r09_evidence_fusion'),
                                        dict(device='cuda', gpu_min_batch=2), address, key)
            checks['mocked_contract_checks'] = contracts(raw)
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
                for cls in (Matched38Agent, EvidenceFusionAgent):
                    model = run.legacy.build()
                    controller = cls(None, raw / ('old' if cls is Matched38Agent else 'new'), mode='retrieval', arm='matched38')
                    model.fusion_agent = controller
                    objective = run.legacy.Counted(problem, calls[0], stochastic=fid == 20)
                    torch.set_rng_state(start)
                    with torch.no_grad():
                        model(population.clone(), objective)
                    values.append(torch.cat(objective.values, 1).numpy())
                    end_rng.append(torch.get_rng_state().numpy())
                frozen = np.load(ROOT / 'experiments/r07_hardware_accelerated_v1/results' / f'BBOB_f{fid:02d}/b00/matched38.npz')['values']
                assert np.array_equal(values[0], frozen), f'Old matched38 replay differs: f{fid}'
                assert np.array_equal(values[1], frozen), f'R09 changed matched38 base: f{fid}'
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
            print('All 22 factorized actions and paired offline prefixes verified', flush=True)
            checks['offline_actions_verified'] = list(PROFILES)
            previous = json.loads((directory.parent / 'PRE_MIGRATION.json').read_text())
            assert all(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == sha
                       for name, sha in previous['previous_records_and_sources_sha256'].items())
            checks['previous_round_records_and_sources_unchanged'] = True
            checks['status'] = 'engineering_checks_passed'
        except BaseException:
            import traceback
            checks.update(status='failed', traceback=traceback.format_exc())
            raise
        finally:
            names = ('roopf/evidence_fusion_agent.py', 'roopf/policy_api.py', 'scripts/collect_r09.py', 'scripts/experiment_r09_full.py',
                     'scripts/run_r09_hardware.py', 'scripts/report_r09.py', 'scripts/verify_r09.py',
                     'scripts/audit_r09_records.py', 'scripts/worldcodes_request_r09.ps1', 'scripts/run_r09_hardware.sh')
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
