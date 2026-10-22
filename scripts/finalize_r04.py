import csv
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import experiment_r04 as run

EXP = run.EXP


def save(name, value):
    run.write_json(EXP / name, value)


def assert_equal(actual, expected, message):
    if actual != expected:
        raise RuntimeError(message)


def horizon_mean(cases, name, horizon):
    means = []
    for function in sorted({case['function'] for case in cases}):
        seeds = sorted({case['seed_index'] for case in cases if case['function'] == function})
        means.append(float(np.mean([np.mean([case['horizon_advantages'][str(horizon)][name]
            for case in cases if case['function'] == function and case['seed_index'] == seed]) for seed in seeds])))
    return float(np.mean(means)) if means else None


def audit_collection():
    settings, binding = run.protocol(), run.freeze()
    receipts, cases, branch_count = [], [], 0
    for seed in settings['training_seed_indices']:
        for function in run.generated().TRAIN_FUNCTIONS:
            folder = EXP / 'collection' / f"s{seed:02d}_{function['fid']}"
            receipt = json.loads((folder / 'COMPLETE.json').read_text())
            assert_equal(receipt['binding'], binding, 'Collection binding mismatch')
            baseline = run.load_pickle(folder / 'BASELINE.pkl.gz')
            observed = np.concatenate([np.frombuffer(call['y'], dtype=np.float32).reshape(call['shape'][:2]) for call in baseline['calls']], axis=1)
            if not np.array_equal(observed, baseline['values']) or observed.shape != (1, 300):
                raise RuntimeError('Baseline objective receipt mismatch')
            assert_equal(hashlib.sha256(baseline['calls'][0]['x']).hexdigest(), baseline['population_sha256'], 'Initial population mismatch')
            assert_equal(hashlib.sha256(baseline['calls'][0]['rng_before']).hexdigest(), baseline['solver_rng_sha256'], 'Initial RNG mismatch')
            assert_equal(len(baseline['records']), 45, 'Incomplete late-stage coverage')
            with gzip.open(folder / 'CASES.json.gz', 'rt') as stream:
                episode_cases = json.load(stream)
            expected_steps = {row['decision_step'] for row in baseline['records'] if any(action['changes_original'] for action in row['actions'].values())}
            assert_equal({row['decision_step'] for row in episode_cases}, expected_steps, 'Conflict retention mismatch')
            branch_paths = set()
            episode_fe = 300
            for case in episode_cases:
                record = baseline['records'][case['decision_step']]
                for key in record:
                    assert_equal(case[key], record[key], 'Historical pre-action evidence changed')
                assert_equal(case['function'], function['fid'], 'Function identity mismatch')
                assert_equal(case['seed_index'], seed, 'Seed identity mismatch')
                branches = {}
                for name, action in case['actions'].items():
                    chosen = action['selected_index']
                    if action['changes_original']:
                        path = folder / f"d{case['decision_step']:02d}_i{chosen}.pkl.gz"
                        if chosen not in branches:
                            branch = run.load_pickle(path)
                            assert_equal(branch['applied'], 1, 'Repeated counterfactual action')
                            assert_equal(branch['context_sha256'], case['context_sha256'], 'Counterfactual context mismatch')
                            assert_equal(branch['stop_nfe'], case['horizon_nfe']['10'], 'Incorrect rollout horizon')
                            prefix = branch['prefix_nfe']
                            if not np.array_equal(branch['values'][:, :prefix], baseline['values'][:, :prefix]):
                                raise RuntimeError('Counterfactual prefix mismatch')
                            assert_equal(branch['values'].shape[1], branch['stop_nfe'], 'Counterfactual length mismatch')
                            assert_equal(branch['native_fe'] + branch['cached_fe'], branch['stop_nfe'], 'Branch evaluation accounting mismatch')
                            paid = sum(int(np.prod(call['shape'])) // 10 for call in branch['calls'])
                            assert_equal(paid, branch['native_fe'], 'Native call count mismatch')
                            if paid > 20:
                                raise RuntimeError('Branch cap exceeded')
                            if path not in branch_paths:
                                episode_fe += paid
                                branch_count += 1
                                branch_paths.add(path)
                            branches[chosen] = branch
                    branch = branches.get(chosen)
                    for horizon in ('1', '5', '10'):
                        stop = min(300, case['nfe_before'] + 2 * int(horizon))
                        assert_equal(case['horizon_nfe'][horizon], stop, 'NFE alignment mismatch')
                        reference = float(baseline['values'][:, :stop].min())
                        best = reference if branch is None else float(branch['values'][:, :stop].min())
                        advantage = float(np.clip((reference - best) / case['scale'], -20, 20))
                        assert_equal(case['horizon_best_values'][horizon][name], best, 'Stored branch best differs')
                        assert_equal(case['horizon_advantages'][horizon][name], advantage, 'Multistep label differs from raw objectives')
                assert_equal(case['advantages'], case['horizon_advantages']['5'], 'Primary target is not five-round gain')
            assert_equal(episode_fe, receipt['native_fe'], 'Episode evaluation total mismatch')
            assert_equal(len(branch_paths), receipt['distinct_branches'], 'Branch deduplication mismatch')
            receipts.append(receipt)
            cases.extend(episode_cases)
    assert_equal(len(receipts), 288, 'Not all eight seeds completed')
    memory = json.loads((EXP / 'offline_memory.json').read_text())
    memory_ids = {case['id'] for case in memory['cases']}
    expected_ids = {case['id'] for case in cases if case['function'] not in run.legacy.HELDOUT}
    assert_equal(memory_ids, expected_ids, 'Calibration/test labels entered memory')
    assert_equal(memory['collection_binding'], binding, 'Memory is bound to different code')
    if memory['test_data_used'] or {case['function'] for case in memory['cases']} & run.legacy.HELDOUT:
        raise RuntimeError('Information boundary violated')
    heldout = [case for case in cases if case['function'] in run.legacy.HELDOUT]
    with gzip.open(EXP / 'CALIBRATION_CASES.json.gz', 'rt') as stream:
        assert_equal({case['id'] for case in json.load(stream)}, {case['id'] for case in heldout}, 'Calibration export differs')
    for key, actions in memory['context_validation'].items():
        for name, value in actions.items():
            changed = [case for case in heldout if case['context_key'] == key and case['actions'][name]['changes_original']]
            episodes = {(case['function'], case['seed_index']) for case in changed}
            functions = {case['function'] for case in changed}
            seeds = {case['seed_index'] for case in changed}
            primary, secondary = horizon_mean(changed, name, 5), horizon_mean(changed, name, 10)
            supported = len(changed) >= 8 and len(episodes) >= 6 and len(functions) >= 2 and len(seeds) >= 3
            supported = bool(supported and primary > 0 and secondary >= 0)
            assert_equal(value['supported'], supported, 'Calibration gate differs from preregistration')
            assert_equal(value['balanced_horizon_advantages']['5'], primary, 'Calibration aggregation mismatch')
            assert_equal(value['balanced_horizon_advantages']['10'], secondary, 'Calibration aggregation mismatch')
    comparison = {}
    for name in run.CONFIGS:
        changed = [case for case in cases if case['actions'][name]['changes_original']]
        values = {horizon: np.asarray([case['horizon_advantages'][horizon][name] for case in changed]) for horizon in ('1', '5', '10')}
        comparison[name] = dict(changed_named_actions=len(changed),
            actual_point_changes=sum(case['actions'][name]['changes_point'] for case in changed),
            signs={horizon: dict(positive=int((value > 0).sum()), zero=int((value == 0).sum()), negative=int((value < 0).sum())) for horizon, value in values.items()},
            no_immediate_gain_but_positive_at_5=int(((values['1'] <= 0) & (values['5'] > 0)).sum()),
            immediate_gain_lost_by_5=int(((values['1'] > 0) & (values['5'] <= 0)).sum()),
            positive_at_5_but_negative_at_10=int(((values['5'] > 0) & (values['10'] < 0)).sum()),
            balanced_mean={horizon: horizon_mean(changed, name, horizon) for horizon in ('1', '5', '10')})
    audit = dict(status='all_raw_labels_verified', baseline_episodes=len(receipts),
        baseline_states=sum(row['baseline_states'] for row in receipts), retained_cases=len(cases),
        conflict_cases=sum(case['fusion_features'][0] == 1 for case in cases),
        memory_cases=len(memory_ids), calibration_cases=len(heldout), distinct_counterfactual_branches=branch_count,
        collection_native_fe=sum(receipt['native_fe'] for receipt in receipts),
        calibration_and_memory_functions_disjoint=True, fresh_seed_indices=list(range(8)),
        cases_with_truncated_5_round_horizon=sum(case['nfe_before'] + 10 > 300 for case in cases),
        cases_with_truncated_10_round_horizon=sum(case['nfe_before'] + 20 > 300 for case in cases),
        target_comparison=comparison)
    previous = json.loads((ROOT / 'experiments/r03_evidence_fusion/offline_memory.json').read_text())
    with gzip.open(ROOT / 'experiments/r03_evidence_fusion/HELDOUT_CASES.json.gz', 'rt') as stream:
        previous_cases = previous['cases'] + json.load(stream)
    audit['coverage_vs_r03'] = dict(r03_baseline_states=len(previous_cases),
        r03_informative_cases=sum(any(action['changes_original'] for action in case['actions'].values()) for case in previous_cases),
        r04_baseline_states=audit['baseline_states'], r04_informative_cases=len(cases),
        qualification='Different fixed seeds and labels; informative means action-index disagreement, not guaranteed positive utility')
    save('LABEL_AUDIT.json', audit)
    return audit, memory


def audit_confirmation(memory):
    rows = json.loads((EXP / 'CONFIRMATORY_RESULTS.json').read_text())
    expected = {(name, seed) for name in run.legacy.HELDOUT for seed in (100, 101)}
    assert_equal({(row['function'], row['seed_index']) for row in rows}, expected, 'Incomplete independent-seed tests')
    summaries = []
    memory_by_id = {case['id']: case for case in memory['cases']}
    applicability = dict(decisions=0, candidate_change_opportunities=0,
        states_with_calibrated_candidate_change=0, states_with_locally_eligible_action=0,
        scheduled_states_with_locally_eligible_action=0,
        calibrated_action_attempts=0, blocked_fewer_than_two_changed_retrieved_cases=0,
        blocked_nonpositive_retrieved_five_round_mean=0,
        blocked_negative_retrieved_ten_round_mean=0,
        local_support_passed=0)
    for row in rows:
        folder = EXP / 'confirmatory' / f"s{row['seed_index']}_{row['function']}"
        arrays = np.load(folder / 'VALUES.npz')
        if arrays['baseline'].shape != (1, 300) or arrays['retrieval'].shape != (1, 300):
            raise RuntimeError('Confirmatory budget differs')
        assert_equal(row['baseline_best'], float(arrays['baseline'].min()), 'Baseline result mismatch')
        assert_equal(row['retrieval_best'], float(arrays['retrieval'].min()), 'Agent result mismatch')
        assert_equal(row['improved'], row['retrieval_best'] < row['baseline_best'], 'Win classification mismatch')
        assert_equal(row['worsened'], row['retrieval_best'] > row['baseline_best'], 'Loss classification mismatch')
        assert_equal(row['agent_stats']['requests'], 0, 'Unexpected API calls')
        assert_equal(row['memory_sha256'], run.legacy.digest((EXP / 'offline_memory.json').read_bytes()), 'Confirmation memory changed')
        with gzip.open(folder / 'DECISIONS.json.gz', 'rt') as stream:
            decisions = json.load(stream)
        assert_equal(len(decisions), 45, 'Missing confirmation logs')
        changed = sum(int(item['selected'][0][0] != item['original'][0][0]) for item in decisions)
        assert_equal(changed, row['agent_stats']['changed'], 'Changed actions miscounted')
        if any(item['configs'] != ['release'] for item in decisions if item['step'] % 5):
            raise RuntimeError('Agent action persisted beyond one round')
        for item in decisions:
            current = item['current'][0]
            applicability['decisions'] += 1
            applicability['candidate_change_opportunities'] += int(any(action['changes_original'] for action in current['actions'].values()))
            eligible = len(item['eligible_configs'][0]) > 1
            applicability['states_with_locally_eligible_action'] += int(eligible)
            applicability['scheduled_states_with_locally_eligible_action'] += int(eligible and item['step'] % 5 == 0)
            neighbors = [memory_by_id[identity] for identity in item['retrieved_case_ids'][0]]
            context = memory['context_validation'][current['context_key']]
            has_calibrated_action = False
            for name, action in current['actions'].items():
                if not action['changes_original'] or not context[name]['supported']:
                    continue
                has_calibrated_action = True
                applicability['calibrated_action_attempts'] += 1
                local = [case for case in neighbors if case['actions'][name]['changes_original']]
                if len(local) < 2:
                    applicability['blocked_fewer_than_two_changed_retrieved_cases'] += 1
                elif np.mean([case['advantages'][name] for case in local]) <= 0:
                    applicability['blocked_nonpositive_retrieved_five_round_mean'] += 1
                elif np.mean([case['horizon_advantages']['10'][name] for case in local]) < 0:
                    applicability['blocked_negative_retrieved_ten_round_mean'] += 1
                else:
                    applicability['local_support_passed'] += 1
            applicability['states_with_calibrated_candidate_change'] += int(has_calibrated_action)
    for name in sorted(run.legacy.HELDOUT):
        selected = [row for row in rows if row['function'] == name]
        baseline = np.asarray([row['baseline_best'] for row in selected])
        agent = np.asarray([row['retrieval_best'] for row in selected])
        summaries.append(dict(function=name, n=len(selected), baseline_mean=float(baseline.mean()),
            baseline_std=float(baseline.std()), retrieval_mean=float(agent.mean()), retrieval_std=float(agent.std()),
            wins=sum(row['improved'] for row in selected), ties=sum(row['unchanged'] for row in selected),
            losses=sum(row['worsened'] for row in selected),
            changed_points=sum(row['agent_stats']['changed_points'] for row in selected)))
    with (EXP / 'FUNCTION_COMPARISON.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    save('FUNCTION_COMPARISON.json', summaries)
    save('APPLICABILITY_AUDIT.json', applicability)
    return rows, summaries


def archives():
    members, archives = [], []
    groups = [(f'COLLECTION_SEED{seed:02d}.tar.xz', sorted((EXP / 'collection').glob(f's{seed:02d}_*'))) for seed in range(8)]
    groups.append(('CONFIRMATORY_AND_SMOKE.tar.xz', [EXP / 'confirmatory', EXP / 'verification']))
    previous_path = EXP / 'ARCHIVE_MANIFEST.json'
    if previous_path.exists():
        previous = json.loads(previous_path.read_text())
        current_paths = {str(path.relative_to(EXP)) for _, folders in groups for folder in folders for path in folder.rglob('*') if path.is_file()}
        assert_equal(current_paths, {entry['path'] for entry in previous['members']}, 'Raw archive inventory changed')
        for entry in previous['members']:
            assert_equal(hashlib.sha256((EXP / entry['path']).read_bytes()).hexdigest(), entry['sha256'], 'Raw archived evidence changed')
        for entry in previous['archives']:
            assert_equal(hashlib.sha256((EXP / entry['path']).read_bytes()).hexdigest(), entry['sha256'], 'Archive shard changed')
        return previous['archives'], previous['members']
    for name, folders in groups:
        files = sorted(path for folder in folders for path in folder.rglob('*') if path.is_file())
        target = EXP / name
        with tarfile.open(target, 'w:xz', preset=3) as archive:
            for path in files:
                relative = str(path.relative_to(EXP))
                archive.add(path, arcname=relative, recursive=False)
                members.append(dict(archive=name, path=relative, bytes=path.stat().st_size,
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        if target.stat().st_size >= 95_000_000:
            raise RuntimeError('Archive shard exceeds safe remote-file size')
        archives.append(dict(path=name, bytes=target.stat().st_size, members=len(files),
            sha256=hashlib.sha256(target.read_bytes()).hexdigest()))
    save('ARCHIVE_MANIFEST.json', dict(archives=archives, members=members,
        raw_local_files_retained=True, archive_exclusion_is_not_deletion=True))
    return archives, members


def main():
    if subprocess.check_output(['git', '-C', str(ROOT), 'diff', '76498f7', '--',
        'experiments/r02_fusion_agent', 'experiments/r03_evidence_fusion']):
        raise RuntimeError('Prior frozen evidence changed')
    audit, memory = audit_collection()
    rows, comparison = audit_confirmation(memory)
    contracts = json.loads((EXP / 'CONTRACT_VERIFICATION.json').read_text())
    assert_equal(contracts['status'], 'passed', 'Contract tests failed')
    supported = [dict(context=context, config=name, **value) for context, actions in memory['context_validation'].items()
        for name, value in actions.items() if value['supported']]
    result = dict(status='complete', baseline_round='R03 76498f7', training_functions=36,
        seeds_per_function=8, baseline_episodes=audit['baseline_episodes'],
        baseline_states=audit['baseline_states'], retained_informative_cases=audit['retained_cases'],
        memory_cases=audit['memory_cases'], calibration_cases=audit['calibration_cases'],
        distinct_counterfactual_branches=audit['distinct_counterfactual_branches'],
        primary_horizon_rounds=5, secondary_horizon_rounds=[1, 10], action_duration_rounds=1,
        supported_context_actions=supported, confirmation_episodes=len(rows),
        confirmatory_wins=sum(row['improved'] for row in rows), confirmatory_ties=sum(row['unchanged'] for row in rows),
        confirmatory_losses=sum(row['worsened'] for row in rows),
        confirmatory_changed_points=sum(row['agent_stats']['changed_points'] for row in rows),
        live_api_calls=0, benchmark_test_evaluations=0,
        collection_training_fe=audit['collection_native_fe'], smoke_training_fe=contracts['smoke_native_training_fe'],
        confirmatory_training_fe=sum(row['native_fe'] for row in rows),
        total_new_training_fe=audit['collection_native_fe'] + contracts['smoke_native_training_fe'] + sum(row['native_fe'] for row in rows),
        functions=comparison, label_target_comparison=audit['target_comparison'],
        coverage_vs_r03=audit['coverage_vs_r03'],
        applicability=json.loads((EXP / 'APPLICABILITY_AUDIT.json').read_text()),
        claims=dict(multiseed_collection_complete=True, multistep_labels_verified_against_raw_objectives=True,
            actual_live_llm_efficacy_tested=False, general_benchmark_improvement_established=False,
            fresh_confirmation_seeds=True, unseen_confirmation_function_families=False),
        limitations=run.protocol()['limitations'])
    if result['total_new_training_fe'] > 650000:
        raise RuntimeError('Total resource cap exceeded')
    save('FINAL_RESULT.json', result)
    sources = list(run.SOURCES) + ['scripts/verify_r04.py', 'scripts/finalize_r04.py', 'scripts/restore_r04.py']
    save('CODE_BINDING.json', {name: run.legacy.digest((ROOT / name).read_bytes()) for name in sources})
    archive_files, members = archives()
    public_files = sorted(path for path in EXP.iterdir() if path.is_file() and path.name not in {'MANIFEST.json', 'COMPLETE.json'})
    manifest = [dict(path=path.name, bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for path in public_files]
    save('MANIFEST.json', manifest)
    save('COMPLETE.json', dict(status='verified', public_artifacts=len(manifest), raw_archive_members=len(members),
        archive_shards=len(archive_files), manifest_sha256=run.legacy.digest((EXP / 'MANIFEST.json').read_bytes())))
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
