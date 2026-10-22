import csv
from decimal import Decimal
import gzip
import hashlib
import json
from pathlib import Path
import pickle
import sys
import tarfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import experiment_r05_full as run

EXP = run.EXP


def save(name, value):
    run.legacy.write_json(EXP / name, value)


def table(name, rows):
    with (EXP / name).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def signature(trajectories):
    rows = [dict(trajectory_id=row['trajectory_id'], current=row['current']) for row in trajectories]
    return hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def payload(record):
    content = record['request']['messages'][-1]['content']
    if content.startswith('{'):
        return json.loads(content)
    return json.loads(content.split('INPUT:\n', 1)[1].split('\nFINAL OUTPUT CONTRACT:', 1)[0])


def audit():
    identity = run.freeze()
    functions, trajectories, details, bindings = [], [], [], {}
    totals = {arm: dict(native_fe=0, cached_fe=0, changed_points=0, changed_indices=0,
        opportunities=0, requests=0, failures=0, nonrelease=0, decisions=0, decision_log_steps=0) for arm in run.ARMS}
    for group, fid in run.legacy.TASKS:
        name = f'{group}_f{fid:02d}'
        folder = EXP / 'results' / name
        complete = json.loads((folder / 'COMPLETE.json').read_text())
        if complete['binding'] != identity or complete['n'] != 100 or not complete['original_final_rng_bitwise_verified']:
            raise RuntimeError('Function did not complete full100 with matching code/RNG')
        baseline = run.legacy.original_values(group, fid)
        arrays = {}
        row = dict(group=group, fid=fid, primary=not (group == 'BBOB' and fid == 20), n=100,
            baseline_mean=float(baseline.min(1).astype(float).mean()), baseline_std=float(baseline.min(1).astype(float).std()))
        for arm in run.ARMS:
            arrays[arm] = np.load(folder / f'{arm}_FULL100.npz')['values']
            values = arrays[arm]
            if values.shape != (100, 300) or not np.isfinite(values).all():
                raise RuntimeError('Invalid complete output')
            parts = []
            for batch in range(10):
                batch_folder = folder / f'b{batch:02d}'
                receipt = json.loads((batch_folder / f'{arm}.json').read_text())
                if receipt['binding'] != identity or receipt['nfe'] != 300 or receipt['n'] != 10:
                    raise RuntimeError('Mixed or incomplete batch')
                part = np.load(batch_folder / f'{arm}.npz')['values']
                if part.shape != (10, 300) or receipt['native_fe'] + receipt['cached_fe'] != 3000:
                    raise RuntimeError('Batch budget mismatch')
                if float(part.min(1).astype(float).mean()) != receipt['mean'] or float(part.min(1).astype(float).std()) != receipt['std']:
                    raise RuntimeError('Batch numerical summary mismatch')
                with gzip.open(batch_folder / 'PAIRED_START.pkl.gz', 'rb') as stream:
                    paired = pickle.load(stream)
                if run.legacy.digest(paired['population'].tobytes()) != receipt['paired_initial_sha256'] or run.legacy.digest(paired['rng'].tobytes()) != receipt['paired_solver_rng_sha256']:
                    raise RuntimeError('Paired initial population or RNG mismatch')
                key = name, batch
                pair_id = receipt['paired_initial_sha256'], receipt['paired_solver_rng_sha256']
                if key in bindings and bindings[key] != pair_id:
                    raise RuntimeError('Arms not paired')
                bindings[key] = pair_id
                if batch == 9 and paired['original_after'].tobytes() != (run.legacy.SNAP / name / 'FULL_RNG.bin').read_bytes():
                    raise RuntimeError('Full100 final RNG mismatch')
                with gzip.open(batch_folder / f'{arm}_DECISIONS.json.gz', 'rt') as stream:
                    logs = json.load(stream)
                expected = 0 if arm == 'off' or group == 'CEC-style' else 45
                if len(logs) != expected:
                    raise RuntimeError('Incomplete semantic logs')
                changed, nonrelease = 0, 0
                for step, decision in enumerate(logs):
                    if step != decision['step'] or len(decision['current']) != 10:
                        raise RuntimeError('Decision batch/step mismatch')
                    for index, current in enumerate(decision['current']):
                        config = decision['configs'][index]
                        if config not in decision['eligible_configs'][index]:
                            raise RuntimeError('Unsupported executed action')
                        if decision['selected'][index][0] != current['actions'][config]['selected_index']:
                            raise RuntimeError('Selected candidate differs from configuration executor')
                        if step % 23 and config != 'release':
                            raise RuntimeError('Multi-step action duration changed')
                        changed += int(decision['selected'][index] != decision['original'][index])
                        nonrelease += int(config != 'release')
                    if arm == 'llm_live' and step % 23 == 0:
                        eligible = [dict(trajectory_id=str(index), current=current) for index, current in enumerate(decision['current'])
                            if len(decision['eligible_configs'][index]) > 1]
                        if eligible:
                            details.append(dict(scope='formal', signature=signature(eligible), task=name,
                                batch=batch, step=step, configs={str(index): decision['configs'][index] for index in range(10)}))
                stats = receipt['agent_stats']
                if changed != stats['changed'] or nonrelease != stats['nonrelease']:
                    raise RuntimeError('Decision behavior count mismatch')
                native_path = batch_folder / f'{arm}_NATIVE_CALLS.pkl.gz'
                if native_path.exists():
                    with gzip.open(native_path, 'rb') as stream:
                        calls = pickle.load(stream)
                    if sum(int(np.prod(call['shape'])) // 10 for call in calls) != receipt['native_fe']:
                        raise RuntimeError('Native calls do not match counted evaluations')
                elif receipt['native_fe']:
                    raise RuntimeError('Missing native objective payment journal')
                total = totals[arm]
                total['native_fe'] += receipt['native_fe']
                total['cached_fe'] += receipt['cached_fe']
                for target, source in [('changed_points', 'changed_points'), ('changed_indices', 'changed'),
                    ('opportunities', 'opportunities'), ('requests', 'requests'), ('failures', 'failures'),
                    ('nonrelease', 'nonrelease'), ('decisions', 'decisions')]:
                    total[target] += stats[source]
                total['decision_log_steps'] += len(logs)
                parts.append(part)
            if not np.array_equal(values, np.concatenate(parts)):
                raise RuntimeError('Full100 file differs from individual batches')
            if arm == 'off' and not np.array_equal(values, baseline):
                raise RuntimeError('Original baseline not bitwise identical for all100')
            best, reference = values.min(1).astype(float), baseline.min(1).astype(float)
            row.update({f'{arm}_mean': float(best.mean()), f'{arm}_std': float(best.std()),
                f'{arm}_trajectory_wins': int((best < reference).sum()),
                f'{arm}_trajectory_ties': int((best == reference).sum()),
                f'{arm}_trajectory_losses': int((best > reference).sum())})
        row['llm_absolute_gain'] = row['baseline_mean'] - row['llm_live_mean']
        row['llm_relative_gain_percent'] = None if abs(row['baseline_mean']) < 1e-12 else row['llm_absolute_gain'] / abs(row['baseline_mean']) * 100
        llm_best, retrieval_best = arrays['llm_live'].min(1), arrays['retrieval'].min(1)
        row['llm_vs_retrieval_trajectory_wins'] = int((llm_best < retrieval_best).sum())
        row['llm_vs_retrieval_trajectory_ties'] = int((llm_best == retrieval_best).sum())
        row['llm_vs_retrieval_trajectory_losses'] = int((llm_best > retrieval_best).sum())
        functions.append(row)
        for index in range(100):
            trajectories.append(dict(group=group, fid=fid, primary=row['primary'], trajectory=index,
                batch=index // 10, row=index % 10, baseline=float(baseline[index].min()),
                off=float(arrays['off'][index].min()), retrieval=float(arrays['retrieval'][index].min()),
                llm_live=float(arrays['llm_live'][index].min())))
    table('FUNCTION_COMPARISON.csv', functions)
    table('TRAJECTORY_COMPARISON.csv', trajectories)
    save('FUNCTION_COMPARISON.json', functions)
    return functions, trajectories, totals, details


def api_audit(details):
    lookup = {}
    for item in details:
        lookup.setdefault(item['signature'], []).append(item)
    rows, scopes = [], {}
    for scope, folder in [('setup_v1', EXP / 'setup_attempt_v1/api_journal'), ('formal', run.JOURNAL)]:
        count = dict(attempts=0, valid=0, failed=0, pending=0, prompt_tokens=0, completion_tokens=0,
            total_tokens=0, usage_missing=0, elapsed_seconds=0.0, provider_credit_units=Decimal('0'))
        for path in sorted(folder.glob('*.json')):
            record = json.loads(path.read_text())
            expected = hashlib.sha256(json.dumps(record['request']).encode()).hexdigest()
            if record['request_sha256'] != expected or path.stem != expected:
                raise RuntimeError('API request identity mismatch')
            request_path = folder / f'{path.stem}.request.payload'
            if request_path.exists() and hashlib.sha256(request_path.read_bytes()).hexdigest() != expected:
                raise RuntimeError('API sent bytes differ from journaled request')
            response = record.get('response', {})
            raw_path = folder / f'{path.stem}.request.payload.response.raw'
            if response and raw_path.exists() and json.loads(raw_path.read_text(encoding='utf-8-sig')) != response:
                raise RuntimeError('Raw provider response differs from saved API record')
            status = record['status']
            count['attempts'] += 1
            count['valid' if status == 'ok' else 'failed' if status == 'failed' else 'pending'] += 1
            count['elapsed_seconds'] += record.get('elapsed_seconds', 0)
            usage = response.get('usage') or {}
            if not usage:
                count['usage_missing'] += 1
            for key in ('prompt_tokens', 'completion_tokens', 'total_tokens'):
                count[key] += int(usage.get(key) or 0)
            if usage.get('credit') is not None:
                count['provider_credit_units'] += Decimal(str(usage['credit']))
            item = payload(record)
            locations = lookup.get(signature(item['trajectories']), []) if scope == 'formal' else []
            if scope == 'formal' and not locations:
                raise RuntimeError('Live API record has no matching executed decision')
            if status == 'ok':
                ids = record['config_ids']
                if len(ids) != len(item['trajectories']):
                    raise RuntimeError('API result cardinality mismatch')
                for selected, request in zip(ids, item['trajectories']):
                    if selected not in request['eligible_configs']:
                        raise RuntimeError('Provider decision unsupported')
                    if scope == 'formal' and any(location['configs'][request['trajectory_id']] != selected for location in locations):
                        raise RuntimeError('Executed selection differs from actual model response')
            rows.append(dict(scope=scope, request_sha256=expected, status=status,
                requested_model=record['request']['model'], provider_reported_model=response.get('model'),
                response_id=response.get('id'), finish_reason=(response.get('choices') or [{}])[0].get('finish_reason'),
                prompt_tokens=usage.get('prompt_tokens'), completion_tokens=usage.get('completion_tokens'),
                total_tokens=usage.get('total_tokens'), provider_credit_units=usage.get('credit'),
                elapsed_seconds=record.get('elapsed_seconds'), error_type=record.get('error_type'),
                error=record.get('error'), executed_locations=';'.join(f"{location['task']}:b{location['batch']:02d}:s{location['step']}" for location in locations)))
        count['provider_credit_units'] = float(count['provider_credit_units'])
        count['valid_fraction'] = count['valid'] / count['attempts'] if count['attempts'] else None
        scopes[scope] = count
    if not scopes['formal']['valid'] or scopes['formal']['pending']:
        raise RuntimeError('Cannot claim complete live-LLM acceptance without completed valid real API calls')
    table('API_REQUESTS.csv', rows)
    result = dict(scopes=scopes, provider_reported_models=sorted({row['provider_reported_model'] for row in rows if row['provider_reported_model']}),
        all_attempts=sum(value['attempts'] for value in scopes.values()),
        all_known_total_tokens=sum(value['total_tokens'] for value in scopes.values()),
        all_reported_credit_units=float(sum(Decimal(str(value['provider_credit_units'])) for value in scopes.values())),
        credit_unit_currency_unspecified=True, actual_cash_charge_not_inferred=True,
        preparation_failures_preserved=True, formal_provider_decisions_matched_to_executed_actions=True)
    save('API_AUDIT.json', result)
    return result


def archive():
    groups = [(f"RESULTS_{group.replace('-', '_')}_f{fid:02d}.tar.xz", [EXP / 'results' / f'{group}_f{fid:02d}']) for group, fid in run.legacy.TASKS]
    groups.append(('API_AND_SETUP.tar.xz', [run.JOURNAL, EXP / 'setup_attempt_v1']))
    previous = EXP / 'ARCHIVE_MANIFEST.json'
    if previous.exists():
        data = json.loads(previous.read_text())
        current = {str(path.relative_to(EXP)) for _, folders in groups for folder in folders for path in folder.rglob('*') if path.is_file()}
        if current != {row['path'] for row in data['members']}:
            raise RuntimeError('Raw inventory changed after archival')
        for row in data['members']:
            if hashlib.sha256((EXP / row['path']).read_bytes()).hexdigest() != row['sha256']:
                raise RuntimeError('Archived raw evidence changed')
        for row in data['archives']:
            if hashlib.sha256((EXP / row['path']).read_bytes()).hexdigest() != row['sha256']:
                raise RuntimeError('Archive shard changed')
        return data
    data = dict(archives=[], members=[], local_raw_files_retained=True)
    for name, folders in groups:
        target = EXP / name
        files = sorted(path for folder in folders for path in folder.rglob('*') if path.is_file())
        with tarfile.open(target, 'w:xz', preset=3) as bundle:
            for path in files:
                relative = str(path.relative_to(EXP))
                bundle.add(path, arcname=relative, recursive=False)
                data['members'].append(dict(archive=name, path=relative, bytes=path.stat().st_size,
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        if target.stat().st_size >= 95_000_000:
            raise RuntimeError('Archive exceeds safe remote shard size')
        data['archives'].append(dict(path=name, bytes=target.stat().st_size,
            sha256=hashlib.sha256(target.read_bytes()).hexdigest()))
    save('ARCHIVE_MANIFEST.json', data)
    return data


def outcomes(functions, arm, group=None, primary=True):
    selected = [row for row in functions if (not primary or row['primary']) and (group is None or row['group'] == group)]
    return dict(functions=len(selected),
        function_wins=sum(row[f'{arm}_mean'] < row['baseline_mean'] for row in selected),
        function_ties=sum(row[f'{arm}_mean'] == row['baseline_mean'] for row in selected),
        function_losses=sum(row[f'{arm}_mean'] > row['baseline_mean'] for row in selected),
        trajectory_wins=sum(row[f'{arm}_trajectory_wins'] for row in selected),
        trajectory_ties=sum(row[f'{arm}_trajectory_ties'] for row in selected),
        trajectory_losses=sum(row[f'{arm}_trajectory_losses'] for row in selected))


def audit_feedback_causality():
    requests = {}
    for path in run.JOURNAL.glob('*.json'):
        record = json.loads(path.read_text())
        item = payload(record)
        requests.setdefault(signature(item['trajectories']), []).append(item)
    matched = 0
    for group, fid in run.legacy.TASKS:
        for batch in range(10):
            path = EXP / 'results' / f'{group}_f{fid:02d}' / f'b{batch:02d}' / 'llm_live_DECISIONS.json.gz'
            with gzip.open(path, 'rt') as stream:
                logs = json.load(stream)
            for decision in logs:
                if decision['step'] % 23:
                    continue
                selected = [dict(trajectory_id=str(index), current=current)
                    for index, current in enumerate(decision['current']) if len(decision['eligible_configs'][index]) > 1]
                if not selected:
                    continue
                matches = requests.get(signature(selected), [])
                if not matches:
                    raise RuntimeError('Missing actual API request for executed epoch')
                for request in matches:
                    for row in request['trajectories']:
                        index = int(row['trajectory_id'])
                        if row['observed_feedback'] != decision['feedback'][index]:
                            raise RuntimeError('Feedback snapshot differs from actual sent input')
                        expected_history = min(decision['step'], 32)
                        if len(row['observed_feedback']['competing_slot_recent']) != expected_history:
                            raise RuntimeError('Feedback includes future or missing active-round observations')
                        if row['current']['candidate_count'] != 37:
                            raise RuntimeError('Full candidate pool missing')
                        matched += 1
    result = dict(status='passed', matched_trajectory_inputs=matched,
        immutable_pre_action_feedback=True, candidate_count=37,
        no_current_objective_in_inputs=True)
    save('API_INPUT_CAUSALITY.json', result)
    return result


def main():
    functions, trajectories, totals, details = audit()
    api = api_audit(details)
    input_causality = audit_feedback_causality()
    setup_native = sum(json.loads(path.read_text())['native_fe'] for path in (EXP / 'setup_attempt_v1/results').glob('*/b*/*.json'))
    by_group = {group: {arm: outcomes(functions, arm, group) for arm in ('retrieval', 'llm_live')} for group in ('BBOB', 'CEC-style', 'UAV')}
    primary_functions = [row for row in functions if row['primary']]
    llm_vs_retrieval = dict(function_wins=sum(row['llm_live_mean'] < row['retrieval_mean'] for row in primary_functions),
        function_ties=sum(row['llm_live_mean'] == row['retrieval_mean'] for row in primary_functions),
        function_losses=sum(row['llm_live_mean'] > row['retrieval_mean'] for row in primary_functions),
        trajectory_wins=sum(row['llm_vs_retrieval_trajectory_wins'] for row in primary_functions),
        trajectory_ties=sum(row['llm_vs_retrieval_trajectory_ties'] for row in primary_functions),
        trajectory_losses=sum(row['llm_vs_retrieval_trajectory_losses'] for row in primary_functions))
    result = dict(status='full_live_model_executed_and_verified', functions=35, trajectories_per_function=100,
        trajectories_per_arm=3500, nfe_per_trajectory=300, dimension=10,
        arms=list(run.ARMS), original_all100_bitwise_equivalent_for_all35=True,
        primary_excludes_bbob20=True, primary={arm: outcomes(functions, arm) for arm in ('retrieval', 'llm_live')},
        by_group=by_group, all35={arm: outcomes(functions, arm, primary=False) for arm in ('retrieval', 'llm_live')},
        behavior_and_evaluation_totals=totals, api=api,
        actual_api_input_causality=input_causality,
        primary_llm_vs_retrieval=llm_vs_retrieval,
        frozen_memory_prior_round_training_evaluations=json.loads((ROOT / 'experiments/r04_multiseed_horizon/FINAL_RESULT.json').read_text())['total_new_training_fe'],
        setup_native_evaluations=setup_native,
        total_new_native_evaluations=setup_native + sum(value['native_fe'] for value in totals.values()),
        virtual_evaluations_per_arm=1050000, formal_api_validity_90pct_gate=api['scopes']['formal']['valid_fraction'] >= .9,
        fees_include_preparation_failures=True, weights_and_memory_frozen=True,
        definitive_general_improvement_claim=False, benchmark_previously_inspected_during_development=True)
    if result['total_new_native_evaluations'] > run.settings()['max_new_native_objective_evaluations']:
        raise RuntimeError('Native evaluation cap exceeded')
    save('FINAL_RESULT.json', result)
    lines = ['完整 R05 自适应证据融合 + DeepSeek 基准结果',
        '35函数 × 每函数100条轨迹 × 每条300次评估；均值±总体标准差（ddof=0）；目标越小越好。',
        '所有3500条原始基准均已逐位验证；主分析排除原始批耦合随机 BBOB20，仍完整报告该函数。',
        f"正式真实API：{api['scopes']['formal']['attempts']}次，合法{api['scopes']['formal']['valid']}次，失败{api['scopes']['formal']['failed']}次。",
        f"准备阶段失败亦保留：{api['scopes']['setup_v1']['attempts']}次；全部已知tokens={api['all_known_total_tokens']}。",
        f"接口返回credit字段合计={api['all_reported_credit_units']}；未返回货币单位，不据此推断人民币或美元费用。",
        f"主分析LLM函数胜/平/负={result['primary']['llm_live']['function_wins']}/{result['primary']['llm_live']['function_ties']}/{result['primary']['llm_live']['function_losses']}。",
        f"LLM实际换点={totals['llm_live']['changed_points']}；LLM失败回退={totals['llm_live']['failures']}。", '',
        '函数 | 原始基准均值±标准差 | 确定性自适应控制均值±标准差 | 真实LLM均值±标准差 | LLM配对轨迹胜/平/负']
    for row in functions:
        lines.append(f"{row['group']}_f{row['fid']:02d} | {row['baseline_mean']:.10g} ± {row['baseline_std']:.10g} | "
            f"{row['retrieval_mean']:.10g} ± {row['retrieval_std']:.10g} | {row['llm_live_mean']:.10g} ± {row['llm_live_std']:.10g} | "
            f"{row['llm_live_trajectory_wins']}/{row['llm_live_trajectory_ties']}/{row['llm_live_trajectory_losses']}")
    lines.extend(['', '说明：真实API调用不等于性能改善。本报告为已用于开发的基准上的完整描述性结果，不是新任务盲测。',
        '完整36候选池开放；反馈更新融合权重和接受门槛；online_only同时移除Residual评分和接纳影响。CEC锁和首个保护Anchor保留。',
        '调用间隔23个活跃轮次，每配置只生效一轮。神经网络权重和离线案例固定，轨迹内反馈统计与融合参数持续更新。旧案例只作情境参考，不将旧动作收益迁移为新候选空间的反事实标签。',
        '目标值采用完整请求精确缓存复用，每条轨迹仍计300次预算；真实LLM决策并未由检索策略替代。',
        '反馈日志改为深拷贝；实际API请求必须与行动前决策快照逐项一致，禁止使用未来目标信息。多项同时修改不能分离各项的因果贡献。'])
    (EXP / 'REPORT.zh-CN.txt').write_text('\n'.join(lines) + '\n')
    bundle = archive()
    save('AUDIT_CODE_BINDING.json', {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        for name in ('AGENTS.md', 'scripts/finalize_r05_full.py', 'scripts/experiment_r05_full.py', 'scripts/verify_r05.py')})
    excluded = {'MANIFEST.json', 'COMPLETE.json'}
    manifest = [dict(path=path.name, bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        for path in sorted(EXP.iterdir()) if path.is_file() and path.name not in excluded]
    save('MANIFEST.json', manifest)
    save('COMPLETE.json', dict(status='verified', full35x100=True, actual_valid_live_api_calls=api['scopes']['formal']['valid'],
        public_artifacts=len(manifest), archived_members=len(bundle['members']),
        manifest_sha256=hashlib.sha256((EXP / 'MANIFEST.json').read_bytes()).hexdigest()))
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
