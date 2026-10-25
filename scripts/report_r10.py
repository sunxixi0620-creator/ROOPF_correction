import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
import gzip
import json
from pathlib import Path
import sys
import tarfile
import collections
import hashlib
from decimal import Decimal

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import experiment_r10_full as run


def audit_api(current_run, write=False):
    from roopf.policy_api_r10 import validate_response, wire_bytes
    from roopf.current_evidence_agent import POLICY_ID, HISTORY_FEATURES, request_context_digest, digest
    journals, consumed, counts, coverage = {}, set(), collections.Counter(), []
    credits, credit_records = Decimal(0), 0
    per_arm = {arm: collections.Counter() for arm in current_run.settings()['arms'] if arm.startswith('llm_')}
    for path in (current_run.EXP / 'api_journal').glob('*.json'):
        entry = json.loads(path.read_text())
        assert entry['status'] != 'pending', 'Unknown live API outcome'
        assert hashlib.sha256(wire_bytes(entry['request'])).hexdigest() == path.stem
        payload = json.loads(entry['request']['messages'][-1]['content'].split('INPUT:\n', 1)[1])
        assert payload['policy_id'] == POLICY_ID
        assert payload['memory_sha256'] == digest(json.loads(current_run.MEMORY.read_text()))
        assert payload['execution_context_sha256'] == digest(entry['execution_context'])
        assert payload['decision_context_sha256'] == entry['decision_context_sha256']
        arm = entry['execution_context']['arm']
        assert arm in per_arm
        if arm == 'llm_no_offline':
            assert all(r['offline'] is None for r in payload['trajectories'])
        if arm == 'llm_no_online':
            assert all(r['measured_history'] is None and all(r['state'][i] is None for i in HISTORY_FEATURES) for r in payload['trajectories'])
        if entry['status'] in ('ok','partial'):
            decoded = validate_response(entry['response'], payload)
            assert all(decoded[k] == entry[k] for k in decoded)
        else:
            assert entry['config_ids'] == ['matched38'] * len(payload['trajectories'])
        assert entry['decision_context_sha256'] not in journals
        journals[entry['decision_context_sha256']] = (path.stem, entry)
        counts[entry['status']] += 1
        per_arm[arm]['attempts'] += 1
        per_arm[arm][entry['status']] += 1
        per_arm[arm]['valid_rows'] += len(entry.get('valid_trajectory_ids', []))
        per_arm[arm]['fallback_rows'] += len(entry.get('row_errors', {}))
        usage = entry.get('response',{}).get('usage',{})
        counts['prompt_tokens'] += usage.get('prompt_tokens',0)
        counts['completion_tokens'] += usage.get('completion_tokens',0)
        credit = usage.get('credit')
        if isinstance(credit,(int,float)):
            credits += Decimal(str(credit)); credit_records += 1
    assert sum(v['attempts'] for v in per_arm.values()) == 580 * len(per_arm), 'Incomplete full API windows'
    for group, fid in current_run.legacy.TASKS:
        for arm in per_arm:
            function_counts = collections.Counter()
            valid_trajectories = set()
            for batch in range(10):
                with gzip.open(current_run.EXP / 'results' / f'{group}_f{fid:02d}/b{batch:02d}/{arm}_DECISIONS.json.gz','rt') as stream:
                    records = json.load(stream)
                requests = [r for r in records if r['request_event']]
                assert len(requests) == (0 if group == 'CEC-style' else 2)
                for record in requests:
                    key = request_context_digest(record['api_trajectories'],record['execution_context'])
                    identity, entry = journals[key]
                    assert record['api_request_sha256'] == identity
                    assert record['execution_context'] == entry['execution_context']
                    assert record['configs'] == entry['config_ids']
                    assert record['applied_generation'] == ['matched38'] * 10
                    assert record['api_row_validation']['valid_trajectory_ids'] == entry['valid_trajectory_ids']
                    for i, t in enumerate(record['api_trajectories']):
                        assert record['current'][i]['generation'] == t['current']
                        assert record['decision_features'][i] == t['decision_features']
                        assert record['current_options'][i] == t['current_options']
                    continuation = records[record['step']+1:record['step']+5]
                    assert len(continuation) == 4
                    for row in continuation:
                        assert row['configs'] == entry['config_ids']
                        assert row['applied_generation'] == [name.split('@')[0] for name in entry['config_ids']]
                    valid_trajectories.update((batch,int(i)) for i in entry['valid_trajectory_ids'])
                    consumed.add(identity)
                    function_counts['attempts'] += 1
                    function_counts[entry['status']] += 1
                    function_counts['valid_rows'] += len(entry['valid_trajectory_ids'])
            coverage.append(dict(group=group,fid=fid,arm=arm,primary=not(group=='BBOB' and fid==20),
                counts=dict(function_counts),trajectories_with_valid_api=len(valid_trajectories),structured_no_intervention=group=='CEC-style'))
    assert consumed == {v[0] for v in journals.values()}, 'Unused or unmapped actual requests'
    assert counts['ok'] + counts['partial'] > 0, 'Full execution has no valid live responses'
    for arm, values in per_arm.items():
        assert values['valid_rows'] > 0, f'No real valid API decision for {arm}'
    result = dict(status='all_actual_requests_mapped_to_executed_decisions',attempts=len(journals),
        valid=counts['ok']+counts['partial'],fully_valid=counts['ok'],partial=counts['partial'],failures=counts['failed'],
        valid_rows=sum(v['valid_rows'] for v in per_arm.values()),fallback_rows=sum(v['fallback_rows'] for v in per_arm.values()),
        prompt_tokens=counts['prompt_tokens'],completion_tokens=counts['completion_tokens'],
        provider_credit_sum=str(credits),responses_with_credit=credit_records,currency='not inferred',
        per_arm={k:dict(v) for k,v in per_arm.items()},coverage=coverage,
        matched_request_identities=len(consumed),generation_delay_verified=True)
    if write:
        current_run.legacy.write_json(current_run.EXP / 'HARDWARE_API_EXECUTION_AUDIT.json',result)
    return result


def verify(current_run):
    binding = current_run.binding()
    rows = list(csv.DictReader((current_run.EXP/'FUNCTION_COMPARISON.csv').open()))
    lookup = {(r['group'],int(r['fid']),r['arm']):r for r in rows}
    count = 0
    for group,fid in current_run.legacy.TASKS:
        folder = current_run.EXP/'results'/f'{group}_f{fid:02d}'
        assert json.loads((folder/'COMPLETE.json').read_text())['binding'] == binding
        arrays = {a:np.load(folder/f'{a}_FULL100.npz')['values'] for a in current_run.settings()['arms']}
        assert np.array_equal(arrays['off'],current_run.legacy.original_values(group,fid))
        frozen = ROOT/'experiments/r07_hardware_accelerated_v1/results'/f'{group}_f{fid:02d}/matched38_FULL100.npz'
        assert np.array_equal(arrays['matched38'],np.load(frozen)['values'])
        for arm,data in arrays.items():
            assert data.shape == (100,300) and np.isfinite(data).all()
            final = data.min(1).astype(float); row=lookup[group,fid,arm]
            assert float(row['mean']) == float(final.mean()) and float(row['std']) == float(final.std(ddof=0))
            for a,prefix in [('off',''),('matched38','matched38_')]:
                ref=arrays[a].min(1).astype(float)
                assert tuple(int(row[prefix+k]) for k in ('wins','ties','losses')) == (int((final<ref).sum()),int((final==ref).sum()),int((final>ref).sum()))
        for batch in range(10):
            receipts=[json.loads((folder/f'b{batch:02d}/{a}.json').read_text()) for a in arrays]
            old=json.loads((frozen.parent/f'b{batch:02d}/matched38.json').read_text())
            for r in receipts:
                assert r['binding']==binding and r['native_fe']+r['cached_fe']==3000
                for key in ('paired_initial_sha256','paired_solver_rng_sha256'):
                    assert r[key]==old[key]
        count+=1
    previous=json.loads((current_run.EXP/'PRE_MIGRATION.json').read_text())
    for name,sha in previous['previous_records_and_sources_sha256'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==sha,name
    for name,sha in json.loads((current_run.EXP/'ARCHIVE_MANIFEST.json').read_text()).items():
        h=hashlib.sha256()
        with (current_run.EXP/name).open('rb') as stream:
            for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
        assert h.hexdigest()==sha,name
    audit=audit_api(current_run)
    result=dict(status='full_records_verified',functions=count,arms=len(arrays),trajectory_records=count*len(arrays)*100,
        evaluation_records=count*len(arrays)*30000,paired_batches=350,exact_original_functions=35,exact_matched38_functions=35,
        previous_scientific_records_unchanged=True,api=audit,primary_excludes_bbob20=True,cec_structured_lock_retained=True)
    current_run.legacy.write_json(current_run.EXP/'INDEPENDENT_VERIFICATION.json',result)
    return result


def archive_results(source_directory, archives, workers=1, progress=None):
    archives.mkdir(exist_ok=True)
    sources = sorted(path for path in source_directory.iterdir() if path.is_dir())

    def pack(source):
        target = archives / (source.name + '.tar.xz')
        with tarfile.open(target, 'w:xz') as archive:
            archive.add(source, arcname='results/' + source.name)
        return target

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(pack, source) for source in sources]
        for completed, future in enumerate(as_completed(futures), 1):
            target = future.result()
            if progress:
                progress(completed, len(sources), target.name)
    return {str(path.relative_to(archives.parent)): run.legacy.digest(path.read_bytes())
            for path in sorted(archives.glob('*.tar.xz'))}


def main(archive_workers=1, archive_progress=None):
    audit_api(run, write=True)
    identity = run.binding()
    arms = run.settings()['arms']
    rows, traces, totals = [], [], {arm: dict(native_fe=0, cached_fe=0, changed_points=0,
        anchor_variant_rounds=0, local_scaled_rounds=0, selected_adjusted_candidates=0, fusion_changed_points=0, nondefault_fusion_rounds=0) for arm in arms}
    api_inputs, reviews_consumed = [], 0
    for group, fid in run.legacy.TASKS:
        folder = run.EXP / 'results' / f'{group}_f{fid:02d}'
        receipt = json.loads((folder / 'COMPLETE.json').read_text())
        if receipt['binding'] != identity:
            raise RuntimeError('Result source binding differs')
        baseline = np.load(folder / 'off_FULL100.npz')['values'].min(1).astype(float)
        matched_values = np.load(folder / 'matched38_FULL100.npz')['values']
        frozen_path = run.ROOT / 'experiments/r07_hardware_accelerated_v1/results' / f'{group}_f{fid:02d}' / 'matched38_FULL100.npz'
        if not np.array_equal(matched_values, np.load(frozen_path)['values']):
            raise RuntimeError('R10 matched38 base differs from the frozen R07 matched38 trajectory')
        matched = matched_values.min(1).astype(float)
        included = not (group == 'BBOB' and fid == 20)
        for arm in arms:
            data = np.load(folder / f'{arm}_FULL100.npz')['values']
            if data.shape != (100, 300) or not np.isfinite(data).all():
                raise RuntimeError('Incomplete trajectory file')
            values = data.min(1).astype(float)
            row = dict(group=group, fid=fid, arm=arm, primary=included,
                mean=float(values.mean()), std=float(values.std()), baseline_mean=float(baseline.mean()),
                baseline_std=float(baseline.std()), wins=int((values < baseline).sum()),
                ties=int((values == baseline).sum()), losses=int((values > baseline).sum()))
            row.update(matched38_mean=float(matched.mean()), matched38_std=float(matched.std()),
                matched38_wins=int((values < matched).sum()), matched38_ties=int((values == matched).sum()),
                matched38_losses=int((values > matched).sum()))
            rows.append(row)
            for index, value in enumerate(values):
                traces.append(dict(group=group, fid=fid, arm=arm, trajectory=index, primary=included,
                    baseline=float(baseline[index]), value=float(value), difference=float(value - baseline[index])))
            for batch in range(10):
                part = folder / f'b{batch:02d}'
                record = json.loads((part / f'{arm}.json').read_text())
                if record['binding'] != identity:
                    raise RuntimeError('Batch source binding differs')
                for key in ('native_fe', 'cached_fe'):
                    totals[arm][key] += record[key]
                for key in ('changed_points', 'anchor_variant_rounds', 'local_scaled_rounds', 'selected_adjusted_candidates', 'fusion_changed_points', 'nondefault_fusion_rounds'):
                    totals[arm][key] += record['agent_stats'].get(key, 0)
                if arm.startswith('llm_'):
                    with gzip.open(part / f'{arm}_DECISIONS.json.gz', 'rt') as stream:
                        for decision in json.load(stream):
                            if decision['request_event']:
                                api_inputs.append(dict(trajectories=decision['api_trajectories'],
                                    execution_context=decision['execution_context']))
                                reviews_consumed += sum(bool(item['current']['reviews']) for item in decision['api_trajectories'])
    for name, records in (('FUNCTION_COMPARISON.csv', rows), ('TRAJECTORY_COMPARISON.csv', traces)):
        with (run.EXP / name).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    journal = [json.loads(path.read_text()) for path in (run.EXP / 'api_journal').glob('*.json')]
    valid = [record for record in journal if record['status'] in ('ok', 'partial')]
    valid_contexts = {record['decision_context_sha256'] for record in valid}
    from roopf.evidence_fusion_agent import request_context_digest
    matched_inputs = sum(request_context_digest(**items) in valid_contexts for items in api_inputs)
    usages = [record.get('response', {}).get('usage', {}) for record in journal]
    api = dict(fully_valid=sum(r['status']=='ok' for r in journal), partial=sum(r['status']=='partial' for r in journal), valid_rows=sum(len(r.get('valid_trajectory_ids',[])) for r in journal), invalid_rows=sum(len(r.get('row_errors',{})) for r in journal), attempts=len(journal), valid=len(valid), failures=len(journal) - len(valid),
        input_batches=len(api_inputs), valid_input_batches_matched=matched_inputs,
        historical_review_rows_in_raw_logs=reviews_consumed, observed_progress5_sent_to_llm=False, usage_records=usages,
        prompt_tokens=sum(row.get('prompt_tokens', 0) for row in usages),
        completion_tokens=sum(row.get('completion_tokens', 0) for row in usages),
        finish_reasons=[record.get('response', {}).get('choices', [{}])[0].get('finish_reason') for record in journal],
        reported_models=sorted({str(record.get('response', {}).get('model')) for record in journal}),
        currency='not inferred')
    if len(journal) > run.settings()['max_new_api_attempts']:
        raise RuntimeError('API request cap exceeded')
    summary = dict(ablation_semantics=dict(llm_no_offline='remove case evidence and support gate; retain current predictions and factual history', llm_no_online='remove selected-observation history from LLM and retrieval features; numeric matched38 foundation unchanged'), status='full_results_collected' if valid else 'full_execution_without_valid_live_api',
        primary_excludes_bbob20=True, cec_structured_lock_retained=True, totals=totals, api=api,
        changed_points_definition='relative to original selector on current generated pool; not original trajectory coordinates',
        comparisons={}, comparisons_to_matched38={}, default_profile='matched38', failure_fallback='matched38',
        frozen_r07_matched38_all_evaluations_equal=True)
    for arm in arms:
        selected = [row for row in rows if row['arm'] == arm and row['primary']]
        summary['comparisons'][arm] = dict(
            function_wins=sum(row['mean'] < row['baseline_mean'] for row in selected),
            function_ties=sum(row['mean'] == row['baseline_mean'] for row in selected),
            function_losses=sum(row['mean'] > row['baseline_mean'] for row in selected),
            trajectory_wins=sum(row['wins'] for row in selected),
            trajectory_ties=sum(row['ties'] for row in selected),
            trajectory_losses=sum(row['losses'] for row in selected))
        summary['comparisons_to_matched38'][arm] = dict(
            function_wins=sum(row['mean'] < row['matched38_mean'] for row in selected),
            function_ties=sum(row['mean'] == row['matched38_mean'] for row in selected),
            function_losses=sum(row['mean'] > row['matched38_mean'] for row in selected),
            trajectory_wins=sum(row['matched38_wins'] for row in selected),
            trajectory_ties=sum(row['matched38_ties'] for row in selected),
            trajectory_losses=sum(row['matched38_losses'] for row in selected))
    run.legacy.write_json(run.EXP / 'FINAL_RESULT.json', summary)
    lines = ['R10 生成策略与离线在线评分融合完整 LLM 实验结果', '', json.dumps(summary, ensure_ascii=False, indent=2), '',
        '本报告仅记录实际完成结果，不自动认定性能改善。',
        '候选评分为启发式，不是已校准的收益置信下界。',
        '全部22动作保留；Full、去离线案例输入、去在线历史反馈均用真实API；不移除在线代理网络或冻结Residual。']
    (run.EXP / 'REPORT.zh-CN.txt').write_text('\n'.join(lines) + '\n')
    archives = run.EXP / 'archives'
    manifest = archive_results(run.EXP / 'results', archives, archive_workers, archive_progress)
    run.legacy.write_json(run.EXP / 'ARCHIVE_MANIFEST.json', manifest)
    print(json.dumps(dict(status=summary['status'], comparisons=summary['comparisons']), ensure_ascii=False))



