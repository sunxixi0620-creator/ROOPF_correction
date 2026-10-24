import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
import gzip
import json
from pathlib import Path
import sys
import tarfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import experiment_r09_full as run


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
            raise RuntimeError('R09 matched38 base differs from the frozen R07 matched38 trajectory')
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
                if arm == 'llm_live':
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
    valid = [record for record in journal if record['status'] == 'ok']
    valid_contexts = {record['decision_context_sha256'] for record in valid}
    from roopf.evidence_fusion_agent import request_context_digest
    matched_inputs = sum(request_context_digest(**items) in valid_contexts for items in api_inputs)
    usages = [record.get('response', {}).get('usage', {}) for record in journal]
    api = dict(attempts=len(journal), valid=len(valid), failures=len(journal) - len(valid),
        input_batches=len(api_inputs), valid_input_batches_matched=matched_inputs,
        review_rows_sent=reviews_consumed, usage_records=usages,
        prompt_tokens=sum(row.get('prompt_tokens', 0) for row in usages),
        completion_tokens=sum(row.get('completion_tokens', 0) for row in usages),
        finish_reasons=[record.get('response', {}).get('choices', [{}])[0].get('finish_reason') for record in journal],
        reported_models=sorted({str(record.get('response', {}).get('model')) for record in journal}),
        currency='not inferred')
    if len(journal) > run.settings()['max_new_api_attempts']:
        raise RuntimeError('API request cap exceeded')
    summary = dict(status='full_results_collected' if valid else 'full_execution_without_valid_live_api',
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
    lines = ['R09 生成策略与离线在线评分融合完整 LLM 实验结果', '', json.dumps(summary, ensure_ascii=False, indent=2), '',
        '本报告仅记录实际完成结果，不自动认定性能改善。',
        '候选评分为启发式，不是已校准的收益置信下界。',
        '所有配置均保留，未按验证性能筛选；无LLM、Anchor单侧、局部单侧与等候选数对照分别列出。']
    (run.EXP / 'REPORT.zh-CN.txt').write_text('\n'.join(lines) + '\n')
    archives = run.EXP / 'archives'
    manifest = archive_results(run.EXP / 'results', archives, archive_workers, archive_progress)
    run.legacy.write_json(run.EXP / 'ARCHIVE_MANIFEST.json', manifest)
    print(json.dumps(dict(status=summary['status'], comparisons=summary['comparisons']), ensure_ascii=False))


if __name__ == '__main__':
    main()
