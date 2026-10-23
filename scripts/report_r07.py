import csv
import gzip
import json
from pathlib import Path
import sys
import tarfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import experiment_r07_full as run


def main():
    identity = run.binding()
    arms = run.settings()['arms']
    rows, traces, totals = [], [], {arm: dict(native_fe=0, cached_fe=0, changed_points=0,
        anchor_variant_rounds=0, local_scaled_rounds=0, selected_adjusted_candidates=0) for arm in arms}
    api_inputs, reviews_consumed = [], 0
    for group, fid in run.legacy.TASKS:
        folder = run.EXP / 'results' / f'{group}_f{fid:02d}'
        receipt = json.loads((folder / 'COMPLETE.json').read_text())
        if receipt['binding'] != identity:
            raise RuntimeError('Result source binding differs')
        baseline = np.load(folder / 'off_FULL100.npz')['values'].min(1).astype(float)
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
                for key in ('changed_points', 'anchor_variant_rounds', 'local_scaled_rounds', 'selected_adjusted_candidates'):
                    totals[arm][key] += record['agent_stats'].get(key, 0)
                if arm == 'llm_live':
                    with gzip.open(part / f'{arm}_DECISIONS.json.gz', 'rt') as stream:
                        for decision in json.load(stream):
                            if decision['request_event']:
                                api_inputs.append(decision['api_trajectories'])
                                reviews_consumed += sum(bool(item['current']['reviews']) for item in decision['api_trajectories'])
    for name, records in (('FUNCTION_COMPARISON.csv', rows), ('TRAJECTORY_COMPARISON.csv', traces)):
        with (run.EXP / name).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    journal = [json.loads(path.read_text()) for path in (run.EXP / 'api_journal').glob('*.json')]
    valid = [record for record in journal if record['status'] == 'ok']
    known_inputs = [record['request']['messages'][-1]['content'] for record in valid]
    matched_inputs = sum(any(json.dumps(items) in content for content in known_inputs) for items in api_inputs)
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
        comparisons={})
    for arm in arms:
        selected = [row for row in rows if row['arm'] == arm and row['primary']]
        summary['comparisons'][arm] = dict(
            function_wins=sum(row['mean'] < row['baseline_mean'] for row in selected),
            function_ties=sum(row['mean'] == row['baseline_mean'] for row in selected),
            function_losses=sum(row['mean'] > row['baseline_mean'] for row in selected),
            trajectory_wins=sum(row['wins'] for row in selected),
            trajectory_ties=sum(row['ties'] for row in selected),
            trajectory_losses=sum(row['losses'] for row in selected))
    run.legacy.write_json(run.EXP / 'FINAL_RESULT.json', summary)
    lines = ['R07 联合生成实验结果', '', json.dumps(summary, ensure_ascii=False, indent=2), '',
        '本报告仅记录实际完成结果，不自动认定性能改善。',
        '候选评分为启发式，不是已校准的收益置信下界。',
        '所有配置均保留，未按验证性能筛选；无LLM、Anchor单侧、局部单侧与等候选数对照分别列出。']
    (run.EXP / 'REPORT.zh-CN.txt').write_text('\n'.join(lines) + '\n')
    archives = run.EXP / 'archives'
    archives.mkdir(exist_ok=True)
    for source in sorted((run.EXP / 'results').iterdir()):
        if source.is_dir():
            with tarfile.open(archives / (source.name + '.tar.xz'), 'w:xz') as archive:
                archive.add(source, arcname='results/' + source.name)
    manifest = {str(path.relative_to(run.EXP)): run.legacy.digest(path.read_bytes())
        for path in sorted(archives.glob('*.tar.xz'))}
    run.legacy.write_json(run.EXP / 'ARCHIVE_MANIFEST.json', manifest)
    print(json.dumps(dict(status=summary['status'], comparisons=summary['comparisons']), ensure_ascii=False))


if __name__ == '__main__':
    main()
