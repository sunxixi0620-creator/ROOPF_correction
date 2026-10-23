import csv
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / 'experiments/r06_event_reviewed_fusion'
OLD = ROOT / 'experiments/r05_adaptive_evidence_fusion'
sys.path.insert(0, str(ROOT / 'scripts'))


def save(name, value):
    (EXP / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def main():
    result = json.loads((EXP / 'FINAL_RESULT.json').read_text())
    rows = json.loads((EXP / 'FUNCTION_COMPARISON.json').read_text())
    old_rows = {(row['group'], row['fid']): row for row in json.loads((OLD / 'FUNCTION_COMPARISON.json').read_text())}
    comparison, mechanism = [], {}
    for row in rows:
        name = f"{row['group']}_f{row['fid']:02d}"
        old = old_rows[row['group'], row['fid']]
        if (old['baseline_mean'], old['baseline_std']) != (row['baseline_mean'], row['baseline_std']):
            raise RuntimeError('Baseline identity differs across rounds')
        new_values = np.load(EXP / 'results' / name / 'llm_live_FULL100.npz')['values']
        old_values = np.load(OLD / 'results' / name / 'llm_live_FULL100.npz')['values']
        new_best, old_best = new_values.min(1), old_values.min(1)
        comparison.append(dict(group=row['group'], fid=row['fid'], primary=row['primary'],
            baseline_mean=row['baseline_mean'], baseline_std=row['baseline_std'],
            r05_mean=old['llm_live_mean'], r05_std=old['llm_live_std'],
            r06_mean=row['llm_live_mean'], r06_std=row['llm_live_std'],
            r06_vs_r05_wins=int((new_best < old_best).sum()),
            r06_vs_r05_ties=int((new_best == old_best).sum()),
            r06_vs_r05_losses=int((new_best > old_best).sum())))
        if row['group'] == 'CEC-style':
            continue
        for arm in ('retrieval', 'llm_live'):
            counts = mechanism.setdefault(arm, dict(rows=0, adapted_weight_rows=0,
                effective_residual_adaptation_rows=0, first_epochs=[], review_epochs=[],
                configs=Counter(), changes_by_step=Counter(), always_release_verified=0,
                first_nonrelease=0, review_nonrelease=0))
            for batch in range(10):
                folder = EXP / 'results' / name / f'b{batch:02d}'
                with gzip.open(folder / f'{arm}_DECISIONS.json.gz', 'rt') as stream:
                    logs = json.load(stream)
                untouched = np.ones(10, dtype=bool)
                for decision in logs:
                    if decision['request_kind'] == 'event':
                        counts['first_epochs'].append(decision['step'])
                    if decision['request_kind'] == 'review':
                        counts['review_epochs'].append(decision['step'])
                    for index, current in enumerate(decision['current']):
                        if current['candidate_count'] != 37:
                            raise RuntimeError('Missing full candidate pool')
                        config = decision['configs'][index]
                        counts['configs'][config] += 1
                        counts['rows'] += 1
                        changed_weight = not np.isclose(current['adaptation']['residual_multiplier'], 1.0)
                        counts['adapted_weight_rows'] += int(changed_weight)
                        counts['effective_residual_adaptation_rows'] += int(changed_weight and current['residual_weight'] != 0)
                        if config != 'release':
                            untouched[index] = False
                            counts['changes_by_step'][decision['step']] += 1
                            counts['first_nonrelease' if decision['request_kind'] == 'event' else 'review_nonrelease'] += 1
                if not (row['group'] == 'BBOB' and row['fid'] == 20):
                    actual = np.load(folder / f'{arm}.npz')['values']
                    baseline = np.load(folder / 'off.npz')['values']
                    if not np.array_equal(actual[untouched], baseline[untouched]):
                        raise RuntimeError('Always-release trajectories differ from baseline')
                    counts['always_release_verified'] += int(untouched.sum())
    save('MECHANISM_AUDIT.json', dict(status='passed', arms=mechanism,
        policy_memory_identity='current R06 full37',
        actual_feedback_and_incumbent_utility_audit=result['actual_api_input_causality']))
    with (EXP / 'R05_R06_COMPARISON.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(comparison[0]))
        writer.writeheader()
        writer.writerows(comparison)
    save('R05_R06_COMPARISON.json', comparison)
    primary = [row for row in comparison if row['primary']]
    summary = dict(function_wins=sum(row['r06_mean'] < row['r05_mean'] for row in primary),
        function_ties=sum(row['r06_mean'] == row['r05_mean'] for row in primary),
        function_losses=sum(row['r06_mean'] > row['r05_mean'] for row in primary),
        trajectory_wins=sum(row['r06_vs_r05_wins'] for row in primary),
        trajectory_ties=sum(row['r06_vs_r05_ties'] for row in primary),
        trajectory_losses=sum(row['r06_vs_r05_losses'] for row in primary))
    save('R05_R06_SUMMARY.json', summary)
    api_records = [json.loads(path.read_text()) for path in (EXP / 'api_journal').glob('*.json')]
    finishes = Counter((record.get('response', {}).get('choices') or [{}])[0].get('finish_reason', 'no_response') for record in api_records)
    decisions = Counter(name for record in api_records if record['status'] == 'ok' for name in record['config_ids'])
    old_api = json.loads((OLD / 'API_AUDIT.json').read_text())['scopes']['formal']
    api = result['api']['scopes']['formal']
    transport = dict(finish_reasons=dict(finishes), valid_trajectory_choices=dict(decisions),
        old_mean_prompt_tokens=old_api['prompt_tokens'] / old_api['attempts'],
        new_mean_prompt_tokens=api['prompt_tokens'] / api['attempts'],
        formal_valid_fraction=api['valid_fraction'],
        incomplete_outputs_executed=False, failed_requests_automatically_retried=False)
    save('TRANSPORT_COMPARISON.json', transport)
    with (EXP / 'REPORT.zh-CN.txt').open('a') as stream:
        stream.write('\n与R05主分析对照：' + json.dumps(summary, ensure_ascii=False) + '\n')
        stream.write('实际反馈闭环核验：' + json.dumps(result['actual_api_input_causality'], ensure_ascii=False) + '\n')
        stream.write('接口压缩与完整性：' + json.dumps(transport, ensure_ascii=False) + '\n')
        stream.write('训练目标评价次数另计入总原生评价；API成功与局部复核覆盖不等于稳定性能改善。\n')
    import restore_r06_full as restore
    restore.EXP = EXP
    restore.restore(write=False)
    save('ARCHIVE_VERIFICATION.json', dict(status='passed', checksums_verified=True))
    manifest = [dict(path=path.name, bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        for path in sorted(EXP.iterdir()) if path.is_file() and path.name not in {'MANIFEST.json', 'COMPLETE.json'}]
    save('MANIFEST.json', manifest)
    complete = json.loads((EXP / 'COMPLETE.json').read_text())
    complete.update(mechanism_audit_passed=True, public_artifacts=len(manifest),
        manifest_sha256=hashlib.sha256((EXP / 'MANIFEST.json').read_bytes()).hexdigest())
    save('COMPLETE.json', complete)
    print(json.dumps(dict(status='passed', versus_r05=summary, transport=transport)))


if __name__ == '__main__':
    main()
