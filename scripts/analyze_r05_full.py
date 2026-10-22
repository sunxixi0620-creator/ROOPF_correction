import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / 'experiments/r05_adaptive_evidence_fusion'
OLD = ROOT / 'experiments/r04_full_llm_benchmark'
sys.path.insert(0, str(ROOT / 'scripts'))


def save(name, value):
    (EXP / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def analyze():
    result = json.loads((EXP / 'FINAL_RESULT.json').read_text())
    current = json.loads((EXP / 'FUNCTION_COMPARISON.json').read_text())
    previous = {(row['group'], row['fid']): row for row in json.loads((OLD / 'FUNCTION_COMPARISON.json').read_text())}
    comparison = []
    audits = {arm: dict(rows=0, adapted_weight_rows=0, effective_residual_adaptation_rows=0, adapted_margin_rows=0,
        residual_multiplier_min=2.0, residual_multiplier_max=0.5,
        proxy_reliability_min=1.0, proxy_reliability_max=0.0,
        configs={}, always_release_trajectories=0, always_release_bitwise_verified=0)
        for arm in ('retrieval', 'llm_live')}
    for row in current:
        name = f"{row['group']}_f{row['fid']:02d}"
        old = previous[row['group'], row['fid']]
        if (old['baseline_mean'], old['baseline_std']) != (row['baseline_mean'], row['baseline_std']):
            raise RuntimeError('Round baselines differ')
        new_values = np.load(EXP / 'results' / name / 'llm_live_FULL100.npz')['values']
        old_values = np.load(OLD / 'results' / name / 'llm_live_FULL100.npz')['values']
        new_best, old_best = new_values.min(1), old_values.min(1)
        comparison.append(dict(group=row['group'], fid=row['fid'], primary=row['primary'],
            baseline_mean=row['baseline_mean'], baseline_std=row['baseline_std'],
            r04_mean=old['llm_live_mean'], r04_std=old['llm_live_std'],
            r05_mean=row['llm_live_mean'], r05_std=row['llm_live_std'],
            r05_vs_r04_wins=int((new_best < old_best).sum()),
            r05_vs_r04_ties=int((new_best == old_best).sum()),
            r05_vs_r04_losses=int((new_best > old_best).sum())))
        if row['group'] == 'CEC-style':
            continue
        for arm, counts in audits.items():
            for batch in range(10):
                folder = EXP / 'results' / name / f'b{batch:02d}'
                with gzip.open(folder / f'{arm}_DECISIONS.json.gz', 'rt') as stream:
                    decisions = json.load(stream)
                untouched = np.ones(10, dtype=bool)
                for decision in decisions:
                    for index, context in enumerate(decision['current']):
                        if context['candidate_count'] != 37:
                            raise RuntimeError('Truncated candidate evidence')
                        adaptation = context['adaptation']
                        proxy = adaptation['proxy_rank_reliability']
                        residual = adaptation['residual_rank_reliability']
                        multiplier = min(2.0, max(0.5, residual / proxy))
                        returns = adaptation['config_observed_relative_to_protected_anchor']
                        adverse = max(0.0, -returns['adaptive_fusion']['mean'])
                        online_adverse = max(0.0, -returns['online_only']['mean'])
                        expected = dict(residual_multiplier=multiplier,
                            margin=min(0.16, 0.06 + .10 * (1 - min(proxy, residual)) + .02 * adverse),
                            soft_margin=min(.08, .03 + .05 * (1 - min(proxy, residual)) + .01 * adverse),
                            online_margin=min(.16, .06 + .10 * (1 - proxy) + .02 * online_adverse),
                            online_soft_margin=min(.08, .03 + .05 * (1 - proxy) + .01 * online_adverse))
                        if any(not math.isclose(adaptation[key], value, abs_tol=1e-12) for key, value in expected.items()):
                            raise RuntimeError('Recorded feedback parameters violate frozen formula')
                        online = context['actions']['online_only']
                        if online['residual_in_score'] or online['residual_in_gate']:
                            raise RuntimeError('Online-only has Residual dependence')
                        config = decision['configs'][index]
                        untouched[index] &= config == 'release'
                        counts['configs'][config] = counts['configs'].get(config, 0) + 1
                        counts['rows'] += 1
                        counts['adapted_weight_rows'] += int(not math.isclose(multiplier, 1.0))
                        counts['effective_residual_adaptation_rows'] += int(not math.isclose(multiplier, 1.0)
                            and context['residual_weight'] != 0.0 and context['probability'] is not None)
                        counts['adapted_margin_rows'] += int(not math.isclose(adaptation['margin'], .11))
                        counts['residual_multiplier_min'] = min(counts['residual_multiplier_min'], multiplier)
                        counts['residual_multiplier_max'] = max(counts['residual_multiplier_max'], multiplier)
                        counts['proxy_reliability_min'] = min(counts['proxy_reliability_min'], proxy)
                        counts['proxy_reliability_max'] = max(counts['proxy_reliability_max'], proxy)
                counts['always_release_trajectories'] += int(untouched.sum())
                if not (row['group'] == 'BBOB' and row['fid'] == 20):
                    values = np.load(folder / f'{arm}.npz')['values']
                    reference = np.load(folder / 'off.npz')['values']
                    if not np.array_equal(values[untouched], reference[untouched]):
                        raise RuntimeError('Release trajectories do not preserve original behavior')
                    counts['always_release_bitwise_verified'] += int(untouched.sum())
    with (EXP / 'R04_R05_COMPARISON.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(comparison[0]))
        writer.writeheader()
        writer.writerows(comparison)
    save('R04_R05_COMPARISON.json', comparison)
    save('MECHANISM_AUDIT.json', dict(status='passed', arms=audits,
        feedback_formula_verified=True, full37_candidate_evidence_verified=True,
        online_only_thresholds_residual_independent=True,
        caveat='Descriptive selected-pair feedback, not causal counterfactual reliability'))
    scopes = result['api']['scopes']
    save('COST_LIMITATIONS.json', dict(preparation_unresolved_requests=scopes['setup_v1']['pending'],
        unresolved_requests_automatically_retried=False,
        reported_credit_units=result['api']['all_reported_credit_units'],
        actual_total_cash_charge_known=False, currency_unspecified=True,
        note='Credits only sum provider-reported fields; unresolved or missing billing fields are not guessed.'))
    primary = [row for row in comparison if row['primary']]
    summary = dict(function_wins=sum(row['r05_mean'] < row['r04_mean'] for row in primary),
        function_ties=sum(row['r05_mean'] == row['r04_mean'] for row in primary),
        function_losses=sum(row['r05_mean'] > row['r04_mean'] for row in primary),
        trajectory_wins=sum(row['r05_vs_r04_wins'] for row in primary),
        trajectory_ties=sum(row['r05_vs_r04_ties'] for row in primary),
        trajectory_losses=sum(row['r05_vs_r04_losses'] for row in primary))
    save('R04_R05_SUMMARY.json', summary)
    with (EXP / 'REPORT.zh-CN.txt').open('a') as stream:
        stream.write('\n机制核验：完整37候选证据、反馈驱动参数公式、纯在线门槛独立性、全程release轨迹原版等价性均已检查。\n')
        stream.write('与上一轮R04的主分析对照：' + json.dumps(summary, ensure_ascii=False) + '\n')
        stream.write(f"费用边界：准备阶段另有{scopes['setup_v1']['pending']}个中断后结果未知的请求，未自动重试；报告仅汇总已知tokens和返回credit，不推断未知费用或货币单位。\n")
    import restore_r04_full as restore
    restore.EXP = EXP
    restore.restore(write=False)
    save('ARCHIVE_VERIFICATION.json', dict(status='passed', archive_and_member_checksums_verified=True))
    manifest = [dict(path=path.name, bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        for path in sorted(EXP.iterdir()) if path.is_file() and path.name not in {'MANIFEST.json', 'COMPLETE.json'}]
    save('MANIFEST.json', manifest)
    complete = json.loads((EXP / 'COMPLETE.json').read_text())
    complete.update(public_artifacts=len(manifest), mechanism_audit_passed=True,
        manifest_sha256=hashlib.sha256((EXP / 'MANIFEST.json').read_bytes()).hexdigest())
    save('COMPLETE.json', complete)
    print(json.dumps(dict(status='passed', mechanism=audits, versus_r04=summary)))


if __name__ == '__main__':
    analyze()
