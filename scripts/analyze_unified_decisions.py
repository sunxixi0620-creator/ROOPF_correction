"""Prespecified early effects and descriptive paid-query intervention diagnostics."""
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from roopf.experiment_io import sha256, write_json


def development():
    out = ROOT/'docs/revision/unified_execution'
    data = pd.read_csv(out/'development.csv')
    pivot = data.pivot(index=['model_seed', 'fid', 'instance', 'population'],
                       columns='method', values='early_utility')
    comparisons = {}
    for a, b in [('full', 'anchor_only'), ('no_residual', 'anchor_only'),
                 ('full', 'late_full'), ('no_residual', 'late_no_residual')]:
        values = (pivot[a]-pivot[b]).to_numpy().reshape(3, 12, 3, 2, 4).mean(-1)
        rng = np.random.default_rng(20281129); samples = []
        for _ in range(5000):
            seeds = rng.integers(0, 3, 3); families = rng.integers(0, 12, 12)
            instances = rng.integers(0, 2, (12, 2))
            samples.append(np.mean([values[seeds][:, family, :, instances[j]].mean()
                                   for j, family in enumerate(families)]))
        comparisons[a+'_vs_'+b] = dict(mean=float(values.mean()),
            ci95=np.quantile(samples, [.025, .975]).tolist(),
            by_training_seed=values.mean((1, 2, 3)).tolist())
    write_json(out/'EARLY_EFFECTS.json', dict(nfe=210, comparisons=comparisons,
        input_sha256=sha256(out/'development.csv'), note='Prespecified auxiliary outcome; not new model selection'))
    print(json.dumps(comparisons, indent=2))


def paid_diagnostics(value):
    values, decisions = value['values'], value['decisions']
    before = np.minimum.accumulate(values)
    improvements = np.maximum(0, before[:-1]-values[1:])
    total = float(before[99]-before[-1])
    observed_gains = []
    bins = {'pre70': {'proposed': 0, 'accepted': 0, 'readiness_rejected': 0,
                     'margin_rejected': 0, 'anchor_ranked_first': 0},
            'post70': {'proposed': 0, 'accepted': 0, 'readiness_rejected': 0,
                      'margin_rejected': 0, 'anchor_ranked_first': 0}}
    for d in decisions:
        second = d['eval_before']+1
        if d['accepted']:
            # Attribution convention: evaluate first anchor, then competing slot.
            observed_gains.append(float(improvements[second-1]))
        row = bins['pre70' if d['eval_before'] < .7*len(values) else 'post70']
        row['proposed'] += int(d['proposed'] >= 2)
        row['accepted'] += int(d['accepted'])
        row['anchor_ranked_first'] += int(d['proposed'] < 2)
        row['readiness_rejected'] += int(d['proposed'] >= 2 and not d['accepted'] and not d['evidence_ready'])
        row['margin_rejected'] += int(d['proposed'] >= 2 and not d['accepted'] and d['evidence_ready'])
    gains = np.asarray(observed_gains)
    assert gains.sum() <= total+max(1e-8, abs(total)*1e-12)
    return dict(portfolio_improvement_events=int((gains > 0).sum()),
        observed_total_improvement=total, observed_portfolio_improvement=float(gains.sum()),
        observed_portfolio_share=float(gains.sum()/total) if total > 0 else None,
        top5_portfolio_share_of_total=float(np.sort(gains)[-5:].sum()/total) if total > 0 else None,
        **{f'{part}_{key}': number for part, values in bins.items() for key, number in values.items()})


def external(run, out):
    rows = []
    for path in sorted((run/'cases').glob('*.pt')):
        meta = json.loads(path.with_suffix('.json').read_text())
        if meta['case']['method'] not in ('no_residual', 'no_residual_score_only'):
            continue
        assert sha256(path) == meta['data_sha256']
        value = torch.load(path, weights_only=False, map_location='cpu')
        row = value['row']
        rows.append({k: row[k] for k in ('dimension', 'budget', 'fid', 'run_index', 'group', 'method')} | paid_diagnostics(value))
    data = pd.DataFrame(rows)
    assert len(data) == 2160
    data.to_csv(out/'decision_diagnostics.csv', index=False)
    data.groupby(['dimension', 'budget', 'method']).mean(numeric_only=True).to_csv(out/'decision_summary.csv')
    write_json(out/'DECISION_DIAGNOSTICS_METHOD.json', dict(
        interpretation='Observed sequential incumbent decreases; not causal share of final benefit versus anchor-only',
        convention='First anchor is evaluated before competing slot; only the latter receives portfolio attribution',
        zero_improvement_shares='Missing, not zero; included in counts and raw gains',
        extra_objective_queries=0, final_metric_source='Same archived performance trajectories'))


if __name__ == '__main__':
    development()
