"""Decision timing from already-paid development runs; no objective queries."""
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from roopf.experiment_io import sha256

run = ROOT/'results/unified_revision_v1_20260928_1840'
out = ROOT/'docs/revision/unified_execution'
rows = []
for seed in range(3):
    for path in sorted((run/f'development_{seed}').glob('*.pt')):
        manifest = json.loads(path.with_suffix('.json').read_text())
        assert sha256(path) == manifest['data_sha256']
        value = torch.load(path, map_location='cpu', weights_only=False)
        method = manifest['case']['variant']
        for row in value['decisions']:
            rows.append(dict(model_seed=seed, method=method, nfe=row['eval_before'],
                admitted=float(row['accepted']), residual_changed=float(row['residual_changed']),
                proposed=float(row['proposed'] >= 2),
                eligibility_rejection=float(row['proposed'] >= 2 and not row['eligible']),
                readiness_rejection=float(row['proposed'] >= 2 and row['eligible'] and
                    not row['accepted'] and not row['evidence_ready']),
                margin_rejection=float(row['proposed'] >= 2 and row['eligible'] and
                    not row['accepted'] and row['evidence_ready'])))
data = pd.DataFrame(rows)
assert len(data) == 432000
summary = data.groupby(['method', 'model_seed', 'nfe']).mean(numeric_only=True)
summary.to_csv(out/'decision_timing.csv')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), constrained_layout=True)
for method, label in [('full', 'Full'), ('no_residual', 'Without residual'),
    ('score_only', 'Score only'), ('late_full', 'Full, 70% start'),
    ('late_no_residual', 'Without residual, 70% start')]:
    values = summary.loc[method].groupby('nfe').mean()
    axes[0].plot(values.index, values.admitted, label=label, lw=1.4)
    if method in ('full', 'score_only', 'late_full'):
        axes[1].plot(values.index, values.residual_changed, label=label, lw=1.4)
for ax in axes:
    ax.axvline(210, color='gray', lw=.8, ls='--')
    ax.set(xlabel='Consumed NFE before decision', ylim=(-.02, 1.02))
    ax.grid(alpha=.2); ax.legend(frameon=False, fontsize=8)
axes[0].set(ylabel='Fraction of opportunities accepted', title='Actual portfolio admission')
axes[1].set(ylabel='Fraction of decisions changed', title='Residual change at the same observed state')
for ext in ('png', 'pdf'):
    fig.savefig(out/f'decision_timing.{ext}', dpi=180)
plt.close(fig)
print('432000 decision records summarized; 0 new objective evaluations')
