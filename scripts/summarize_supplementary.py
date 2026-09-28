"""Recompute paired summaries, conditional bootstrap intervals, and plots."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def main():
    p = argparse.ArgumentParser(); p.add_argument('directory', type=Path); args = p.parse_args()
    root = args.directory
    if not (root/'COMPLETE').exists():
        raise RuntimeError('Refuse final summary before all predeclared runs complete')
    rows = list(csv.DictReader((root/'raw_results.csv').open()))
    keys = [(x['case'], x['variant'], x['warmup'], x['seed']) for x in rows]
    assert len(keys) == len(set(keys))
    assert all(int(x['actual_nfe']) == 300 for x in rows)
    rng = np.random.default_rng(9202801)
    summary = []
    configurations = sorted({(x['variant'], x['warmup']) for x in rows})
    for case in dict.fromkeys(x['case'] for x in rows):
        reference = {x['seed']: float(x['final']) for x in rows if x['case'] == case and x['variant'] == 'full' and float(x['warmup']) == .7}
        for variant, warmup in configurations:
            if variant == 'full' and float(warmup) == .7:
                continue
            current = {x['seed']: float(x['final']) for x in rows if x['case'] == case and x['variant'] == variant and x['warmup'] == warmup}
            if not current:
                continue
            assert current.keys() == reference.keys(), (case, variant, warmup)
            seeds = sorted(reference)
            full = np.array([reference[s] for s in seeds]); other = np.array([current[s] for s in seeds])
            diff = full - other
            bootstrap = diff[rng.integers(0, len(diff), size=(5000, len(diff)))].mean(1)
            lo, hi = np.quantile(bootstrap, [.025, .975])
            summary.append({'case': case, 'comparator': variant, 'comparator_warmup': warmup,
                'n_pairs': len(seeds), 'full_mean': full.mean(), 'comparator_mean': other.mean(),
                'full_std': full.std(ddof=1), 'comparator_std': other.std(ddof=1),
                'full_minus_comparator': diff.mean(), 'ci95_lower': lo, 'ci95_upper': hi,
                'wins': int((diff < 0).sum()), 'ties': int((diff == 0).sum()), 'losses': int((diff > 0).sum())})
    write_csv(root/'paired_summary.csv', summary)
    groups = []
    for variant, warmup in configurations:
        for group in ('bbob', 'zero', 'shifted'):
            subset = [x for x in summary if x['comparator'] == variant and x['comparator_warmup'] == warmup and (x['case'].startswith('bbob') if group == 'bbob' else x['case'].endswith(group))]
            if subset:
                groups.append({'group': group, 'comparator': variant, 'warmup': warmup,
                    'functions': len(subset), 'mean_wins': sum(x['full_minus_comparator'] < 0 for x in subset),
                    'mean_ties': sum(x['full_minus_comparator'] == 0 for x in subset),
                    'mean_losses': sum(x['full_minus_comparator'] > 0 for x in subset),
                    'pair_wins': sum(x['wins'] for x in subset), 'pair_ties': sum(x['ties'] for x in subset), 'pair_losses': sum(x['losses'] for x in subset)})
    write_csv(root/'group_summary.csv', groups)
    mechanisms = []
    for variant, warmup in configurations:
        subset = [x for x in rows if x['variant'] == variant and x['warmup'] == warmup]
        batches = {(x['case'], x['algorithm_seed']): x for x in subset}
        sampled = sum(int(x['sampled_decisions']) for x in subset)
        opportunities = sum(int(x['gate_opportunities']) for x in subset)
        adopted = sum(int(x['portfolio_evals']) for x in subset)
        mechanisms.append({'variant': variant, 'warmup': warmup, 'trajectories': len(subset),
            'gate_opportunities': opportunities, 'portfolio_evals': adopted,
            'acceptance_rate': adopted/opportunities if opportunities else 0,
            'incumbent_improvement_events': sum(int(x['portfolio_incumbent_improvements']) for x in subset),
            'sampled_decisions': sampled, 'residual_decision_changes_sampled': sum(int(x['residual_changes_sampled']) for x in subset),
            'timed_batches': len(batches),
            'median_batch_seconds_excluding_observer': np.median([float(x['online_seconds_excluding_observer']) for x in batches.values()]),
            'median_amortized_seconds_per_trajectory': np.median([float(x['online_seconds_excluding_observer'])/int(x['batch_size']) for x in batches.values()]),
            'median_proxy_seconds_per_batch': np.median([float(x['proxy_seconds']) for x in batches.values()])})
    write_csv(root/'mechanism_runtime_summary.csv', mechanisms)
    lines = ['# Supplementary experiment results', '',
        'Development-instance evidence, not independent confirmation. Full method is the final ROOPF; comparators are controlled ablations or schedules.', '',
        f'{len(rows)} trajectories, each 300 objective evaluations. Bootstrap intervals condition on each fixed instance and resample paired initializations; they do not establish family-level generalization. No multiplicity-adjusted significance claims are made.', '',
        '| Group | Comparator | Warm-up | Function mean W/T/L | Paired endpoint W/T/L |',
        '|---|---|---:|---|---|']
    for x in groups:
        lines.append(f"| {x['group']} | {x['comparator']} | {x['warmup']} | {x['mean_wins']}/{x['mean_ties']}/{x['mean_losses']} | {x['pair_wins']}/{x['pair_ties']}/{x['pair_losses']} |")
    lines += ['', 'W/T/L is full/default against comparator. Tiny exact floating-point differences are counted, so assess paired_summary.csv effect sizes and intervals before interpreting counts.', '',
        'If parallel_execution.json exists, these runtimes are scheduling diagnostics, not controlled timing comparisons. Runtime is instrumented CPU batch wall time with observer overhead removed; component timers are nested. Amortized seconds are throughput, not isolated single-trajectory latency. CEC residual routing is disabled by the selected final implementation. f20 is excluded for predeclared objective stochasticity/batch coupling.']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    cases = list(dict.fromkeys(x['case'] for x in rows))
    fig, ax = plt.subplots(figsize=(11, max(3, .24*len(cases))))
    plots = [c for c in configurations if c != ('full', '0.7')]
    for i, (variant, warmup) in enumerate(plots):
        data = {x['case']: x for x in summary if x['comparator'] == variant and x['comparator_warmup'] == warmup}
        xs=[]; ys=[]
        for j, case in enumerate(cases):
            if case in data and data[case]['full_mean'] >= 0 and data[case]['comparator_mean'] >= 0:
                xs.append(np.log10((data[case]['full_mean']+1e-12)/(data[case]['comparator_mean']+1e-12)))
                ys.append(j+(i-(len(plots)-1)/2)*.14)
        ax.scatter(xs, ys, s=18, label=f'{variant}, warm-up={warmup}')
    ax.axvline(0, color='gray', linewidth=1); ax.set_yticks(range(len(cases)), cases)
    ax.set_xlabel('log10(full mean objective / comparator mean objective); negative favors full')
    ax.legend(fontsize=8); ax.grid(axis='x', alpha=.2); fig.tight_layout()
    fig.savefig(root/'paired_effects.pdf'); fig.savefig(root/'paired_effects.png', dpi=170); plt.close(fig)
    print(json.dumps({'trajectories':len(rows), 'pairwise_case_comparisons':len(summary), 'groups':groups}), flush=True)


if __name__ == '__main__':
    main()
