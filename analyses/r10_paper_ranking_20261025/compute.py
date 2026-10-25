"""Rank current R10 against archived paper tables; no optimization or API calls."""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
ARCHIVE = ROOT / 'experiments/r02_fusion_agent/baseline_52ec388/docs/revision/zip_replication820_results'
CURRENT = ROOT / 'experiments/r10_current_candidate_fusion/FUNCTION_COMPARISON.csv'


def read(path):
    with path.open(newline='') as handle:
        return list(csv.DictReader(handle))


def ranks(values):
    """Average rank for exact numerical ties, lower values are better."""
    return {name: 1 + sum(other < value for other in values.values())
            + .5 * (sum(other == value for other in values.values()) - 1)
            for name, value in values.items()}


def write(name, rows):
    with (OUT / name).open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    sources = {str(CURRENT.relative_to(ROOT)): hashlib.sha256(CURRENT.read_bytes()).hexdigest()}
    functions = []
    details = []
    for row in read(CURRENT):
        if row['arm'] != 'llm_live':
            continue
        group, fid = row['group'], int(row['fid'])
        source = ARCHIVE / f'{group}_f{fid:02d}' / 'COMPARISONS.csv'
        sources[str(source.relative_to(ROOT))] = hashlib.sha256(source.read_bytes()).hexdigest()
        archive = read(source)
        assert float(row['baseline_mean']) == float(archive[0]['full_mean'])
        values = {r['method']: float(r['baseline_mean']) for r in archive}
        assert len(values) == (9 if group == 'UAV' else 14)
        values['R10 Full'] = float(row['mean'])
        function_ranks = ranks(values)
        functions.append(dict(group=group, fid=fid, values=values, ranks=function_ranks))
        for method, value in values.items():
            details.append(dict(group=group, fid=fid, method=method, mean=value,
                                rank=function_ranks[method], primary=not (group == 'BBOB' and fid == 20)))
    assert len(functions) == 35
    summaries = []
    scopes = [('BBOB23_exclude_f20', 'BBOB', 20), ('BBOB24_reference_only', 'BBOB', None),
              ('CEC6_reference_only', 'CEC-style', None), ('UAV5_reference_only', 'UAV', None)]
    for scope, group, exclude in scopes:
        selected = [r for r in functions if r['group'] == group and r['fid'] != exclude]
        averages = {name: sum(r['ranks'][name] for r in selected) / len(selected)
                    for name in selected[0]['values']}
        overall = ranks(averages)
        for method in sorted(averages, key=averages.get):
            wins = sum(r['values']['R10 Full'] < r['values'][method] for r in selected)
            ties = sum(r['values']['R10 Full'] == r['values'][method] for r in selected)
            losses = len(selected) - wins - ties
            summaries.append(dict(scope=scope, method=method, functions=len(selected),
                                  average_rank=averages[method], overall_rank=overall[method],
                                  full_function_wins=wins, full_function_ties=ties,
                                  full_function_losses=losses))
    write('RANKS.csv', summaries)
    write('FUNCTION_RANKS.csv', details)
    metadata = dict(
        current_method='R10 llm_live (Full)',
        method_set='Replace archived ROOPF with current R10; exclude matched38 and all historical agent versions.',
        ranking='Rank per-function final means ascending with average ranks for exact ties; average across functions; average ranks for tied overall scores.',
        limitations=[
            'Other algorithms use archived paper means, not new paired reruns.',
            'No significance testing is implied by these reference ranks.',
            'BBOB20 has batch-coupled stochastic semantics and is excluded from primary BBOB23.',
            'CEC archived offset identity is unconfirmed; current six functions retain structured locks with no agent intervention.',
            'UAV uses the recovered proxy protocol; the original released ZIP lacked a complete UAV entry point.',
            'No pooled 35-function rank: comparator sets differ by suite.'
        ], sources_sha256=sources)
    (OUT / 'PROVENANCE.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n')
    for row in summaries:
        print(row['scope'], row['overall_rank'], row['method'], f"{row['average_rank']:.6f}")


if __name__ == '__main__':
    main()
