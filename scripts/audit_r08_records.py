"""Independently check complete R08 records and package shareable scientific evidence.

No objective evaluations, API calls, parameter changes or modifications to frozen
records occur here. Existing raw files remain in place. Missing functions fail.
"""
import argparse
import collections
import csv
from decimal import Decimal
import gzip
import hashlib
import json
from pathlib import Path
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', action='store_true')
    args = parser.parse_args()
    import numpy as np
    import run_r08_hardware as hardware
    p = ROOT / 'experiments/r08_matched38_agent'
    runtime = json.loads((p / 'HARDWARE_RUNTIME.json').read_text())
    run, _ = hardware.configure(str(p), runtime)
    binding = json.loads((p / 'CODE_BINDING.json').read_text())
    assert run.binding() == binding, 'Current code/environment differs from frozen run'
    rows = list(csv.DictReader((p / 'FUNCTION_COMPARISON.csv').open()))
    by_key = {(r['group'], int(r['fid']), r['arm']): r for r in rows}
    totals = {arm: collections.Counter() for arm in run.settings()['arms']}
    counts = collections.Counter()
    for group, fid in run.legacy.TASKS:
        folder = p / 'results' / f'{group}_f{fid:02d}'
        receipt = json.loads((folder / 'COMPLETE.json').read_text())
        assert receipt['binding'] == binding and receipt['n'] == 100 and receipt['nfe'] == 300
        data = {arm: np.load(folder / f'{arm}_FULL100.npz')['values'] for arm in run.settings()['arms']}
        original = run.legacy.original_values(group, fid)
        assert np.array_equal(data['off'], original), 'Frozen original changed'
        prior = ROOT / 'experiments/r07_hardware_accelerated_v1/results' / f'{group}_f{fid:02d}/matched38_FULL100.npz'
        assert np.array_equal(data['matched38'], np.load(prior)['values']), 'Frozen matched38 changed'
        baseline, matched = data['off'].min(1).astype(float), data['matched38'].min(1).astype(float)
        for arm, values in data.items():
            assert values.shape == (100, 300) and np.isfinite(values).all()
            final = values.min(1).astype(float)
            row = by_key[group, fid, arm]
            assert float(row['mean']) == float(final.mean()) and float(row['std']) == float(final.std(ddof=0))
            for reference, prefix in ((baseline, ''), (matched, 'matched38_')):
                assert (int(row[prefix + 'wins']), int(row[prefix + 'ties']), int(row[prefix + 'losses'])) == (
                    int((final < reference).sum()), int((final == reference).sum()), int((final > reference).sum()))
            counts['trajectory_records'] += 100
            counts['evaluation_records'] += 30000
        for batch in range(10):
            receipts = []
            for arm in run.settings()['arms']:
                part = json.loads((folder / f'b{batch:02d}/{arm}.json').read_text())
                assert part['binding'] == binding
                assert part['native_fe'] + part['cached_fe'] == 3000
                totals[arm].update({k: part[k] for k in ('native_fe', 'cached_fe')})
                totals[arm].update({k: part['agent_stats'].get(k, 0) for k in ('changed_points', 'failures')})
                receipts.append(part)
            assert len({r['paired_initial_sha256'] for r in receipts}) == 1
            assert len({r['paired_solver_rng_sha256'] for r in receipts}) == 1
            counts['paired_batches'] += 1
        counts['functions'] += 1
    saved_writer = hardware.write_json
    hardware.write_json = lambda *_: None
    try:
        api_audit = hardware.audit_execution(run)
    finally:
        hardware.write_json = saved_writer
    credits, credit_records = Decimal(0), 0
    for path in (p / 'api_journal').glob('*.json'):
        record = json.loads(path.read_text())
        assert hashlib.sha256(json.dumps(record['request']).encode()).hexdigest() == path.stem
        payload = json.JSONDecoder().raw_decode(record['request']['messages'][-1]['content'].split('INPUT:\n', 1)[1])[0]
        assert payload['policy_id'] == 'r08_matched38_default_full_joint_agent_v1'
        assert payload['failure_fallback'] == 'matched38'
        credit = record.get('response', {}).get('usage', {}).get('credit')
        if isinstance(credit, (int, float)):
            credits += Decimal(str(credit))
            credit_records += 1
    manifest = json.loads((p / 'ARCHIVE_MANIFEST.json').read_text())
    assert len(manifest) == 35
    for name, sha in manifest.items():
        assert digest(p / name) == sha, name
    previous = json.loads((p / 'PRE_MIGRATION.json').read_text())
    assert all(digest(ROOT / name) == sha for name, sha in previous['previous_scientific_records_sha256'].items())
    summary = dict(status='complete_records_independently_verified', counts=dict(counts),
                   per_arm_totals=totals, exact_original_functions=35, exact_frozen_matched38_functions=35,
                   archives_verified=35, all_api_requests_full_identity_checked=True,
                   api={k: v for k, v in api_audit.items() if k != 'provider_reported_billing_fields'},
                   provider_credit_sum=str(credits), responses_with_credit=credit_records, currency='not inferred',
                   previous_scientific_records_unchanged=True, audit_script_sha256=digest(Path(__file__)))
    if args.bundle:
        publication = p / 'publication'
        publication.mkdir(exist_ok=False)
        # Credential contents are checked in memory and never printed or packaged.
        key = (Path.home() / '.config/roopf_agent/worldcodes.env').read_text().strip().split('=', 1)[1].encode()
        journal = sorted((p / 'api_journal').glob('*.json'))
        for path in journal:
            assert key not in path.read_bytes(), 'Credential found in API evidence'
        with tarfile.open(publication / 'API_JOURNAL.tar.xz', 'w:xz') as archive:
            for path in journal:
                archive.add(path, arcname=str(path.relative_to(p)))
        with tarfile.open(publication / 'EVALUATION_EVIDENCE.tar.xz', 'w:xz') as archive:
            for group, fid in run.legacy.TASKS:
                folder = p / 'results' / f'{group}_f{fid:02d}'
                paths = [folder / 'COMPLETE.json', *folder.glob('*_FULL100.npz'),
                         *folder.glob('b??/*.json'), *folder.glob('b??/PAIRED_START.pkl.gz')]
                for path in sorted(paths):
                    archive.add(path, arcname=str(path.relative_to(p)))
        with gzip.open(publication / 'API_EXECUTED_DECISIONS.jsonl.gz', 'wt') as stream:
            for group, fid in run.legacy.TASKS:
                for batch in range(10):
                    path = p / 'results' / f'{group}_f{fid:02d}/b{batch:02d}/llm_live_DECISIONS.json.gz'
                    with gzip.open(path, 'rt') as source:
                        records = json.load(source)
                    prior_reviews = None
                    for record in records:
                        reviews = record.get('completed_reviews_after')
                        changed = reviews != prior_reviews and any(reviews or [])
                        if record['request_event'] or changed:
                            stream.write(json.dumps(dict(group=group, fid=fid, batch=batch, decision=record)) + '\n')
                        prior_reviews = reviews
        hashes = {path.name: digest(path) for path in publication.iterdir() if path.is_file()}
        (publication / 'MANIFEST.json').write_text(json.dumps(dict(sha256=hashes,
            excluded_full_raw_archive_note='Full 35-function raw archives remain local; hashes in ARCHIVE_MANIFEST.json',
            api_credentials_included=False), indent=2) + '\n')
        summary['published_evidence_sha256'] = hashes
    target = p / 'INDEPENDENT_VERIFICATION.json'
    with target.open('x') as stream:
        json.dump(summary, stream, indent=2)
        stream.write('\n')
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    main()
