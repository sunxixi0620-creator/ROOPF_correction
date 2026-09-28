"""Check archived evidence against completed local outputs before publication."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from archive_remaining import NAMES, ROOT, digest


def main():
    archive = ROOT / 'artifacts/remaining_20260928'
    manifest = json.loads((archive / 'manifest.json').read_text())
    assert set(manifest) == set(NAMES)
    report = {'stages': [], 'original_checkpoints_unchanged': True}
    frozen = {
        'anchor_policy_d10.pt': 'a265d448ad4a2d191eed23dafa36e03e9475a7b95e824bb04d9a88715031a06d',
        'residual_selector_generated36_d10.pt': '42bac7c6792b6980ad149eff08ce4b2d93ce3ffba51d3d869dc0d1625b3a9d01',
    }
    for name, expected in frozen.items():
        assert digest(ROOT / 'checkpoints' / name) == expected, name
    expected_rows = {
        'remaining_operators': 2340, 'remaining_replay': 1260,
        'remaining_uav': 1050, 'remaining_gp_ei': 680,
        'remaining_surr_rlde': 680, 'random_shortlist': 180,
        'native20_eval': 5760, 'order_original_eval': 960,
        'order_curated_eval': 960, 'serial_external_timing': 72,
    }
    for name in NAMES:
        root = ROOT / 'results' / name
        assert (root / 'COMPLETE').exists(), name
        recorded = {}
        for part in manifest[name]:
            assert digest(archive / part['archive']) == part['sha256']
            for rel, info in part['files'].items():
                assert rel not in recorded
                path = ROOT / rel
                assert path.stat().st_size == info['bytes']
                assert digest(path) == info['sha256'], rel
                recorded[rel] = info
        actual = {str(p.relative_to(ROOT)) for p in root.rglob('*') if p.is_file()}
        assert actual <= set(recorded), (name, actual - set(recorded))
        record = {'stage': name, 'archived_files_verified': len(recorded)}
        short = name.removesuffix('_20260928')
        if short in expected_rows:
            if (root / 'all_rows.json').exists():
                rows = pd.DataFrame(json.loads((root / 'all_rows.json').read_text()))
            else:
                rows = pd.read_csv(root / 'raw_results.csv')
            assert len(rows) == expected_rows[short], name
            keys = [k for k in ['suite', 'case', 'fid', 'instance', 'seed',
                               'dimension', 'budget', 'method', 'mode',
                               'force_at', 'force_kind'] if k in rows]
            assert not rows.duplicated(keys).any(), (name, keys)
            assert np.isfinite(rows.final).all()
            assert (rows.actual_nfe == (rows.budget if 'budget' in rows else 300)).all()
            traces = list(root.glob('*.npy')) + list(root.glob('*.npz'))
            if short != 'serial_external_timing':
                assert len(traces) == len(rows), (name, len(traces))
            for path in traces:
                data = np.load(path)
                if isinstance(data, np.lib.npyio.NpzFile):
                    with data:
                        trace = data['trace'] if 'trace' in data.files else data['trail']
                else:
                    trace = data
                assert np.isfinite(trace).all(), path
                assert (np.diff(trace, axis=-1) <= 0).all(), path
            record.update(trajectories=len(rows), main_points=int(rows.actual_nfe.sum()),
                          traces_checked=len(traces), unique_ids_and_budget=True)
        if short.startswith('retrain_d'):
            history = json.loads((root / 'history.json').read_text())
            complete = json.loads((root / 'COMPLETE').read_text())
            assert len(history) == complete['epochs'] == 80
            assert [h['epoch'] for h in history] == list(range(80))
            assert complete['point_evaluations'] == 110592000
            assert digest(root / 'anchor.pt') == complete['checkpoint_sha256']
            record.update(training_epochs=80, training_points=complete['point_evaluations'])
        report['stages'].append(record)
    counts = json.loads((ROOT / 'docs/experiments/FINAL_COMPLETION_COUNTS_20260928.json').read_text())
    assert sum(s.get('trajectories', 0) for s in report['stages']) == counts['new_trajectories']
    assert sum(s.get('main_points', 0) for s in report['stages']) == counts['new_main_points']
    target = ROOT / 'docs/experiments/remaining_tables/final_integrity_checks.json'
    target.write_text(json.dumps(report, indent=2))
    print('Verified all 15 stages, original checkpoints, archives, budgets and traces.')


if __name__ == '__main__':
    main()
