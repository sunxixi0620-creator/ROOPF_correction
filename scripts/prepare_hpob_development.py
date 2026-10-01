"""Metadata-only source/development eligibility audit; never reads test labels."""
import json
from pathlib import Path
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from roopf.experiment_io import sha256, write_json
DATA = ROOT/'results/hpob_source_preparation/data'
OUT = ROOT/'docs/research/hpob_preparation'


def main():
    protocol = ROOT/'docs/experiments/HPOB_DEVELOPMENT_INTAKE_PROTOCOL.md'
    identity = dict(protocol_sha256=sha256(protocol),
        source_sha256=sha256(Path(__file__)),
        download=json.loads((DATA.parent/'DOWNLOAD.json').read_text()))
    for r in identity['download']['files']:
        assert sha256(DATA/r['name']) == r['sha256']
    assert not (DATA/'meta-test-dataset.json').exists()
    train = json.loads((DATA/'meta-train-dataset.json').read_text())
    val = json.loads((DATA/'meta-validation-dataset.json').read_text())
    init = json.loads((DATA/'bo-initializations.json').read_text())
    spaces = sorted(set(train) & set(val), key=int)
    assert len(spaces) == 16
    test_ids = {s: sorted(set(init[s])-set(train[s])-set(val[s]), key=int) for s in spaces}
    heldout = set().union(*(set(val[s]) | set(test_ids[s]) for s in spaces))
    source_ids = set().union(*(set(train[s]) for s in spaces))
    rows = []
    for s in spaces:
        original = sorted(train[s], key=int)
        strict = [d for d in original if d not in heldout]
        for d, task in sorted(val[s].items(), key=lambda p: int(p[0])):
            # No extraction, statistics, normalization or ranking of target y.
            x = np.asarray(task['X'], dtype=np.float64)
            unique, indices = np.unique(x, axis=0, return_index=True)
            valid_x = bool(x.ndim == 2 and 1 <= x.shape[1] <= 64 and np.isfinite(x).all()
                           and x.min() >= -1e-6 and x.max() <= 1+1e-6)
            eligible = bool(valid_x and len(unique) >= 105)
            rows.append(dict(space=s, dataset=d, dim=x.shape[1], points=len(x),
                unique_points=len(unique), canonical_indices=sorted(indices.tolist()),
                official_sources=original, strict_sources=strict,
                eligible_official=eligible and len(original) >= 5,
                eligible_strict=eligible and len(strict) >= 5,
                official_initializations=init.get(s, {}).get(d, {})))
    result = dict(identity=identity, spaces=spaces, validation_tasks=len(rows),
        inferred_test_ids=test_ids, source_dataset_ids=len(source_ids),
        heldout_dataset_ids=len(heldout), overlap_dataset_ids=sorted(source_ids&heldout,key=int),
        eligible_official=sum(r['eligible_official'] for r in rows),
        eligible_strict=sum(r['eligible_strict'] for r in rows),
        test_responses_downloaded=False, optimization_calls=0, training_epochs=0,
        target_response_statistics_computed=False, tasks=rows)
    write_json(OUT/'DEVELOPMENT_MANIFEST.json',result)
    summary={k:v for k,v in result.items() if k not in ('tasks','identity','inferred_test_ids','overlap_dataset_ids')}
    summary['overlap_dataset_ids_count']=len(result['overlap_dataset_ids'])
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    main()
