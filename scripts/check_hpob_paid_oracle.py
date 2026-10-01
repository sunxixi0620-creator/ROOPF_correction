"""Information-flow contracts on fabricated data: zero HPO-B response queries."""
import json
from pathlib import Path
import sys
import tempfile
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from roopf.hpob_paid_oracle import DevelopmentOracle, initialization
from roopf.experiment_io import write_json, sha256


def main():
    task = dict(space='1', dataset='2', canonical_indices=list(range(8)),
                eligible_raw_disjoint=True, official_initializations={'test0': [0, 1, 2, 3, 4]})
    assert initialization(task, 'test0') == ([0, 1, 2, 3, 4], 'official')
    assert initialization(task, 'test1') == initialization(task, 'test1')
    with tempfile.TemporaryDirectory(prefix='roopf_hpob_oracle_') as tmp:
        path = Path(tmp)/'meta-validation-dataset.json'
        x = np.linspace(0, 1, 16).reshape(8, 2).tolist()
        path.write_text(json.dumps({'1': {'2': dict(X=x, y=[[100.+k] for k in range(8)])}}))
        oracle = DevelopmentOracle(path, task, budget=2)
        assert np.array_equal(oracle.X, x) and not hasattr(oracle, 'y')
        assert oracle.query(3) == 103.
        for k in (3, -1, 8):
            try:
                oracle.query(k)
                raise AssertionError('invalid/repeated query accepted')
            except ValueError:
                pass
        assert oracle.query(0) == 100.
        try:
            oracle.query(4)
            raise AssertionError('budget overflow accepted')
        except ValueError:
            pass
        result = oracle.close()
        assert result['calls'] == 2 and result['paid'] == [3, 0]
    write_json(ROOT/'docs/research/hpob_preparation/ORACLE_CONTRACTS.json',
        dict(benchmark_objective_calls=0, fabricated_queries=2,
             separate_response_process=True, raw_paid_responses=True,
             duplicate_invalid_and_overbudget_rejected=True,
             test_responses_downloaded=False,
             source_sha256=sha256(ROOT/'roopf/hpob_paid_oracle.py')))
    print('HPO-B oracle contracts passed; zero benchmark queries.')


if __name__ == '__main__':
    main()
