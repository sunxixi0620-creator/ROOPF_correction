"""Paid-response interface for development HPO-B tables.

The target response table lives in a spawned evaluator process. The optimizer
gets public candidate X and replies to explicit paid index queries only.
This is an experiment information-flow boundary, not an OS security sandbox.
"""
import json
import multiprocessing as mp
from pathlib import Path
import numpy as np


def _serve(path, space, dataset, canonical, budget, conn):
    try:
        # Validation-only entry point; no generic path capable of loading test.
        path = Path(path)
        if path.name != 'meta-validation-dataset.json':
            raise ValueError('Development evaluator accepts only validation split')
        tables = json.loads(path.read_text())
        task = tables[str(space)][str(dataset)]
        x = np.asarray(task['X'], dtype=np.float64)[canonical]
        y = np.asarray(task['y'], dtype=np.float64).reshape(-1)[canonical]
        del task, tables
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError('Nonfinite benchmark data')
        if len(x) < budget:
            raise ValueError('Insufficient unique candidates for budget')
        conn.send(dict(ok=True, X=x, budget=budget, maximize=True))
        paid, values = [], []
        while True:
            request = conn.recv()
            op = request.get('op')
            if op == 'close':
                conn.send(dict(ok=True, paid=paid, values=values, calls=len(paid)))
                return
            if op != 'query':
                conn.send(dict(ok=False, error='Only paid query or close is allowed'))
                continue
            k = request.get('index')
            if type(k) is not int or k < 0 or k >= len(x):
                conn.send(dict(ok=False, error='Invalid candidate index'))
            elif k in paid:
                conn.send(dict(ok=False, error='Candidate already observed'))
            elif len(paid) >= budget:
                conn.send(dict(ok=False, error='Paid budget exhausted'))
            else:
                paid.append(k)
                values.append(float(y[k]))
                conn.send(dict(ok=True, value=float(y[k]), calls=len(paid)))
    except Exception as exc:
        conn.send(dict(ok=False, error=repr(exc)))
    finally:
        conn.close()


class DevelopmentOracle:
    def __init__(self, path, task, budget=105):
        if not task.get('eligible_raw_disjoint', False):
            raise ValueError('Task must pass the OpenML raw-dataset disjoint source rule')
        if Path(path).name != 'meta-validation-dataset.json':
            raise ValueError('Development oracle only accepts the validation split')
        ctx = mp.get_context('spawn')
        parent, child = ctx.Pipe()
        self._connection = parent
        self._process = ctx.Process(target=_serve,
            args=(str(path), task['space'], task['dataset'], task['canonical_indices'], budget, child))
        self._process.start()
        child.close()
        message = self._receive()
        self.X = message['X']
        self.X.setflags(write=False)
        self.budget = message['budget']
        self.maximize = message['maximize']
        self.observed_indices, self.observed_values = [], []

    def _receive(self):
        message = self._connection.recv()
        if not message['ok']:
            raise ValueError(message['error'])
        return message

    def query(self, index):
        self._connection.send(dict(op='query', index=index))
        message = self._receive()
        self.observed_indices.append(index)
        self.observed_values.append(message['value'])
        assert message['calls'] == len(self.observed_indices)
        return message['value']

    def close(self):
        self._connection.send(dict(op='close'))
        message = self._receive()
        self._connection.close()
        self._process.join(timeout=10)
        assert not self._process.is_alive()
        return message


def initialization(task, seed_id):
    """Return five canonical indices, using public official indices where valid."""
    from roopf.experiment_io import seed_for
    mapping = {old: new for new, old in enumerate(task['canonical_indices'])}
    official = task['official_initializations'].get(seed_id, [])
    if len(official) == 5 and len(set(official)) == 5 and all(k in mapping for k in official):
        return [mapping[k] for k in official], 'official'
    seed = seed_for('hpob_development_initial', task['space'], task['dataset'], seed_id)
    indices = np.random.default_rng(seed).choice(len(mapping), 5, replace=False)
    return indices.tolist(), 'input_only_permutation'
