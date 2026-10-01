"""Fixed-task online baseline audit; no learned model tuning or new task draws."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import json
import multiprocessing as mp
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import continuous_prior_replication as parent
from roopf.fitted_online import TrustRegion, propose, replay_score
from roopf.experiment_io import CaseStore, sha256, write_json, save_torch, seed_for

RUN = ROOT / 'results/strong_online_baselines_v1'
OUT = ROOT / 'docs/revision/strong_online_baselines'
METHODS = ('GP_LogEI', 'TuRBO_LogEI')
SOURCES = ('scripts/strong_online_baselines.py', 'roopf/fitted_online.py',
           'docs/experiments/STRONG_ONLINE_BASELINES_PROTOCOL.md',
           'requirements-strong-online.txt')


def freeze():
    parent.verify()
    OUT.mkdir(parents=True, exist_ok=True)
    if not (RUN / 'identity.json').exists():
        import botorch, gpytorch, scipy, linear_operator
        write_json(RUN / 'identity.json', dict(parent=sha256(parent.RUN/'identity.json'),
            sources={p: sha256(ROOT/p) for p in SOURCES}, time=time.time(),
            packages=dict(torch=torch.__version__, botorch=botorch.__version__, gpytorch=gpytorch.__version__,
                          scipy=scipy.__version__, linear_operator=linear_operator.__version__)))
    return verify()


def verify():
    parent.verify()
    z = json.loads((RUN/'identity.json').read_text())
    assert sha256(parent.RUN/'identity.json') == z['parent']
    for p, h in z['sources'].items():
        assert sha256(ROOT/p) == h, p
    return z


def rollout(f, i, method, budget=300, role=parent.ROLE, device='cpu'):
    torch.set_num_threads(1)
    start = time.perf_counter()
    task = parent.parent.ProceduralTask(f, i, role, dim=20)
    x = parent.st.inputs(f, i, role, 10)
    y = task.calfitness(x[None])[0]
    initial_x, initial_y = x.clone(), y.clone()
    state = TrustRegion(best=float(y.min())) if method == 'TuRBO_LogEI' else None
    previous = None
    trace = []
    while len(x) < budget:
        if state and state.length < .5**7:
            # New local data after restart; keep global incumbent and paid ledger.
            n = min(10, budget-len(x))
            start_at = len(x)
            seed = seed_for('strong_online_restart', f, i, role, state.restarts) % (2**31-1)
            q = torch.quasirandom.SobolEngine(20, scramble=True, seed=seed).draw(n) * 10 - 5
            assert torch.cdist(q.double(), x.double()).min() > 1e-6
            v = task.calfitness(q[None])[0]
            x, y = torch.cat((x, q)), torch.cat((y, v))
            state = TrustRegion(best=float(v.min()), start=start_at, restarts=state.restarts+1)
            trace.append(dict(restart=True, paid_before=start_at, count=n, points=q))
            previous = None
            continue
        before = task.points
        q, info, previous = propose(x, y, method, (f, i, role), state, previous, device)
        assert task.points == before and task.diagnostic_points == 0
        value = task.calfitness(q[None])[0, 0]
        info.update(paid_before=len(x), restart=False)
        trace.append(info)
        x, y = torch.cat((x, q)), torch.cat((y, value[None]))
        if state:
            state.update(float(value))
    assert task.points == budget and task.diagnostic_points == 0
    return dict(fid=f, instance=i, method=method, x=x, y=y, initial_x=initial_x, initial_y=initial_y,
                calls=task.points, teacher_calls=0, trace=trace, seconds=time.perf_counter()-start, device=device)


def jobs():
    return [(f, i, m) for f in range(36) for i in range(2) for m in METHODS]


def collect(j):
    store = CaseStore(RUN/'cases', verify())
    name = '_'.join(map(str, j))
    with store.lock(name):
        if store.load(name, dict(job=j)) is None:
            try:
                store.save(name, dict(job=j), rollout(*j))
            except Exception as exc:
                write_json(RUN/'errors'/f'{name}.json', dict(job=j, error=repr(exc)))
                raise


def contracts():
    verify()
    if (OUT/'CONTRACTS.json').exists():
        return
    results = {}
    for m in METHODS:
        a = rollout(0, 0, m, 12, 'strong_online_contract_v1')
        b = rollout(0, 0, m, 12, 'strong_online_contract_v1')
        assert torch.equal(a['x'], b['x']) and torch.equal(a['y'], b['y'])
        for v in (a, b):
            for info in v['trace']:
                n = info['paid_before']
                assert abs(replay_score(v['x'][:n], v['y'][:n], m, info)-info['logei']) < 1e-6
        results[m] = [a, b]
    # Check expansion, contraction, and restart trigger independently of objectives.
    state = TrustRegion(best=0.)
    for _ in range(20):
        state.update(1.)
    assert state.length == .4
    for n in range(10):
        state.update(-float(n+1))
    assert state.length == .8
    for _ in range(160):
        state.update(1.)
    assert state.length < .5**7
    save_torch(RUN/'contracts.pt', results)
    write_json(OUT/'CONTRACTS.json', dict(calls=48, deterministic=True, paid_only=True,
        posterior_replay=True, trust_region_transitions=True, epochs=0))


def run(workers):
    verify()
    assert (OUT/'CONTRACTS.json').exists()
    start = time.time()
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('spawn')) as ex:
        fs = {ex.submit(collect, j): j for j in jobs()}
        for n, f in enumerate(as_completed(fs), 1):
            f.result()
            write_json(RUN/'STATUS.json', dict(completed=n, total=144, elapsed=time.time()-start))
            print(f'{n}/144 {time.time()-start:.1f}s {fs[f]}', flush=True)
    write_json(RUN/'COMPLETE.json', dict(cases=144, calls=43200, seconds=time.time()-start, workers=workers))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('phase', choices=['freeze', 'contracts', 'run'])
    p.add_argument('--workers', type=int, default=12)
    a = p.parse_args()
    run(a.workers) if a.phase == 'run' else globals()[a.phase]()
