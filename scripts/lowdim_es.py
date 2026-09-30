"""Fixed rank-four output adaptation versus full-coordinate ES; audit only."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
from pathlib import Path
import sys, json, time, multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import es_reliability as parent
from roopf.complementary_proposal import build_context
from roopf.revision_tasks import ProceduralTask, population
from roopf.experiment_io import CaseStore, sha256, fingerprint, seed_for, write_json, save_torch
from scripts.unified_revision import bounded
RUN = ROOT / 'results/lowdim_es_20260930'
OUT = ROOT / 'docs/revision/lowdim_es'
SOURCES = ['scripts/lowdim_es.py', 'docs/experiments/LOWDIM_ES_PROTOCOL.md']
METHODS = ('Low', 'Full')
GROUPS = ('Base', 'Low_A', 'Low_B', 'Full_A', 'Full_B')


def basis():
    g = torch.Generator().manual_seed(seed_for('lowdim_es_v1', 'basis'))
    return torch.linalg.qr(torch.randn(200, 4, generator=g), mode='reduced').Q


def noise(seed, method, group, direction):
    g = torch.Generator().manual_seed(seed_for('lowdim_es_v1', seed, method, group, direction))
    return torch.randn(200 if method == 'Low' else 58504, generator=g)


def make(seed, method='Full', offset=None):
    p = parent.old.proposal(seed)
    p.load_state_dict(torch.load(parent.PARENT / f'selected_{seed}.pt', weights_only=False))
    if offset is not None:
        scale = torch.load(parent.PARENT / f'origin_{seed}.pt', weights_only=False)['scale']
        full = offset
        if method == 'Low':
            full = torch.zeros(58504)
            at = 0
            for name, param in p.named_parameters():
                n = param.numel()
                if name == 'context.2.weight':
                    full[at:at+n] = (offset[:160].reshape(40, 4) @ basis().T).flatten()
                elif name == 'context.2.bias':
                    full[at:at+n] = offset[160:]
                at += n
            assert at == 58504
        vector_to_parameters(parameters_to_vector(p.parameters()) + scale * full, p.parameters())
    return build_context(proposal=p).eval().requires_grad_(False)


def simulate(spec):
    torch.set_num_threads(1)
    offset = None
    method = spec.get('method', 'Full')
    if spec['role'] == 'estimation':
        offset = spec['sign'] * .02 * noise(spec['seed'], method, spec['group'], spec['direction'])
    elif spec.get('step'):
        path = RUN / spec['step']
        assert sha256(path) == spec['step_hash']
        offset = torch.load(path, weights_only=False)
    elif spec.get('zero'):
        offset = torch.zeros(200)
    model = make(spec['seed'], method, offset)
    role, f, i = spec['role'], spec['fid'], spec['instance']
    count = 4 if role == 'transfer' else 2
    split = 'lowdim_es_v1_' + role
    task = ProceduralTask(f, i, split, dim=20)
    x = population(f, i, split, count=count, dim=20)
    initial_x_sha = fingerprint(x)
    torch.manual_seed(seed_for('lowdim_es_v1', role, f, i, 'policy'))
    before = fingerprint(model.state_dict())
    start = time.perf_counter()
    with torch.no_grad():
        _, trail, nfe, points = model(x, task)
    assert nfe == 600 and task.points == 600 * count and task.diagnostic_points == 0
    assert torch.isfinite(trail).all() and before == fingerprint(model.state_dict())
    return dict(utility=bounded(task.initial_values.min(1).values, trail[:, -1], task.initial_values.std(1)),
        trail=trail, points_sha=fingerprint(points), initial_x_sha=initial_x_sha,
        initial_y_sha=fingerprint(task.initial_values), main_calls=task.points, teacher_calls=0,
        seconds=time.perf_counter()-start)


def verify():
    identity = json.loads((RUN / 'identity.json').read_text())
    for p, h in identity['sources'].items():
        assert sha256(ROOT / p) == h
    parent.verify()
    assert identity['parent_identity'] == sha256(parent.RUN / 'identity.json')
    return identity


def worker(spec):
    store = CaseStore(RUN / 'cases', verify())
    key = fingerprint(spec)[:32]
    with store.lock(key):
        value = store.load(key, spec)
        if value is None:
            value = simulate(spec)
            store.save(key, spec, value)
    return value


def interval(delta, level=.975):
    recipe = delta.reshape(3, 12, 3, 2, 4).mean((0, 2, 3, 4))
    rng = np.random.default_rng(20261004)
    boot = recipe[rng.integers(0, 12, (5000, 12))].mean(1)
    lo, hi = np.quantile(boot, [(1-level)/2, 1-(1-level)/2])
    seeds = delta.mean((1, 2, 3))
    return dict(mean=float(delta.mean()), lower=float(lo), upper=float(hi), level=level,
        seed_means=seeds.tolist(), passed=bool(delta.mean() >= .0001 and lo > 0 and (seeds > 0).all()))


def contracts():
    path = RUN / 'CONTRACTS.json'
    if path.exists():
        return
    q = basis()
    assert torch.allclose(q.T @ q, torch.eye(4), atol=1e-6)
    for s in range(3):
        base = make(s).proposal
        zero = make(s, 'Low', torch.zeros(200)).proposal
        changed = make(s, 'Low', .02 * noise(s, 'Low', 'A', 0)).proposal
        assert fingerprint(base.state_dict()) == fingerprint(zero.state_dict())
        modified = [k for k in base.state_dict() if not torch.equal(base.state_dict()[k], changed.state_dict()[k])]
        assert modified == ['context.2.weight', 'context.2.bias']
    specs = [dict(role='contract', seed=0, method=m, fid=0, instance=0, zero=z)
             for m, z in [('Full', False), ('Low', True)]]
    a, b = [worker(spec) for spec in specs]
    assert a['points_sha'] == b['points_sha'] and torch.equal(a['trail'], b['trail'])
    write_json(path, dict(zero_offset_exact=True, changed_only_output_layer=True,
        orthonormal=True, calls=2400, point_sha=a['points_sha']))


def execute(pool, jobs, phase):
    values = []
    for value in pool.map(worker, jobs):
        values.append(value)
        if len(values) % 32 == 0 or len(values) == len(jobs):
            write_json(RUN / 'STATUS.json', dict(status=phase, completed=len(values), total=len(jobs), workers=pool._max_workers))
    return values


def main():
    torch.set_num_threads(1)
    RUN.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    if (RUN / 'COMPLETE.json').exists():
        print('Already complete; no simulations launched.')
        return
    start = time.perf_counter()
    if not (RUN / 'identity.json').exists():
        parent.verify()
        write_json(RUN / 'identity.json', dict(version='lowdim_es_v1',
            sources={p: sha256(ROOT / p) for p in SOURCES}, parent_identity=sha256(parent.RUN / 'identity.json')))
    verify()
    contracts()
    available = parent.available_gib()
    workers = 32 if available >= 26 else 24 if available >= 22 else 16
    assert available >= 17, 'Insufficient memory headroom'
    write_json(RUN / 'RESOURCES.json', dict(workers=workers, available_gib=available,
        threads_per_worker=1, device='cpu', benchmark='es_reliability_20260930/RESOURCES.json'))
    fids = np.random.default_rng(20261003).permutation(36)[:6].tolist()
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('spawn')) as pool:
        jobs = [dict(role='estimation', seed=s, method=m, group=g, direction=d, sign=sign, fid=f, instance=0)
                for s in range(3) for m in METHODS for g in ('A', 'B') for d in range(8) for sign in (1, -1) for f in fids]
        values = execute(pool, jobs, 'estimation')
        scores = np.array([float(v['utility'].mean()) for v in values]).reshape(3, 2, 2, 8, 2, 6)
        directions = []
        for s in range(3):
            for mi, m in enumerate(METHODS):
                grads = []
                for gi, g in enumerate(('A', 'B')):
                    diff = (scores[s, mi, gi, :, 0] - scores[s, mi, gi, :, 1]).mean(1)
                    grad = sum(float(diff[d]) * noise(s, m, g, d) for d in range(8)) / .32
                    assert torch.isfinite(grad).all()
                    phi = torch.nn.Parameter(torch.zeros_like(grad))
                    opt = torch.optim.Adam([phi], lr=.01)
                    phi.grad = -grad
                    torch.nn.utils.clip_grad_norm_([phi], 1)
                    opt.step()
                    save_torch(RUN / f'step_{s}_{m}_{g}.pt', phi.detach())
                    save_torch(RUN / f'gradient_{s}_{m}_{g}.pt', grad)
                    grads.append(grad)
                directions.append(dict(seed=s, method=m, norm_A=float(grads[0].norm()),
                    norm_B=float(grads[1].norm()), cosine=float(torch.nn.functional.cosine_similarity(*grads, dim=0))))
        write_json(RUN / 'DIRECTIONS.json', directions)
        jobs = []
        for s in range(3):
            for group in GROUPS:
                method = group.split('_')[0] if group != 'Base' else 'Full'
                step = f'step_{s}_{group}.pt' if group != 'Base' else None
                for f in range(36):
                    for i in range(2):
                        jobs.append(dict(role='transfer', seed=s, method=method, group=group,
                            fid=f, instance=i, step=step, step_hash=sha256(RUN/step) if step else None))
        values = execute(pool, jobs, 'transfer')
        matrix = {g: np.zeros((3, 36, 2, 4)) for g in GROUPS}
        for spec, value in zip(jobs, values):
            matrix[spec['group']][spec['seed'], spec['fid'], spec['instance']] = value['utility'].numpy()
        save_torch(RUN / 'utilities.pt', matrix)
    primary = {g: interval(matrix[g]-matrix['Base']) for g in ('Low_A', 'Low_B')}
    descriptive = {g+'-Base': interval(matrix[g]-matrix['Base'], .95) for g in ('Full_A', 'Full_B')}
    descriptive.update({f'Low_{g}-Full_{g}': interval(matrix[f'Low_{g}']-matrix[f'Full_{g}'], .95) for g in ('A', 'B')})
    result = dict(primary=primary, descriptive=descriptive, low_reliable=all(v['passed'] for v in primary.values()),
        estimation_calls=1382400, transfer_calls=2592000, contract_calls=2400, total_calls=3976800,
        teacher_calls=0, seconds=time.perf_counter()-start, workers=workers, directions=directions,
        all_workers_joined=True, development_only=True)
    write_json(RUN / 'COMPLETE.json', result)
    write_json(OUT / 'RESULTS.json', result)
    write_json(RUN / 'STATUS.json', dict(status='complete'))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
