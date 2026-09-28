"""Bounded, resumable execution of the preregistered unified revision."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import argparse
import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from roopf.anchor_backbone import AnchorPolicyBackbone
from roopf.revision_tasks import ProceduralTask, population
from roopf.unified import build_unified, VARIANTS
from roopf.experiment_io import (CaseStore, canonical, fingerprint, save_torch,
                                 seed_for, sha256, write_json)

SOURCES = ['scripts/unified_revision.py', 'roopf/unified.py', 'roopf/model.py',
           'roopf/anchor_backbone.py', 'roopf/revision_tasks.py',
           'roopf/experiment_io.py', 'roopf/residual_features.py',
           'docs/experiments/UNIFIED_REVISION_PROTOCOL.md',
           'docs/experiments/UNIFIED_EXTERNAL_PROTOCOL.md']
TEMPLATE = ROOT/'checkpoints/residual_selector_generated36_d10.pt'


def identity(dimension=10):
    return dict(version='unified_revision_v1', sources={p: sha256(ROOT/p) for p in SOURCES},
        originals={p.name: sha256(p) for p in (ROOT/'checkpoints').glob('*.pt')},
        torch=torch.__version__, numpy=np.__version__, dimension=dimension, budget=300,
        model_seeds=[0, 1, 2], max_epochs=80, development_instances=2, populations=4)


def verify(run):
    run = Path(run)
    expected = json.loads((run/'identity.json').read_text())
    if expected != canonical(identity(expected['dimension'])):
        raise ValueError('Frozen source/configuration changed; cannot resume this run')
    return run, expected


def bounded(initial, final, scale):
    g = ((initial-final)/scale.clamp_min(1e-8)).clamp_min(0)
    return g/(1+g)


def validate_anchor(model):
    device = next(model.parameters()).device
    dimension = model.generator.dim
    scores = []
    points = 0
    model.eval()
    with torch.no_grad():
        for fid in range(36):
            task = ProceduralTask(fid, 0, 'anchor_validation', dim=dimension, device=device)
            pop = population(fid, 0, 'anchor_validation', count=2, dim=dimension, device=device)
            _, trail, nfe, _ = model(pop, task)
            initial_y = task.initial_values
            assert nfe == 300 and task.points == 600
            assert torch.isfinite(trail).all()
            scores.extend(bounded(initial_y.min(1).values, trail[:, -1], initial_y.std(1)).cpu().tolist())
            points += task.points
    return float(np.mean(scores)), points


def train_anchor(job):
    run, seed = job
    run, protocol = verify(run)
    dimension = protocol['dimension']
    out = run/f'anchor_{seed}'
    out.mkdir(exist_ok=True)
    store = CaseStore(out/'state', protocol)
    torch.set_num_threads(1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    start = time.perf_counter()
    with store.lock('training'):
        if (out/'COMPLETE.json').exists():
            done = json.loads((out/'COMPLETE.json').read_text())
            assert sha256(out/'selected.pt') == done['selected_sha256']
            return done
        torch.manual_seed(seed_for('anchor', seed, 'init'))
        model = AnchorPolicyBackbone(dimension, 200, 100).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=.001)
        state = store.load('training', {'seed': seed})
        history, begin, stale, train_points, val_points, elapsed = [], 1, 0, 0, 0, 0.
        if state is not None:
            model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer'])
            history = state['history']; begin = state['epoch']+1
            best, selected_epoch, stale = state['best'], state['selected_epoch'], state['stale']
            train_points, val_points, elapsed = state['train_points'], state['val_points'], state['seconds']
            best_state = state['selected_model']
            save_torch(out/'selected.pt', best_state)
        else:
            best, cost = validate_anchor(model); val_points += cost; selected_epoch = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            save_torch(out/'selected.pt', best_state)
            history.append(dict(epoch=0, validation=best, loss=None))
            print('ANCHOR', seed, 'validation0', best, flush=True)
        for epoch in range(begin, 81):
            if epoch > 20 and stale >= 3:
                break
            model.train(); optimizer.zero_grad(); total = 0.
            for fid in range(36):
                torch.manual_seed(seed_for('anchor', seed, epoch, fid, 'dropout'))
                task = ProceduralTask(fid, epoch, 'anchor_training', dim=dimension, device=device)
                pop = population(fid, epoch, 'anchor_training', count=16, dim=dimension, device=device)
                _, _, nfe, candidates = model(pop, task)
                parent = task.initial_values
                child = task.calfitness(candidates)
                loss = ((child.min(1).values-parent.min(1).values)/parent.std(1).clamp_min(1e-8)).mean()/4
                assert torch.isfinite(loss)
                loss.backward(); total += float(loss.detach())*4/36
                assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
                train_points += task.points
                if (fid+1)%4 == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
                    optimizer.step(); optimizer.zero_grad()
                del candidates, child, loss, task
            score = None
            if epoch%5 == 0:
                score, cost = validate_anchor(model); val_points += cost
                if score > best+1e-5:
                    best, stale, selected_epoch = score, 0, epoch
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    save_torch(out/'selected.pt', best_state)
                else:
                    stale += 1
            history.append(dict(epoch=epoch, loss=total, validation=score,
                seconds=elapsed+time.perf_counter()-start, training_points=train_points))
            write_json(out/'history.json', history)
            state = dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                history=history, epoch=epoch, best=best, selected_epoch=selected_epoch,
                stale=stale, train_points=train_points, val_points=val_points,
                seconds=elapsed+time.perf_counter()-start, selected_model=best_state)
            store.save('training', {'seed': seed}, state)
            print('ANCHOR', seed, epoch, 'val', score, 'best', best, flush=True)
        done = dict(seed=seed, epochs=history[-1]['epoch'], selected_epoch=selected_epoch,
            best_validation=best, selected_sha256=sha256(out/'selected.pt'), training_points=train_points,
            validation_points=val_points, seconds=elapsed+time.perf_counter()-start,
            peak_gpu_bytes=torch.cuda.max_memory_allocated() if device.type == 'cuda' else 0)
        write_json(out/'COMPLETE.json', done)
        return done


def task_identity(task, pop, seed, variant):
    return dict(fid=task.fid, instance=task.instance, split=task.split,
        parameter_seed=task.parameter_seed, parameters=task.params,
        population_seed=seed_for('revision_v1', task.split, 'population', task.fid, task.instance, task.dim),
        population=pop, policy_seed=seed, variant=variant)


def collect(job):
    run, model_seed, split, fid, instance = job
    run, protocol = verify(run); torch.set_num_threads(1)
    anchor = run/f'anchor_{model_seed}/selected.pt'
    store = CaseStore(run/f'labels_{model_seed}', {**protocol, 'anchor_sha256': sha256(anchor)})
    task = ProceduralTask(fid, instance, split, dim=protocol['dimension'])
    pop = population(fid, instance, split, count=2, dim=protocol['dimension'])
    policy_seed = seed_for('revision_v1', split, 'policy', fid, instance)
    case_identity = task_identity(task, pop, policy_seed, 'no_residual')
    key = f'{split}_{fid:02d}_{instance}'
    with store.lock(key):
        value = store.load(key, case_identity)
        if value is not None:
            return key
        opt = build_unified(anchor, TEMPLATE, 'no_residual', dim=protocol['dimension'])
        records = []
        def teacher(features, truth, incumbent, nfe):
            records.append((features.reshape(-1, 27),
                (truth < incumbent[:, None]-1e-12).float().reshape(-1), nfe))
        opt.teacher_observer = teacher
        torch.manual_seed(policy_seed)
        _, trail, nfe, points = opt(pop.clone(), task)
        assert nfe == 300 and task.points == 600 and task.diagnostic_points == 7600
        value = dict(x=torch.cat([v[0] for v in records]), y=torch.cat([v[1] for v in records]),
            nfe=torch.tensor([v[2] for v in records]), trail=trail, points=points,
            main_points=task.points, teacher_points=task.diagnostic_points,
            policy_rng_after=torch.get_rng_state())
        store.save(key, case_identity, value)
    return key


def labels(run, seed, split):
    run, protocol = verify(run)
    anchor = run/f'anchor_{seed}/selected.pt'
    store = CaseStore(run/f'labels_{seed}', {**protocol, 'anchor_sha256': sha256(anchor)})
    cases = []
    for fid in range(36):
        for instance in range(2 if split == 'residual_training' else 1):
            task = ProceduralTask(fid, instance, split, dim=protocol['dimension'])
            pop = population(fid, instance, split, count=2, dim=protocol['dimension'])
            ps = seed_for('revision_v1', split, 'policy', fid, instance)
            key = f'{split}_{fid:02d}_{instance}'
            value = store.load(key, task_identity(task, pop, ps, 'no_residual'))
            if value is None:
                raise ValueError(f'Missing label artifact {key}')
            cases.append(value)
    return torch.cat([v['x'] for v in cases]), torch.cat([v['y'] for v in cases])


def train_residual(job):
    run, seed = job
    run, protocol = verify(run); torch.set_num_threads(1)
    out = run/f'residual_{seed}'; out.mkdir(exist_ok=True)
    if (out/'COMPLETE.json').exists():
        done = json.loads((out/'COMPLETE.json').read_text())
        assert done['selected_sha256'] == sha256(out/'selected.pt')
        return done
    start = time.perf_counter()
    x, y = labels(run, seed, 'residual_training')
    vx, vy = labels(run, seed, 'residual_validation')
    mean, std = x.mean(0), x.std(0, unbiased=False).clamp_min(1e-6)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x, vx = ((x-mean)/std).to(dev), ((vx-mean)/std).to(dev)
    y, vy = y.to(dev), vy.to(dev)
    torch.manual_seed(seed_for('residual', seed, 'init'))
    model = nn.Sequential(nn.Linear(27, 128), nn.LayerNorm(128), nn.GELU(),
        nn.Linear(128, 128), nn.GELU(), nn.Linear(128, 1)).to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    history = []; best = float('inf'); stale = 0
    # Restart an interrupted small residual fit deterministically; partial histories
    # never count as separate models and selected checkpoint is rewritten atomically.
    for epoch in range(1, 61):
        model.train(); total = 0.
        for ix in torch.randperm(len(x), device=dev).split(4096):
            optimizer.zero_grad()
            loss = nn.functional.binary_cross_entropy_with_logits(model(x[ix]).flatten(), y[ix])
            assert torch.isfinite(loss)
            loss.backward(); optimizer.step(); total += float(loss.detach())*len(ix)
        model.eval()
        with torch.no_grad():
            value = sum(float(nn.functional.binary_cross_entropy_with_logits(model(a).flatten(), b,
                reduction='sum')) for a, b in zip(vx.split(16384), vy.split(16384)))/len(vx)
        if value < best-1e-5:
            best, stale, selected_epoch = value, 0, epoch
            payload = torch.load(TEMPLATE, map_location='cpu', weights_only=False)
            payload = {k: payload[k] for k in ['feature_names']}
            payload.update(state_dict={'net.'+k: v.detach().cpu() for k, v in model.state_dict().items()},
                mean=mean.tolist(), std=std.tolist(), epoch=epoch, model_seed=seed,
                target='incumbent_improvement', feature_context='unified38_actual_network_input',
                anchor_sha256=sha256(run/f'anchor_{seed}/selected.pt'))
            save_torch(out/'selected.pt', payload)
        else:
            stale += 1
        history.append(dict(epoch=epoch, train_bce=total/len(x), validation_bce=value))
        write_json(out/'history.json', history)
        print('RESIDUAL', seed, epoch, 'val', value, flush=True)
        if stale >= 8:
            break
    done = dict(seed=seed, epochs=epoch, selected_epoch=selected_epoch, validation_bce=best,
        train_rows=len(x), validation_rows=len(vx), training_positive_rate=float(y.mean()),
        selected_sha256=sha256(out/'selected.pt'), seconds=time.perf_counter()-start)
    write_json(out/'COMPLETE.json', done)
    return done


def evaluate_case(job):
    run, seed, fid, instance = job
    run, protocol = verify(run); torch.set_num_threads(1)
    anchor, residual = run/f'anchor_{seed}/selected.pt', run/f'residual_{seed}/selected.pt'
    store = CaseStore(run/f'development_{seed}', {**protocol,
        'anchor_sha256': sha256(anchor), 'residual_sha256': sha256(residual)})
    ps = seed_for('revision_v1', 'development', 'policy', fid, instance)
    pop = population(fid, instance, 'development', count=4, dim=protocol['dimension'])
    rows = []
    for variant in VARIANTS:
        task = ProceduralTask(fid, instance, 'development', dim=protocol['dimension'])
        case_identity = task_identity(task, pop, ps, variant)
        key = f'{fid:02d}_{instance}_{variant}'
        with store.lock(key):
            value = store.load(key, case_identity)
            if value is None:
                model = build_unified(anchor, residual, variant, dim=protocol['dimension'])
                torch.manual_seed(ps)
                start = time.perf_counter()
                _, trail, nfe, points = model(pop.clone(), task)
                initial = task.initial_values
                seconds = time.perf_counter()-start
                assert nfe == 300 and task.points == 1200 and task.diagnostic_points == 0
                assert torch.isfinite(trail).all() and (trail[:, 1:] <= trail[:, :-1]).all()
                value = dict(trail=trail, points=points, decisions=model.decisions,
                    initial_best=initial.min(1).values, initial_std=initial.std(1),
                    nfe=nfe, seconds=seconds, population_sha256=fingerprint(pop))
                store.save(key, case_identity, value)
        u = bounded(value['initial_best'], value['trail'][:, -1], value['initial_std'])
        early = bounded(value['initial_best'], value['trail'][:, 54], value['initial_std'])  # NFE210
        for i in range(4):
            ds = [d for d in value['decisions'] if d['batch'] == i]
            rows.append(dict(model_seed=seed, fid=fid, instance=instance, population=i,
                method=variant, initial=float(value['initial_best'][i]), final=float(value['trail'][i, -1]),
                utility=float(u[i]), early_utility=float(early[i]), nfe=value['nfe'],
                accepted=sum(d['accepted'] for d in ds),
                early_accepted=sum(d['accepted'] for d in ds if d['eval_before'] < 210),
                residual_changed=sum(d['residual_changed'] for d in ds),
                batch_seconds=value['seconds'], population_sha256=value['population_sha256']))
    return rows


def summarize(run, rows):
    data = pd.DataFrame(rows).sort_values(['model_seed', 'fid', 'instance', 'population', 'method'])
    data.to_csv(run/'development.csv', index=False)
    assert len(data) == 5184
    pivot = data.pivot(index=['model_seed', 'fid', 'instance', 'population'], columns='method', values='utility')
    rng = np.random.default_rng(20281129)
    # Shared paired resampling, clustered by12 coefficient families. All3 fixed
    # scales and4 populations retained inside a sampled family/instance.
    draws = [(rng.integers(0, 3, 3), rng.integers(0, 12, 12), rng.integers(0, 2, (12, 2)))
             for _ in range(5000)]
    def effect(a, b, risk=False):
        if risk:
            diff = ((pivot[b]-pivot['anchor_only'] < -.02).astype(float)
                    -(pivot[a]-pivot['anchor_only'] < -.02).astype(float))
        else:
            diff = pivot[a]-pivot[b]
        values = diff.to_numpy().reshape(3, 12, 3, 2, 4).mean(-1)
        samples = []
        for seeds, families, instances in draws:
            sample = [values[seeds][:, family, :, instances[j]].mean()
                      for j, family in enumerate(families)]
            samples.append(float(np.mean(sample)))
        lo, hi = np.quantile(samples, [.025, .975])
        by_seed = values.mean((1, 2, 3)).tolist()
        mean = float(values.mean())
        return dict(mean=mean, ci95=[float(lo), float(hi)], by_seed=by_seed,
            practical_pass=bool(mean >= .005 and lo > 0 and all(v > 0 for v in by_seed)))
    comparisons = {a+'_vs_'+b: effect(a, b) for a, b in [
        ('full', 'anchor_only'), ('no_residual', 'anchor_only'), ('full', 'no_residual'),
        ('full', 'score_only'), ('full', 'late_full'), ('no_residual', 'late_no_residual')]}
    comparisons['protection_risk_reduction'] = effect('full', 'score_only', risk=True)
    pass_full = comparisons['full_vs_anchor_only']['practical_pass']
    pass_simple = comparisons['no_residual_vs_anchor_only']['practical_pass']
    full_residual = comparisons['full_vs_no_residual']['practical_pass']
    survivor = 'full' if pass_full and full_residual else ('no_residual' if pass_simple else ('full' if pass_full else None))
    result = dict(comparisons=comparisons, surviving_candidate=survivor,
        decision='STOP_BEFORE_FINAL_TEST' if survivor is None else 'REQUIRES_FROZEN_EXTERNAL_PROTOCOL',
        residual_independent_gain=full_residual,
        protection_supported=bool(comparisons['full_vs_score_only']['ci95'][0] >= -.005 and
                                  comparisons['protection_risk_reduction']['ci95'][0] > 0),
        method_means=data.groupby('method')[['utility', 'early_utility', 'accepted', 'early_accepted', 'residual_changed']].mean().to_dict('index'),
        protocol_sha256=sha256(ROOT/'docs/experiments/UNIFIED_REVISION_PROTOCOL.md'))
    write_json(run/'DECISION.json', result)
    return result


def pool_run(function, jobs, workers):
    with ProcessPoolExecutor(workers, mp_context=mp.get_context('spawn')) as executor:
        futures = [executor.submit(function, job) for job in jobs]
        results = []
        for i, future in enumerate(as_completed(futures)):
            results.append(future.result())
            if (i+1)%12 == 0 or i+1 == len(futures):
                print(function.__name__, i+1, '/', len(futures), flush=True)
    return results  # context joins all workers before the next phase or COMPLETE


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['prepare', 'all', 'anchor', 'labels', 'residual', 'evaluate'])
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--gpu-workers', type=int, default=3)
    parser.add_argument('--dimension', type=int, choices=[10, 20], default=10)
    args = parser.parse_args()
    run = args.run.resolve()
    if args.mode == 'prepare':
        run.mkdir(parents=True, exist_ok=False)
        write_json(run/'identity.json', identity(args.dimension))
        write_json(run/'status.json', {'stage': 'prepared', 'pid': os.getpid()})
        print(run, flush=True)
        return
    verify(run)
    start = time.perf_counter()
    write_json(run/'status.json', {'stage': args.mode, 'pid': os.getpid()})
    if args.mode in ('all', 'anchor'):
        write_json(run/'status.json', {'stage': 'anchor_training', 'pid': os.getpid()})
        pool_run(train_anchor, [(str(run), s) for s in range(3)], args.gpu_workers)
    if args.mode in ('all', 'labels'):
        write_json(run/'status.json', {'stage': 'label_collection', 'pid': os.getpid()})
        jobs = [(str(run), s, split, f, i) for s in range(3)
                for split, count in [('residual_training', 2), ('residual_validation', 1)]
                for f in range(36) for i in range(count)]
        pool_run(collect, jobs, args.workers)
    if args.mode in ('all', 'residual'):
        write_json(run/'status.json', {'stage': 'residual_training', 'pid': os.getpid()})
        pool_run(train_residual, [(str(run), s) for s in range(3)], args.gpu_workers)
        write_json(run/'SELECTION_FROZEN.json', {str(p.relative_to(run)): sha256(p)
            for p in run.glob('*/selected.pt')})
    if args.mode in ('all', 'evaluate'):
        write_json(run/'status.json', {'stage': 'development_evaluation', 'pid': os.getpid()})
        frozen = json.loads((run/'SELECTION_FROZEN.json').read_text())
        assert len(frozen) == 6 and all(sha256(run/p) == h for p, h in frozen.items())
        jobs = [(str(run), s, f, i) for s in range(3) for f in range(36) for i in range(2)]
        results = pool_run(evaluate_case, jobs, args.workers)
        result = summarize(run, [row for batch in results for row in batch])
        write_json(run/'COMPLETE.json', dict(seconds_this_session=time.perf_counter()-start,
            decision=result['decision'], all_workers_joined=True))
        write_json(run/'status.json', {'stage': 'complete', 'decision': result['decision']})
        print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
