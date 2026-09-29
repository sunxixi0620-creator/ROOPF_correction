"""Verified continuation to1000, isolated from all completed experiment sources."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import contextlib
import json
import multiprocessing as mp
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from roopf.anchor_backbone import AnchorPolicyBackbone
from roopf.revision_tasks import ProceduralTask, population
from roopf.experiment_io import CaseStore, canonical, save_torch, seed_for, sha256, write_json
from scripts.unified_revision import validate_anchor

SOURCES = ['scripts/complementarity_training.py', 'scripts/unified_revision.py',
           'roopf/anchor_backbone.py', 'roopf/revision_tasks.py', 'roopf/experiment_io.py',
           'docs/experiments/COMPLEMENTARITY_TRAINING_PROTOCOL.md']
PREFIX = {10: ROOT/'results/unified_revision_v1_20260928_1840',
          20: ROOT/'results/unified_native20_20260928'}
MILESTONES = (80, 160, 320, 640, 1000)


def verify(run):
    run = Path(run)
    identity = json.loads((run/'identity.json').read_text())
    for name, digest in identity['sources'].items():
        if sha256(ROOT/name) != digest:
            raise ValueError(f'Frozen training source changed: {name}')
    if identity['torch'] != torch.__version__ or identity['numpy'] != np.__version__:
        raise ValueError('Training environment mismatch')
    return run, identity


def prepare(run):
    run = Path(run)
    if (run/'identity.json').exists():
        verify(run)
        return
    run.mkdir(parents=True, exist_ok=True)
    sources = {p: sha256(ROOT/p) for p in SOURCES}
    prefixes = {}
    for dim in (10, 20):
        old = json.loads((PREFIX[dim]/'identity.json').read_text())
        for seed in range(3):
            path = PREFIX[dim]/f'anchor_{seed}/state'
            store = CaseStore(path, old)
            state = store.load('training', {'seed': seed})
            assert state is not None and state['epoch'] == (30 if dim == 10 and seed == 1 else 80)
            assert state['history'][-1]['epoch'] == state['epoch']
            assert state['optimizer']['state']
            key = f'd{dim}_s{seed}'
            destination = run/'prefix'/f'{key}.pt'
            destination.parent.mkdir(exist_ok=True)
            shutil.copyfile(path/'training.pt', destination)
            assert sha256(destination) == sha256(path/'training.pt')
            prefixes[key] = dict(source=str(path/'training.pt'), sha256=sha256(destination),
                epoch=state['epoch'], source_identity=old,
                source_manifest=json.loads((path/'training.json').read_text()))
    identity = dict(version='complementarity_training_v1', sources=sources, prefixes=prefixes,
        torch=torch.__version__, numpy=np.__version__, max_epochs=1000,
        width=200, batch=16, lr=.001, accumulation=4, validation_every=5,
        early_stop=False, dimensions=[10, 20], seeds=[0, 1, 2], milestones=MILESTONES)
    write_json(run/'identity.json', identity)
    for name in SOURCES:
        target = run/'source_snapshot'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/name, target)
    write_json(run/'STATUS.json', dict(status='prepared', completed=0, expected=6))


def epoch_step(model, optimizer, dim, seed, epoch, device):
    model.train()
    optimizer.zero_grad()
    total, points = 0., 0
    for fid in range(36):
        torch.manual_seed(seed_for('anchor', seed, epoch, fid, 'dropout'))
        task = ProceduralTask(fid, epoch, 'anchor_training', dim=dim, device=device)
        pop = population(fid, epoch, 'anchor_training', count=16, dim=dim, device=device)
        _, _, nfe, candidates = model(pop, task)
        parent = task.initial_values
        child = task.calfitness(candidates)
        loss = ((child.min(1).values-parent.min(1).values)/parent.std(1).clamp_min(1e-8)).mean()/4
        assert nfe == 300 and torch.isfinite(loss)
        loss.backward()
        total += float(loss.detach())*4/36
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        points += task.points
        if (fid+1)%4 == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
            optimizer.step()
            optimizer.zero_grad()
    assert points == 288000
    return total, points


def train(job):
    run, dim, seed = job
    run, identity = verify(run)
    torch.set_num_threads(1)
    key = f'd{dim}_s{seed}'
    out = run/key
    out.mkdir(exist_ok=True)
    store = CaseStore(out/'state', identity)
    with store.lock('training'), (out/'worker.log').open('a', buffering=1) as log, contextlib.redirect_stdout(log):
        if (out/'COMPLETE.json').exists():
            done = json.loads((out/'COMPLETE.json').read_text())
            assert sha256(out/'selected.pt') == done['selected_sha256']
            return done
        state = store.load('training', {'dimension': dim, 'seed': seed})
        prefix = identity['prefixes'][key]
        if state is None:
            source = run/'prefix'/f'{key}.pt'
            assert sha256(source) == prefix['sha256']
            state = torch.load(source, map_location='cpu', weights_only=False)
            state.update(extension_seconds=0., prefix_train_points=state['train_points'],
                         prefix_val_points=state['val_points'], prefix_epoch=state['epoch'])
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = AnchorPolicyBackbone(dim, 200, 100).to(device)
        model.load_state_dict(state['model'], strict=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=.001)
        optimizer.load_state_dict(state['optimizer'])
        assert all(group['lr'] == .001 for group in optimizer.param_groups)
        assert all(int(v['step']) == state['epoch']*9 for v in optimizer.state.values())
        save_torch(out/'selected.pt', state['selected_model'])
        if state['epoch'] in MILESTONES:
            save_torch(out/f'epoch{state["epoch"]}.pt', state)
        start = time.perf_counter()
        previous_seconds = state['extension_seconds']
        for epoch in range(state['epoch']+1, 1001):
            loss, points = epoch_step(model, optimizer, dim, seed, epoch, device)
            score = None
            state['train_points'] += points
            if epoch%5 == 0:
                score, cost = validate_anchor(model)
                state['val_points'] += cost
                if score > state['best']+1e-5:
                    state['best'], state['selected_epoch'] = score, epoch
                    state['selected_model'] = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                    save_torch(out/'selected.pt', state['selected_model'])
            elapsed = previous_seconds+time.perf_counter()-start
            state['history'].append(dict(epoch=epoch, loss=loss, validation=score,
                extension_seconds=elapsed, training_points=state['train_points'], updates=epoch*9))
            state.update(model=model.state_dict(), optimizer=optimizer.state_dict(), epoch=epoch,
                         extension_seconds=elapsed)
            store.save('training', {'dimension': dim, 'seed': seed}, state)
            write_json(out/'history.json', state['history'])
            write_json(out/'PROGRESS.json', dict(dimension=dim, seed=seed, epoch=epoch,
                selected_epoch=state['selected_epoch'], best_validation=state['best'],
                extension_seconds=elapsed, updates=epoch*9, pid=os.getpid()))
            if epoch in MILESTONES:
                save_torch(out/f'epoch{epoch}.pt', state)
            print(key, epoch, 'validation', score, 'best', state['best'], flush=True)
        done = dict(dimension=dim, seed=seed, epoch=state['epoch'],
            selected_epoch=state['selected_epoch'], best_validation=state['best'],
            selected_sha256=sha256(out/'selected.pt'), parameter_count=sum(p.numel() for p in model.parameters()),
            training_points=state['train_points'], validation_points=state['val_points'],
            extension_training_points=state['train_points']-state['prefix_train_points'],
            extension_validation_points=state['val_points']-state['prefix_val_points'],
            extension_seconds=state['extension_seconds'], updates=9000,
            extension_updates=9*(1000-state['prefix_epoch']), device=str(device),
            peak_gpu_bytes=torch.cuda.max_memory_allocated() if device.type == 'cuda' else 0)
        write_json(out/'COMPLETE.json', done)
        return done


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['prepare', 'train'])
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--workers', type=int, default=3)
    args = parser.parse_args()
    if args.mode == 'prepare':
        prepare(args.run)
        return
    verify(args.run)
    # Three20D runs first: their previous bests lie near the80-epoch boundary.
    jobs = [(args.run, dim, seed) for dim in (20, 10) for seed in range(3)]
    write_json(args.run/'STATUS.json', dict(status='training', expected=6, completed=0, supervisor_pid=os.getpid()))
    completed = []
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context('spawn')) as executor:
        futures = [executor.submit(train, job) for job in jobs]
        for future in as_completed(futures):
            completed.append(future.result())
            write_json(args.run/'STATUS.json', dict(status='training', expected=6,
                completed=len(completed), models=completed, supervisor_pid=os.getpid()))
    write_json(args.run/'COMPLETE.json', dict(status='complete_anchor_duration',
        all_workers_joined=True, models=completed))
    write_json(args.run/'STATUS.json', dict(status='complete_anchor_duration', expected=6, completed=6))


if __name__ == '__main__':
    main()
