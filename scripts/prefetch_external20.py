"""Overlap external inference only after each native anchor has finished selection."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'scripts')]
import unified_external as exp
from roopf.experiment_io import sha256, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--training-pid', type=int, required=True)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args(); run = args.run.resolve()
    exp.verify(str(run)); pending = set(range(3))
    while pending:
        ready = [s for s in sorted(pending) if (exp.NATIVE20/f'anchor_{s}/COMPLETE.json').exists()]
        if not ready:
            if not Path(f'/proc/{args.training_pid}').exists():
                raise RuntimeError('Training stopped without all selected anchors')
            time.sleep(15)
            continue
        seed = ready[0]
        checkpoint = exp.checkpoint(20, seed)
        weight_hash = sha256(checkpoint)
        jobs = [(str(run), d, b, f, r, m) for d, b in exp.CONDITIONS[1:]
                for f in range(1, 13) for r in range(seed*10, (seed+1)*10) for m in exp.METHODS[:3]]
        start = time.perf_counter()
        write_json(run/f'prefetch20_{seed}.json', dict(status='running', anchor_sha256=weight_hash))
        with ProcessPoolExecutor(args.workers, mp_context=mp.get_context('spawn')) as pool:
            futures = [pool.submit(exp.worker, job) for job in jobs]
            for i, future in enumerate(as_completed(futures)):
                future.result()
                if (i+1)%60 == 0:
                    print('native20 inference', seed, i+1, '/', len(jobs), flush=True)
        assert sha256(checkpoint) == weight_hash
        write_json(run/f'prefetch20_{seed}.json', dict(status='complete', all_workers_joined=True,
            cases=len(jobs), anchor_sha256=weight_hash, seconds=time.perf_counter()-start))
        pending.remove(seed)
    print('All native20 inference workers joined', flush=True)


if __name__ == '__main__':
    main()
