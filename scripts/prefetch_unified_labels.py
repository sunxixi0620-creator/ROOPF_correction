"""Collect a completed seed while other independent anchors are training."""
import argparse
import json
import os
from pathlib import Path
import time
from unified_revision import verify, collect, pool_run
from roopf.experiment_io import sha256, write_json


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--seed', type=int, required=True, choices=[0, 1, 2])
    p.add_argument('--workers', type=int, default=8)
    a = p.parse_args()
    run, _ = verify(a.run)
    done = json.loads((run/f'anchor_{a.seed}/COMPLETE.json').read_text())
    checkpoint = run/f'anchor_{a.seed}/selected.pt'
    assert sha256(checkpoint) == done['selected_sha256']
    write_json(run/f'prefetch_{a.seed}.json', {'pid': os.getpid(), 'status': 'running',
        'anchor_sha256': done['selected_sha256'], 'source_sha256': sha256(__file__)})
    start = time.perf_counter()
    jobs = [(str(run), a.seed, split, fid, i)
            for split, count in [('residual_training', 2), ('residual_validation', 1)]
            for fid in range(36) for i in range(count)]
    pool_run(collect, jobs, a.workers)
    assert sha256(checkpoint) == done['selected_sha256']
    write_json(run/f'prefetch_{a.seed}.json', {'status': 'complete', 'all_workers_joined': True,
        'cases': len(jobs), 'seconds': time.perf_counter()-start,
        'anchor_sha256': done['selected_sha256'], 'source_sha256': sha256(__file__)})
