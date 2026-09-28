"""Use idle CPU capacity on the remaining frozen cases in reverse order."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from pathlib import Path
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'scripts')]
import unified_external as exp
from roopf.experiment_io import write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=12)
    args = parser.parse_args(); run = args.run.resolve()
    exp.verify(str(run))
    jobs = [(str(run), d, b, f, r, m) for d, b in exp.CONDITIONS for f in range(1, 13)
            for r in range(30) for m in exp.METHODS[-2:]]
    # Existence only determines scheduling; actual cache reuse always verifies hashes
    # inside exp.worker. A lock spans checking, optimization and atomic publication.
    jobs = [job for job in reversed(jobs) if not (run/'cases'/
        f'd{job[1]}_b{job[2]}_f{job[3]:02d}_r{job[4]:02d}_{job[5]}.json').exists()]
    start = time.perf_counter()
    with ProcessPoolExecutor(args.workers, mp_context=mp.get_context('spawn')) as pool:
        futures = [pool.submit(exp.worker, job) for job in jobs]
        for i, future in enumerate(as_completed(futures)):
            future.result()
            if (i+1)%30 == 0 or i+1 == len(jobs):
                print('baseline overlap', i+1, '/', len(jobs), 'seconds', time.perf_counter()-start, flush=True)
    write_json(run/'baseline_overlap_COMPLETE.json', dict(all_workers_joined=True,
        scheduled_cases=len(jobs), seconds=time.perf_counter()-start, algorithm_changes=False))


if __name__ == '__main__':
    main()
