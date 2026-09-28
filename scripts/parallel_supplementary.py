"""Resume a predeclared supplementary protocol with independent CPU workers.

Performance RNG/batch definitions are unchanged. Parallel wall times are not
used as uncontended runtime evidence. Only the parent writes aggregate files.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing
import os
from pathlib import Path


def worker(job):
    from supplementary_experiments import run, write_csv
    import numpy as np
    dest=Path(job['dest'])
    if (dest/'COMPLETE').exists():
        return json.loads((dest/'rows.json').read_text()), False
    dest.mkdir(exist_ok=True)
    rows,opt,trail,instance=run(job['case'],job['variant'],job['seeds'],job['algorithm_seed'],
        job['warmup'],job['diagnostic_stride'],job['shift_seed'])
    (dest/'rows.json').write_text(json.dumps(rows,indent=2))
    (dest/'instance.json').write_text(json.dumps(instance,indent=2))
    write_csv(dest/'candidates.csv',opt.candidate_samples);write_csv(dest/'decisions.csv',opt.decisions)
    np.savez_compressed(dest/'trails.npz',trail=trail.cpu().numpy(),nfe=np.arange(102,301,2),seeds=job['seeds'])
    (dest/'COMPLETE').write_text('Exact budget, finite values, monotonic trace and candidate count passed.\n')
    return rows, True


def main():
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);p.add_argument('--workers',type=int,default=8)
    args=p.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='', 'CPU workers require CUDA_VISIBLE_DEVICES empty'
    from supplementary_experiments import ROOT, r, write_csv
    protocol=json.loads((args.directory/'protocol.json').read_text())
    for rel,h in protocol['hashes'].items():
        assert r.sha256(ROOT/rel)==h, f'Source mismatch: {rel}'
    jobs=[];existing=[];rows=[]
    for case_index,case in enumerate(protocol['cases_expanded']):
        variants=protocol['variants'].split(',')
        order=variants[case_index%len(variants):]+variants[:case_index%len(variants)]
        for start in range(0,protocol['seeds'],protocol['batch_size']):
            seeds=list(range(protocol['seed_start']+start,protocol['seed_start']+min(protocol['seeds'],start+protocol['batch_size'])))
            for warmup in map(float,protocol['warmups'].split(',')):
                for variant in order:
                    dest=args.directory/f'{case}_{variant}_w{warmup:.4f}_s{seeds[0]}'
                    job={'dest':str(dest),'case':case,'variant':variant,'seeds':seeds,
                        'algorithm_seed':20264000+int(case.split('_')[1][1:])*100+start,
                        'warmup':warmup,'diagnostic_stride':protocol['diagnostic_stride'],'shift_seed':protocol['shift_seed']}
                    if (dest/'COMPLETE').exists():
                        rows.extend(json.loads((dest/'rows.json').read_text()));existing.append(dest.name)
                    else:jobs.append(job)
    schedule={'workers':args.workers,'source_sha256':r.sha256(Path(__file__)),
        'existing_complete_batches':existing,'parallel_batches':[Path(j['dest']).name for j in jobs],
        'runtime_warning':'Parallel and previously serial times are not pooled as controlled runtime measurements. Use separate serial runtime replication.'}
    (args.directory/'parallel_execution.json').write_text(json.dumps(schedule,indent=2))
    print(json.dumps({'existing_trajectories':len(rows),'pending_batches':len(jobs),'workers':args.workers}),flush=True)
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        futures={pool.submit(worker,j):j for j in jobs}
        for future in as_completed(futures):
            result,executed=future.result();rows.extend(result)
            write_csv(args.directory/'raw_results.csv',rows)
            j=futures[future]
            print(json.dumps({'case':j['case'],'variant':j['variant'],'completed_trajectories':len(rows)}),flush=True)
    expected=len(protocol['cases_expanded'])*len(protocol['variants'].split(','))*len(protocol['warmups'].split(','))*protocol['seeds']
    assert len(rows)==expected
    assert len({(x['case'],x['variant'],x['warmup'],x['seed']) for x in rows})==expected
    write_csv(args.directory/'raw_results.csv',rows)
    (args.directory/'COMPLETE').write_text(f'{len(rows)} trajectories completed.\n')

if __name__=='__main__':main()
