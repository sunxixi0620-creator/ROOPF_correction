"""Engineering only; paired original RNG, all timing settings, zero API requests."""
import gzip,json,sys,time
from pathlib import Path
import numpy as np
import torch
import experiment_r14_timing as run


def main():
    directory=run.EXP/'engineering_checks';directory.mkdir(exist_ok=True)
    run.configure(str(run.EXP),run.read(run.EXP/'HARDWARE_RUNTIME.json'))
    checks=[]
    for group,fid in [('BBOB',1),('CEC-style',1),('UAV',1)]:
        cache=run.calls_for(group,fid)[0]
        start,source,pop,rng,schedule,baseline,basepoints=run.original_batch(group,fid,0,cache)
        canonical=None
        for arm in ['canonical',*run.ARMS]:
            model=run.build('fe_210' if arm=='canonical' else arm,mode='collect',canonical=arm=='canonical')
            agent=model.fusion_agent;agent.forced={5:{i:'anchor_sharp@online' for i in range(10)},15:{i:'local_wide@residual_boost' for i in range(10)}}
            records=[];agent.active_record=records.append
            measured=run.recorder(run.RUN.legacy.problem_for(group,fid),cache,schedule)
            torch.set_rng_state(rng);began=time.monotonic()
            with torch.no_grad():_,trail,nfe,executed=model(pop.clone(),measured)
            values=torch.cat(measured.values,1).numpy();points=executed.numpy().reshape(10,100,2,10);refs=torch.stack(model.r14_references,1).numpy()
            assert nfe==300 and np.array_equal(values.min(1),trail[:,-1].numpy())
            assert np.array_equal(torch.get_rng_state().numpy(),start['original_after'])
            run.check_decisions(records,points,refs,210 if arm=='canonical' else run.THRESHOLDS[arm],False)
            assert agent.stats['requests']==0
            if arm=='canonical':canonical=(values,points,records)
            if arm=='fe_210':
                assert np.array_equal(values,canonical[0]) and np.array_equal(points,canonical[1])
                assert records==canonical[2],'Default210 changed canonical full decisions'
            if measured.new_calls:cache=run.RUN.storage.merged_calls(cache,measured.new_calls)
            check=dict(group=group,fid=fid,arm=arm,trajectories=10,nfe=300,objective_rng_boundaries=201,api_calls=0,
                canonical_210_exact_match=arm=='fe_210',elapsed_seconds=time.monotonic()-began)
            checks.append(check);print(json.dumps(dict(event='engineering_pass',**check)),flush=True)
    frozen=run.read(run.EXP/'FROZEN_PARENT.json')
    for p,h in frozen['sha256'].items():assert run.sha(run.ROOT/p)==h,p
    result=dict(status='engineering_passed',not_full_live_validation=True,checks=checks,new_api_requests=0,
        all_1389_frozen_parent_files_unchanged=True,fixed_llm_FE=[220,240],canonical210_bitwise_match=True)
    run.write_json(directory/'RESULT.json',result)

if __name__=='__main__':
    try:main()
    except Exception:
        import traceback
        d=run.EXP/'engineering_checks';d.mkdir(exist_ok=True)
        run.write_json(d/f'FAILURE_{time.time_ns()}.json',dict(traceback=traceback.format_exc()));raise
