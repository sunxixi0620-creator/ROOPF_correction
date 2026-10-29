"""No-API schedule boundary, full numerical default, case isolation checks."""
import copy,gzip,json,time
from pathlib import Path
import numpy as np
import torch
import r18_common as common

def check():
    import experiment_r18_schedule as run
    folder=common.EXP/'engineering_checks';folder.mkdir(exist_ok=True)
    if (folder/'RESULT.json').exists():
        prior=common.read(folder/'RESULT.json');assert prior['binding']==common.binding();return prior
    run.configure();checks=[]
    for group,fid in [('BBOB',1),('CEC-style',1),('UAV',1)]:
        cache=run.calls_for(group,fid)[0]
        start,source,pop,rng,rng_schedule,baseline,basepoints=run.original_batch(group,fid,0,cache)
        canonical=None
        # Cover every generation and fusion value at the two events, with varying rows.
        from roopf.scheduled_evidence_agent import ACTION_NAMES
        actions1={i:ACTION_NAMES[(2*i+3)%22] for i in range(10)}
        actions2={i:ACTION_NAMES[(2*i+4)%22] for i in range(10)}
        for cid in run.ARMS:
            model=run.build('S02_L05' if cid=='canonical' else cid,mode='collect',canonical=cid=='canonical')
            config=common.CONFIGS['S02_L05' if cid=='canonical' else cid]
            agent=model.fusion_agent;records=[];agent.active_record=records.append
            steps=(config['first_call_active_step'],config['second_call_active_step'])
            agent.forced={steps[0]:actions1,steps[1]:actions2}
            measured=run.recorder(run.RUN.legacy.problem_for(group,fid),cache,rng_schedule)
            torch.set_rng_state(rng)
            with torch.no_grad():_,trail,nfe,executed=model(pop.clone(),measured)
            values=torch.cat(measured.values,1).numpy();points=executed.numpy().reshape(10,100,2,10)
            refs=torch.stack(model.r15_references,1).numpy()
            assert nfe==300 and len(measured.rng_before)==201 and np.array_equal(values.min(1),trail[:,-1].numpy())
            assert np.array_equal(torch.get_rng_state().numpy(),start['original_after'])
            run.check_decisions(records,points,refs,config,False)
            assert agent.stats['requests']==0 and not agent.uncertain_api_outcome
            if measured.new_calls:cache=run.RUN.storage.merged_calls(cache,measured.new_calls)
            row=dict(group=group,config=cid,n=10,fe=300,objective_rng_boundaries=201,api_calls=0,window_and_rng_checks=True)
            checks.append(row);print(json.dumps(dict(event='engineering_pass',**row)),flush=True)
    # Case loader must reject another configuration without silently changing step labels.
    from roopf.scheduled_evidence_agent import ScheduledEvidenceAgent,POLICY_ID,digest
    cfg=common.schedule('S01_L03');bad=folder/'raw/wrong_memory.json'
    common.write(bad,dict(policy_id=POLICY_ID,cases=[],coverage_radii={},schedule=cfg,schedule_signature=digest(cfg)))
    try:ScheduledEvidenceAgent(bad,folder/'no_api',common.schedule('S02_L05'),mode='collect')
    except ValueError:pass
    else:raise AssertionError('Cross-config memory accepted')
    common.validate_frozen()
    result=dict(status='engineering_passed',binding=common.binding(),checks=checks,all28_schedule_boundaries=True,
                cross_config_case_rejected=True,all_groups_checked=3,real_api_calls=0,not_full_live_validation=True)
    common.write(folder/'RESULT.json',result);return result

if __name__=='__main__':check()
