"""Engineering-only execution checks; no API calls and no efficacy claim."""
from pathlib import Path
import gzip
import json
import sys
import time
import traceback
import numpy as np
import torch
import experiment_r13_cec_uniform as run


def main():
    directory=run.EXP/'engineering_checks';directory.mkdir(exist_ok=True)
    assert Path(sys.prefix).name=='ooneurgo'
    runtime=run.readj(run.EXP/'HARDWARE_RUNTIME.json')
    run.configure(str(run.EXP),runtime)
    import roopf.model as current
    import roopf.strict_component_ablations as strict
    problem=run.RUN.legacy.problem_for('CEC-style',1)
    assert current._r13_frozen_structured_predicate(problem)
    assert not current._is_structured_system_problem(problem) and not strict._is_structured_system_problem(problem)
    assert problem.getfunname()=='cecf1'
    original=run.build('frozen_baseline')
    archived=sys.modules[type(original).__module__]
    assert archived._is_structured_system_problem(problem),'Archived baseline was patched'
    paired,source=run.paired_start(1,0)
    population=torch.from_numpy(paired['population'].copy());rng=torch.from_numpy(paired['rng'].copy())
    calls=run.calls_for(1)[0]
    checks=[];schedule=None
    for arm in run.ARMS:
        model=original if arm=='frozen_baseline' else run.build(arm,mode='collect')
        records=[];agent=getattr(model,'fusion_agent',None)
        if agent:
            agent.active_record=records.append;agent.execution_context=dict(engineering_only=True,arm=arm)
            agent.forced={5:{i:'anchor_sharp@online' for i in range(10)},15:{i:'local_wide@residual_boost' for i in range(10)}}
        if arm=='llm_no_online_generators':model.objective_rng_schedule=schedule
        measured=run.recorder(problem,calls,schedule)
        torch.set_rng_state(rng);began=time.monotonic()
        with torch.no_grad():_,trail,nfe,executed=model(population.clone(),measured)
        values=torch.cat(measured.values,1).numpy();points=executed.numpy().reshape(10,100,2,10)
        assert nfe==300 and values.shape==(10,300) and np.array_equal(values.min(1),trail[:,-1].numpy())
        assert torch.get_rng_state().numpy().tobytes()==paired['original_after'].tobytes()
        if arm=='frozen_baseline':
            assert np.array_equal(values,np.load(source/'off.npz')['values'])
            schedule=measured.rng_before
            assert len(schedule)==201
            assert all(run.sha_bytes(s.numpy().tobytes())==calls[i][0]['rng'] for i,s in enumerate(schedule))
        else:
            assert model.r13_counts['residual_forward']>0
            if agent:
                run.check_decisions(records,points,arm,agent,live=False)
                assert model.r13_counts['prepare_generation']==45
                assert agent.stats['requests']==0
            else:assert not records and arm=='no_agent'
            if arm=='llm_no_online_generators':assert not any(model.r13_counts['generators'].values())
            if arm=='llm_no_anchor':assert model.baseline_model is None
        checks.append(dict(arm=arm,trajectories=10,nfe=300,objective_rng_boundaries=201,
                           elapsed_seconds=time.monotonic()-began,route_counts=model.r13_counts,
                           agent_rounds=len(records),api_calls=0))
        print(json.dumps(dict(event='engineering_pass',**checks[-1])),flush=True)
    # BBOB/UAV/generated names retain exactly the old boolean policy.
    class Named:
        def __init__(self,name):self.name=name
        def getfunname(self):return self.name
    for name in ('1','bbob1','uav1','generated36:rbf_sparse','hpof1'):
        p=Named(name)
        assert current._is_structured_system_problem(p)==current._r13_frozen_structured_predicate(p)
    for name,expected in run.readj(run.PARENT/'CODE_BINDING.json')['code'].items():assert run.sha(run.ROOT/name)==expected
    result=dict(status='engineering_checks_passed',not_full_live_validation=True,
                objective_names_and_implementations_unchanged=True,archived_baseline_unchanged=True,
                non_cec_predicate_unchanged=True,api_calls=0,checks=checks)
    run.write_json(directory/'RESULT.json',result)


if __name__=='__main__':
    try:main()
    except Exception:
        directory=run.EXP/'engineering_checks';directory.mkdir(exist_ok=True)
        run.write_json(directory/f'FAILURE_{time.time_ns()}.json',dict(traceback=traceback.format_exc()))
        raise
