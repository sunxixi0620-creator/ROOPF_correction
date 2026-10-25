"""Engineering checks only: real objectives and forced actions, zero API calls."""
import gzip
import hashlib
import json
from pathlib import Path
import pickle
import traceback

import experiment_r12_strict_ablations as run
import numpy as np
import torch


def main():
    p=run.EXP;target=p/'engineering_checks';target.mkdir(exist_ok=True)
    run.configure(str(p),json.loads((run.PARENT/'HARDWARE_RUNTIME.json').read_text()))
    checks=[]
    for group,fid in [('BBOB',9),('CEC-style',1),('BBOB',20)]:
        source=run.PARENT/'results'/f'{group}_f{fid:02d}'
        with gzip.open(source/'b00/PAIRED_START.pkl.gz','rb') as stream:paired=pickle.load(stream)
        calls,_=run.RUN.storage.full_calls(group,fid);problem=run.RUN.legacy.problem_for(group,fid)
        noagent=run.build('no_agent');torch.set_rng_state(torch.from_numpy(paired['rng'].copy()))
        first=run.recorder(problem,calls[0])
        with torch.no_grad():_,_,nfe,_=noagent(torch.from_numpy(paired['population'].copy()),first)
        original=np.load(source/'b00/off.npz')['values']
        assert nfe==300 and np.array_equal(torch.cat(first.values,1).numpy(),original)
        assert all(hashlib.sha256(s.numpy().tobytes()).hexdigest()==calls[0][i][0]['rng'] for i,s in enumerate(first.rng_before))
        assert all(v==100 for v in noagent.strict_generator_counts.values())
        model=run.build('llm_no_online_generators',mode='collect');agent=model.fusion_agent
        agent.execution_context=dict(engineering=True,group=group,fid=fid)
        agent.forced={5:{i:'anchor_sharp@online' for i in range(10)},15:{i:'joint_soft_fine@residual_boost' for i in range(10)}}
        records=[];agent.active_record=records.append;model.objective_rng_schedule=first.rng_before
        measured=run.recorder(problem,calls[0],first.rng_before);torch.set_rng_state(torch.from_numpy(paired['rng'].copy()))
        with torch.no_grad():_,trail,nfe,_=model(torch.from_numpy(paired['population'].copy()),measured)
        values=torch.cat(measured.values,1).numpy()
        assert nfe==300 and values.shape==(10,300) and np.array_equal(values.min(1),trail[:,-1].numpy())
        assert np.array_equal(values[:,:210],original[:,:210])
        assert model.neural_generator is None and not any(model.strict_generator_counts.values())
        assert agent.stats['requests']==0 and len(records)==(0 if group=='CEC-style' else 45)
        assert torch.get_rng_state().numpy().tobytes()==paired['original_after'].tobytes()
        if group=='CEC-style':assert np.array_equal(values,original)
        else:
            assert set(c['candidate_count'] for record in records for c in record['current'])=={1,2}
            current=records[5]['current'][0]
            assert len(current['candidates'])==1
            assert len(records[6]['current'][0]['candidates'])==2
        checks.append(dict(group=group,fid=fid,trajectories=10,nfe=300,no_agent_all_values_equal_frozen_original=True,
            no_online_generator_calls=0,paired_objective_rng_boundaries=201,api_calls=0,engineering_only=True))
        print('PASS',checks[-1],flush=True)
    (target/'RESULT.json').write_text(json.dumps(dict(status='passed',not_full_live_validation=True,checks=checks),indent=2)+'\n')


if __name__=='__main__':
    try:main()
    except Exception:
        (run.EXP/'engineering_checks/LATEST_FAILURE.txt').write_text(traceback.format_exc())
        raise
