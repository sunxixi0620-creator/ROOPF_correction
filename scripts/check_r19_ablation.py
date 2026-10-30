"""No-API engineering check of the selected schedule and strict removals."""
import sys,json,torch,numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import experiment_r15_schedule as reference
import experiment_r13_cec_uniform as factory
import r15_common as common
from roopf.scheduled_evidence_agent_r18 import ScheduledEvidenceAgent
from roopf.scheduled_ablation_r19 import ScheduledNoAnchor,ScheduledNoOnline
E=ROOT/'experiments/r19_s01l08_component_ablations';RUNTIME=common.RUNTIME

def main():
 reference.configure();factory.configure(str(E),RUNTIME)
 for group,fid in [('BBOB',1),('CEC-style',1),('UAV',1)]:
  cache=reference.calls_for(group,fid)[0]
  start,source,pop,rng,boundaries,_,_=reference.original_batch(group,fid,0,cache)
  for arm in ['llm_full','llm_no_anchor','llm_no_online_generators','no_agent']:
   agent=None
   if arm!='no_agent':
    cls={'llm_full':ScheduledEvidenceAgent,'llm_no_anchor':ScheduledNoAnchor,'llm_no_online_generators':ScheduledNoOnline}[arm]
    agent=cls(ROOT/'experiments/r15_llm_schedule_matched/cases/S01_L08/offline_memory.json',E/'engineering_checks/no_api',schedule=common.schedule('S01_L08'),mode='collect',arm='joint',budget={'used':0,'limit':2100},api_config=common.PROTOCOL['fixed_conditions']['api'])
    factory.agent_for=lambda _arm,mode='collect':agent
    agent.forced={0:{i:'local_wide@online' for i in range(10)},10:{i:'local_wide@online' for i in range(10)}}
   model=factory.build(arm,mode='collect')
   records=[]
   if agent:agent.active_record=records.append
   if arm=='llm_no_online_generators':model.objective_rng_schedule=boundaries
   measured=reference.recorder(factory.RUN.legacy.problem_for(group,fid),cache,boundaries)
   torch.set_rng_state(rng)
   with torch.no_grad():_,trail,nfe,executed=model(pop.clone(),measured)
   vals=torch.cat(measured.values,1).numpy();points=executed.numpy().reshape(10,100,2,10)
   assert nfe==300 and np.array_equal(vals.min(1),trail[:,-1].numpy())
   assert np.array_equal(torch.get_rng_state().numpy(),start['original_after'])
   assert np.array_equal(points[:,:,0],torch.stack(model.r13_references,1).numpy()[:,:,0])
   if agent:
    assert len(records)==45 and model.r13_counts['prepare_generation']==45
    assert [r['step'] for r in records if r['configs'][0]=='local_wide@online']==list(range(8))+list(range(10,18))
    assert [r['step'] for r in records if r['applied_generation'][0]=='local_wide']==list(range(1,8))+list(range(11,18))
    assert agent.stats['requests']==0
   if arm=='llm_no_anchor':assert model.baseline_model is None and model.r13_counts['anchor_forward']==0
   if arm=='llm_no_online_generators':assert model.neural_generator is None and not any(model.r13_counts['generators'].values())
   if arm=='no_agent':assert model.fusion_agent is None and model.agentic_controller is None
   print(json.dumps(dict(group=group,arm=arm,status='passed',fe=nfe,api_calls=0)),flush=True)
 print(json.dumps({'status':'engineering_passed','groups':3,'arms':4,'real_api_calls':0}))
if __name__=='__main__':main()
