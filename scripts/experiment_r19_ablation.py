"""R19: S01_L08 full and strict module ablations, paired live validation."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
import sys,json,gzip,time,traceback,hashlib,tempfile,secrets,fcntl,multiprocessing as mp
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import r15_common as common
from hardware_backend import serve,write_json
E=ROOT/'experiments/r19_s01l08_component_ablations';ARMS=('llm_full','llm_no_anchor','llm_no_online_generators','no_agent')
SOURCES=['scripts/experiment_r19_ablation.py','scripts/report_r19_ablation.py','scripts/check_r19_ablation.py','roopf/scheduled_ablation_r19.py','roopf/scheduled_evidence_agent_r18.py','roopf/policy_api_r18.py','roopf/no_anchor_ablation.py','roopf/strict_component_ablations.py']
MEM=ROOT/'experiments/r15_llm_schedule_matched/cases/S01_L08/offline_memory.json'
RUN=None

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def binding():
 common.validate_frozen()
 return dict(protocol_sha256=sha(E/'PROTOCOL_DRAFT.json'),source_sha256={s:sha(ROOT/s) for s in SOURCES},memory_sha256=sha(MEM),parent_r15_binding_sha256=sha(common.EXP/'CODE_BINDING.json'),runtime_sha256=common.digest(common.RUNTIME))
def configure(address=None,auth=None):
 global RUN
 import experiment_r15_schedule as ref
 import experiment_r13_cec_uniform as factory
 ref.configure(address=address,authkey=auth)
 RUN=factory.configure(str(E),common.RUNTIME,address,auth)

def check_records(records,points,refs,arm):
 import numpy as np
 assert len(records)==45 and [r['step'] for r in records if r['request_event']]==[0,10]
 for k,r in enumerate(records):
  assert r['step']==k
  for i,c in enumerate(r['current']):
   assert c['generation']['nfe_before']==210+2*k
   assert np.array_equal(points[i,55+k,0],c['protected_point'])
   assert np.array_equal(points[i,55+k,1],c['candidates'][c['selected_index']])
   assert c['candidate_count'] in ((1,2) if arm=='llm_no_online_generators' else (38,))
  if k in (0,10):plans=r['configs']
  active=any(s<=k<s+8 for s in (0,10))
  assert r['configs']==(plans if active else ['release']*len(r['configs']))
  expected=['matched38']*len(r['configs']) if k in (0,10) else ([n.split('@')[0] for n in plans] if active else ['release']*len(r['configs']))
  assert r['applied_generation']==expected

def task(t):
 try:return run_batch(t)
 except Exception:
  group,fid,b=t;write_json(E/'failures'/f'{group}_{fid}_{b}_{time.time_ns()}.json',dict(task=t,traceback=traceback.format_exc()));raise

def run_batch(t):
 import numpy as np,torch,pickle
 import experiment_r15_schedule as ref
 import experiment_r13_cec_uniform as factory
 from roopf.scheduled_evidence_agent_r18 import ScheduledEvidenceAgent
 from roopf.scheduled_ablation_r19 import ScheduledNoAnchor,ScheduledNoOnline
 group,fid,b=t;folder=E/'results'/f'{group}_f{fid:02d}'/f'b{b:02d}';folder.mkdir(parents=True,exist_ok=True)
 ident=binding();marker=folder/'BATCH_COMPLETE.json'
 if marker.exists():assert json.loads(marker.read_text())['binding']==ident;return
 cache=ref.calls_for(group,fid)[b]
 for p in folder.glob('*_NATIVE_CALLS.pkl.gz'):
  with gzip.open(p,'rb') as f:cache=RUN.storage.merged_calls(cache,pickle.load(f))
 start,source,pop,rng,boundaries,basevalues,basepoints=ref.original_batch(group,fid,b,cache)
 archive=ROOT/'experiments/r15_llm_schedule_matched/results'/f'{group}_f{fid:02d}'/f'b{b:02d}'/'S01_L08.npz'
 with np.load(archive) as a:archvalues=a['values'].copy();archpoints=a['executed_points'].copy()
 np.savez_compressed(folder/'REFERENCES.npz',original_values=basevalues,original_points=basepoints,archived_full_values=archvalues,archived_full_points=archpoints)
 hashes=dict(paired_initial_sha256=sha_bytes(pop.numpy().tobytes()),paired_solver_rng_sha256=sha_bytes(rng.numpy().tobytes()))
 for arm in ARMS:
  receipt=folder/(arm+'.json')
  if receipt.exists():assert json.loads(receipt.read_text())['binding']==ident;continue
  agent=None
  if arm!='no_agent':
   cls={'llm_full':ScheduledEvidenceAgent,'llm_no_anchor':ScheduledNoAnchor,'llm_no_online_generators':ScheduledNoOnline}[arm]
   agent=cls(MEM,E/'api_journal',schedule=common.schedule('S01_L08'),mode='llm',arm='joint',budget={'used':0,'limit':2100},api_config=common.PROTOCOL['fixed_conditions']['api'])
   agent.execution_context=dict(round='r19_s01l08_component_ablations',group=group,fid=fid,batch=b,arm=arm,online_admission_start_fe=210,llm_decision_fe=[210,230],schedule_signature=common.schedule('S01_L08'),memory_sha256=agent.memory_hash,**hashes)
   factory.agent_for=lambda _arm,mode='llm':agent
  model=factory.build(arm,mode='llm');records=[]
  if agent:agent.active_record=records.append
  if arm=='llm_no_online_generators':model.objective_rng_schedule=boundaries
  measured=ref.recorder(RUN.legacy.problem_for(group,fid),cache,boundaries)
  torch.set_rng_state(rng);begin=time.monotonic()
  with torch.no_grad():_,trail,nfe,executed=model(pop.clone(),measured)
  vals=torch.cat(measured.values,1).numpy();points=executed.numpy().reshape(10,100,2,10);refs=torch.stack(model.r13_references,1).numpy()
  assert nfe==300 and vals.shape==(10,300) and np.isfinite(vals).all() and np.array_equal(vals.min(1),trail[:,-1].numpy())
  assert measured.native_fe+measured.cached_fe==3000 and len(measured.rng_before)==201
  assert torch.equal(torch.get_rng_state(),torch.from_numpy(start['original_after'].copy())) and np.array_equal(points[:,:,0],refs[:,:,0])
  if agent:
   check_records(records,points,refs,arm);assert agent.stats['requests']+agent.stats['api_reused']==2 and not agent.uncertain_api_outcome
   for rec in records:
    if rec['request_event']:
     journal=json.loads((E/'api_journal'/(rec['api_request_sha256']+'.json')).read_text());assert journal['config_ids']==rec['configs']
  else:assert model.fusion_agent is None and model.agentic_controller is None
  if arm=='llm_no_anchor':assert model.baseline_model is None and model.r13_counts['anchor_forward']==0
  if arm=='llm_no_online_generators':assert model.neural_generator is None and not any(model.r13_counts['generators'].values())
  np.savez_compressed(folder/(arm+'.npz'),values=vals,executed_points=points,reference_points=refs)
  with gzip.open(folder/(arm+'_DECISIONS.json.gz'),'wt') as f:json.dump(records,f,allow_nan=False)
  if measured.new_calls:RUN.storage.save_pickle(folder/(arm+'_NATIVE_CALLS.pkl.gz'),measured.new_calls);cache=RUN.storage.merged_calls(cache,measured.new_calls)
  data=dict(group=group,fid=fid,batch=b,arm=arm,n=10,nfe=300,**hashes,binding=ident,native_fe=measured.native_fe,cached_fe=measured.cached_fe,agent_stats={} if not agent else agent.stats,route_counts=model.r13_counts,elapsed_seconds=time.monotonic()-begin,objective_rng_boundaries_matched=201,coordinate_changes_vs_original=int((points!=basepoints).any(-1).sum()),coordinate_changes_vs_archived_full=int((points!=archpoints).any(-1).sum()))
  write_json(receipt,data);print(json.dumps(dict(event='arm_complete',group=group,fid=fid,batch=b,arm=arm,valid_rows=data['agent_stats'].get('valid_rows',0))),flush=True)
 write_json(marker,dict(task=t,binding=ident,status='complete'))

def sha_bytes(v):return hashlib.sha256(v).hexdigest()

def main():
 import experiment_r15_schedule as ref
 ident=binding();write_json(E/'CODE_BINDING.json',ident)
 lock=(E/'.run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 runtime=common.RUNTIME;ctx=mp.get_context('spawn');gpu=None
 try:
  with tempfile.TemporaryDirectory(prefix='r19_gpu_') as temp:
   address=str(Path(temp)/'cuda.sock');auth=secrets.token_bytes(32);ready,child=ctx.Pipe(duplex=False)
   gpu=ctx.Process(target=serve,args=(address,auth,runtime['visible_devices'],child,str(E/'GPU_PROGRESS.json'),runtime['gpu_concurrency']));gpu.start();child.close();assert ready.poll(60);info=ready.recv();assert info['status']=='ready'
   write_json(E/'EXECUTION_STATUS.json',dict(status='running',phase='full_live',pid=os.getpid(),started=time.strftime('%Y-%m-%dT%H:%M:%S%z')))
   tasks=[(g,f,b) for b in range(10) for g,n in [('BBOB',24),('CEC-style',6),('UAV',5)] for f in range(1,n+1)];done=0;begin=time.monotonic();last=0
   with ctx.Pool(24,initializer=configure,initargs=(address,auth)) as pool:
    pending=pool.imap_unordered(task,tasks,chunksize=1)
    while done<len(tasks):
     try:pending.next(timeout=1);done+=1
     except mp.TimeoutError:pass
     if time.monotonic()-last>20:
      count=sum(1 for a in ARMS for _ in E.glob(f'results/*/b??/{a}.json'));j={x:0 for x in ['ok','partial','failed','pending']}
      for p in (E/'api_journal').glob('*.json'):j[json.loads(p.read_text())['status']]+=1
      data=dict(event='progress',completed_arm_batches=count,total_arm_batches=1400,finished_functions_batches=done,elapsed_seconds=int(time.monotonic()-begin),api=j);write_json(E/'PROGRESS.json',data);print(json.dumps(data),flush=True);last=time.monotonic()
   from report_r19_ablation import report
   report()
 except BaseException:
  err=dict(status='partial_or_failed',traceback=traceback.format_exc());write_json(E/f'FAILURE_{time.time_ns()}.json',err);write_json(E/'EXECUTION_STATUS.json',err);raise
 finally:
  if gpu and gpu.is_alive():gpu.terminate();gpu.join()
if __name__=='__main__':main()
