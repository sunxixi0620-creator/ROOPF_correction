"""R14: one admission threshold, fixed FE220/240 live-LLM decisions, paired full suite."""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
import argparse,fcntl,gzip,hashlib,importlib.util,json,multiprocessing as mp,secrets,sys,tempfile,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from hardware_backend import serve,write_json
EXP=ROOT/'experiments/r14_online_entry_timing'
ARMS=('fe_210','fe_100','fe_130','fe_150','fe_180','fe_230','fe_260')
THRESHOLDS={a:int(a[3:]) for a in ARMS}
RUN=None
VARIANT=None
CALLS={}


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def digest(b):return hashlib.sha256(b).hexdigest()
def configure(directory,runtime,address=None,authkey=None):
    global EXP,RUN,VARIANT
    EXP=Path(directory)
    import experiment_r13_cec_uniform as parent
    RUN=parent.configure(str(EXP),runtime,address,authkey)
    RUN.SOURCES=tuple(dict.fromkeys((*RUN.SOURCES,'scripts/experiment_r14_timing.py','scripts/report_r14_timing.py',
        'scripts/check_r14_timing.py','experiments/r14_online_entry_timing/variant_model.py')))
    spec=importlib.util.spec_from_file_location('roopf.r14_timing_variant',EXP/'variant_model.py')
    VARIANT=importlib.util.module_from_spec(spec);sys.modules[spec.name]=VARIANT;spec.loader.exec_module(VARIANT)
    import roopf.model as frozen
    VARIANT._is_structured_system_problem=frozen._is_structured_system_problem
    return RUN


def build(arm,mode='llm',canonical=False):
    from roopf.current_evidence_agent import CurrentEvidenceAgent
    model=RUN.legacy.build()
    if not canonical:
        model.__class__=VARIANT.ROOPFOptimizer;model.online_entry_fe=THRESHOLDS[arm]
    agent=CurrentEvidenceAgent(EXP/'offline_memory.json',EXP/('api_journal' if mode=='llm' else 'engineering_checks/no_api'),
        mode=mode,arm='joint',budget={'used':0,'limit':4900},api_config=RUN.settings()['api'])
    model.fusion_agent=agent
    refs=[];fn=model._baseline_candidates
    def capture(*args,**kwargs):
        value=fn(*args,**kwargs);refs.append(value.detach().cpu().clone());return value
    model._baseline_candidates=capture;model.r14_references=refs
    return model


def calls_for(group,fid):
    key=(group,fid)
    if key not in CALLS:CALLS[key]=RUN.storage.full_calls(group,fid)[0]
    return CALLS[key]


def paired(group,fid,batch):
    path=ROOT/'experiments/r10_current_candidate_fusion/results'/f'{group}_f{fid:02d}'/f'b{batch:02d}'
    with gzip.open(path/'PAIRED_START.pkl.gz','rb') as f:value=__import__('pickle').load(f)
    return value,path


def recorder(problem,cache,schedule=None):
    import torch
    class Recorded(RUN.legacy.Counted):
        def __init__(self):
            super().__init__(problem,cache,stochastic=True);self.rng_before=[]
        def calfitness(self,x):
            before=torch.get_rng_state().clone()
            if schedule is not None:assert torch.equal(before,schedule[len(self.rng_before)]),'Objective RNG pairing changed'
            self.rng_before.append(before)
            return super().calfitness(x)
    return Recorded()


def save_pickle(path,value):RUN.storage.save_pickle(path,value)
def original_batch(group,fid,batch,cache):
    import torch,numpy as np
    start,source=paired(group,fid,batch);population=torch.from_numpy(start['population'].copy());rng=torch.from_numpy(start['rng'].copy())
    original=RUN.legacy.build(original=True);problem=RUN.legacy.problem_for(group,fid);measured=recorder(problem,cache)
    torch.set_rng_state(rng)
    with torch.no_grad():_,_,nfe,points=original(population.clone(),measured)
    values=torch.cat(measured.values,1).numpy()
    assert nfe==300 and np.array_equal(values,np.load(source/'off.npz')['values'])
    assert torch.equal(torch.get_rng_state(),torch.from_numpy(start['original_after'].copy()))
    assert len(measured.rng_before)==201
    return start,source,population,rng,measured.rng_before,values,points.numpy().reshape(10,100,2,10)


def frozen_final(group,fid,batch,source,baseline_points):
    import numpy as np
    if group=='CEC-style':
        with np.load(ROOT/'experiments/r13_cec_uniform/results'/f'{group}_f{fid:02d}'/f'b{batch:02d}/llm_full.npz') as data:
            return data['values'].copy(),data['executed_points'].copy()
    values=np.load(source/'llm_live.npz')['values'];points=baseline_points.copy()
    with gzip.open(source/'llm_live_DECISIONS.json.gz','rt') as f:records=json.load(f)
    assert len(records)==45
    for r in records:
        k=55+r['step']
        for i,c in enumerate(r['current']):
            points[i,k,0]=c['protected_point'];points[i,k,1]=c['candidates'][c['selected_index']]
    return values,points


def check_decisions(records,points,refs,threshold,live):
    import numpy as np
    assert len(records)==45 and np.array_equal(points[:,:,0],refs[:,:,0])
    assert [r['step'] for r in records if r['request_event']]==([5,15] if live else [])
    for k,r in enumerate(records):
        fe=210+2*k
        for i,c in enumerate(r['current']):
            assert c['generation']['nfe_before']==fe and c['candidate_count']==38
            assert np.array_equal(points[i,55+k,0],c['protected_point'])
            assert np.array_equal(points[i,55+k,1],c['candidates'][c['selected_index']])
            if fe<threshold:assert c['selected_index']==0
        if fe<threshold:assert np.array_equal(points[:,55+k],refs[:,55+k])
    before=max(0,(threshold-100)//2)
    assert np.array_equal(points[:,:before],refs[:,:before]),'Online point admitted before parameter threshold'


def run_batch(task):
    import numpy as np,torch,pickle
    group,fid,batch=task;folder=EXP/'results'/f'{group}_f{fid:02d}'/f'b{batch:02d}';folder.mkdir(parents=True,exist_ok=True)
    identity=RUN.binding();marker=folder/'BATCH_COMPLETE.json'
    if marker.exists():assert read(marker)['binding']==identity;return read(marker)
    cache=calls_for(group,fid)[batch]
    for p in sorted(folder.glob('*_NATIVE_CALLS.pkl.gz')):
        with gzip.open(p,'rb') as f:cache=RUN.storage.merged_calls(cache,pickle.load(f))
    start,source,population,rng,schedule,baseline_values,baseline_points=original_batch(group,fid,batch,cache)
    frozen_values,frozen_points=frozen_final(group,fid,batch,source,baseline_points)
    hashes={'paired_initial_sha256':digest(population.numpy().tobytes()),'paired_solver_rng_sha256':digest(rng.numpy().tobytes())}
    assert hashes['paired_initial_sha256']==read(source/'llm_live.json')['paired_initial_sha256']
    np.savez_compressed(folder/'REFERENCES.npz',frozen_final_values=frozen_values,frozen_final_points=frozen_points,
        original_values=baseline_values,original_points=baseline_points)
    for arm in ARMS:
        receipt=folder/(arm+'.json')
        if receipt.exists():assert read(receipt)['binding']==identity;continue
        model=build(arm);agent=model.fusion_agent;records=[];agent.active_record=records.append
        agent.execution_context=dict(round='r14_online_entry_timing',group=group,fid=fid,batch=batch,arm=arm,
            online_admission_start_fe=THRESHOLDS[arm],llm_decision_fe=[220,240],**hashes)
        problem=RUN.legacy.problem_for(group,fid);measured=recorder(problem,cache,schedule)
        torch.set_rng_state(rng);began=time.monotonic()
        with torch.no_grad():_,trail,nfe,executed=model(population.clone(),measured)
        values=torch.cat(measured.values,1).numpy();points=executed.numpy().reshape(10,100,2,10)
        refs=torch.stack(model.r14_references,1).numpy()
        assert nfe==300 and values.shape==(10,300) and np.isfinite(values).all()
        assert np.array_equal(values.min(1),trail[:,-1].numpy())
        assert measured.native_fe+measured.cached_fe==3000 and len(measured.rng_before)==201
        assert torch.equal(torch.get_rng_state(),torch.from_numpy(start['original_after'].copy()))
        check_decisions(records,points,refs,THRESHOLDS[arm],True)
        assert agent.stats['requests']+agent.stats['api_reused']==2
        assert not agent.resource_exhausted and not agent.uncertain_api_outcome
        for r in records:
            if r['request_event']:
                row=read(EXP/'api_journal'/(r['api_request_sha256']+'.json'))
                assert row['config_ids']==r['configs']
        np.savez_compressed(folder/(arm+'.npz'),values=values,executed_points=points,reference_points=refs)
        with gzip.open(folder/(arm+'_DECISIONS.json.gz'),'wt') as f:json.dump(records,f,allow_nan=False)
        if measured.new_calls:
            save_pickle(folder/(arm+'_NATIVE_CALLS.pkl.gz'),measured.new_calls);cache=RUN.storage.merged_calls(cache,measured.new_calls)
        row=dict(group=group,fid=fid,batch=batch,arm=arm,threshold=THRESHOLDS[arm],n=10,nfe=300,**hashes,binding=identity,
            native_fe=measured.native_fe,cached_fe=measured.cached_fe,elapsed_seconds=time.monotonic()-began,agent_stats=agent.stats,
            coordinate_changes_vs_frozen_final=int((points!=frozen_points).any(-1).sum()),
            coordinate_changes_vs_original=int((points!=baseline_points).any(-1).sum()),
            objective_rng_boundaries_matched=201,final_rng_matches_frozen_original=True,llm_call_fe=[220,240],
            actual_online_evaluated_points=int((points[:,:,1]!=refs[:,:,1]).any(-1).sum()),cec_unlocked=True)
        write_json(receipt,row)
        print(json.dumps(dict(event='arm_batch_complete',group=group,fid=fid,batch=batch,arm=arm,api=agent.stats['requests'],
            api_failures=agent.stats['failures'],valid_rows=agent.stats['valid_rows'],elapsed_seconds=round(row['elapsed_seconds'],2))),flush=True)
    result=dict(group=group,fid=fid,batch=batch,status='complete',binding=identity);write_json(marker,result);return result


def worker(task):
    try:return run_batch(task)
    except Exception:
        folder=EXP/'failures';folder.mkdir(exist_ok=True)
        write_json(folder/f'{task[0]}_f{task[1]:02d}_b{task[2]:02d}_{time.time_ns()}.json',dict(task=task,traceback=traceback.format_exc()));raise


def progress(started,done,total,stats):
    from run_r10_hardware import process_tree_usage
    complete=sum(1 for arm in ARMS for _ in EXP.glob(f'results/*/b??/{arm}.json'))
    states={'ok':0,'partial':0,'failed':0,'pending':0}
    for p in (EXP/'api_journal').glob('*.json'):states[read(p)['status']]+=1
    _,rss,processes=process_tree_usage();elapsed=time.monotonic()-started
    data=dict(event='progress',completed_tasks=done,total_tasks=total,completed_arm_batches=complete,total_arm_batches=2450,
        actual_api=states,expected_api_requests=4900,elapsed_seconds=round(elapsed),rss_gib=round(rss/1024**3,2),processes=processes,
        eta_seconds=round(elapsed*(2450-complete)/complete) if complete else None)
    if stats.exists():data['gpu']=read(stats)
    write_json(EXP/'PROGRESS.json',data);print(json.dumps(data),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=['all','run','report','verify'],default='all',nargs='?')
    p.add_argument('--workers',type=int,default=24);p.add_argument('--progress-seconds',type=float,default=10)
    args=p.parse_args();assert Path(sys.prefix).name=='ooneurgo' and 1<=args.workers<=24
    runtime=read(EXP/'HARDWARE_RUNTIME.json');assert runtime['workers']==args.workers
    stamp=str(time.time_ns());gpu=None;began=time.monotonic();done=0
    from contextlib import ExitStack
    with (EXP/'.run.lock').open('a') as lock,ExitStack() as cleanup:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            temporary=cleanup.enter_context(tempfile.TemporaryDirectory(prefix='r14_gpu_'));address=str(Path(temporary)/'cuda.sock');auth=secrets.token_bytes(32)
            stats=EXP/'GPU_PROGRESS.json'
            if args.phase in ('all','run'):
                ctx=mp.get_context('spawn');ready,child=ctx.Pipe(duplex=False)
                gpu=ctx.Process(target=serve,args=(address,auth,runtime['visible_devices'],child,str(stats),runtime['gpu_concurrency']))
                gpu.start();child.close();assert ready.poll(60),'CUDA initialization timeout'
                info=ready.recv();assert info['status']=='ready',info
                write_json(EXP/f'HARDWARE_{stamp}.json',info);ready.close();print(json.dumps(dict(event='gpu_ready',**info)),flush=True)
            configure(str(EXP),runtime);identity=RUN.binding();binding=EXP/'CODE_BINDING.json'
            if binding.exists():assert read(binding)==identity,'Frozen R14 source changed'
            else:write_json(binding,identity)
            if args.phase in ('all','run'):
                tasks=[(g,f,b) for b in range(10) for g,n in [('BBOB',24),('CEC-style',6),('UAV',5)] for f in range(1,n+1)]
                last=0
                with mp.get_context('spawn').Pool(args.workers,initializer=configure,initargs=(str(EXP),runtime,address,auth)) as pool:
                    pending=pool.imap_unordered(worker,tasks,chunksize=1)
                    while done<len(tasks):
                        try:pending.next(timeout=1);done+=1
                        except mp.TimeoutError:pass
                        if time.monotonic()-last>=args.progress_seconds:progress(began,done,len(tasks),stats);last=time.monotonic()
                progress(began,done,350,stats)
            if args.phase in ('all','report','verify'):
                from report_r14_timing import report
                result=report(EXP,verify=args.phase in ('all','verify'));print(json.dumps(result),flush=True)
        except BaseException:
            write_json(EXP/f'FAILURE_{stamp}.json',dict(status='partial_or_failed',completed_tasks=done,traceback=traceback.format_exc()));raise
        finally:
            if gpu is not None:
                if gpu.is_alive():
                    from multiprocessing.connection import Client
                    try:
                        with Client(address,family='AF_UNIX',authkey=auth) as c:c.send(dict(op='shutdown'))
                    except OSError:pass
                gpu.join(timeout=10)
                if gpu.is_alive():gpu.terminate();gpu.join(timeout=5)

if __name__=='__main__':main()
