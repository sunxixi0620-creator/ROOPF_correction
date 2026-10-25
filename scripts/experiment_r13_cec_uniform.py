"""Complete CEC-only live-LLM Full/baseline/three-component remeasurement."""
from __future__ import annotations
import os
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
import argparse
import fcntl
import gzip
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import pickle
import secrets
import sys
import tempfile
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from hardware_backend import serve,write_json
from run_r10_hardware import Progress,configure as configure_r10
PARENT=ROOT/'experiments/r10_current_candidate_fusion'
EXP=ROOT/'experiments/r13_cec_uniform'
ARMS=('frozen_baseline','no_agent','llm_full','llm_no_anchor','llm_no_online_generators')
LIVE_ARMS=ARMS[2:]
GENERATORS=('neural_generator','_current_to_pbest','_covariance_elite','_trust_region','_coordinate_pattern','_opposition_restart')
EXTRA=('roopf/cec_uniform.py','roopf/no_anchor_ablation.py','roopf/strict_component_ablations.py',
       'roopf/benchmarks/cecfunctions.py','roopf/benchmarks/utils.py','scripts/experiment_r13_cec_uniform.py',
       'scripts/check_r13_cec_uniform.py','scripts/run_r13_cec_uniform.sh')
RUN=None
CALL_CACHE={}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def readj(path):return json.loads(Path(path).read_text())


def configure(directory,runtime,address=None,authkey=None):
    global EXP,RUN
    EXP=Path(directory)
    RUN,_=configure_r10(directory,runtime,address,authkey)
    RUN.SOURCES=tuple(dict.fromkeys((*RUN.SOURCES,*EXTRA)))
    from roopf.cec_uniform import install_execution_policy
    policy=install_execution_policy()
    assert policy['baseline_module_patched'] is False
    return RUN


def agent_for(arm,mode='llm'):
    from roopf.current_evidence_agent import CurrentEvidenceAgent
    from roopf.no_anchor_ablation import NoAnchorAgent
    from roopf.strict_component_ablations import NoOnlineGeneratorsAgent
    cls={'llm_full':CurrentEvidenceAgent,'llm_no_anchor':NoAnchorAgent,
         'llm_no_online_generators':NoOnlineGeneratorsAgent}[arm]
    return cls(RUN.MEMORY,EXP/('api_journal' if mode=='llm' else 'engineering_checks/no_api'),
        mode=mode,arm='joint',budget=dict(used=0,limit=RUN.settings()['max_new_api_attempts']),
        api_config=RUN.settings()['api'])


def build(arm,mode='llm'):
    import torch
    from roopf.no_anchor_ablation import install as remove_anchor
    from roopf.strict_component_ablations import NoOnlineGeneratorsOptimizer
    model=RUN.legacy.build(original=arm=='frozen_baseline')
    if arm=='llm_no_anchor':model=remove_anchor(model)
    if arm=='llm_no_online_generators':model.__class__=NoOnlineGeneratorsOptimizer
    if arm in LIVE_ARMS:model.fusion_agent=agent_for(arm,mode)
    elif arm=='no_agent':assert model.fusion_agent is None and model.agentic_controller is None
    # Every non-baseline arm shares the same CUDA proxy, independent of Agent presence.
    client=getattr(RUN,'_hardware_proxy_client',None)
    if arm!='frozen_baseline' and client is not None:
        proxy=model.surrogate
        def predict(archive_x,archive_y,candidates,problem,distance_scale=.15):
            return client.predict(proxy,archive_x,archive_y,candidates,problem,distance_scale)
        proxy.predict=predict
    model.r13_counts=dict(generators={name:0 for name in GENERATORS},residual_forward=0,
                          anchor_forward=0,anchor_temperature_proposal=0,prepare_generation=0)
    model.r13_references=[]
    for name in GENERATORS:
        owner=model.neural_generator if name=='neural_generator' else model
        attr='forward' if name=='neural_generator' else name
        original=getattr(owner,attr)
        def guarded(*args,_name=name,_fn=original,**kwargs):
            model.r13_counts['generators'][_name]+=1
            if arm=='llm_no_online_generators':raise AssertionError('Removed generator executed: '+_name)
            return _fn(*args,**kwargs)
        setattr(owner,attr,guarded)
    if arm=='llm_no_online_generators':
        def forbidden_pool(*args,**kwargs):raise AssertionError('Removed candidate bank executed')
        model._candidate_pool=forbidden_pool;model.neural_generator=None
    if model.router_model is not None:
        def residual_hook(*args):model.r13_counts['residual_forward']+=1
        model.router_model.register_forward_hook(residual_hook)
    if model.baseline_model is not None:
        generator=model.baseline_model.generator
        def anchor_hook(*args):model.r13_counts['anchor_forward']+=1
        generator.register_forward_hook(anchor_hook)
        if hasattr(generator,'competing_proposal'):
            original=generator.competing_proposal
            def temperature(*args,_fn=original,**kwargs):
                model.r13_counts['anchor_temperature_proposal']+=1
                return _fn(*args,**kwargs)
            generator.competing_proposal=temperature
    original=model._baseline_candidates
    def reference(*args,**kwargs):
        result=original(*args,**kwargs)
        model.r13_references.append(result.detach().cpu().clone())
        return result
    model._baseline_candidates=reference
    if getattr(model,'fusion_agent',None) is not None:
        original_prepare=model.fusion_agent.prepare_generation
        def prepare(*args,**kwargs):
            model.r13_counts['prepare_generation']+=1
            return original_prepare(*args,**kwargs)
        model.fusion_agent.prepare_generation=prepare
    return model


def calls_for(fid):
    if fid not in CALL_CACHE:CALL_CACHE[fid]=RUN.storage.full_calls('CEC-style',fid)[0]
    return CALL_CACHE[fid]


def paired_start(fid,batch):
    path=PARENT/'results'/f'CEC-style_f{fid:02d}'/f'b{batch:02d}'
    with gzip.open(path/'PAIRED_START.pkl.gz','rb') as f:paired=pickle.load(f)
    return paired,path


def recorder(problem,cache,schedule=None):
    import torch
    class Recorded(RUN.legacy.Counted):
        def __init__(self):
            super().__init__(problem,cache,stochastic=True)
            self.rng_before=[];self.points=[]
        def calfitness(self,candidates):
            before=torch.get_rng_state().clone()
            if schedule is not None:assert torch.equal(before,schedule[len(self.rng_before)]),'Objective RNG pairing failed'
            self.rng_before.append(before);self.points.append(candidates.detach().cpu().clone())
            return super().calfitness(candidates)
    return Recorded()


def check_decisions(records,points,arm,agent,live):
    import numpy as np
    assert len(records)==45
    assert agent.stats['requests']+agent.stats.get('api_reused',0)==(2 if live else 0)
    assert len([r for r in records if r['request_event']])==(2 if live else 0)
    configs=None
    for k,r in enumerate(records):
        assert r['step']==k and len(r['current'])==10
        assert all(c['generation']['nfe_before']==210+2*k for c in r['current'])
        assert all(c['generation']['no_current_objective'] for c in r['current'])
        if k in (5,15):configs=list(r['configs'])
        expected=configs if k in (5,6,7,8,9,15,16,17,18,19) else ['release']*10
        assert r['configs']==expected
        applied=['matched38']*10 if k in (5,15) else ([n.split('@')[0] for n in configs] if k in (6,7,8,9,16,17,18,19) else ['release']*10)
        assert r['applied_generation']==applied
        for i,c in enumerate(r['current']):
            assert np.array_equal(points[i,55+k,0],np.asarray(c['protected_point'],dtype=np.float32))
            assert np.array_equal(points[i,55+k,1],np.asarray(c['candidates'][c['selected_index']],dtype=np.float32))
            assert c['residual_support'] is not None
            if arm=='llm_no_online_generators':
                assert c['candidate_count'] in (1,2) and all(op==-2 for op in c['candidate_ops'])
            else:assert c['candidate_count']==38
            if arm=='llm_no_anchor':assert c['anchor']['enabled'] is False and c['extra_origin']=='native_extra'


def run_batch(task):
    import numpy as np
    import torch
    fid,batch=task;group='CEC-style';name=f'{group}_f{fid:02d}'
    target=EXP/'results'/name/f'b{batch:02d}';target.mkdir(parents=True,exist_ok=True)
    identity=RUN.binding();marker=target/'BATCH_COMPLETE.json'
    if marker.exists():
        value=readj(marker);assert value['binding']==identity;return value
    paired,source=paired_start(fid,batch)
    population=torch.from_numpy(paired['population'].copy());rng=torch.from_numpy(paired['rng'].copy())
    hashes=dict(paired_initial_sha256=sha_bytes(population.numpy().tobytes()),
                paired_solver_rng_sha256=sha_bytes(rng.numpy().tobytes()))
    frozen=readj(source/'llm_live.json');assert all(hashes[k]==frozen[k] for k in hashes)
    problem=RUN.legacy.problem_for(group,fid)
    assert problem.getfunname()==f'cecf{fid}'
    cache=calls_for(fid)[batch]
    # Load exact objective cache entries from previous successful arms on resume.
    for path in sorted(target.glob('*_NATIVE_CALLS.pkl.gz')):
        with gzip.open(path,'rb') as f:cache=RUN.storage.merged_calls(cache,pickle.load(f))
    schedule=None
    for arm in ARMS:
        receipt_path=target/(arm+'.json')
        if receipt_path.exists():
            assert readj(receipt_path)['binding']==identity
            if arm=='frozen_baseline':
                with gzip.open(target/'OBJECTIVE_RNG_SCHEDULE.pkl.gz','rb') as f:schedule=pickle.load(f)
            continue
        model=build(arm);agent=getattr(model,'fusion_agent',None);records=[]
        if agent is not None:
            agent.active_record=records.append
            agent.execution_context=dict(round='r13_cec_uniform',group=group,fid=fid,batch=batch,arm=arm,**hashes,
                execution_policy='same_as_bbob_no_cec_structural_exclusion',
                frozen_parent_binding_sha256=sha(PARENT/'CODE_BINDING.json'))
        if arm=='llm_no_online_generators':
            assert schedule is not None;model.objective_rng_schedule=schedule
        measured=recorder(problem,cache,schedule)
        torch.set_rng_state(rng);began=time.monotonic()
        with torch.no_grad():_,trail,nfe,executed=model(population.clone(),measured)
        values=torch.cat(measured.values,1).numpy()
        points=executed.numpy().reshape(10,100,2,10)
        refs=torch.stack(model.r13_references,1).numpy()
        assert nfe==300 and values.shape==(10,300) and np.isfinite(values).all()
        assert measured.native_fe+measured.cached_fe==3000 and len(measured.rng_before)==201
        assert np.array_equal(values.min(1),trail[:,-1].numpy())
        assert np.array_equal(points[:,:,0],refs[:,:,0])
        assert torch.get_rng_state().numpy().tobytes()==paired['original_after'].tobytes()
        if arm=='frozen_baseline':
            assert np.array_equal(values,np.load(source/'off.npz')['values'])
            assert np.array_equal(points,refs) and not records and model.r13_counts['residual_forward']==0
            schedule=measured.rng_before
            assert all(sha_bytes(s.numpy().tobytes())==calls_for(fid)[batch][i][0]['rng'] for i,s in enumerate(schedule))
            RUN.storage.save_pickle(target/'OBJECTIVE_RNG_SCHEDULE.pkl.gz',schedule)
        else:
            assert model.r13_counts['residual_forward']>0,'Residual CEC exclusion still active'
            if agent is None:assert arm=='no_agent' and not records
            else:
                assert model.r13_counts['prepare_generation']==45
                assert not agent.resource_exhausted and not agent.uncertain_api_outcome
                check_decisions(records,points,arm,agent,live=True)
            if arm=='llm_no_online_generators':assert model.neural_generator is None and not any(model.r13_counts['generators'].values())
            elif arm=='llm_no_anchor':assert model.baseline_model is None and model.r13_counts['anchor_forward']==0
            else:assert all(n>=100 for n in model.r13_counts['generators'].values())
        np.savez_compressed(target/(arm+'.npz'),values=values,executed_points=points,reference_points=refs)
        with gzip.open(target/(arm+'_DECISIONS.json.gz'),'wt') as f:json.dump(records,f,allow_nan=False)
        if measured.new_calls:
            RUN.storage.save_pickle(target/(arm+'_NATIVE_CALLS.pkl.gz'),measured.new_calls)
            cache=RUN.storage.merged_calls(cache,measured.new_calls)
        receipt=dict(group=group,fid=fid,batch=batch,arm=arm,n=10,nfe=300,**hashes,binding=identity,
            native_fe=measured.native_fe,cached_fe=measured.cached_fe,elapsed_seconds=time.monotonic()-began,
            agent_present=agent is not None,agent_stats={} if agent is None else dict(agent.stats),
            route_counts=model.r13_counts,objective_rng_boundaries_matched=201,
            final_rng_matches_frozen_original=True,baseline_archived_cpu=arm=='frozen_baseline',
            objective_name=problem.getfunname(),cec_structural_exclusion=arm=='frozen_baseline')
        write_json(receipt_path,receipt)
        print(json.dumps(dict(event='arm_batch_complete',fid=fid,batch=batch,arm=arm,
                             api=receipt['agent_stats'].get('requests',0),native_fe=measured.native_fe,
                             cached_fe=measured.cached_fe)),flush=True)
    value=dict(group=group,fid=fid,batch=batch,status='complete',binding=identity)
    write_json(marker,value);return value


def sha_bytes(data):return hashlib.sha256(data).hexdigest()


def worker(task):
    try:return run_batch(task)
    except Exception:
        folder=EXP/'failures';folder.mkdir(exist_ok=True)
        write_json(folder/f'f{task[0]:02d}_b{task[1]:02d}_{time.time_ns()}.json',dict(task=task,traceback=traceback.format_exc()))
        raise


class R13Progress(Progress):
    def event(self,**data):
        if 'total_arm_batches' in data:
            completed=sum(1 for arm in ARMS for _ in self.directory.glob(f'results/*/b??/{arm}.json'))
            data.update(completed_arm_batches=completed,total_arm_batches=300,arm_batch_percent=round(completed/3,1))
            data.pop('eta_arm_batch_seconds',None)
        super().event(**data)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',nargs='?',default='all',choices=['all','run','report','verify'])
    parser.add_argument('--workers',type=int,default=24);parser.add_argument('--progress-seconds',type=float,default=10)
    args=parser.parse_args();assert Path(sys.prefix).name=='ooneurgo' and 1<=args.workers<=24
    runtime=readj(EXP/'HARDWARE_RUNTIME.json');assert runtime['workers']==args.workers
    with (EXP/'.run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        stamp=time.strftime('%Y%m%d_%H%M%S');stats=EXP/f'GPU_STATS_{stamp}.json'
        progress=R13Progress(EXP,args.progress_seconds,stats);gpu=None
        try:
            with tempfile.TemporaryDirectory(prefix='roopf_r13_') as temp:
                address,authkey=str(Path(temp)/'cuda.sock'),secrets.token_bytes(32)
                if args.phase in ('all','run'):
                    ctx=mp.get_context('spawn');ready,child=ctx.Pipe(duplex=False)
                    gpu=ctx.Process(target=serve,args=(address,authkey,runtime['visible_devices'],child,str(stats),runtime['gpu_concurrency']))
                    gpu.start();child.close();assert ready.poll(60),'CUDA init timeout'
                    metadata=ready.recv();ready.close();assert metadata['status']=='ready',metadata
                    write_json(EXP/f'HARDWARE_{stamp}.json',metadata);progress.event(event='gpu_ready',**metadata)
                else:address=authkey=None
                run=configure(str(EXP),runtime);identity=run.binding();bound=EXP/'CODE_BINDING.json'
                if bound.exists():assert readj(bound)==identity,'Frozen round source changed'
                else:write_json(bound,identity)
                phases=['run','report','verify'] if args.phase=='all' else [args.phase]
                for phase in phases:
                    if phase=='run':
                        tasks=[(f,b) for b in range(10) for f in range(1,7)]
                        progress.begin('run',60)
                        with mp.get_context('spawn').Pool(args.workers,initializer=configure,
                                initargs=(str(EXP),runtime,address,authkey)) as pool:
                            pending=pool.imap_unordered(worker,tasks,chunksize=1)
                            while progress.done<60:
                                try:value=pending.next(timeout=1)
                                except mp.TimeoutError:continue
                                progress.done+=1;progress.event(event='batch_complete',fid=value['fid'],batch=value['batch'],completed=progress.done,total=60)
                    else:
                        progress.begin(phase,1)
                        from report_r13_cec_uniform import report
                        result=report(run,verify=phase=='verify');progress.done=1
                        progress.event(event='report_result',status=result['status'])
                    progress.event(event='stage_complete',stage=phase)
        except BaseException:
            write_json(EXP/f'FAILURE_{stamp}.json',dict(status='partial_or_failed',traceback=traceback.format_exc()));raise
        finally:
            if gpu is not None:
                if gpu.is_alive():
                    from multiprocessing.connection import Client
                    try:
                        with Client(address,family='AF_UNIX',authkey=authkey) as connection:connection.send(dict(op='shutdown'))
                    except OSError:pass
                gpu.join(timeout=5)
                if gpu.is_alive():gpu.terminate();gpu.join(timeout=5)
            progress.close()


if __name__=='__main__':main()
