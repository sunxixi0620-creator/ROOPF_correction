"""Full strict module ablations of frozen R10; no existing result is overwritten."""
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
EXP=ROOT/'experiments/r12_strict_component_ablations'
PARENT=ROOT/'experiments/r10_current_candidate_fusion'
EXTRA=('roopf/strict_component_ablations.py','scripts/experiment_r12_strict_ablations.py',
 'scripts/report_r12_strict_ablations.py','scripts/run_r12_strict_ablations.sh',
 'scripts/generate_r12_sources.py','scripts/r12_model_header.txt','scripts/check_r12_strict_ablations.py')
GENERATORS=('neural_generator','_current_to_pbest','_covariance_elite','_trust_region','_coordinate_pattern','_opposition_restart')


def configure(directory,runtime,address=None,authkey=None):
    global EXP,RUN
    EXP=Path(directory);RUN,_=configure_r10(directory,runtime,address,authkey)
    RUN.SOURCES=tuple(dict.fromkeys((*RUN.SOURCES,*EXTRA)))
    return RUN


def build(arm,mode='llm'):
    from roopf.strict_component_ablations import NoOnlineGeneratorsOptimizer,NoOnlineGeneratorsAgent
    model=RUN.legacy.build()
    client=getattr(RUN,'_hardware_proxy_client',None)
    if client is not None:
        proxy=model.surrogate
        def predict(archive_x,archive_y,candidates,problem,distance_scale=.15):
            return client.predict(proxy,archive_x,archive_y,candidates,problem,distance_scale)
        proxy.predict=predict
    model.strict_generator_counts={name:0 for name in GENERATORS}
    if arm=='llm_no_online_generators':
        model.__class__=NoOnlineGeneratorsOptimizer
        model.fusion_agent=NoOnlineGeneratorsAgent(RUN.MEMORY,EXP/'api_journal',mode=mode,arm='joint',
            budget=dict(used=0,limit=RUN.settings()['max_new_api_attempts']),api_config=RUN.settings()['api'])
    else:
        assert arm=='no_agent' and model.fusion_agent is None and model.agentic_controller is None
    for name in GENERATORS:
        owner=model.neural_generator if name=='neural_generator' else model
        attribute='forward' if name=='neural_generator' else name
        original=getattr(owner,attribute)
        def guarded(*args,_name=name,_original=original,**kwargs):
            model.strict_generator_counts[_name]+=1
            if arm=='llm_no_online_generators':raise AssertionError('Removed online generator called: '+_name)
            return _original(*args,**kwargs)
        setattr(owner,attribute,guarded)
    if arm=='llm_no_online_generators':
        def forbidden_pool(*args,**kwargs):raise AssertionError('Removed online generator bank called')
        model._candidate_pool=forbidden_pool
        model.neural_generator=None
    return model


def recorder(problem,cache,schedule=None):
    import torch
    class Recorded(RUN.legacy.Counted):
        def __init__(self):
            super().__init__(problem,cache,stochastic=True)
            self.rng_before=[];self.points=[]
        def calfitness(self,candidates):
            state=torch.get_rng_state().clone()
            if schedule is not None:assert torch.equal(state,schedule[len(self.rng_before)]),'Objective RNG pairing broke'
            self.rng_before.append(state);self.points.append(candidates.detach().cpu().clone())
            return super().calfitness(candidates)
    return Recorded()


def run_function(task):
    import numpy as np
    import torch
    group,fid=task;name=f'{group}_f{fid:02d}'
    source=PARENT/'results'/name;folder=EXP/'results'/name
    folder.mkdir(parents=True,exist_ok=True)
    identity=RUN.binding();marker=folder/'COMPLETE.json'
    if marker.exists():
        receipt=json.loads(marker.read_text());assert receipt['binding']==identity;return receipt
    models={arm:build(arm) for arm in RUN.settings()['arms']}
    problem=RUN.legacy.problem_for(group,fid)
    original_calls,_=RUN.storage.full_calls(group,fid)
    for batch in range(10):
        target=folder/f'b{batch:02d}';target.mkdir(exist_ok=True)
        with gzip.open(source/f'b{batch:02d}/PAIRED_START.pkl.gz','rb') as stream:paired=pickle.load(stream)
        population=torch.from_numpy(paired['population'].copy());rng=torch.from_numpy(paired['rng'].copy())
        hashes=dict(paired_initial_sha256=hashlib.sha256(population.numpy().tobytes()).hexdigest(),
            paired_solver_rng_sha256=hashlib.sha256(rng.numpy().tobytes()).hexdigest())
        frozen=json.loads((source/f'b{batch:02d}/llm_live.json').read_text())
        assert all(hashes[k]==frozen[k] for k in hashes)
        cache=original_calls[batch]
        for path in sorted((source/f'b{batch:02d}').glob('*_NATIVE_CALLS.pkl.gz')):
            with gzip.open(path,'rb') as stream:cache=RUN.storage.merged_calls(cache,pickle.load(stream))
        schedule=None
        for arm in RUN.settings()['arms']:
            receipt_path=target/(arm+'.json')
            if receipt_path.exists():
                assert json.loads(receipt_path.read_text())['binding']==identity
                if arm=='no_agent':
                    with gzip.open(target/'OBJECTIVE_RNG_SCHEDULE.pkl.gz','rb') as stream:schedule=pickle.load(stream)
                continue
            model=models[arm];model.strict_generator_counts={name:0 for name in GENERATORS};records=[]
            if arm=='llm_no_online_generators':
                assert schedule is not None
                model.objective_rng_schedule=schedule
                agent=model.fusion_agent;agent.active_record=records.append
                agent.execution_context=dict(group=group,fid=fid,batch=batch,arm=arm,**hashes,
                    strict_ablation='all_online_generators_removed')
            measured=recorder(problem,cache,schedule if arm!='no_agent' else None)
            torch.set_rng_state(rng);began=time.monotonic()
            with torch.no_grad():_,trail,nfe,executed=model(population.clone(),measured)
            values=torch.cat(measured.values,1).numpy()
            assert nfe==300 and values.shape==(10,300) and np.isfinite(values).all()
            assert np.array_equal(values.min(1),trail[:,-1].numpy())
            assert len(measured.rng_before)==201
            assert torch.get_rng_state().numpy().tobytes()==paired['original_after'].tobytes()
            if arm=='no_agent':
                assert model.fusion_agent is None and model.agentic_controller is None and not records
                assert all(v==100 for v in model.strict_generator_counts.values())
                schedule=measured.rng_before
                assert all(hashlib.sha256(state.numpy().tobytes()).hexdigest()==original_calls[batch][i][0]['rng'] for i,state in enumerate(schedule))
                RUN.storage.save_pickle(target/'OBJECTIVE_RNG_SCHEDULE.pkl.gz',schedule)
            else:
                assert not any(model.strict_generator_counts.values())
                assert len(records)==(0 if group=='CEC-style' else 45)
                assert not agent.resource_exhausted and not agent.uncertain_api_outcome
                for record in records:
                    assert all(c['candidate_count'] in (1,2) and all(op==-2 for op in c['candidate_ops']) for c in record['current'])
            executed=executed.numpy().reshape(10,100,2,10)
            np.savez_compressed(target/(arm+'.npz'),values=values,executed_points=executed)
            with gzip.open(target/(arm+'_DECISIONS.json.gz'),'wt') as stream:json.dump(records,stream,allow_nan=False)
            if measured.new_calls:RUN.storage.save_pickle(target/(arm+'_NATIVE_CALLS.pkl.gz'),measured.new_calls)
            receipt=dict(group=group,fid=fid,batch=batch,arm=arm,n=10,nfe=300,**hashes,binding=identity,
                native_fe=measured.native_fe,cached_fe=measured.cached_fe,elapsed_seconds=time.monotonic()-began,
                generator_calls=model.strict_generator_counts,agent_present=model.fusion_agent is not None,
                agent_stats={} if arm=='no_agent' else dict(agent.stats),objective_rng_boundaries_matched=201,
                final_rng_matches_frozen_original=True)
            write_json(receipt_path,receipt)
            print(json.dumps(dict(event='batch_complete',group=group,fid=fid,batch=batch,arm=arm)),flush=True)
        assert schedule is not None
    for arm in RUN.settings()['arms']:
        values=np.concatenate([np.load(folder/f'b{b:02d}/{arm}.npz')['values'] for b in range(10)])
        np.savez_compressed(folder/(arm+'_FULL100.npz'),values=values)
    receipt=dict(status='complete',group=group,fid=fid,n=100,nfe=300,binding=identity)
    write_json(marker,receipt);return receipt


def worker(task):
    try:return run_function(task)
    except Exception:
        write_json(EXP/'results'/f'{task[0]}_f{task[1]:02d}'/'ERROR.json',dict(traceback=traceback.format_exc()))
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',nargs='?',default='all',choices=['all','run','report','verify'])
    parser.add_argument('--workers',type=int,default=24);parser.add_argument('--progress-seconds',type=float,default=5)
    args=parser.parse_args();assert Path(sys.prefix).name=='ooneurgo' and 1<=args.workers<=24
    runtime=json.loads((PARENT/'HARDWARE_RUNTIME.json').read_text());runtime['workers']=args.workers
    runtime['no_agent_proxy_backend']='same CUDA proxy as Full, independently of Agent presence'
    with (EXP/'.run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        path=EXP/'HARDWARE_RUNTIME.json'
        if path.exists():assert json.loads(path.read_text())==runtime
        else:write_json(path,runtime)
        stamp=time.strftime('%Y%m%d_%H%M%S');stats=EXP/f'GPU_STATS_{stamp}.json'
        progress=Progress(EXP,args.progress_seconds,stats);gpu=None
        try:
            with tempfile.TemporaryDirectory(prefix='roopf_r12_') as temp:
                address,authkey=str(Path(temp)/'cuda.sock'),secrets.token_bytes(32)
                if args.phase in ('all','run'):
                    context=mp.get_context('spawn');ready,child=context.Pipe(duplex=False)
                    gpu=context.Process(target=serve,args=(address,authkey,runtime['visible_devices'],child,str(stats),runtime['gpu_concurrency']))
                    gpu.start();child.close();assert ready.poll(60),'GPU init timeout'
                    metadata=ready.recv();ready.close();assert metadata['status']=='ready',metadata
                    progress.event(event='gpu_ready',**metadata);write_json(EXP/f'HARDWARE_{stamp}.json',metadata)
                else:address=authkey=None
                run=configure(str(EXP),runtime);identity=run.binding();frozen=EXP/'CODE_BINDING.json'
                if frozen.exists():assert json.loads(frozen.read_text())==identity
                else:write_json(frozen,identity)
                phases=['run','report','verify'] if args.phase=='all' else [args.phase]
                for phase in phases:
                    if phase=='run':
                        progress.begin('run',35)
                        with mp.get_context('spawn').Pool(args.workers,initializer=configure,initargs=(str(EXP),runtime,address,authkey)) as pool:
                            pending=pool.imap_unordered(worker,run.legacy.TASKS,chunksize=1)
                            while progress.done<35:
                                try:receipt=pending.next(timeout=1)
                                except mp.TimeoutError:continue
                                progress.done+=1;progress.event(event='function_complete',group=receipt['group'],fid=receipt['fid'],completed=progress.done,total=35)
                    else:
                        progress.begin(phase,1)
                        from report_r12_strict_ablations import report
                        result=report(run,verify=phase=='verify');progress.done=1
                        progress.event(event='report_result',status=result['status'])
                    progress.event(event='stage_complete',stage=phase)
                progress.event(event='requested_phases_complete',phases=phases)
        except BaseException:
            write_json(EXP/f'FAILURE_{stamp}.json',dict(status='partial_or_failed',traceback=traceback.format_exc()));raise
        finally:
            if gpu is not None:
                if gpu.is_alive():
                    from multiprocessing.connection import Client
                    try:
                        with Client(address,family='AF_UNIX',authkey=authkey) as conn:conn.send(dict(op='shutdown'))
                    except OSError:pass
                gpu.join(timeout=5)
                if gpu.is_alive():gpu.terminate();gpu.join(timeout=5)
            progress.close()


if __name__=='__main__':main()
