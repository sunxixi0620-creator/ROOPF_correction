"""Hardware-benchmarked fixed-center ES reliability; no training loop."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
from pathlib import Path
import sys,json,time,resource
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import torch
import numpy as np
from torch.nn.utils import parameters_to_vector,vector_to_parameters
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import terminal_training as old
from roopf.complementary_proposal import build_context
from roopf.revision_tasks import ProceduralTask,population
from roopf.experiment_io import CaseStore,sha256,write_json,save_torch,fingerprint,seed_for
from scripts.unified_revision import bounded
RUN=ROOT/'results/es_reliability_20260930';OUT=ROOT/'docs/revision/es_reliability'
PARENT=ROOT/'results/terminal_training_20260930'
SOURCES=['scripts/es_reliability.py','docs/experiments/ES_RELIABILITY_PROTOCOL.md']

def noise(s,g,d):return torch.randn(58504,generator=torch.Generator().manual_seed(seed_for('es_reliability',s,g,d)))

def make(s,offset=None,device='cpu'):
    p=old.proposal(s);p.load_state_dict(torch.load(PARENT/f'selected_{s}.pt',weights_only=False))
    if offset is not None:
        origin=torch.load(PARENT/f'origin_{s}.pt',weights_only=False)
        vector_to_parameters(parameters_to_vector(p.parameters())+origin['scale']*offset,p.parameters())
    return build_context(proposal=p).eval().requires_grad_(False).to(device)

def simulate(spec):
    torch.set_num_threads(1);device=spec.get('device','cpu');s=spec['seed'];offset=None
    if spec['role']=='estimation':offset=spec['sign']*.02*noise(s,spec['group'],spec['direction'])
    elif spec.get('group') in ('A','B'):
        p=RUN/f"step_s{s}_{spec['group']}.pt";assert sha256(p)==spec['step_hash'];offset=torch.load(p,weights_only=False)
    model=make(s,offset,device)
    split='reliability_v1_'+spec['role'];count=4 if spec['role']=='transfer' else 2
    task=ProceduralTask(spec['fid'],spec['instance'],split,dim=20,device=device)
    x=population(spec['fid'],spec['instance'],split,count=count,dim=20,device=device)
    torch.manual_seed(seed_for('reliability_v1',spec['role'],spec['fid'],spec['instance'],'policy'))
    before=fingerprint(model.state_dict())
    if device=='cuda':torch.cuda.synchronize()
    start=time.perf_counter()
    with torch.no_grad():_,trail,nfe,points=model(x,task)
    if device=='cuda':torch.cuda.synchronize()
    elapsed=time.perf_counter()-start
    assert nfe==600 and task.points==600*count and task.diagnostic_points==0 and torch.isfinite(trail).all()
    assert before==fingerprint(model.state_dict())
    return dict(utility=bounded(task.initial_values.min(1).values,trail[:,-1],task.initial_values.std(1)).cpu(),
        trail=trail.cpu(),points_sha=fingerprint(points),main_calls=task.points,teacher_calls=0,
        seconds=elapsed,rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)

def verify():
    identity=json.loads((RUN/'identity.json').read_text())
    for p,h in identity['sources'].items():assert sha256(ROOT/p)==h
    for s,h in enumerate(identity['models']):assert sha256(PARENT/f'selected_{s}.pt')==h
    old.verify();return identity

def worker(spec):
    store=CaseStore(RUN/'cases',verify());key=fingerprint(spec)[:32]
    with store.lock(key):
        result=store.load(key,spec)
        if result is None:result=simulate(spec);store.save(key,spec,result)
    return result

def warm(_):
    torch.set_num_threads(1);return os.getpid()

def available_gib():
    lines=Path('/proc/meminfo').read_text().splitlines()
    return int(next(x for x in lines if x.startswith('MemAvailable:')).split()[1])/1024**2

def benchmark():
    path=RUN/'RESOURCES.json'
    if path.exists():return json.loads(path.read_text())
    jobs=[dict(role='timing',seed=0,fid=i%36,instance=i//36) for i in range(64)]
    runs={};reference=None;rss=0
    for workers in (16,24,32):
        if workers>16 and available_gib()-workers*rss/1024<8:
            runs[str(workers)]=dict(skipped='memory headroom');continue
        with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn')) as pool:
            list(pool.map(warm,range(workers*2)))
            start=time.perf_counter();results=list(pool.map(simulate,jobs));elapsed=time.perf_counter()-start
        hashes=[x['points_sha'] for x in results]
        if reference is None:reference=hashes
        else:assert hashes==reference
        rss=max(x['rss_mb'] for x in results)
        runs[str(workers)]=dict(seconds=elapsed,trajectories_per_second=128/elapsed,worker_peak_rss_mb=rss,calls=76800)
        write_json(RUN/'RESOURCE_PROGRESS.json',runs)
    comparisons={}
    for device in ('cpu','cuda'):
        if device=='cuda' and not torch.cuda.is_available():continue
        spec=dict(role='device_timing',seed=0,fid=0,instance=0,device=device)
        t=time.perf_counter();value=simulate(spec)
        comparisons[device]=dict(inference_seconds=value['seconds'],including_setup_seconds=time.perf_counter()-t,calls=value['main_calls'])
    chosen=min((w for w in runs if 'seconds' in runs[w]),key=lambda w:runs[w]['seconds'])
    report=dict(cpu_available=os.cpu_count(),memory_available_gib=available_gib(),worker_trials=runs,
        device_trials=comparisons,chosen_workers=int(chosen),cpu_fingerprints_identical=True,
        scientific_device='cpu',note='CUDA timing only; do not mix numerical/RNG paths',
        calls=sum(v.get('calls',0) for v in runs.values())+sum(v['calls'] for v in comparisons.values()))
    write_json(path,report);write_json(OUT/'RESOURCES.json',report);return report

def interval(delta):
    recipes=delta.reshape(3,12,3,2,4).mean((0,2,3,4))
    rng=np.random.default_rng(20261002);d=recipes[rng.integers(0,12,(5000,12))].mean(1)
    lo,hi=np.quantile(d,[.0125,.9875]);seeds=delta.mean((1,2,3))
    return dict(mean=float(delta.mean()),lower=float(lo),upper=float(hi),seed_means=seeds.tolist(),
        passed=bool(delta.mean()>=.0001 and lo>0 and (seeds>0).all()))

def main():
    RUN.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
    if (RUN/'COMPLETE.json').exists():print('Already complete; do not rerun.');return
    if not (RUN/'identity.json').exists():
        identity=dict(version='es_reliability_v1',sources={p:sha256(ROOT/p) for p in SOURCES},
            models=[sha256(PARENT/f'selected_{s}.pt') for s in range(3)])
        write_json(RUN/'identity.json',identity)
    verify();write_json(RUN/'STATUS.json',dict(status='resource_benchmark'))
    resources=benchmark()
    fids=np.random.default_rng(20261001).permutation(36)[:6].tolist()
    with ProcessPoolExecutor(max_workers=resources['chosen_workers'],mp_context=mp.get_context('spawn')) as pool:
        write_json(RUN/'STATUS.json',dict(status='estimating_directions',workers=resources['chosen_workers']))
        jobs=[dict(role='estimation',seed=s,group=g,direction=d,sign=sign,fid=f,instance=0)
              for s in range(3) for g in ('A','B') for d in range(8) for sign in (1,-1) for f in fids]
        results=list(pool.map(worker,jobs))
        scores=np.array([float(v['utility'].mean()) for v in results]).reshape(3,2,8,2,6)
        estimates=[]
        for s in range(3):
            grads=[]
            for gi,g in enumerate(('A','B')):
                differences=(scores[s,gi,:,0]-scores[s,gi,:,1]).mean(1)
                grad=sum(float(differences[d])*noise(s,g,d) for d in range(8))/.32
                assert torch.isfinite(grad).all()
                phi=torch.nn.Parameter(torch.zeros(58504));opt=torch.optim.Adam([phi],lr=.01)
                phi.grad=-grad;torch.nn.utils.clip_grad_norm_([phi],1);opt.step()
                save_torch(RUN/f'step_s{s}_{g}.pt',phi.detach());save_torch(RUN/f'gradient_s{s}_{g}.pt',grad)
                grads.append(grad)
            estimates.append(dict(seed=s,norm_A=float(grads[0].norm()),norm_B=float(grads[1].norm()),
                cosine=float(torch.nn.functional.cosine_similarity(grads[0],grads[1],dim=0))))
        write_json(RUN/'DIRECTIONS.json',estimates);write_json(OUT/'DIRECTIONS.json',estimates)
        write_json(RUN/'STATUS.json',dict(status='transfer',workers=resources['chosen_workers']))
        jobs=[dict(role='transfer',seed=s,group=g,fid=f,instance=i,
            step_hash=sha256(RUN/f'step_s{s}_{g}.pt') if g!='Base' else None)
            for s in range(3) for g in ('Base','A','B') for f in range(36) for i in range(2)]
        results=list(pool.map(worker,jobs));matrix={g:np.zeros((3,36,2,4)) for g in ('Base','A','B')}
        for spec,value in zip(jobs,results):matrix[spec['group']][spec['seed'],spec['fid'],spec['instance']]=value['utility'].numpy()
        effects={g:interval(matrix[g]-matrix['Base']) for g in ('A','B')}
        save_torch(RUN/'utilities.pt',matrix)
    result=dict(all_workers_joined=True,resources=resources,directions=estimates,effects=effects,
        locally_reliable=all(e['passed'] for e in effects.values()),estimation_calls=691200,transfer_calls=1555200,
        teacher_calls=0,interpretation='cosine alone is not a high-dimensional reliability test')
    write_json(RUN/'COMPLETE.json',result);write_json(OUT/'RESULTS.json',result)
    write_json(RUN/'STATUS.json',dict(status='complete'));print(json.dumps(result),flush=True)

if __name__=='__main__':main()
