"""Matched novel-region branch continuations; diagnostic oracle, never deployment."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
from pathlib import Path
import sys,json,time,types,multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import torch
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import es_reliability as parent
from roopf.online_portfolio import build_online
from roopf.revision_tasks import ProceduralTask,population
from roopf.experiment_io import CaseStore,sha256,fingerprint,seed_for,write_json,save_torch
from scripts.unified_revision import bounded
RUN=ROOT/'results/region_potential_20260930';OUT=ROOT/'docs/revision/region_potential'
SOURCES=['scripts/region_potential.py','docs/experiments/REGION_POTENTIAL_PROTOCOL.md']
STEPS=(0,105);MENUS=('Iso','Pop','Center','Offline_0','Offline_1','Offline_2')


def make(budget=600):return build_online(20,budget).requires_grad_(False)


def setup(fid,pop_id,role='region_potential_v1'):
    task=ProceduralTask(fid,0,role,dim=20)
    x=population(fid,0,role,count=2,dim=20)[pop_id:pop_id+1].clone()
    return task,x,seed_for(role,fid,pop_id,'policy')


def instrument(model,steps,action=None):
    snapshots={};orig_pool=model._candidate_pool;orig_score=model.scores
    current={}
    def pool(self,x,fitness,problem,state,raw,memory):
        current.update(x=x,fitness=fitness,memory=memory)
        return orig_pool(x,fitness,problem,state,raw,memory)
    def scores(self,candidates,prior,ax,ay,problem,remaining,stagnation):
        score=orig_score(candidates,prior,ax,ay,problem,remaining,stagnation)
        step=(self.evalnum-100)//2
        if step in steps:
            chosen=torch.argsort(score,dim=1,stable=True)[:,:2]
            state=dict(x=current['x'].clone(),fitness=current['fitness'].clone(),memory=current['memory'].clone(),
                ax=ax.clone(),ay=ay.clone(),candidates=candidates.clone(),prior=prior.clone(),scores=score.clone(),
                chosen=chosen.clone(),remaining=remaining,stagnation=stagnation.clone(),rng=torch.get_rng_state().clone())
            state['fingerprint']=fingerprint(state);snapshots[step]=state
            if action is not None and step==action['step']:
                assert state['fingerprint']==action['fingerprint']
                if action['replace']:
                    first=int(chosen[0,0]);slot=1 if first==0 else 0
                    candidates[0,slot]=action['point']
                    score=score.clone();floor=score.min()-2
                    score[0,first]=floor;score[0,slot]=floor+1
                    assert torch.argsort(score,dim=1,stable=True)[0,:2].tolist()==[first,slot]
        return score
    model._candidate_pool=types.MethodType(pool,model);model.scores=types.MethodType(scores,model)
    return snapshots


def rollout(model,fid,pop_id,role='region_potential_v1'):
    task,x,ps=setup(fid,pop_id,role);initial_x=x.clone()
    before=fingerprint(model.state_dict());torch.manual_seed(ps)
    with torch.no_grad():_,trail,nfe,points=model(x,task)
    assert nfe==model.MaxNFE and task.points==model.MaxNFE and task.diagnostic_points==0
    assert torch.isfinite(trail).all() and fingerprint(model.state_dict())==before
    return dict(trail=trail,points=points,initial_x=initial_x,initial=task.initial_values,main_calls=task.points)


def verify():
    identity=json.loads((RUN/'identity.json').read_text())
    for p,h in identity['sources'].items():assert sha256(ROOT/p)==h
    parent.verify();assert identity['parent_identity']==sha256(parent.RUN/'identity.json')
    return identity


def contracts():
    path=RUN/'CONTRACTS.json'
    if path.exists():return
    role='region_contract_v1'
    a=rollout(make(114),4,0,role)
    model=make(114);snap=instrument(model,(2,));b=rollout(model,4,0,role)
    assert torch.equal(a['points'],b['points']) and torch.equal(a['trail'],b['trail'])
    state=snap[2]
    model=make(114);instrument(model,(2,),dict(step=2,fingerprint=state['fingerprint'],replace=False))
    c=rollout(model,4,0,role)
    assert torch.equal(b['points'],c['points']) and torch.equal(b['trail'],c['trail'])
    point=torch.full((20,),4.5)
    model=make(114);instrument(model,(2,),dict(step=2,fingerprint=state['fingerprint'],replace=True,point=point))
    d=rollout(model,4,0,role)
    assert torch.equal(d['points'][:,:4],b['points'][:,:4])
    assert torch.equal(d['points'][:,4],b['points'][:,4]) and torch.equal(d['points'][0,5],point)
    save_torch(RUN/'contract_trajectories.pt',dict(uninstrumented=a,instrumented=b,noop=c,forced=d))
    write_json(path,dict(calls=456,zero_extra_fitness=True,no_op_exact=True,prefix_exact=True,forced_slot_exact=True))


def menus(state,base,step,fid,pop_id):
    best=state['x'][0,0];dim=20
    g=torch.Generator().manual_seed(seed_for('region_menus_v1',fid,pop_id,step))
    directions={'Iso':torch.randn(2,dim,generator=g)}
    indices=torch.randint(0,100,(2,2),generator=g)
    directions['Pop']=state['x'][0,indices[:,0]]-state['x'][0,indices[:,1]]
    directions['Center']=torch.stack((-best,state['x'][0].mean(0)-best))
    intent=state['candidates'].gather(1,state['chosen'][:,:,None].expand(-1,-1,20))
    scores=state['scores'].gather(1,state['chosen'])
    task,_,_=setup(fid,pop_id)
    for s in range(3):
        model=parent.make(s).proposal
        before=fingerprint(model.state_dict())
        with torch.no_grad():
            pts=model(state['x'],state['fitness'],intent,scores,state['remaining'],state['stagnation'],task)
        assert fingerprint(model.state_dict())==before and task.points==0 and task.diagnostic_points==0
        directions[f'Offline_{s}']=pts[0]-best
    known=torch.cat((base['initial_x'][0],base['points'][0,:2*step],state['candidates'][0]),0)
    result={}
    for name in MENUS:
        d=directions[name];norm=d.square().mean(1).sqrt()
        points=[];metadata=[]
        for j in range(2):
            for radius in (.1,.2):
                raw=best+d[j]/norm[j].clamp_min(1e-12)*(10*radius)
                point=raw.clamp(-5,5)
                distance=float(((known-point).square().mean(1).sqrt()/10).min())
                valid=bool(norm[j]>1e-12 and distance>=.05)
                points.append(point)
                metadata.append(dict(direction=j,radius=radius,eligible=valid,min_distance=distance,
                    clipped=bool(not torch.equal(raw,point)),actual_radius=float((point-best).square().mean().sqrt()/10)))
        result[name]=dict(points=torch.stack(points),metadata=metadata)
    return result


def worker(job):
    torch.set_num_threads(1);fid,pop_id=job;identity=verify();store=CaseStore(RUN/'cases',identity)
    spec=dict(fid=fid,population=pop_id);key=f'f{fid}_p{pop_id}'
    with store.lock(key):
        existing=store.load(key,spec)
        if existing is not None:return dict(key=key,calls=existing['main_calls'])
        start=time.perf_counter();model=make();snap=instrument(model,STEPS);base=rollout(model,fid,pop_id)
        init=base['initial'].min(1).values;scale=base['initial'].std(1)
        branches={};candidate_menus={};calls=600;prefix_calls=0;valid=0
        for step in STEPS:
            state=snap[step];candidates=menus(state,base,step,fid,pop_id);candidate_menus[step]=candidates
            for name in MENUS:
                for k in range(4):
                    eligible=candidates[name]['metadata'][k]['eligible']
                    point=candidates[name]['points'][k]
                    if eligible:
                        model=make();instrument(model,(step,),dict(step=step,fingerprint=state['fingerprint'],replace=True,point=point))
                        branch=rollout(model,fid,pop_id);calls+=600;valid+=1;prefix_calls+=100+2*step
                        assert torch.equal(branch['points'][:,:2*step],base['points'][:,:2*step])
                        assert torch.equal(branch['trail'][:,:step],base['trail'][:,:step])
                        assert torch.equal(branch['points'][:,2*step],base['points'][:,2*step])
                        assert torch.equal(branch['points'][0,2*step+1],point)
                    else:branch=base
                    def advantage(at):
                        return float((bounded(init,branch['trail'][:,at],scale)-bounded(init,base['trail'][:,at],scale))[0])
                    branches[f'{step}_{name}_{k}']=dict(eligible=eligible,trail=branch['trail'],points_sha=fingerprint(branch['points']),
                        immediate=advantage(step),after20=advantage(min(step+10,249)),after60=advantage(min(step+30,249)),
                        final=advantage(249),main_calls=600 if eligible else 0)
        value=dict(base=base,snapshots=snap,menus=candidate_menus,branches=branches,main_calls=calls,
            branch_calls=valid*600,redundant_prefix_calls=prefix_calls,off_rollout_teacher_calls=0,seconds=time.perf_counter()-start)
        store.save(key,spec,value)
        return dict(key=key,calls=calls)


def interval(delta,threshold):
    # 3 model seeds x36 configurations x2 populations x2 stages; controls reused.
    recipe=delta.reshape(3,12,3,2,2).mean((0,2,3,4))
    rng=np.random.default_rng(20261005);boot=recipe[rng.integers(0,12,(5000,12))].mean(1)
    lo,hi=np.quantile(boot,[.00625,.99375]);seeds=delta.mean((1,2,3))
    return dict(mean=float(delta.mean()),lower=float(lo),upper=float(hi),seed_means=seeds.tolist(),
        threshold=threshold,passed=bool(delta.mean()>=threshold and lo>0 and (seeds>0).all()))


def summarize():
    identity=verify();store=CaseStore(RUN/'cases',identity)
    potential={m:np.zeros((36,2,2)) for m in MENUS};allmean={m:np.zeros((36,2,2)) for m in MENUS}
    diagnostics={m:dict(actions=0,eligible=0,clipped=0,positive_states=0,delayed_positive_states=0) for m in MENUS}
    calls=0;branch_calls=0;prefix=0
    for f in range(36):
        for p in range(2):
            v=store.load(f'f{f}_p{p}',dict(fid=f,population=p));assert v is not None
            calls+=v['main_calls'];branch_calls+=v['branch_calls'];prefix+=v['redundant_prefix_calls']
            for si,step in enumerate(STEPS):
                for m in MENUS:
                    rows=[v['branches'][f'{step}_{m}_{k}'] for k in range(4)]
                    gains=[r['final'] for r in rows];best=int(np.argmax(gains));gain=max(0.,gains[best])
                    potential[m][f,p,si]=gain;allmean[m][f,p,si]=np.mean(gains)
                    d=diagnostics[m];d['actions']+=4;d['eligible']+=sum(r['eligible'] for r in rows)
                    d['clipped']+=sum(x['clipped'] for x in v['menus'][step][m]['metadata'])
                    d['positive_states']+=int(gain>1e-12)
                    d['delayed_positive_states']+=int(gain>1e-12 and rows[best]['immediate']<=0)
    offline=np.stack([potential[f'Offline_{s}'] for s in range(3)])
    effects={'Offline-Base':interval(offline,.001)}
    effects.update({f'Offline-{m}':interval(offline-potential[m][None],.0005) for m in ('Iso','Pop','Center')})
    save_torch(RUN/'aggregates.pt',dict(potential=potential,allmean=allmean))
    report=dict(effects=effects,passed=all(v['passed'] for v in effects.values()),diagnostics=diagnostics,
        menu_potential_mean={m:float(x.mean()) for m,x in potential.items()},menu_all_candidates_mean={m:float(x.mean()) for m,x in allmean.items()},
        stage_potential={str(100+2*step):{m:float(x[:,:,si].mean()) for m,x in potential.items()} for si,step in enumerate(STEPS)},
        base_calls=72*600,branch_calls=branch_calls,redundant_prefix_calls=prefix,contract_calls=456,
        total_calls=calls+456,actual_trajectories=calls//600,off_rollout_teacher_calls=0,oracle_assisted=True,
        all_workers_joined=True,development_only=True)
    assert report['total_calls']<=2117256
    return report


def main():
    torch.set_num_threads(1);RUN.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
    if (RUN/'COMPLETE.json').exists():print('Already complete; no simulations launched.');return
    start=time.perf_counter()
    if not (RUN/'identity.json').exists():
        parent.verify();write_json(RUN/'identity.json',dict(version='region_potential_v1',
            sources={p:sha256(ROOT/p) for p in SOURCES},parent_identity=sha256(parent.RUN/'identity.json')))
    verify();contracts()
    available=parent.available_gib();workers=32 if available>=26 else 24 if available>=22 else 16
    assert available>=17
    write_json(RUN/'RESOURCES.json',dict(workers=workers,available_gib=available,device='cpu',threads_per_worker=1))
    with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn')) as pool:
        jobs=[(f,p) for f in range(36) for p in range(2)]
        write_json(RUN/'STATUS.json',dict(status='branches',completed=0,total=72,workers=workers))
        for n,_ in enumerate(pool.map(worker,jobs),1):
            write_json(RUN/'STATUS.json',dict(status='branches',completed=n,total=72,workers=workers))
    result=summarize();result['seconds']=time.perf_counter()-start
    write_json(RUN/'COMPLETE.json',result);write_json(OUT/'RESULTS.json',result)
    write_json(RUN/'STATUS.json',dict(status='complete'));print(json.dumps(result),flush=True)


if __name__=='__main__':main()
