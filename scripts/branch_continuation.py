"""Single oracle intervention; exact prefix restoration and independent RNG."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
from pathlib import Path
import sys,json,types,time,shutil
import torch
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import complementary_proposal_study as parent
from roopf.complementary_proposal import ComplementaryProposal,build_context
from roopf.revision_tasks import ProceduralTask,population
from roopf.experiment_io import CaseStore,sha256,write_json,fingerprint,seed_for
from scripts.unified_revision import bounded
OLD=ROOT/'results/complementary_proposal_20260930'
RUN=ROOT/'results/branch_continuation_20260930'
OUT=ROOT/'docs/revision/branch_continuation'
STEPS=(35,105,210)
SOURCES=['scripts/branch_continuation.py','docs/experiments/BRANCH_CONTINUATION_PROTOCOL.md']

def make(seed,budget=600):
    p=ComplementaryProposal(parent.ANCHORS[seed]).eval().requires_grad_(False)
    p.load_state_dict(torch.load(OLD/f'model_{seed}/selected.pt',weights_only=False))
    model=build_context(proposal=p,budget=budget).eval().requires_grad_(False)
    return model

def setup(fid,pop_id,role='branch_v1'):
    task=ProceduralTask(fid,0,role,dim=20)
    x=population(fid,0,role,count=4,dim=20)[pop_id:pop_id+1].clone()
    return task,x,seed_for(role,fid,pop_id,'policy')

def instrument(model,steps,intervention=None):
    snapshots={};orig_pool=model._candidate_pool;orig_score=model.scores
    def pool(self,x,fitness,problem,state,raw,memory):
        self.audit_memory=memory.clone()
        return orig_pool(x,fitness,problem,state,raw,memory)
    def score(self,candidates,prior,ax,ay,problem,rem,stag):
        result=orig_score(candidates,prior,ax,ay,problem,rem,stag)
        step=(self.evalnum-100)//2
        if step in steps:
            chosen=torch.argsort(result,dim=1,stable=True)[:,:2]
            state=dict(x=self.current_x.clone(),fitness=self.current_fitness.clone(),ax=ax.clone(),ay=ay.clone(),
                candidates=candidates.clone(),scores=result.clone(),prior=prior.clone(),memory=self.audit_memory.clone(),
                rng=torch.get_rng_state().clone(),stagnation=stag.clone(),remaining=rem,chosen=chosen.clone())
            state['fingerprint']=fingerprint(state)
            snapshots[step]=state
            if intervention is not None and step==intervention['step']:
                assert state['fingerprint']==intervention['fingerprint']
                if intervention['replace']:
                    first=int(chosen[0,0]);new=intervention['index']
                    assert new not in chosen[0].tolist() and new in (0,1)
                    # Preserve first/second order, every other score unchanged.
                    result=result.clone();floor=result.min()-2
                    result[0,first]=floor;result[0,new]=floor+1
                    assert torch.argsort(result,dim=1,stable=True)[0,:2].tolist()==[first,new]
        return result
    model._candidate_pool=types.MethodType(pool,model);model.scores=types.MethodType(score,model)
    return snapshots

def rollout(model,fid,pop_id,role='branch_v1'):
    task,x,ps=setup(fid,pop_id,role);before=fingerprint(model.state_dict())
    torch.manual_seed(ps)
    with torch.no_grad():_,trail,nfe,points=model(x,task)
    assert task.diagnostic_points==0 and task.points==model.MaxNFE and nfe==model.MaxNFE
    assert fingerprint(model.state_dict())==before
    return dict(trail=trail,points=points,initial=task.initial_values,main_calls=task.points)

def contracts():
    torch.set_num_threads(1)
    role='branch_contract'
    a=rollout(make(0,114),4,0,role)
    model=make(0,114);snap=instrument(model,(2,))
    b=rollout(model,4,0,role)
    assert torch.equal(a['points'],b['points']) and torch.equal(a['trail'],b['trail'])
    state=snap[2]
    model=make(0,114);instrument(model,(2,),dict(step=2,fingerprint=state['fingerprint'],replace=False))
    c=rollout(model,4,0,role)
    assert torch.equal(b['points'],c['points']) and torch.equal(b['trail'],c['trail'])
    # A fixture may replace a non-extra index solely to exercise continuation.
    new=next((i for i in (0,1) if i not in state['chosen'][0].tolist()),None)
    calls=342
    if new is not None:
        model=make(0,114);instrument(model,(2,),dict(step=2,fingerprint=state['fingerprint'],replace=True,index=new))
        d=rollout(model,4,0,role);calls+=114
        assert torch.equal(d['points'][:,:4],b['points'][:,:4])
        assert torch.equal(d['points'][:,5],state['candidates'][:,new])
    else:raise AssertionError('Fixture must exercise a replacement')
    return dict(passed=True,base_exact=True,no_op_exact=True,prefix_exact=True,slot_replacement=True,
        main_calls=calls,teacher_calls=0)

def verify():
    identity=json.loads((RUN/'identity.json').read_text())
    for p,h in identity['sources'].items():assert sha256(ROOT/p)==h
    for s,h in enumerate(identity['proposals']):assert sha256(OLD/f'model_{s}/selected.pt')==h
    parent.verify(OLD)
    return identity

def worker(job):
    seed,fid,pop_id=job
    torch.set_num_threads(1);identity=verify();store=CaseStore(RUN/'cases',identity)
    case=dict(seed=seed,fid=fid,population=pop_id);key=f's{seed}_f{fid}_p{pop_id}'
    with store.lock(key):
        cached=store.load(key,case)
        if cached is not None:return cached['rows']
        start=time.perf_counter()
        model=make(seed);snap=instrument(model,STEPS)
        base=rollout(model,fid,pop_id)
        teacher,_,_=setup(fid,pop_id)
        rows=[];branches={};labels={}
        init=base['initial'].min(1).values;scale=base['initial'].std(1)
        for step in STEPS:
            state=snap[step];chosen=state['chosen']
            points=torch.cat((state['candidates'][:,:2],state['candidates'].gather(1,chosen[:,:,None].expand(-1,-1,20))),1)
            with torch.no_grad():values=teacher.diagnostic_fitness(points)
            target=min(float(state['fitness'][0,0]),float(values[0,2:].min()))
            best,index=values[0,:2].min(0);index=int(index)
            replace=bool(float(best)<target-1e-12)
            if replace:assert index not in chosen[0].tolist()
            action=dict(step=step,fingerprint=state['fingerprint'],replace=replace,index=index)
            model=make(seed);instrument(model,(step,),action)
            branch=rollout(model,fid,pop_id)
            assert torch.equal(branch['points'][:,:2*step],base['points'][:,:2*step])
            assert torch.equal(branch['trail'][:,:step],base['trail'][:,:step])
            if not replace:
                assert torch.equal(branch['points'],base['points']) and torch.equal(branch['trail'],base['trail'])
            else:
                assert torch.equal(branch['points'][:,2*step],state['candidates'][:,int(chosen[0,0])])
                assert torch.equal(branch['points'][:,2*step+1],state['candidates'][:,index])
                assert branch['trail'][0,step]<base['trail'][0,step]
            def advantage(at):
                return float((bounded(init,branch['trail'][:,at],scale)-bounded(init,base['trail'][:,at],scale))[0])
            final_delta=float(base['trail'][0,-1]-branch['trail'][0,-1])
            rows.append(dict(seed=seed,fid=fid,population=pop_id,nfe=100+2*step,opportunity=replace,
                immediate=advantage(step),after20=advantage(min(step+10,249)),after60=advantage(min(step+30,249)),
                final=advantage(249),final_win=final_delta>1e-12,final_loss=final_delta< -1e-12,
                final_tie=abs(final_delta)<=1e-12))
            branches[step]=branch;labels[step]=dict(values=values,action=action)
        assert teacher.diagnostic_points==12
        result=dict(rows=rows,base=base,branches=branches,snapshots=snap,labels=labels,
            main_calls=2400,teacher_calls=12,seconds=time.perf_counter()-start)
        store.save(key,case,result)
        return rows

def summary(g):
    metrics=['immediate','after20','after60','final']
    cluster=g.assign(recipe=g.fid//3).groupby('recipe')[metrics].mean().reindex(range(12))
    rng=np.random.default_rng(20261001);indices=rng.integers(0,12,(5000,12))
    result={}
    for m in metrics:
        a=cluster[m].to_numpy();lo,hi=np.quantile(a[indices].mean(1),[.025,.975])
        result[m]=dict(mean=float(a.mean()),lower=float(lo),upper=float(hi))
    op=g[g.opportunity]
    result.update(states=len(g),opportunities=len(op),conditional_final_mean=float(op.final.mean()) if len(op) else None,
        conditional_immediate_mean=float(op.immediate.mean()) if len(op) else None,
        opportunity_final_wins=int(op.final_win.sum()),opportunity_final_losses=int(op.final_loss.sum()),
        opportunity_final_ties=int(op.final_tie.sum()),seed_final_means=g.groupby('seed').final.mean().to_dict())
    return result

def main():
    RUN.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
    if not (RUN/'identity.json').exists():
        checks=contracts()
        identity=dict(version='branch_continuation_v1',sources={p:sha256(ROOT/p) for p in SOURCES},
            proposals=[sha256(OLD/f'model_{s}/selected.pt') for s in range(3)],parent=sha256(OLD/'identity.json'))
        write_json(RUN/'identity.json',identity);write_json(RUN/'CONTRACTS.json',checks)
        for name in SOURCES:
            target=RUN/'source_snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,target)
    verify();write_json(RUN/'STATUS.json',dict(status='running',expected=432))
    outputs=parent.parallel(worker,[(s,f,p) for s in range(3) for f in range(36) for p in range(4)],12)
    df=pd.DataFrame([row for _,rows in outputs for row in rows]);assert len(df)==1296
    df.to_csv(OUT/'pairs.csv',index=False)
    aggregate=summary(df);stages={str(n):summary(g) for n,g in df.groupby('nfe')}
    e=aggregate['final']
    passed=e['mean']>=.001 and e['lower']>0 and all(v>0 for v in aggregate['seed_final_means'].values())
    route='prioritize_selector' if passed else ('below_practical_threshold_for_single_intervention' if e['upper']<.001 else 'inconclusive')
    result=dict(all_workers_joined=True,aggregate=aggregate,stages=stages,direction=route,
        base_trajectories=432,branch_trajectories=1296,replay_calls=1036800,redundant_prefix_calls=432000,
        teacher_calls=5184,contract=json.loads((RUN/'CONTRACTS.json').read_text()),oracle_assisted=True)
    verify();write_json(OUT/'RESULTS.json',result);write_json(RUN/'COMPLETE.json',result)
    write_json(RUN/'STATUS.json',dict(status='complete'));print(json.dumps(result),flush=True)

if __name__=='__main__':main()
