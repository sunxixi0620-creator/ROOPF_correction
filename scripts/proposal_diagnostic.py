"""Post-replay teachers, frozen models, paired same-state candidate diagnostics."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import json
from pathlib import Path
import sys
import types
import time
import zipfile
import numpy as np
import pandas as pd
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import complementary_proposal_study as prev
from roopf.complementary_proposal import ComplementaryProposal,build_context
from roopf.online_portfolio import OnlinePortfolio
from roopf.anchor_backbone import AnchorPolicyBackbone
from roopf.experiment_io import CaseStore,sha256,write_json
from scripts.unified_revision import bounded
OLD=ROOT/'results/complementary_proposal_20260930'
RUN=ROOT/'results/proposal_diagnostic_20260930'
OUT=ROOT/'docs/revision/proposal_diagnostic'
STEPS=(0,35,70,105,140,175,210,245)

def worker(job):
    method,seed,fid,instance=job
    torch.set_num_threads(1)
    identity=json.loads((RUN/'identity.json').read_text())
    for name,digest in identity['sources'].items():assert sha256(ROOT/name)==digest
    _,prior=prev.verify(OLD)
    frozen=json.loads((OLD/'MODELS_FROZEN.json').read_text())
    for s,h in enumerate(frozen['selected']):assert sha256(OLD/f'model_{s}/selected.pt')==h
    oldstore=CaseStore(OLD/'search',dict(identity=prior,frozen=frozen))
    case=dict(method=method,seed=seed,fid=fid,instance=instance)
    key=f'{method}_s{seed}_f{fid}_i{instance}'
    reference=oldstore.load(key,case)
    assert reference is not None
    store=CaseStore(RUN/'cases',identity)
    with store.lock(key):
        cached=store.load(key,case)
        if cached is not None:return cached
        proposals={}
        for s in (range(3) if method=='O' else [seed]):
            p=ComplementaryProposal(prev.ANCHORS[s]).eval().requires_grad_(False)
            p.load_state_dict(torch.load(OLD/f'model_{s}/selected.pt',weights_only=False))
            a=AnchorPolicyBackbone(20,200,100).eval().requires_grad_(False)
            a.load_state_dict(torch.load(prev.ANCHORS[s],weights_only=False))
            proposals[s]=(p,a)
        model=build_context(proposal=None if method=='O' else proposals[seed][0]).eval().requires_grad_(False)
        snapshots=[]
        original=model.scores
        def capture(self,candidates,prior,ax,ay,problem,remaining,stagnation):
            step=(self.evalnum-100)//2
            if step in STEPS:
                snapshots.append(dict(step=step,candidates=candidates.clone(),prior=prior.clone(),
                    ax=ax.clone(),ay=ay.clone(),x=self.current_x.clone(),fitness=self.current_fitness.clone(),
                    remaining=remaining,stagnation=stagnation.clone()))
            return original(candidates,prior,ax,ay,problem,remaining,stagnation)
        model.scores=types.MethodType(capture,model)
        task,pop,policy=prev.setup('search',fid,instance)
        torch.manual_seed(policy)
        start=time.perf_counter()
        with torch.no_grad():_,trail,nfe,points=model(pop,task)
        assert torch.equal(points,reference['points']) and torch.equal(trail,reference['trail'])
        assert task.points==2400 and task.diagnostic_points==0 and nfe==600 and len(snapshots)==8
        teacher,_,_=prev.setup('search',fid,instance)
        rows=[]
        with torch.no_grad():
            for snap in snapshots:
                c,prior,ax,ay=(snap[k] for k in ('candidates','prior','ax','ay'))
                x,fit,rem,stag=(snap[k] for k in ('x','fitness','remaining','stagnation'))
                scores=OnlinePortfolio.scores(model,c,prior,ax,ay,task,rem,stag)
                idx=scores.argsort(dim=1,stable=True)[:,:2]
                intent=c.gather(1,idx[:,:,None].expand(-1,-1,20))
                isc=scores.gather(1,idx)
                for s,(p,a) in proposals.items():
                    new=p(x,fit,intent,isc,rem,stag,task)
                    old=a.generator(x,task,fit)
                    # Exactly42 teacher calls/state; no influence on replay.
                    vals=teacher.diagnostic_fitness(torch.cat((c,new,old),1))
                    vo,vn,va=vals[:,:38],vals[:,38:40],vals[:,40:42]
                    nc=torch.cat((new,c[:,2:]),1);ac=torch.cat((old,c[:,2:]),1)
                    ns=OnlinePortfolio.scores(model,nc,prior,ax,ay,task,rem,stag)
                    aas=OnlinePortfolio.scores(model,ac,prior,ax,ay,task,rem,stag)
                    ni=ns.argsort(dim=1,stable=True)[:,:2];ai=aas.argsort(dim=1,stable=True)[:,:2]
                    nvals=torch.cat((vn,vo[:,2:]),1);avals=torch.cat((va,vo[:,2:]),1)
                    incumbent=fit[:,0];scale=fit.std(1).clamp_min(1e-8)
                    target=torch.minimum(incumbent,vo.gather(1,idx).min(1).values)
                    nsel=nvals.gather(1,ni).min(1).values
                    asel=avals.gather(1,ai).min(1).values
                    nbest=vn.min(1).values;abest=va.min(1).values
                    oracle_o=torch.minimum(incumbent,vo.min(1).values)
                    oracle_n=torch.minimum(incumbent,nvals.min(1).values)
                    init_scale=task.initial_values.std(1)
                    base=p.backbone.generator(x,task,fit)
                    features=torch.cat(((intent-x[:,:1]).flatten(1)/10,isc.clamp(-10,10),
                        x.new_full((4,1),rem),stag[:,None]),1)
                    correction=.5*torch.tanh(p.context(features)).reshape(4,2,20)
                    metrics=dict(potential_new=bounded(target,nbest,scale),potential_old=bounded(target,abest,scale),
                        chosen_new=bounded(incumbent,nsel,init_scale),chosen_O=bounded(incumbent,target,init_scale),
                        chosen_old=bounded(incumbent,asel,init_scale),
                        oracle_increment=bounded(oracle_o,oracle_n,init_scale),
                        oracle_gap=bounded(torch.minimum(incumbent,nsel),oracle_n,init_scale),
                        opportunity=nbest<target-1e-12,realized=nsel<target-1e-12,
                        fullpool_opportunity=nbest<oracle_o-1e-12,
                        selected=(ni<2).sum(1)/2,
                        pair_distance=(new[:,0]-new[:,1]).norm(dim=1),
                        nearest_shared=torch.cdist(new,c[:,2:]).amin(2).mean(1),
                        saturation=(correction.abs()>.49).float().mean((1,2)),
                        clipped=((base+correction).abs()>5).float().mean((1,2)))
                    for b in range(4):rows.append(dict(behavior=method,seed=s,fid=fid,instance=instance,
                        population=b,step=snap['step'],**{k:float(v[b]) for k,v in metrics.items()}))
        assert task.diagnostic_points==0 and task.points==2400
        expected=32*42*len(proposals)
        assert teacher.diagnostic_points==expected
        result=dict(rows=rows,replay_calls=task.points,teacher_calls=expected,exact_parity=True,
            seconds=time.perf_counter()-start)
        store.save(key,case,result)
        return result

def main():
    RUN.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
    sources=['scripts/proposal_diagnostic.py','docs/experiments/PROPOSAL_DIAGNOSTIC_PROTOCOL.md']
    identity=dict(version='proposal_diagnostic_v1',sources={p:sha256(ROOT/p) for p in sources},
        parent_identity=sha256(OLD/'identity.json'),models=sha256(OLD/'MODELS_FROZEN.json'))
    if (RUN/'identity.json').exists():assert json.loads((RUN/'identity.json').read_text())==identity
    else:write_json(RUN/'identity.json',identity)
    write_json(RUN/'STATUS.json',dict(status='replaying_and_diagnosing',expected=288))
    jobs=[(m,s,f,i) for m in ('O','New') for s in ([-1] if m=='O' else range(3))
          for f in range(36) for i in range(2)]
    results=prev.parallel(worker,jobs,12)
    rows=[r for _,v in results for r in v['rows']]
    df=pd.DataFrame(rows);assert len(df)==13824
    df['potential_advantage']=df.potential_new-df.potential_old
    df['chosen_advantage']=df.chosen_new-df.chosen_O
    df['chosen_advantage_old']=df.chosen_new-df.chosen_old
    df['recipe']=df.fid//3
    df.to_csv(OUT/'states.csv',index=False)
    metrics=['potential_new','potential_old','potential_advantage','chosen_advantage','chosen_advantage_old',
             'oracle_increment','oracle_gap','opportunity','fullpool_opportunity','selected',
             'pair_distance','nearest_shared','saturation','clipped']
    report={}
    rng=np.random.default_rng(20261001)
    indices=rng.integers(0,12,(5000,12))
    for behavior,g in df.groupby('behavior'):
        means=g.groupby('recipe')[metrics].mean().reindex(range(12))
        report[behavior]={}
        for metric in metrics:
            a=means[metric].to_numpy();lo,hi=np.quantile(a[indices].mean(1),[.025,.975])
            report[behavior][metric]=dict(mean=float(a.mean()),lower=float(lo),upper=float(hi))
        opportunity=g[g.opportunity>0]
        report[behavior]['opportunity_count']=len(opportunity)
        report[behavior]['realized_fraction_given_opportunity']=float(opportunity.realized.mean())
    df.groupby(['behavior','step'])[metrics+['realized']].mean().to_csv(OUT/'by_step.csv')
    df.groupby(['behavior','seed'])[metrics+['realized']].mean().to_csv(OUT/'by_seed.csv')
    complete=dict(all_workers_joined=True,exact_replay_batches=288,state_records=len(df),
        replay_calls=sum(v['replay_calls'] for _,v in results),teacher_calls=sum(v['teacher_calls'] for _,v in results),
        models_frozen=True,results=report)
    assert complete['replay_calls']==691200 and complete['teacher_calls']==580608
    write_json(OUT/'RESULTS.json',complete);write_json(RUN/'COMPLETE.json',complete)
    write_json(RUN/'STATUS.json',dict(status='complete'))
    print(json.dumps(complete),flush=True)

if __name__=='__main__':main()
