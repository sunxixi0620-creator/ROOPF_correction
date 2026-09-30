"""Antithetic parameter-space training on complete closed-loop terminal reward."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import sys,json,time,shutil
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import numpy as np
import torch
from torch.nn.utils import parameters_to_vector,vector_to_parameters
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import complementary_proposal_study as parent
from roopf.complementary_proposal import ComplementaryProposal,build_context
from roopf.online_portfolio import build_online
from roopf.anchor_backbone import AnchorPolicyBackbone
from roopf.revision_tasks import ProceduralTask,population
from roopf.experiment_io import CaseStore,sha256,write_json,save_torch,seed_for,fingerprint
from scripts.unified_revision import bounded
RUN=ROOT/'results/terminal_training_20260930';OUT=ROOT/'docs/revision/terminal_training'
OLD=ROOT/'results/complementary_proposal_20260930'
SOURCES=['scripts/terminal_training.py','docs/experiments/TERMINAL_TRAINING_PROTOCOL.md']


def proposal(seed):
    p=ComplementaryProposal(parent.ANCHORS[seed]).eval().requires_grad_(False)
    p.load_state_dict(torch.load(OLD/f'model_{seed}/selected.pt',weights_only=False));return p

def noise(seed,update,direction,n):
    return torch.randn(n,generator=torch.Generator().manual_seed(seed_for('terminal_es',seed,update,direction)))

def verify():
    ident=json.loads((RUN/'identity.json').read_text())
    for p,h in ident['sources'].items():assert sha256(ROOT/p)==h
    for s,h in enumerate(ident['proposals']):assert sha256(OLD/f'model_{s}/selected.pt')==h
    parent.verify(OLD);return ident

def job(role,method,seed,fid,instance=0,center=None,update=0,direction=-1,sign=0):
    return dict(role=role,method=method,seed=seed,fid=fid,instance=instance,center=center,
        center_hash=sha256(RUN/center) if center else None,update=update,direction=direction,sign=sign)

def evaluate(spec):
    torch.set_num_threads(1);ident=verify();store=CaseStore(RUN/'cases',ident)
    key=fingerprint(spec)[:32]
    with store.lock(key):
        value=store.load(key,spec)
        if value is not None:return value
        start=time.perf_counter();m=spec['method'];s=spec['seed']
        if m=='O':model=build_online(20,600)
        elif m=='A':
            model=AnchorPolicyBackbone(20,200,100).eval().requires_grad_(False)
            model.load_state_dict(torch.load(parent.ANCHORS[s],weights_only=False));model.MaxNFE=600
        else:
            p=proposal(s)
            if spec['center']:
                assert sha256(RUN/spec['center'])==spec['center_hash']
                coords=torch.load(RUN/spec['center'],weights_only=False)
                origin=torch.load(RUN/f'origin_{s}.pt',weights_only=False)
                offset=coords
                if spec['sign']:
                    offset=offset+spec['sign']*.02*noise(s,spec['update'],spec['direction'],len(coords))
                vector_to_parameters(origin['theta']+origin['scale']*offset,p.parameters())
            model=build_context(proposal=p).eval().requires_grad_(False)
        split='terminal_v1_'+spec['role'];count=2 if spec['role'] in ('train','timing') else 4
        f,i=spec['fid'],spec['instance'];task=ProceduralTask(f,i,split,dim=20)
        pop=population(f,i,split,count=count,dim=20)
        torch.manual_seed(seed_for('terminal_v1',spec['role'],f,i,'policy'))
        before=fingerprint(model.state_dict())
        with torch.no_grad():_,trail,nfe,points=model(pop,task)
        assert nfe==600 and task.points==600*count and task.diagnostic_points==0
        assert before==fingerprint(model.state_dict()) and torch.isfinite(trail).all()
        value=dict(utility=bounded(task.initial_values.min(1).values,trail[:,-1],task.initial_values.std(1)),
            trail=trail,initial=task.initial_values,points_sha256=fingerprint(points),main_calls=task.points,
            teacher_calls=0,seconds=time.perf_counter()-start)
        if spec['role']=='confirmation':value['points']=points
        store.save(key,spec,value);return value

def execute(pool,jobs):return list(pool.map(evaluate,jobs))

def interval(a,level=.95):
    # seeds x36 x observations; recipe clusters preserve all scales/seeds.
    x=a.reshape(3,12,3,-1).mean((0,2,3))
    rng=np.random.default_rng(20261001);d=x[rng.integers(0,12,(5000,12))].mean(1)
    lo,hi=np.quantile(d,[(1-level)/2,1-(1-level)/2])
    return dict(mean=float(a.mean()),lower=float(lo),upper=float(hi),
        seed_means=a.mean(tuple(range(1,a.ndim))).tolist())

def validation(pool,centers):
    o=execute(pool,[job('validation','O',-1,f) for f in range(36)])
    b=np.stack([v['utility'].numpy() for v in o])
    a=[]
    for s in range(3):
        values=execute(pool,[job('validation','F',s,f,center=centers[s]) for f in range(36)])
        a.append(np.stack([v['utility'].numpy() for v in values])-b)
    return np.array(a)

def prepare():
    RUN.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
    if (RUN/'identity.json').exists():verify();return
    parent.verify(OLD)
    ident=dict(version='terminal_training_v1',sources={p:sha256(ROOT/p) for p in SOURCES},
        proposals=[sha256(OLD/f'model_{s}/selected.pt') for s in range(3)],sigma=.02,lr=.01,directions=8,max_updates=30,workers=16)
    write_json(RUN/'identity.json',ident)
    for s in range(3):
        p=proposal(s);theta=parameters_to_vector(p.parameters()).detach().clone()
        scales=torch.cat([torch.full_like(w.flatten(),max(float(w.square().mean().sqrt()),.01)) for w in p.parameters()])
        assert len(theta)==58504
        save_torch(RUN/f'origin_{s}.pt',dict(theta=theta,scale=scales))
        save_torch(RUN/f'center_s{s}_u0.pt',torch.zeros_like(theta))
        e=noise(s,1,0,len(theta));assert torch.equal(e,noise(s,1,0,len(theta)))
        assert torch.allclose((theta+scales*.02*e)+(theta-scales*.02*e),2*theta,atol=1e-6,rtol=1e-6)
    for name in SOURCES:
        target=RUN/'source_snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,target)

def main():
    torch.set_num_threads(1);prepare();started=time.perf_counter()
    with ProcessPoolExecutor(max_workers=16,mp_context=mp.get_context('spawn')) as pool:
        # Timing and center-zero parity use a separate task role.
        write_json(RUN/'STATUS.json',dict(status='timing_and_contracts'))
        timing_jobs=[job('timing','F',s,f,center=f'center_s{s}_u0.pt') for s in range(3) for f in (0,9,18,27)]
        t=time.perf_counter();timed=execute(pool,timing_jobs);elapsed=time.perf_counter()-t
        references=execute(pool,[job('timing','F',s,f) for s in range(3) for f in (0,9,18,27)])
        for a,b in zip(timed,references):
            assert torch.equal(a['trail'],b['trail']) and a['points_sha256']==b['points_sha256']
        write_json(RUN/'TIMING.json',dict(first_wave_seconds=elapsed,mean_worker_seconds=np.mean([x['seconds'] for x in timed]),
            upper_training_seconds=30*294*np.mean([x['seconds'] for x in timed])/16*1.4,
            timing_contract_calls=28800,parity_passed=True))
        phis=[torch.nn.Parameter(torch.zeros(58504)) for _ in range(3)]
        optimizers=[torch.optim.Adam([p],lr=.01) for p in phis]
        store=CaseStore(RUN/'state',verify())
        saved=store.load('training',{'kind':'terminal_es'})
        if saved:
            for s in range(3):
                phis[s].data.copy_(saved['phis'][s]);optimizers[s].load_state_dict(saved['optimizers'][s])
            start_update=saved['update']+1;selected=saved['selected'];best=saved['best'];history=saved['history'];stale=saved['stale']
            initial=saved['initial']
        else:
            write_json(RUN/'STATUS.json',dict(status='initial_validation'))
            initial=validation(pool,[f'center_s{s}_u0.pt' for s in range(3)])
            save_torch(RUN/'initial_validation.pt',initial)
            best=initial.mean((1,2)).tolist();selected=[0,0,0];history=[];stale=0;start_update=1
        order=np.random.default_rng(20261001).permutation(36)
        for update in range(start_update,31):
            t=time.perf_counter();fids=order[((update-1)%6)*6:((update-1)%6+1)*6].tolist()
            write_json(RUN/'STATUS.json',dict(status='training',update=update,max_updates=30,selected=selected))
            ojobs=[job('train','O',-1,f,update) for f in fids]
            o=execute(pool,ojobs);obase=np.array([float(v['utility'].mean()) for v in o])
            jobs=[job('train','F',s,f,update,f'center_s{s}_u{update-1}.pt',update,d,sign)
                  for s in range(3) for d in range(8) for sign in (1,-1) for f in fids]
            values=execute(pool,jobs)
            rewards=np.array([float(v['utility'].mean()) for v in values]).reshape(3,8,2,6)-obase[None,None,None,:]
            signals=[]
            for s in range(3):
                differences=(rewards[s,:,0]-rewards[s,:,1]).mean(1)
                grad=sum(float(differences[d])*noise(s,update,d,58504) for d in range(8))/(16*.02)
                assert torch.isfinite(grad).all()
                norm=float(grad.norm());signals.append(dict(norm=norm,direction_differences=differences.tolist(),reward=float(rewards[s].mean())))
                phis[s].grad=-grad;torch.nn.utils.clip_grad_norm_([phis[s]],1)
                optimizers[s].step();optimizers[s].zero_grad(set_to_none=True)
                save_torch(RUN/f'center_s{s}_u{update}.pt',phis[s].detach())
            entry=dict(update=update,signals=signals,seconds=time.perf_counter()-t)
            if update%5==0:
                write_json(RUN/'STATUS.json',dict(status='validation',update=update,selected=selected))
                v=validation(pool,[f'center_s{s}_u{update}.pt' for s in range(3)])
                save_torch(RUN/f'validation_u{update}.pt',v);entry['validation']=v.mean((1,2)).tolist()
                changed=False
                for s in range(3):
                    if entry['validation'][s]>best[s]+1e-5:best[s]=entry['validation'][s];selected[s]=update;changed=True
                stale=0 if changed else stale+1
            history.append(entry);write_json(RUN/'history.json',history)
            with store.lock('training'):
                store.save('training',{'kind':'terminal_es'},dict(phis=[p.detach() for p in phis],optimizers=[o.state_dict() for o in optimizers],
                    update=update,selected=selected,best=best,history=history,initial=initial,stale=stale))
            write_json(RUN/'PROGRESS.json',dict(update=update,selected_updates=selected,best_validation=best,
                last_update_seconds=entry['seconds'],elapsed_seconds=time.perf_counter()-started))
            if all(v['norm']==0 for v in signals):break
            if update>=10 and stale>=2:break
        centers=[f'center_s{s}_u{selected[s]}.pt' for s in range(3)]
        chosen=validation(pool,centers)
        effects=dict(vs_O=interval(chosen),vs_initial=interval(chosen-initial))
        gate=all(effects[k]['mean']>=threshold and effects[k]['lower']>0 and min(effects[k]['seed_means'])>0
                 for k,threshold in [('vs_O',.005),('vs_initial',.001)])
        frozen={}
        for s in range(3):
            p=proposal(s);origin=torch.load(RUN/f'origin_{s}.pt',weights_only=False)
            phi=torch.load(RUN/centers[s],weights_only=False);vector_to_parameters(origin['theta']+origin['scale']*phi,p.parameters())
            save_torch(RUN/f'selected_{s}.pt',p.state_dict());frozen[str(s)]=sha256(RUN/f'selected_{s}.pt')
        write_json(RUN/'MODELS_FROZEN.json',dict(selected_updates=selected,hashes=frozen,centers=centers))
        write_json(RUN/'VALIDATION_GATE.json',dict(passed=gate,effects=effects))
        confirmation=None
        if gate:
            write_json(RUN/'STATUS.json',dict(status='confirmation'))
            jobs=[job('confirmation',m,s,f,i,centers[s] if m=='Terminal' else None)
                  for m in ('O','A','Initial','Terminal') for s in ([-1] if m=='O' else range(3)) for f in range(36) for i in range(2)]
            values=execute(pool,jobs);mat={m:np.zeros((3,36,2,4)) for m in ('O','A','Initial','Terminal')}
            for spec,v in zip(jobs,values):
                m,s,f,i=(spec[k] for k in ('method','seed','fid','instance'))
                if m=='O':mat[m][:,f,i]=v['utility'].numpy()
                else:mat[m][s,f,i]=v['utility'].numpy()
            confirmation={m:interval(mat['Terminal']-mat[m],1-.05/3) for m in ('O','A','Initial')}
            for e in confirmation.values():e['passed']=e['mean']>=.005 and e['lower']>0 and min(e['seed_means'])>0
            save_torch(RUN/'confirmation_utilities.pt',mat)
    verify();result=dict(all_workers_joined=True,updates=update,selected_updates=selected,validation_gate=gate,
        validation=effects,confirmation=confirmation,elapsed_seconds=time.perf_counter()-started)
    write_json(RUN/'COMPLETE.json',result);write_json(OUT/'RESULTS.json',result)
    write_json(RUN/'STATUS.json',dict(status='complete'))
    # Independently count committed calls, including timing/validation; no teachers.
    costs={}
    for meta in (RUN/'cases').glob('*.json'):
        if meta.name=='identity.json':continue
        d=json.loads(meta.read_text());assert sha256(meta.with_suffix('.pt'))==d['data_sha256']
        v=torch.load(meta.with_suffix('.pt'),weights_only=False);role=d['case']['role']
        costs[role]=costs.get(role,0)+v['main_calls'];assert v['teacher_calls']==0
    write_json(OUT/'COSTS.json',costs)
    (OUT/'CONCLUSIONS.zh-CN.md').write_text('# 终局收益训练可行性\n\n'+json.dumps(result,ensure_ascii=False,indent=2)+'\n\n共享生成函数族，属于开发证据；门槛未通过则停止，不追加变体。\n')
    print(json.dumps(result),flush=True)

if __name__=='__main__':main()
