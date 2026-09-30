"""Single calibrated-selector experiment with immutable contextual proposals."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
from pathlib import Path
import sys,json,time,shutil
import numpy as np
import pandas as pd
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import complementary_proposal_study as parent
from roopf.selection_calibration import ScoreCalibration,build_calibrated
from roopf.complementary_proposal import ComplementaryProposal,build_context
from roopf.online_portfolio import build_online
from roopf.revision_tasks import ProceduralTask,population
from roopf.experiment_io import CaseStore,sha256,write_json,save_torch,seed_for,fingerprint
from scripts.unified_revision import bounded
OLD=ROOT/'results/complementary_proposal_20260930'
RUN=ROOT/'results/selection_calibration_20260930'
OUT=ROOT/'docs/revision/selection_calibration'
SOURCES=['scripts/selection_calibration_study.py','roopf/selection_calibration.py','docs/experiments/SELECTION_CALIBRATION_PROTOCOL.md']

def proposal(seed):
    p=ComplementaryProposal(parent.ANCHORS[seed]).eval().requires_grad_(False)
    p.load_state_dict(torch.load(OLD/f'model_{seed}/selected.pt',weights_only=False))
    return p

def setup(role,fid,instance,count=4):
    split='selection_v1_'+role
    return (ProceduralTask(fid,instance,split,dim=20),population(fid,instance,split,count=count,dim=20),
            seed_for('selection_v1',role,fid,instance,'policy'))

def verify():
    identity=json.loads((RUN/'identity.json').read_text())
    for p,h in identity['sources'].items():assert sha256(ROOT/p)==h,p
    for s,h in enumerate(identity['proposals']):assert sha256(OLD/f'model_{s}/selected.pt')==h
    parent.verify(OLD)
    return identity

def contracts():
    torch.set_num_threads(1);calls=0
    for budget in (113,600):
        outputs=[]
        for mode in ('base','capture','zero'):
            p=proposal(0)
            model=(build_context(proposal=p,budget=budget).eval().requires_grad_(False) if mode=='base' else
                   build_calibrated(p,ScoreCalibration() if mode=='zero' else None,budget,mode=='capture'))
            task,pop,seed=setup('contract',4,0,2)
            before=fingerprint(p.state_dict())
            torch.manual_seed(seed)
            with torch.no_grad():_,trail,nfe,points=model(pop,task)
            assert nfe==budget and task.points==2*budget and task.diagnostic_points==0
            assert fingerprint(p.state_dict())==before
            outputs.append((trail,points,torch.get_rng_state()));calls+=task.points
        assert all(torch.equal(a,b) and torch.equal(a,c) for a,b,c in zip(*outputs))
    c=ScoreCalibration();f=torch.randn(2,38,10)
    c(f).square().sum().backward()
    assert sum(p.numel() for p in c.parameters())==769
    return dict(passed=True,exact_parity=True,odd_budget=True,main_calls=calls,teacher_calls=0)

def collect(job):
    seed,role,fid,instance=job
    torch.set_num_threads(1);identity=verify();store=CaseStore(RUN/'data',identity)
    case=dict(seed=seed,role=role,fid=fid,instance=instance);key=f's{seed}_{role}_f{fid}_i{instance}'
    with store.lock(key):
        if store.load(key,case) is not None:return key
        p=proposal(seed);before=fingerprint(p.state_dict())
        model=build_calibrated(p,capture=True)
        task,pop,ps=setup(role,fid,instance)
        torch.manual_seed(ps)
        with torch.no_grad():_,trail,nfe,points=model(pop,task)
        assert nfe==600 and task.points==2400 and task.diagnostic_points==0 and len(model.records)==8
        assert fingerprint(p.state_dict())==before
        data={k:torch.cat([r[k] for r in model.records]) for k in model.records[0]}
        teacher,_,_=setup(role,fid,instance)
        with torch.no_grad():data['values']=teacher.diagnostic_fitness(data['candidates'])
        assert teacher.diagnostic_points==1216
        store.save(key,case,dict(data=data,main_calls=2400,teacher_calls=1216,trail=trail,points=points))
    return key

def dataset(seed,role,device):
    store=CaseStore(RUN/'data',verify());parts=[]
    for fid in range(36):
        for i in range(2 if role=='train' else 1):
            v=store.load(f's{seed}_{role}_f{fid}_i{i}',dict(seed=seed,role=role,fid=fid,instance=i))
            assert v is not None;parts.append(v['data'])
    return {k:torch.cat([v[k] for v in parts]).to(device) for k in ('features','scores','values','incumbent','scale')}

@torch.no_grad()
def validate(model,data):
    scores=data['scores']+model(data['features'])
    indices=torch.argsort(scores,dim=1,stable=True)[:,:2]
    values=data['values'].gather(1,indices).min(1).values
    return float(bounded(data['incumbent'],values,data['scale']).mean())

def train(seed):
    torch.set_num_threads(1);verify();out=RUN/f'calibrator_{seed}';out.mkdir(exist_ok=True)
    if (out/'COMPLETE.json').exists():
        v=json.loads((out/'COMPLETE.json').read_text());assert sha256(out/'selected.pt')==v['sha256'];return v
    device='cuda' if torch.cuda.is_available() else 'cpu'
    train_data=dataset(seed,'train',device);val_data=dataset(seed,'validation',device)
    torch.manual_seed(seed_for('calibration_train',seed))
    model=ScoreCalibration().to(device);optimizer=torch.optim.Adam(model.parameters(),lr=.001)
    z=((train_data['values']-train_data['incumbent'][:,None])/train_data['scale'][:,None]).clamp(-5,5)
    targets=torch.softmax(-z/.2,dim=1)
    best=validate(model,val_data);initial=best;selected=0;stale=0;history=[]
    save_torch(out/'selected.pt',{k:v.detach().cpu() for k,v in model.state_dict().items()})
    start=time.perf_counter()
    for epoch in range(1,81):
        model.train();order=torch.randperm(len(z),device=device);loss_sum=0
        for ids in order.split(128):
            optimizer.zero_grad(set_to_none=True)
            score=train_data['scores'][ids]+model(train_data['features'][ids])
            loss=-(targets[ids]*torch.log_softmax(-score/.2,dim=1)).sum(1).mean()
            assert torch.isfinite(loss);loss.backward()
            assert all(torch.isfinite(p.grad).all() for p in model.parameters())
            optimizer.step();loss_sum+=float(loss.detach())/18
        score=None
        if epoch%2==0:
            model.eval();score=validate(model,val_data)
            if score>best+1e-5:
                best,selected,stale=score,epoch,0
                save_torch(out/'selected.pt',{k:v.detach().cpu() for k,v in model.state_dict().items()})
            else:stale+=1
        history.append(dict(epoch=epoch,loss=loss_sum,validation=score))
        write_json(out/'history.json',history)
        write_json(out/'PROGRESS.json',dict(epoch=epoch,selected=selected,validation=best))
        if stale>=8:break
    save_torch(out/'last.pt',dict(model=model.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch))
    result=dict(seed=seed,epoch=epoch,selected_epoch=selected,initial_validation=initial,
        selected_validation=best,sha256=sha256(out/'selected.pt'),seconds=time.perf_counter()-start,
        updates=18*epoch,parameters=769,device=device)
    write_json(out/'COMPLETE.json',result);return result

def evaluate(job):
    method,seed,fid,instance=job
    torch.set_num_threads(1);identity=verify()
    frozen=json.loads((RUN/'MODELS_FROZEN.json').read_text())
    for s,h in enumerate(frozen['calibrators']):assert sha256(RUN/f'calibrator_{s}/selected.pt')==h
    store=CaseStore(RUN/'search',dict(identity=identity,frozen=frozen))
    case=dict(method=method,seed=seed,fid=fid,instance=instance);key=f'{method}_s{seed}_f{fid}_i{instance}'
    with store.lock(key):
        cached=store.load(key,case)
        if cached is not None:return cached
        start=time.perf_counter()
        if method=='O':model=build_online(20,600)
        elif method=='Base':model=build_context(proposal=proposal(seed)).eval().requires_grad_(False)
        else:
            c=ScoreCalibration();c.load_state_dict(torch.load(RUN/f'calibrator_{seed}/selected.pt',weights_only=False))
            model=build_calibrated(proposal(seed),c)
        before=fingerprint(model.state_dict())
        task,pop,ps=setup('search',fid,instance);torch.manual_seed(ps)
        with torch.no_grad():_,trail,nfe,points=model(pop,task)
        assert nfe==600 and task.points==2400 and task.diagnostic_points==0
        assert fingerprint(model.state_dict())==before and torch.isfinite(trail).all()
        assert points.shape==(4,500,20) and (trail[:,1:]<=trail[:,:-1]).all()
        selected=sum(sum(i<2 for i in r['selected']) for r in model.decisions)
        result=dict(utility=bounded(task.initial_values.min(1).values,trail[:,-1],task.initial_values.std(1)),
            trail=trail,points=points,main_calls=2400,teacher_calls=0,selected_extra=selected,
            seconds=time.perf_counter()-start,initial=task.initial_values)
        store.save(key,case,result);return result

def main():
    RUN.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
    if not (RUN/'identity.json').exists():
        checks=contracts()
        identity=dict(version='selection_calibration_v1',sources={p:sha256(ROOT/p) for p in SOURCES},
            proposals=[sha256(OLD/f'model_{s}/selected.pt') for s in range(3)],parent=sha256(OLD/'identity.json'))
        write_json(RUN/'identity.json',identity);write_json(RUN/'CONTRACTS.json',checks)
        for name in SOURCES:
            dest=RUN/'source_snapshot'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,dest)
    verify()
    write_json(RUN/'STATUS.json',dict(status='collecting',batches=324))
    parent.parallel(collect,[(s,r,f,i) for s in range(3) for r in ('train','validation')
        for f in range(36) for i in range(2 if r=='train' else 1)],12)
    write_json(RUN/'STATUS.json',dict(status='training'))
    trained=parent.parallel(train,list(range(3)),3)
    write_json(RUN/'MODELS_FROZEN.json',dict(calibrators=[sha256(RUN/f'calibrator_{s}/selected.pt') for s in range(3)]))
    write_json(RUN/'STATUS.json',dict(status='paid_search'))
    jobs=[(m,s,f,i) for m in ('O','Base','Cal') for s in ([-1] if m=='O' else range(3)) for f in range(36) for i in range(2)]
    results=parent.parallel(evaluate,jobs,12)
    matrix={m:np.zeros((3,36,2,4)) for m in ('O','Base','Cal')};rows=[]
    for job,v in results:
        m,s,f,i=job
        if m=='O':matrix[m][:,f,i]=v['utility'].numpy()
        else:matrix[m][s,f,i]=v['utility'].numpy()
        rows.append(dict(method=m,seed=s,fid=f,instance=i,seconds=v['seconds'],selected_extra=v['selected_extra']))
    pd.DataFrame(rows).to_csv(OUT/'runtime_selection.csv',index=False)
    effects={m:parent.interval(matrix['Cal']-matrix[m],.975,.005) for m in ('Base','O')}
    save_torch(RUN/'utilities.pt',matrix)
    verify()
    result=dict(all_workers_joined=True,training=[v for _,v in trained],effects=effects,
        success=all(v['passed'] for v in effects.values()),behavior_calls=777600,label_calls=393984,
        search_calls=1209600,search_trajectories=2016,contract=json.loads((RUN/'CONTRACTS.json').read_text()),
        inference_teacher_calls=0,proposal_weights_unchanged=True)
    write_json(RUN/'COMPLETE.json',result);write_json(OUT/'RESULTS.json',result)
    write_json(RUN/'STATUS.json',dict(status='complete'));print(json.dumps(result),flush=True)

if __name__=='__main__':main()
