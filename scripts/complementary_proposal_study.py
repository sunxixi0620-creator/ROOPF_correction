"""Bounded conditional-proposal pilot: collect, train, potential gate, paid gate."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing as mp
from pathlib import Path
import shutil
import sys
import time
import numpy as np
import torch
import torch.nn.functional as functional
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from roopf.complementary_proposal import ComplementaryProposal, build_context
from roopf.online_portfolio import build_online
from roopf.free_allocation import build_free
from roopf.anchor_backbone import AnchorPolicyBackbone
from roopf.revision_tasks import ProceduralTask, population
from roopf.experiment_io import CaseStore, save_torch, sha256, seed_for, write_json
from scripts.unified_revision import bounded

ANCHORS = [ROOT/f'results/complementarity_four_group_20260929/d20/anchor_{s}/selected.pt' for s in range(3)]
SOURCES = ['scripts/complementary_proposal_study.py', 'roopf/complementary_proposal.py',
    'roopf/online_portfolio.py', 'roopf/free_allocation.py', 'roopf/model.py',
    'roopf/anchor_backbone.py', 'roopf/revision_tasks.py', 'roopf/experiment_io.py',
    'scripts/unified_revision.py', 'docs/experiments/COMPLEMENTARY_PROPOSAL_PROTOCOL.md']
OUT = ROOT/'docs/revision/complementary_proposal'


def setup(role, fid, instance, count=4):
    split = 'complementary_v1_'+role
    task = ProceduralTask(fid, instance, split, dim=20)
    pop = population(fid, instance, split, count=count, dim=20)
    return task, pop, seed_for('complementary_v1', role, fid, instance, 'policy')


def verify(run):
    run = Path(run)
    identity = json.loads((run/'identity.json').read_text())
    for name, digest in identity['sources'].items():
        assert sha256(ROOT/name) == digest, name
    for s, digest in enumerate(identity['anchors']):
        assert sha256(ANCHORS[s]) == digest
    assert identity['torch'] == torch.__version__
    return run, identity


def contracts():
    torch.set_num_threads(1)
    calls = 0
    # Capturing/re-scoring cannot change behavior or RNG; test odd budgets too.
    for budget, models in ((b, pair) for b in (113,600) for pair in
            ((build_online(20,b),build_context(dim=20,budget=b,capture=True)),
             (build_free(ANCHORS[0],20,b),build_context(anchor=ANCHORS[0],budget=b)))):
        outputs = []
        for model in models:
            task, pop, seed = setup('contract', 4, 0, 2)
            torch.manual_seed(seed)
            with torch.no_grad():
                _, trail, nfe, points = model(pop, task)
            assert nfe == budget and task.points == 2*budget and task.diagnostic_points == 0
            outputs.append((trail, points, torch.get_rng_state()))
            calls += task.points
        assert all(torch.equal(a,b) for a,b in zip(*outputs))
    task, pop, seed = setup('contract', 4, 0, 2)
    collector = build_context(capture=True, budget=113)
    torch.manual_seed(seed)
    collector(pop, task)
    calls += task.points
    snap = collector.snapshots[0]
    proposal = ComplementaryProposal(ANCHORS[0]).eval()
    with torch.no_grad():
        pred = proposal(snap['x'], snap['fitness'], snap['intent'], snap['intent_scores'],
                        snap['remaining'], snap['stagnation'], task)
        old = proposal.backbone.generator(snap['x'], task, snap['fitness'])
    assert torch.equal(pred, old)
    pred = proposal(snap['x'], snap['fitness'], snap['intent'], snap['intent_scores'],
                    snap['remaining'], snap['stagnation'], task)
    task.diagnostic_fitness(pred).mean().backward()
    assert all(torch.isfinite(p.grad).all() for p in proposal.parameters() if p.grad is not None)
    assert proposal.context[-1].weight.grad.abs().sum() > 0
    assert task.points == 226
    return dict(passed=True, O_capture_exact=True, old_Free_exact=True,
        zero_head_exact=True, odd_budget=True, finite_gradients=True,
        main_calls=calls, teacher_calls=task.diagnostic_points)


def prepare(run):
    if (run/'identity.json').exists():
        verify(run)
        return
    result = contracts()
    run.mkdir(parents=True, exist_ok=True)
    identity = dict(version='complementary_proposal_v1', sources={p:sha256(ROOT/p) for p in SOURCES},
        anchors=[sha256(p) for p in ANCHORS], torch=torch.__version__, numpy=np.__version__,
        condition=[20,600], train_instances=2, validation_instances=1, check_instances=1,
        search_instances=2, max_epochs=60, lr=.0001, patience_checks=8, seeds=[0,1,2])
    write_json(run/'identity.json', identity)
    write_json(run/'CONTRACTS.json', result)
    for name in SOURCES:
        dest=run/'source_snapshot'/name
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/name,dest)


def collect(job):
    run, role, fid, instance = job
    run, identity = verify(run)
    torch.set_num_threads(1)
    store = CaseStore(run/'data', identity)
    key = f'{role}_f{fid}_i{instance}'
    case = dict(role=role, fid=fid, instance=instance)
    with store.lock(key):
        if store.load(key, case) is not None:
            return key
        task, pop, seed = setup(role, fid, instance)
        model = build_context(capture=True)
        torch.manual_seed(seed)
        start=time.perf_counter()
        with torch.no_grad():
            _, trail, nfe, points = model(pop, task)
        assert nfe==600 and task.points==2400 and task.diagnostic_points==0
        assert len(model.snapshots)==8
        data = {k:torch.cat([s[k] for s in model.snapshots])
                for k in ('x','fitness','intent','intent_scores','stagnation')}
        data['remaining'] = torch.cat([torch.full((4,),s['remaining']) for s in model.snapshots])
        # Teachers only AFTER completing O, with a distinct task object.
        teacher, _, _ = setup(role, fid, instance)
        with torch.no_grad():
            data['threshold'] = torch.minimum(data['fitness'][:,0],
                teacher.diagnostic_fitness(data['intent']).min(1).values)
        data['scale'] = data['fitness'].std(1).clamp_min(1e-8)
        store.save(key,case,dict(data=data, trail=trail, points=points, main_calls=task.points,
            teacher_calls=teacher.diagnostic_points, seconds=time.perf_counter()-start))
    return key


def records(run, identity, role, device):
    store=CaseStore(run/'data',identity)
    result=[]
    for fid in range(36):
        for instance in range(2 if role=='train' else 1):
            case=dict(role=role,fid=fid,instance=instance)
            value=store.load(f'{role}_f{fid}_i{instance}',case)
            assert value is not None
            task=ProceduralTask(fid,instance,'complementary_v1_'+role,dim=20,device=device)
            data={k:v.to(device) for k,v in value['data'].items()}
            result.append((fid,instance,task,data))
    return result


def proposals(model, task, data):
    return model(data['x'],data['fitness'],data['intent'],data['intent_scores'],
        data['remaining'],data['stagnation'],task)


@torch.no_grad()
def utilities(model, rows, old=False):
    result=[]
    model.eval()
    for fid, instance, task, data in rows:
        p=(model.backbone.generator(data['x'],task,data['fitness']) if old else proposals(model,task,data))
        values=task.diagnostic_fitness(p).min(1).values
        u=bounded(data['threshold'],values,data['scale'])
        result.append(u.cpu().numpy())
    return np.array(result)


def train(job):
    run, seed=job
    run,identity=verify(run)
    torch.set_num_threads(1)
    out=run/f'model_{seed}'
    out.mkdir(exist_ok=True)
    if (out/'COMPLETE.json').exists():
        complete=json.loads((out/'COMPLETE.json').read_text())
        assert sha256(out/'selected.pt')==complete['selected_sha256']
        return complete
    device='cuda' if torch.cuda.is_available() else 'cpu'
    training=records(run,identity,'train',device)
    validation=records(run,identity,'validation',device)
    torch.manual_seed(seed_for('complementary_training',seed))
    model=ComplementaryProposal(ANCHORS[seed]).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=.0001)
    best=float(utilities(model,validation).mean())
    selected_epoch=0
    save_torch(out/'selected.pt',{k:v.detach().cpu() for k,v in model.state_dict().items()})
    history=[dict(epoch=0,validation=best)]
    stale=0
    start=time.perf_counter()
    for epoch in range(1,61):
        model.train()
        generator=np.random.default_rng(seed_for('complementary_order',seed,epoch))
        loss_sum=0.
        for index in generator.permutation(len(training)):
            _,_,task,data=training[index]
            optimizer.zero_grad(set_to_none=True)
            pred=proposals(model,task,data)
            z=(task.diagnostic_fitness(pred)-data['threshold'][:,None])/data['scale'][:,None]
            smooth=-.05*(torch.logsumexp(-z/.05,dim=1)-np.log(2))
            loss=(.05*functional.softplus(smooth/.05)).mean()
            assert torch.isfinite(loss)
            loss.backward()
            assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
            torch.nn.utils.clip_grad_norm_(model.parameters(),10)
            optimizer.step()
            loss_sum+=float(loss.detach())/len(training)
        score=None
        if epoch%2==0:
            score=float(utilities(model,validation).mean())
            if score>best+1e-5:
                best,selected_epoch,stale=score,epoch,0
                save_torch(out/'selected.pt',{k:v.detach().cpu() for k,v in model.state_dict().items()})
            else:
                stale+=1
        history.append(dict(epoch=epoch,loss=loss_sum,validation=score,seconds=time.perf_counter()-start))
        write_json(out/'history.json',history)
        write_json(out/'PROGRESS.json',dict(epoch=epoch,selected_epoch=selected_epoch,validation=best,device=device))
        save_torch(out/'last.pt',dict(model=model.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch))
        if stale>=8:
            break
    complete=dict(seed=seed,epoch=epoch,selected_epoch=selected_epoch,validation=best,
        selected_sha256=sha256(out/'selected.pt'),seconds=time.perf_counter()-start,
        train_teacher_calls=sum(t.diagnostic_points for _,_,t,_ in training),
        validation_teacher_calls=sum(t.diagnostic_points for _,_,t,_ in validation),
        parameters=sum(p.numel() for p in model.parameters()),device=device)
    write_json(out/'COMPLETE.json',complete)
    return complete


def interval(delta, level, practical):
    # Shape seeds x configurations x observations: all configurations preserved.
    by_recipe=delta.reshape(3,12,3,-1).mean((0,2,3))
    rng=np.random.default_rng(20260930)
    samples=by_recipe[rng.integers(0,12,(5000,12))].mean(1)
    tail=(1-level)/2
    lower,upper=np.quantile(samples,[tail,1-tail])
    per_seed=delta.mean(tuple(range(1,delta.ndim)))
    mean=float(delta.mean())
    return dict(mean=mean,lower=float(lower),upper=float(upper),confidence=level,
        seed_means=per_seed.tolist(),passed=bool(mean>=practical and lower>0 and (per_seed>0).all()))


def proposal_gate(run):
    run,identity=verify(run)
    torch.set_num_threads(1)
    rows=records(run,identity,'check','cpu')
    all_new,all_old=[],[]
    for seed in range(3):
        model=ComplementaryProposal(ANCHORS[seed]).eval()
        all_old.append(utilities(model,rows,old=True))
        model.load_state_dict(torch.load(run/f'model_{seed}/selected.pt',weights_only=False))
        all_new.append(utilities(model,rows))
    new,old=np.array(all_new),np.array(all_old)
    result=interval(new-old,.95,.001)
    result.update(new_utility=float(new.mean()),old_utility=float(old.mean()),
        teacher_calls=sum(t.diagnostic_points for _,_,t,_ in rows))
    save_torch(run/'proposal_utilities.pt',dict(new=new,old=old))
    write_json(run/'PROPOSAL_GATE.json',result)
    return result


def search(job):
    run,method,seed,fid,instance=job
    run,identity=verify(run)
    torch.set_num_threads(1)
    frozen=json.loads((run/'MODELS_FROZEN.json').read_text())
    for s,digest in enumerate(frozen['selected']):
        assert sha256(run/f'model_{s}/selected.pt')==digest
    store=CaseStore(run/'search',dict(identity=identity,frozen=frozen))
    case=dict(method=method,seed=seed,fid=fid,instance=instance)
    key=f'{method}_s{seed}_f{fid}_i{instance}'
    with store.lock(key):
        value=store.load(key,case)
        if value is not None:
            return value['utility']
        if method=='O':
            model=build_online(20,600)
        elif method=='A':
            model=AnchorPolicyBackbone(20,200,100).eval().requires_grad_(False)
            model.load_state_dict(torch.load(ANCHORS[seed],weights_only=False))
            model.MaxNFE=600
        elif method=='Old':
            model=build_context(anchor=ANCHORS[seed])
        else:
            p=ComplementaryProposal(ANCHORS[seed])
            p.load_state_dict(torch.load(run/f'model_{seed}/selected.pt',weights_only=False))
            model=build_context(proposal=p).eval().requires_grad_(False)
        task,pop,policy_seed=setup('search',fid,instance)
        torch.manual_seed(policy_seed)
        start=time.perf_counter()
        with torch.no_grad():
            _,trail,nfe,points=model(pop,task)
        assert nfe==600 and task.points==2400 and task.diagnostic_points==0
        assert points.shape==(4,500,20) and torch.isfinite(trail).all()
        assert (trail[:,1:]<=trail[:,:-1]).all()
        utility=bounded(task.initial_values.min(1).values,trail[:,-1],task.initial_values.std(1))
        store.save(key,case,dict(utility=utility,trail=trail,points=points,main_calls=task.points,
            teacher_calls=0,extra_selected=getattr(model,'extra_selected',None),
            seconds=time.perf_counter()-start))
        return utility


def parallel(fn,jobs,workers):
    result=[]
    with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn')) as pool:
        futures={pool.submit(fn,job):job for job in jobs}
        for future in as_completed(futures):
            result.append((futures[future],future.result()))
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',default='results/complementary_proposal_20260930')
    parser.add_argument('--workers',type=int,default=12)
    parser.add_argument('--checks-only',action='store_true')
    args=parser.parse_args()
    if args.checks_only:
        print(json.dumps(contracts()),flush=True)
        return
    run=ROOT/args.run
    prepare(run)
    start=time.perf_counter()
    write_json(run/'STATUS.json',dict(status='collecting'))
    parallel(collect,[(run,role,f,i) for role in ('train','validation','check')
        for f in range(36) for i in range(2 if role=='train' else 1)],args.workers)
    write_json(run/'STATUS.json',dict(status='training',workers=3))
    parallel(train,[(run,s) for s in range(3)],3)
    write_json(run/'MODELS_FROZEN.json',dict(selected=[sha256(run/f'model_{s}/selected.pt') for s in range(3)]))
    gate=proposal_gate(run)
    search_effects=None
    if gate['passed']:
        write_json(run/'STATUS.json',dict(status='paid_search',proposal_gate=gate))
        jobs=[(run,m,s,f,i) for m in ('A','O','Old','New') for s in ([-1] if m=='O' else range(3))
              for f in range(36) for i in range(2)]
        results=parallel(search,jobs,args.workers)
        matrices={m:np.zeros((3,36,2,4)) for m in ('A','O','Old','New')}
        for job,u in results:
            _,m,s,f,i=job
            if m=='O':
                matrices[m][:,f,i]=u.numpy()
            else:
                matrices[m][s,f,i]=u.numpy()
        search_effects={m:interval(matrices['New']-matrices[m],.975 if m!='A' else .95,.005)
                        for m in ('O','Old','A')}
        save_torch(run/'search_utilities.pt',matrices)
        write_json(run/'SEARCH_GATE.json',search_effects)
    run,identity=verify(run)
    store=CaseStore(run/'data',identity)
    behavior=teachers=0
    for role in ('train','validation','check'):
        for f in range(36):
            for i in range(2 if role=='train' else 1):
                value=store.load(f'{role}_f{f}_i{i}',dict(role=role,fid=f,instance=i))
                behavior+=value['main_calls'];teachers+=value['teacher_calls']
    training=[json.loads((run/f'model_{s}/COMPLETE.json').read_text()) for s in range(3)]
    report=dict(status='complete',all_workers_joined=True,proposal=gate,search=search_effects,
        training=training,behavior_calls=behavior,label_calls=teachers,
        search_calls=1728000 if gate['passed'] else 0,
        search_trajectories=2880 if gate['passed'] else 0,
        elapsed_seconds=time.perf_counter()-start,
        next_external_confirmation_authorized=False,
        success=bool(search_effects and all(search_effects[m]['passed'] for m in ('O','Old'))))
    write_json(run/'COMPLETE.json',report)
    write_json(run/'STATUS.json',report)
    OUT.mkdir(parents=True,exist_ok=True)
    write_json(OUT/'RESULTS.json',report)
    text=['# 互补提案机制试验','',
          '本轮是同一生成函数族的新实例机制试验，不是外部泛化证明。',
          '新模块依赖在线上下文；A 是原独立 anchor，不是将新模块关闭上下文后的模型。','',
          f"提案潜力门槛：{'通过' if gate['passed'] else '未通过'}；均值差 {gate['mean']:.6f}，95%CI [{gate['lower']:.6f}, {gate['upper']:.6f}]。"]
    if search_effects:
        for m,v in search_effects.items():
            text.append(f"\nNew−{m}：{v['mean']:.6f}，CI [{v['lower']:.6f}, {v['upper']:.6f}]，门槛 {'通过' if v['passed'] else '未通过'}。")
    else:
        text.append('\n按预先规则停止，未打开端到端搜索集。潜力不足不能用增加 epoch 或切换条件掩盖。')
    text.append('\nResidual 未参与本轮；原 residual 负结果保持。完整数字与调用成本见 RESULTS.json。')
    (OUT/'CONCLUSIONS.zh-CN.md').write_text('\n'.join(text)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    main()
