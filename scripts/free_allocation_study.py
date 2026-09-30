"""Single bounded allocation diagnosis; conditional confirmation, no tuning loop."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing as mp
from pathlib import Path
import shutil
import sys
import time
import zipfile
import numpy as np
import pandas as pd
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from roopf.free_allocation import build_free
from roopf.online_portfolio import build_online
from roopf.revision_tasks import ProceduralTask,population
from roopf.experiment_io import CaseStore,canonical,fingerprint,seed_for,sha256,write_json
from scripts import complementarity_four_group as previous
from scripts.unified_revision import bounded

OLD=ROOT/'results/complementarity_four_group_20260929'
CONDITIONS=previous.CONDITIONS
OUT=ROOT/'docs/revision/free_allocation'
SOURCES=['scripts/free_allocation_study.py','roopf/free_allocation.py',
         'roopf/online_portfolio.py','docs/experiments/FREE_ALLOCATION_PROTOCOL.md']


def verify(run):
    run=Path(run); identity=json.loads((run/'identity.json').read_text())
    for name,digest in identity['sources'].items():assert sha256(ROOT/name)==digest,name
    previous.verify(OLD)
    assert sha256(OLD/'identity.json')==identity['previous_identity_sha256']
    assert sha256(OLD/'MODELS_FROZEN.json')==identity['models_manifest_sha256']
    frozen=json.loads((OLD/'MODELS_FROZEN.json').read_text())
    for name,digest in frozen['models'].items():assert sha256(OLD/name)==digest,name
    return run,identity


def checks():
    torch.set_num_threads(1)
    calls=0
    for d in (10,20):
        pop=population(4,0,'allocation_contract',count=2,dim=d)
        results=[]
        for model in (build_online(d,301),build_free(None,d,301)):
            task=ProceduralTask(4,0,'allocation_contract',dim=d)
            torch.manual_seed(6381)
            _,trail,nfe,points=model(pop.clone(),task)
            assert nfe==301 and task.points==602 and task.diagnostic_points==0
            results.append((trail,points,torch.get_rng_state()));calls+=task.points
        assert all(torch.equal(a,b) for a,b in zip(*results))
        model=build_free(OLD/f'd{d}/anchor_0/selected.pt',d,113)
        assert model.router_model is None
        task=ProceduralTask(4,0,'allocation_contract',dim=d)
        original=model._baseline_candidates
        captured=[]
        def probe(x,fitness,problem,rest):
            proposal=original(x,fitness,problem,rest)
            expected=model.baseline_model.generator(x,problem,fitness)
            assert torch.equal(proposal,expected)
            captured.append(True)
            return proposal
        model._baseline_candidates=probe
        torch.manual_seed(6381)
        _,trail,nfe,points=model(pop.clone(),task)
        assert nfe==113 and task.points==226 and task.diagnostic_points==0
        assert len(captured)==7 and all(v['anchor_calls']==1 and v['residual_calls']==0 for v in model.decisions)
        assert all(len(set(v['selected']))==len(v['selected']) for v in model.decisions)
        calls+=task.points
    OUT.mkdir(parents=True,exist_ok=True)
    write_json(OUT/'CONTRACTS.json',dict(passed=True,dimensions=[10,20],
        anchor_free_exact_O_parity=True,extra_proposals_exact_anchor=True,
        odd_budgets_exact=True,residual_calls=0,teacher_calls=0,objective_calls=calls))


def prepare(run):
    run=Path(run)
    if (run/'identity.json').exists():verify(run);return
    previous.verify(OLD)
    assert json.loads((OLD/'COMPLETE.json').read_text())['all_workers_joined']
    checks()
    run.mkdir(parents=True,exist_ok=True)
    identity=dict(version='free_allocation_v1',sources={p:sha256(ROOT/p) for p in SOURCES},
        previous_identity_sha256=sha256(OLD/'identity.json'),models_manifest_sha256=sha256(OLD/'MODELS_FROZEN.json'),
        old_run=str(OLD),conditions=CONDITIONS,development_new_trajectories=2592,
        confirmation_split='allocation_v1_confirmation',torch=torch.__version__,numpy=np.__version__)
    write_json(run/'identity.json',identity)
    for p in SOURCES:
        q=run/'source_snapshot'/p;q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/p,q)
    write_json(run/'STATUS.json',dict(status='prepared'))


def spec(stage,d,b,s,m,f,i):
    if stage=='development':return previous.case_spec(d,b,s,m,f,i)
    split='allocation_v1_confirmation'
    task=ProceduralTask(f,i,split,dim=d)
    pop=population(f,i,split,count=4,dim=d)
    ps=seed_for('allocation_v1',split,d,f,i,'policy')
    return task,pop,dict(dim=d,budget=b,seed=s,method=m,fid=f,instance=i,
        task_parameters=task.params,population=pop,policy_seed=ps)


def key(d,b,s,m,f,i):return previous.case_key(d,b,s,m,f,i)


def evaluate(job):
    run,stage,d,b,s,m,f,i=job
    run,identity=verify(run);torch.set_num_threads(1)
    task,pop,case=spec(stage,d,b,s,m,f,i)
    store=CaseStore(run/stage,dict(protocol=identity,stage=stage))
    case_id=key(d,b,s,m,f,i)
    with store.lock(case_id):
        result=store.load(case_id,case)
        if result is None:
            start=time.perf_counter()
            model=(build_free(OLD/f'd{d}/anchor_{s}/selected.pt',d,b) if m=='Free'
                   else previous.make_model(OLD,d,b,s,m))
            torch.manual_seed(case['policy_seed'])
            with torch.no_grad():_,trail,nfe,points=model(pop.clone(),task)
            assert nfe==b and task.points==4*b and task.diagnostic_points==0
            assert points.shape==(4,b-100,d) and torch.isfinite(trail).all()
            assert (trail[:,1:]<=trail[:,:-1]).all()
            result=dict(trail=trail,points=points,initial_best=task.initial_values.min(1).values,
                initial_std=task.initial_values.std(1),main_points=task.points,teacher_points=0,nfe=nfe,
                seconds=time.perf_counter()-start,decisions=getattr(model,'decisions',[]),
                population_sha256=fingerprint(pop),task_parameters=task.params)
            store.save(case_id,case,result)
    return case_id


def jobs(run,stage):
    methods=['Free'] if stage=='development' else ['A','O','F','Free']
    return [(run,stage,d,b,s,m,f,i) for d,b in CONDITIONS for m in methods
            for s in ([-1] if m=='O' else range(3)) for f in range(36) for i in range(2)]


def execute_stage(run,stage,workers):
    write_json(run/'STATUS.json',dict(status='running_'+stage,expected_cases=len(jobs(run,stage))))
    with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn')) as ex:
        fs=[ex.submit(evaluate,j) for j in jobs(run,stage)]
        for f in as_completed(fs):f.result()


def analyze(run,stage):
    run,identity=verify(run);torch.set_num_threads(1)
    store=CaseStore(run/stage,dict(protocol=identity,stage=stage))
    _,old_identity=previous.verify(OLD)
    old_store=CaseStore(OLD/'cases',dict(protocol=old_identity,
        frozen=json.loads((OLD/'MODELS_FROZEN.json').read_text())))
    rows=[];curves=[];reused=0
    for d,b in CONDITIONS:
        for m in ('A','O','F','Free'):
            for s in ([-1] if m=='O' else range(3)):
                for f in range(36):
                    for i in range(2):
                        task,pop,case=spec(stage,d,b,s,m,f,i)
                        source=old_store if stage=='development' and m!='Free' else store
                        value=source.load(key(d,b,s,m,f,i),case)
                        assert value is not None and value['nfe']==b and value['main_points']==4*b
                        assert value['population_sha256']==fingerprint(pop) and value['teacher_points']==0
                        if source is old_store:reused+=4
                        u=bounded(value['initial_best'],value['trail'][:,-1],value['initial_std'])
                        for p in range(4):
                            decisions=[x for x in value['decisions'] if x['batch']==p]
                            selected_anchor=(sum(sum(op==-2 for op in x.get('chosen_ops',[])) for x in decisions) if m=='Free' else None)
                            rows.append(dict(dim=d,budget=b,method=m,seed=s,fid=f,instance=i,population=p,
                                initial=float(value['initial_best'][p]),initial_std=float(value['initial_std'][p]),
                                final=float(value['trail'][p,-1]),utility=float(u[p]),selected_anchor=selected_anchor))
                        cu=bounded(value['initial_best'][:,None],value['trail'],value['initial_std'][:,None]).mean(0)
                        for t,v in enumerate(cu):curves.append(dict(dim=d,budget=b,method=m,nfe=min(b,102+2*t),utility=float(v)))
    data=pd.DataFrame(rows);assert len(data)==8640
    out=OUT/stage;out.mkdir(parents=True,exist_ok=True)
    data.to_csv(out/'results.csv',index=False)
    pd.DataFrame(curves).groupby(['dim','budget','method','nfe'],as_index=False).utility.mean().to_csv(out/'curves.csv',index=False)
    comparisons={}
    for d,b in CONDITIONS:
        part=data[(data.dim==d)&(data.budget==b)];matrix={}
        for m in ('A','O','F','Free'):
            a=part[part.method==m]
            if m=='O':a=pd.concat([a.assign(seed=s) for s in range(3)])
            matrix[m]=a.sort_values(['seed','fid','instance','population']).utility.to_numpy().reshape(3,12,3,2,4)
        rng=np.random.default_rng(20260930+d+b)
        draws=[(rng.integers(0,3,3),rng.integers(0,12,12),rng.integers(0,2,(12,2))) for _ in range(5000)]
        effects={}
        for m in ('A','O','F'):
            raw=matrix['Free']-matrix[m];values=raw.mean(-1)
            samples=[np.mean([values[ss][:,fam,:,ii[j]].mean() for j,fam in enumerate(ff)]) for ss,ff,ii in draws]
            adjusted=np.quantile(samples,[.05/6,1-.05/6]).tolist()
            per_seed=values.mean((1,2,3)).tolist();mean=float(values.mean())
            effects['Free_vs_'+m]=dict(mean=mean,ci95=np.quantile(samples,[.025,.975]).tolist(),
                ci_adjusted=adjusted,by_seed=per_seed,degradation_rate=float((raw<-.02).mean()),
                practical_pass=bool(mean>=.005 and adjusted[0]>0 and all(x>0 for x in per_seed)))
        comparisons[f'd{d}_b{b}']=effects
    write_json(out/'COMPARISONS.json',comparisons)
    passed=all(x['practical_pass'] for x in comparisons['d20_b600'].values())
    write_json(out/'DECISION.json',dict(target='d20_b600',all_three_target_contrasts_pass=passed,
        next_step=('new_instance_confirmation' if stage=='development' and passed else 'stop_and_report'),
        stage=stage,automatic_parameter_search=False))
    write_json(out/'VERIFICATION.json',dict(passed=True,total_rows=8640,reused_trajectories=reused,
        new_trajectories=8640-reused,new_main_points=1036800 if stage=='development' else 3456000,
        all_case_hashes_verified=True,teacher_points=0,all_workers_joined=True))
    lines=['# 自由分配评估槽位：'+stage,'',
        'Free与O的核心差异是两个额外提案来自冻结anchor还是均匀采样。',
        '开发阶段复用已经看过的四组任务；确认阶段（若触发）仅为共享配方的新实例，不是全新函数族。','',
        '| 条件 | 对照 | 平均有界改善差 | 校正区间 | 通过门槛 |','|---|---|---:|---|---|']
    for cond,es in comparisons.items():
        for name,v in es.items():lines.append(f'| {cond} | {name} | {v["mean"]:.6f} | {v["ci_adjusted"]} | {v["practical_pass"]} |')
    lines+=['',f'预定20维600预算的三项联合验收：{passed}。',
        'Free−F检验取消固定槽位和门控的整体分配变化，不能单独归因于某一个条件。',
        'Free−O检验相同在线选择规则下，离线提案相对两个随机提案的增量价值。']
    (out/'RESULTS.zh-CN.md').write_text('\n'.join(lines)+'\n')
    return passed


def archive(run,stages):
    run,identity=verify(run)
    target=ROOT/'artifacts/free_allocation_v1';target.mkdir(parents=True,exist_ok=True)
    files=[]
    for stage in stages:
        for d,b in CONDITIONS:
            methods=['Free'] if stage=='development' else ['A','O','F','Free']
            for m in methods:
                for s in ([-1] if m=='O' else range(3)):
                    path=target/f'{stage}_d{d}_b{b}_{m}_s{s}.zip'
                    with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED) as z:
                        folder=run/stage;z.write(folder/'identity.json',f'{stage}/identity.json')
                        prefix=f'd{d}_b{b}_{m}_s{s if s>=0 else "none"}_'
                        for p in sorted(folder.glob(prefix+'*')):
                            if p.suffix in ('.pt','.json'):z.write(p,str(p.relative_to(run)))
                    assert path.stat().st_size<95_000_000
                    files.append(dict(file=path.name,sha256=sha256(path)))
    with zipfile.ZipFile(target/'sources.zip','w',zipfile.ZIP_DEFLATED) as z:
        for name in SOURCES:z.write(ROOT/name,name)
    files.append(dict(file='sources.zip',sha256=sha256(target/'sources.zip')))
    write_json(target/'manifest.json',dict(files=files,identity=identity,
        reused_models_and_cases='artifacts/complementarity_v1'))
    shutil.copyfile(run/'identity.json',OUT/'identity.json')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','all'])
    parser.add_argument('--run',type=Path,required=True);parser.add_argument('--workers',type=int,default=12)
    args=parser.parse_args()
    if args.mode=='prepare':prepare(args.run);return
    verify(args.run)
    try:
        execute_stage(args.run,'development',args.workers)
        gate=analyze(args.run,'development');stages=['development']
        if gate:
            execute_stage(args.run,'confirmation',args.workers)
            analyze(args.run,'confirmation');stages.append('confirmation')
        archive(args.run,stages)
        done=dict(status='complete',stages=stages,confirmation_triggered=gate,all_workers_joined=True)
        write_json(args.run/'COMPLETE.json',done);write_json(args.run/'STATUS.json',done);write_json(OUT/'COMPLETE.json',done)
    except Exception as e:
        write_json(args.run/'FAILED.json',dict(error=repr(e)));raise


if __name__=='__main__':main()
