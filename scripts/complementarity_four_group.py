"""Frozen four-group execution after completed long training, with automatic handoff."""
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
import zipfile
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from roopf.anchor_backbone import AnchorPolicyBackbone
from roopf.online_portfolio import build_online
from roopf.unified import build_unified
from roopf.revision_tasks import ProceduralTask, population
from roopf.experiment_io import CaseStore, canonical, fingerprint, seed_for, sha256, write_json
from scripts import unified_revision as legacy
from scripts.complementarity_training import verify as verify_training

CONDITIONS = ((10, 300), (20, 300), (20, 600))
METHODS = ('A', 'O', 'F', 'FR')
SPLIT = 'complementarity_v1_confirmation'
SOURCES = ['scripts/complementarity_four_group.py', 'roopf/online_portfolio.py',
    'roopf/unified.py', 'roopf/model.py', 'roopf/anchor_backbone.py', 'roopf/revision_tasks.py',
    'roopf/experiment_io.py', 'roopf/residual_features.py', 'scripts/unified_revision.py',
    'scripts/complementarity_training.py', 'scripts/check_complementarity_contracts.py',
    'docs/experiments/COMPLEMENTARITY_FOUR_GROUP_PROTOCOL.md',
    'docs/experiments/COMPLEMENTARITY_TRAINING_PROTOCOL.md']


def verify(run):
    run = Path(run)
    identity = json.loads((run/'identity.json').read_text())
    for name, digest in identity['sources'].items():
        if sha256(ROOT/name) != digest:
            raise ValueError(f'Frozen four-group source changed: {name}')
    training = Path(identity['training_run'])
    verify_training(training)
    assert sha256(training/'identity.json') == identity['training_identity_sha256']
    assert sha256(legacy.TEMPLATE) == identity['template_sha256']
    return run, identity


def prepare(run, training):
    run, training = Path(run), Path(training).resolve()
    if (run/'identity.json').exists():
        verify(run)
        return
    verify_training(training)
    assert json.loads((ROOT/'docs/revision/complementarity/CONTRACTS.json').read_text())['passed']
    run.mkdir(parents=True, exist_ok=True)
    identity = dict(version='complementarity_four_group_v1',
        sources={p:sha256(ROOT/p) for p in SOURCES}, training_run=str(training),
        training_identity_sha256=sha256(training/'identity.json'),
        template_sha256=sha256(legacy.TEMPLATE), conditions=CONDITIONS,
        methods=METHODS, split=SPLIT, torch=torch.__version__, numpy=np.__version__,
        instances=2, populations=4, model_seeds=[0, 1, 2], main_trajectories=8640)
    write_json(run/'identity.json', identity)
    for name in SOURCES:
        target = run/'source_snapshot'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/name, target)
    write_json(run/'STATUS.json', dict(status='waiting_for_anchor_training'))


def stage_verify(path):
    path = Path(path)
    _, root_id = verify(path.parent)
    child = json.loads((path/'identity.json').read_text())
    assert child['parent_identity_sha256'] == sha256(path.parent/'identity.json')
    for seed, digest in child['anchors'].items():
        assert sha256(path/f'anchor_{seed}/selected.pt') == digest
    return path, child


def freeze_anchors(run):
    run, identity = verify(run)
    training = Path(identity['training_run'])
    completed = json.loads((training/'COMPLETE.json').read_text())
    assert completed['all_workers_joined'] and len(completed['models']) == 6
    for dim in (10, 20):
        stage = run/f'd{dim}'
        stage.mkdir(exist_ok=True)
        hashes = {}
        for seed in range(3):
            source = training/f'd{dim}_s{seed}'
            meta = json.loads((source/'COMPLETE.json').read_text())
            assert meta['epoch'] == 1000
            assert sha256(source/'selected.pt') == meta['selected_sha256']
            out = stage/f'anchor_{seed}'
            out.mkdir(exist_ok=True)
            for name in ('selected.pt', 'COMPLETE.json', 'history.json'):
                target = out/name
                if target.exists():
                    assert sha256(target) == sha256(source/name)
                else:
                    shutil.copyfile(source/name, target)
            hashes[str(seed)] = meta['selected_sha256']
        child = dict(version='complementarity_stage', dimension=dim,
            parent_identity_sha256=sha256(run/'identity.json'), sources=identity['sources'], anchors=hashes)
        if (stage/'identity.json').exists():
            assert json.loads((stage/'identity.json').read_text()) == canonical(child)
        else:
            write_json(stage/'identity.json', child)
    write_json(run/'ANCHORS_FROZEN.json', dict(training_complete_sha256=sha256(training/'COMPLETE.json'),
        models={f'd{d}_s{s}':sha256(run/f'd{d}/anchor_{s}/selected.pt') for d in (10, 20) for s in range(3)}))


def label_job(job):
    legacy.verify = stage_verify
    return legacy.collect(job)


def residual_job(job):
    legacy.verify = stage_verify
    result = legacy.train_residual(job)
    assert result['train_rows'] == 547200 and result['validation_rows'] == 273600
    return result


def pool_jobs(fn, jobs, workers):
    values = []
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('spawn')) as executor:
        futures = [executor.submit(fn, job) for job in jobs]
        for f in as_completed(futures):
            values.append(f.result())
    return values


def make_model(run, dim, budget, seed, method):
    stage = Path(run)/f'd{dim}'
    if method == 'O':
        assert seed == -1
        return build_online(dim, budget)
    anchor, residual = stage/f'anchor_{seed}/selected.pt', stage/f'residual_{seed}/selected.pt'
    if method == 'A':
        model = AnchorPolicyBackbone(dim, 200, 100).eval().requires_grad_(False)
        model.MaxNFE = budget
        model.load_state_dict(torch.load(anchor, map_location='cpu', weights_only=False), strict=True)
        return model
    return build_unified(anchor, residual, 'full' if method == 'FR' else 'no_residual', dim=dim, budget=budget)


def case_spec(dim, budget, seed, method, fid, instance):
    task = ProceduralTask(fid, instance, SPLIT, dim=dim)
    pop = population(fid, instance, SPLIT, count=4, dim=dim)
    policy_seed = seed_for('complementarity_v1', SPLIT, dim, fid, instance, 'policy')
    spec = dict(dim=dim, budget=budget, seed=seed, method=method, fid=fid, instance=instance,
                task_parameters=task.params, population=pop, policy_seed=policy_seed)
    return task, pop, spec


def case_key(dim, budget, seed, method, fid, instance):
    return f'd{dim}_b{budget}_{method}_s{seed if seed >= 0 else "none"}_f{fid:02d}_i{instance}'


def run_case(run, dim, budget, seed, method, fid, instance):
    task, pop, spec = case_spec(dim, budget, seed, method, fid, instance)
    start = time.perf_counter()
    model = make_model(run, dim, budget, seed, method)
    torch.manual_seed(spec['policy_seed'])
    with torch.no_grad():
        _, trail, nfe, points = model(pop.clone(), task)
    seconds = time.perf_counter()-start
    assert nfe == budget and task.points == 4*budget and task.diagnostic_points == 0
    assert torch.isfinite(trail).all() and (trail[:, 1:] <= trail[:, :-1]).all()
    assert points.shape == (4, budget-100, dim)
    return dict(trail=trail, points=points, initial_best=task.initial_values.min(1).values,
        initial_std=task.initial_values.std(1), nfe=nfe, main_points=task.points,
        teacher_points=0, seconds=seconds, decisions=getattr(model, 'decisions', []),
        population_sha256=fingerprint(pop), policy_rng_after=torch.get_rng_state())


def evaluate(job):
    run, dim, budget, seed, method, fid, instance = job
    run, identity = verify(run)
    torch.set_num_threads(1)
    frozen = json.loads((run/'MODELS_FROZEN.json').read_text())
    if method != 'O':
        for label in ('anchor', 'residual'):
            path = run/f'd{dim}/{label}_{seed}/selected.pt'
            assert sha256(path) == frozen['models'][str(path.relative_to(run))]
    store = CaseStore(run/'cases', dict(protocol=identity, frozen=frozen))
    _, _, spec = case_spec(dim, budget, seed, method, fid, instance)
    key = case_key(dim, budget, seed, method, fid, instance)
    with store.lock(key):
        value = store.load(key, spec)
        if value is None:
            value = run_case(run, dim, budget, seed, method, fid, instance)
            store.save(key, spec, value)
    return key


def jobs(run):
    return [(run, d, b, s, m, f, i) for d, b in CONDITIONS for m in METHODS
            for s in ([-1] if m == 'O' else range(3)) for f in range(36) for i in range(2)]


def analyze(run):
    run, identity = verify(run)
    frozen = json.loads((run/'MODELS_FROZEN.json').read_text())
    store = CaseStore(run/'cases', dict(protocol=identity, frozen=frozen))
    rows, curves = [], []
    for _, d, b, s, m, f, i in jobs(run):
        _, pop, spec = case_spec(d, b, s, m, f, i)
        value = store.load(case_key(d, b, s, m, f, i), spec)
        assert value is not None and value['nfe'] == b and value['main_points'] == 4*b
        assert value['teacher_points'] == 0 and value['population_sha256'] == fingerprint(pop)
        u = legacy.bounded(value['initial_best'], value['trail'][:, -1], value['initial_std'])
        for p in range(4):
            ds = [x for x in value['decisions'] if x['batch'] == p]
            rows.append(dict(dim=d, budget=b, seed=s, method=m, fid=f, instance=i, population=p,
                initial=float(value['initial_best'][p]), final=float(value['trail'][p,-1]),
                utility=float(u[p]), nfe=b, seconds_batch=value['seconds'],
                accepted=sum(x.get('accepted', False) for x in ds),
                residual_changed=sum(x.get('residual_changed', False) for x in ds)))
        cu = legacy.bounded(value['initial_best'][:,None], value['trail'], value['initial_std'][:,None]).mean(0)
        for t, u in enumerate(cu):
            curves.append(dict(dim=d, budget=b, seed=s, method=m, fid=f, instance=i,
                nfe=min(b,102+2*t), utility=float(u)))
    data = pd.DataFrame(rows)
    assert len(data) == 8640 and int(data.nfe.sum()) == 3456000
    out = ROOT/'docs/revision/complementarity/four_group'
    out.mkdir(parents=True, exist_ok=True)
    data.to_csv(out/'results.csv', index=False)
    pd.DataFrame(curves).groupby(['dim','budget','method','nfe'],as_index=False).utility.mean().to_csv(out/'curves.csv',index=False)
    comparisons = {}
    for d,b in CONDITIONS:
        part = data[(data.dim == d)&(data.budget == b)]
        matrix = {}
        index = ['seed','fid','instance','population']
        for method in METHODS:
            a = part[part.method == method]
            if method == 'O':
                a = pd.concat([a.assign(seed=s) for s in range(3)])
            matrix[method] = a.sort_values(index).utility.to_numpy().reshape(3,12,3,2,4)
        rng = np.random.default_rng(20260929+d+b)
        draws = [(rng.integers(0,3,3),rng.integers(0,12,12),rng.integers(0,2,(12,2))) for _ in range(5000)]
        results = {}
        for a,c in [('F','A'),('F','O'),('FR','A'),('FR','O'),('FR','F')]:
            raw = matrix[a]-matrix[c]
            values = raw.mean(-1)
            samples = [np.mean([values[ss][:,family,:,ii[j]].mean() for j,family in enumerate(ff)])
                       for ss,ff,ii in draws]
            ci95 = np.quantile(samples,[.025,.975]).tolist()
            ci99 = np.quantile(samples,[.005,.995]).tolist()
            per_seed = values.mean((1,2,3)).tolist()
            mean = float(values.mean())
            results[a+'_vs_'+c] = dict(mean=mean,ci95=ci95,ci99=ci99,by_seed=per_seed,
                degradation_rate=float((raw < -.02).mean()),
                practical_pass=bool(mean>=.005 and ci99[0]>0 and all(x>0 for x in per_seed)))
        comparisons[f'd{d}_b{b}'] = dict(effects=results,
            F_complementarity=results['F_vs_A']['practical_pass'] and results['F_vs_O']['practical_pass'],
            FR_complementarity=results['FR_vs_A']['practical_pass'] and results['FR_vs_O']['practical_pass'],
            residual_increment=results['FR_vs_F']['practical_pass'])
    write_json(out/'COMPARISONS.json',comparisons)
    write_json(out/'VERIFICATION.json',dict(passed=True,cases=2160,trajectories=8640,
        main_points=3456000,teacher_points=0,online_independent_trajectories=864,
        online_results_shared_not_replicated=True,all_hashes_verified=True,all_workers_joined=True))
    return out, comparisons


def timing(run):
    torch.set_num_threads(1)
    run, identity = verify(run)
    frozen = json.loads((run/'MODELS_FROZEN.json').read_text())
    original = CaseStore(run/'cases',dict(protocol=identity,frozen=frozen))
    repeats = CaseStore(run/'timing',dict(protocol=identity,frozen=frozen,timing=True))
    rows=[]
    for d,b in ((10,300),(20,600)):
        for f in (0,24):
            for m in METHODS:
                s=-1 if m=='O' else 0
                key=case_key(d,b,s,m,f,0)
                _,_,spec=case_spec(d,b,s,m,f,0)
                with repeats.lock(key):
                    result=repeats.load(key,spec)
                    if result is None:
                        result=run_case(run,d,b,s,m,f,0)
                        repeats.save(key,spec,result)
                reference=original.load(key,spec)
                assert torch.equal(result['trail'],reference['trail']) and torch.equal(result['points'],reference['points'])
                rows.append(dict(dim=d,budget=b,fid=f,method=m,populations=4,seconds_batch=result['seconds']))
    out=ROOT/'docs/revision/complementarity/four_group'
    out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(out/'serial_timing.csv',index=False)
    write_json(out/'TIMING.json',dict(batches=16,trajectories=64,extra_points=28800,
        exact_trajectory_repeats=True,shared_host=True,added_to_main_N=False))


def finalize(run):
    out, comparisons = analyze(run)
    training=Path(json.loads((run/'identity.json').read_text())['training_run'])
    duration=[]
    training_rows=[]
    for d in (10,20):
        for s in range(3):
            source=training/f'd{d}_s{s}'
            meta=json.loads((source/'COMPLETE.json').read_text())
            history=json.loads((source/'history.json').read_text())
            prefix_best=max(x['validation'] for x in history if x['epoch']<=80 and x['validation'] is not None)
            duration.append(dict(**meta, best_through80=prefix_best,
                selected_minus_best80=meta['best_validation']-prefix_best,
                near_cap=meta['selected_epoch']>=950))
            for row in history:
                training_rows.append(dict(dimension=d,seed=s,epoch=row['epoch'],
                    loss=row['loss'],validation=row['validation']))
            shutil.copyfile(source/'history.json',out/f'anchor_d{d}_s{s}_history.json')
    write_json(out/'ANCHOR_DURATION.json',duration)
    pd.DataFrame(training_rows).to_csv(out/'anchor_training_curves.csv',index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    training_frame=pd.DataFrame(training_rows)
    for ax,d in zip(axes,(10,20)):
        for s in range(3):
            frame=training_frame[(training_frame.dimension==d)&(training_frame.seed==s)].dropna(subset=['validation'])
            ax.plot(frame.epoch,frame.validation,label=f'Seed {s}')
        ax.axvline(80,color='gray',linestyle=':');ax.set_title(f'{d}D anchor validation')
        ax.set_xlabel('Epoch');ax.set_ylabel('Bounded improvement');ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(out/'anchor_training.png',dpi=180);fig.savefig(out/'anchor_training.pdf');plt.close(fig)
    for name in ('identity.json','ANCHORS_FROZEN.json','MODELS_FROZEN.json'):
        shutil.copyfile(run/name,out/name)
    costs=dict(main_points=3456000,main_trajectories=8640,
        label_behavior_points=388800,label_teacher_points=4924800,
        timing_extra_points=28800,label_train_rows_per_model=547200,label_validation_rows_per_model=273600)
    costs.update(anchor_training_points_including_prefix=sum(x['training_points'] for x in duration),
        anchor_validation_points_including_prefix=sum(x['validation_points'] for x in duration),
        new_anchor_training_points=sum(x['extension_training_points'] for x in duration),
        new_anchor_validation_points=sum(x['extension_validation_points'] for x in duration))
    write_json(out/'COSTS.json',costs)
    lines=['# 离线、在线与融合：四组确认结果','',
        '同一训练配方族的新实例确认；不是完全未见函数族或新的CEC独立测试。',
        'A为充分预算训练后按验证集选中的anchor，O不加载预训练权重，F不含residual，FR含匹配重训residual。',
        'O仅运行一次，再作为三个anchor种子的共享配对对照；不扩大其独立样本数。','',
        '| 条件 | 对照 | 平均有界改善差 | 99%区间（条件内五对照校正） | 通过实用门槛 |',
        '|---|---|---:|---|---|']
    for condition,result in comparisons.items():
        for key,v in result['effects'].items():
            lines.append(f'| {condition} | {key} | {v["mean"]:.6f} | {v["ci99"]} | {v["practical_pass"]} |')
    lines+=['','F或FR必须同时通过相对A和O的验收才能支持该条件的互补性。',
            'FR相对F的独立收益另行验收；区间跨零不表示等效。没有根据结果修改在线对照或扩大样本。']
    (out/'RESULTS.zh-CN.md').write_text('\n'.join(lines)+'\n')
    archive=ROOT/'artifacts/complementarity_v1'
    archive.mkdir(parents=True,exist_ok=True)
    entries=[]
    for d,b in CONDITIONS:
        for m in METHODS:
            for s in ([-1] if m=='O' else range(3)):
                target=archive/f'd{d}_b{b}_{m}_s{s}.zip'
                with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as z:
                    for path in sorted((run/'cases').glob(f'd{d}_b{b}_{m}_s{s if s>=0 else "none"}_*')):
                        if path.suffix in ('.json','.pt'):z.write(path,str(path.relative_to(run)))
                    z.write(run/'cases/identity.json','cases/identity.json')
                assert target.stat().st_size<95_000_000
                entries.append(dict(file=target.name,sha256=sha256(target)))
    for d in (10,20):
        for s in range(3):
            for module in ('anchor','residual'):
                target=archive/f'{module}_d{d}_s{s}.pt'
                shutil.copyfile(run/f'd{d}/{module}_{s}/selected.pt',target)
                entries.append(dict(file=target.name,sha256=sha256(target)))
            for split in ('residual_training','residual_validation'):
                target=archive/f'labels_d{d}_s{s}_{split}.zip'
                with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as z:
                    folder=run/f'd{d}/labels_{s}'
                    z.write(folder/'identity.json',f'd{d}/labels_{s}/identity.json')
                    for path in sorted(folder.glob(split+'_*')):
                        if path.suffix in ('.pt','.json'):z.write(path,str(path.relative_to(run)))
                assert target.stat().st_size<95_000_000
                entries.append(dict(file=target.name,sha256=sha256(target)))
            target=archive/f'training_d{d}_s{s}.zip'
            with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as z:
                for path in sorted((training/f'd{d}_s{s}').glob('epoch*.pt')):
                    z.write(path,str(path.relative_to(training)))
                for name in ('history.json','COMPLETE.json'):
                    z.write(training/f'd{d}_s{s}'/name,f'd{d}_s{s}/{name}')
                for name in ('history.json','COMPLETE.json'):
                    z.write(run/f'd{d}/residual_{s}'/name,f'd{d}/residual_{s}/{name}')
            assert target.stat().st_size<95_000_000
            entries.append(dict(file=target.name,sha256=sha256(target)))
    with zipfile.ZipFile(archive/'sources.zip','w',zipfile.ZIP_DEFLATED) as z:
        for name in SOURCES:z.write(ROOT/name,name)
    entries.append(dict(file='sources.zip',sha256=sha256(archive/'sources.zip')))
    write_json(archive/'manifest.json',dict(files=entries,source_identity=json.loads((run/'identity.json').read_text())))
    done=dict(status='complete_four_group',all_workers_joined=True,trajectories=8640)
    write_json(run/'COMPLETE.json',done)
    write_json(out/'COMPLETE.json',done)
    write_json(run/'STATUS.json',done)


def execute(run, workers):
    run, identity = verify(run)
    training=Path(identity['training_run'])
    while not (training/'COMPLETE.json').exists():
        verify(run)
        status=json.loads((training/'STATUS.json').read_text())
        if status.get('supervisor_pid'):
            os.kill(status['supervisor_pid'],0)
        time.sleep(30)
    freeze_anchors(run)
    write_json(run/'STATUS.json',dict(status='collecting_matched_residual_labels'))
    label_jobs=[(run/f'd{d}',s,split,f,i) for d in (10,20) for s in range(3)
                for split in ('residual_training','residual_validation') for f in range(36)
                for i in range(2 if split=='residual_training' else 1)]
    pool_jobs(label_job,label_jobs,workers)
    write_json(run/'STATUS.json',dict(status='training_matched_residuals'))
    residuals=pool_jobs(residual_job,[(run/f'd{d}',s) for d in (10,20) for s in range(3)],3)
    frozen=dict(models={str(p.relative_to(run)):sha256(p) for d in (10,20) for s in range(3)
        for p in [run/f'd{d}/anchor_{s}/selected.pt',run/f'd{d}/residual_{s}/selected.pt']},
        selected_without_confirmation_results=True,residual_training=residuals)
    write_json(run/'MODELS_FROZEN.json',frozen)
    write_json(run/'STATUS.json',dict(status='evaluating_four_groups',expected_cases=2160))
    pool_jobs(evaluate,jobs(run),workers)
    write_json(run/'STATUS.json',dict(status='serial_timing_and_verification'))
    timing(run)
    finalize(run)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=['prepare','all','finalize'])
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--training',type=Path,default=ROOT/'results/complementarity_20260929')
    parser.add_argument('--workers',type=int,default=12)
    args=parser.parse_args()
    if args.mode=='prepare':prepare(args.run,args.training)
    elif args.mode=='finalize':finalize(args.run)
    else:
        try:execute(args.run,args.workers)
        except Exception as e:
            write_json(args.run/'FAILED.json',dict(error=repr(e),pid=os.getpid()))
            raise


if __name__=='__main__':main()
