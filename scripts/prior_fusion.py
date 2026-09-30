"""Frozen predictive prior + prequential mean fusion: matched closed-loop test."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
from pathlib import Path
import sys,json,time,multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor,as_completed
import torch
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import fewshot_prior as prior
from scripts import label_repeatability as repeat
from roopf.revision_tasks import ProceduralTask
from roopf.online_portfolio import build_online
from roopf.experiment_io import CaseStore,sha256,fingerprint,seed_for,write_json,save_torch
from scripts.unified_revision import bounded
RUN=ROOT/'results/prior_fusion_20260930';OUT=ROOT/'docs/revision/prior_fusion'
SOURCES=['scripts/prior_fusion.py','docs/experiments/PRIOR_FUSION_PROTOCOL.md','roopf/online_portfolio.py','roopf/model.py','roopf/revision_tasks.py']
METHODS=('A','O','F','Fixed','Fshuffled','Fanalytic')


def verify():
    identity=json.loads((RUN/'identity.json').read_text())
    for p,h in identity['sources'].items():assert sha256(ROOT/p)==h
    assert sha256(prior.RUN/'identity.json')==identity['prior_identity'];prior.verify()
    for p,h in identity['models'].items():assert sha256(prior.RUN/p)==h
    return identity


def model_for(fid,seed,kind):
    fold=(fid//3)%3;model=prior.net(fold,seed)
    model.load_state_dict(torch.load(prior.RUN/f'model_{fold}_{kind}_{seed}.pt',weights_only=False)['state'])
    return model.eval().requires_grad_(False)


@torch.no_grad()
def rollout(fid,instance,pop,seed,method,budget=300,role='prior_fusion_v1',forced_weight=None):
    task=ProceduralTask(fid,instance,role,dim=20)
    g=torch.Generator().manual_seed(seed_for(role,'initial',fid,instance,pop))
    ax=10*torch.rand(1,20,20,generator=g)-5;ay=task.calfitness(ax);initial_x=ax.clone();initial_y=ay.clone()
    initial=ay.min(1).values;scale=ay.std(1).clamp_min(1e-8)
    needs_prior=method not in ('O',);needs_online=method!='A'
    model=model_for(fid,seed,'shuffled' if method=='Fshuffled' else 'correct') if needs_prior and method!='Fanalytic' else None
    model_before=fingerprint(model.state_dict()) if model is not None else None
    surrogate=build_online(20,budget).surrogate if needs_online else None
    errors=torch.ones(2);records=[];trail=[];points=[];pool_hashes=[]
    start=time.perf_counter()
    for step in range(budget-20):
        n=20+step;remaining=(budget-n)/budget
        pg=torch.Generator().manual_seed(seed_for(role,'pool',fid,instance,pop,step))
        best=ax[0,ay[0].argmin()]
        pool=torch.cat((10*torch.rand(64,20,generator=pg)-5,
            (best+10*(.05+.35*remaining)*torch.randn(64,20,generator=pg)).clamp(-5,5)),0)[None]
        cg=torch.Generator().manual_seed(seed_for(role,'context',fid,instance,pop,step))
        ci=torch.randperm(n,generator=cg)[:min(40,n)]
        cx,cy=ax[:,ci],ay[:,ci];mean=cy.mean(1,keepdim=True);std=cy.std(1,keepdim=True,unbiased=False).clamp_min(1e-6)
        before=task.points
        if method=='Fanalytic':
            cr=(cx/5).square().mean(-1);qr=(pool/5).square().mean(-1)
            pm=mean+std*(qr-cr.mean(1,keepdim=True))/cr.std(1,keepdim=True,unbiased=False).clamp_min(.01)
        elif needs_prior:pm=model(cx,(cy-mean)/std,pool)*std+mean
        else:pm=None
        if needs_online:om,_=surrogate.predict(ax,ay,pool,task)
        else:om=None
        assert task.points==before and task.diagnostic_points==0
        if method=='A':w=1.;score=pm
        elif method=='O':w=0.;score=om
        else:
            w=.5 if method=='Fixed' else float((errors[1]+.05)/(errors.sum()+.1))
            if forced_weight is not None:w=forced_weight
            score=w*pm+(1-w)*om
        score=score.clone();score[torch.cdist(pool,ax).amin(2)<1e-6]=torch.inf
        index=int(score[0].argmin());assert torch.isfinite(score[0,index])
        chosen=pool[:,index:index+1];py=float(pm[0,index]) if pm is not None else None;oy=float(om[0,index]) if om is not None else None
        y=task.calfitness(chosen)
        if method in ('F','Fshuffled','Fanalytic'):
            errors=.9*errors+.1*((torch.tensor([py,oy])-y[0,0])/scale[0]).square()
        records.append(dict(nfe=n+1,prior_weight=w,prior_prediction=py,online_prediction=oy,value=float(y[0,0]),context_indices=ci))
        pool_hashes.append(fingerprint(pool));points.append(chosen)
        ax=torch.cat((ax,chosen),1);ay=torch.cat((ay,y),1);trail.append(float(ay.min()))
    assert task.points==budget and task.diagnostic_points==0 and ax.shape[1]==budget
    if model is not None:assert fingerprint(model.state_dict())==model_before
    return dict(initial_x=initial_x,initial=initial_y,points=torch.cat(points,1),values=ay[:,20:],trail=torch.tensor(trail),
        records=records,pool_hashes=pool_hashes,calls=task.points,utility=float(bounded(initial,ay.min(1).values,scale)[0]),seconds=time.perf_counter()-start)


def contracts():
    if (RUN/'CONTRACTS.json').exists():return
    outputs={}
    for key,method,w in [('A','A',None),('O','O',None),('forceA','F',1.),('forceO','F',0.),('F','F',None),('Fixed','Fixed',None)]:
        outputs[key]=rollout(0,0,0,0,method,24,'prior_fusion_contract',w)
    assert torch.equal(outputs['A']['points'],outputs['forceA']['points'])
    assert torch.equal(outputs['O']['points'],outputs['forceO']['points'])
    assert torch.equal(outputs['F']['points'][:,:1],outputs['Fixed']['points'][:,:1])
    assert len({fingerprint(v['initial_x']) for v in outputs.values()})==1
    save_torch(RUN/'contracts.pt',outputs)
    write_json(RUN/'CONTRACTS.json',dict(calls=144,weight_endpoint_equivalence=True,common_initialization=True,prequential_first_step_fixed_equivalence=True))


def collect(job):
    torch.set_num_threads(1);fid,instance,pop=job;store=CaseStore(RUN/'cases',verify());key=f'f{fid}_i{instance}_p{pop}'
    spec=dict(fid=fid,instance=instance,population=pop)
    with store.lock(key):
        if store.load(key,spec) is not None:return key
        outputs={}
        for method in ('O','Fanalytic'):outputs[f'{method}_shared']=rollout(fid,instance,pop,0,method)
        for seed in range(3):
            for method in ('A','F','Fixed','Fshuffled'):outputs[f'{method}_{seed}']=rollout(fid,instance,pop,seed,method)
        assert len({fingerprint(v['initial_x']) for v in outputs.values()})==1
        assert len({fingerprint(v['initial']) for v in outputs.values()})==1
        assert len({v['pool_hashes'][0] for v in outputs.values()})==1
        store.save(key,spec,dict(outputs=outputs,calls=sum(v['calls'] for v in outputs.values())))
        return key


def interval(delta,main=True):
    # seed, recipe, scale, instance, population
    recipe=delta.reshape(3,12,3,2,2).mean((0,2,3,4));rng=np.random.default_rng(20261010)
    boot=recipe[rng.integers(0,12,(10000,12))].mean(1);alpha=.05/8 if main else .025
    lo,hi=np.quantile(boot,[alpha,1-alpha]);seeds=delta.mean((1,2,3));threshold=.005 if main else .001
    return dict(mean=float(delta.mean()),lower=float(lo),upper=float(hi),seed_means=seeds.tolist(),threshold=threshold,
        passed=bool(delta.mean()>=threshold and lo>0 and (seeds>0).all()),primary=main)


def summarize():
    store=CaseStore(RUN/'cases',verify());arrays={m:np.zeros((3,36,2,2)) for m in METHODS};calls=0;times={m:[] for m in METHODS};weights=[]
    curves={m:np.zeros((3,36,2,2,4)) for m in METHODS}
    for f in range(36):
        for i in range(2):
            for p in range(2):
                v=store.load(f'f{f}_i{i}_p{p}',dict(fid=f,instance=i,population=p));calls+=v['calls']
                for s in range(3):
                    for m in METHODS:
                        row=v['outputs'][f'{m}_shared' if m in ('O','Fanalytic') else f'{m}_{s}'];arrays[m][s,f,i,p]=row['utility']
                        init=row['initial'].min(1).values;scale=row['initial'].std(1)
                        for j,nfe in enumerate((40,100,200,300)):curves[m][s,f,i,p,j]=float(bounded(init,row['trail'][nfe-21:nfe-20],scale)[0])
                        if m not in ('O','Fanalytic') or s==0:times[m].append(row['seconds'])
                        if m=='F':weights.append([r['prior_weight'] for r in row['records']])
    assert calls==604800
    effects={f'F-{m}':interval(arrays['F']-arrays[m],m!='Fixed') for m in ('A','O','Fshuffled','Fanalytic','Fixed')}
    save_torch(RUN/'aggregates.pt',dict(utilities=arrays,curves=curves,weights=np.array(weights)))
    return dict(effects=effects,passed=all(effects['F-'+m]['passed'] for m in ('A','O','Fshuffled','Fanalytic')),
        means={m:float(x.mean()) for m,x in arrays.items()},mean_curves={m:a.mean((0,1,2,3)).tolist() for m,a in curves.items()},
        calls=calls,contract_calls=144,total_calls=calls+144,trajectories=2016,shared_controls=288,
        mean_prior_weight=float(np.mean(weights)),concurrent_seconds_per_trajectory={m:float(np.mean(v)) for m,v in times.items()},
        development_only=True,online_control_is_mean_ranking_not_entire_old_O=True)


def report(r):
    rows=['# 冻结先验的闭环融合','',
        '主门槛全部通过；只支持开发族内闭环结果，尚需独立外部验证。' if r['passed'] else '闭环融合未通过全部主门槛。保留先验预测正面证据，停止此融合原型的权重/结构搜索。','',
        '| 比较 | 终局效用差 | 区间 | 三种子效应 | 通过 |','|---|---:|---|---|---|']
    for name,e in r['effects'].items():rows.append(f"| {name} | {e['mean']:+.6f} | [{e['lower']:+.6f},{e['upper']:+.6f}] | {e['seed_means']} | {e['passed']} |")
    rows+=['','四项主比较使用98.75%区间与0.005实质门槛；F−Fixed为次要95%区间/0.001门槛，不替代主验收。',
        f"20D/300NFE，20初始化＋280次顺序决策，共2016条轨迹、{r['total_calls']:,}次调用（含144契约检查），耗时{r['seconds']/60:.2f}分钟。O/Fanalytic跨三个模型种子共享，没有重复计独立样本。",
        f"F平均先验权重={r['mean_prior_weight']:.4f}。权重按选点前预测、选点后真值更新；不是全域校准或可靠性保证。",
        'A的网络权重冻结但仍读取当前context；O只拟合当前archive；全部方法使用共同候选生成规则和均值排序。O并不是旧完整OnlinePortfolio的原采集策略，因此结果不能替换历史O成绩。',
        'Fanalytic是无预训练的径向先验融合对照；Fshuffled使用打乱源标签的冻结模型；Fixed去掉自适应权重。没有使用旧residual，不把本轮新增方法归于原解锁模型。',
        '只检验现有生成族新实例，先验模型按recipe留出。权重/候选池/预算未根据本轮结果调整。CPU并发秒数包含资源争用，不当作串行速度。']
    (OUT/'CONCLUSIONS.zh-CN.md').write_text('\n'.join(rows)+'\n')


def main():
    torch.set_num_threads(1);RUN.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
    if (RUN/'COMPLETE.json').exists():print('Already complete; no rerun.');return
    start=time.perf_counter()
    if not (RUN/'identity.json').exists():
        parent=json.loads((prior.RUN/'COMPLETE.json').read_text());assert parent['passed']
        check=json.loads((prior.OUT/'VERIFICATION.json').read_text());assert check['prediction_arrays_exactly_reproduced']
        write_json(RUN/'identity.json',dict(version='prior_fusion_v1',sources={p:sha256(ROOT/p) for p in SOURCES},
            prior_identity=sha256(prior.RUN/'identity.json'),models=json.loads((prior.RUN/'MODELS_FROZEN.json').read_text())))
    verify();contracts();available=repeat.old.parent.parent.available_gib();workers=32 if available>=26 else 24 if available>=22 else 16;assert available>=17
    write_json(RUN/'RESOURCES.json',dict(workers=workers,available_gib=available,device='cpu',threads=1))
    with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn')) as pool:
        jobs=[(f,i,p) for f in range(36) for i in range(2) for p in range(2)]
        fs=[pool.submit(collect,j) for j in jobs]
        for n,f in enumerate(as_completed(fs),1):f.result();write_json(RUN/'STATUS.json',dict(status='running',completed=n,total=144,workers=workers))
    r=summarize();r.update(seconds=time.perf_counter()-start,all_workers_joined=True)
    write_json(RUN/'COMPLETE.json',r);write_json(OUT/'RESULTS.json',r);report(r);write_json(RUN/'STATUS.json',dict(status='complete'))
    print(json.dumps(r),flush=True)


if __name__=='__main__':main()
