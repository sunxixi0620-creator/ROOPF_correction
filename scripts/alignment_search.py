"""LOO-selected source representation/full-space models, matched closed loop."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
from pathlib import Path
import sys,json,time,multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor,as_completed
import torch
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import shared_structure as parent
from scripts.unified_revision import bounded
from roopf.experiment_io import CaseStore,sha256,fingerprint,seed_for,write_json,save_torch
RUN=ROOT/'results/alignment_search_20260930';OUT=ROOT/'docs/revision/alignment_search'
SOURCES=['scripts/alignment_search.py','docs/experiments/ALIGNMENT_SEARCH_PROTOCOL.md']
METHODS=('T','O','S','Fixed','RadialS','WrongS')


def features(x,B=None,radial=False):
    z=x/5
    if B is not None:z=z@B
    return torch.cat((z,z.square().mean(-1,keepdim=True) if radial else parent.qfeatures(z)),-1)


def fit(a,y):
    xm=a.mean(0);ym=y.mean();a=a-xm;z=y-ym;n,d=a.shape;lam=.01
    if n<=d:
        inv=torch.cholesky_inverse(torch.linalg.cholesky(a@a.T+lam*torch.eye(n,dtype=a.dtype)))
        beta=a.T@(inv@z);denom=lam*inv.diag()-1/n
    else:
        inv=torch.cholesky_inverse(torch.linalg.cholesky(a.T@a+lam*torch.eye(d,dtype=a.dtype)))
        beta=inv@(a.T@z);denom=1-1/n-(a@inv*a).sum(1)
    assert denom.min()>1e-10
    residual=(z-a@beta)/denom
    return dict(beta=beta,xmean=xm,ymean=ym,loo=residual.square().mean(),residual=residual)


def predict(model,x):return (x-model['xmean'])@model['beta']+model['ymean']


def verify():
    identity=json.loads((RUN/'identity.json').read_text())
    for p,h in identity['sources'].items():assert sha256(ROOT/p)==h
    parent.verify();assert sha256(parent.RUN/'identity.json')==identity['parent_identity']
    for p,h in identity['models'].items():assert sha256(parent.RUN/p)==h
    return identity


@torch.no_grad()
def rollout(group,t,mismatch,seed,method,budget=300,role='alignment_search_v1',force=None):
    task=parent.Task(group,t,role,mismatch)
    gen=torch.Generator().manual_seed(seed_for(role,'initial',group,t));x=10*torch.rand(20,20,generator=gen,dtype=torch.float64)-5;y=task.calfitness(x)
    initial_x=x.clone();initial_y=y.clone();initial=y.min();scale=y.std().clamp_min(1e-8)
    use_t=method!='O';use_o=method!='T';B=None
    if use_t and method!='RadialS':B=torch.load(parent.RUN/f'model_g{(group+1)%12 if method=="WrongS" else group}_s{seed}.pt',weights_only=False)['B']
    records=[];points=[];values=[];trail=[];start=time.perf_counter()
    for step in range(budget-20):
        n=20+step;remaining=(budget-n)/budget;g=torch.Generator().manual_seed(seed_for(role,'pool',group,t,step))
        q=torch.cat((10*torch.rand(64,20,generator=g,dtype=torch.float64)-5,
            (x[y.argmin()]+10*(.05+.35*remaining)*torch.randn(64,20,generator=g,dtype=torch.float64)).clamp(-5,5)))
        before=task.points
        if use_t:
            tm=fit(features(x,B,method=='RadialS'),y);tp=predict(tm,features(q,B,method=='RadialS'))
        if use_o:
            om=fit(features(x),y);op=predict(om,features(q))
        if method=='T':weight=1.;score=tp
        elif method=='O':weight=0.;score=op
        else:
            weight=.5 if method=='Fixed' else float(tm['loo']<om['loo'])
            if force is not None:weight=force
            score=weight*tp+(1-weight)*op
        assert task.points==before
        score=score.clone();score[torch.cdist(q,x).min(1).values<1e-8]=torch.inf
        index=int(score.argmin());assert torch.isfinite(score[index]);chosen=q[index:index+1];value=task.calfitness(chosen)
        records.append(dict(nfe=n+1,weight=weight,loo_t=float(tm['loo']) if use_t else None,loo_o=float(om['loo']) if use_o else None,
            pred_t=float(tp[index]) if use_t else None,pred_o=float(op[index]) if use_o else None,value=float(value[0])))
        x=torch.cat((x,chosen));y=torch.cat((y,value));points.append(chosen);values.append(value);trail.append(y.min())
    assert task.points==budget and len(x)==budget
    return dict(initial_x=initial_x,initial=initial_y,points=torch.cat(points),values=torch.cat(values),trail=torch.stack(trail),records=records,
        utility=float(bounded(initial,y.min(),scale)),calls=task.points,seconds=time.perf_counter()-start)


def contracts():
    if (RUN/'CONTRACTS.json').exists():return
    g=torch.Generator().manual_seed(20261012);errors=[]
    for d in (5,20):
        a=torch.randn(12,d,generator=g,dtype=torch.float64);y=torch.randn(12,generator=g,dtype=torch.float64);model=fit(a,y);direct=[]
        for i in range(12):
            mask=torch.arange(12)!=i;other=fit(a[mask],y[mask]);direct.append(y[i]-predict(other,a[i]))
        assert torch.allclose(torch.stack(direct),model['residual'],atol=1e-9,rtol=1e-9);errors.append(float((torch.stack(direct)-model['residual']).abs().max()))
    outputs={}
    for name,method,force in [('T','T',None),('O','O',None),('forced_T','S',1.),('forced_O','S',0.)]:outputs[name]=rollout(0,0,False,0,method,24,'alignment_contract',force)
    assert torch.equal(outputs['T']['points'],outputs['forced_T']['points']) and torch.equal(outputs['O']['points'],outputs['forced_O']['points'])
    assert len({fingerprint(v['initial_x']) for v in outputs.values()})==1
    save_torch(RUN/'contracts.pt',outputs);write_json(RUN/'CONTRACTS.json',dict(calls=96,loo_bruteforce_max_errors=errors,forced_endpoints_exact=True))


def collect(job):
    torch.set_num_threads(1);group,t,mismatch=job;spec=dict(group=group,task=t,mismatch=mismatch);store=CaseStore(RUN/'cases',verify());key=f'g{group}_t{t}_m{int(mismatch)}'
    with store.lock(key):
        if store.load(key,spec) is not None:return key
        outputs={}
        for method in ('O','RadialS'):outputs[f'{method}_shared']=rollout(group,t,mismatch,0,method)
        for seed in range(3):
            for method in ('T','S','Fixed','WrongS'):outputs[f'{method}_{seed}']=rollout(group,t,mismatch,seed,method)
        assert len({fingerprint(v['initial_x']) for v in outputs.values()})==1
        store.save(key,spec,dict(outputs=outputs,calls=sum(v['calls'] for v in outputs.values())));return key


def interval(delta,noninferior=False,secondary=False):
    # source seed x group x task x condition, or condition-specific without final axis.
    groups=delta.mean(tuple(i for i in range(delta.ndim) if i!=1));rng=np.random.default_rng(20261012)
    boot=groups[rng.integers(0,12,(10000,12))].mean(1);alpha=.025 if secondary else .005;lo,hi=np.quantile(boot,[alpha,1-alpha]);seeds=delta.mean(tuple(range(1,delta.ndim)))
    threshold=-.005 if noninferior else .001 if secondary else .005
    passed=(lo>threshold and (seeds>=threshold).all()) if noninferior else (delta.mean()>=threshold and lo>0 and (seeds>0).all())
    return dict(mean=float(delta.mean()),lower=float(lo),upper=float(hi),seed_means=seeds.tolist(),threshold=threshold,passed=bool(passed),noninferiority=noninferior,secondary=secondary)


def summarize():
    store=CaseStore(RUN/'cases',verify());arrays={m:np.zeros((3,12,4,2)) for m in METHODS};curves={m:np.zeros((3,12,4,2,5)) for m in METHODS};weights={m:[] for m in ('S','WrongS','RadialS')};calls=0
    for group in range(12):
        for t in range(4):
            for mi in range(2):
                v=store.load(f'g{group}_t{t}_m{mi}',dict(group=group,task=t,mismatch=bool(mi)));assert v is not None;calls+=v['calls']
                for seed in range(3):
                    for method in METHODS:
                        row=v['outputs'][f'{method}_shared' if method in ('O','RadialS') else f'{method}_{seed}'];arrays[method][seed,group,t,mi]=row['utility'];init=row['initial'].min();scale=row['initial'].std()
                        for j,n in enumerate((20,40,100,200,300)):curves[method][seed,group,t,mi,j]=0. if n==20 else float(bounded(init,row['trail'][n-21],scale))
                        if method in weights and (method!='RadialS' or seed==0):weights[method].append(dict(condition=mi,weights=[r['weight'] for r in row['records']]))
    assert calls==403200
    effects={f'S-{m}':interval(arrays['S']-arrays[m]) for m in ('T','O','RadialS','WrongS')}
    effects['mismatch:S-O_noninferiority']=interval(arrays['S'][...,1]-arrays['O'][...,1],noninferior=True)
    effects['S-Fixed']=interval(arrays['S']-arrays['Fixed'],secondary=True)
    save_torch(RUN/'aggregates.pt',dict(utilities=arrays,curves=curves,weights=weights))
    return dict(effects=effects,passed=all(e['passed'] for e in effects.values() if not e['secondary']),
        condition_means={m:a.mean((0,1,2)).tolist() for m,a in arrays.items()},mean_curves={m:a.mean((0,1,2)).tolist() for m,a in curves.items()},
        transfer_selection_rate={m:[float(np.mean([r['weights'] for r in rows if r['condition']==mi])) for mi in range(2)] for m,rows in weights.items()},
        calls=calls,contract_calls=96,total_calls=calls+96,trajectories=1344,controlled_only=True,not_original_A_O_F=True)


def report(r):
    rows=['# 共享结构失配下的闭环模型选择','',
        '五项主验收全部通过，仅支持本受控分布。' if r['passed'] else '主验收未全部通过，停止该LOO选择规则的局部搜索。','',
        '| 比较 | 终局效用差 | 区间 | 通过 |','|---|---:|---|---|']
    for name,e in r['effects'].items():rows.append(f"| {name} | {e['mean']:+.6f} | [{e['lower']:+.6f},{e['upper']:+.6f}] | {e['passed']} |")
    rows+=['','五主比较99%区间；S−Fixed为次要95%区间。mismatch非劣界为−0.005，其它主比较实质门槛+0.005。',
        'T含冻结源表示和在线系数拟合，O为目标全空间二次回归，S用精确留一误差选择一个模型；不是原纯离线anchor/原完整ROOPF比较。',
        f"共1344条20D/300NFE轨迹，20点初始化，{r['total_calls']:,}次目标调用（包括96契约调用），无额外目标查询用于模型选择。耗时{r['seconds']/60:.2f}分钟。",
        '', '| 方法 | matched均值 | mismatch均值 |','|---|---:|---:|']
    for m,a in r['condition_means'].items():rows.append(f'| {m} | {a[0]:.6f} | {a[1]:.6f} |')
    rows+=['',f"选择迁移预测器的比例（matched/mismatch）：{r['transfer_selection_rate']}。比例不等于收益贡献或真实相关性识别准确率。",
        '匹配/失配50/50是预定受控评测分布，不声称真实应用比例。已知二次形式、秩3假设和合成共享子空间使本研究成为机制正对照，不是外部泛化或方法新颖性的证据。',
        '没有调整正则、秩、选模阈值和主预算；原解锁checkpoint、原神经先验和旧residual未改变。']
    (OUT/'CONCLUSIONS.zh-CN.md').write_text('\n'.join(rows)+'\n')


def main():
    torch.set_num_threads(1);RUN.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
    if (RUN/'COMPLETE.json').exists():print('Already complete; no rerun.');return
    start=time.perf_counter()
    if not (RUN/'identity.json').exists():
        assert json.loads((parent.RUN/'COMPLETE.json').read_text())['passed']
        assert json.loads((parent.OUT/'VERIFICATION.json').read_text())['all_predictions_recomputed']
        write_json(RUN/'identity.json',dict(version='alignment_search_v1',sources={p:sha256(ROOT/p) for p in SOURCES},parent_identity=sha256(parent.RUN/'identity.json'),models=json.loads((parent.RUN/'MODELS_FROZEN.json').read_text())))
    verify();contracts();available=parent.resource_parent.old.parent.parent.available_gib();workers=32 if available>=26 else 24 if available>=22 else 16;assert available>=17
    write_json(RUN/'RESOURCES.json',dict(workers=workers,available_gib=available,threads=1,device='cpu'))
    with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn')) as pool:
        fs=[pool.submit(collect,(g,t,bool(m))) for g in range(12) for t in range(4) for m in range(2)]
        for n,f in enumerate(as_completed(fs),1):f.result();write_json(RUN/'STATUS.json',dict(stage='running',completed=n,total=96))
    r=summarize();r.update(seconds=time.perf_counter()-start,all_workers_joined=True);write_json(RUN/'COMPLETE.json',r);write_json(OUT/'RESULTS.json',r);report(r)
    write_json(RUN/'STATUS.json',dict(stage='complete'));print(json.dumps(r),flush=True)


if __name__=='__main__':main()
