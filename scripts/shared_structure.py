"""Controlled transfer of learned quadratic subspaces; no ROOPF claim or tuning."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
from pathlib import Path
import sys,json,time,multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor,as_completed
import torch
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from roopf.experiment_io import CaseStore,sha256,fingerprint,seed_for,write_json,save_torch
from roopf.online_portfolio import build_online
from scripts import radial_prior_audit as radial
from scripts import label_repeatability as resource_parent
RUN=ROOT/'results/shared_structure_20260930';OUT=ROOT/'docs/revision/shared_structure'
SOURCES=['scripts/shared_structure.py','scripts/radial_prior_audit.py','docs/experiments/PRIOR_STRUCTURE_PROTOCOL.md','roopf/model.py','roopf/online_portfolio.py']
COUNTS=(10,20,40);METHODS=('Learned','Random','Wrong','RadialLinear','OnlineFull','OnlineRidge')


def basis(group):
    g=torch.Generator().manual_seed(seed_for('shared_structure_basis_v1',group))
    return torch.linalg.qr(torch.randn(20,3,generator=g,dtype=torch.float64)).Q


class Task:
    def __init__(self,group,instance,role,mismatch=False):
        self.fun=dict(xlb=-5.,xub=5.);self.points=0;self.mismatch=mismatch
        g=torch.Generator().manual_seed(seed_for('shared_structure_task_v1',group,instance,role))
        B=basis(group);raw=torch.randn(20,3,generator=g,dtype=torch.float64);C=torch.linalg.qr(raw-B@(B.T@raw)).Q
        self.params=dict(B=B,C=C,c=6*torch.rand(20,generator=g,dtype=torch.float64)-3,
            lam=.5+torch.rand(3,generator=g,dtype=torch.float64),scale=.8+.4*torch.rand((),generator=g,dtype=torch.float64),
            bias=2*torch.rand((),generator=g,dtype=torch.float64)-1)
    def calfitness(self,x):
        self.points+=x.numel()//20;p=self.params
        if self.mismatch:x=torch.roll(x,7,-1)
        z=x-p['c'];y=p['scale']*((((z@p['B']).square())*p['lam']).mean(-1)+.2*(z@p['C']).square().mean(-1)+.01*z.square().mean(-1))+p['bias']
        assert torch.isfinite(y).all();return y


def qfeatures(x):
    ii,jj=torch.triu_indices(x.shape[-1],x.shape[-1]);return x[...,ii]*x[...,jj]


def verify():
    identity=json.loads((RUN/'identity.json').read_text())
    for p,h in identity['sources'].items():assert sha256(ROOT/p)==h
    return identity


def contracts():
    if (RUN/'CONTRACTS.json').exists():return
    g=torch.Generator().manual_seed(20261011);x=10*torch.rand(64,20,generator=g,dtype=torch.float64)-5
    a=Task(0,0,'contracts');b=Task(0,0,'contracts',True)
    ya=a.calfitness(torch.roll(x,7,-1));yb=b.calfitness(x);assert torch.equal(ya,yb)
    B=a.params['B'];C=a.params['C'];assert torch.allclose(B.T@C,torch.zeros(3,3,dtype=B.dtype),atol=1e-12)
    # Recover a known full quadratic matrix after removing intercept and linear terms.
    x2=torch.randn(12,80,20,generator=g,dtype=torch.float64)
    H=B@torch.diag(torch.tensor([.5,1.,1.5],dtype=B.dtype))@B.T
    linear=torch.randn(12,20,generator=g,dtype=torch.float64);bias=torch.randn(12,1,generator=g,dtype=torch.float64)
    y2=torch.einsum('bni,ij,bnj->bn',x2,H,x2)+(x2*linear[:,None]).sum(-1)+bias
    recovered=fit_subspace(x2*5,y2);assert torch.linalg.norm(recovered['B']@recovered['B'].T-B@B.T)<1e-4
    # All numerical test arrays are synthetic algebra; only the task contract queried f.
    save_torch(RUN/'contracts.pt',dict(x=x,matched_permuted=ya,mismatch=yb,true_H=H,recovered=recovered))
    write_json(RUN/'CONTRACTS.json',dict(calls=a.points+b.points,coordinate_permutation_exact=True,orthogonal_residual_subspace=True,known_quadratic_recovery=True))


def fit_subspace(x,y):
    # Receives observed coordinates/labels only, never the simulator's true B/C/c.
    z=x/5;quad=qfeatures(z);linear=torch.cat((torch.ones_like(z[...,:1]),z),-1)
    Q=torch.linalg.qr(linear,mode='reduced').Q
    rq=quad-Q@(Q.transpose(1,2)@quad);ry=y[:,:,None]-Q@(Q.transpose(1,2)@y[:,:,None])
    gram=(rq.transpose(1,2)@rq).sum(0);rhs=(rq.transpose(1,2)@ry).sum(0)
    regularization=1e-5*float(gram.diag().mean());coeff=torch.linalg.solve(gram+regularization*torch.eye(210,dtype=x.dtype),rhs)[:,0]
    ii,jj=torch.triu_indices(20,20);H=torch.zeros(20,20,dtype=x.dtype);H[ii,jj]=coeff/torch.where(ii==jj,1.,2.)
    H=H+H.T-torch.diag(H.diag());eigenvalues,U=torch.linalg.eigh(H);learned=U[:,-3:]
    return dict(B=learned,H=H,eigenvalues=eigenvalues,regularization=regularization,rank=3)


def source(job):
    torch.set_num_threads(1);group,seed=job;store=CaseStore(RUN/'source',verify());spec=dict(group=group,seed=seed);key=f'g{group}_s{seed}'
    with store.lock(key):
        if store.load(key,spec) is not None:return key
        xs=[];ys=[];calls=0;role=f'shared_source_v1_{seed}'
        for t in range(64):
            task=Task(group,t,role);g=torch.Generator().manual_seed(seed_for('shared_source_points',group,seed,t))
            x=10*torch.rand(80,20,generator=g,dtype=torch.float64)-5;y=task.calfitness(x);calls+=task.points;xs.append(x);ys.append(y)
        x,y=torch.stack(xs),torch.stack(ys);before=time.perf_counter();model=fit_subspace(x,y)
        store.save(key,spec,dict(x=x,y=y,model=model,calls=calls,fit_seconds=time.perf_counter()-before))
        return key


def frozen_models():
    marker=RUN/'MODELS_FROZEN.json'
    if marker.exists():
        frozen=json.loads(marker.read_text())
        for p,h in frozen.items():assert sha256(RUN/p)==h
        return
    store=CaseStore(RUN/'source',verify());models={}
    for group in range(12):
        for seed in range(3):
            v=store.load(f'g{group}_s{seed}',dict(group=group,seed=seed));assert v is not None
            path=RUN/f'model_g{group}_s{seed}.pt';save_torch(path,v['model']);models[path.name]=sha256(path)
    write_json(marker,models)


def predict_ridge(cx,cy,qx,kind,B=None):
    z=cx/5;w=qx/5
    if B is not None:z=z@B;w=w@B
    def features(x):
        return torch.cat((torch.ones_like(x[...,:1]),x,x.square().mean(-1,keepdim=True) if kind=='radial' else qfeatures(x)),-1)
    a,b=features(z),features(w);d=a.shape[-1];ridge=torch.eye(d,dtype=a.dtype)*.01;ridge[0,0]=1e-6
    beta=torch.linalg.solve(a.transpose(1,2)@a+ridge,a.transpose(1,2)@cy[:,:,None])
    return (b@beta).squeeze(-1)


def collect(job):
    torch.set_num_threads(1);group,mismatch=job;store=CaseStore(RUN/'target',verify());spec=dict(group=group,mismatch=mismatch);key=f'g{group}_m{int(mismatch)}'
    with store.lock(key):
        if store.load(key,spec) is not None:return key
        frozen=json.loads((RUN/'MODELS_FROZEN.json').read_text());models={}
        for j in (group,(group+1)%12):
            for seed in range(3):
                name=f'model_g{j}_s{seed}.pt';assert sha256(RUN/name)==frozen[name];models[j,seed]=torch.load(RUN/name,weights_only=False)['B']
        xs=[];ys=[];calls=0
        for t in range(32):
            task=Task(group,t,'shared_target_v1',mismatch);g=torch.Generator().manual_seed(seed_for('shared_target_points',group,t))
            cx=10*torch.rand(40,20,generator=g,dtype=torch.float64)-5
            box=10*torch.rand(256,20,generator=g,dtype=torch.float64)-5
            sphere=torch.randn(256,20,generator=g,dtype=torch.float64);sphere=4.5*sphere/torch.linalg.vector_norm(sphere,dim=1,keepdim=True)
            assert torch.allclose(sphere.square().sum(1),torch.full((256,),20.25,dtype=torch.float64),atol=1e-12)
            x=torch.cat((cx,box,sphere));y=task.calfitness(x);calls+=task.points;xs.append(x);ys.append(y)
        x,y=torch.stack(xs),torch.stack(ys);predictions={};bounds=type('Bounds',(),{'fun':dict(xlb=-5.,xub=5.)})()
        surrogate=build_online(20,300).surrogate
        for n in COUNTS:
            cx=x[:,:n];rawcy=y[:,:n];mean=rawcy.mean(1,keepdim=True);std=rawcy.std(1,keepdim=True,unbiased=False).clamp_min(1e-8);cy=(rawcy-mean)/std;qx=x[:,40:]
            controls={}
            controls['RadialLinear']=predict_ridge(cx,cy,qx,'radial')
            controls['OnlineFull']=predict_ridge(cx,cy,qx,'full')
            with torch.no_grad():mu,_=surrogate.predict(cx,rawcy,qx,bounds)
            controls['OnlineRidge']=(mu-mean)/std
            for seed in range(3):
                g=torch.Generator().manual_seed(seed_for('shared_random_subspace',group,seed));random=torch.linalg.qr(torch.randn(20,3,generator=g,dtype=torch.float64)).Q
                methods=dict(controls)
                for name,B in [('Learned',models[group,seed]),('Wrong',models[(group+1)%12,seed]),('Random',random)]:methods[name]=predict_ridge(cx,cy,qx,'subspace',B)
                for name,pred in methods.items():assert torch.isfinite(pred).all();predictions[f'{name}_{n}_{seed}']=pred
        store.save(key,spec,dict(x=x,y=y,predictions=predictions,calls=calls,context_counts=COUNTS))
        return key


def summarize():
    store=CaseStore(RUN/'target',verify())
    arrays={m:{metric:np.zeros((2,2,3,3,12)) for metric in ('mse','regret','utility')} for m in METHODS};calls=0
    for group in range(12):
        for mi in range(2):
            row=store.load(f'g{group}_m{mi}',dict(group=group,mismatch=bool(mi)));assert row is not None;calls+=row['calls'];x,y=row['x'],row['y']
            for ci,n in enumerate(COUNTS):
                cy=y[:,:n];mean=cy.mean(1,keepdim=True);std=cy.std(1,keepdim=True,unbiased=False).clamp_min(1e-8);truth=(y[:,40:]-mean)/std
                for pi in range(2):
                    target=truth[:,pi*256:(pi+1)*256]
                    for seed in range(3):
                        for method in METHODS:
                            pred=row['predictions'][f'{method}_{n}_{seed}'][:,pi*256:(pi+1)*256]
                            chosen=target[torch.arange(32),pred.argmin(1)];regret=chosen-target.min(1).values
                            improvement=(((cy.min(1).values-mean[:,0])/std[:,0])-chosen).clamp_min(0)
                            arrays[method]['mse'][mi,pi,seed,ci,group]=float((pred-target).square().mean())
                            arrays[method]['regret'][mi,pi,seed,ci,group]=float(regret.mean())
                            arrays[method]['utility'][mi,pi,seed,ci,group]=float((improvement/(1+improvement)).mean())
    rng=np.random.default_rng(20261011);ix=rng.integers(0,12,(10000,12));effects={};mismatch_effects={}
    for mi in range(2):
        for pi,pool in enumerate(('box','shell')):
            for control in METHODS[1:]:
                for metric in ('mse','regret'):
                    a,b=arrays['Learned'][metric][mi,pi],arrays[control][metric][mi,pi]
                    delta=(b-a)/np.maximum(b,1e-8) if metric=='mse' else b-a
                    groups=delta.mean((0,1));boot=groups[ix].mean(1);lo,hi=np.quantile(boot,[.00125,.99875]);seeds=delta.mean((1,2));threshold=.05 if metric=='mse' else .01
                    stat=dict(mean=float(delta.mean()),lower=float(lo),upper=float(hi),seed_means=seeds.tolist(),by_context=delta.mean((0,2)).tolist(),
                        threshold=threshold,passed=bool(delta.mean()>=threshold and lo>0 and (seeds>0).all()))
                    (effects if mi==0 else mismatch_effects)[f'{pool}:{metric}:Learned-{control}']=stat
    source_store=CaseStore(RUN/'source',verify());source_calls=0;distances=[]
    for group in range(12):
        for seed in range(3):
            row=source_store.load(f'g{group}_s{seed}',dict(group=group,seed=seed));source_calls+=row['calls'];B=row['model']['B'];true=basis(group)
            distances.append(float(torch.linalg.norm(B@B.T-true@true.T)/6**.5))
    assert calls==423936 and source_calls==184320
    save_torch(RUN/'aggregates.pt',arrays)
    return dict(effects=effects,mismatch_effects=mismatch_effects,passed=all(v['passed'] for v in effects.values()),
        passed_comparisons=sum(v['passed'] for v in effects.values()),source_calls=source_calls,target_calls=calls,contract_calls=128,total_calls=calls+source_calls+128,
        mean_projection_distance=float(np.mean(distances)),projection_distances=distances,
        means={m:{metric:array.mean((2,3,4)).tolist() for metric,array in values.items()} for m,values in arrays.items()},
        controlled_positive_control_only=True,no_full_search_or_component_necessity_claim=True)


def report(r):
    lines=['# 共享方向结构的可迁移性试验','',
        '预设20项matched比较全部通过；仅证明受控共享方向分布的可迁移信息，不是通用融合或原ROOPF成功。' if r['passed'] else f"预设20项matched比较通过{r['passed_comparisons']}项，联合门槛未通过；不依据结果修改秩、正则、任务分布或训练量。",'',
        '| matched主比较 | 平均差 | 99.75%区间 | 通过 |','|---|---:|---|---|']
    for name,e in r['effects'].items():lines.append(f"| {name} | {e['mean']:+.5f} | [{e['lower']:+.5f},{e['upper']:+.5f}] | {e['passed']} |")
    lines+=['', 'MSE项是相对降低，regret项是候选池归一化选择遗憾的绝对降低；不是全局最优regret或终局效用。',
        f"学到的子空间投影误差平均{r['mean_projection_distance']:.6f}，仅用于事后核验；算法未读取真实子空间。",
        '12个独立方向组共享同一解析生成结构，不能称12个不同函数族。三个源数据种子各自闭式拟合，无神经网络epoch、无测试调参。',
        f"目标context10/20/40；同半径候选均在半径4.5球面。源训练{r['source_calls']:,}次、目标标签{r['target_calls']:,}次、契约128次，共{r['total_calls']:,}次；候选标签采集全部算作研究成本。",
        '', '| mismatch诊断 | 平均差 | 99.75%区间 |','|---|---:|---|']
    for name,e in r['mismatch_effects'].items():lines.append(f"| {name} | {e['mean']:+.5f} | [{e['lower']:+.5f},{e['upper']:+.5f}] |")
    lines+=['','mismatch通过固定坐标置换破坏源目标方向对应，目标分布的旋转对称边际仍相同。失配条件不得隐藏，也不用于更换matched主验收。',
        '没有完整A/O/F/FR搜索，不自动把条件化预测器叫作纯离线模型；下一项须单独定义在线修正及失配机制后冻结验证。',
        f"本轮耗时约{r['seconds']/60:.2f}分钟，源拟合与分组评测使用单线程CPU进程并行，实际资源见RESOURCES.json。"]
    (OUT/'CONCLUSIONS.zh-CN.md').write_text('\n'.join(lines)+'\n')


def main():
    torch.set_num_threads(1);RUN.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
    if (RUN/'COMPLETE.json').exists():print('Already complete; no experiments relaunched.');return
    start=time.perf_counter()
    if not (RUN/'identity.json').exists():write_json(RUN/'identity.json',dict(version='shared_structure_v1',sources={p:sha256(ROOT/p) for p in SOURCES}))
    verify();contracts();available=resource_parent.old.parent.parent.available_gib();workers=32 if available>=26 else 24 if available>=22 else 16;assert available>=17
    write_json(RUN/'RESOURCES.json',dict(source_workers=workers,target_workers=min(workers,24),available_gib=available,threads=1,device='cpu'))
    with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn')) as pool:
        fs=[pool.submit(source,(g,s)) for g in range(12) for s in range(3)]
        for n,f in enumerate(as_completed(fs),1):f.result();write_json(RUN/'STATUS.json',dict(stage='source',completed=n,total=36))
    frozen_models()
    with ProcessPoolExecutor(max_workers=min(workers,24),mp_context=mp.get_context('spawn')) as pool:
        fs=[pool.submit(collect,(g,bool(m))) for g in range(12) for m in range(2)]
        for n,f in enumerate(as_completed(fs),1):f.result();write_json(RUN/'STATUS.json',dict(stage='target',completed=n,total=24))
    r=summarize();r.update(seconds=time.perf_counter()-start,all_workers_joined=True)
    write_json(RUN/'COMPLETE.json',r);write_json(OUT/'RESULTS.json',r);report(r);write_json(RUN/'STATUS.json',dict(stage='complete'))
    print(json.dumps(r),flush=True)


if __name__=='__main__':main()
