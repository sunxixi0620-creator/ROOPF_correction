"""Verify paid-history decisions and report the frozen correction experiment."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
import sys,json,argparse,time,zipfile
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import torch
from scipy.spatial.distance import pdist
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import frozen_prior_correction as study
from roopf.experiment_io import CaseStore,sha256,fingerprint,write_json
RUN,OUT=study.RUN,study.OUT

def load_case(job):
    f,s,m=job;spec=dict(fid=f,seed=s,method=m)
    v=CaseStore(RUN/'cases',study.verify()).load(f'f{f}_s{s}_{m}',spec);assert v is not None
    return v

def verify_case(job):
    torch.set_num_threads(1);v=load_case(job);f,s,m=job
    ix,iy=v['initial_x'],v['initial_y'];x=torch.cat((ix,v['points']));y=torch.cat((iy,v['values']))
    assert torch.equal(ix,study.initial(f,'frozen_prior_correction_v1'))
    assert len(x)==600 and v['calls']==600 and v['teacher_calls']==0 and torch.isfinite(y).all()
    assert pdist(x.numpy()).min()>1e-6
    trail=np.minimum.accumulate(y.numpy())[100:];assert np.array_equal(trail,v['trail'])
    g=max(0,float(iy.min()-y.min()))/max(float(iy.std()),1e-8);assert abs(g/(1+g)-v['utility'])<1e-12
    assert v['gp_final_count']==(0 if m=='P' else (600 if m=='O100' else 560))
    errors=[]
    for step in (0,199,499):
        n=100+step;ax,ay=x[:n],y[:n];p,gp=study.make_state(f,s,m,ax,ay)
        assert p.identity==v['prior_identity']
        # Rebuild the exact targets used in the incremental run: initial float32
        # normalization, then scalar normalization and pool-batch prior predictions.
        # Renormalizing all history in float32 would change the saved targets.
        if gp is not None:
            _,initial_gp=study.make_state(f,s,m,ix,iy)
            recorded=v['decisions'][:step]
            appended=recorded[:,5] if m=='Blend' else recorded[:,5]-recorded[:,1]
            gp=study.GP(gp.x,np.r_[initial_gp.y,appended])
        q=study.pool_for(ax,ay.numpy(),f,step,'frozen_prior_correction_v1',600)
        i,pm,mu,sd,score=study.decision(m,p,gp,ax,ay.numpy(),q);old=v['decisions'][step]
        assert i==int(old[0]) and np.array_equal(q[i],v['points'][step].numpy())
        delta=np.abs(np.array([pm[i],mu[i],sd[i],score])-old[1:5]);errors.append(float(delta.max()))
        assert np.allclose([pm[i],mu[i],sd[i],score],old[1:5],rtol=1e-7,atol=1e-8), (job,step,delta.tolist())
        p.verify()
    return dict(max_error=max(errors),prefix=f,initial=fingerprint(dict(x=ix,y=iy)))

def verify_diagnostics():
    store=CaseStore(RUN/'diagnostics',study.verify());error=0.;counts=0
    for fid in range(36):
        v=store.load(f'f{fid}',dict(fid=fid));assert v is not None and v['calls']==228
        for key,r in v['outputs'].items():
            method,seed=key.split('_');p,gp=study.make_state(fid,int(seed),method,v['initial_x'],v['initial_y'])
            _,pm,mu,sd,_=study.decision(method,p,gp,v['initial_x'],v['initial_y'].numpy(),v['query_x'])
            for new,old in ((pm,r['prior']),(mu,r['mean']),(sd,r['std'])):
                assert np.allclose(new,old,atol=1e-10,rtol=1e-10);error=max(error,float(abs(new-old).max()))
            assert np.allclose(r['target'],(v['query_y']-p.center)/p.scale)
            counts+=1
    return dict(recomputed_prediction_sets=counts,max_error=error,extra_objective_calls=0)

def verify(workers):
    study.verify();assert (RUN/'SEARCH_COMPLETE.json').exists();initial={};errors=[];start=time.time()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        fs=[pool.submit(verify_case,j) for j in study.jobs()]
        for n,future in enumerate(as_completed(fs),1):
            r=future.result();initial.setdefault(r['prefix'],set()).add(r['initial']);errors.append(r['max_error'])
            if n%108==0:print(f'verified {n}/648 trajectories',flush=True)
    assert len(initial)==36 and all(len(v)==1 for v in initial.values())
    diag=verify_diagnostics();write_json(OUT/'VERIFICATION.json',dict(trajectories=648,all_budgets_and_trails=True,common_initialization=True,immutable_context=True,
        sampled_decisions_recomputed=1944,max_incremental_vs_dense_score_error=max(errors),diagnostics=diag,original_checkpoints_unchanged=True,
        extra_objective_calls=0,seconds=time.time()-start))

def summarize():
    identity=study.verify();store=CaseStore(RUN/'cases',identity);arr={m:np.zeros((3,36)) for m in study.METHODS};curves={m:np.zeros((3,36,501)) for m in study.METHODS}
    costs={m:[] for m in study.METHODS};calibration={m:[] for m in study.METHODS};calls=0;filehashes={};percase=[]
    for f,s,m in study.jobs():
        name=f'f{f}_s{s}_{m}';v=store.load(name,dict(fid=f,seed=s,method=m));assert v is not None;calls+=v['calls']
        iy=v['initial_y'].numpy();initial=float(iy.min());scale=max(float(iy.std(ddof=1)),1e-8);final=float(min(initial,v['values'].min()))
        g=max(initial-final,0)/scale;u=g/(1+g);assert abs(u-v['utility'])<1e-6
        ys=np.r_[initial,np.minimum.accumulate(np.r_[initial,v['values'].numpy()])[1:]]
        gain=np.maximum(initial-ys,0)/scale;curve=gain/(1+gain)
        slots=[s] if m in study.SEEDED else range(3)
        for seed in slots:arr[m][seed,f]=u;curves[m][seed,f]=curve
        rec=v['decisions'];mse=float(np.mean((rec[:,5]-rec[:,2])**2));nugget=study.NOISE*(.25 if m=='Blend' else 1.)
        cover=None if m=='P' else float(np.mean(abs(rec[:,5]-rec[:,2])<=1.96*np.sqrt(rec[:,3]**2+nugget)))
        calibration[m].append(dict(mse=mse,coverage=cover));costs[m].append(v['seconds'])
        percase.append(dict(fid=f,seed=s,method=m,utility=u,loop_seconds=v['seconds'],prediction_mse=mse,coverage95=cover))
        filehashes[str((RUN/'cases'/f'{name}.pt').relative_to(RUN))]=sha256(RUN/'cases'/f'{name}.pt')
    assert calls==388800
    rng=np.random.default_rng(280931);samples=rng.integers(0,12,(10000,12));contrasts=[]
    for other in ('P','O100','Shuffled','Analytic','O60','Mean','Blend'):
        primary=other in ('P','O100','Shuffled','Analytic');alpha=.05/(2*(4 if primary else 3));threshold=.005 if primary else .001
        delta=arr['F']-arr[other];groups=delta.reshape(3,12,3).mean((0,2));low,high=np.quantile(groups[samples].mean(1),[alpha,1-alpha]);seed=delta.mean(1)
        contrasts.append(dict(control=other,primary=primary,mean=float(delta.mean()),lower=float(low),upper=float(high),threshold=threshold,seed_effects=seed.tolist(),
            passed=bool(delta.mean()>=threshold and low>0 and (seed>0).all())))
    diag=CaseStore(RUN/'diagnostics',identity);diagnostics={m:[] for m in study.METHODS if m!='Mean'}
    for f in range(36):
        v=diag.load(f'f{f}',dict(fid=f))
        for name,r in v['outputs'].items():
            m=name.split('_')[0];err=r['mean']-r['target'];noise=study.NOISE*(.25 if m=='Blend' else 1.)
            diagnostics[m].append(dict(mse=float(np.mean(err**2)),coverage=None if m=='P' else float(np.mean(abs(err)<=1.96*np.sqrt(r['std']**2+noise)))))
    diagmeans={m:dict(mse=float(np.mean([v['mse'] for v in vals])),coverage95=None if m=='P' else float(np.mean([v['coverage'] for v in vals]))) for m,vals in diagnostics.items()}
    result=dict(method_means={m:float(v.mean()) for m,v in arr.items()},contrasts=contrasts,joint_pass=all(c['passed'] for c in contrasts if c['primary']),
        curve_nfe=[100,200,300,400,600],curve_means={m:v.mean((0,1))[[0,100,200,300,500]].tolist() for m,v in curves.items()},
        mean_curve_utility={m:float(v.mean()) for m,v in curves.items()},initial_predictive_diagnostic=diagmeans,
        selected_point_diagnostic={m:dict(mse=float(np.mean([v['mse'] for v in vals])),coverage95=None if m=='P' else float(np.mean([v['coverage'] for v in vals]))) for m,vals in calibration.items()},
        loop_seconds_per_trajectory={m:float(np.mean(v)) for m,v in costs.items()},timing_excludes_initial_model_setup=True,
        calls=dict(search=388800,diagnostic=8208,contracts=416,total=397424),neural_epochs=0,trajectories=648,identity_sha256=sha256(RUN/'identity.json'))
    write_json(OUT/'RESULTS.json',result);write_json(OUT/'rows.json',percase);np.savez_compressed(OUT/'arrays.npz',**{f'utility_{m}':v for m,v in arr.items()},**{f'curve_{m}':v for m,v in curves.items()})
    write_json(OUT/'RAW_FILES.json',filehashes)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8,4.5))
    for m,v in curves.items():ax.plot(np.arange(100,601),v.mean((0,1)),label=m)
    ax.set(xlabel='Real objective evaluations',ylabel='Mean bounded improvement',title='Frozen-prior correction: 20D development experiment');ax.legend(ncol=4);fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(OUT/f'curves.{ext}',dpi=150)
    plt.close(fig)
    lines=['# 固定先验＋在线GP修正：完整搜索结果','',f"600NFE主验收：{'通过' if result['joint_pass'] else '未通过'}。不是原解锁版，也不是旧residual获得验证。",'',
        '20D/600NFE，100点共同初始化，固定40点条件先验；修正模型从另外60点起拟合。O100使用全部观测。所有先验权重冻结，新增神经网络训练0轮。36配置（12配方×3尺度）各一个新实例和初始化，三模型种子；仍属开发证据，不能当作新函数族确认。',
        '648条轨迹、388800次主评估；独立预测诊断8208次、契约416次，总397424次。O100/O60/Analytic共享，不将训练种子复制计为独立运行。','',
        '|方法|600NFE平均有界改善↑|','|---|---:|']
    lines += [f'|{m}|{v:.6f}|' for m,v in result['method_means'].items()]
    lines += ['', '|比较F−控制|差值|调整区间|验收|类型|','|---|---:|---|---|---|']
    lines += [f"|{c['control']}|{c['mean']:+.6f}|[{c['lower']:+.6f},{c['upper']:+.6f}]|{'通过' if c['passed'] else '未通过'}|{'主' if c['primary'] else '次'}|" for c in contrasts]
    lines += ['', '主比较98.75%区间/0.005门槛；次比较98.3333%区间/0.001门槛；均要求三个种子正效应，按12配方聚类。区间不代表跨未见函数族显著性。',
        'P=固定条件先验；F=先验+GP局部误差修正/EI；Mean=F只按均值选点；Shuffled=打乱训练关系的先验；Blend=固定0.5均值混合/相应GP方差缩放；Analytic=径向先验+同样GP修正。',
        'O100/O60均为同核、全量 eligible archive 的固定超参数精确GP，不冒称最强BO。所有方法候选生成规则相同；局部候选中心随轨迹改变，不声称坐标全程一致。', '',
        '|初始化后的独立预测诊断|标准化MSE↓|95%区间覆盖率|','|---|---:|---:|']
    for m,v in diagmeans.items():lines.append(f"|{m}|{v['mse']:.6f}|{'—' if v['coverage95'] is None else format(v['coverage95'],'.1%')}|")
    lines += ['', '诊断查询未参与任何搜索、选模或参数调整。先验不确定性未建模；95%模型区间不等于已校准总置信度。在线已选点覆盖率另存RESULTS，受选择偏差影响。',
        '记录曲线是事先规定的辅助指标，不能用某个有利时刻替代600NFE主结论。任何主门槛失败就停止此候选，不扫描核参数、改40/60划分或补训residual。',
        'loop_seconds是并发环境中的搜索循环时间，排除了初始先验构建和GP拟合，不作为完整串行部署成本。研究总墙钟见SEARCH_COMPLETE/DIAGNOSTICS_COMPLETE。']
    (OUT/'CONCLUSIONS.zh-CN.md').write_text('\n'.join(lines)+'\n');print(json.dumps({k:result[k] for k in ('joint_pass','method_means','contrasts','initial_predictive_diagnostic')},indent=2),flush=True)

def archive():
    assert (OUT/'VERIFICATION.json').exists();study.verify();dest=ROOT/'artifacts/frozen_prior_correction_v1';dest.mkdir(exist_ok=True)
    files=sorted([p for d in (RUN,OUT) for p in d.rglob('*') if p.is_file() and p.suffix!='.lock']+[ROOT/p for p in study.SOURCES]+[Path(__file__)])
    files=sorted(set(files));manifest={str(p.relative_to(ROOT)):sha256(p) for p in files};batches=[files[i:i+100] for i in range(0,len(files),100)];zips={}
    for i,batch in enumerate(batches):
        zpath=dest/f'frozen_prior_correction_v1.part{i+1:03d}.zip'
        with zipfile.ZipFile(zpath,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
            for p in batch:z.write(p,str(p.relative_to(ROOT)))
        with zipfile.ZipFile(zpath) as z:
            assert z.testzip() is None
            for n in z.namelist():
                import hashlib
                assert hashlib.sha256(z.read(n)).hexdigest()==manifest[n]
        zips[zpath.name]=sha256(zpath)
    write_json(dest/'MANIFEST.json',dict(files=manifest,archives=zips,parent_models='artifacts/fewshot_prior_v1',total_calls=397424))
    print(f'Archived and checked {len(files)} files in {len(batches)} parts.',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['verify','report','archive']);p.add_argument('--workers',type=int,default=24);a=p.parse_args()
    if a.phase=='verify':verify(a.workers)
    elif a.phase=='report':summarize()
    else:archive()
