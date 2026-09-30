"""Read-only cold-start replay, censored endpoint analysis and artifact export."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,argparse,time,zipfile,hashlib
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import torch
from scipy.spatial.distance import pdist
from scipy.stats import qmc
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import cold_start as st
from roopf.experiment_io import CaseStore,sha256,write_json,seed_for,fingerprint
RUN,OUT=st.RUN,st.OUT

def load(j):return CaseStore(RUN/'cases',st.verify()).load('_'.join(map(str,j)),dict(job=j))
def verify_case(j):
    torch.set_num_threads(1);v=load(j);f,inst,seed,m=j;x,y=v['x'],v['y'];assert v['calls']==600 and v['teacher_calls']==0
    assert x.shape==(600,20) and y.shape==(600,) and torch.isfinite(y).all() and pdist(x.numpy()).min()>1e-6
    assert torch.equal(x[:10],st.inputs(f,inst,'cold_start_search_v1',10))
    p=st.base.Prior(f,seed,'zero' if m in ('O','S') else 'correct',x[:10],y[:10]);assert p.identity==v['prior_identity']
    if m=='W':assert v['switch']['count']==40 and np.array_equal(v['switch']['targets'],(y[:40].numpy().astype(float)-p.center)/p.scale)
    assert np.array_equal(v['trace'][:,6],np.array([float(m in ('P','F') or (m=='W' and n<40)) for n in range(10,600)]))
    maxerr=0
    for step in (0,29,30,289,589):
        n=10+step;active=m in ('P','F') or(m=='W' and n<40);gp=None
        if m!='P':
            if m=='W' and n>=40:targets=(y[:n].numpy().astype(float)-p.center)/p.scale
            else:
                initial=(y[:10].numpy().astype(float)-p.center)/p.scale-(p(x[:10].numpy()) if active else 0)
                r=v['trace'][:step];targets=np.r_[initial,r[:,5]-r[:,1]]
            gp=st.base.GP(x[:n].numpy(),targets)
        if m=='S' and n<40:
            sobol=qmc.Sobol(20,scramble=True,seed=seed_for('cold_start_search_v1','sobol',f,inst)%(2**32)).random_base2(5)[:30].astype('float32')*10-5;q=sobol[step:step+1]
        else:q=st.pool(x[:n],y[:n].numpy(),f,inst,'cold_start_search_v1',step)
        values=st.choose(q,x[:n],y[:n].numpy(),p,gp,active,m);assert values[0]==int(v['trace'][step,0]);assert np.array_equal(q[values[0]],x[n].numpy())
        old=v['trace'][step,:5];error=float(abs(np.array(values)-old).max());maxerr=max(maxerr,error)
        assert np.allclose(values,old,rtol=1e-7,atol=1e-8),(j,step,error)
    p.verify();return dict(error=maxerr,task=(f,inst),initial=fingerprint(dict(x=x[:10],y=y[:10])))
def verify(workers):
    st.verify();assert (RUN/'SEARCH_COMPLETE.json').exists();errors=[];initial={}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for n,r in enumerate(ex.map(verify_case,st.jobs()),1):
            errors.append(r['error']);initial.setdefault(r['task'],set()).add(r['initial'])
            if n%132==0:print(f'verified {n}/792',flush=True)
    assert len(initial)==72 and all(len(v)==1 for v in initial.values())
    # Recalculate diagnostics only from saved observations, never objective calls.
    ds=CaseStore(RUN/'diagnostics',st.verify());sets=0
    for f in range(36):
        v=ds.load('f'+str(f),dict(job=f));assert v['calls']==168;x,y=v['x'],v['y']
        for k,r in v['metrics'].items():
            ns,ss,m=k.split('_');n,s=int(ns),int(ss);p=st.base.Prior(f,s,'shuffled' if m=='Shuffled' else ('zero' if m=='O' else 'correct'),x[:n],y[:n]);mu=p(x[40:].numpy())
            if m in ('O','F'):
                gp=st.base.GP(x[:n].numpy(),(y[:n].numpy().astype(float)-p.center)/p.scale-p(x[:n].numpy()));mu+=gp.predict(x[40:].numpy())[0]
            assert np.allclose(mu,r['prediction'],rtol=1e-10,atol=1e-10);sets+=1
    write_json(OUT/'VERIFICATION.json',dict(cases=792,decisions=3960,diagnostic_predictions=sets,max_score_error=max(errors),common_initialization=True,budget_and_unique=True,switch_rebuild=True,original_unchanged=True,extra_calls=0))

def report():
    st.verify();OUT.mkdir(parents=True,exist_ok=True);curves={m:np.zeros((3,36,2,591)) for m in st.METHODS};times={m:np.zeros((3,36,2,3)) for m in st.METHODS};success={m:np.zeros((3,36,2,3)) for m in st.METHODS};timing={m:[] for m in st.METHODS};calls=0
    for j in st.jobs():
        f,i,s,m=j;v=load(j);y=v['y'].numpy().astype(float);init=y[:10];scale=max(float(init.std(ddof=1)),1e-8);g=np.maximum(0,init.min()-np.minimum.accumulate(y)[9:])/scale;u=g/(1+g)
        assert len(u)==591;ts=[];ss=[]
        for goal in (.25,.5,1.):
            hits=np.flatnonzero(g[:291]>=goal);ts.append(int(hits[0]+10) if len(hits) else 301);ss.append(bool(len(hits)))
        slots=[s] if m in ('P','F','W') else range(3)
        for k in slots:curves[m][k,f,i]=u;times[m][k,f,i]=ts;success[m][k,f,i]=ss
        timing[m].append(v['seconds']);calls+=v['calls']
    assert calls==475200
    rng=np.random.default_rng(202609301);idx=rng.integers(0,12,(10000,12))
    def estimate(delta):
        groups=delta.reshape(3,12,3,2).mean((0,2,3));ci=np.quantile(groups[idx].mean(1),[.0125,.9875]);return dict(mean=float(delta.mean()),lower=float(ci[0]),upper=float(ci[1]),seed_means=delta.mean((1,2)).tolist())
    comparisons=[]
    for other in ('O','S','P','F'):
        eff=estimate(times[other][...,1]-times['W'][...,1]);end=estimate(curves['W'][...,-1]-curves[other][...,-1]);att=float((success['W'][...,1]-success[other][...,1]).mean())
        passed=eff['mean']>=5 and eff['lower']>0 and min(eff['seed_means'])>0 and end['lower']>-.005 and att>=0
        comparisons.append(dict(control=other,primary=other in ('O','S'),restricted_time_saved=eff,terminal_utility_delta=end,attainment_delta=att,passed=bool(passed)))
    diagnostic={};ds=CaseStore(RUN/'diagnostics',st.verify())
    for f in range(36):
        v=ds.load('f'+str(f),dict(job=f))
        for k,r in v['metrics'].items():
            n,s,m=k.split('_');diagnostic.setdefault(n+'_'+m,[]).append({a:r[a] for a in ('mse','recall','regret','improves')})
    diag={k:{a:float(np.mean([r[a] for r in rows])) for a in rows[0]} for k,rows in diagnostic.items()}
    endpoints=(10,20,40,80,150,300,600)
    result=dict(joint_pass=all(c['passed'] for c in comparisons if c['primary']),comparisons=comparisons,
        restricted_time300={m:v.mean((0,1,2)).tolist() for m,v in times.items()},attainment300={m:v.mean((0,1,2)).tolist() for m,v in success.items()},
        curve_nfe=endpoints,curve_means={m:v.mean((0,1,2))[[t-10 for t in endpoints]].tolist() for m,v in curves.items()},
        auc10_300={m:float(v[...,:291].mean()) for m,v in curves.items()},diagnostics=diag,
        seconds_per_trajectory={m:float(np.mean(v)) for m,v in timing.items()},calls=dict(search=475200,diagnostics=6048,contracts=180,total=481428),epochs=0)
    write_json(OUT/'RESULTS.json',result);np.savez_compressed(OUT/'arrays.npz',**{f'curve_{m}':v for m,v in curves.items()},**{f'time_{m}':v for m,v in times.items()},**{f'success_{m}':v for m,v in success.items()})
    lines=['# 冷启动固定窗口：开发实验结果','',f"联合验收：{'通过' if result['joint_pass'] else '未通过'}。本轮不属于原解锁版/旧residual验证。",'',
        '20维，36配置×2新实例；10点共同初始化，W辅助至累计40次，随后使用全部历史拟合纯在线GP。持续运行至600次，主要考察300次前缀；不是为300预算单独调整过候选策略。P/F/W各3模型种子，O/S共享。792条轨迹，总481428次目标评估，新增神经训练0轮。',
        '训练先验覆盖10/20/40上下文；5点是外推诊断。任务家族已用于开发，不是独立确认。', '',
        '主要目标：比初始10点最佳值低0.5个初始样本标准差。未达标记为删失，受限时间记301，仍计入统计。时间从第一次真实评估起算。', '',
        '|方法|300次内达到率|受限平均达到时间↓|600次有界改善↑|','|---|---:|---:|---:|']
    for m in st.METHODS:lines.append(f"|{m}|{result['attainment300'][m][1]:.1%}|{result['restricted_time300'][m][1]:.3f}|{result['curve_means'][m][-1]:.6f}|")
    lines+=['','O纯在线；P纯离线先验引导；F持续先验＋在线GP纠偏；W固定40次退出；S共同10点后增加30个Sobol点，再切换纯在线。', '',
        '|W相对对照|平均节省评估|97.5%区间|600次质量差|97.5%区间|联合条件|','|---|---:|---|---:|---|---|']
    for c in comparisons:
        a,b=c['restricted_time_saved'],c['terminal_utility_delta'];lines.append(f"|{c['control']}|{a['mean']:+.3f}|[{a['lower']:+.3f},{a['upper']:+.3f}]|{b['mean']:+.6f}|[{b['lower']:+.6f},{b['upper']:+.6f}]|{'通过' if c['passed'] else '未通过'}|")
    lines+=['','仅O/S是主比较；要求各至少节省5次、区间下界>0、三个模型种子均正，终局差下界>−0.005且达到率不下降。P/F仅描述，不能以它们替代主验收。12配方聚类bootstrap10000次；模型种子不能当独立任务。', '',
        '|上下文数|方法|预测MSE↓|真实最优十分位召回↑|选点真值标准化遗憾↓|选点改善比例↑|','|---|---|---:|---:|---:|---:|']
    for n in (5,10,20,40):
        for m in ('P','Shuffled','O','F'):
            r=diag[f'{n}_{m}'];lines.append(f"|{n}|{m}|{r['mse']:.4f}|{r['recall']:.3f}|{r['regret']:.4f}|{r['improves']:.1%}|")
    lines+=['','独立诊断真值没有进入搜索或调参。诊断按预测均值选点，不能等同EI收益。F用全部上下文拟合残差，方差未刻画数据依赖先验的不确定性；不宣称校准。',
        '本轮没有自适应退出、打乱先验闭环或额外重训。任何主门槛失败停止本候选；不根据结果改变目标难度、辅助窗口、核或预算。若通过也只进入另行冻结的机制验证，不直接视为论文结论。']
    (OUT/'CONCLUSIONS.zh-CN.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,2,figsize=(11,4))
    for m,v in curves.items():
        axs[0].plot(np.arange(10,601),v.mean((0,1,2)),label=m)
        tt=times[m][...,1];axs[1].plot(np.arange(10,301),[(tt<=n).mean() for n in range(10,301)],label=m)
    axs[0].set(xlabel='Paid objective evaluations',ylabel='Mean bounded improvement');axs[1].set(xlabel='Paid objective evaluations',ylabel='Target attainment rate')
    for ax in axs:ax.axvline(40,ls=':',color='gray');ax.legend()
    fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(OUT/f'curves.{ext}',dpi=160)
    print(json.dumps(result['comparisons'],indent=2),flush=True)

def archive():
    assert (OUT/'VERIFICATION.json').exists();st.verify();dest=ROOT/'artifacts/cold_start_v1';dest.mkdir(exist_ok=True)
    files=sorted(set([p for d in (RUN,OUT) for p in d.rglob('*') if p.is_file() and p.suffix!='.lock']+[ROOT/p for p in st.SOURCES]+[Path(__file__)]));manifest={str(p.relative_to(ROOT)):sha256(p) for p in files};parts={}
    for k in range(0,len(files),100):
        zp=dest/f'cold_start_v1.part{k//100+1:03d}.zip'
        with zipfile.ZipFile(zp,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
            for p in files[k:k+100]:z.write(p,str(p.relative_to(ROOT)))
        with zipfile.ZipFile(zp) as z:
            assert z.testzip() is None
            for name in z.namelist():assert hashlib.sha256(z.read(name)).hexdigest()==manifest[name]
        parts[zp.name]=sha256(zp)
    write_json(dest/'MANIFEST.json',dict(files=manifest,archives=parts,parents=['artifacts/fewshot_prior_v1','artifacts/frozen_prior_correction_v1'],total_calls=481428))
    print(f'Archived {len(files)} files in {len(parts)} verified parts',flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['verify','report','archive']);p.add_argument('--workers',type=int,default=24);a=p.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    if a.phase=='verify':verify(a.workers)
    elif a.phase=='report':report()
    else:archive()
