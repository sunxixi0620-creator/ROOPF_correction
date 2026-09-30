"""Zero-query verification and reporting of restricted metric oracle."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,argparse,zipfile,hashlib
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import torch
from scipy.spatial.distance import pdist
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import oracle_metric as study
from roopf.experiment_io import CaseStore,write_json,sha256,fingerprint
st=study.st;RUN,OUT=study.RUN,study.OUT
PAIRS=(('T40','O'),('T40','WA'),('T40','R40'),('T600','O'),('T600','WA'))
def load(j):return CaseStore(RUN/'cases',study.verify()).load('_'.join(map(str,j)),dict(job=j))
def check(j):
    torch.set_num_threads(1);f,i,m=j;v=load(j);x,y=v['x'],v['y'];assert len(y)==600 and v['calls']==600 and v['teacher_calls']==0 and torch.isfinite(y).all() and pdist(x.numpy()).min()>1e-6
    assert torch.equal(x[:10],st.inputs(f,i,study.ROLE,10));p=st.base.Prior(f,0,'analytic' if m=='WA' else 'zero',x[:10],y[:10]);z=(y.numpy().astype(float)-p.center)/p.scale;err=0
    if m.startswith('T') or m=='R40':
        task=study.ProceduralTask(f,i,study.ROLE,dim=20);A=study.transform(task,m,study.ROLE,f,i);assert np.array_equal(A,v['A']) and abs(np.trace(A@A.T)-20)<1e-4 and task.points==0
    else:A=np.eye(20)
    if m in ('T40','R40'):assert np.array_equal(v['switch'],z[:40])
    if m=='WA':assert np.array_equal(v['switch']['targets'],z[:40])
    for step in (0,29,30,289,589):
        n=step+10;active=m=='WA' and n<40;B=A if m=='T600' or(m in ('T40','R40') and n<40) else np.eye(20)
        targets=z[:n]
        if active:targets=np.r_[z[:10]-p(x[:10].numpy()),v['trace'][:step,5]-v['trace'][:step,1]]
        gp=st.base.GP(x[:n].numpy(),targets) if m in ('O','WA') else study.MetricGP(x[:n].numpy(),targets,B);q=st.pool(x[:n],y[:n].numpy(),f,i,study.ROLE,step);new=st.choose(q,x[:n],y[:n].numpy(),p,gp,active,'O');old=v['trace'][step,:5]
        assert new[0]==int(old[0]) and np.array_equal(q[new[0]],x[n].numpy()) and np.allclose(new,old,rtol=1e-7,atol=1e-8),(j,step)
        err=max(err,float(abs(np.array(new)-old).max()))
    return (f,i),fingerprint(dict(x=x[:10],y=y[:10])),err
def verify():
    study.verify();assert (RUN/'COMPLETE.json').exists();ids={};errors=[]
    with ProcessPoolExecutor(max_workers=24) as ex:
        for n,(j,h,e) in enumerate(ex.map(check,study.jobs()),1):
            ids.setdefault(j,set()).add(h);errors.append(e)
            if n%144==0:print(f'verified {n}/720',flush=True)
    assert len(ids)==144 and all(len(v)==1 for v in ids.values())
    for f in range(36):
        for i in range(4):
            a=load((f,i,'T40'));b=load((f,i,'T600'));assert torch.equal(a['x'][:40],b['x'][:40]) and torch.equal(a['y'][:40],b['y'][:40])
    write_json(OUT/'VERIFICATION.json',dict(cases=720,decisions=3600,shared_initializations=True,budgets_and_unique=True,metric_parameters_exact=True,T40_T600_common_prefix=144,max_error=max(errors),extra_calls=0))
def report():
    a={m:{'time':np.zeros((36,4,3)),'success':np.zeros((36,4,3)),'curve':np.zeros((36,4,591))} for m in study.METHODS};calls=0;rows=[]
    for j in study.jobs():
        f,i,m=j;v=load(j);y=v['y'].numpy().astype(float);ini=y[:10];g=np.maximum(0,ini.min()-np.minimum.accumulate(y)[9:])/max(ini.std(ddof=1),1e-8);ts=[];ss=[]
        for goal in (.25,.5,1.):
            hits=np.flatnonzero(g[:291]>=goal);ts.append(int(hits[0]+10) if len(hits) else 301);ss.append(bool(len(hits)))
        a[m]['time'][f,i]=ts;a[m]['success'][f,i]=ss;a[m]['curve'][f,i]=g/(1+g);calls+=v['calls'];rows.append(dict(fid=f,instance=i,method=m,times=ts,success=ss,terminal=float(g[-1]/(1+g[-1])),seconds=v['seconds']))
    assert calls==432000;index=np.random.default_rng(202609306).integers(0,12,(10000,12))
    def est(d):
        groups=d.reshape(12,3,4).mean((1,2));lo,hi=np.quantile(groups[index].mean(1),[.005,.995]);return dict(mean=float(d.mean()),lower=float(lo),upper=float(hi),recipe_means=groups.tolist())
    contrasts=[]
    for m,c in PAIRS:
        t=est(a[c]['time'][...,1]-a[m]['time'][...,1]);u=est(a[m]['curve'][...,-1]-a[c]['curve'][...,-1]);s=est(a[m]['success'][...,1]-a[c]['success'][...,1]);eff=t['mean']>=5 and t['lower']>0
        contrasts.append(dict(method=m,control=c,time_saved=t,terminal_delta=u,attainment_delta=s,efficiency_pass=bool(eff),strict_joint_pass=bool(eff and u['lower']>-.005 and s['mean']>=0)))
    r=dict(contrasts=contrasts,T40_mechanism_pass=all(c['efficiency_pass'] for c in contrasts if c['method']=='T40'),T600_potential_pass=all(c['efficiency_pass'] for c in contrasts if c['method']=='T600'),means={m:dict(time=v['time'].mean((0,1)).tolist(),attainment=v['success'].mean((0,1)).tolist(),terminal=float(v['curve'][...,-1].mean())) for m,v in a.items()},calls=dict(search=432000,contracts=90,total=432090),epochs=0,privileged=True)
    write_json(OUT/'RESULTS.json',r);write_json(OUT/'rows.json',rows);np.savez_compressed(OUT/'arrays.npz',**{m+'_'+k:v for m,d in a.items() for k,v in d.items()})
    lines=['# 真实二次项度量：受限oracle诊断','', '特权诊断，不是可部署算法，不是全部真实结构的性能上界。仅提供当前任务二次项的方向与轴谱；不提供平移、最优点、梯度或候选真值，不建模其余非线性项的完整结构。', '', '|方法|300次目标达到率|受限平均达到时间↓|600次有界改善↑|','|---|---:|---:|---:|']
    for m,v in r['means'].items():lines.append(f"|{m}|{v['attainment'][1]:.3%}|{v['time'][1]:.3f}|{v['terminal']:.6f}|")
    lines+=['','O=原在线GP；WA=原解析均值先验至40次退出；T40=真实二次度量至40次后切回各向同性；T600=真实二次度量持续；R40=相同真实轴谱、独立随机方向至40次退出。所有metric组均为零均值。所有方法初始10点共同，全部评估计费。', '', '|比较|节省评估|99%区间|达到率差|终局差|效率/联合验收|','|---|---:|---|---:|---:|---|']
    for c in contrasts:
        t=c['time_saved'];lines.append(f"|{c['method']}−{c['control']}|{t['mean']:+.3f}|[{t['lower']:+.3f},{t['upper']:+.3f}]|{c['attainment_delta']['mean']:+.3%}|{c['terminal_delta']['mean']:+.6f}|{c['efficiency_pass']}/{c['strict_joint_pass']}|")
    lines+=['',f"T40相对O/WA/R40的机制效率门槛：{r['T40_mechanism_pass']}；T600相对O/WA的持续度量潜力门槛：{r['T600_potential_pass']}。", '',
        '主要目标为初始最佳再改善0.5个初始样本标准差；300次未达标作为删失，受限均值使用301。效率门槛节省至少5次且区间下界正；联合保护另要求600效用差下界>−0.005且平均达到率不下降。',
        '按12配方聚类10000次，每端点内5比较的99%区间，不是15端点联合覆盖。无训练种子，因为没有学习；144个新任务实例，同历史函数分布，不能称未见函数族确认。', '', '|配方|T40−O节省|T40−WA节省|T40−R40节省|T600−O节省|T600−WA节省|','|---|---:|---:|---:|---:|---:|']
    for recipe in range(12):lines.append('|'+str(recipe)+'|'+'|'.join(f"{c['time_saved']['recipe_means'][recipe]:+.3f}" for c in contrasts)+'|')
    lines+=['','逐配方结果仅作完整展示，不据此挑选函数或宣称子集成功。失败只能说明当前二次度量＋固定核＋当前候选机制没有通过门槛，不能证明所有特征表示无用。成功也不说明此独立随机方向可以从历史任务直接获得。',
        '原生成器的旋转和平移逐实例重采样；此前shared_structure正对照刻意共享rank3方向，已完成，不重复。仅正交旋转不改变各向同性距离，已经代数核验。本轮对metric归一化trace=20以限制平均距离尺度变化，但不是最优核调参。',
        '新增主评估432000＋契约90＝432090次，新增神经训练0轮。未使用神经先验的成绩代替原组件结论。']
    (OUT/'CONCLUSIONS.zh-CN.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,2,figsize=(11,4))
    for m,v in a.items():
        axs[0].plot(np.arange(10,601),v['curve'].mean((0,1)),label=m);tt=v['time'][...,1];axs[1].plot(np.arange(10,301),[(tt<=n).mean() for n in range(10,301)],label=m)
    axs[0].set(xlabel='Paid evaluations',ylabel='Bounded improvement');axs[1].set(xlabel='Paid evaluations',ylabel='Target attainment')
    for ax in axs:ax.axvline(40,color='gray',ls=':');ax.legend()
    fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(OUT/f'curves.{ext}',dpi=160)
    print(json.dumps(r,indent=2),flush=True)
def archive():
    study.verify();assert (OUT/'VERIFICATION.json').exists();dest=ROOT/'artifacts/oracle_metric_v1';dest.mkdir(exist_ok=True)
    files=sorted(set([p for d in (RUN,OUT) for p in d.rglob('*') if p.is_file() and p.suffix!='.lock']+[ROOT/p for p in study.SOURCES]+[Path(__file__)]));manifest={str(p.relative_to(ROOT)):sha256(p) for p in files};parts={}
    for k in range(0,len(files),100):
        zp=dest/f'oracle_metric_v1.part{k//100+1:03d}.zip'
        with zipfile.ZipFile(zp,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
            for p in files[k:k+100]:z.write(p,str(p.relative_to(ROOT)))
        with zipfile.ZipFile(zp) as z:
            assert z.testzip() is None
            for name in z.namelist():assert hashlib.sha256(z.read(name)).hexdigest()==manifest[name]
        parts[zp.name]=sha256(zp)
    write_json(dest/'MANIFEST.json',dict(files=manifest,archives=parts,parent='artifacts/cold_start_components_v1',new_calls=432090));print(f'Archived {len(files)} files',flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['verify','report','archive']);a=p.parse_args();OUT.mkdir(parents=True,exist_ok=True);globals()[a.phase]()
