"""Replay, summarize and archive frozen component study, no objective queries."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,argparse,zipfile,hashlib
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import torch
from scipy.stats import qmc
from scipy.spatial.distance import pdist
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import cold_start_components as study
from roopf.experiment_io import CaseStore,write_json,sha256,seed_for,fingerprint
st=study.st;RUN,OUT=study.RUN,study.OUT

def load(j):return CaseStore(RUN/'cases',study.verify()).load('_'.join(map(str,j)),dict(job=j))
def check(j):
    torch.set_num_threads(1);v=load(j);f,i,s,m=j;x,y=v['x'],v['y'];assert v['calls']==600 and v['teacher_calls']==0 and len(y)==600 and torch.isfinite(y).all() and pdist(x.numpy()).min()>1e-6
    assert torch.equal(x[:10],st.inputs(f,i,study.ROLE,10))
    kind='zero' if m in ('O','S') else ('shuffled' if m=='WS' else ('analytic' if m=='WA' else 'correct'))
    p=st.base.Prior(f,s,kind,x[:10],y[:10]);assert p.identity==v['prior_identity'];err=0;exits=m in ('W','WS','WA')
    if exits:assert np.array_equal(v['switch']['targets'],(y[:40].numpy().astype(float)-p.center)/p.scale)
    expected=np.array([float(m in ('P','F') or (exits and n<40)) for n in range(10,600)])
    assert np.array_equal(v['trace'][:,6],expected)
    for step in (0,29,30,289,589):
        n=10+step;active=bool(expected[step]);gp=None
        if m!='P':
            if exits and n>=40:targets=(y[:n].numpy().astype(float)-p.center)/p.scale
            else:
                init=(y[:10].numpy().astype(float)-p.center)/p.scale-(p(x[:10].numpy()) if active else 0);tr=v['trace'][:step];targets=np.r_[init,tr[:,5]-tr[:,1]]
            gp=st.base.GP(x[:n].numpy(),targets)
        if m=='S' and n<40:
            sobol=qmc.Sobol(20,scramble=True,seed=seed_for(study.ROLE,'sobol',f,i)%(2**32)).random_base2(5)[:30].astype('float32')*10-5;q=sobol[step:step+1]
        else:q=st.pool(x[:n],y[:n].numpy(),f,i,study.ROLE,step)
        new=st.choose(q,x[:n],y[:n].numpy(),p,gp,active,m);old=v['trace'][step,:5]
        assert new[0]==int(old[0]) and np.array_equal(q[new[0]],x[n].numpy()) and np.allclose(new,old,rtol=1e-7,atol=1e-8),(j,step)
        err=max(err,float(abs(np.array(new)-old).max()))
    p.verify();return ((f,i),fingerprint(dict(x=x[:10],y=y[:10])),err)
def verify():
    study.verify();assert (RUN/'COMPLETE.json').exists();initial={};errors=[]
    with ProcessPoolExecutor(max_workers=24) as ex:
        for n,(key,h,e) in enumerate(ex.map(check,study.jobs()),1):
            initial.setdefault(key,set()).add(h);errors.append(e)
            if n%324==0:print(f'verified {n}/3240',flush=True)
    assert len(initial)==216 and all(len(v)==1 for v in initial.values())
    # Full prefix agreement across W/F and original initial paid states.
    for f in range(36):
        for i in range(6):
            for s in range(3):
                a=load((f,i,s,'W'));b=load((f,i,s,'F'));assert torch.equal(a['x'][:40],b['x'][:40]) and torch.equal(a['y'][:40],b['y'][:40])
    write_json(OUT/'VERIFICATION.json',dict(cases=3240,decisions=16200,common_initialization=True,budget_unique=True,W_F_prefix_pairs=648,source_and_weights_frozen=True,max_error=max(errors),extra_calls=0))
def report():
    study.verify();a={m:{'time':np.zeros((3,36,6,3)),'success':np.zeros((3,36,6,3)),'curve':np.zeros((3,36,6,591))} for m in study.METHODS};timing={m:[] for m in study.METHODS};calls=0;case_rows=[]
    for j in study.jobs():
        f,i,s,m=j;v=load(j);y=v['y'].numpy().astype(float);initial=y[:10];scale=max(initial.std(ddof=1),1e-8);g=np.maximum(0,initial.min()-np.minimum.accumulate(y)[9:])/scale;times=[];success=[]
        for target in (.25,.5,1.):
            hit=np.flatnonzero(g[:291]>=target);times.append(int(hit[0]+10) if len(hit) else 301);success.append(bool(len(hit)))
        for seed in ([s] if m in study.SEEDED else range(3)):
            a[m]['time'][seed,f,i]=times;a[m]['success'][seed,f,i]=success;a[m]['curve'][seed,f,i]=g/(1+g)
        calls+=v['calls'];timing[m].append(v['seconds']);case_rows.append(dict(fid=f,instance=i,seed=s,method=m,time=times,success=success,terminal=float(g[-1]/(1+g[-1])),seconds=v['seconds']))
    assert calls==1944000;idx=np.random.default_rng(202609305).integers(0,12,(10000,12));alpha=.05/(2*6)
    def estimate(d):
        grouped=d.reshape(3,12,3,6).mean((0,2,3));lo,hi=np.quantile(grouped[idx].mean(1),[alpha,1-alpha]);return dict(mean=float(d.mean()),lower=float(lo),upper=float(hi),seed_means=d.mean((1,2)).tolist())
    def positive(e,threshold):return e['mean']>=threshold and e['lower']>0 and min(e['seed_means'])>0
    contrasts=[]
    for m in ('O','WS','WA','P','F','S'):
        t=estimate(a[m]['time'][...,1]-a['W']['time'][...,1]);u=estimate(a['W']['curve'][...,-1]-a[m]['curve'][...,-1]);s=estimate(a['W']['success'][...,1]-a[m]['success'][...,1]);eff=bool(positive(t,5))
        contrasts.append(dict(control=m,time_saved=t,terminal_delta=u,attainment_delta=s,efficiency_pass=eff,terminal_improvement_pass=bool(positive(u,.005)),strict_joint_pass=bool(eff and u['lower']>-.005 and s['mean']>=0)))
    c={v['control']:v for v in contrasts};claims=dict(learned_knowledge_efficiency=all(c[m]['efficiency_pass'] for m in ('O','WS','WA')),beyond_space_filling=c['S']['efficiency_pass'],online_adaptation_terminal=c['P']['terminal_improvement_pass'],exit_efficiency=c['F']['efficiency_pass'],original_strict_joint=all(c[m]['strict_joint_pass'] for m in ('O','S')))
    r=dict(claims=claims,contrasts=contrasts,means={m:dict(time=v['time'].mean((0,1,2)).tolist(),attainment=v['success'].mean((0,1,2)).tolist(),terminal=float(v['curve'][...,-1].mean()),auc10_300=float(v['curve'][...,:291].mean()),mean_run_seconds=float(np.mean(timing[m]))) for m,v in a.items()},calls=dict(search=1944000,contracts=180,total=1944180),epochs=0,interval_level=1-.05/6,concurrent_timing=True)
    write_json(OUT/'RESULTS.json',r);write_json(OUT/'rows.json',case_rows);np.savez_compressed(OUT/'arrays.npz',**{m+'_'+k:v for m,d in a.items() for k,v in d.items()})
    lines=['# 冷启动七组组件对照','', '配置冻结后的同分布新实例检验；任务家族有历史开发接触，不是未见函数族确认。先验均为冻结权重，新增训练0轮。',
    '36配置×6新实例，W/WS/P/F三个模型种子，O/WA/S共享。3240条600次轨迹；新目标调用1944180（主搜索1944000＋契约180）。', '',
    'W=正确先验辅助至40次后撤出；O=纯在线GP；WS=打乱训练关系先验且同样退出；WA=解析径向先验且同样退出；P=冻结先验均值选点无在线GP；F=持续融合；S=空间填充后在线GP。初始10点全部计费，退出保留全部数据。', '',
    '|方法|300次达到率|受限平均达到时间↓|600次有界改善↑|并发下平均运行秒数|','|---|---:|---:|---:|---:|']
    for m,v in r['means'].items():lines.append(f"|{m}|{v['attainment'][1]:.3%}|{v['time'][1]:.3f}|{v['terminal']:.6f}|{v['mean_run_seconds']:.3f}|")
    lines+=['','目标为初始最佳值改善0.5个初始样本标准差。未达标记受限时间301，不删除；.25/1目标仅附加报告。运行秒数包括初始化/模型加载但受并发影响，不能当串行部署成本。', '',
    '|W相对|节省评估|99.1667%区间|达标率差|终局质量差|99.1667%区间|','|---|---:|---|---:|---:|---|']
    for x in contrasts:
        t,u=x['time_saved'],x['terminal_delta'];lines.append(f"|{x['control']}|{t['mean']:+.3f}|[{t['lower']:+.3f},{t['upper']:+.3f}]|{x['attainment_delta']['mean']:+.3%}|{u['mean']:+.6f}|[{u['lower']:+.6f},{u['upper']:+.6f}]|")
    lines+=['', '|预先定义的主张|验收|','|---|---|']
    names={'learned_knowledge_efficiency':'有效学习先验相对O/WS/WA均有实质效率增益','beyond_space_filling':'相对空间填充有实质效率增益','online_adaptation_terminal':'在线适应相对P有实质终局增益','exit_efficiency':'退出相对持续融合有实质效率增益','original_strict_joint':'原O/S效率＋达标率＋终局联合保护条件'}
    for k,v in claims.items():lines.append(f"|{names[k]}|{'通过' if v else '未通过'}|")
    lines+=['','效率要求节省至少5次、区间下界>0、三个种子方向为正；在线适应主张要求600次效用增加至少0.005、区间下界>0、各种子正。退出若仅终局略好，不能替代效率主张。',
    '区间按12配方聚类、10000重采样，每个端点内校正6个比较；不是18个端点的联合覆盖保证。终局非劣容忍−0.005和平均达标率不下降的旧保护条件继续单列，不因效率主张通过而改判。',
    '所有信号只使用已支付观测，GP未完整建模数据依赖先验的不确定性。WS/WA为匹配结构对照，函数ID只用于编排持出模型，不输入优化器。P仍用付费最佳点生成局部候选，并非完全不读取在线值。',
    '不按结果改窗口、预算、核或目标；不将本轮归给原anchor/旧residual。不自动进入新架构训练或未见函数族外部测试。']
    (OUT/'CONCLUSIONS.zh-CN.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,2,figsize=(12,4.5))
    for m,v in a.items():
        axs[0].plot(np.arange(10,601),v['curve'].mean((0,1,2)),label=m);tt=v['time'][...,1];axs[1].plot(np.arange(10,301),[(tt<=n).mean() for n in range(10,301)],label=m)
    axs[0].set(xlabel='Paid evaluations',ylabel='Mean bounded improvement');axs[1].set(xlabel='Paid evaluations',ylabel='Target attainment rate')
    for ax in axs:ax.axvline(40,color='gray',ls=':');ax.legend(ncol=2)
    fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(OUT/f'curves.{ext}',dpi=160)
    print(json.dumps(r,indent=2),flush=True)
def archive():
    study.verify();assert (OUT/'VERIFICATION.json').exists();dest=ROOT/'artifacts/cold_start_components_v1';dest.mkdir(exist_ok=True)
    files=sorted(set([p for d in (RUN,OUT) for p in d.rglob('*') if p.is_file() and p.suffix!='.lock']+[ROOT/p for p in study.SOURCES]+[Path(__file__)]));manifest={str(p.relative_to(ROOT)):sha256(p) for p in files};parts={}
    for k in range(0,len(files),100):
        zp=dest/f'cold_start_components_v1.part{k//100+1:03d}.zip'
        with zipfile.ZipFile(zp,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
            for p in files[k:k+100]:z.write(p,str(p.relative_to(ROOT)))
        with zipfile.ZipFile(zp) as z:
            assert z.testzip() is None
            for n in z.namelist():assert hashlib.sha256(z.read(n)).hexdigest()==manifest[n]
        parts[zp.name]=sha256(zp)
    write_json(dest/'MANIFEST.json',dict(files=manifest,archives=parts,parent='artifacts/cold_start_v1',new_calls=1944180));print(f'Archived {len(files)} files in {len(parts)} parts',flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['verify','report','archive']);a=p.parse_args();OUT.mkdir(parents=True,exist_ok=True);globals()[a.phase]()
