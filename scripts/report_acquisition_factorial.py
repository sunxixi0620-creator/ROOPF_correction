"""Audit, summarize and archive acquisition factorial without objective calls."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse,json,hashlib,zipfile
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from pathlib import Path
import sys
import numpy as np
import torch
from scipy.spatial.distance import pdist
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import acquisition_factorial as s
from roopf.acquisition_search import original_scores
from roopf.experiment_io import CaseStore,sha256,write_json,fingerprint

def load(j):return CaseStore(s.RUN/'cases',s.verify()).load('_'.join(map(str,j)),dict(job=j))

@torch.no_grad()
def check(j):
    f,i,m,e=j
    return check_values(j,load(j),f,i,m,e,0,s.ROLE)

@torch.no_grad()
def check_values(j,v,f,i,m,e,seed,role):
    torch.set_num_threads(1)
    x,y=v['x'],v['y'];assert v['calls']==300 and v['teacher_calls']==0
    assert len(x)==len(y)==300 and torch.isfinite(y).all() and pdist(x.numpy()).min()>1e-6
    assert torch.equal(x[:10],s.st.inputs(f,i,role,10))
    p=s.st.base.Prior(f,seed,{'O':'zero','WA':'analytic','W':'correct'}[m],x[:10],y[:10])
    assert p.identity==v['prior_identity']
    mode=np.array([float(m!='O' and n<40) for n in range(10,300)])
    assert np.array_equal(mode,v['trace'][:,6])
    if m!='O':assert np.array_equal(v['switch']['targets'],(y[:40].numpy().astype(float)-p.center)/p.scale)
    maxerror=0;near_ties=0;distinct_ties=0;max_tie_gap=0.;max_tie_distance=0.
    for step in (0,29,30,139,289):
        n=10+step;active=bool(mode[step])
        if not active:targets=(y[:n].numpy().astype(float)-p.center)/p.scale
        else:targets=np.r_[(y[:10].numpy().astype(float)-p.center)/p.scale-p(x[:10].numpy()),v['trace'][:step,5]-v['trace'][:step,1]]
        gp=s.st.base.GP(x[:n].numpy(),targets)
        q=s.st.pool(x[:n],y[:n].numpy(),f,i,role,step);old=v['trace'][step,:5]
        if e=='pool':
            new=s.st.choose(q,x[:n],y[:n].numpy(),p,gp,active,m)
            assert new[0]==old[0] and np.array_equal(q[new[0]],x[n].numpy())
        else:
            extra=v['acquisition'][step]['extra'];q=np.vstack((q,extra))
            le,pm,mu,sd=original_scores(q,x[:n],y[:n].numpy(),p,gp,active);k=int(old[0]);winner=int(le.argmax())
            assert np.array_equal(q[k],x[n].numpy())
            gap=float(le.max()-le[k]);assert gap<1e-6,(j,step,gap)
            near_ties+=int(winner!=k)
            if winner!=k:
                distinct_ties+=int(not np.array_equal(q[winner],q[k]))
                max_tie_gap=max(max_tie_gap,gap)
                max_tie_distance=max(max_tie_distance,float(np.linalg.norm(q[winner].astype(float)-q[k])))
            new=(k,pm[k],mu[k],sd[k],np.exp(le[k]))
            assert abs(le[k]-v['acquisition'][step]['logei'])<1e-6
        error=float(abs(np.asarray(new)-old).max());maxerror=max(maxerror,error)
        assert np.allclose(new,old,rtol=1e-6,atol=1e-7),(j,step,new,old)
    p.verify()
    if e=='continuous':assert all(a['logei']>=a['original_logei'] for a in v['acquisition'])
    return dict(key=(f,i),initial=fingerprint(dict(x=x[:10],y=y[:10])),error=maxerror,near_ties=near_ties,
                distinct_ties=distinct_ties,max_tie_gap=max_tie_gap,max_tie_distance=max_tie_distance)

def verify():
    s.verify();assert (s.RUN/'COMPLETE.json').exists();initial={};errors=[];ties=0;distinct=0;gap=0.;distance=0.
    with ProcessPoolExecutor(max_workers=16) as ex:
        for k,r in enumerate(ex.map(check,s.jobs()),1):
            initial.setdefault(tuple(r['key']),set()).add(r['initial']);errors.append(r['error']);ties+=r['near_ties']
            distinct+=r['distinct_ties'];gap=max(gap,r['max_tie_gap']);distance=max(distance,r['max_tie_distance'])
    assert len(initial)==72 and all(len(v)==1 for v in initial.values())
    write_json(s.OUT/'VERIFICATION.json',dict(cases=432,decisions=2160,common_initialization=True,unique_paid_points=True,
               monotone_acquisition_decisions=216*290,max_replay_error=max(errors),numerical_near_ties=ties,
               distinct_point_near_ties=distinct,max_near_tie_logei_gap=gap,max_near_tie_point_distance=distance,
               near_tie_rule='same saved point; log-acquisition gap below 1e-6; predictive values agree',extra_calls=0))

@torch.no_grad()
def gpu_check():
    s.verify();assert torch.cuda.is_available();torch.set_num_threads(1)
    errors=[];queries=0
    for f in range(36):
        for i in range(2):
            v=load((f,i,'W','continuous'));assert v is not None
            p=s.st.base.Prior(f,0,'correct',v['x'][:10],v['y'][:10])
            model=deepcopy(p.model).cuda();cx=p.cx[None].cuda();cy=((p.cy-p.center)/p.scale)[None].cuda()
            # All candidate priors from the active 30 decisions, not just winners.
            qs=[]
            for step in range(30):
                n=step+10;q=s.st.pool(v['x'][:n],v['y'][:n].numpy(),f,i,s.ROLE,step)
                qs.append(np.vstack((q,v['acquisition'][step]['extra'])))
            q=np.vstack(qs);observed=[]
            for batch in np.array_split(q,9):
                observed.append(model(cx,cy,torch.from_numpy(batch)[None].cuda())[0].cpu().numpy())
            observed=np.concatenate(observed);expected=p(q)
            err=float(abs(observed-expected).max());assert np.allclose(observed,expected,rtol=1e-5,atol=1e-5)
            errors.append(err);queries+=len(q);p.verify()
    write_json(s.OUT/'GPU_PRIOR_PARITY.json',dict(states=2160,candidate_predictions=queries,
                max_absolute_error=max(errors),objective_calls=0,training_epochs=0,
                purpose='Independent GPU verification that all continuous W candidates use the same frozen prior'))

def report():
    s.verify();assert (s.OUT/'VERIFICATION.json').exists()
    data={f'{e}_{m}':dict(time=np.zeros((36,2)),attainment=np.zeros((36,2)),curve=np.zeros((36,2,291))) for e in s.ENGINES for m in s.PRIORS}
    rows=[];opt=[];calls=0
    for j in s.jobs():
        f,i,m,e=j;v=load(j);y=v['y'].numpy().astype(float);scale=max(y[:10].std(ddof=1),1e-8)
        g=np.maximum(0,y[:10].min()-np.minimum.accumulate(y)[9:])/scale
        hit=np.flatnonzero(g>=.5);t=int(hit[0]+10) if len(hit) else 301
        a=data[f'{e}_{m}'];a['time'][f,i]=t;a['attainment'][f,i]=bool(len(hit));a['curve'][f,i]=g/(1+g)
        calls+=v['calls'];rows.append(dict(fid=f,instance=i,method=m,engine=e,time=t,attainment=bool(len(hit)),terminal=float(a['curve'][f,i,-1]),seconds=v['seconds']))
        if e=='continuous':opt.extend(v['acquisition'])
    assert calls==129600
    idx=np.random.default_rng(2026093017).integers(0,12,(10000,12))
    def estimate(d,comparisons=1):
        groups=d.reshape(12,3,2).mean((1,2));alpha=.05/(2*comparisons);lo,hi=np.quantile(groups[idx].mean(1),[alpha,1-alpha])
        return dict(mean=float(d.mean()),lower=float(lo),upper=float(hi),recipe_means=groups.tolist())
    primary={}
    for control in ('O','WA'):
        effect=(data[f'continuous_{control}']['time']-data['continuous_W']['time'])-(data[f'pool_{control}']['time']-data['pool_W']['time'])
        r=estimate(effect,2);r['pass']=bool(r['mean']>=5 and r['lower']>0);primary[control]=r
    pairs=[(f'engine_{m}',f'continuous_{m}',f'pool_{m}') for m in s.PRIORS]
    pairs += [('continuous_W_O','continuous_W','continuous_O'),('continuous_W_WA','continuous_W','continuous_WA'),('pool_W_WA','pool_W','pool_WA')]
    secondary={}
    for name,treatment,control in pairs:
        a,b=data[treatment],data[control];effect=estimate(b['time']-a['time'],6)
        effect['efficiency_pass']=bool(effect['mean']>=5 and effect['lower']>0)
        secondary[name]=dict(time_saved=effect,attainment_delta=estimate(a['attainment']-b['attainment']),terminal_delta=estimate(a['curve'][...,-1]-b['curve'][...,-1]))
    means={key:dict(time=float(a['time'].mean()),attainment=float(a['attainment'].mean()),terminal=float(a['curve'][...,-1].mean()),auc=float(a['curve'].mean()),concurrent_mean_seconds=float(np.mean([r['seconds'] for r in rows if f"{r['engine']}_{r['method']}"==key]))) for key,a in data.items()}
    result=dict(means=means,primary_interactions=primary,secondary=secondary,
         solver=dict(decisions=len(opt),continuous_selected=float(np.mean([r['continuous_selected'] for r in opt])),
                     median_logei_gain=float(np.median([r['logei']-r['original_logei'] for r in opt])),
                     mean_function_evaluations=float(np.mean([r['nfev'] for r in opt])),
                     scipy_success_fraction=float(np.mean([r['success'] for r in opt])),
                     original_rank_discrepancies=sum(r['original_rank_changed'] for r in opt)),
         calls=dict(main=calls,contracts=360,total=calls+360),epochs=0,model_seeds=[0],budget=300,
         inference='Development, 12 recipe clusters, one fixed model seed; not external confirmation')
    write_json(s.OUT/'RESULTS.json',result);write_json(s.OUT/'rows.json',rows)
    np.savez_compressed(s.OUT/'arrays.npz',**{key+'_'+metric:values for key,a in data.items() for metric,values in a.items()})
    lines=['# 采集求解器 × 先验：六组开发实验','', '432 条轨迹，全部预算 300、共同初始化 10 点；固定模型种子 0，W/WA 在 40 次撤出。主调用 129,600，契约调用 360，新增训练 0。原先验、GP 参数、600 次带宽计划不变。', '', '|求解器/先验|受限平均达到时间↓|300次达到率|300次有界改善↑|', '|---|---:|---:|---:|']
    for key,v in means.items():lines.append(f"|{key}|{v['time']:.3f}|{v['attainment']:.3%}|{v['terminal']:.6f}|")
    lines+=['', '质量目标为初始最好值再改善 0.5 个初始样本标准差；未达标记为 301，不删除失败。', '', '|主要交互：新求解器增加的W相对收益|平均增加的评估节省|97.5%区间|实用交互门槛|','|---|---:|---|---|']
    for key,r in primary.items():lines.append(f"|W相对{key}|{r['mean']:+.3f}|[{r['lower']:+.3f},{r['upper']:+.3f}]|{r['pass']}|")
    lines+=['','交互正值表示新引擎扩大 W 相对控制的效率优势；门槛均值至少 5 次且区间下界正。','', '|次要比较|节省评估|99.1667%区间|效率门槛|达到率差|300次终局差|','|---|---:|---|---|---:|---:|']
    for key,r in secondary.items():
        t=r['time_saved'];lines.append(f"|{key}|{t['mean']:+.3f}|[{t['lower']:+.3f},{t['upper']:+.3f}]|{t['efficiency_pass']}|{r['attainment_delta']['mean']:+.3%}|{r['terminal_delta']['mean']:+.6f}|")
    lines+=['', '12 配方聚类、10,000 bootstrap；主要两项和次要六项各自校正，未声称所有端点联合覆盖。达到率/终局完整 95% 描述区间见 RESULTS.json。一模型种子不是多种子确认，300 次结果不代表 600 次非劣。', '',
        'pool 为原 128 点、原 EI；continuous 为八起点有界 LogEI 优化，保留原池，最终按原预测器评分。求解器同时记录优化停止状态；达到迭代上限不是异常失效，不声称求得全局最优。', '',
        f"连续候选被选比例 {result['solver']['continuous_selected']:.3%}；SciPy success 比例 {result['solver']['scipy_success_fraction']:.3%}；原池 raw-EI/LogEI 排名差异 {result['solver']['original_rank_discrepancies']} 次。", '',
        '本轮属于已接触函数家族上的开发机制研究。学习先验若未超过 WA，不再调同一模型的窗口/轮次。原解锁版与旧 residual 的身份和结论不变。运行计时需区分单任务示例与并发墙钟。']
    (s.OUT/'CONCLUSIONS.zh-CN.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,2,figsize=(12,4.5))
    colors={'O':'#3b69a3','W':'#ba493e','WA':'#3b8656'}
    for e in s.ENGINES:
        for m in s.PRIORS:
            a=data[f'{e}_{m}'];style='--' if e=='pool' else '-'
            axs[0].plot(np.arange(10,301),a['curve'].mean((0,1)),style,color=colors[m],label=f'{e}/{m}')
            axs[1].plot(np.arange(10,301),[(a['time']<=n).mean() for n in range(10,301)],style,color=colors[m],label=f'{e}/{m}')
    axs[0].set(xlabel='Paid evaluations',ylabel='Mean bounded improvement');axs[1].set(xlabel='Paid evaluations',ylabel='Target attainment')
    for ax in axs:ax.axvline(40,color='gray',ls=':');ax.legend(fontsize=8)
    fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(s.OUT/f'curves.{ext}',dpi=160)
    print(json.dumps(result,indent=2),flush=True)

def archive():
    s.verify();assert (s.OUT/'VERIFICATION.json').exists();dest=ROOT/'artifacts/acquisition_factorial_v1';dest.mkdir(exist_ok=True)
    files=sorted(set([p for d in (s.RUN,s.OUT) for p in d.rglob('*') if p.is_file() and p.suffix!='.lock']+[ROOT/p for p in s.SOURCES]+[Path(__file__),ROOT/'scripts/check_acquisition_search.py']))
    manifest={str(p.relative_to(ROOT)):sha256(p) for p in files};parts={}
    for k in range(0,len(files),100):
        path=dest/f'acquisition_factorial_v1.part{k//100+1:03d}.zip'
        with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
            for p in files[k:k+100]:z.write(p,str(p.relative_to(ROOT)))
        with zipfile.ZipFile(path) as z:
            assert z.testzip() is None
            for n in z.namelist():assert hashlib.sha256(z.read(n)).hexdigest()==manifest[n]
        parts[path.name]=sha256(path)
    write_json(dest/'MANIFEST.json',dict(files=manifest,archives=parts,main_calls=129600,contract_calls=360))
    print(f'Archived {len(files)} files in {len(parts)} parts',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['verify','gpu_check','report','archive']);a=p.parse_args();globals()[a.phase]()
