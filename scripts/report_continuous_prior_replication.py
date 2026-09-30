"""Independent accounting and cluster inference for the three-seed replication."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse,json,zipfile,hashlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from copy import deepcopy
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import continuous_prior_replication as s
from scripts.report_acquisition_factorial import check_values
from roopf.experiment_io import CaseStore,write_json,sha256

def load(j):return CaseStore(s.RUN/'cases',s.verify()).load('_'.join(map(str,j)),dict(job=j))
def check(j):
    f,i,m,seed=j
    return check_values(j,load(j),f,i,m,'continuous',seed,s.ROLE)

def verify():
    s.verify();assert (s.RUN/'COMPLETE.json').exists();initial={};errors=[];ties=0;distinct=0;gap=0.;distance=0.
    with ProcessPoolExecutor(max_workers=16) as ex:
        for r in ex.map(check,s.jobs()):
            initial.setdefault(tuple(r['key']),set()).add(r['initial']);errors.append(r['error']);ties+=r['near_ties']
            distinct+=r['distinct_ties'];gap=max(gap,r['max_tie_gap']);distance=max(distance,r['max_tie_distance'])
    assert len(initial)==144 and all(len(v)==1 for v in initial.values())
    write_json(s.OUT/'VERIFICATION.json',dict(cases=720,decisions=3600,initial_tasks=144,unique_paid_points=True,
        common_initialization=True,monotone_acquisition_decisions=720*290,max_replay_error=max(errors),
        numerical_near_ties=ties,distinct_point_near_ties=distinct,max_near_tie_logei_gap=gap,max_near_tie_point_distance=distance,extra_calls=0))

@torch.no_grad()
def gpu_check():
    s.verify();torch.set_num_threads(1);assert torch.cuda.is_available();errors=[];queries=0
    # All 432 W trajectories: active-stage selected points plus their original pools.
    for f in range(36):
        for seed in range(3):
            for i in range(4):
                v=load((f,i,'W',seed));p=s.st.base.Prior(f,seed,'correct',v['x'][:10],v['y'][:10])
                model=deepcopy(p.model).cuda();cx=p.cx[None].cuda();cy=((p.cy-p.center)/p.scale)[None].cuda()
                qs=[]
                for step in range(30):
                    n=step+10;qs.append(np.vstack((s.st.pool(v['x'][:n],v['y'][:n].numpy(),f,i,s.ROLE,step),v['acquisition'][step]['extra'])))
                q=np.vstack(qs);observed=[]
                for batch in np.array_split(q,9):observed.append(model(cx,cy,torch.from_numpy(batch)[None].cuda())[0].cpu().numpy())
                observed=np.concatenate(observed);expected=p(q);err=float(abs(observed-expected).max())
                assert np.allclose(observed,expected,rtol=1e-5,atol=1e-5);errors.append(err);queries+=len(q);p.verify()
    write_json(s.OUT/'GPU_PRIOR_PARITY.json',dict(trajectories=432,states=432*30,candidate_predictions=queries,max_absolute_error=max(errors),objective_calls=0,training_epochs=0))

def report():
    s.verify();assert (s.OUT/'VERIFICATION.json').exists()
    arrays={m:dict(time=np.zeros((3,36,4)),attainment=np.zeros((3,36,4)),curve=np.zeros((3,36,4,291))) for m in ('O','WA','W')}
    rows=[];calls=0;statuses=[];rank_changes=0;selected=[]
    for j in s.jobs():
        f,i,m,seed=j;v=load(j);y=v['y'].numpy().astype(float);scale=max(y[:10].std(ddof=1),1e-8)
        g=np.maximum(0,y[:10].min()-np.minimum.accumulate(y)[9:])/scale;hit=np.flatnonzero(g>=.5);t=int(hit[0]+10) if len(hit) else 301;u=g/(1+g)
        for k in ([seed] if m=='W' else range(3)):
            arrays[m]['time'][k,f,i]=t;arrays[m]['attainment'][k,f,i]=bool(len(hit));arrays[m]['curve'][k,f,i]=u
        rows.append(dict(fid=f,instance=i,method=m,seed=seed,time=t,attainment=bool(len(hit)),terminal=float(u[-1]),seconds=v['seconds']))
        calls+=v['calls'];statuses.extend(z['status'] for z in v['acquisition']);rank_changes+=sum(z['original_rank_changed'] for z in v['acquisition']);selected.extend(z['continuous_selected'] for z in v['acquisition'])
    assert calls==216000
    idx=np.random.default_rng(2026093019).integers(0,12,(10000,12))
    def estimate(d):
        groups=d.reshape(3,12,3,4).mean((0,2,3));lo,hi=np.quantile(groups[idx].mean(1),[.0125,.9875])
        return dict(mean=float(d.mean()),lower=float(lo),upper=float(hi),seed_means=d.mean((1,2)).tolist(),recipe_means=groups.tolist())
    contrasts={}
    for m in ('O','WA'):
        a,b=arrays['W'],arrays[m];t=estimate(b['time']-a['time']);u=estimate(a['curve'][...,-1]-b['curve'][...,-1]);att=estimate(a['attainment']-b['attainment'])
        efficiency=bool(t['mean']>=5 and t['lower']>0 and min(t['seed_means'])>0)
        contrasts[m]=dict(time_saved=t,terminal_delta=u,attainment_delta=att,efficiency_pass=efficiency,
                         protective_pass=bool(u['lower']>-.005 and att['mean']>=0),joint_pass=bool(efficiency and u['lower']>-.005 and att['mean']>=0))
    means={m:dict(time=float(a['time'].mean()),attainment=float(a['attainment'].mean()),terminal=float(a['curve'][...,-1].mean()),auc=float(a['curve'].mean()),seed_times=a['time'].mean((1,2)).tolist(),seed_terminal=a['curve'][...,-1].mean((1,2)).tolist()) for m,a in arrays.items()}
    result=dict(means=means,contrasts=contrasts,joint_pass=all(r['joint_pass'] for r in contrasts.values()),calls=dict(main=calls,contracts=360,total=calls+360),epochs=0,
        solver=dict(decisions=len(statuses),status_counts={str(k):statuses.count(k) for k in sorted(set(statuses))},continuous_selection=float(np.mean(selected)),original_rank_discrepancies=rank_changes),
        scope='Three frozen model seeds, new instances in historically exposed task family, 300 paid evaluations',shared_controls_independent_cases=288)
    write_json(s.OUT/'RESULTS.json',result);write_json(s.OUT/'rows.json',rows);np.savez_compressed(s.OUT/'arrays.npz',**{m+'_'+k:v for m,a in arrays.items() for k,v in a.items()})
    lines=['# 连续采集求解器下的三种子复验','', '720 条轨迹，36 配置 × 4 新实例；W 使用三个冻结训练种子，O/WA 每任务只运行一次并共享。主调用 216,000，契约 360，新增训练 0。固定预算 300，初始化 10，先验于 40 次退出。', '', '|方法|受限平均达到时间↓|300次达到率|300次有界改善↑|','|---|---:|---:|---:|']
    for m,v in means.items():lines.append(f"|{m}|{v['time']:.3f}|{v['attainment']:.3%}|{v['terminal']:.6f}|")
    lines+=['', '|W 相对|节省评估|97.5%区间|三个训练种子节省|效率/保护/联合|','|---|---:|---|---|---|']
    for m,r in contrasts.items():
        t=r['time_saved'];lines.append(f"|{m}|{t['mean']:+.3f}|[{t['lower']:+.3f},{t['upper']:+.3f}]|{', '.join(f'{v:+.3f}' for v in t['seed_means'])}|{r['efficiency_pass']}/{r['protective_pass']}/{r['joint_pass']}|")
    lines+=['','|W 相对|达到率差及97.5%区间|300次终局差及97.5%区间|','|---|---|---|']
    for m,r in contrasts.items():
        a,u=r['attainment_delta'],r['terminal_delta'];lines.append(f"|{m}|{a['mean']:+.3%} [{a['lower']:+.3%},{a['upper']:+.3%}]|{u['mean']:+.6f} [{u['lower']:+.6f},{u['upper']:+.6f}]|")
    lines+=['',f"两项联合门槛均通过：{result['joint_pass']}。", '',
        '效率门槛：均值节省至少5次、区间下界正、三个种子方向均正。保护：终局下界>−0.005、平均达到率不下降。区间按12配方聚类10,000次，两个比较在各端点内校正，不是全部端点的单一联合覆盖声明。', '',
        '本次假设在看到六组开发结果后另行冻结。不能替换六组中未通过的主要交互检验，不能称未见函数族确认，也不能证明600次非劣或旧residual/固定退出必要。', '',
        '同一家族的这一有界复验无论结果如何均完整保留，不根据结果增补种子、改窗口、延长预算或继续追显著性。']
    (s.OUT/'CONCLUSIONS.zh-CN.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,2,figsize=(12,4.5))
    for m,a in arrays.items():
        axs[0].plot(np.arange(10,301),a['curve'].mean((0,1,2)),label=m)
        axs[1].plot(np.arange(10,301),[(a['time']<=n).mean() for n in range(10,301)],label=m)
    axs[0].set(xlabel='Paid evaluations',ylabel='Mean bounded improvement');axs[1].set(xlabel='Paid evaluations',ylabel='Target attainment')
    for ax in axs:ax.axvline(40,color='gray',ls=':');ax.legend()
    fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(s.OUT/f'curves.{ext}',dpi=160)
    print(json.dumps(result,indent=2),flush=True)

def archive():
    s.verify();assert (s.OUT/'VERIFICATION.json').exists();dest=ROOT/'artifacts/continuous_prior_replication_v1';dest.mkdir(exist_ok=True)
    files=sorted(set([p for d in (s.RUN,s.OUT) for p in d.rglob('*') if p.is_file() and p.suffix!='.lock']+[ROOT/p for p in s.SOURCES]+[Path(__file__),ROOT/'scripts/report_acquisition_factorial.py']))
    manifest={str(p.relative_to(ROOT)):sha256(p) for p in files};parts={}
    for k in range(0,len(files),100):
        path=dest/f'continuous_prior_replication_v1.part{k//100+1:03d}.zip'
        with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
            for p in files[k:k+100]:z.write(p,str(p.relative_to(ROOT)))
        with zipfile.ZipFile(path) as z:
            assert z.testzip() is None
            for n in z.namelist():assert hashlib.sha256(z.read(n)).hexdigest()==manifest[n]
        parts[path.name]=sha256(path)
    write_json(dest/'MANIFEST.json',dict(files=manifest,archives=parts,main_calls=216000,contract_calls=360,parent='artifacts/acquisition_factorial_v1'))
    print(f'Archived {len(files)} files in {len(parts)} parts',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['verify','gpu_check','report','archive']);a=p.parse_args();globals()[a.phase]()
