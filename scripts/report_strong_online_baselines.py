"""Accounting, GPU score replay, paired analysis and archival of online references."""
import os
for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[k] = '1'
import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.distance import pdist
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import strong_online_baselines as s
from roopf.fitted_online import replay_score, TrustRegion
from roopf.experiment_io import CaseStore, write_json, sha256, fingerprint


def load(j):
    return CaseStore(s.RUN/'cases', s.verify()).load('_'.join(map(str, j)), dict(job=j))


def verify():
    s.verify()
    assert (s.RUN/'COMPLETE.json').exists()
    torch.set_num_threads(1)
    parent = CaseStore(s.parent.RUN/'cases', s.parent.verify())
    cpu_errors, gpu_errors, warnings, restarts = [], [], [], 0
    tasks = {}
    for j in s.jobs():
        f, i, m = j
        v = load(j)
        x, y = v['x'], v['y']
        assert v['calls'] == len(x) == len(y) == 300 and v['teacher_calls'] == 0
        assert torch.isfinite(x).all() and torch.isfinite(y).all()
        assert x.min() >= -5 and x.max() <= 5 and pdist(x.numpy()).min() > 1e-6
        p = parent.load(f'{f}_{i}_O_0', dict(job=(f, i, 'O', 0)))
        assert torch.equal(x[:10], p['x'][:10]) and torch.equal(y[:10], p['y'][:10])
        tasks.setdefault((f, i), set()).add(fingerprint(dict(x=x[:10], y=y[:10])))
        paid = 10
        state = TrustRegion(best=float(y[:10].min())) if m == 'TuRBO_LogEI' else None
        for info in v['trace']:
            assert info['paid_before'] == paid
            if info['restart']:
                assert state and state.length < .5**7
                n = info['count']
                assert n == min(10, 300-paid) and torch.equal(info['points'], x[paid:paid+n])
                state = TrustRegion(best=float(y[paid:paid+n].min()), start=paid, restarts=state.restarts+1)
                paid += n
                restarts += 1
                continue
            assert torch.equal(info['point'], x[paid])
            from dataclasses import asdict
            assert info['state'] == (asdict(state) if state else None)
            point = (x[paid].double()+5)/10
            bounds = info['bounds']
            assert (point >= bounds[0]-1e-7).all() and (point <= bounds[1]+1e-7).all()
            if state:
                ls_raw = info['parameters']['covar_module.base_kernel.raw_lengthscale'].flatten()
                ls = .005 + (4-.005)*ls_raw.sigmoid()
                w = ls/ls.log().mean().exp()
                local = y[state.start:paid]
                center = (x[state.start+int(local.argmin())].double()+5)/10
                expected = torch.stack(((center-w*state.length/2).clamp(0,1),
                                        (center+w*state.length/2).clamp(0,1)))
                assert torch.allclose(bounds, expected, atol=1e-10, rtol=1e-10)
            else:
                assert torch.equal(bounds, torch.stack((torch.zeros(20),torch.ones(20))).double())
            warnings.extend(info['warnings'])
            if paid in (10, 39, 40, 149, 299):
                error = abs(replay_score(x[:paid], y[:paid], m, info)-info['logei'])
                assert error < 1e-6, (j, paid, error)
                cpu_errors.append(error)
                if torch.cuda.is_available():
                    error = abs(replay_score(x[:paid], y[:paid], m, info, 'cuda')-info['logei'])
                    assert error < 1e-5, (j, paid, error)
                    gpu_errors.append(error)
            if state:
                state.update(float(y[paid]))
            paid += 1
        assert paid == 300
    assert len(tasks) == 72 and all(len(v) == 1 for v in tasks.values())
    write_json(s.OUT/'VERIFICATION.json', dict(cases=144, tasks=72, calls=43200,
        cpu_replayed=len(cpu_errors), gpu_replayed=len(gpu_errors),
        max_cpu_error=max(cpu_errors), max_gpu_error=max(gpu_errors) if gpu_errors else None,
        warnings_count=len(warnings), warning_examples=sorted(set(warnings))[:20], restarts=restarts,
        exact_initialization=True, unique_paid_points=True, trust_region_reconstructed=True,
        diagnostic_calls=0, verification_objective_calls=0))


def report():
    assert (s.OUT/'VERIFICATION.json').exists()
    prior = np.load(s.parent.OUT/'arrays.npz')
    data = {m: {k:prior[f'{m}_{k}'][:, :, :2].mean(0) for k in ('time','attainment','curve')}
            for m in ('O','WA','W')}
    rows = []
    for m in s.METHODS:
        data[m] = dict(time=np.zeros((36,2)), attainment=np.zeros((36,2)), curve=np.zeros((36,2,291)))
    for j in s.jobs():
        f, i, m = j
        v = load(j)
        y = v['y'].numpy().astype(float)
        scale = max(y[:10].std(ddof=1), 1e-8)
        gain = np.maximum(0, y[:10].min()-np.minimum.accumulate(y)[9:])/scale
        hit = np.flatnonzero(gain >= .5)
        t = int(hit[0]+10) if len(hit) else 301
        data[m]['time'][f,i] = t
        data[m]['attainment'][f,i] = bool(len(hit))
        data[m]['curve'][f,i] = gain/(1+gain)
        rows.append(dict(fid=f, instance=i, method=m, time=t, attained=bool(len(hit)),
            terminal=float(data[m]['curve'][f,i,-1]), concurrent_elapsed_seconds=v['seconds'],
            fitting_seconds=sum(r.get('fit_seconds',0) for r in v['trace'])))
    indices = np.random.default_rng(2026100101).integers(0,12,(10000,12))
    def estimate(d):
        groups = d.reshape(12,3,2).mean((1,2))
        lo, hi = np.quantile(groups[indices].mean(1), [.00625,.99375])
        return dict(mean=float(d.mean()),lower=float(lo),upper=float(hi),recipe_means=groups.tolist())
    contrasts = {}
    for m in ('W','WA'):
        for b in s.METHODS:
            a, c = data[m], data[b]
            t = estimate(c['time']-a['time'])
            contrasts[f'{m}_vs_{b}'] = dict(time_saved=t,
                attainment_delta=estimate(a['attainment']-c['attainment']),
                terminal_delta=estimate(a['curve'][...,-1]-c['curve'][...,-1]),
                practical_efficiency=bool(t['mean']>=5 and t['lower']>0))
    means = {m:dict(time=float(a['time'].mean()),attainment=float(a['attainment'].mean()),
                 terminal=float(a['curve'][...,-1].mean()),auc=float(a['curve'].mean())) for m,a in data.items()}
    result = dict(means=means, contrasts=contrasts, new_calls=43200, contracts=48, epochs=0,
        tasks=72, new_trajectories=144, reused_trajectories=360,
        scope='Existing development instances; TuRBO-1 sequential LogEI variant, not original TS',
        run=json.loads((s.RUN/'COMPLETE.json').read_text()))
    write_json(s.OUT/'RESULTS.json',result)
    write_json(s.OUT/'rows.json',rows)
    np.savez_compressed(s.OUT/'arrays.npz',**{m+'_'+k:v for m,a in data.items() for k,v in a.items()})
    lines=['# 标准拟合 GP 与 TuRBO-1／LogEI 开发参照','',
        '固定既有36配置×前两个实例，共72任务。两组新增144条轨迹，每条预算300（含共同初始10点）。新增43,200次目标调用、48次契约；离线训练0轮。既有O/WA/W复用360条，W平均三个模型种子。', '',
        '|方法|受限平均达到时间↓|300次达到率↑|终局有界改善↑|', '|---|---:|---:|---:|']
    for m,a in means.items():
        lines.append(f"|{m}|{a['time']:.3f}|{a['attainment']:.3%}|{a['terminal']:.6f}|")
    lines += ['', '|比较（正值利于先验方法）|节省评估|98.75%区间|实用效率门槛|终局差|', '|---|---:|---|---|---:|']
    for m,a in contrasts.items():
        t=a['time_saved'];lines.append(f"|{m}|{t['mean']:+.3f}|[{t['lower']:+.3f},{t['upper']:+.3f}]|{a['practical_efficiency']}|{a['terminal_delta']['mean']:+.6f}|")
    lines += ['', '区间按12配方聚类、10,000 bootstrap、每端点四比较校正。目标为初始最好值改善0.5个初始样本标准差，未达记301。没有将原来144任务的总体均值拿来和本次72任务直接比较。', '',
        'TuRBO_LogEI 为官方教程设置的单信赖域、q=1、解析LogEI变体；成功容忍10、初始10、超参数每步重新拟合。不是原作者Thompson sampling实现，不能将本次排名概括为超过所有TuRBO实现。', '',
        '这两组同时改变核、超参数学习和采集求解，属于外部在线参照，不替代O/WA/W的结构消融。既有函数族经过多轮开发，本次也不是独立外部测试。学习先验相对WA的必要性仍由之前冻结复验回答。', '',
        '运行成本见rows.json和COMPLETE记录：每轨迹elapsed受12进程竞争影响，不能直接与之前16进程的旧方法计时作单任务加速比。标准GP拟合与采集均有有限迭代上限，不声称全局最优或每次严格收敛。']
    (s.OUT/'CONCLUSIONS.zh-CN.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1,2,figsize=(12,4.5))
    for m,a in data.items():
        axes[0].plot(np.arange(10,301),a['curve'].mean((0,1)),label=m)
        # W attainment curves require averaging indicators over seeds, not thresholding mean time.
        times = prior['W_time'][:,:,:2] if m=='W' else a['time']
        axes[1].plot(np.arange(10,301),[(times<=n).mean() for n in range(10,301)],label=m)
    axes[0].set(xlabel='Paid evaluations',ylabel='Bounded improvement')
    axes[1].set(xlabel='Paid evaluations',ylabel='Target attainment')
    for ax in axes:ax.legend()
    fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(s.OUT/f'curves.{ext}',dpi=160)
    print(json.dumps(result,indent=2))


def archive():
    assert (s.OUT/'RESULTS.json').exists()
    s.verify()
    dest=ROOT/'artifacts/strong_online_baselines_v1'
    dest.mkdir(exist_ok=True)
    files=sorted(set([p for d in (s.RUN,s.OUT) for p in d.rglob('*') if p.is_file() and p.suffix!='.lock']+
        [ROOT/p for p in s.SOURCES]+[Path(__file__)]))
    manifest={str(p.relative_to(ROOT)):sha256(p) for p in files}
    parts={}
    for k in range(0,len(files),100):
        path=dest/f'strong_online_baselines_v1.part{k//100+1:03d}.zip'
        with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
            for p in files[k:k+100]:z.write(p,str(p.relative_to(ROOT)))
        with zipfile.ZipFile(path) as z:
            assert z.testzip() is None
            for name in z.namelist():assert hashlib.sha256(z.read(name)).hexdigest()==manifest[name]
        parts[path.name]=sha256(path)
    write_json(dest/'MANIFEST.json',dict(files=manifest,archives=parts,new_calls=43248))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('phase',choices=['verify','report','archive'])
    a=p.parse_args()
    globals()[a.phase]()
