"""Predeclared grouped development analysis, never called by the optimizer."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import json,sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import hpob_transfer_development_v1 as s
from roopf.experiment_io import CaseStore,write_json,sha256
from roopf.hpob_transfer_v1 import build_gp,Predictor,standardize,rank_weights,log_ei,taf_scores


def replay_case(v,x,source):
    """No fit and no new labels: rebuild selected full-pool decisions from params."""
    errors=[];name=s.key(v['task']);method=v['method'];seed=v['seed']
    for n in (5,20,40,104):
        info=v['trace'][n-5];paid=[r['index'] for r in v['paid'][:n]]
        if method=='P':scores=source.mean(0).clone()
        else:
            y=torch.tensor([r['value'] for r in v['paid'][:n]],dtype=torch.double)
            z,mu,sd=standardize(y)
            assert abs(float(mu)-info['y_mean'])<1e-12 and abs(float(sd)-info['y_scale'])<1e-12
            prior=source.mean(0) if method=='A' else torch.zeros(len(x),dtype=torch.double)
            model=build_gp(x[paid],z-prior[paid])
            with torch.no_grad():
                for k,p in model.named_parameters():p.copy_(info['parameters'][k])
            model.eval();pred=Predictor(model);mean,var=pred.moments(x);mean+=prior
            scores=log_ei(mean,var,mean[paid].max())
            if method=='R':
                w,_=rank_weights(source[:,paid].numpy(),pred.loo_mean().numpy(),z.numpy(),(name,seed,method),n)
                assert np.max(np.abs(w-np.array(info['weights'])))<1e-12
                scores=taf_scores(scores,source,paid,w)
                available=torch.ones(len(x),dtype=torch.bool);available[paid]=False
                if torch.isneginf(scores[available]).all():
                    scores=torch.zeros_like(scores);assert info['score_kind']=='raw_zero_acquisition'
        scores[paid]=-float('inf')
        assert int(scores.argmax())==info['selected'],(name,seed,method,n,'argmax mismatch')
        err=abs(float(scores[info['selected']])-info['score']);assert err<1e-7,(name,seed,method,n,err)
        errors.append(err)
    return errors


def main():
    torch.set_num_threads(1)
    identity=s.verify();assert (s.RUN/'RUN_COMPLETE.json').exists()
    cache_check=json.loads((s.OUT/'CACHE_VERIFICATION.json').read_text());assert cache_check['passed']
    store=CaseStore(s.RUN/'cases',identity);rows=[];expected={};calls=0;recovered=0;all_warnings=0;groups={}
    ts=s.tasks('development');replay_errors=[]
    for t in ts:
        name=s.key(t);groups.setdefault(t['group'],[]).append(name)
        x=torch.from_numpy(np.load(s.RUN/'public'/f'{name}.npy'))
        source=CaseStore(s.RUN/'caches',identity).load(name,dict(task=t))['means']
        for seed in range(5):
            for method in s.METHODS:
                job=(name,seed,method);v=store.load(f'{name}_{seed}_{method}',dict(job=job));assert v is not None
                assert v['calls']==len(v['paid'])==105 and len(v['trace'])==100
                ix=[r['index'] for r in v['paid']];assert len(set(ix))==105
                signature=v['paid'][:5]
                expected.setdefault((name,seed),signature);assert expected[name,seed]==signature
                for n,r in enumerate(v['trace'],5):
                    assert r['paid_before']==n and r['selected']==ix[n] and np.isfinite(r['score'])
                    if method=='R':assert abs(sum(r['weights'])-1)<1e-12
                    recovered+=int(r.get('recovered',False));all_warnings+=len(r['warnings'])
                assert all(np.isfinite([r['value'] for r in v['paid']]))
                journal=json.loads((s.RUN/'journals'/f'{name}_{seed}_{method}.json').read_text())
                assert journal==v['paid'] and v['source_cache_sha256']==sha256(s.RUN/'caches'/f'{name}.pt')
                replay_errors.extend(replay_case(v,x,source))
                curve=np.array(v['metrics']['regret']);assert np.isfinite(curve).all() and np.all(np.diff(curve)<=1e-12)
                calls+=105;rows.append(dict(task=name,group=t['group'],space=t['space'],seed=seed,method=method,
                    metrics=v['metrics'],seconds=v['seconds'],fit_seconds=sum(r.get('seconds',0) for r in v['trace']),
                    source_weights=[1-r['weights'][-1] for r in v['trace']] if method=='R' else None))
    assert calls==138600 and len(rows)==1320 and len(groups)==11
    names=sorted(groups);metrics=('cold_auc','auc','terminal','restricted_time','attained')
    values={m:{k:[] for k in metrics} for m in s.METHODS};curves={m:[] for m in s.METHODS}
    for group in names:
        for method in s.METHODS:
            r=[v for v in rows if v['group']==group and v['method']==method]
            # Each task has exactly five seeds, so this equals seed->task->group.
            for k in metrics:values[method][k].append(np.mean([v['metrics'][k] for v in r]))
            curves[method].append(np.mean([v['metrics']['regret'] for v in r],axis=0))
    bs=np.random.default_rng(2026100102).integers(0,len(names),(10000,len(names)))
    def estimate(d):
        d=np.asarray(d);lo,hi=np.quantile(d[bs].mean(1),[.025/3,1-.025/3])
        return dict(mean=float(d.mean()),lower=float(lo),upper=float(hi),groups=d.tolist())
    contrasts={}
    for other in ('O','A','P'):
        contrasts['R_vs_'+other]={k:estimate(np.asarray(values[other][k])-np.asarray(values['R'][k])) for k in ('cold_auc','auc','terminal','restricted_time')}
    c=contrasts['R_vs_O'];benefit=c['cold_auc'];terminal=c['terminal']
    # terminal difference is O-R; negative of its lower bound is upper bound R-O.
    gate=benefit['mean']>=.01 and benefit['lower']>0 and -terminal['lower']<=.01
    means={m:{k:float(np.mean(v)) for k,v in kv.items()} for m,kv in values.items()}
    result=dict(means=means,contrasts=contrasts,neural_entry_gate=bool(gate),target_calls=calls,
        source_labels=json.loads((s.RUN/'EXPORTS.json').read_text())['source_labels'],neural_epochs=0,
        tasks=len(ts),groups=len(groups),seeds=5,trajectories=len(rows),cold_fit_recoveries=recovered,
        warnings=all_warnings,confirmation_calls=0,scope='raw-grouped HPO-B-derived development, not official test',
        identity_sha256=sha256(s.RUN/'identity.json'))
    write_json(s.OUT/'RESULTS.json',result);write_json(s.OUT/'ROWS.json',rows)
    write_json(s.OUT/'VERIFICATION.json',dict(cases=1320,calls=138600,unique_indices=True,
        paid_step_alignment=True,paired_initialization=True,cache_check=cache_check,
        full_pool_replayed=len(replay_errors),max_score_error=max(replay_errors),passed=True))
    lines=['# HPO-B派生开发集迁移对照','',
        '66任务、11原始数据集组、每组方法5个配对种子，每条预算105含初始5点。所有组固定后一次性报告；最终确认集调用0，神经训练0轮。',
        '', '|方法|冷启动regret AUC↓|全程AUC↓|终局regret↓|受限达到时间↓|','|---|---:|---:|---:|---:|']
    for m,a in means.items():lines.append(f"|{m}|{a['cold_auc']:.6f}|{a['auc']:.6f}|{a['terminal']:.6f}|{a['restricted_time']:.3f}|")
    lines+=['','O=纯在线GP，P=静态源均值，A=固定源均值＋在线残差GP，R=RGPE-TAF/Matern重实现。',
        '按原始数据集组等权聚合，主比较为R-O/R-A/R-P；正值表示R的regret更低。',
        '', '|比较|冷启动改善|98.3333%组bootstrap区间|','|---|---:|---|']
    for k,a in contrasts.items():
        v=a['cold_auc'];lines.append(f"|{k}|{v['mean']:+.6f}|[{v['lower']:+.6f}, {v['upper']:+.6f}]|")
    lines+=['',f'预先规定的神经先验进入条件：{gate}。这只判断是否有值得继续研究的迁移空间；不证明神经网络不可替代，也不等于独立确认。',
            '本实验是自定义原始数据集隔离的HPO-B派生开发集，不能称为官方HPO-B测试成绩。完整成本、诊断与各任务记录见RESULTS/ROWS。']
    (s.OUT/'CONCLUSIONS.zh-CN.md').write_text('\n'.join(lines)+'\n')
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(7,4.5))
    for m,a in curves.items():ax.plot(np.arange(5,106),np.asarray(a).mean(0)[4:],label=m)
    ax.set(xlabel='Paid evaluations',ylabel='Normalized simple regret',title='HPO-B-derived development (raw-group average)');ax.legend();fig.tight_layout()
    for ext in ('pdf','png'):fig.savefig(s.OUT/f'curves.{ext}',dpi=160)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
