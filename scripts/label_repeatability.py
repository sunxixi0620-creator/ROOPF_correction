"""Fixed-state, paired future-RNG audit. No model training or oracle deployment."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
from pathlib import Path
import sys,json,time,types,multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import long_value as old
from roopf.experiment_io import CaseStore,sha256,fingerprint,seed_for,write_json,save_torch
from scripts.unified_revision import bounded
RUN=ROOT/'results/label_repeatability_20260930';OUT=ROOT/'docs/revision/label_repeatability'
SOURCES=['scripts/label_repeatability.py','docs/experiments/LABEL_REPEATABILITY_PROTOCOL.md','roopf/online_portfolio.py','roopf/model.py']


def verify():
    identity=json.loads((RUN/'identity.json').read_text())
    for p,h in identity['sources'].items():assert sha256(ROOT/p)==h
    assert sha256(old.RUN/'identity.json')==identity['old_identity']
    assert sha256(old.RUN/'dataset.pt')==identity['old_dataset']
    return identity


def selection():
    data=torch.load(old.RUN/'dataset.pt',weights_only=False);ids=data['id'];rows=[]
    for recipe in range(12):
        for step in old.STEPS:
            pool=[tuple(map(int,k)) for k in ids if k[0]//3==recipe and k[3]==step]
            chosen=sorted(pool,key=lambda k:seed_for('repeatability_states_v1',k))[:2]
            for k in chosen:
                indices=[m*4+seed_for('repeatability_action_v1',k,m)%4 for m in range(3)]
                rows.append(dict(id=list(k),recipe=recipe,indices=indices))
    assert len(rows)==48 and len({tuple(r['id']) for r in rows})==48
    return rows


def branch(f,i,p,step,state,point=None,stream=None,budget=600,role='long_value_v1'):
    model=old.parent.make(budget)
    snapshots=old.parent.instrument(model,(step,),dict(step=step,fingerprint=state['fingerprint'],replace=point is not None,point=point))
    score=model.scores;original_rand=torch.rand;private=[];events=[]
    def tracked_rand(*args,**kwargs):
        generator=kwargs.get('generator')
        if generator is not None and len(args)==1 and args[0]==(1,2,20):
            if not private:private.append(generator)
            assert private[0] is generator,'Unexpected additional private RNG'
        return original_rand(*args,**kwargs)
    def hooked(self,*args,**kwargs):
        result=score(*args,**kwargs)
        current=(self.evalnum-100)//2
        if current==step:
            assert len(private)==1
            before=fingerprint((torch.get_rng_state(),private[0].get_state()))
            if stream is not None:
                torch.manual_seed(seed_for('repeatability_global_v1',f,i,p,step,stream,role))
                private[0].manual_seed(seed_for('repeatability_private_v1',f,i,p,step,stream,role))
            events.append(dict(before=before,global_after=fingerprint(torch.get_rng_state()),private_after=fingerprint(private[0].get_state())))
        return result
    model.scores=types.MethodType(hooked,model)
    torch.rand=tracked_rand
    try:out=old.rollout(model,f,i,p,role)
    finally:torch.rand=original_rand
    assert len(events)==1 and snapshots[step]['fingerprint']==state['fingerprint']
    out['rng_event']=events[0]
    return out


def contracts():
    path=RUN/'CONTRACTS.json'
    if path.exists():return
    role='repeatability_contract_v1';f,i,p,step=3,0,0,2
    a=old.rollout(old.parent.make(114),f,i,p,role)
    model=old.parent.make(114);snap=old.parent.instrument(model,(step,));b=old.rollout(model,f,i,p,role);st=snap[step]
    c=branch(f,i,p,step,st,budget=114,role=role)
    d=branch(f,i,p,step,st,stream=0,budget=114,role=role)
    e=branch(f,i,p,step,st,stream=0,budget=114,role=role)
    h=branch(f,i,p,step,st,stream=1,budget=114,role=role)
    point=torch.full((20,),4.5);j=branch(f,i,p,step,st,point,0,114,role)
    assert torch.equal(a['points'],b['points']) and torch.equal(b['points'],c['points'])
    assert torch.equal(d['points'],e['points']) and torch.equal(d['trail'],e['trail'])
    assert not torch.equal(d['points'][:,6:],h['points'][:,6:])
    for key in ('global_after','private_after'):assert d['rng_event'][key]!=h['rng_event'][key]
    for v in (d,e,h,j):assert torch.equal(v['points'][:,:5],a['points'][:,:5])
    assert torch.equal(d['points'][:,:6],a['points'][:,:6]) and torch.equal(j['points'][0,5],point)
    assert d['rng_event']==j['rng_event']
    save_torch(RUN/'contracts.pt',dict(original=a,instrumented=b,noop=c,stream0=d,repeat=e,stream1=h,forced=j))
    write_json(path,dict(calls=7*114,same_stream_exact=True,private_and_global_rng_reset=True,unmodified_prefix=True,forced_slot_exact=True))


def collect(job):
    torch.set_num_threads(1);n,stream=job;identity=verify();row=json.loads((RUN/'SELECTION.json').read_text())[n]
    f,i,p,step=row['id'];spec=dict(state=n,stream=stream);store=CaseStore(RUN/'cases',identity);key=f's{n}_r{stream}'
    with store.lock(key):
        if store.load(key,spec) is not None:return key
        previous=CaseStore(old.RUN/'cases',json.loads((old.RUN/'identity.json').read_text())).load(f'f{f}_i{i}_p{p}',dict(fid=f,instance=i,population=p))
        st=next(v for v in previous['states'] if v['step']==step);snapshot=st['snapshot']
        base=branch(f,i,p,step,snapshot,stream=stream);initial=base['initial'].min(1).values;scale=base['initial'].std(1)
        assert torch.equal(base['points'][:,:2*step+2],previous['base']['points'][:,:2*step+2])
        trajectories=[];gains=[];eligible=[];calls=600
        for k in row['indices']:
            valid=st['metadata'][k]['eligible'];eligible.append(valid)
            if valid:
                out=branch(f,i,p,step,snapshot,st['points'][k],stream);calls+=600
                assert torch.equal(out['points'][:,:2*step+1],base['points'][:,:2*step+1])
                assert torch.equal(out['points'][0,2*step+1],st['points'][k])
                assert out['rng_event']==base['rng_event']
            else:out=base
            trajectories.append(out)
            gains.append(float((bounded(initial,out['trail'][:,-1],scale)-bounded(initial,base['trail'][:,-1],scale))[0]))
        store.save(key,spec,dict(row=row,base=base,branches=trajectories,gains=gains,eligible=eligible,calls=calls,
            redundant_prefix_calls=(calls//600)*(100+2*step),source_state_sha=fingerprint(st['snapshot']),points=st['points'][row['indices']]))
        return key


def interval(values,rows):
    recipe=np.array([np.mean(values[[j for j,r in enumerate(rows) if r['recipe']==k]]) for k in range(12)])
    rng=np.random.default_rng(20261008);boot=recipe[rng.integers(0,12,(10000,12))].mean(1)
    lo,hi=np.quantile(boot,[.0125,.9875])
    return dict(mean=float(values.mean()),lower=float(lo),upper=float(hi),recipe_means=recipe.tolist())


def summarize():
    rows=json.loads((RUN/'SELECTION.json').read_text());store=CaseStore(RUN/'cases',verify())
    delta=np.zeros((48,16,3));eligible=np.zeros((48,3),bool);calls=prefix=0
    for n in range(48):
        for stream in range(16):
            v=store.load(f's{n}_r{stream}',dict(state=n,stream=stream));assert v is not None
            delta[n,stream]=v['gains'];eligible[n]=v['eligible'];calls+=v['calls'];prefix+=v['redundant_prefix_calls']
    halves=[delta[:,:8].mean(1),delta[:,8:].mean(1)];gains=[];choices=[]
    for train,test in ((0,1),(1,0)):
        choice=np.concatenate((np.zeros((48,1)),halves[train]),1).argmax(1)
        evaluated=np.concatenate((np.zeros((48,1)),halves[test]),1)[np.arange(48),choice]
        gains.append(evaluated);choices.append(choice)
    cross=np.mean(gains,0);random=delta.mean((1,2));effects={}
    for name,d,hs in [('Crossfit-Base',cross,gains),('Crossfit-Random',cross-random,[gains[0]-halves[1].mean(1),gains[1]-halves[0].mean(1)])]:
        stat=interval(d,rows);stat['half_means']=[float(h.mean()) for h in hs]
        stat['passed']=bool(stat['mean']>=.0001 and stat['lower']>0 and min(stat['half_means'])>0);effects[name]=stat
    a,b=halves[0][eligible],halves[1][eligible];means=delta.mean(1)[eligible]
    within=float(delta.var(1,ddof=1)[eligible].mean());between=float(np.var(means,ddof=1));signal=max(0.,between-within/16)
    signmask=(abs(a)>1e-12)&(abs(b)>1e-12)
    preflight=json.loads((RUN/'PREFLIGHT.json').read_text())['calls'] if (RUN/'PREFLIGHT.json').exists() else 0
    result=dict(effects=effects,passed=all(s['passed'] for s in effects.values()),states=48,streams=16,
        eligible_actions=int(eligible.sum()),calls=calls,contract_calls=798,preflight_calls=preflight,total_calls=calls+798+preflight,
        redundant_prefix_calls=prefix,actual_trajectories=calls//600,
        half_mean_correlation=float(np.corrcoef(a,b)[0,1]),nonzero_sign_pairs=int(signmask.sum()),
        nonzero_sign_agreement=float(((a[signmask]>0)==(b[signmask]>0)).mean()),
        within_action_variance=within,between_means_variance=between,corrected_signal_variance=signal,
        single_label_reliability=signal/(signal+within) if signal+within else 0.,
        mean16_reliability=signal/(signal+within/16) if signal+within else 0.,
        optimistic_full16_oracle=float(np.maximum(0,delta.mean(1).max(1)).mean()),
        random_mean=float(random.mean()),stage_crossfit={str(100+2*s):float(cross[[j for j,r in enumerate(rows) if r['id'][3]==s]].mean()) for s in old.STEPS},
        development_only=True,no_deployable_selector=True)
    save_torch(RUN/'aggregates.pt',dict(delta=delta,eligible=eligible,choices=choices,crossfit=cross,random=random))
    assert calls<=1843200
    return result


def report(result):
    original=json.loads((ROOT/'docs/revision/long_value/VERIFICATION.json').read_text())['original_weights']
    for p,h in original.items():assert sha256(ROOT/p)==h
    write_json(OUT/'VERIFICATION.json',dict(original_weights=original,cases=768,source_identity=verify(),prefix_assertions=True,paired_both_rng=True))
    lines=['# 标签重复性审查','',
        '跨随机流可利用潜力通过预设门槛；仍未证明决策前可预测或可部署。' if result['passed'] else '跨随机流可利用潜力未通过预设联合门槛。停止当前长期终局标签路线的局部训练搜索。','',
        '| 比较 | 平均效用差 | 97.5%区间 | 两个评价方向 | 通过 |','|---|---:|---|---|---|']
    for name,e in result['effects'].items():lines.append(f"| {name} | {e['mean']:+.7f} | [{e['lower']:+.7f},{e['upper']:+.7f}] | {e['half_means']} | {e['passed']} |")
    lines+=['',f"48个固定状态，每状态16个后续随机流；{result['eligible_actions']}/144个动作合格。未根据历史标签选择状态或动作。",
        f"两半动作均值相关={result['half_mean_correlation']:.4f}，非零符号一致率={result['nonzero_sign_agreement']:.4f}（{result['nonzero_sign_pairs']}对）。",
        f"方差分解的单标签可靠性估计={result['single_label_reliability']:.4f}，16次均值可靠性={result['mean16_reliability']:.4f}。这些是描述性总体估计，不是单状态可预测性或泛化保证。",
        f"全部16流事后选优={result['optimistic_full16_oracle']:.7f}，仍有选择乐观性；主结果只用独立另一半流评价。",
        f"实际调用{result['total_calls']:,}（含798次独立契约核验及{result['preflight_calls']}次预检调用），计算耗时{result['seconds']/60:.2f}分钟。全部原始分支保存，初始前缀/第一点/插入点/双RNG配对逐条核验。",
        '', '0.0001是诊断门槛，不是历史完整搜索的0.005门槛。结果限于现有开发配方、抽中状态、三个动作与固定后续策略。未训练新模型；未改变原解锁权重。',
        '下一步执行另行冻结的少样本先验价值对照；本轮结果不允许宣称原residual或离线在线融合必要。']
    (OUT/'CONCLUSIONS.zh-CN.md').write_text('\n'.join(lines)+'\n')


def main():
    torch.set_num_threads(1);RUN.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
    if (RUN/'COMPLETE.json').exists():
        print('Already complete; verifying and reporting only.');result=json.loads((RUN/'COMPLETE.json').read_text());report(result);return
    start=time.perf_counter()
    if not (RUN/'identity.json').exists():
        old.verify();write_json(RUN/'identity.json',dict(version='repeatability_v1',sources={p:sha256(ROOT/p) for p in SOURCES},
            old_identity=sha256(old.RUN/'identity.json'),old_dataset=sha256(old.RUN/'dataset.pt')))
        write_json(RUN/'SELECTION.json',selection())
    verify();assert json.loads((RUN/'SELECTION.json').read_text())==selection();contracts()
    available=old.parent.parent.available_gib();workers=32 if available>=26 else 24 if available>=22 else 16;assert available>=17
    write_json(RUN/'RESOURCES.json',dict(workers=workers,available_gib=available,threads=1,device='cpu'))
    jobs=[(n,r) for n in range(48) for r in range(16)]
    with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn')) as pool:
        fs=[pool.submit(collect,j) for j in jobs]
        for n,f in enumerate(as_completed(fs),1):
            f.result();write_json(RUN/'STATUS.json',dict(status='running',completed=n,total=768,workers=workers))
    result=summarize();result.update(seconds=time.perf_counter()-start,all_workers_joined=True)
    write_json(RUN/'COMPLETE.json',result);write_json(OUT/'RESULTS.json',result);report(result)
    write_json(RUN/'STATUS.json',dict(status='complete'));print(json.dumps(result),flush=True)


if __name__=='__main__':main()
