"""Verify heldout grouped value diagnostics and archive all evidence."""
from pathlib import Path
import sys,json,platform,zipfile
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import long_value as study
from roopf.experiment_io import CaseStore,sha256,fingerprint,write_json
from scripts.unified_revision import bounded


def main():
    torch.set_num_threads(1);run,out=study.RUN,study.OUT
    if not (run/'COMPLETE.json').exists():raise SystemExit('Study incomplete; no simulations or report started.')
    result=json.loads((run/'COMPLETE.json').read_text());identity=study.verify();store=CaseStore(run/'cases',identity)
    assert len(list((run/'cases').glob('*.pt')))==288
    total=0;prefix=0;eligible=0;feature_states=0
    dataset=torch.load(run/'dataset.pt',weights_only=False)
    assert sha256(run/'dataset.pt')==result['costs']['dataset_sha']
    by_id={tuple(k):j for j,k in enumerate(dataset['id'])};assert len(by_id)==576
    for f in range(36):
        for i in range(2):
            for p in range(4):
                v=store.load(f'f{f}_i{i}_p{p}',dict(fid=f,instance=i,population=p));base=v['base']
                assert base['main_calls']==600 and v['off_rollout_calls']==0
                init=base['initial'].min(1).values;scale=base['initial'].std(1);calls=600;pc=0
                for st in v['states']:
                    step=st['step'];snapshot=st['snapshot'];j=by_id[(f,i,p,step)]
                    assert snapshot['fingerprint']==fingerprint({k:a for k,a in snapshot.items() if k!='fingerprint'})
                    points,metadata=study.candidates(snapshot,base,step,f,i,p)
                    assert torch.equal(points,st['points']) and metadata==st['metadata']
                    task,_,_=study.setup(f,i,p);geo,fused,proxy=study.features(snapshot,points,metadata,task)
                    assert task.points==0 and task.diagnostic_points==0
                    assert fingerprint((geo,fused,proxy))==st['feature_sha'];feature_states+=1
                    for k,a in [('geo',geo),('fusion',fused),('proxy',proxy),('eligible',st['eligible']),('y',st['labels'])]:
                        assert np.array_equal(a.numpy(),dataset[k][j])
                    for k,branch in enumerate(st['branches']):
                        label=float((bounded(init,branch['trail'][:,-1],scale)-bounded(init,base['trail'][:,-1],scale))[0])
                        assert label==float(st['labels'][k])
                        assert torch.equal(branch['trail'][:,:step],base['trail'][:,:step])
                        if st['eligible'][k]:
                            assert branch['calls']==600;eligible+=1;calls+=600;pc+=100+2*step
                        else:
                            assert branch['calls']==0 and torch.equal(branch['trail'],base['trail'])
                            assert branch['points_sha']==fingerprint(base['points'])
                assert calls==v['calls'] and pc==v['redundant_prefix_calls'];total+=calls;prefix+=pc
    assert total==result['costs']['calls'] and total+456==result['costs']['total_calls']
    frozen=json.loads((run/'MODELS_FROZEN.json').read_text());assert len(frozen)==18
    for path,h in frozen.items():assert sha256(run/path)==h
    records=torch.load(run/'decisions.pt',weights_only=False);assert len(records)==9
    model_rows=[]
    for fold in range(3):
        fm=result['costs']['folds'][fold]
        assert not(set(fm['train'])&set(fm['val']) or set(fm['train'])&set(fm['test']) or set(fm['val'])&set(fm['test']))
        assert sha256(run/f'trainval_{fold}.pt')==fm['trainval_sha'] and sha256(run/f'heldout_{fold}.pt')==fm['heldout_sha']
        tv=torch.load(run/f'trainval_{fold}.pt',weights_only=False);test=torch.load(run/f'heldout_{fold}.pt',weights_only=False)
        for name,data in [('train',tv['train']),('val',tv['val']),('test',test)]:
            assert sorted(set((data['id'][:,0]//3).tolist()))==fm[name]
        for seed in range(3):
            rec=next(r for r in records if r['fold']==fold and r['seed']==seed)
            assert np.array_equal(rec['ids'],test['id'])
            for kind in ('geo','fusion'):
                stem=f'model_{fold}_{kind}_{seed}';ck=torch.load(run/(stem+'.pt'),weights_only=False);meta=json.loads((run/(stem+'.json')).read_text())
                raw=tv['train'][kind][tv['train']['eligible']]
                assert np.array_equal(ck['mean'],raw.mean(0)) and np.array_equal(ck['std'],np.maximum(raw.std(0),.001))
                assert ck['trainval_sha']==fm['trainval_sha']
                best=-float('inf');selected=None
                for h in meta['history']:
                    if h['validation_gain']>best+1e-8:best=h['validation_gain'];selected=h['epoch']
                assert selected==meta['selected_epoch']==ck['selected_epoch']
                model=study.net(test[kind].shape[-1]);model.load_state_dict(ck['state']);model.eval()
                with torch.no_grad():pred=model(torch.tensor(np.clip((test[kind]-ck['mean'])/ck['std'],-10,10))).squeeze(-1).numpy()/1000
                assert np.array_equal(pred,rec['choices'][kind]['predictions'])
                dec=study.policy(pred,test)
                assert dec['mask'].sum()==48 and np.array_equal(dec['gain'],rec['outputs'][kind])
                assert np.array_equal(dec['choice'],rec['choices'][kind]['choice'])
                model_rows.append(dict(fold=fold,seed=seed,kind=kind,selected=selected,executed=meta['epochs'],validation_gain=best))
            for method,scores in [('proxy',test['proxy']),('constant',np.zeros_like(test['proxy']))]:
                dec=study.policy(scores,test);assert dec['mask'].sum()==48
                assert np.array_equal(dec['gain'],rec['outputs'][method])
            expected=(test['y']*test['eligible']).sum(1)/test['eligible'].sum(1)*.25
            assert np.array_equal(expected,rec['outputs']['random'])
    arrays=torch.load(run/'effects.pt',weights_only=False)
    for name,stats in result['effects'].items():
        other=name.split('-')[1];delta=arrays['fusion'] if other=='Base' else arrays['fusion']-arrays[other]
        assert study.interval(delta)==stats
    original=json.loads((ROOT/'docs/revision/es_reliability/VERIFICATION.json').read_text())['original_weights']
    for p,h in original.items():assert sha256(ROOT/p)==h
    write_json(out/'VERIFICATION.json',dict(cases=288,feature_states=feature_states,eligible_branches=eligible,
        total_calls=total+456,redundant_prefix_calls=prefix,features_recomputed_without_objective_calls=True,
        labels_recomputed=True,recipe_splits_disjoint_per_model=True,training_only_standardization=True,
        checkpoints_verified=18,heldout_predictions_recomputed=True,matched_quota_verified=True,original_weights=original))
    write_json(out/'ENVIRONMENT.json',dict(python=sys.version,torch=str(torch.__version__),numpy=np.__version__,platform=platform.platform()))
    rows=[]
    for name,e in result['effects'].items():
        seeds=', '.join(f'{x:+.7f}' for x in e['seed_means'])
        rows.append(f"| {name} | {e['mean']:+.7f} | [{e['lower']:+.7f}, {e['upper']:+.7f}] | {seeds} | {'通过' if e['passed'] else '未通过'} |")
    trained=[]
    for r in model_rows:
        trained.append(f"| {r['fold']} | {r['kind']} | {r['seed']} | {r['selected']} | {r['executed']} | {r['validation_gain']:+.7f} |")
    if result['fusion_passed']:
        conclusion='固定配额的分组测试中，融合价值模型通过全部预设识别与在线信息增量门槛。可以进入另行冻结的可部署门控及完整搜索验证；本轮还不是完整融合算法成绩。'
        decision='下一步只固定这一价值模型与特征，使用训练/验证数据确定因果可执行的干预规则，然后在新实例上比较完整搜索、纯在线、纯离线价值策略以及匹配随机干预。测试时不能跨未来状态排序分配配额。'
    elif result['recognition_passed']:
        conclusion='长期价值融合模型通过相对不干预、代理、随机与常数策略的识别门槛，但在线特征相对几何价值模型的增量尚未通过。因此不能据此认定融合必要。'
        decision='保留长期价值可预测性的有限证据，停止扩大“融合必要”的主张。后续完整搜索如继续，必须同时保留几何价值对照并在新实例检验；不按本轮测试结果追加特征或调干预比例。'
    else:
        conclusion='当前长期价值模型未通过预设的提前识别门槛。按协议停止此模型的特征/损失/干预率搜索，不启动大规模重训或完整搜索扩展。'
        decision=('保留全部种子和分组结果。事后可获益分支不能直接转化为可学习的提前选择证据；本轮也不能推出所有长期价值模型均无效。'
            '当前论文只能使用已有受支持结论，不能声称已证明离线与在线双向必要或原 residual 必要。'
            '若继续研究，优先独立检验标签重复性：在同一状态和候选上使用多个后续随机流，与不干预分支配对，区分稳定的期望收益与单次续跑噪声。'
            '须先固定抽样与判据；本轮未执行这项诊断，也不据此追加网络、轮次或特征搜索。')
    means='\n'.join(f"| {m} | {v:+.7f} |" for m,v in result['mean_utilities'].items())
    report=f'''# 长期干预收益的提前识别实验

{conclusion}

使用新的程序生成实例，36配置×2实例×4种群，共288条纯在线 O 基准轨迹；在100/310 NFE的两个时点生成576个状态。每个状态固定随机、种群差分、中心/质心三类菜单，共12个候选。符合几何新颖性条件的候选分别干预一次，继续到600 NFE，标签为相对同一 Base 的终局效用差。本轮没有训练或使用离线神经提案生成器。

Geo使用41个决策前几何/已观察适应度特征；Fusion增加4个在线代理预测特征。网络分别为41/45→64→32→1（SiLU），参数量4,801/5,057。目标为终局收益×1000，MSE，Adam学习率0.001、weight_decay0.0001、batch256。最多200轮，每5轮检查验证集25%配额收益，连续4次未改善早停；允许保留零输出初始化模型。所有特征在分支执行之前提取。

三个外层分组，每组4个测试配方、6个训练配方、2个早停验证配方。同一配方全部尺度、实例、种群与时点在每个模型内属于同一分区。每组/模型3个初始化种子，共18个模型，全部先冻结再测试。标准化只使用训练集合格候选。函数族有历史开发接触，外层训练集也存在重叠，因此下面区间是按12配方聚类得到的开发诊断，不是完全独立最终验证。

测试时先选各状态的最高分合格候选，再在每个测试分组的192个状态中选48个干预。每个种子合计576状态、144次干预。该配额评估需要查看整批无标签特征分数，不是在线可部署的因果门控。Random报告恰好抽K个状态并均匀抽候选的精确期望；Constant使用固定哈希打破零分并列，隔离非学习选择偏好。候选和状态选择不使用测试标签。

| 主要比较 | 平均效用差（含未干预状态的0） | 调整后的99%区间 | 三个初始化种子均值 | 门槛 |
|---|---:|---|---|---|
{chr(10).join(rows)}

预设各比较均须均值≥0.0001、区间下界>0、三个种子均值>0。5个比较作 Bonferroni 调整；前4类(Base/Proxy/Random/Constant)是识别门槛，另加Geo是在线信息增量门槛。效用差不是百分比，也不能替代论文0.005实质收益门槛。

| 策略/描述性诊断 | 平均终局效用差 |
|---|---:|
{means}

`*_all`强制所有状态干预，仅为描述性诊断；`fusion_random_candidates`在融合策略选中的状态内均匀选点，用于区分状态选择与候选选择。不依据这些补充结果更换主配额或策略。

每个状态/候选的标签只来自一次后续随机轨迹。本轮没有分解候选的稳定期望收益与续跑噪声，因此无法把未通过门槛单独归因于模型容量、训练不足或长期价值本身不可预测。

| 外层组 | 模型 | 种子 | 选中轮次 | 实际轮次 | 选模验证收益 |
|---|---|---|---:|---:|---:|
{chr(10).join(trained)}

选模验证收益不是独立证据；选中0轮意味着保留初始化，不应称为学到新知识。

资源：采集并行度见 RESOURCES.json，训练 CPU/CUDA 用合成数据预先计时后只按速度选设备，结果为 `{result['training_resources']['device']}`；计时 `{result['training_resources']['seconds_per_100_steps']}` 秒/100步。计算总耗时约{result['seconds']/60:.2f}分钟。共288条 Base、{eligible}条合格分支，真实调用{total:,}次，加契约核验456次，共{total+456:,}次，其中重复重放前缀{prefix:,}次。全部分支是昂贵的标签采集成本；训练、预测和报告复核没有新增目标调用。

核验：逐批哈希与身份、576个状态的特征无查询重算、终局标签、按配方隔离、训练集标准化、18个checkpoint与测试预测、固定配额及统计均通过。保存完整Base坐标，分支保存best-value轨迹和坐标指纹；前缀及实际插入点在执行时逐条断言。原解锁模型哈希保持不变。

{decision}

这是新的价值模型研究，不能追溯解释为原 residual 已有效，也不是最终 F>A/O 完整搜索对照。
'''
    (out/'CONCLUSIONS.zh-CN.md').write_text(report)
    (out/'NEXT_DECISION.zh-CN.md').write_text('# 后续决策\n\n'+conclusion+'\n\n'+decision+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(7.5,4))
    for n,(name,e) in enumerate(result['effects'].items()):
        ax.plot([e['lower'],e['upper']],[n,n],color='#2166ac',lw=2);ax.plot(e['mean'],n,'o',color='#2166ac')
    ax.axvline(0,color='black',lw=.7);ax.set_yticks(range(5),list(result['effects']));ax.invert_yaxis()
    ax.set_xlabel('Paired terminal utility difference at fixed 25% quota')
    ax.set_title('Grouped development evaluation; adjusted 99% intervals')
    fig.tight_layout();fig.savefig(out/'effects.png',dpi=180);fig.savefig(out/'effects.pdf');plt.close(fig)
    artifact=ROOT/'artifacts/long_value_v1';artifact.mkdir(parents=True,exist_ok=True)
    files=[p for p in run.rglob('*') if p.is_file() and p.suffix!='.lock']+[ROOT/p for p in study.SOURCES]+[Path(__file__)]+list(out.glob('*'))
    groups=[[],[],[],[]]
    for p in sorted(set(files)):
        if p.parent==run/'cases' and p.stem.startswith('f'):groups[1+int(p.stem.split('_')[0][1:])//12].append(p)
        else:groups[0].append(p)
    manifest=[]
    for i,paths in enumerate(groups):
        archive=artifact/f'audit_{i}.zip'
        with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
            for p in paths:z.write(p,p.relative_to(ROOT))
        with zipfile.ZipFile(archive) as z:assert z.testzip() is None
        assert archive.stat().st_size<95_000_000
        manifest.append(dict(archive=archive.name,bytes=archive.stat().st_size,sha256=sha256(archive),members=len(paths)))
    write_json(artifact/'MANIFEST.json',manifest)
    (artifact/'README.md').write_text('长期价值预测实验完整归档。四个ZIP解压到同一仓库根目录。audit_0含数据集、划分、18个模型、协议/脚本和报告；其余按函数分片保存原始分支数据。父机制试验见 ../region_potential_v1。\n\n复核：`.venv/bin/python scripts/report_long_value.py`，不调用目标函数或重新训练。\n')
    readme=ROOT/'README.md';marker='**长期价值提前识别实验已完成。**'
    if marker not in readme.read_text():
        first,rest=readme.read_text().split('\n',1)
        readme.write_text(first+'\n\n'+marker+' '+conclusion+' [详细结果](docs/revision/long_value/CONCLUSIONS.zh-CN.md) · [归档](artifacts/long_value_v1/README.md)。\n'+rest)
    print(json.dumps(dict(effects=result['effects'],recognition_passed=result['recognition_passed'],fusion_passed=result['fusion_passed'],total_calls=total+456),indent=2))


if __name__=='__main__':main()
