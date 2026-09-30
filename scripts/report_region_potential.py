"""Verify and archive completed region potential diagnostic; no objective calls."""
from pathlib import Path
import sys,json,zipfile,platform
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import region_potential as study
from roopf.experiment_io import CaseStore,sha256,fingerprint,write_json
from scripts.unified_revision import bounded


def main():
    torch.set_num_threads(1);run,out=study.RUN,study.OUT
    if not (run/'COMPLETE.json').exists():raise SystemExit('Incomplete study: no reporting or simulation started.')
    result=json.loads((run/'COMPLETE.json').read_text());identity=study.verify();store=CaseStore(run/'cases',identity)
    assert len(list((run/'cases').glob('*.pt')))==72
    calls=0;branches=0;ineligible=0;prefix_calls=0;eligible_by_menu={m:0 for m in study.MENUS}
    for f in range(36):
        for p in range(2):
            case=store.load(f'f{f}_p{p}',dict(fid=f,population=p));assert case is not None
            base=case['base'];assert base['main_calls']==600 and base['points'].shape==(1,500,20)
            assert base['trail'].shape==(1,250) and torch.isfinite(base['trail']).all()
            assert case['off_rollout_teacher_calls']==0
            init=base['initial'].min(1).values;scale=base['initial'].std(1)
            case_calls=600;case_prefix=0
            for step in study.STEPS:
                state=case['snapshots'][step]
                assert state['fingerprint']==fingerprint({k:v for k,v in state.items() if k!='fingerprint'})
                expected_best=init[0] if step==0 else base['trail'][0,step-1]
                assert torch.equal(state['fitness'][0,0],expected_best)
                known=torch.cat((base['initial_x'][0],base['points'][0,:2*step],state['candidates'][0]),0)
                for m in study.MENUS:
                    menu=case['menus'][step][m]
                    for k in range(4):
                        row=case['branches'][f'{step}_{m}_{k}'];meta=menu['metadata'][k];point=menu['points'][k]
                        distance=float(((known-point).square().mean(1).sqrt()/10).min())
                        assert distance==meta['min_distance'] and row['eligible']==meta['eligible']
                        assert torch.isfinite(point).all() and (point.abs()<=5).all()
                        assert len(row['points_sha'])==64
                        assert torch.equal(row['trail'][:,:step],base['trail'][:,:step])
                        for metric,at in [('immediate',step),('after20',min(step+10,249)),('after60',min(step+30,249)),('final',249)]:
                            gain=float((bounded(init,row['trail'][:,at],scale)-bounded(init,base['trail'][:,at],scale))[0])
                            assert gain==row[metric]
                        if row['eligible']:
                            assert distance>=.05 and row['main_calls']==600
                            branches+=1;case_calls+=600;case_prefix+=100+2*step;eligible_by_menu[m]+=1
                        else:
                            assert row['main_calls']==0 and row['points_sha']==fingerprint(base['points'])
                            assert torch.equal(row['trail'],base['trail']);ineligible+=1
            assert case_calls==case['main_calls'] and case_prefix==case['redundant_prefix_calls']
            assert case_calls-600==case['branch_calls']
            calls+=case_calls;prefix_calls+=case_prefix
    recomputed=study.summarize()
    for k,v in recomputed.items():assert result[k]==v
    assert branches+ineligible==3456 and calls+456==result['total_calls']
    assert calls//600==result['actual_trajectories']
    original=json.loads((ROOT/'docs/revision/es_reliability/VERIFICATION.json').read_text())['original_weights']
    for p,h in original.items():assert sha256(ROOT/p)==h
    write_json(out/'VERIFICATION.json',dict(cases=72,eligible_branches=branches,ineligible_actions=ineligible,
        main_calls=calls,contract_calls=456,total_calls=calls+456,redundant_prefix_calls=prefix_calls,
        hashes_and_snapshots_verified=True,novelty_recomputed=True,all_gains_recomputed=True,
        statistics_recomputed=True,original_weights=original,
        branch_prefix_coordinates='asserted during execution; archived branch points are fingerprints only'))
    write_json(out/'ENVIRONMENT.json',dict(python=sys.version,torch=str(torch.__version__),numpy=np.__version__,platform=platform.platform()))
    primary=[]
    for name,e in result['effects'].items():
        seeds=', '.join(f'{x:+.7f}' for x in e['seed_means'])
        primary.append(f"| {name} | {e['mean']:+.7f} | [{e['lower']:+.7f}, {e['upper']:+.7f}] | {seeds} | {'通过' if e['passed'] else '未通过'} |")
    menus=[]
    for m,d in result['diagnostics'].items():
        menus.append(f"| {m} | {result['menu_potential_mean'][m]:+.7f} | {result['menu_all_candidates_mean'][m]:+.7f} | {d['eligible']}/{d['actions']} | {d['delayed_positive_states']}/{d['positive_states']} |")
    if result['passed']:
        conclusion='本轮方向外推区域候选通过预设潜力门槛；证据支持继续设计训练实验，但尚未证明实际可学习选择器或完整融合方法有效。'
        next_step='固定当前菜单和控制策略，在训练专用实例采集分支终局标签，使用单独验证集训练能够在评估前选点/弃用的模型。冻结后必须与同预算纯在线、纯离线和非学习探索做完整搜索对照；不得将本轮事后最优值当作部署成绩。'
    elif result['effects']['Offline-Base']['passed']:
        conclusion='离线方向候选存在达到预设门槛的事后长期潜力，但没有通过相对所有非学习探索的增量验收。当前证据不足以支撑对该生成器继续重训。'
        next_step=('停止当前方向外推方案的重训和参数搜索。若继续方法研究，下一项应检验长期干预收益是否能由决策前信息预测：'
            '固定简单探索候选，使用完整分支终局收益作为标签，按生成配方隔离训练与验证，比较离线价值预测、在线代理评分和随机干预。'
            '必须固定并匹配干预次数，避免把少干预的收益误认成识别能力；没有提前识别能力就停止，不能从事后最佳值直接跳到大规模重训。'
            '这属于新的长期价值学习问题，区别于此前以即时改进为标签的排序校准；需另行冻结协议，本轮没有启动。'
            '只有可预测性成立，才在新实例完整搜索中验证融合、纯在线和纯离线的差异。当前结果仍不能证明离线知识或原 residual 的必要性。')
    else:
        conclusion='本轮离线方向外推候选未通过预设长期潜力门槛，整体可行性验收未通过。按协议停止此方案，不追加半径/时点搜索或完整重训。'
        next_step='停止当前方向外推路线并保留全部负面结果。当前证据支持的论文主张应与已有 anchor 增强结果对齐；离线与在线双向必要性仍未证明。若继续新方法研究，应明确不同的信息来源或任务结构，不能把这一候选的失败泛化为全部探索方法都无效，也不应继续围绕同一生成器追加小幅变体。'
    report=f'''# 新区域方向的长期价值诊断

{conclusion}

本轮使用纯在线 O 作为共同起点与后续策略，20D/600 NFE。36 个配置、2 个独立初始种群，共72条 Base 轨迹。在预先固定的100和310 NFE处，保留第一个在线评估点，用一个探索点替换第二个，然后继续运行到600 NFE。

离线信息来自三个冻结的终局训练选中模型（更新0/20/10）。每个模型提供2个方向，每个方向使用1和2个原始坐标单位的 RMS 位移（归一化半径0.1/0.2），共4个点；网络没有在本轮重新训练。对照分别使用随机高斯方向 Iso、当前种群差分 Pop，以及边界中心/当前种群质心方向 Center，每组同样4个点和同样半径。

候选到全部已评估点及当前38个在线候选的最小归一化 RMS 距离须≥0.05。几何不合格者弃用，直接复用 Base；不重采样，不查看目标真值来决定是否合格。裁剪后的实际距离可能不同，逐项元数据已保存。几何新颖性不代表一定进入另一个吸引域。

下表的“潜力”是每个状态4个候选完成后续搜索后，事后取最佳终局增益，并允许不干预（0）。这是使用终局真值的诊断标签，不是可部署算法成绩。三类非学习控制同样允许事后选优；控制轨迹在三个离线种子之间复用，没有作为独立样本重复计数。

| 主要潜力对照 | 平均效用差 | 调整后的98.75%区间 | 三个模型种子均值 | 门槛 |
|---|---:|---|---|---|
{chr(10).join(primary)}

12个生成配方聚类重采样5,000次，对4个主要比较作 Bonferroni 调整。Offline−Base平均值须≥0.001；相对每个简单探索控制须≥0.0005；各比较还须区间下界>0且三个模型种子均值>0。所有条件须同时满足。效用差不是百分比；这些是诊断门槛，不是论文融合版0.005实质收益门槛。

| 菜单 | 事后最优潜力均值 | 全部4个候选的平均干预收益 | 合格点/总点数 | 即时未改善的终局赢家/有终局赢家的状态 |
|---|---:|---:|---|---|
{chr(10).join(menus)}

“全部候选平均”包括弃用点的零增益，用于判断有多少收益依赖事后挑选。各时点结果在 RESULTS.json 中完整保留，不按较好时点改方案。

资源与成本：32个 CPU 工作进程配置（实际见 RESOURCES.json），每进程1线程。计算耗时约{result['seconds']/60:.2f}分钟。72条 Base，加{branches}条实际候选分支，另有{ineligible}个不合格动作复用 Base。实际科学轨迹共{result['actual_trajectories']}条，真实调用{calls:,}次；契约核验456次，总计{result['total_calls']:,}次。其中分支标签成本{result['branch_calls']:,}次，包含重复重放前缀{prefix_calls:,}次。没有分支外的额外目标查询，但整批分支显然是昂贵的诊断/离线标签成本，不能藏入单条600预算的部署成绩。

已核验72个逐批数据身份/哈希、全部快照指纹、几何距离、弃用行为、即时及终局收益和汇总统计。前缀坐标与指定插入点在执行时逐条断言；归档保存完整 Base 坐标和快照，分支保存 best-value 轨迹及坐标指纹，未保存每条分支的完整坐标。原解锁权重哈希不变。

{next_step}

本轮只覆盖既有程序生成函数族的新实例、一次区域干预、固定两个时点与两个半径，以及当前三个已训练模型。其事后最大值仅限这些候选和固定后续策略，不是所有探索策略的上界。不能据此宣称完全独立外部泛化、F>A/O成立或 residual/保护模块必要。
'''
    (out/'CONCLUSIONS.zh-CN.md').write_text(report)
    (out/'NEXT_DECISION.zh-CN.md').write_text('# 本轮后续决策\n\n'+conclusion+'\n\n'+next_step+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(7.5,3.8))
    for i,(name,e) in enumerate(result['effects'].items()):
        ax.plot([e['lower'],e['upper']],[i,i],color='#2166ac',lw=2)
        ax.plot(e['mean'],i,'o',color='#2166ac')
    ax.axvline(0,color='black',lw=.7);ax.set_yticks(range(4),list(result['effects']));ax.invert_yaxis()
    ax.set_xlabel('Difference in retrospective terminal potential')
    ax.set_title('Oracle-assisted diagnostic; adjusted 98.75% intervals')
    fig.tight_layout();fig.savefig(out/'effects.png',dpi=180);fig.savefig(out/'effects.pdf');plt.close(fig)
    artifact=ROOT/'artifacts/region_potential_v1';artifact.mkdir(parents=True,exist_ok=True)
    # Split archives by data/configuration to keep every member comfortably below GitHub limit.
    files=[p for p in run.rglob('*') if p.is_file() and p.suffix!='.lock']
    files += [ROOT/p for p in study.SOURCES]+[Path(__file__)]+list(out.glob('*'))
    groups=[[],[],[],[]]
    for p in sorted(set(files)):
        if p.parent==run/'cases' and p.stem.startswith('f'):
            f=int(p.stem.split('_')[0][1:]);groups[1+f//12].append(p)
        else:groups[0].append(p)
    manifests=[]
    for i,paths in enumerate(groups):
        archive=artifact/f'audit_{i}.zip'
        with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
            for p in paths:z.write(p,p.relative_to(ROOT))
        with zipfile.ZipFile(archive) as z:assert z.testzip() is None
        assert archive.stat().st_size<95_000_000
        manifests.append(dict(archive=archive.name,bytes=archive.stat().st_size,sha256=sha256(archive),members=len(paths)))
    write_json(artifact/'MANIFEST.json',manifests)
    (artifact/'README.md').write_text('区域方向潜力诊断归档。四个 ZIP 解压到同一仓库根目录。audit_0 为协议/脚本/统计，其余为按函数分片的快照与轨迹。父权重见 ../terminal_training_v1，父审计见 ../es_reliability_v1。\n\n复核：`.venv/bin/python scripts/report_region_potential.py`（不会调用目标函数或重跑实验）。\n')
    readme=ROOT/'README.md';marker='**区域方向的长期潜力诊断已完成。**'
    if marker not in readme.read_text():
        first,rest=readme.read_text().split('\n',1)
        readme.write_text(first+'\n\n'+marker+' '+conclusion+' [完整结果](docs/revision/region_potential/CONCLUSIONS.zh-CN.md) · [归档](artifacts/region_potential_v1/README.md)。\n'+rest)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
