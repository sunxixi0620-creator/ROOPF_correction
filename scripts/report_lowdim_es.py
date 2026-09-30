"""Read-only scientific verification, reporting and archival of lowdim ES audit."""
from pathlib import Path
import sys, json, zipfile, platform
from collections import Counter
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import lowdim_es as study
from roopf.experiment_io import CaseStore, sha256, fingerprint, write_json


def main():
    torch.set_num_threads(1)
    run, out = study.RUN, study.OUT
    if not (run / 'COMPLETE.json').exists():
        raise SystemExit('Study incomplete: no report or simulation started.')
    result = json.loads((run / 'COMPLETE.json').read_text())
    write_json(out / 'ENVIRONMENT.json', dict(python=sys.version, platform=platform.platform(),
        torch=str(torch.__version__), numpy=np.__version__, cuda=torch.version.cuda))
    identity = study.verify()
    store = CaseStore(run / 'cases', identity)
    counts, calls = Counter(), Counter()
    matrices = {g: np.zeros((3, 36, 2, 4)) for g in study.GROUPS}
    values, initial = {}, {}
    for path in sorted((run / 'cases').glob('*.json')):
        if path.name == 'identity.json':
            continue
        spec = json.loads(path.read_text())['case']
        key = path.stem
        assert key == fingerprint(spec)[:32]
        value = store.load(key, spec)
        assert value is not None
        values[key] = value
        role, s, f, i = spec['role'], spec['seed'], spec['fid'], spec['instance']
        counts[role] += 1
        calls[role] += value['main_calls']
        assert value['main_calls'] == (2400 if role == 'transfer' else 1200)
        assert value['teacher_calls'] == 0
        assert torch.isfinite(value['trail']).all() and torch.isfinite(value['utility']).all()
        assert len(value['points_sha']) == 64
        pair = (role, f, i)
        init = (value['initial_x_sha'], value['initial_y_sha'])
        if pair in initial:
            assert initial[pair] == init
        initial[pair] = init
        if role == 'transfer':
            if spec['step']:
                assert spec['step_hash'] == sha256(run / spec['step'])
            matrices[spec['group']][s, f, i] = value['utility'].numpy()
    assert counts == dict(contract=2, estimation=1152, transfer=1080)
    assert calls == dict(contract=2400, estimation=1382400, transfer=2592000)
    assert sum(calls.values()) == result['total_calls']
    # Recompute every estimator and Adam step from the committed scientific cases.
    fids = np.random.default_rng(20261003).permutation(36)[:6].tolist()
    for s in range(3):
        for m in study.METHODS:
            for g in ('A', 'B'):
                diffs = []
                for d in range(8):
                    signed = []
                    for sign in (1, -1):
                        utilities = []
                        for f in fids:
                            spec = dict(role='estimation', seed=s, method=m, group=g,
                                        direction=d, sign=sign, fid=f, instance=0)
                            utilities.append(float(values[fingerprint(spec)[:32]]['utility'].mean()))
                        signed.append(np.array(utilities))
                    diffs.append((signed[0]-signed[1]).mean())
                grad = sum(float(diffs[d])*study.noise(s,m,g,d) for d in range(8))/.32
                assert torch.equal(grad, torch.load(run/f'gradient_{s}_{m}_{g}.pt', weights_only=False))
                phi = torch.nn.Parameter(torch.zeros_like(grad))
                opt = torch.optim.Adam([phi], lr=.01)
                phi.grad = -grad
                torch.nn.utils.clip_grad_norm_([phi],1)
                opt.step()
                assert torch.equal(phi.detach(), torch.load(run/f'step_{s}_{m}_{g}.pt', weights_only=False))
    stored = torch.load(run / 'utilities.pt', weights_only=False)
    for group in study.GROUPS:
        assert np.array_equal(matrices[group], stored[group])
    for g, stats in result['primary'].items():
        assert study.interval(matrices[g]-matrices['Base']) == stats
    for comparison, stats in result['descriptive'].items():
        left, right = comparison.split('-')
        assert study.interval(matrices[left]-matrices[right], .95) == stats
    assert result['low_reliable'] == all(x['passed'] for x in result['primary'].values())
    original = json.loads((ROOT / 'docs/revision/es_reliability/VERIFICATION.json').read_text())['original_weights']
    for p, h in original.items():
        assert sha256(ROOT/p) == h
    write_json(out / 'VERIFICATION.json', dict(counts=dict(counts), calls=dict(calls), total_calls=sum(calls.values()),
        case_hashes_verified=True, initial_populations_and_values_matched=True,
        gradients_and_steps_recomputed=True, statistics_recomputed=True, original_weights=original))
    rows = []
    for name, stats in result['primary'].items():
        seeds = ', '.join(f'{x:+.8f}' for x in stats['seed_means'])
        rows.append(f"| {name} − Base | {stats['mean']:+.8f} | [{stats['lower']:+.8f}, {stats['upper']:+.8f}] | {seeds} | {'通过' if stats['passed'] else '未通过'} |")
    secondary = []
    for name, stats in result['descriptive'].items():
        secondary.append(f"| {name} | {stats['mean']:+.8f} | [{stats['lower']:+.8f}, {stats['upper']:+.8f}] |")
    if result['low_reliable']:
        conclusion = '200 参数低维更新的两组方向均通过预设局部可靠性门槛。可进入另行冻结协议的有界重训；本轮没有证明最终融合优于纯在线，也没有证明低维优于全参数。'
        next_step = '下一步固定此 200 参数方案和选模规则，进行有界终局重训；只有冻结后的独立确认通过，才扩大验证。不要按本轮描述性对照挑选方向或改秩。'
    else:
        conclusion = '200 参数低维更新没有同时通过两组方向的局部可靠性门槛。按冻结协议停止该方案，不追加完整重训或秩/步长搜索。'
        next_step = ('下一步转向一个明确的机制问题：离线经验能否提出当前在线搜索尚未覆盖、且能改善后续终局收益的区域。'
                     '先固定区域提案的定义，与等数量、等评估预算的非学习探索作对照；标签须基于完整后续搜索价值，不能只用即时改进。'
                     '本轮结果不足以否定其他低维参数化或所有终局训练方法，也不能直接证明探索重设计会成功。')
    minutes = result['seconds']/60
    report = f'''# 低维输出适配与全参数 ES 的匹配诊断

{conclusion}

从三个终局训练选中模型（更新 0/20/10）出发。Low 固定主干和上下文隐藏层，仅调整上下文输出层：固定正交基 Q∈R^(200×4)，40×4 系数矩阵 C 形成权重增量 C Qᵀ，另调整 40 个输出偏置，共 200 个优化参数。零增量逐参数、逐轨迹与原模型一致。Full 使用全部 58,504 个参数；两者为研究候选，不替换原解锁版。

每种方案、每个模型种子估计独立 A/B 两组方向；每组 8 个正负扰动、尺度 0.02，使用相同 6 个配置及初始种群估计终局效用方向。随后从同一模型位置分别使用新 Adam 做一步更新（学习率 0.01，梯度范数裁剪 1）。不继承原优化器动量。相同每坐标尺度和步长不等于相同总扰动范数；低维方案还限制了可表达的更新，因此不能将差异纯粹归因于维数。

迁移检查使用新的实例，36 配置 × 2 实例 × 4 种群 × 3 模型种子；比较 Base、Low_A/B、Full_A/B，全部 20D/600 NFE。没有按结果选取更新方向。

| 主要对照 | 平均终局效用差 | 97.5% 区间 | 三个种子均值 | 局部门槛 |
|---|---:|---|---|---|
{chr(10).join(rows)}

主要门槛：两组均须平均增益≥0.0001、Bonferroni 调整后的区间下界>0、三个种子均值均>0。12 个生成配方聚类重采样 5,000 次。该门槛是训练更新诊断，不是论文方法的 0.005 实质增益门槛；效用差不是百分比。

| 描述性对照 | 平均终局效用差 | 未调整的 95% 区间 |
|---|---:|---|
{chr(10).join(secondary)}

上表用于解释结果，不能以未调整的多个对照宣称确认性优越性。A/B 在 Low 与 Full 之间并非同一参数方向，只匹配分配、任务、种群和搜索随机流。余弦值不作为可靠性门槛。

资源与成本：CPU {result['workers']} 个进程，每进程 1 线程，复用上一轮吞吐测试的配置。本轮耗时约 {minutes:.2f} 分钟。方向估计 1,152 批/2,304 条轨迹；迁移检查 1,080 批/4,320 条轨迹；零偏移契约 2 批/4 条轨迹。共 3,976,800 次真实函数评估，额外教师评估 0 次。逐批数据身份、哈希、初始种群/目标值配对通过；全部梯度、Adam 更新及统计已从原始批次重算。存储 best-value 轨迹和评估点指纹，不含完整评估坐标。

{next_step}

本轮是既有程序生成函数族的新实例开发证据，覆盖三个已选模型位置与一个固定更新设置；不是完全未见函数族测试，不能代替 F 对 A/O 的最终对照，也不证明 residual 或保护门控的必要性。原解锁模型两个权重哈希保持不变。
'''
    (out / 'CONCLUSIONS.zh-CN.md').write_text(report)
    (out / 'NEXT_DECISION.zh-CN.md').write_text('# 下一步研究决策\n\n' + next_step + '\n\n本轮所有结果均保留，不挑选种子或方向。\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels = ['Low A − Base', 'Low B − Base', 'Full A − Base', 'Full B − Base']
    stats = list(result['primary'].values()) + [result['descriptive'][x] for x in ('Full_A-Base','Full_B-Base')]
    fig, ax = plt.subplots(figsize=(7.5,3.6))
    for i, st in enumerate(stats):
        ax.plot([st['lower'],st['upper']],[i,i], color='#2166ac' if i<2 else '#777777', lw=2)
        ax.plot(st['mean'],i,'o',color='#2166ac' if i<2 else '#777777')
    ax.axvline(0,color='black',lw=.7)
    ax.set_yticks(range(4),labels)
    ax.invert_yaxis()
    ax.set_xlabel('Paired terminal utility difference (higher is better)')
    ax.set_title('Low: adjusted 97.5% intervals; Full: descriptive 95% intervals')
    fig.tight_layout()
    fig.savefig(out/'effects.png',dpi=180)
    fig.savefig(out/'effects.pdf')
    plt.close(fig)
    artifact = ROOT/'artifacts/lowdim_es_v1'
    artifact.mkdir(parents=True,exist_ok=True)
    files = [p for p in run.rglob('*') if p.is_file() and p.suffix != '.lock']
    files += [ROOT/p for p in study.SOURCES]+[Path(__file__)]+list(out.glob('*'))
    archive = artifact/'audit.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(set(files)):
            z.write(p,p.relative_to(ROOT))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
    write_json(artifact/'MANIFEST.json',dict(archive=archive.name,sha256=sha256(archive),bytes=archive.stat().st_size,members=len(set(files))))
    (artifact/'README.md').write_text('低维/全参数 ES 匹配诊断归档。audit.zip 保存源代码、协议、更新方向、逐批轨迹及报告。父模型见 ../terminal_training_v1；父审计见 ../es_reliability_v1。\n\n复核：`.venv/bin/python scripts/report_lowdim_es.py`，不会重跑实验。\n')
    readme=ROOT/'README.md'
    marker='**低维 ES 匹配诊断已完成。**'
    if marker not in readme.read_text():
        first,rest=readme.read_text().split('\n',1)
        readme.write_text(first+'\n\n'+marker+' '+conclusion+' [详细结果](docs/revision/lowdim_es/CONCLUSIONS.zh-CN.md) · [归档](artifacts/lowdim_es_v1/README.md)。\n'+rest)
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
