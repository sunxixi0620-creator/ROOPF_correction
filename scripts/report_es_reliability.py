"""Verify and archive the completed fixed-center audit; never run simulations."""
from pathlib import Path
import sys, json, zipfile
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import es_reliability as study
from roopf.experiment_io import CaseStore, sha256, fingerprint, write_json


def main():
    run, out = study.RUN, study.OUT
    result = json.loads((run / 'COMPLETE.json').read_text())
    identity = study.verify()
    store = CaseStore(run / 'cases', identity)
    counts, calls = {}, {}
    matrix = {g: np.zeros((3, 36, 2, 4)) for g in ('Base', 'A', 'B')}
    for path in sorted((run / 'cases').glob('*.json')):
        if path.name == 'identity.json':
            continue
        spec = json.loads(path.read_text())['case']
        assert path.stem == fingerprint(spec)[:32]
        value = store.load(path.stem, spec)
        role = spec['role']
        counts[role] = counts.get(role, 0) + 1
        calls[role] = calls.get(role, 0) + value['main_calls']
        assert value['teacher_calls'] == 0
        assert torch.isfinite(value['trail']).all()
        assert len(value['points_sha']) == 64
        if role == 'transfer':
            s, g = spec['seed'], spec['group']
            if g != 'Base':
                assert spec['step_hash'] == sha256(run / f'step_s{s}_{g}.pt')
            matrix[g][s, spec['fid'], spec['instance']] = value['utility'].numpy()
    assert counts == {'estimation': 576, 'transfer': 648}
    assert calls == {'estimation': 691200, 'transfer': 1555200}
    saved = torch.load(run / 'utilities.pt', weights_only=False)
    for g in matrix:
        assert np.array_equal(matrix[g], saved[g])
    for g in ('A', 'B'):
        assert study.interval(matrix[g] - matrix['Base']) == result['effects'][g]
    original = {
        'checkpoints/anchor_policy_d10.pt': 'a265d448ad4a2d191eed23dafa36e03e9475a7b95e824bb04d9a88715031a06d',
        'checkpoints/residual_selector_generated36_d10.pt': '42bac7c6792b6980ad149eff08ce4b2d93ce3ffba51d3d869dc0d1625b3a9d01',
    }
    for p, h in original.items():
        assert sha256(ROOT / p) == h
    write_json(out / 'VERIFICATION.json', dict(counts=counts, calls=calls,
        resource_calls=result['resources']['calls'], total_calls=sum(calls.values()) + result['resources']['calls'],
        original_weights=original, case_hashes_verified=True, statistics_recomputed=True))
    rows = []
    for g, e in result['effects'].items():
        seeds = ', '.join(f'{x:+.8f}' for x in e['seed_means'])
        rows.append(f"| {g} − Base | {e['mean']:+.8f} | [{e['lower']:+.8f}, {e['upper']:+.8f}] | {seeds} | {'通过' if e['passed'] else '未通过'} |")
    passed = result['locally_reliable']
    conclusion = ('两组更新均通过局部可靠性门槛；这只支持当前点的一步更新有效，不能证明收敛。后续优先研究提案的长期互补价值。' if passed else
        '两组更新没有同时通过局部可靠性门槛。当前全参数、少方向 ES 的稳定一步改进能力未获支持；不能据此认定网络结构或离线融合没有潜力。下一步优先限定一个低维更新方案，先检查更新可靠性，再决定是否完整重训。')
    text = f'''# 固定模型位置的 ES 更新可靠性审计

{conclusion}

三个终局训练选中模型（更新 0/20/10）分别估计 A/B 两组独立方向。每组 8 个正负配对扰动，全部 58,504 个参数，归一化扰动尺度 0.02；使用新建 Adam 做一步更新，学习率 0.01、梯度范数裁剪 1。此处不延续原 Adam 动量，因此审计的是局部新优化器更新，不是原训练下一步的精确复现。

在另一组新实例上固定比较 Base、A、B：36 个配置 × 2 个实例 × 4 条轨迹 × 3 个训练种子。每条 20D、600 次真实评估。所有更新均报告，不按迁移结果选取较好的一组。

| 对照 | 平均终局效用差 | 97.5% 区间 | 三个种子均值 | 门槛 |
|---|---:|---|---|---|
{chr(10).join(rows)}

区间按 12 个生成配方聚类重采样 5,000 次，对两个比较使用 Bonferroni 调整。每组须平均增益至少 0.0001、区间下界大于零且三个种子均正；两组须同时通过。这是更新诊断门槛，不能替代论文方法 0.005 的实质增益要求。效用差不是百分比。

两组梯度余弦接近零本身不是失败判据：高维、少方向估计即使含有有效信号，也可能接近正交。本次证据仅限三个模型位置和固定步长，不能单独区分方向数、步长、优化器与任务噪声的责任。

资源实测：16/24/32 个 CPU 进程分别约 9.36/9.91/10.91 条轨迹每秒；32 进程较 16 进程吞吐提升约 16.6%，相同任务轨迹指纹一致。WSL 可见 32 逻辑 CPU、约 62 GiB 内存。单个小批量完整搜索 CPU 约 1.82 秒，CUDA 约 7.13 秒；本次统一使用 CPU 32 进程。GPU 结果只用于单任务计时，不是全面的 GPU 性能结论。

成本：方向估计 691,200 次、迁移检查 1,555,200 次、资源计时 232,800 次，合计 2,479,200 次真实评估；额外教师标签评估为零。科学部分 1,224 批、3,744 条轨迹。已核验逐批数据哈希、身份、预算并从逐批文件重算统计。保存 best-value 轨迹和评估点指纹，没有保存完整评估坐标。

本轮仍使用既有程序生成函数族的新实例，属于开发诊断，不是独立外部验证。原解锁模型两个权重哈希不变。本轮未追加网络结构、训练变体或超参数搜索。
'''
    (out / 'CONCLUSIONS.zh-CN.md').write_text(text)
    artifact = ROOT / 'artifacts/es_reliability_v1'
    artifact.mkdir(parents=True, exist_ok=True)
    files = [p for p in run.rglob('*') if p.is_file() and p.suffix != '.lock']
    files += [ROOT / p for p in study.SOURCES] + [Path(__file__)]
    files += list(out.glob('*'))
    archive = artifact / 'audit.zip'
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(set(files)):
            z.write(p, p.relative_to(ROOT))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
    write_json(artifact / 'MANIFEST.json', dict(archive=archive.name, sha256=sha256(archive),
        bytes=archive.stat().st_size, members=len(set(files))))
    (artifact / 'README.md').write_text('ES 更新可靠性审计归档。audit.zip 保存源代码、协议、方向/更新、逐批数据与统计；父模型见 ../terminal_training_v1。\n\n复核：`.venv/bin/python scripts/report_es_reliability.py`（不会重新运行实验）。\n')
    readme = ROOT / 'README.md'
    marker = '**ES 更新可靠性审计已完成。**'
    if marker not in readme.read_text():
        body = readme.read_text()
        first, rest = body.split('\n', 1)
        readme.write_text(first + '\n\n' + marker + ' ' + conclusion + ' [结果与资源实测](docs/revision/es_reliability/CONCLUSIONS.zh-CN.md) · [归档](artifacts/es_reliability_v1/README.md)。\n' + rest)
    print(json.dumps(result['effects'], indent=2))


if __name__ == '__main__':
    main()
