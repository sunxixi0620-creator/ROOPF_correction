"""Verify and archive the bounded revision; never changes model selection."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import zipfile
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'scripts')]
from roopf.experiment_io import fingerprint, save_torch, sha256, write_json
from roopf.revision_tasks import ProceduralTask, population
from unified_revision import verify


def provenance(run):
    """Save actual parameters plus explicit population fingerprints, not just seeds."""
    torch.set_num_threads(1)
    run, _ = verify(run)
    path = run/'task_parameters.pt'
    if path.exists():
        meta = json.loads((run/'task_parameters.json').read_text())
        assert meta['sha256'] == sha256(path)
        return
    definitions = []
    for split, instances, count in [('anchor_training', range(1, 81), 16),
          ('anchor_validation', range(1), 2), ('residual_training', range(2), 2),
          ('residual_validation', range(1), 2), ('development', range(2), 4)]:
        for fid in range(36):
            for inst in instances:
                task = ProceduralTask(fid, inst, split)
                pop = population(fid, inst, split, count=count)
                definitions.append(dict(split=split, fid=fid, instance=inst,
                    parameter_seed=task.parameter_seed, params=task.params,
                    population_sha256=fingerprint(pop), population_count=count))
    save_torch(path, definitions)
    write_json(run/'task_parameters.json', {'sha256': sha256(path), 'count': len(definitions),
        'includes_unused_post_early_stop_training_instances': True,
        'meaning': 'predeclared task parameters, not extra evaluated independent samples'})


def finalize(run):
    run, identity = verify(run)
    assert (run/'COMPLETE.json').exists()
    checks = []
    # A complete archive must verify all content hashes and case identities.
    for manifest in sorted(run.glob('*_*/*.json')):
        meta = json.loads(manifest.read_text())
        if 'data_sha256' not in meta:
            continue
        data = manifest.with_suffix('.pt')
        assert sha256(data) == meta['data_sha256'], str(data)
        owner = json.loads((manifest.parent/'identity.json').read_text())
        assert fingerprint({'run': owner, 'case': meta['case']}) == meta['identity_sha256']
        checks.append(str(data.relative_to(run)))
    assert len([x for x in checks if x.startswith('labels_')]) == 324
    assert len([x for x in checks if x.startswith('development_')]) == 1296
    frozen = json.loads((run/'SELECTION_FROZEN.json').read_text())
    assert len(frozen) == 6 and all(sha256(run/p) == h for p, h in frozen.items())
    assert all(sha256(ROOT/'checkpoints'/name) == h for name, h in identity['originals'].items())
    for p in run.glob('prefetch_*.json'):
        done = json.loads(p.read_text())
        assert done['status'] == 'complete' and done['all_workers_joined']
    provenance(run)
    decision = json.loads((run/'DECISION.json').read_text())
    out = ROOT/'docs/revision/unified_execution'
    out.mkdir(exist_ok=True)
    for name in ['DECISION.json', 'development.csv', 'identity.json', 'SELECTION_FROZEN.json', 'COMPLETE.json', 'task_parameters.json', 'environment.json', 'seed_roles.json']:
        shutil.copyfile(run/name, out/name)
    summaries = {}
    for stage in ['anchor', 'residual']:
        for seed in range(3):
            key = f'{stage}_{seed}'
            summaries[key] = json.loads((run/key/'COMPLETE.json').read_text())
            shutil.copyfile(run/key/'history.json', out/(key+'_history.json'))
    write_json(out/'TRAINING.json', summaries)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), constrained_layout=True)
    for seed in range(3):
        h = json.loads((run/f'anchor_{seed}/history.json').read_text())
        h = [r for r in h if r['validation'] is not None]
        axes[0].plot([r['epoch'] for r in h], [r['validation'] for r in h], marker='.', label=f'Seed {seed}')
        h = json.loads((run/f'residual_{seed}/history.json').read_text())
        axes[1].plot([r['epoch'] for r in h], [r['validation_bce'] for r in h], label=f'Seed {seed}')
    axes[0].set(xlabel='Epoch', ylabel='Validation bounded improvement', title='Anchor validation (higher is better)')
    axes[1].set(xlabel='Epoch', ylabel='Validation BCE', title='Residual validation (lower is better)')
    for ax in axes:
        ax.legend(frameon=False); ax.grid(alpha=.2)
    for extension in ['png', 'pdf']:
        fig.savefig(out/f'training_curves.{extension}', dpi=180)
    plt.close(fig)
    pairs = [k for k in decision['comparisons'] if k != 'protection_risk_reduction']
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    for i, pair in enumerate(pairs):
        row = decision['comparisons'][pair]
        ax.plot(row['ci95'], [i, i], color='#245a81')
        ax.plot(row['mean'], i, 'o', color='#245a81')
    ax.axvline(0, color='black', lw=.8)
    ax.axvline(.005, color='#a65f00', lw=.8, linestyle='--', label='Predeclared practical threshold')
    ax.set(yticks=range(len(pairs)), yticklabels=pairs,
        xlabel='Paired bounded-improvement difference (95% clustered bootstrap interval)')
    ax.invert_yaxis(); ax.grid(axis='x', alpha=.2); ax.legend(frameon=False)
    for extension in ['png', 'pdf']:
        fig.savefig(out/f'mechanism_effects.{extension}', dpi=180)
    plt.close(fig)
    costs = dict(anchor_training_points=sum(summaries[f'anchor_{s}']['training_points'] for s in range(3)),
        anchor_validation_points=sum(summaries[f'anchor_{s}']['validation_points'] for s in range(3)),
        label_behavior_points=324*600, label_teacher_points=324*7600,
        development_main_points=5184*300, development_trajectories=5184,
        note='GPU preflight and contract tests reported separately; no duplicate validation/teacher rows counted as independent tasks')
    write_json(out/'COSTS.json', costs)
    artifact = ROOT/'artifacts/unified_revision_v1'
    artifact.mkdir(exist_ok=True)
    for stage in ['anchor', 'residual']:
        for seed in range(3):
            shutil.copyfile(run/f'{stage}_{seed}/selected.pt', artifact/f'{stage}_{seed}.pt')
    # Keep each archive comfortably below hosted git file-size limits.
    archives = []
    for seed in range(3):
        for stage in ['labels', 'development']:
            zpath = artifact/f'{stage}_{seed}.zip'
            with zipfile.ZipFile(zpath, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
                for p in sorted((run/f'{stage}_{seed}').glob('*')):
                    if p.suffix in ('.pt', '.json'):
                        z.write(p, str(p.relative_to(run)))
            if zpath.stat().st_size >= 95_000_000:
                raise ValueError(f'Archive needs splitting: {zpath}')
            archives.append(dict(file=zpath.name, bytes=zpath.stat().st_size, sha256=sha256(zpath)))
    shutil.copyfile(run/'task_parameters.pt', artifact/'task_parameters.pt')
    write_json(artifact/'manifest.json', dict(run=run.name, archive_files=archives,
        checkpoint_hashes={p.name: sha256(p) for p in artifact.glob('*.pt')}, source_identity=identity))
    write_json(out/'VERIFICATION.json', dict(status='passed', case_artifacts=len(checks),
        all_case_content_and_identity_hashes=True, six_frozen_models=True,
        original_checkpoints_unchanged=True, workers_joined=json.loads((run/'COMPLETE.json').read_text())['all_workers_joined']))
    lines = ['# 统一候选：受控重训与机制验收', '',
        '本报告依据预登记协议；原最终版未替换。新训练源没有使用测试ELA或测试成绩筛选。',
        '开发评测为已知12配方族、各3尺度的新实例，不称为完全未见函数族或最终独立测试。', '',
        f"当前决策：`{decision['decision']}`；后续候选：`{decision['surviving_candidate']}`。", '',
        '## 完整优化效应', '', '| 对照 | 配对平均差 | 95%区间 | 实用改善验收 |', '|---|---:|---|---|']
    for key, row in decision['comparisons'].items():
        if key == 'protection_risk_reduction':
            continue
        lines.append(f"| {key} | {row['mean']:.6f} | [{row['ci95'][0]:.6f}, {row['ci95'][1]:.6f}] | {row['practical_pass']} |")
    lines += ['', '效应使用预定有界归一化改善，正值表示前者较好；不是胜率或目标值百分比。',
        '区间按12配方族、实例和3个训练种子配对重采样；只有3个训练种子，训练方差估计仍有限。', '',
        '![配对效应与区间](mechanism_effects.png)', '',
        f"Residual独立收益验收：{decision['residual_independent_gain']}。保护机制收益/风险验收：{decision['protection_supported']}。", '',
        '## 训练记录', '', '| 模型 | 训练结束轮次 | 验证选中轮次 |', '|---|---:|---:|']
    for key, row in summaries.items():
        lines.append(f"| {key} | {row['epochs']} | {row['selected_epoch']} |")
    lines += ['', '完整数值、成本、实际参数与原始轨迹已归档。候选构造、训练源及部分训练口径发生变化，',
        '只能由共享同anchor的消融解释新候选内部机制；不能将新旧权重差异全归因于residual。', '',
        '![验证曲线](training_curves.png)', '',
        '## 按预登记规则行动', '']
    if decision['surviving_candidate'] is None:
        lines += ['完整方法及预定简化配置均未通过总体验收，本轮按计划停止，未打开最终测试。',
                  '不追加loss、epoch或阈值实验。原论文已有融合证据保留，新候选负面结果单独报告。']
    else:
        lines += ['候选通过开发验收；这不等于通过独立外部验证。先按冻结规则完成强基线与最终任务来源协议，',
                  '再执行最终扩展。尚未完成的强基线/中维/未用任务源不能写作已通过。']
    (out/'RESULTS.zh-CN.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'verified_cases': len(checks), 'decision': decision['decision'], 'artifacts': archives}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['provenance', 'finalize'])
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    (provenance if args.mode == 'provenance' else finalize)(args.run)
