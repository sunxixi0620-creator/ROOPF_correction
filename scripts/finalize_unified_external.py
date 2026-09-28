"""Audit, paired analysis, serial timing and archival; no external model selection."""
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
import argparse
import json
from pathlib import Path
import shutil
import sys
import zipfile
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'scripts')]
import unified_external as exp
from roopf.experiment_io import CaseStore, fingerprint, save_torch, sha256, write_json
from roopf.revision_tasks import ProceduralTask, population

OUT = ROOT/'docs/revision/unified_external'
ARCHIVE = ROOT/'artifacts/unified_external_v1'


def paired_effect(values):
    """Values ordered by function, training/run group, run within group."""
    values = np.asarray(values).reshape(12, 3, 10)
    rng = np.random.default_rng(2026092804)
    f = rng.integers(12, size=(5000, 12))
    g = rng.integers(3, size=(5000, 3))
    r = rng.integers(10, size=(5000, 12, 3, 10))
    draws = values[f[:, :, None, None], g[:, None, :, None], r].mean((1, 2, 3))
    low, high = np.quantile(draws, [.025, .975])
    by_function = values.mean((1, 2))
    return dict(mean=float(values.mean()), ci95=[float(low), float(high)],
        by_group=values.mean((0, 2)).tolist(), by_function=by_function.tolist(),
        function_wins=int((by_function > .01).sum()),
        function_ties=int((np.abs(by_function) <= .01).sum()),
        function_losses=int((by_function < -.01).sum()))


def serial_timing(run):
    identity = exp.verify(str(run))
    for mode in ('baselines', 'learned10', 'learned20'):
        assert json.loads((run/f'{mode}_COMPLETE.json').read_text())['all_workers_joined']
    assert json.loads((run/'baseline_overlap_COMPLETE.json').read_text())['all_workers_joined']
    store = CaseStore(run/'timing', identity)
    rows = []
    for dim, budget, fid, run_index in exp.TIMING_PANEL:
        for method in exp.METHODS:
            spec = exp.case_spec(dim, budget, fid, run_index, method)
            key = f'd{dim}_b{budget}_f{fid:02d}_r{run_index:02d}_{method}'
            with store.lock(key):
                value = store.load(key, spec)
                if value is None:
                    value = exp.execute(spec); store.save(key, spec, value)
            # Same seeds/settings: repeats are timing observations, never added to N.
            original = CaseStore(run/'cases', identity).load(key, spec)
            assert np.array_equal(value['points'], original['points'])
            assert np.array_equal(value['values'], original['values'])
            rows.append(value['row'])
            print('timing', key, value['row']['seconds'], flush=True)
    write_json(run/'TIMING_COMPLETE.json', dict(cases=len(rows), objective_evaluations=9000,
        same_trajectories=True, sequential_in_this_process=True, exclusive_host=False))
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT/'serial_timing.csv', index=False)


def audit(run):
    identity = exp.verify(str(run))
    assert json.loads((exp.NATIVE20/'COMPLETE.json').read_text())['all_workers_joined']
    for mode, count in [('baselines', 2160), ('learned10', 1080), ('learned20', 2160)]:
        done = json.loads((run/f'{mode}_COMPLETE.json').read_text())
        assert done['all_workers_joined'] and done['cases'] == count
    for path in run.glob('prefetch20_*.json'):
        done = json.loads(path.read_text())
        assert done['status'] == 'complete' and done['all_workers_joined']
    assert json.loads((run/'baseline_overlap_COMPLETE.json').read_text())['all_workers_joined']
    for name, digest in json.loads((ROOT/'docs/revision/unified_execution/STAGE1.json').read_text())['original_checkpoints'].items():
        assert sha256(ROOT/'checkpoints'/name) == digest
    frozen = json.loads((run/'NATIVE20_FROZEN.json').read_text())
    assert all(sha256(exp.checkpoint(20, int(s))) == h for s, h in frozen.items())
    store = CaseStore(run/'cases', identity)
    rows, seeds = [], []
    for dim, budget in exp.CONDITIONS:
        for fid in range(1, 13):
            for index in range(30):
                initial_points, initial_values = None, None
                for method in exp.METHODS:
                    spec = exp.case_spec(dim, budget, fid, index, method)
                    key = f'd{dim}_b{budget}_f{fid:02d}_r{index:02d}_{method}'
                    value = store.load(key, spec)
                    assert value is not None, key
                    assert value['points'].shape == (budget, dim)
                    assert value['values'].shape == value['trace'].shape == (budget,)
                    assert np.isfinite(value['values']).all()
                    assert np.array_equal(np.minimum.accumulate(value['values']), value['trace'])
                    assert value['row']['final'] == min(value['values'])
                    assert value['row']['actual_nfe'] == budget
                    if method in exp.METHODS[:3]:
                        if initial_points is None:
                            initial_points, initial_values = value['points'][:100], value['values'][:100]
                        else:
                            assert np.array_equal(initial_points, value['points'][:100])
                            assert np.array_equal(initial_values, value['values'][:100])
                        assert all(not x['residual_active'] for x in value['decisions'])
                    if method == 'cma_es':
                        seeds.append(spec['effective_optimizer_seed'])
                    rows.append(value['row'])
    assert len(rows) == 5400 and len(seeds) == len(set(seeds)) == 1080
    data = pd.DataFrame(rows)
    assert data.actual_nfe.sum() == 2160000
    return data, identity


def archive(run, identity, native):
    ARCHIVE.mkdir(exist_ok=True, parents=True)
    for seed in range(3):
        shutil.copyfile(exp.checkpoint(20, seed), ARCHIVE/f'anchor20_{seed}.pt')
    archives = []
    for dim, budget in exp.CONDITIONS:
        for method in exp.METHODS:
            target = ARCHIVE/f'd{dim}_b{budget}_{method}.zip'
            with zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as z:
                z.write(run/'cases/identity.json', 'cases/identity.json')
                for path in sorted((run/'cases').glob(f'd{dim}_b{budget}_f*_r*_{method}.*')):
                    if path.suffix in ('.pt', '.json'):
                        z.write(path, str(path.relative_to(run)))
            assert target.stat().st_size < 95_000_000
            archives.append(dict(file=target.name, bytes=target.stat().st_size, sha256=sha256(target)))
    with zipfile.ZipFile(ARCHIVE/'sources_and_timing.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        for path in sorted(set(exp.SOURCES) | set(identity['native20_training']['sources'])):
            z.write(ROOT/path, path)
        z.write(Path(__file__), 'scripts/finalize_unified_external.py')
        z.write(ROOT/'scripts/analyze_unified_decisions.py', 'scripts/analyze_unified_decisions.py')
        z.write(ROOT/'scripts/prefetch_external20.py', 'scripts/prefetch_external20.py')
        z.write(ROOT/'scripts/overlap_external_baselines.py', 'scripts/overlap_external_baselines.py')
        z.write(ROOT/'roopf/__init__.py', 'roopf/__init__.py')
        for path in sorted((run/'timing').glob('*')):
            if path.suffix in ('.pt', '.json'):
                z.write(path, str(path.relative_to(run)))
    # Explicit 20D task parameters; same parameter/population role across model seeds.
    definitions = []
    torch.set_num_threads(1)
    for split, epochs, count in [('anchor_training', range(1, max(x['epochs'] for x in native.values())+1), 16),
                                 ('anchor_validation', range(1), 2)]:
        for epoch in epochs:
            for fid in range(36):
                task = ProceduralTask(fid, epoch, split, dim=20)
                pop = population(fid, epoch, split, dim=20, count=count)
                definitions.append(dict(split=split, fid=fid, instance=epoch,
                    parameter_seed=task.parameter_seed, params=task.params,
                    population_sha256=fingerprint(pop), population_count=count))
    save_torch(ARCHIVE/'native20_task_parameters.pt', definitions)
    write_json(ARCHIVE/'manifest.json', dict(archives=archives, source_identity=identity,
        extra_files={p.name: sha256(p) for p in ARCHIVE.iterdir() if p.name not in
                     [x['file'] for x in archives]+['manifest.json']},
        task_parameter_sets=len(definitions), note='No objective evaluation for exporting task parameters'))


def finalize(run):
    OUT.mkdir(parents=True, exist_ok=True)
    data, identity = audit(run)
    data.to_csv(OUT/'results.csv', index=False)
    result = {}
    for dim, budget in exp.CONDITIONS:
        block = data[(data.dimension == dim) & (data.budget == budget)]
        index = pd.MultiIndex.from_product((range(1, 13), range(30)), names=('fid', 'run_index'))
        pivot = block.pivot(index=['fid', 'run_index'], columns='method', values='log_error').reindex(index)
        assert not pivot.isna().any().any()
        comparisons = {method: paired_effect(pivot[method]-pivot.no_residual)
                       for method in exp.METHODS[1:]}
        util = block.pivot(index=['fid', 'run_index'], columns='method', values='utility').reindex(index)
        selected_risk = util.no_residual-util.anchor_only < -.02
        score_risk = util.no_residual_score_only-util.anchor_only < -.02
        comparisons['protection_risk_reduction'] = paired_effect(score_risk.astype(float)-selected_risk.astype(float))
        comparisons['risk_rates'] = dict(selected=float(selected_risk.mean()), score_only=float(score_risk.mean()))
        result[f'd{dim}_b{budget}'] = comparisons
    write_json(OUT/'COMPARISONS.json', result)
    write_json(OUT/'ANALYSIS_IDENTITY.json', dict(
        scripts={name: sha256(ROOT/'scripts'/name) for name in
                 ('finalize_unified_external.py', 'analyze_unified_decisions.py')},
        results_csv_sha256=sha256(OUT/'results.csv'),
        comparisons_sha256=sha256(OUT/'COMPARISONS.json'),
        bootstrap_draws=5000, nominal_pointwise_coverage=.95,
        multiple_comparison_adjustment=False))
    data.groupby(['dimension', 'budget', 'fid', 'method']).agg(
        error_mean=('error', 'mean'), error_median=('error', 'median'),
        error_std=('error', 'std'), log_error_mean=('log_error', 'mean'),
        runs=('error', 'count'), accepted_mean=('accepted', 'mean'),
        early_accepted_mean=('early_accepted', 'mean')).to_csv(OUT/'function_summary.csv')
    native = {str(s): json.loads((exp.NATIVE20/f'anchor_{s}/COMPLETE.json').read_text()) for s in range(3)}
    write_json(OUT/'NATIVE20_TRAINING.json', native)
    shutil.copyfile(exp.NATIVE20/'COMPLETE.json', OUT/'NATIVE20_COMPLETE.json')
    for s in range(3):
        shutil.copyfile(exp.NATIVE20/f'anchor_{s}/history.json', OUT/f'anchor20_{s}_history.json')
    timing = json.loads((run/'TIMING_COMPLETE.json').read_text())
    costs = dict(external_trajectories=5400, external_main_points=2160000,
        external_teacher_points=0, timing_repeat_points=9000, timing_repeats_added_to_N=False,
        declared_optimum_implementation_check_points=24,
        synthetic_interface_preflight_points=2100,
        native20_anchor_training_points=sum(x['training_points'] for x in native.values()),
        native20_anchor_validation_points=sum(x['validation_points'] for x in native.values()),
        native20_anchor_process_seconds=sum(x['seconds'] for x in native.values()),
        max_native20_process_seconds=max(x['seconds'] for x in native.values()),
        prior_stage_costs=json.loads((ROOT/'docs/revision/unified_execution/COSTS.json').read_text()),
        shared_host_timing=True,
        audit_preflight_note='Other procedural gradient/parity checks are documented separately; no fully instrumented aggregate for every historical audit invocation')
    write_json(OUT/'COSTS.json', costs)
    from analyze_unified_decisions import external
    external(run, OUT)
    for name in ('identity.json', 'NATIVE20_FROZEN.json', 'baselines_COMPLETE.json',
                 'learned10_COMPLETE.json', 'learned20_COMPLETE.json', 'TIMING_COMPLETE.json',
                 'environment.json', 'resource_schedule.json', 'baseline_overlap_COMPLETE.json'):
        shutil.copyfile(run/name, OUT/name)
    archive(run, identity, native)
    write_json(OUT/'VERIFICATION.json', dict(passed=True, case_count=5400,
        all_case_hashes_and_identities_verified=True, exact_budget=True,
        all_learned_initializations_paired=True, teacher_queries=0,
        cma_32bit_seed_collisions=0, original_checkpoints_unchanged=True,
        all_workers_joined=True, timing=timing))
    plots(data, result, native)
    report(result, native)
    write_json(run/'COMPLETE.json', dict(status='complete', all_workers_joined=True,
        external_selection_performed=False, primary_method='no_residual',
        cases=5400, objective_evaluations=2160000))
    shutil.copyfile(run/'COMPLETE.json', OUT/'COMPLETE.json')
    print(json.dumps(result, indent=2), flush=True)


def plots(data, result, native):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5), constrained_layout=True)
    labels = ['Same anchor', 'No protection', 'CMA-ES', 'GP-EI']
    for ax, (key, comparisons) in zip(axes, result.items()):
        for j, method in enumerate(exp.METHODS[1:]):
            row = comparisons[method]
            ax.plot(row['ci95'], [j, j], color='#245a81'); ax.plot(row['mean'], j, 'o', color='#245a81')
        ax.axvline(0, color='black', lw=.8)
        ax.set(title=key, yticks=range(4), yticklabels=labels, xlabel='Paired log10-error advantage')
        ax.invert_yaxis(); ax.grid(axis='x', alpha=.2)
    for ext in ('png', 'pdf'):
        fig.savefig(OUT/f'external_effects.{ext}', dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 3.5), constrained_layout=True)
    for s in range(3):
        history = json.loads((OUT/f'anchor20_{s}_history.json').read_text())
        history = [x for x in history if x['validation'] is not None]
        ax.plot([x['epoch'] for x in history], [x['validation'] for x in history], marker='.', label=f'Seed {s}')
    ax.set(xlabel='Epoch', ylabel='Validation bounded improvement', title='Native 20D anchor validation')
    ax.legend(frameon=False); ax.grid(alpha=.2)
    for ext in ('png', 'pdf'):
        fig.savefig(OUT/f'native20_training.{ext}', dpi=180)
    plt.close(fig)


def report(result, native):
    lines = ['# 冻结后的 CEC2022 完整验证', '',
        '本轮只检验阶段三选定的 no_residual 候选；没有用外部成绩重新选择方法。原最终版保持冻结。',
        '采用版本与源数据哈希固定的 opfunu CEC2022 全12函数，含混合与组合函数；',
        '10维/300NFE、20维/300NFE、20维/600NFE，每方法每函数每条件30次，共5400条轨迹。',
        '20维anchor独立训练并由生成验证集选模，未使用测试目标或ELA筛选训练函数。',
        '每个模型先完成验证选模与哈希冻结，再评估该模型；不同维度/种子的训练和评测有时间重叠，训练与选择规则始终固定。',
        '作者此前对其他BBOB/CEC来源的开发接触仍须披露。本次不证明整个研究历史与所有数学原语隔离。', '',
        '## 主要比较', '',
        '优势=对照log10误差−候选log10误差，正值表示候选较好。区间为函数、训练/运行组、组内运行的5000次配对重采样。',
        '三种条件分别报告；不把同一函数在不同条件下视为独立的新函数族。逐对照名义95%区间未做多重比较校正。', '',
        '| 条件 | 对照 | 候选优势 | 95%区间 | 函数胜/平/负 |', '|---|---|---:|---|---|']
    for key, comp in result.items():
        for method in exp.METHODS[1:]:
            row = comp[method]
            lines.append(f"| {key} | {method} | {row['mean']:.4f} | [{row['ci95'][0]:.4f}, {row['ci95'][1]:.4f}] | {row['function_wins']}/{row['function_ties']}/{row['function_losses']} |")
    lines += ['', '函数胜/平/负按30运行的平均log误差差计算，绝对差≤0.01为平；不代替统计检验。',
        '训练种子只有3个，外部每组分配不同运行；训练随机性与运行组变异未完全分离。',
        '置信区间限于本套件的重采样，不是普遍可靠性或非退化保证。', '',
        '![外部配对效应](external_effects.png)', '', '## 保护风险', '',
        '风险为初始标准差归一化的有界改善相对同一anchor降低超过0.02的运行比例。',
        '两种融合配置共享相同初始100点；无保护对照也去掉residual，能够单独比较保护规则。', '',
        '| 条件 | 候选退化比例 | 无保护退化比例 | 风险降低及95%区间 |', '|---|---:|---:|---|']
    for key, comp in result.items():
        risk = comp['protection_risk_reduction']; rates = comp['risk_rates']
        lines.append(f"| {key} | {rates['selected']:.4f} | {rates['score_only']:.4f} | {risk['mean']:.4f} [{risk['ci95'][0]:.4f}, {risk['ci95'][1]:.4f}] |")
    lines += ['', '## 20维训练与计算成本', '', '| 种子 | 训练结束epoch | 验证选中epoch |', '|---|---:|---:|']
    for s, row in native.items():
        lines.append(f"| {s} | {row['epochs']} | {row['selected_epoch']} |")
    lines += ['', '![20维验证曲线](native20_training.png)', '',
        '线上计时见serial_timing.csv，固定F1/F8、10维300与20维600，每方法各4次按序计时。',
        '这些重复额外消耗9000次目标评估，不加入主性能样本数；宿主机器并非独占。',
        '串行计时包含模型构造/载入和在线搜索，不含Python启动、opfunu实例载入、缓存核验及结果写盘。',
        'objective_seconds只包目标evaluate计算；边界映射、张量转换等计入overhead_seconds。',
        '并行原始seconds包含资源竞争，少数任务还保留了暂停等待时间，不能用其做串行速度排名。',
        '主评测216万次目标调用、所有离线训练/标签及验证费用分列在COSTS.json。',
        '初始化计入预算；原生GP使用20点LHS、CMA使用10点种群，学习方法保留100点初始化。',
        '所有方法使用固定边界仿射接口；优化器不访问已知最优值。神经状态为float32，',
        '目标计算和最终统计为float64。此精度差异作为实现限制披露。', '',
        '## 论文边界', '',
        '本外部表检验简化候选，不为residual增益背书。阶段三残差独立收益未成立。',
        '存在anchor回退不等于保护已有效；必须结合上面的匹配对照与风险区间逐条件解释。',
        '20维是中等维度证据，不是高维、噪声或约束优化证据；不新增这些未经测试的主张。',
        '若强基线占优，保留结果并收紧论文定位，不以本表启动另一轮局部调参。']
    (OUT/'RESULTS.zh-CN.md').write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('timing', 'finalize'))
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    (serial_timing if args.mode == 'timing' else finalize)(args.run.resolve())
