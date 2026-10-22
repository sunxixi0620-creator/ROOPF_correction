from pathlib import Path
import csv
import hashlib
import json
import gzip
import lzma
import shutil
import subprocess
import tarfile

import numpy as np

ROOT = Path('/home/skq/repos/roopf_agent/ROOPF')
EXP = ROOT / 'experiments/r02_fusion_agent'
OUT = EXP / 'results'


def save_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')


def main():
    verified = json.loads((EXP / 'VERIFICATION.json').read_text())
    assert verified['status'] == 'passed' and verified['paired_functions'] == 35
    assert (OUT / 'COMPLETE.json').exists()
    rows = list(csv.DictReader((OUT / 'FUNCTION_COMPARISON.csv').open()))
    local_rows = [row for row in rows if row['mode'] == 'retrieval']
    live_rows = [row for row in rows if row['mode'] == 'llm_live']
    assert len(local_rows) == len(live_rows) == 35
    groups = {}
    for group in ['BBOB', 'CEC-style', 'UAV']:
        groups[group] = {}
        for mode, selected_rows in [('retrieval', local_rows), ('llm_live', live_rows)]:
            selected = [row for row in selected_rows if row['group'] == group and not (group == 'BBOB' and row['fid'] == '20')]
            groups[group][mode] = dict(functions=len(selected), wins=sum(float(row['mean_difference']) < 0 for row in selected),
                ties=sum(float(row['mean_difference']) == 0 for row in selected), losses=sum(float(row['mean_difference']) > 0 for row in selected),
                changed_decisions=sum(int(row['changed_decisions']) for row in selected), decisions=sum(int(row['decisions']) for row in selected))
    native_total = 0
    for path in OUT.rglob('*.json'):
        data = json.loads(path.read_text())
        if isinstance(data, dict) and 'mode' in data and 'native_fe' in data:
            native_total += data['native_fe']
    assert native_total <= 210000
    formal = [json.loads(path.read_text()) for path in (OUT / 'api_journal/json_contract_v4').glob('*.json')]
    all_api = [json.loads(path.read_text()) for path in (OUT / 'api_journal').rglob('*.json')]
    formal_valid = sum(row['status'] == 'ok' for row in formal)
    api_rate = formal_valid / max(1, len(formal))
    main_group = groups['BBOB']['llm_live']
    local_by_task = {(row['group'], row['fid']): row for row in local_rows}
    llm_vs_local = dict(wins=0, ties=0, losses=0)
    for row in live_rows:
        if row['group'] == 'BBOB' and row['fid'] == '20':
            continue
        delta = float(row['mean']) - float(local_by_task[(row['group'], row['fid'])]['mean'])
        llm_vs_local['wins' if delta < 0 else 'losses' if delta > 0 else 'ties'] += 1
    admitted = main_group['wins'] > main_group['losses'] and api_rate >= .9 and llm_vs_local['wins'] > llm_vs_local['losses']
    report = dict(status='completed_pilot', performance_improvement_admitted=admitted, n_per_function=10,
        paired_original_n=10, preserved_original_n=100, groups=groups, llm_vs_retrieval=llm_vs_local,
        formal_api=dict(requests=len(formal), valid=formal_valid, failed=len(formal)-formal_valid, valid_rate=api_rate,
                       total_tokens=sum(row.get('response', {}).get('usage', {}).get('total_tokens', 0) for row in formal),
                       elapsed_seconds=sum(row.get('elapsed_seconds', 0) for row in formal)),
        all_api=dict(requests=len(all_api), valid=sum(row['status'] == 'ok' for row in all_api),
                     failed=sum(row['status'] != 'ok' for row in all_api), recorded_total_tokens_lower_bound=sum(row.get('response', {}).get('usage', {}).get('total_tokens', 0) for row in all_api)),
        native_test_fe_including_failed_attempts=native_total, offline_training_function_fe=77220,
        currency_cost='Provider does not supply verified currency cost; token and timing receipts preserved',
        baseline_commit='52ec388b7812214f98fbfd84624a1dd65a52a0d8',
        limitations=['10-trajectory pilot, not 100-trajectory independent confirmation', 'BBOB-f20 excluded from primary comparisons',
                     'CEC structured lock retained', 'UAV recovered protocol', 'Only gate-threshold alternatives survived offline validation; weight-changing alternatives rejected'])
    save_json(EXP / 'FINAL_RESULT.json', report)
    lines = ['# R02 离线—在线融合 Agent 实验报告', '',
        '原始对照来自52ec388，35函数×6组×100轨迹，400文件已独立复制并逐字核验；本轮只写实验副本，不写原仓库或原支付记录。原仓库期间存在其他提交更新，当前HEAD单独记录在SOURCE_REFERENCE_CHECK.json，不作为更换对照的依据。', '',
        '本轮已完成35函数、每函数10条配对轨迹、每条300 FE的pilot。原版/off逐调用只读回放，所有目标序列与原始前10轨迹逐字相同。另运行同一经验库的确定性检索融合，以及实际deepseek-v4-flash融合。', '',
        '离线仅用训练函数重建61560条完整池记录及1620个融合状态；29函数建立经验库，7函数筛选配置。修改点位于原Protected selector之前，连接离线经验、Anchor、Proxy、Residual及当前任务已观察反馈；六算子、冻结权重、Anchor保护槽和原干预窗口保留。', '',
        '## 与配对原版的函数均值胜/平/负', '', '|范围|检索式融合|LLM融合|', '|---|---|---|']
    for group, modes in groups.items():
        rendered = ['/'.join(str(modes[mode][key]) for key in ['wins', 'ties', 'losses']) for mode in ['retrieval', 'llm_live']]
        lines.append('|' + group + ('（排除f20）' if group == 'BBOB' else '') + '|' + '|'.join(rendered) + '|')
    lines += ['', 'LLM相对检索式融合的函数均值胜/平/负：' + '/'.join(str(llm_vs_local[key]) for key in ['wins', 'ties', 'losses']) + '。', '',
        f"正式API {len(formal)}次，合法{formal_valid}次，失败{len(formal)-formal_valid}次，有效率{api_rate:.2%}；记录token {report['formal_api']['total_tokens']}。实验及接口日志共{len(all_api)}次请求，另有Luna两个失败健康测试和DeepSeek一个成功健康测试；早期失败全部保留。", '',
        f'新增真实测试FE（含失败组）{native_total}；训练函数FE 77220另计。命中原支付的精确响应不重复计为原生费用，但计入每轨迹300逻辑FE。中转站未给出可核货币费用，不能用返回credit=0宣称免费。', '',
        '## 逐函数完整数值', '', '以下均为目标终值均值±总体标准差；越小越好。原100只作描述，配对比较使用相同前10条。', '',
        '|函数|原100轨迹|原配对10轨迹|检索融合10轨迹|LLM融合10轨迹|', '|---|---|---|---|---|']
    for row in live_rows:
        local = local_by_task[(row['group'], row['fid'])]
        numbers = [(row['historical_100_mean'], row['historical_100_std']), (row['paired_baseline_mean'], row['paired_baseline_std']),
                   (local['mean'], local['std']), (row['mean'], row['std'])]
        lines.append('|' + row['group'] + '_f' + row['fid'] + '|' + '|'.join(f'{float(mean):.9g} ± {float(std):.9g}' for mean, std in numbers) + '|')
    lines += ['', '## 科学结论及边界', '',
        '本轮' + ('达到预设初步准入条件，但仍需要独立确认。' if admitted else '没有达到预设性能改进准入条件，不能宣称加入API或Agent已经改善ROOPF。'), '',
        '离线验证淘汰proxy/residual两项权重变更配置；最终允许release、cautious、balanced，主要区别为接受门槛。本轮不能证明自适应证据权重带来收益。一步离线优势也不等同于终局收益。', '',
        '在线误差来自已选择点上的条件反馈，不能解释为对所有未选候选的无偏可靠性估计。模型名称为中转站回报身份，不等于已经独立核实上游模型身份；thinking关闭请求是否被遵守以实际回执为准。', '',
        'CEC原structured_system_lock未解除，因此该组可能完全不调用Agent。UAV沿用原复现恢复协议；BBOB-f20随机耦合单列。保留原有缺失TF1–TF3训练权重的边界，本轮不补称训练集消融完成。', '',
        '早期WSL直连失败后使用Windows系统网络；短请求成功但长请求受thinking截断和数组长度问题影响。所有接口尝试均归档，正式试验统一采用显式trajectory_id映射、JSON输出合同和thinking关闭参数；个别无效输出按事前协议回退，不重复发送。', '',
        '旧接口阶段decision日志的反馈字段可能引用任务结束后的内存视图，不能用于逐步因果分析；实际API请求已按调用时序独立落盘。正式ID合同阶段已在记录时复制反馈快照。', '',
        '## 保存与复用', '',
        'FUNCTION_COMPARISON.csv保存逐函数统计，PAIRED_TRAJECTORIES.csv保存逐轨迹比较，npz保存每次目标值。API_AND_DECISION_JOURNAL.tar.xz无损保存完整请求、响应、失败和决策日志；JOURNAL_MANIFEST.json提供每个原文件的SHA256。密钥保存在项目外，不进入提交。']
    (EXP / 'REPORT.zh-CN.md').write_text('\n'.join(lines) + '\n')

    journal_files = sorted(set((OUT / 'api_journal').rglob('*')) | set(OUT.rglob('*_decisions.json')))
    journal_files = [path for path in journal_files if path.is_file()]
    entries = []
    with tarfile.open(EXP / 'API_AND_DECISION_JOURNAL.tar.xz', 'w:xz', preset=6) as archive:
        for path in journal_files:
            archive.add(path, arcname=str(path.relative_to(EXP)), recursive=False)
            entries.append(dict(path=str(path.relative_to(EXP)), bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    with tarfile.open(EXP / 'API_AND_DECISION_JOURNAL.tar.xz', 'r:xz') as archive:
        for entry in entries:
            assert hashlib.sha256(archive.extractfile(entry['path']).read()).hexdigest() == entry['sha256']
    save_json(EXP / 'JOURNAL_MANIFEST.json', dict(status='verified_lossless', files=entries))
    pool_raw = gzip.decompress((EXP / 'OFFLINE_FULL_POOL.jsonl.gz').read_bytes())
    pool_archive = EXP / 'OFFLINE_FULL_POOL.jsonl.xz'
    pool_archive.write_bytes(lzma.compress(pool_raw, preset=6))
    assert lzma.decompress(pool_archive.read_bytes()) == pool_raw
    save_json(EXP / 'POOL_COMPRESSION.json', dict(status='verified_lossless', raw_sha256=hashlib.sha256(pool_raw).hexdigest(), gzip_bytes=(EXP / 'OFFLINE_FULL_POOL.jsonl.gz').stat().st_size, xz_bytes=pool_archive.stat().st_size))
    key = Path('/home/skq/.config/roopf_agent/worldcodes.env').read_text().strip().split('=', 1)[1].encode()
    assert not any(key in path.read_bytes() for path in ROOT.rglob('*') if path.is_file() and '.git' not in path.parts)
    source = Path('/home/skq/repos/ROOPF_improve/ROOPF')
    current_head = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    assert subprocess.run(['git', '-C', str(source), 'merge-base', '--is-ancestor', report['baseline_commit'], current_head]).returncode == 0
    assert not subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain'], text=True).strip()
    save_json(EXP / 'SOURCE_REFERENCE_CHECK.json', dict(baseline_commit=report['baseline_commit'], current_source_head=current_head,
        baseline_still_in_history=True, source_worktree_clean=True, baseline_snapshot_hashes_verified=True,
        source_has_other_updates=current_head != report['baseline_commit']))
    desktop = Path('/mnt/c/Users/skq/Desktop/ROOPF_Agent_R02')
    for filename in ['REPORT.zh-CN.md', 'FINAL_RESULT.json', 'VERIFICATION.json']:
        shutil.copy2(EXP / filename, desktop / filename)
    shutil.copy2(OUT / 'FUNCTION_COMPARISON.csv', desktop / 'FUNCTION_COMPARISON.csv')
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
