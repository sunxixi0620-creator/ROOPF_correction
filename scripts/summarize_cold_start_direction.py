"""Present independent cold-start studies without pooling or changing endpoints."""
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'docs/revision/cold_start_direction';OUT.mkdir(parents=True,exist_ok=True)
rep=json.loads((ROOT/'docs/revision/cold_start_replication/RESULTS.json').read_text());cov=json.loads((ROOT/'docs/revision/cold_start_coverage/RESULTS.json').read_text())
lines=['# 冷启动方向：扩大复核与固定覆盖保护','',
'本轮按两个先后冻结的协议推进：先保持W完全不变扩大实例复核，确认速度收益和覆盖率差异；随后仅检验一个H候选——早期30次评估中固定10次Sobol、20次融合选点，40次时撤出先验。没有神经重训、窗口/比例扫描或旧residual实验。',
'两轮均使用新任务角色，但函数家族已有开发接触，所以是同分布新实例证据；不是未见函数族确认，不能追溯归给原解锁ROOPF。', '',
'|研究|方法|300次目标达到率|受限平均达到时间|600次有界改善|','|---|---|---:|---:|---:|']
for name,r in [('扩大实例复核',rep),('固定覆盖保护',cov)]:
 for m,v in r['means'].items():lines.append(f"|{name}|{m}|{v['success'][1]:.3%}|{v['time'][1]:.3f}|{v['terminal']:.6f}|")
lines+=['','目标为初始10点最佳值再改善0.5个初始样本标准差；未在300次内达标记受限时间301，全部保留。各研究的任务实例不同，不能直接拿跨研究均值比较优劣。', '', '|研究/候选相对|节省评估|97.5%区间|达到率差|终局质量差|主验收|','|---|---:|---|---:|---:|---|']
for name,r,candidate in [('复核',rep,'W'),('覆盖保护',cov,'H')]:
 for c in r['contrasts']:
  a=c['time_saved'];lines.append(f"|{name}/{candidate}−{c['control']}|{a['mean']:+.3f}|[{a['lower']:+.3f},{a['upper']:+.3f}]|{c['attainment_delta']['mean']:+.3%}|{c['terminal_delta']['mean']:+.6f}|{'描述性' if c.get('primary') is False else ('通过' if c['passed'] else '未通过')}|")
lines+=['',f"扩大复核联合验收：{'通过' if rep['joint_pass'] else '未通过'}；覆盖保护联合验收：{'通过' if cov['joint_pass'] else '未通过'}。", '',
'验收要求相对O/S分别节省至少5次、配方聚类97.5%区间下界为正、三个模型种子均正，终局差下界高于−0.005，并且平均目标达到率不下降。原来未通过的结果不会被新候选覆盖。',
'新增评估：复核864000；覆盖候选主搜索1036800，契约45；本轮共1900845次。累计此前冷启动搜索/诊断481428次，合计2382273次。只读审查与重放不增加调用。新增神经训练0轮。', '',
'## 论文定位边界', '',
'可以围绕离线条件先验的早期样本效率讨论：离线先验在少观测阶段改善搜索速度，后续保留付费数据并撤掉先验。预测均值与在线GP纠偏是新方法身份，不能把它写成原冻结anchor或原离线residual已被验证。',
'“达到同等质量更快”与“预算内达到率不降低”必须同时报告。终局非劣只是在预设0.005有界效用容忍度下通过，不等于严格相同或终局更优。未达到任务的301是统计截断值，不是成功时间。',
'需要补齐的证据仍包括正确先验对打乱先验/匹配无学习先验的完整闭环比较、在线适应的独立贡献，以及未参与开发的任务分布验证。门槛未通过时不自动启动这些确认，不宣称全部组件必要。']
if not cov['joint_pass']:lines+=['','## 本轮停止决策','','按冻结协议停止H，不扫描Sobol比例、退出窗口或追加epoch。保留W的重复早期加速证据以及其覆盖率限制；不把新增保护视为有效改进。若继续研究，需要另立清晰的新假设与未使用的数据，不能以“必须完美”为由重复调整同一任务上的结果。']
(OUT/'CONCLUSIONS.zh-CN.md').write_text('\n'.join(lines)+'\n')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axs=plt.subplots(2,2,figsize=(11,7))
for row,(name,path) in enumerate([('Replication','cold_start_replication'),('Fixed coverage','cold_start_coverage')]):
 a=np.load(ROOT/f'docs/revision/{path}/arrays.npz')
 for m in (('W','O','S') if row==0 else ('H','W','O','S')):
  curve=a[m+'_u'].mean((0,1,2));tt=a[m+'_time'][...,1];axs[row,0].plot(np.arange(10,601),curve,label=m);axs[row,1].plot(np.arange(10,301),[(tt<=n).mean() for n in range(10,301)],label=m)
 axs[row,0].set(title=name,ylabel='Bounded improvement',xlabel='Paid evaluations');axs[row,1].set(title=name,ylabel='Target attainment',xlabel='Paid evaluations')
 for ax in axs[row]:ax.axvline(40,color='gray',linestyle=':');ax.legend()
fig.tight_layout()
for ext in ('png','pdf'):fig.savefig(OUT/f'curves.{ext}',dpi=160)
