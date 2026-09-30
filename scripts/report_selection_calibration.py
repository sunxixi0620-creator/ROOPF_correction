"""Verify, visualize and archive selection-only experiment; no objective calls."""
from pathlib import Path
import sys,json,zipfile
import numpy as np
import pandas as pd
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from roopf.experiment_io import sha256,write_json
run=ROOT/'results/selection_calibration_20260930'
out=ROOT/'docs/revision/selection_calibration'
r=json.loads((run/'COMPLETE.json').read_text());assert r['all_workers_joined']
identity=json.loads((run/'identity.json').read_text())
for name,digest in identity['sources'].items():assert sha256(ROOT/name)==digest
for seed,h in enumerate(identity['proposals']):assert sha256(ROOT/f'results/complementary_proposal_20260930/model_{seed}/selected.pt')==h
counts={};costs={}
for folder in ('data','search'):
 count=main=teacher=0
 for manifest in sorted((run/folder).glob('*.json')):
  if manifest.name=='identity.json':continue
  meta=json.loads(manifest.read_text())
  assert sha256(manifest.with_suffix('.pt'))==meta['data_sha256']
  v=torch.load(manifest.with_suffix('.pt'),weights_only=False)
  count+=1;main+=v['main_calls'];teacher+=v['teacher_calls']
 counts[folder]=count;costs[folder]=dict(main=main,teacher=teacher)
assert counts=={'data':324,'search':504}
assert costs['data']==dict(main=777600,teacher=393984)
assert costs['search']==dict(main=1209600,teacher=0)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,ax=plt.subplots(figsize=(8,3.4),layout='constrained')
for i,m in enumerate(('Base','O')):
 e=r['effects'][m]
 ax.plot([e['lower'],e['upper']],[i,i],lw=2,color='#28649a');ax.plot(e['mean'],i,'o',color='#28649a')
ax.axvline(0,color='gray',linestyle='--');ax.axvline(.005,color='#b26700',linestyle=':',label='Practical threshold .005')
ax.set_yticks([0,1],['Cal - current selection','Cal - online O'])
ax.set_xlabel('Bounded final utility difference (97.5% CI)');ax.legend();ax.grid(axis='x',alpha=.2)
fig.savefig(out/'effects.png',dpi=180);fig.savefig(out/'effects.pdf');plt.close(fig)
training=sorted(r['training'],key=lambda x:x['seed'])
text=['# 固定提案的选择校准实验','',
 '20D/600，三个提案模型全程冻结，仅训练769参数的最终分数校准器。原解锁权重和旧residual保持不变。',
 '使用新生成实例，但与此前研究共享函数族，属于开发证据；不能称为外部未见函数族验证。','',
 '|种子|实际训练轮数|验证选中轮数|初始验证收益|选中验证收益|','|---|---:|---:|---:|---:|']
for v in training:text.append(f"|{v['seed']}|{v['epoch']}|{v['selected_epoch']}|{v['initial_validation']:.6f}|{v['selected_validation']:.6f}|")
text+=['','|完整搜索比较|平均收益差|97.5%区间|预设验收|','|---|---:|---|---|']
for m,e in r['effects'].items():text.append(f"|Cal−{m}|{e['mean']:.6f}|[{e['lower']:.6f}, {e['upper']:.6f}]|{'通过' if e['passed'] else '未通过'}|")
text+=['','验收要求两个比较均满足平均提升≥0.005、调整后区间下界>0、三个种子方向均为正。',
 f"本轮结论：{'通过内部验收，仍需独立确认' if r['success'] else '未通过预定验收，按协议停止，不追加特征/阈值/epoch变体'}。",'',
 '完成2,016条正式搜索轨迹、1,209,600次付费评估；行为采集777,600次，离线教师393,984次，契约检查4,278次。正式搜索没有教师调用。',
 '三个比较组为Base（当前提案+当前排序）、Cal（同一提案+校准排序）、O（纯在线）；O每个实例只运行一次，作为三个种子的共享配对对照。',
 '运行耗时保存在runtime_selection.csv，是并发工作进程墙钟时间，不充当严格串行延迟基准。校准特征增加一次代理预测/拟合，计算开销不计入函数评估数但不能忽略。',
 '协议中的“online proxy retraining”指不另行改变代理训练方法；运行时正常按已观测档案重新拟合岭回归。',
 '复现：在仓库根目录运行 `.venv/bin/python scripts/selection_calibration_study.py`。源文件冻结后不得修改再复用相同目录；完整数据和模型见归档。','',
 '[冻结协议](../../experiments/SELECTION_CALIBRATION_PROTOCOL.md) · [结果图](effects.png) · [完整归档](../../../artifacts/selection_calibration_v1/README.md)']
(out/'CONCLUSIONS.zh-CN.md').write_text('\n'.join(text)+'\n')
dest=ROOT/'artifacts/selection_calibration_v1';dest.mkdir(parents=True,exist_ok=True)
groups={}
for p in sorted(run.rglob('*')):
 if not p.is_file() or p.suffix=='.lock':continue
 rel=p.relative_to(run)
 if rel.parts[0] in ('data','search') and p.stem!='identity':group=rel.parts[0]+'_'+p.name.split('_')[0]
 else:group='metadata_models_sources'
 groups.setdefault(group,[]).append(p)
manifest={}
for name,paths in groups.items():
 target=dest/(name+'.zip')
 with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
  for p in paths:z.write(p,str(p.relative_to(run)))
  if name=='metadata_models_sources':z.write(__file__,'reporting_source/report_selection_calibration.py')
 with zipfile.ZipFile(target) as z:assert z.testzip() is None
 assert target.stat().st_size<95000000
 manifest[target.name]=dict(sha256=sha256(target),bytes=target.stat().st_size)
write_json(dest/'MANIFEST.json',manifest)
write_json(out/'PACKAGE_VERIFICATION.json',dict(cases=counts,costs=costs,source_hashes_passed=True,
 proposal_hashes_passed=True,case_hashes_passed=True,zip_crc_passed=True,archives=len(manifest)))
(dest/'README.md').write_text('# Selection calibration v1\n\nExtract allZIPs into results/selection_calibration_20260930/.\nIncludes source/protocol snapshot, full labeled data, calibrators, training histories and all paid search trajectories.\nFrozen proposal dependency: artifacts/complementary_proposal_v1.\nSee docs/revision/selection_calibration/CONCLUSIONS.zh-CN.md. This is shared-family development evidence.\n')
print(json.dumps(manifest,indent=2))
