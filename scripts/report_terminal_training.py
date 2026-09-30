"""Completed terminal ES audit/report/archive. Never resumes training."""
from pathlib import Path
import sys,json,zipfile
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from roopf.experiment_io import sha256,write_json,fingerprint
from scripts.terminal_training import verify,proposal,interval
run=ROOT/'results/terminal_training_20260930';out=ROOT/'docs/revision/terminal_training'
identity=verify();result=json.loads((run/'COMPLETE.json').read_text());assert result['all_workers_joined']
frozen=json.loads((run/'MODELS_FROZEN.json').read_text())
initial=torch.load(run/'initial_validation.pt',weights_only=False)
selected=[]
for s,u in enumerate(result['selected_updates']):
 assert sha256(run/f'selected_{s}.pt')==frozen['hashes'][str(s)]
 data=initial if u==0 else torch.load(run/f'validation_u{u}.pt',weights_only=False)
 selected.append(data[s])
 if u==0:
  original=proposal(s).state_dict();saved=torch.load(run/f'selected_{s}.pt',weights_only=False)
  assert all(torch.equal(original[k],saved[k]) for k in original)
selected=np.array(selected)
for key,array in [('vs_O',selected),('vs_initial',selected-initial)]:
 e=interval(array)
 assert all(abs(e[k]-result['validation'][key][k])<1e-12 for k in ('mean','lower','upper'))
counts={};costs={};centers={}
for p in sorted((run/'cases').glob('*.json')):
 if p.name=='identity.json':continue
 meta=json.loads(p.read_text());assert sha256(p.with_suffix('.pt'))==meta['data_sha256']
 case=meta['case'];v=torch.load(p.with_suffix('.pt'),weights_only=False)
 assert fingerprint({'run':identity,'case':case})==meta['identity_sha256']
 assert v['teacher_calls']==0 and torch.isfinite(v['utility']).all()
 role=case['role'];counts[role]=counts.get(role,0)+1;costs[role]=costs.get(role,0)+v['main_calls']
 if case['center']:centers[case['center']]=case['center_hash']
for name,h in centers.items():assert sha256(run/name)==h
assert counts==dict(timing=24,train=8820,validation=792)
assert costs==json.loads((out/'COSTS.json').read_text())
assert not result['validation_gate'] and result['confirmation'] is None
history=json.loads((run/'history.json').read_text())
assert len(history)==30 and all(all(s['norm']>0 for s in h['signals']) for h in history)
curves=[initial.mean((1,2))]+[np.array(h['validation']) for h in history if 'validation' in h]
updates=[0]+[h['update'] for h in history if 'validation' in h]
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(1,2,figsize=(11,3.6),layout='constrained')
for s in range(3):
 axes[0].plot(updates,np.array(curves)[:,s],marker='o',label=f'Seed {s}')
 u=result['selected_updates'][s];axes[0].scatter([u],[selected[s].mean()],s=90,facecolors='none',edgecolors='black',zorder=3)
axes[0].axhline(0,color='gray',linestyle='--');axes[0].set_xlabel('ES updates')
axes[0].set_ylabel('Validation terminal reward vs O');axes[0].legend();axes[0].grid(alpha=.2)
for i,key in enumerate(('vs_O','vs_initial')):
 e=result['validation'][key];axes[1].plot([e['lower'],e['upper']],[i,i],lw=2,color='#28649a')
 axes[1].plot(e['mean'],i,'o',color='#28649a')
axes[1].set_yticks([0,1],['Selected - O','Selected - initial']);axes[1].axvline(0,color='gray',linestyle='--')
axes[1].set_xlabel('Selection-set difference (descriptive 95% CI)');axes[1].grid(axis='x',alpha=.2)
fig.savefig(out/'training.png',dpi=180);fig.savefig(out/'training.pdf');plt.close(fig)
text=['# 终局收益闭环重训结果','',
 '**本轮完成，未通过预登记可行性门槛，确认集未打开。** 三个种子均完成30次ES更新，耗时2241.99秒（约37分22秒）。原解锁权重不变，不替换当前方法。','',
 '更新全部58,504个提案参数，固定在线候选池、代理拟合、排序和评估名额。每次更新8对参数扰动，在6个训练任务上跑完整600次预算，Adam学习率0.01，归一化扰动幅度0.02；梯度估计不通过离散排序反向传播。','',
 '|种子|实际更新数|验证选中更新|初始相对O收益|选中相对O收益|相对重训前提升|',
 '|---|---:|---:|---:|---:|---:|']
for s,u in enumerate(result['selected_updates']):text.append(f'|{s}|30|{u}|{initial[s].mean():.6f}|{selected[s].mean():.6f}|{(selected[s]-initial[s]).mean():.6f}|')
text+=['','|比较|平均收益差|描述性95%区间|预定均值门槛|','|---|---:|---|---:|']
for k,t in [('vs_O',.005),('vs_initial',.001)]:
 e=result['validation'][k];text.append(f"|{k}|{e['mean']:.6f}|[{e['lower']:.6f}, {e['upper']:.6f}]|{t}|")
text+=['',
 '两个平均提升均未达到预设门槛；种子0保留第0次检查点，相对重训前提升为0，也不满足所有种子正向的要求。',
 '**这些是选模验证集上的统计，不是独立确认结果。** 同一验证集选择多个检查点后再汇报其区间会有选择乐观性，区间在零以上不能据此宣称泛化收益显著。所有任务仍共享既有生成函数族。',
 '每轮均检测到有限且非零的参数扰动信号，说明训练管线确实进行了更新，但这不等于估计方差足够小，也不证明优化收敛。30次更新不是30个完整数据epoch，更不等于原anchor的1000轮训练。',
 '种子0未改善；另两个种子在第20和第10次更新选中。此结果仅说明本次固定预算、8方向的ES方案没有达到验收，不能推导所有终局训练或融合方法都不可能成功。',
 '按协议停止本轮，不重新开启已完成训练、不追加迭代/扰动幅度/宽度扫描、不启动失败门槛后的确认实验。原论文双向融合及所有组件必要性仍未建立。',
 '训练调用10,584,000次，验证1,900,800次，计时/复现检查28,800次，总计12,513,600次；没有额外教师查询或目标导数。对应9,636个已核验批次、20,856条完整600预算轨迹。',
 '下一步应整理最终方法取舍与论文证据边界。若选择新的方法研究，应另立具体假设与资源上限；不把本轮失败解释成自动追加大训练的授权。','',
 '[协议](../../experiments/TERMINAL_TRAINING_PROTOCOL.md) · [训练曲线](training.png) · [调用成本](COSTS.json) · [归档](../../../artifacts/terminal_training_v1/README.md)']
(out/'CONCLUSIONS.zh-CN.md').write_text('\n'.join(text)+'\n')
dest=ROOT/'artifacts/terminal_training_v1';dest.mkdir(parents=True,exist_ok=True)
groups={}
for p in sorted(run.rglob('*')):
 if not p.is_file() or p.suffix=='.lock':continue
 rel=p.relative_to(run)
 if rel.parts[0]=='cases' and p.stem!='identity':
  meta=json.loads(p.with_suffix('.json').read_text());group='cases_'+meta['case']['role']
 else:group='models_sources_state'
 groups.setdefault(group,[]).append(p)
manifest={}
for group,paths in groups.items():
 target=dest/(group+'.zip')
 with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
  for p in paths:z.write(p,str(p.relative_to(run)))
  if group=='models_sources_state':z.write(__file__,'reporting_source/report_terminal_training.py')
 with zipfile.ZipFile(target) as z:assert z.testzip() is None
 assert target.stat().st_size<95000000
 manifest[target.name]=dict(sha256=sha256(target),bytes=target.stat().st_size)
write_json(dest/'MANIFEST.json',manifest)
write_json(out/'PACKAGE_VERIFICATION.json',dict(case_counts=counts,costs=costs,all_hashes_passed=True,
 selected_weights_passed=True,statistics_reproduced=True,zero_update_fallback_exact=True,zip_crc_passed=True))
(dest/'README.md').write_text('# Terminal-reward training feasibility v1\n\nExtract allZIPs into results/terminal_training_20260930/.\nIncludes sources, protocol, all ES centers, selected models, optimizer/resume state and verified evaluation cases.\nTraining/validation cases retain trails, initial values and point hashes; full evaluated points can be regenerated from frozen sources/seeds/centers.\nParent dependencies: artifacts/complementary_proposal_v1 and complementarity_v1.\nValidation gate failed; no confirmation set opened. Do NOT resume completed training.\nSee docs/revision/terminal_training/CONCLUSIONS.zh-CN.md.\n')
print(json.dumps(manifest,indent=2))
