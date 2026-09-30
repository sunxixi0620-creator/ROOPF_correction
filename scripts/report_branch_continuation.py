"""Archive and report branch diagnostic; no further objective calls."""
from pathlib import Path
import sys,json,zipfile
import pandas as pd
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from roopf.experiment_io import sha256,write_json
run=ROOT/'results/branch_continuation_20260930';out=ROOT/'docs/revision/branch_continuation'
r=json.loads((run/'COMPLETE.json').read_text());assert r['all_workers_joined']
identity=json.loads((run/'identity.json').read_text())
for p,h in identity['sources'].items():assert sha256(ROOT/p)==h
for s,h in enumerate(identity['proposals']):assert sha256(ROOT/f'results/complementary_proposal_20260930/model_{s}/selected.pt')==h
main=teachers=cases=0
for p in sorted((run/'cases').glob('*.json')):
 if p.name=='identity.json':continue
 assert sha256(p.with_suffix('.pt'))==json.loads(p.read_text())['data_sha256']
 v=torch.load(p.with_suffix('.pt'),weights_only=False)
 main+=v['main_calls'];teachers+=v['teacher_calls'];cases+=1
 assert len(v['rows'])==3
assert (cases,main,teachers)==(432,1036800,5184)
df=pd.read_csv(out/'pairs.csv');df['recipe']=df.fid//3
conditional={}
for label,g in [('all',df)]+[(str(n),g) for n,g in df.groupby('nfe')]:
 q=g[g.opportunity]
 rng=np.random.default_rng(20261001)
 indices=rng.integers(0,12,(5000,12))
 count=q.groupby('recipe').size().reindex(range(12),fill_value=0).to_numpy()
 sums=q.groupby('recipe')[['immediate','final']].sum().reindex(range(12),fill_value=0)
 den=count[indices].sum(1);valid=den>0
 result={}
 for metric in ('immediate','final'):
  means=sums[metric].to_numpy()[indices].sum(1)[valid]/den[valid]
  result[metric]=dict(mean=float(q[metric].mean()) if len(q) else None,
   lower=float(np.quantile(means,.025)) if len(means) else None,
   upper=float(np.quantile(means,.975)) if len(means) else None,
   nonempty_bootstrap_replicates=int(valid.sum()))
 conditional[label]=result
write_json(out/'CONDITIONAL_INTERVALS.json',conditional)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(1,2,figsize=(11,3.6),layout='constrained')
for n,g in df.groupby('nfe'):
 axes[0].plot(range(4),g[['immediate','after20','after60','final']].mean(),marker='o',label=f'Intervene at NFE {n}')
axes[0].set_xticks(range(4),['Immediate','+20 evals','+60 evals','Final'])
axes[0].set_ylabel('Mean bounded utility advantage (all states)');axes[0].legend();axes[0].grid(alpha=.2)
for i,(n,g) in enumerate(df.groupby('nfe')):
 q=g[g.opportunity]
 for k,(metric,color) in enumerate([('final_win','#34865a'),('final_tie','#aaaaaa'),('final_loss','#b65151')]):
  prior=sum(q[m].sum() for m,_ in [('final_win','#34865a'),('final_tie','#aaaaaa'),('final_loss','#b65151')][:k])
  axes[1].bar(i,int(q[metric].sum()),bottom=prior,color=color,label=metric if i==0 else None)
axes[1].set_xticks(range(3),[170,310,520]);axes[1].set_xlabel('Evaluations before intervention')
axes[1].set_ylabel('True opportunities');axes[1].legend()
fig.savefig(out/'continuation.png',dpi=180);fig.savefig(out/'continuation.pdf');plt.close(fig)
text=['# 单次有益提案干预的长期收益诊断','',
 '冻结未校准的上下文提案版本New。36个配置、每配置一个新实例、四个独立初始种群、三个固定模型种子。432条原轨迹，各在170/310/520次评估处分支，共1,296条分支。',
 '先完整运行原轨迹，再在独立副本中查询真值。当两个新提案中的最好者优于已有最好值及原拟选两点，保留第一个原选择，用该新提案替换第二个位置；否则不干预。之后恢复原策略。',
 '这是一种借助事后真值的单次干预诊断，不是可部署算法，不是所有可行干预或融合策略的上界。','',
 '|分支位置|预定状态数|真实机会数|即时收益差（全部状态）|最终收益差（全部状态）|最终95%区间|机会中最终胜/平/负|',
 '|---|---:|---:|---:|---:|---|---|']
for n,s in r['stages'].items():
 e=s['final'];text.append(f"|{n}|{s['states']}|{s['opportunities']}|{s['immediate']['mean']:.6f}|{e['mean']:.6f}|[{e['lower']:.6f}, {e['upper']:.6f}]|{s['opportunity_final_wins']}/{s['opportunity_final_ties']}/{s['opportunity_final_losses']}|")
a=r['aggregate'];e=a['final']
text+=['',f"三个阶段等权汇总：最终收益差{e['mean']:.6f}，95%区间[{e['lower']:.6f}, {e['upper']:.6f}]。",
 f"按预先规则，本轮方向判断：`{r['direction']}`。",'',
 '优先继续选择机制开发的门槛为汇总最终收益≥0.001、区间下界>0且三个种子均值均为正。条件机会统计不是另一个可替代的验收终点；阶段表不用于事后挑选成功阶段。',
 '区间按12配方簇bootstrap5000次，为当前三个模型上的描述性区间，不能当作新函数族泛化证据。',
 '全部干预前状态/随机数指纹和前缀点/轨迹一致；不干预分支要求全程一致；修改只发生在一个评估位置。',
 '实际计算调用1,036,800次，其中432,000次为分支重复前缀；事后教师5,184次；独立契约检查456次。这些调用不是新的可部署算法测试成绩。',
 '复现：`.venv/bin/python scripts/branch_continuation.py`。源代码、协议、全部状态、动作、标签、原轨迹和分支轨迹均有身份哈希与归档。','',
 '[预登记协议](../../experiments/BRANCH_CONTINUATION_PROTOCOL.md) · [阶段图](continuation.png) · [逐对结果](pairs.csv) · [归档](../../../artifacts/branch_continuation_v1/README.md)']
(out/'CONCLUSIONS.zh-CN.md').write_text('\n'.join(text)+'\n')
dest=ROOT/'artifacts/branch_continuation_v1';dest.mkdir(parents=True,exist_ok=True)
groups={}
for p in sorted(run.rglob('*')):
 if not p.is_file() or p.suffix=='.lock':continue
 rel=p.relative_to(run)
 name='cases_'+p.name.split('_')[0] if rel.parts[0]=='cases' and p.stem!='identity' else 'metadata_sources'
 groups.setdefault(name,[]).append(p)
manifest={}
for name,paths in groups.items():
 target=dest/(name+'.zip')
 with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
  for p in paths:z.write(p,str(p.relative_to(run)))
  if name=='metadata_sources':z.write(__file__,'reporting_source/report_branch_continuation.py')
 with zipfile.ZipFile(target) as z:assert z.testzip() is None
 assert target.stat().st_size<95000000
 manifest[target.name]=dict(sha256=sha256(target),bytes=target.stat().st_size)
write_json(dest/'MANIFEST.json',manifest)
write_json(out/'PACKAGE_VERIFICATION.json',dict(cases=cases,main_calls=main,teacher_calls=teachers,
 hashes_passed=True,zip_crc_passed=True,original_weights_preserved=True))
(dest/'README.md').write_text('# One-intervention continuation v1\n\nExtract allZIPs into results/branch_continuation_20260930/.\nIncludes full states, oracle action labels, base/branch trails and points, frozen sources and protocol.\nProposal checkpoint dependency: complementary_proposal_v1. Oracle-assisted diagnosis, not deployable performance.\nSee docs/revision/branch_continuation/CONCLUSIONS.zh-CN.md.\n')
print(json.dumps(manifest,indent=2))
