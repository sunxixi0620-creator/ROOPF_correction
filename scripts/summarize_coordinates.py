import csv,json,sys
from pathlib import Path
import numpy as np
root=Path(sys.argv[1]).resolve()
assert (root/'COMPLETE').exists()
rows=list(csv.DictReader((root/'raw_results.csv').open()))
variants=list(dict.fromkeys(r['variant'] for r in rows))
cases=list(dict.fromkeys(r['case'] for r in rows))
summary=[]
rng=np.random.default_rng(9407)
for case in cases:
    base={r['seed']:float(r['final']) for r in rows if r['case']==case and r['variant']=='legacy'}
    for mode in variants:
        rr=sorted([r for r in rows if r['case']==case and r['variant']==mode],key=lambda r:int(r['seed']))
        a=np.array([float(r['final']) for r in rr]);b=np.array([base[r['seed']] for r in rr])
        delta=a-b;tol=1e-10+1e-6*np.abs(b)
        boot=delta[rng.integers(0,len(a),size=(10000,len(a)))].mean(axis=1)
        ci=np.quantile(boot,[.025,.975])
        summary.append({'case':case,'variant':mode,'n':len(a),'mean':a.mean(),'sample_std':a.std(ddof=1),'median':np.median(a),'mean_improvement_percent':100*(1-a.mean()/b.mean()) if b.mean()!=0 else float('nan'),'wins':int((delta < -tol).sum()),'ties':int((np.abs(delta)<=tol).sum()),'losses':int((delta>tol).sum()),'paired_mean_difference':delta.mean(),'bootstrap95_low':ci[0],'bootstrap95_high':ci[1],'batch_seconds':float(rr[0]['batch_seconds']),'mean_portfolio_evals':np.mean([float(r['portfolio_evals']) for r in rr])})
with (root/'summary.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=summary[0]);w.writeheader();w.writerows(summary)
names={'legacy':'原版','unlocked':'仅解除候选锁','centroid_slot':'精英中心第二槽位','relative_slot':'局部坐标模型第二槽位'}
lines=['# 坐标改进实验：具体数值结果','','## 协议','','10维；初始种群100；总评估300；每条件10个优化种子；每函数一个固定随机移位实例和一个零移位实例。第一锚点始终保留。表中为终点目标值均值 ± 样本标准差，越小越好。所有目标最优值为0。','',f'已完成{len(rows)}条轨迹，总计{len(rows)*300}次候选级目标评估。全部独立预算检查通过。模型未重训，坐标适配为启发式实验。','']
for condition,title in [('zero','零移位回归结果'),('shifted','固定随机移位结果')]:
    lines.extend([f'## {title}','','|函数|'+'|'.join(names[v] for v in variants)+'|','|---|'+'---:|'*len(variants)])
    for case in cases:
        if not case.endswith('_'+condition):continue
        cells=[]
        for v in variants:
            s=next(s for s in summary if s['case']==case and s['variant']==v)
            cells.append(f"{s['mean']:.6g} ± {s['sample_std']:.3g}")
        lines.append('|'+case+'|'+'|'.join(cells)+'|')
    lines.append('')
lines.extend(['## 配对对照原版','','正改善率表示均值降低；负值表示退化。胜平负按差值容差1e-10+1e-6×|baseline|计，不代表统计显著性。','','|条件|版本|均值改善率|胜/平/负|配对平均差95% bootstrap区间|','|---|---|---:|---|---|'])
for s in summary:
    if s['variant']=='legacy':continue
    lines.append(f"|{s['case']}|{names[s['variant']]}|{s['mean_improvement_percent']:.2f}%|{s['wins']}/{s['ties']}/{s['losses']}|[{s['bootstrap95_low']:.6g}, {s['bootstrap95_high']:.6g}]|")
lines.extend(['','## 解释边界','','区间来自10个种子的配对均值差bootstrap，未作多重比较校正，属于探索性描述。一个随机移位实例不能代表跨实例泛化；观察这些结果后作出的设计改变必须在新实例验证。原版对照在同一批量环境重跑，不能与前一轮单轨迹数值逐位比较。','', '“精英中心”是重要简单对照：若局部坐标模型不能优于它，不能把坐标适配的收益归因于学习能力。移位收益需与零移位回归损失同时报告。','','原始数据：raw_results.csv；实例：instances.json；参数/哈希：protocol.json；收敛轨迹：各npz；付费候选：各candidates.csv。'])
(root/'REPORT.zh-CN.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps(summary,ensure_ascii=False,default=float))
