"""Wait for declared development runs, then freeze choices and confirm once."""
import json,time,zipfile,hashlib,sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from training_validation import OUT,ROOT,sha,evaluate
from roopf.anchor_backbone import AnchorPolicyBackbone

def metrics(q,y):
    y=np.asarray(y,dtype=float);order=np.argsort(-q);k=max(1,len(y)//20)
    return {'n':len(y),'positive_rate':float(y.mean()),'mean_prediction':float(q.mean()),
            'brier':float(np.mean((q-y)**2)),'top5_precision':float(y[order[:k]].mean())}

def summarize():
    report={'scope':'one paired continuation stream; generated instance-held-out development and post-selection confirmation',
            'historical_label_availability':json.loads((OUT/'historical_label_availability.json').read_text())}
    report['selection']={a:json.loads((OUT/a/'selection.json').read_text()) for a in ['legacy','shared_scale']}
    report['validation_curve']={a:[{k:v for k,v in json.loads(p.read_text()).items() if k!='rows'} for p in sorted((OUT/a).glob('validation_*.json'))] for a in ['legacy','shared_scale']}
    # Every model is fixed before opening confirmation outcomes.
    selection_file=OUT/'frozen_confirmation_selection.json'
    selection={'selected':report['selection'],'checkpoints':{a:sha(OUT/a/'selected.pt') for a in report['selection']}}
    if selection_file.exists():assert json.loads(selection_file.read_text())==selection
    else:selection_file.write_text(json.dumps(selection,indent=2))
    confirm=OUT/'confirmation_results';confirm.mkdir(exist_ok=True)
    model=AnchorPolicyBackbone(10,200,100).cuda().eval()
    choices={'original_final':ROOT/'checkpoints/anchor_policy_d10.pt','start_epoch80':OUT/'legacy/epoch_080.pt',
             'legacy_selected':OUT/'legacy/selected.pt','shared_scale_selected':OUT/'shared_scale/selected.pt'}
    for name,path in choices.items():
        dest=confirm/(name+'.json')
        if dest.exists():continue
        model.load_state_dict(torch.load(path,map_location='cuda',weights_only=True))
        res=evaluate(model,'confirmation');res['checkpoint_sha256']=sha(path)
        dest.write_text(json.dumps(res,indent=2));print('confirmation',name,res['score'],flush=True)
    results={n:json.loads((confirm/(n+'.json')).read_text()) for n in choices}
    rows=[]
    for n,res in results.items():rows.extend([{'method':n,**r} for r in res['rows']])
    df=pd.DataFrame(rows);df.to_csv(OUT/'confirmation_raw.csv',index=False)
    base=df[df.method=='start_epoch80'].set_index(['fid','instance','seed_index'])['final'];comparisons={}
    for n in choices:
        d=df[df.method==n].set_index(['fid','instance','seed_index'])['final']-base
        means=d.groupby(level=['fid','instance']).mean()
        comparisons[n]={'bounded_gain':results[n]['score'],'case_mean_wtl':[int((means<0).sum()),int((means==0).sum()),int((means>0).sum())],
                        'trajectory_wtl':[int((d<0).sum()),int((d==0).sum()),int((d>0).sum())]}
    report['confirmation_vs_epoch80']=comparisons
    aux=pd.read_csv(OUT/'auxiliary/raw_results.csv');base=aux[aux.method=='full'].set_index(['fid','instance','seed_index'])['final'];ac={}
    for mode in ['memory_only','uniform']:
        d=aux[aux.method==mode].set_index(['fid','instance','seed_index'])['final']-base;m=d.groupby(level=['fid','instance']).mean()
        ac[mode]={'case_mean_wtl_vs_full':[int((m<0).sum()),int((m==0).sum()),int((m>0).sum())],
                  'trajectory_wtl_vs_full':[int((d<0).sum()),int((d==0).sum()),int((d>0).sum())]}
    report['auxiliary']=ac
    holds=set(json.loads((ROOT/'artifacts/training_provenance/residual_training_metrics.json').read_text())['holdout_fids'])
    tasks=torch.load(OUT/'validation.pt',map_location='cpu',weights_only=False);labels=[]
    for i,t in enumerate(tasks):
        with np.load(OUT/'auxiliary'/f'case_{i:02d}_labels.npz') as x:
            d={k:x[k] for k in ['q','improved_best','beats_second','beats_best_anchor','eval_before']}
        d['holdout']=np.full(len(d['q']),'generated36:'+t['fid'] in holds);labels.append(d)
    allx={k:np.concatenate([d[k] for d in labels]) for k in labels[0]};rr={}
    for name,mask in [('all',np.ones(len(allx['q']),dtype=bool)),('late',allx['eval_before']>=210),
                      ('residual_holdout_late',allx['holdout']&(allx['eval_before']>=210))]:
        q=allx['q'][mask];rr[name]={k:metrics(q,allx[k][mask]) for k in ['improved_best','beats_second','beats_best_anchor']}
        rr[name]['incumbent_second_disagreement']=float(np.mean(allx['improved_best'][mask]!=allx['beats_second'][mask]))
        rr[name]['best_anchor_second_disagreement']=float(np.mean(allx['beats_best_anchor'][mask]!=allx['beats_second'][mask]))
    report['residual_label_audit']=rr
    report['counts']={'anchor_extra_training_points':sum(json.loads((OUT/a/'COMPLETE').read_text())['training_points'] for a in ['legacy','shared_scale']),
        'validation_runs':sum(len(list((OUT/a).glob('validation_*.json')))*288 for a in ['legacy','shared_scale']),
        'confirmation_runs':len(df),'auxiliary_runs':len(aux),'auxiliary_teacher_points':1094400,
        'preflight':'288anchor validation trajectories and two four-trajectory full runs, with15200teacher points; excluded from formal counts'}
    (OUT/'REPORT.json').write_text(json.dumps(report,indent=2))
    return report

def publish(r):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    doc=ROOT/'docs/experiments';fig,ax=plt.subplots(figsize=(7,4))
    for arm,h in r['validation_curve'].items():ax.plot([x['epoch'] for x in h],[x['score'] for x in h],marker='o',label=arm)
    ax.set(xlabel='Total epoch (continuation begins at80)',ylabel='Validation bounded normalized gain',title='Same start, different training objectives');ax.legend();fig.tight_layout()
    fig.savefig(doc/'TRAINING_VALIDATION_CURVES.png',dpi=170);fig.savefig(doc/'TRAINING_VALIDATION_CURVES.pdf');plt.close(fig)
    lines=['# 训练充分性开发实验结果','',
      '本轮是同一80轮训练状态的配对续训，不是从头重训或外部基准领先证明。原最终10-D权重不变。',
      '验证集与确认集均为已知36生成函数的新实例，独立参数种子；不声称全新函数族。只有一个续训随机流。','',
      '## 验证选模','', '| 训练目标 | 验证选中总轮次 | 验证得分 |','|---|---:|---:|']
    for a,s in r['selection'].items():lines.append(f"| {a} | {s['epoch']} | {s['score']:.6f} |")
    lines+=['','停止轮数不影响衰减日程：两组均在80轮后令多样性项权重为0；其余训练设置相同。选模指标为归一化真实改善的有界变换，而不是训练loss。',
        '', '## 冻结选择后的确认集','', '| 方法 | 确认得分 | 相对80轮起点的72条件均值胜/平/负 |','|---|---:|---|']
    for a,s in r['confirmation_vs_epoch80'].items():lines.append(f"| {a} | {s['bounded_gain']:.6f} | {'/'.join(map(str,s['case_mean_wtl']))} |")
    lines+=['','胜平负是描述性比较，没有多重比较显著性声明。实例与搜索初始化不能当成多个训练种子。',
            '', '## 未训练gate的必要性对照','']
    for a,s in r['auxiliary'].items():lines.append(f"- {a}相对full：72条件均值胜/平/负为{'/'.join(map(str,s['case_mean_wtl_vs_full']))}。")
    lines+=['','memory_only仅移除随机gate logits、保留成功记忆；uniform同时去除成功记忆。本轮未训练新的辅助网络，也未移除候选生成网络。',
            '', '## Residual标签','',
            '恢复的原3,240,000行日志没有anchor真值，无法直接重标为实际替换标签。新诊断保留两个anchor真值，明确比较第二个anchor；额外教师求值不更新在线状态。']
    for name,d in r['residual_label_audit'].items():lines.append(f"- {name}：incumbent与第二anchor标签分歧率{d['incumbent_second_disagreement']:.2%}；best-of-two与第二anchor标签分歧率{d['best_anchor_second_disagreement']:.2%}。")
    lines+=['','不同标签的正例率、Brier和top5精度见机器可读REPORT.json。标签分歧本身不证明改标签就能提高最终优化效果；本轮没有在验证集上重新训练residual。',
            '', '## 范围与交付','',json.dumps(r['counts'],ensure_ascii=False,indent=2),
            '', '图见TRAINING_VALIDATION_CURVES.pdf；数据、模型与哈希见artifacts/training_validation。所有原有外部测试成绩均未用于选模。']
    (doc/'TRAINING_VALIDATION_RESULTS.zh-CN.md').write_text('\n'.join(lines)+'\n')
    (OUT/'COMPLETE').write_text('Both continuations, frozen-choice confirmation, auxiliary and label diagnostics complete.\n')
    artifact=ROOT/'artifacts/training_validation';artifact.mkdir(exist_ok=True);paths=sorted(p for p in OUT.rglob('*') if p.is_file());groups=[];group=[];size=0
    for p in paths:
        n=p.stat().st_size;assert n<40_000_000,p
        if group and size+n>40_000_000:groups.append(group);group=[];size=0
        group.append(p);size+=n
    if group:groups.append(group)
    manifest=[]
    for i,g in enumerate(groups,1):
        zpath=artifact/f'part{i:03d}.zip';members={}
        with zipfile.ZipFile(zpath,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
            for p in g:
                rel=str(p.relative_to(ROOT));z.write(p,rel);members[rel]={'bytes':p.stat().st_size,'sha256':sha(p)}
        with zipfile.ZipFile(zpath) as z:assert z.testzip() is None
        manifest.append({'archive':zpath.name,'sha256':sha(zpath),'files':members})
    (artifact/'manifest.json').write_text(json.dumps(manifest,indent=2));(artifact/'REPORT.json').write_text(json.dumps(r,indent=2))
    (artifact/'README.md').write_text('Extract all ZIP parts at repository root; independent ZIPs, no concatenation. Manifest contains SHA256 hashes. This is one paired continuation development study, not a replacement of the frozen final model.\n')

if __name__=='__main__':
    for arm in ['legacy','shared_scale','auxiliary']:
        while not (OUT/arm/'COMPLETE').exists():time.sleep(15)
    publish(summarize());print('Development study complete and archived.',flush=True)
