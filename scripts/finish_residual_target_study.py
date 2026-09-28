"""Audit, summarize and archive the completed residual target experiment."""
import json,time,zipfile
import numpy as np
import pandas as pd
import torch
from residual_target_study import ROOT,OUT,SEEDS,net,payload,sha,save,data

def main():
    while not (OUT/'COMPLETE').exists():time.sleep(10)
    torch.set_num_threads(4)
    protocol=json.loads((OUT/'protocol.json').read_text())
    assert protocol['script']==sha(ROOT/'scripts/residual_target_study.py')
    assert protocol['protocol']==sha(ROOT/'docs/experiments/RESIDUAL_TARGET_PROTOCOL.md')
    assert protocol['tasks']==sha(OUT/'tasks.pt')
    for name,h in protocol['original'].items():assert sha(ROOT/'checkpoints'/name)==h
    tasks=torch.load(OUT/'tasks.pt',weights_only=False);assert len(tasks)==144
    # Assert disjoint seeds both within and between roles/splits.
    seeds=[q['seed'] for q in tasks];assert len(set(seeds))==144
    assert len(set(seeds+[s+10000000 for s in seeds]+[s+20000000 for s in seeds]))==432
    feature_names=payload()['feature_names'];best_col=feature_names.index('fitness_best_before')
    for i,q in enumerate(tasks):
        d=np.load(OUT/f'case_{i:03d}.npz');assert d['x'].shape==(15200,len(feature_names));assert np.isfinite(d['x']).all()
        fit=d['fit'];assert fit.shape==(400,38) and np.isfinite(fit).all()
        assert np.array_equal(d['second'],(fit<fit[:,1:2]).reshape(-1))
        # Logger comparison runs in float32, including its epsilon subtraction.
        incumbent=(torch.tensor(fit.reshape(-1),dtype=torch.float32)<torch.tensor(d['x'][:,best_col])-1e-12).numpy()
        assert np.array_equal(d['incumbent'],incumbent)
    tr=data('train');co=data('confirmation');mean=tr['x'].mean(0);std=np.maximum(tr['x'].std(0),1e-6)
    models={'original':ROOT/'checkpoints/residual_selector_generated36_d10.pt',**{f'{t}_{s}':OUT/f'{t}_{s}.pt' for t in ['incumbent','second'] for s in SEEDS}}
    selections={};metrics={};cal=[]
    for tag,path in models.items():
        p=torch.load(path,map_location='cpu',weights_only=False);m=net();m.load_state_dict({k.removeprefix('net.'):v for k,v in p['state_dict'].items()});m.eval()
        if tag!='original':
            assert np.array_equal(np.array(p['mean'],dtype=np.float32),mean) and np.array_equal(np.array(p['std'],dtype=np.float32),std)
            h=json.loads((OUT/(tag+'_history.json')).read_text());best=float('inf');chosen=None
            for row in h:
                if row['validation_bce']<best-1e-5:best=row['validation_bce'];chosen=row['epoch']
            assert chosen==p['epoch'];assert sha(path)==json.loads((OUT/'SELECTION_FROZEN.json').read_text())[tag]
            selections[tag]=dict(selected_epoch=chosen,epochs_run=len(h),validation_bce=best)
        x=torch.tensor((co['x']-np.array(p['mean'],dtype=np.float32))/np.maximum(np.array(p['std'],dtype=np.float32),1e-6))
        with torch.no_grad():pred=torch.cat([torch.sigmoid(m(z).view(-1)) for z in x.split(16384)]).numpy()
        assert np.isfinite(pred).all();metrics[tag]={}
        # Both labels evaluated for every model, but cross-target scores are descriptive, not calibrated claims.
        for target in ['incumbent','second']:
            for phase in ['all','late']:
                mask=np.ones(len(pred),dtype=bool) if phase=='all' else co['eval_before']>=210
                y=co[target][mask];pr=pred[mask];clip=np.clip(pr.astype(float),1e-7,1-1e-7);prior=float(tr[target].mean())
                metrics[tag][target+'_'+phase]=dict(n=len(y),prevalence=float(y.mean()),bce=float(-(y*np.log(clip)+(1-y)*np.log(1-clip)).mean()),brier=float(((pr-y)**2).mean()),constant_train_prior_brier=float(((prior-y)**2).mean()))
                for b in range(10):
                    selected=(pr>=b/10)&((pr<(b+1)/10) if b<9 else (pr<=1))
                    if selected.any():cal.append(dict(model=tag,target=target,phase=phase,bin=b,n=int(selected.sum()),mean_prediction=float(pr[selected].mean()),positive_rate=float(y[selected].mean())))
                # Consecutive38 rows form one pool; report portfolio-only top5.
                poolmask=mask.reshape(-1,38)[:,0];pp=pred.reshape(-1,38)[poolmask,2:];yy=co[target].reshape(-1,38)[poolmask,2:]
                idx=np.argsort(-pp,axis=1,kind='stable')[:,:5]
                metrics[tag][target+'_'+phase]['portfolio_top5_precision']=float(np.take_along_axis(yy,idx,axis=1).mean())
                metrics[tag][target+'_'+phase]['portfolio_prevalence']=float(yy.mean())
    raw=pd.read_csv(OUT/'confirmation.csv');assert len(raw)==864 and not raw.duplicated(['method','fid','seed_index']).any();assert (raw.nfe==300).all();assert np.isfinite(raw.final).all()
    comparisons={}
    for name,g in raw.groupby('method'):
        rec={'bounded_gain':float(g.bounded_gain.mean())}
        for control in ['original','no_residual']:
            baseline=raw[raw.method==control].set_index(['fid','seed_index']).final
            diff=g.set_index(['fid','seed_index']).final-baseline;means=diff.groupby(level='fid').mean()
            rec['wtl_vs_'+control]=[int((means<0).sum()),int((means==0).sum()),int((means>0).sum())]
        comparisons[name]=rec
    report=dict(scope='Two training seeds; known36 families, fresh instance splits; no external test selection',selections=selections,metrics=metrics,optimization=comparisons,budgets=dict(training_candidate_rows=1094400,validation_candidate_rows=547200,confirmation_candidate_rows=547200,collection_main_points=172800,collection_teacher_points=2188800,confirmation_optimization_points=259200,confirmation_initial_points=14400,formal_trajectory_executions=1440,preflight_parity_main_points=2400,preflight_parity_teacher_points=15200),checks='passed: labels, seed splits, finite features, training-only normalization, early-stop selection, frozen hashes, paired final rows')
    save(OUT/'REPORT.json',report);pd.DataFrame(cal).to_csv(OUT/'calibration.csv',index=False)
    lines=['# Residual训练目标对照结果','',report['scope'],'','原始最终模型保持不变。新模型仅改变residual训练数据/目标；同一新数据上的两种目标使用相同结构和随机种子。新数据包含anchor行，与历史训练集不同，因此不能把新旧模型全部差异归因于标签。','','## 早停与选模','','| 模型 | 实际训练轮数 | 选中轮次 | 验证BCE |','|---|---:|---:|---:|']
    for n,s in selections.items():lines.append(f"| {n} | {s['epochs_run']} | {s['selected_epoch']} | {s['validation_bce']:.6f} |")
    lines+=['','## 完整ROOPF确认评测','','每个方法36个新实例×4条轨迹，300NFE。得分是归一化改善的有界变换，不是成功率。胜平负按36个函数实例均值计算，未作显著性声明。','','| 模型 | 得分↑ | 相对原最终版胜/平/负 | 相对无residual胜/平/负 |','|---|---:|---|---|']
    for n,s in comparisons.items():lines.append(f"| {n} | {s['bounded_gain']:.6f} | {'/'.join(map(str,s['wtl_vs_original']))} | {'/'.join(map(str,s['wtl_vs_no_residual']))} |")
    lines+=['','## 后期候选标签表现','','以下为所有38候选的Brier；top5仅取36个portfolio候选。不同标签对应不同任务，BCE/Brier不能跨任务直接解释为同一种性能。','','| 模型 | 目标 | 正例率 | Brier↓ | 训练先验常数Brier↓ | portfolio top5精度 | portfolio正例率 |','|---|---|---:|---:|---:|---:|---:|']
    for n,mm in metrics.items():
        for t in ['incumbent','second']:
            z=mm[t+'_late'];lines.append(f"| {n} | {t} | {z['prevalence']:.4f} | {z['brier']:.4f} | {z['constant_train_prior_brier']:.4f} | {z['portfolio_top5_precision']:.4f} | {z['portfolio_prevalence']:.4f} |")
    lines+=['','校准分箱详见归档calibration.csv。标签确认集由无residual行为策略产生，不能等同于新模型部署后的状态分布；完整优化确认评测单独执行以检验实际收益。验证与确认仍为已知函数族的新实例。','', '数据预算：', '```json',json.dumps(report['budgets'],indent=2),'```']
    (ROOT/'docs/experiments/RESIDUAL_TARGET_RESULTS.zh-CN.md').write_text('\n'.join(lines)+'\n')
    dest=ROOT/'artifacts/residual_target_study';dest.mkdir(exist_ok=True);files=sorted(p for p in OUT.iterdir() if p.is_file());groups=[];group=[];size=0
    for p in files:
        if group and size+p.stat().st_size>40_000_000:groups.append(group);group=[];size=0
        group.append(p);size+=p.stat().st_size
    if group:groups.append(group)
    manifest=[]
    for i,group in enumerate(groups):
        path=dest/f'part{i+1:03d}.zip'
        with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED) as z:
            for p in group:z.write(p,p.name)
        with zipfile.ZipFile(path) as z:assert z.testzip() is None
        manifest.append(dict(file=path.name,sha256=sha(path),members={p.name:sha(p) for p in group}))
    save(dest/'manifest.json',manifest);save(dest/'REPORT.json',report)
    (dest/'README.md').write_text('Restore all ZIP files into results/residual_target_study. Manifest records ZIP and member SHA256. See docs/experiments/RESIDUAL_TARGET_PROTOCOL.md and RESIDUAL_TARGET_RESULTS.zh-CN.md.\n')
    save(ROOT/'docs/experiments/RESIDUAL_TARGET_INTEGRITY.json',dict(passed=True,files=len(files),archives=len(groups),original_checkpoints_unchanged=True))
    print('AUDIT AND ARCHIVE COMPLETE',flush=True)
if __name__=='__main__':main()
