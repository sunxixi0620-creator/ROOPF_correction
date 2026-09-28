import json,time,zipfile,hashlib,multiprocessing
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import pandas as pd
import torch
from residual_ranking_study import ROOT,OUT,PREV,SEEDS,NAMES,path,net,sha,save,load
DIAG=['original']+[f'{t}_{s}' for t in ['ranking','magnitude'] for s in SEEDS]
def diagnostic(i,n):
 import residual_decision_audit as a
 a.path=path
 p=OUT/f'diag_{i:02d}_{n}.csv'
 if p.exists():return
 q=torch.load(OUT/'tasks.pt',weights_only=False)[i];z=a.run(q,n,True)
 reference=pd.read_csv(OUT/'confirmation.csv');reference=reference[(reference.method==n)&(reference.fid==q['fid'])].sort_values('seed_index')
 assert np.array_equal(z['trail'][:,-1].numpy(),reference.final.to_numpy().astype(np.float32))
 pd.DataFrame(z['records']).to_csv(p,index=False)
def wtl(x):return [int((x<0).sum()),int((x==0).sum()),int((x>0).sum())]
def main():
 while not (OUT/'COMPLETE').exists():time.sleep(10)
 p=json.loads((OUT/'protocol.json').read_text());assert p['script']==sha(ROOT/'scripts/residual_ranking_study.py');assert p['protocol']==sha(ROOT/'docs/experiments/RESIDUAL_RANKING_PROTOCOL.md')
 assert p['anchor']==sha(ROOT/'checkpoints/anchor_policy_d10.pt')
 for name,h in p['data'].items():assert sha(PREV/name)==h
 for name,h in p['checkpoints'].items():assert sha(path(name))==h
 save(OUT/'diagnostic_protocol.json',dict(protocol=sha(ROOT/'docs/experiments/RESIDUAL_RANKING_DIAGNOSTIC_PROTOCOL.md'),observer=sha(ROOT/'scripts/residual_decision_audit.py'),script=sha(__file__)))
 with ProcessPoolExecutor(8,mp_context=multiprocessing.get_context('spawn')) as ex:
  fs=[ex.submit(diagnostic,i,n) for i in range(36) for n in DIAG]
  for j,f in enumerate(as_completed(fs)):
   f.result()
   if (j+1)%20==0:print('DIAGNOSTIC',j+1,180,flush=True)
 torch.set_num_threads(4);tx,_=load('train');mean=tx.reshape(-1,tx.shape[-1]).mean(0);std=np.maximum(tx.reshape(-1,tx.shape[-1]).std(0),1e-6)
 ds=[np.load(OUT/f'case_{i:02d}_teacher.npz') for i in range(36)];x=np.concatenate([d['x'] for d in ds]);fit=np.concatenate([d['fit'] for d in ds]);ev=np.concatenate([d['eval_before'] for d in ds]).reshape(-1,38)[:,0];assert np.isfinite(x).all() and np.isfinite(fit).all()
 selections={};metrics={};ij=np.triu_indices(36,1)
 for n in [n for n in NAMES if n!='no_residual']:
  pp=torch.load(path(n),weights_only=False,map_location='cpu');model=net();model.load_state_dict({k.removeprefix('net.'):v for k,v in pp['state_dict'].items()});model.eval()
  if n.startswith(('ranking','magnitude')):
   assert np.array_equal(np.array(pp['mean'],dtype=np.float32),mean);assert np.array_equal(np.array(pp['std'],dtype=np.float32),std)
   assert sha(path(n))==json.loads((OUT/'SELECTION_FROZEN.json').read_text())[n]
   h=json.loads((OUT/(n+'_history.json')).read_text());best=float('inf');chosen=None
   for row in h:
    if row['validation_loss']<best-1e-5:best=row['validation_loss'];chosen=row['epoch']
   assert chosen==pp['epoch'];selections[n]=dict(epochs_run=len(h),selected_epoch=chosen,validation_loss=best)
  xx=torch.tensor((x-np.array(pp['mean'],dtype=np.float32))/np.maximum(np.array(pp['std'],dtype=np.float32),1e-6))
  with torch.no_grad():logits=torch.cat([model(z).view(-1) for z in xx.split(16384)]).numpy().reshape(-1,38)
  assert np.isfinite(logits).all();metrics[n]={}
  for phase in ['all','late']:
   mask=np.ones(len(fit),dtype=bool) if phase=='all' else ev>=210
   truth=fit[mask,2:];scores=torch.sigmoid(torch.tensor(logits[mask,2:])).numpy();sign=np.sign(truth[:,ij[1]]-truth[:,ij[0]]);pred=np.sign(scores[:,ij[0]]-scores[:,ij[1]]);valid=sign!=0
   concordance=float(((sign==pred)&valid).sum()+.5*((pred==0)&valid).sum())/max(valid.sum(),1)
   chosen=scores.argmax(1);top=truth[np.arange(len(truth)),chosen];scale=fit[mask].std(1).clip(1e-8);regret=(top-truth.min(1))/scale
   metrics[n][phase]=dict(portfolio_pair_concordance=concordance,mean_normalized_top1_regret=float(regret.mean()),median_normalized_top1_regret=float(np.median(regret)),top1_beats_second=float((top<fit[mask,1]).mean()))
 raw=pd.read_csv(OUT/'confirmation.csv');assert len(raw)==1152 and not raw.duplicated(['method','fid','seed_index']).any();assert (raw.nfe==300).all();comparisons={}
 for n,g in raw.groupby('method'):
  z=dict(bounded_gain=float(g.bounded_gain.mean()))
  for base in ['original','no_residual']:
   diff=g.set_index(['fid','seed_index']).final-raw[raw.method==base].set_index(['fid','seed_index']).final
   z['wtl_vs_'+base]=wtl(diff.groupby(level='fid').mean())
  comparisons[n]=z
 decisions=pd.concat([pd.read_csv(f) for f in sorted(OUT.glob('diag_*.csv'))],ignore_index=True);assert len(decisions)==32400
 diagnostics={}
 for n,d in decisions.groupby('model'):
  c=d[d.changed==1];diagnostics[n]=dict(decisions=len(d),changed=len(c),fraction=len(c)/len(d),immediate_wtl=wtl(c.selected-c.counterfactual),incumbent_improvements=int((c.selected<c.incumbent-1e-12).sum()))
 report=dict(selections=selections,metrics=metrics,optimization=comparisons,diagnostics=diagnostics,budgets=json.loads((OUT/'COMPLETE').read_text()),scope='Known generated families; all scores/thresholds frozen; no probability calibration interpretation for ranking/magnitude scores')
 report['budgets'].update(diagnostic_trajectories=720,diagnostic_main_points=216000,diagnostic_teacher_points=97200)
 save(OUT/'REPORT.json',report)
 lines=['# 排序与改善幅度目标实验','',report['scope'],'','## 验证选模','','| 模型 | 训练轮数 | 选中轮次 | 验证目标损失 |','|---|---:|---:|---:|']
 for n,z in selections.items():lines.append(f"| {n} | {z['epochs_run']} | {z['selected_epoch']} | {z['validation_loss']:.6f} |")
 lines+=['','不同目标损失不可直接比较。ranking为非相等真值候选对的logistic损失；magnitude为标准化改善幅度有界分数的MSE。现有sigmoid与门控不变，分数不解释为概率。','','## 完整ROOPF新确认实例','','| 方法 | 得分↑ | 相对原最终版胜/平/负 | 相对无residual胜/平/负 |','|---|---:|---|---|']
 for n,z in comparisons.items():lines.append(f"| {n} | {z['bounded_gain']:.6f} | {'/'.join(map(str,z['wtl_vs_original']))} | {'/'.join(map(str,z['wtl_vs_no_residual']))} |")
 lines+=['','得分是归一化改善的有界变换，不是成功率。WTL按36个函数实例均值计算，没有显著性声明。','','## 后期确认候选池的排序质量','','候选池由无residual行为策略产生。使用部署时的float32 sigmoid评分，只比较36个portfolio候选，忽略真值相等的配对，预测相等计半分。top1 regret为选中候选相对池内最佳候选的目标值差除以38候选的真值标准差。','','| 模型 | 配对排序正确率↑ | 平均标准化top1 regret↓ |','|---|---:|---:|']
 for n,z in metrics.items():z=z['late'];lines.append(f"| {n} | {z['portfolio_pair_concordance']:.4f} | {z['mean_normalized_top1_regret']:.4f} |")
 lines+=['','## 同状态实际改选','','| 模型 | 改选数/6480 | 比例 | 即时胜/平/负 | 改选后改善历史最优 |','|---|---:|---:|---|---:|']
 for n,z in diagnostics.items():lines.append(f"| {n} | {z['changed']} | {z['fraction']:.2%} | {'/'.join(map(str,z['immediate_wtl']))} | {z['incumbent_improvements']} |")
 lines+=['','不同模型访问状态不同，跨模型改选统计不等同于相同状态上直接比较所有模型。诊断重复运行的最终值与普通评测逐轨迹一致。','','## 预算','```json',json.dumps(report['budgets'],indent=2),'```']
 (ROOT/'docs/experiments/RESIDUAL_RANKING_RESULTS.zh-CN.md').write_text('\n'.join(lines)+'\n')
 dest=ROOT/'artifacts/residual_ranking_study';dest.mkdir(exist_ok=True);files=sorted(f for f in OUT.iterdir() if f.is_file());groups=[];g=[];size=0
 for f in files:
  if g and size+f.stat().st_size>40000000:groups.append(g);g=[];size=0
  g.append(f);size+=f.stat().st_size
 if g:groups.append(g)
 manifest=[]
 for i,g in enumerate(groups):
  zp=dest/f'part{i+1:03d}.zip'
  with zipfile.ZipFile(zp,'w',zipfile.ZIP_DEFLATED) as z:
   for f in g:z.write(f,f.name)
  hs={f.name:sha(f) for f in g}
  with zipfile.ZipFile(zp) as z:
   assert z.testzip() is None
   for n,h in hs.items():assert hashlib.sha256(z.read(n)).hexdigest()==h
  manifest.append(dict(file=zp.name,sha256=sha(zp),members=hs))
 save(dest/'manifest.json',manifest);save(dest/'REPORT.json',report)
 (dest/'README.md').write_text('Restore ZIP contents into results/residual_ranking_study. Verify SHA256 manifest. Training input data remain in artifacts/residual_target_study. See docs/experiments/RESIDUAL_RANKING_*.\n')
 save(ROOT/'docs/experiments/RESIDUAL_RANKING_INTEGRITY.json',dict(passed=True,archived_files=len(files),original_checkpoints_unchanged=True))
 print('RANKING STUDY COMPLETE',flush=True)
if __name__=='__main__':main()
