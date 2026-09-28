import json,time,zipfile
import numpy as np
import pandas as pd
from residual_decision_audit import ROOT,OUT,NAMES,path,sha,save

def wtl(delta):return [int((delta<0).sum()),int((delta==0).sum()),int((delta>0).sum())]
def main():
 while not (OUT/'COMPLETE').exists():time.sleep(10)
 p=json.loads((OUT/'protocol.json').read_text())
 assert p['script']==sha(ROOT/'scripts/residual_decision_audit.py') and p['protocol']==sha(ROOT/'docs/experiments/RESIDUAL_DECISION_PROTOCOL.md')
 assert p['tasks']==sha(OUT/'tasks.pt') and p['anchor']==sha(ROOT/'checkpoints/anchor_policy_d10.pt')
 for n,h in p['checkpoints'].items():assert sha(path(n))==h
 decisions=pd.concat([pd.read_csv(f) for f in sorted(OUT.glob('*_decisions.csv'))],ignore_index=True)
 cases=[json.loads(f.read_text()) for f in sorted(OUT.glob('[0-9][0-9]_*.json'))];assert len(cases)==180
 replay=pd.DataFrame([row for c in cases for row in c['rows']]);assert not decisions.duplicated(['fid','model','batch','eval_before']).any()
 assert np.isfinite(decisions[['anchor','selected','counterfactual','incumbent','scale']]).all().all()
 assert (decisions[decisions.changed==0].selected==decisions[decisions.changed==0].counterfactual).all()
 results={}
 for name,d in decisions.groupby('model'):
  changed=d[d.changed==1];r=replay[replay.model==name].copy()
  matched=r.merge(d[['fid','model','batch','eval_before','changed']],left_on=['fid','model','batch','disabled_round'],right_on=['fid','model','batch','eval_before'],validate='one_to_one')
  cr=matched[matched.changed==1];delta=changed.selected-changed.counterfactual
  gain=(changed.counterfactual-changed.selected)/changed.scale
  result=dict(decisions=len(d),changed=len(changed),changed_fraction=float(d.changed.mean()),immediate_wtl=wtl(delta),changed_choice_beats_anchor=int((changed.selected<changed.anchor).sum()),changed_choice_improves_incumbent=int((changed.selected<changed.incumbent-1e-12).sum()),median_normalized_immediate_gain=float(gain.median()) if len(gain) else None,mean_bounded_immediate_gain=float((gain/(1+gain.abs())).mean()) if len(gain) else None,replay_trajectories=len(r),replay_final_wtl=wtl(r.factual-r.replay),replay_changed_trajectories=len(cr),replay_changed_final_wtl=wtl(cr.factual-cr.replay),unchanged_at_round_final_wtl=wtl(matched[matched.changed==0].factual-matched[matched.changed==0].replay))
  # Counts per function expose concentration without inventing additive causal attribution.
  per=changed.assign(local_win=delta<0,local_loss=delta>0,improves_best=changed.selected<changed.incumbent-1e-12).groupby('fid').agg(changes=('changed','size'),local_wins=('local_win','sum'),local_losses=('local_loss','sum'),incumbent_improvements=('improves_best','sum'))
  per.to_csv(OUT/(name+'_by_function.csv'));results[name]=result
 report=dict(results=results,budgets=json.loads((OUT/'COMPLETE').read_text()),scope='Identical-state local comparisons plus first-changed-round ablation, not an additive decomposition of total residual gains.')
 report['budgets']['main_points']=300*(report['budgets']['factual_trajectories']+report['budgets']['replay_trajectories'])
 report['budgets']['preflight_trajectories']=40
 # Ten parity runs, of which five had teacher queries at the same observer sites as corresponding formal first cases.
 report['budgets']['preflight_main_points']=12000
 report['budgets']['preflight_teacher_points']=sum(c['teacher_points'] for c in cases if c['fid']==cases[0]['fid'])
 save(OUT/'REPORT.json',report);decisions.to_csv(OUT/'decisions.csv',index=False);replay.to_csv(OUT/'replays.csv',index=False)
 lines=['# Residual决策与单次干预诊断','',report['scope'],'','全部权重冻结。36个新实例，每实例4条轨迹；均为已知生成函数族，不是外部泛化测试。五个checkpoint都通过诊断开关的点、轨迹与随机状态一致性检查。','','## 相同状态下的选择变化','','胜/平/负表示真实目标值更低/相同/更高，比较实际选择与同状态下关闭residual所得选择。改选不一定改善当前历史最优。','','| 模型 | 决策数 | 改选数（比例） | 改选后即时胜/平/负 | 改选后击败第二anchor | 改选后改善历史最优 |','|---|---:|---:|---|---:|---:|']
 for n,z in results.items():lines.append(f"| {n} | {z['decisions']} | {z['changed']} ({z['changed_fraction']:.2%}) | {'/'.join(map(str,z['immediate_wtl']))} | {z['changed_choice_beats_anchor']} | {z['changed_choice_improves_incumbent']} |")
 lines+=['','## 仅撤销首次发生改选的轮次','','同一实例四条轨迹共同取最早发生改选的轮次；只在这一轮关闭residual，此后恢复。重放前缀逐点完全一致，替换点与同状态反事实选择完全一致。下表以保留原干预的最终目标值相对撤销该轮干预计胜平负。','','| 模型 | 重放轨迹数 | 全部重放最终胜/平/负 | 当轮确实改选的轨迹数 | 确实改选子集最终胜/平/负 |','|---|---:|---|---:|---|']
 for n,z in results.items():lines.append(f"| {n} | {z['replay_trajectories']} | {'/'.join(map(str,z['replay_final_wtl']))} | {z['replay_changed_trajectories']} | {'/'.join(map(str,z['replay_changed_final_wtl']))} |")
 lines+=['','不能将这次单轮撤销等同于全程移除residual，也不能将即时收益相加解释最终收益。不同模型访问的状态可能不同，跨模型改选次数不是同一状态上的随机对照。本轮未训练或调参。','','## 预算','```json',json.dumps(report['budgets'],indent=2),'```','', '原始决策、重放轨迹和逐函数统计见artifacts/residual_decision_audit。']
 (ROOT/'docs/experiments/RESIDUAL_DECISION_RESULTS.zh-CN.md').write_text('\n'.join(lines)+'\n')
 dest=ROOT/'artifacts/residual_decision_audit';dest.mkdir(exist_ok=True);files=sorted(f for f in OUT.iterdir() if f.is_file());groups=[];g=[];size=0
 for f in files:
  if g and size+f.stat().st_size>40000000:groups.append(g);g=[];size=0
  g.append(f);size+=f.stat().st_size
 if g:groups.append(g)
 manifest=[]
 for i,g in enumerate(groups):
  zpath=dest/f'part{i+1:03d}.zip'
  with zipfile.ZipFile(zpath,'w',zipfile.ZIP_DEFLATED) as z:
   for f in g:z.write(f,f.name)
  hashes={f.name:sha(f) for f in g}
  import hashlib
  with zipfile.ZipFile(zpath) as z:
   assert z.testzip() is None
   for name,h in hashes.items():assert hashlib.sha256(z.read(name)).hexdigest()==h
  manifest.append(dict(file=zpath.name,sha256=sha(zpath),members=hashes))
 save(dest/'manifest.json',manifest);save(dest/'REPORT.json',report)
 (dest/'README.md').write_text('Extract all ZIP files into results/residual_decision_audit. Verify manifest hashes. Run scripts/finish_residual_decision_audit.py to regenerate summaries. Protocol and interpretation are under docs/experiments/RESIDUAL_DECISION_*.\n')
 save(ROOT/'docs/experiments/RESIDUAL_DECISION_INTEGRITY.json',dict(passed=True,archived_files=len(files),checkpoints_unchanged=True,parity=json.loads((OUT/'PARITY.json').read_text())))
 print('DECISION AUDIT COMPLETE',flush=True)
if __name__=='__main__':main()
