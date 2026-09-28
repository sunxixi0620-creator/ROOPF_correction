import os
os.environ.setdefault('OMP_NUM_THREADS','1')
import copy,json,multiprocessing,zipfile,hashlib
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import pandas as pd
import torch
from training_validation import ROOT,Task,source,sha
from supplementary_experiments import build
from residual_target_study import save
from roopf.model import ROOPFOptimizer,_is_local_proxy_friendly_problem
import run_roopf as r
OUT=ROOT/'results/score_consistency_audit'
def setup():
 OUT.mkdir(exist_ok=True);m=source();m.DEVICE=torch.device('cpu');tasks=[]
 for i,f0 in enumerate(sorted(m.TRAIN_FUNCTIONS,key=lambda f:f['fid'])):
  f=copy.deepcopy(f0);seed=78000000+100*i;r.set_seed(seed);m.gen_train_offset(10,f)
  pop=10*torch.rand(4,100,10,generator=torch.Generator().manual_seed(seed+10000000))-5
  tasks.append(dict(fid=f['fid'],seed=seed,params=f['params'],pop=pop))
 torch.save(tasks,OUT/'tasks.pt');save(OUT/'protocol.json',dict(script=sha(__file__),protocol=sha(ROOT/'docs/experiments/SCORE_CONSISTENCY_PROTOCOL.md'),tasks=sha(OUT/'tasks.pt'),checkpoints={p.name:sha(p) for p in (ROOT/'checkpoints').glob('*.pt')}))
def run(q,variant,diagnostic):
 torch.set_num_threads(1);r.DEVICE=torch.device('cpu');m=source();f=copy.deepcopy(next(f for f in m.TRAIN_FUNCTIONS if f['fid']==q['fid']));f['params']=q['params'];task=Task(f,m);opt=build(variant)
 native=opt._select_by_acquisition;cache={};rows=[];teacher=0;attrs=['last_acquisition','last_surrogate_mu','last_surrogate_sigma','last_router_prob']
 def hook(*args,**kw):
  idx=native(*args,**kw)
  if not diagnostic:return idx
  saved={k:getattr(opt,k) for k in attrs};score=opt.last_acquisition;mu=opt.last_surrogate_mu;sigma=opt.last_surrogate_sigma
  with torch.random.fork_rng(devices=[]):
   k=dict(kw);k['use_learned_router']=False
   ROOPFOptimizer._select_by_acquisition(opt,*args,**k);base=opt.last_acquisition
  for k,v in saved.items():setattr(opt,k,v)
  coefficient=.06 if 'local_op_boost' in opt.ablation and _is_local_proxy_friendly_problem(args[4]) else .03
  prior=-coefficient*torch.log(args[3].clamp_min(1e-8))
  uncertainty=-torch.exp(opt.log_kappa).clamp(.15,4.)*sigma
  if 'mean_only' in opt.ablation or 'no_uncertainty' in opt.ablation:uncertainty=torch.zeros_like(sigma)
  elif 'weak_uncertainty' in opt.ablation:uncertainty=.25*uncertainty
  parts=dict(mu=mu,uncertainty=uncertainty,prior=prior,other=base-mu-uncertainty-prior,residual=score-base,score=score)
  assert torch.allclose(sum(parts[k] for k in ['mu','uncertainty','prior','other','residual']),score,rtol=1e-5,atol=1e-5)
  cache[args[2].shape[1]]=dict(parts=parts,args=args,kw=kw,idx=idx,prior=args[3])
  return idx
 opt._select_by_acquisition=hook
 def observe(s):
  nonlocal teacher
  pool=cache[36];comb=cache[3];idx=s['selected'];saved={k:getattr(opt,k) for k in attrs}
  with torch.random.fork_rng(devices=[]):
   truth=task.calfitness(s['combined']);teacher+=12
   args=list(comb['args']);args[3]=torch.cat([torch.ones((4,1)),pool['prior'].gather(1,idx)],1)
   try:
    ROOPFOptimizer._select_by_acquisition(opt,*args,**comb['kw'])
    chosen=opt.protected_index(opt.last_acquisition,opt.last_router_prob,s['archive_y'].std(1,keepdim=True,unbiased=False).clamp_min(1e-8),s['portfolio_seen'],s['portfolio_best_count'],variant=='full')
   finally:
    for k,v in saved.items():setattr(opt,k,v)
  for b in range(4):
   z=dict(fid=q['fid'],variant=variant,batch=b,eval_before=opt.evalnum,anchor=float(truth[b,0]),first=float(truth[b,1]),second=float(truth[b,2]),native_choice=int(s['extra_idx'][b,0]),prior_choice=int(chosen[b,0]),native_truth=float(truth[b,s['extra_idx'][b,0]]),prior_truth=float(truth[b,chosen[b,0]]),scale=float(truth[b].std(unbiased=False).clamp_min(1e-8)))
   for key in ['mu','uncertainty','prior','other','residual','score']:
    for j in range(2):
     z[f'pool_{key}_{j}']=float(pool['parts'][key][b,idx[b,j]])
     z[f'combined_{key}_{j}']=float(comb['parts'][key][b,j+1])
   rows.append(z)
 opt.supplement_gate_observer=observe if diagnostic else None;r.set_seed(q['seed']+20000000)
 with torch.no_grad():_,trail,nfe,points=opt(q['pop'].clone(),task)
 assert nfe==300 and task.points==1200+teacher
 return trail,points,torch.get_rng_state(),rows,teacher

def worker(i,v):
 p=OUT/f'{i:02d}_{v}.json'
 if p.exists():return json.loads(p.read_text())
 q=torch.load(OUT/'tasks.pt',weights_only=False)[i];a=run(q,v,False);b=run(q,v,True)
 assert all(torch.equal(a[k],b[k]) for k in [0,1,2])
 pd.DataFrame(b[3]).to_csv(OUT/f'{i:02d}_{v}.csv',index=False);np.savez_compressed(OUT/f'{i:02d}_{v}.npz',trail=b[0].numpy(),points=b[1].numpy())
 z=dict(fid=q['fid'],variant=v,teacher=b[4],parity=True);save(p,z);return z

def finish(results):
 d=pd.concat([pd.read_csv(p) for p in sorted(OUT.glob('[0-9][0-9]_*.csv'))],ignore_index=True);assert len(d)==12960;stats={}
 for v,g in d.groupby('variant'):
  gap=g.combined_score_1-g.combined_score_0;flip=gap<0;change=g.prior_choice!=g.native_choice;diff=g.prior_truth-g.native_truth
  z=dict(states=len(g),order_reversed=int(flip.sum()),reversal_true_wtl=[int((g[flip]['second']<g[flip]['first']).sum()),int((g[flip]['second']==g[flip]['first']).sum()),int((g[flip]['second']>g[flip]['first']).sum())],prior_preserving_changed=int(change.sum()),prior_preserving_true_wtl=[int((diff[change]<0).sum()),int((diff[change]==0).sum()),int((diff[change]>0).sum())])
  for key in ['mu','uncertainty','prior','other','residual','score']:
   delta0=g[f'combined_{key}_0']-g[f'pool_{key}_0'];delta1=g[f'combined_{key}_1']-g[f'pool_{key}_1']
   z[key+'_mean_abs_change']=float(pd.concat([delta0.abs(),delta1.abs()]).mean())
   # Matched-state one-component rollback, purely an attribution diagnostic.
   reverted=gap-(delta1-delta0)
   z[key+'_reversals_removed_if_rolled_back']=int((flip&(reverted>=0)).sum())
  stats[v]=z
 p=json.loads((OUT/'protocol.json').read_text());assert p['script']==sha(__file__)
 for name,h in p['checkpoints'].items():assert sha(ROOT/'checkpoints'/name)==h
 report=dict(stats=stats,budget=dict(main_trajectories=576,main_points=172800,teacher_points=sum(z['teacher'] for z in results)),scope='Own-policy states; score decomposition and conditional prior-preserving choices, not full-policy results')
 save(OUT/'REPORT.json',report)
 lines=['# 两轮排序评分一致性诊断','','所有36×2配置均通过诊断开关点、轨迹与RNG一致性检查。组件回退为同状态数值诊断，各组件影响可能重叠，不能相加归因。','','| 配置 | 状态数 | 短名单内顺序反转 | 反转后真实胜/平/负 | 保留prior后改选 | 改选真实胜/平/负 |','|---|---:|---:|---|---:|---|']
 for v,z in stats.items():lines.append(f"| {v} | {z['states']} | {z['order_reversed']} | {'/'.join(map(str,z['reversal_true_wtl']))} | {z['prior_preserving_changed']} | {'/'.join(map(str,z['prior_preserving_true_wtl']))} |")
 lines+=['','## 组件变化','','平均绝对变化为原始评分单位，不能跨函数解释为统一效果。回退数表示只恢复该项的池内值时，多少次原反转不再发生。','','| 配置 | 组件 | 平均绝对变化 | 回退后消除的反转数 |','|---|---|---:|---:|']
 for v,z in stats.items():
  for key in ['mu','uncertainty','prior','other','residual']:lines.append(f"| {v} | {key} | {z[key+'_mean_abs_change']:.6g} | {z[key+'_reversals_removed_if_rolled_back']} |")
 lines+=['','other包含结构、guard和operator修正；residual为开启与关闭learned router的分差。候选顺序变化不自动等于错误；真值胜负为第二个候选相对第一个候选。','','```json',json.dumps(report['budget'],indent=2),'```']
 (ROOT/'docs/experiments/SCORE_CONSISTENCY_RESULTS.zh-CN.md').write_text('\n'.join(lines)+'\n')
 dest=ROOT/'artifacts/score_consistency_audit';dest.mkdir(exist_ok=True);fs=sorted(p for p in OUT.iterdir() if p.is_file());zp=dest/'data.zip'
 with zipfile.ZipFile(zp,'w',zipfile.ZIP_DEFLATED) as z:
  for p in fs:z.write(p,p.name)
 hs={p.name:sha(p) for p in fs}
 with zipfile.ZipFile(zp) as z:
  assert z.testzip() is None
  for name,h in hs.items():assert hashlib.sha256(z.read(name)).hexdigest()==h
 save(dest/'manifest.json',dict(file=zp.name,sha256=sha(zp),members=hs));save(dest/'REPORT.json',report)
 (dest/'README.md').write_text('Restore data.zip into results/score_consistency_audit; verify manifest hashes. See SCORE_CONSISTENCY_PROTOCOL and RESULTS in docs/experiments.\n')
 save(ROOT/'docs/experiments/SCORE_CONSISTENCY_INTEGRITY.json',dict(passed=True,cases=72,archived_files=len(fs),checkpoints_unchanged=True))
def main():
 if not (OUT/'protocol.json').exists():setup()
 assert json.loads((OUT/'protocol.json').read_text())['script']==sha(__file__)
 results=[]
 with ProcessPoolExecutor(8,mp_context=multiprocessing.get_context('spawn')) as ex:
  fs=[ex.submit(worker,i,v) for i in range(36) for v in ['full','no_residual']]
  for j,f in enumerate(as_completed(fs)):results.append(f.result());print(j+1,'/72',flush=True)
 finish(results);print('SCORE AUDIT COMPLETE',flush=True)
if __name__=='__main__':main()
