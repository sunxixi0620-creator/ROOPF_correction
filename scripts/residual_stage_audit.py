import os
os.environ.setdefault('OMP_NUM_THREADS','1')
import copy,json,multiprocessing,zipfile,hashlib
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import pandas as pd
import torch
from training_validation import ROOT,Task,source,sha
from supplementary_experiments import build
from residual_target_study import save,net
from residual_top_study import path,NAMES
from roopf.model import ROOPFOptimizer
import run_roopf as r
OUT=ROOT/'results/residual_stage_audit'
def setup():
 OUT.mkdir(exist_ok=True);m=source();m.DEVICE=torch.device('cpu');qs=[]
 for i,f0 in enumerate(sorted(m.TRAIN_FUNCTIONS,key=lambda f:f['fid'])):
  f=copy.deepcopy(f0);seed=77000000+100*i;r.set_seed(seed);m.gen_train_offset(10,f)
  pop=10*torch.rand(4,100,10,generator=torch.Generator().manual_seed(seed+10000000))-5
  qs.append(dict(fid=f['fid'],seed=seed,params=f['params'],pop=pop))
 torch.save(qs,OUT/'tasks.pt');save(OUT/'protocol.json',dict(script=sha(__file__),protocol=sha(ROOT/'docs/experiments/RESIDUAL_STAGE_PROTOCOL.md'),tasks=sha(OUT/'tasks.pt'),checkpoints={n:sha(path(n)) for n in NAMES if n!='no_residual'},anchor=sha(ROOT/'checkpoints/anchor_policy_d10.pt')))
def run(q,observe):
 torch.set_num_threads(1);r.DEVICE=torch.device('cpu');m=source();f=copy.deepcopy(next(f for f in m.TRAIN_FUNCTIONS if f['fid']==q['fid']));f['params']=q['params'];task=Task(f,m)
 opt=build('no_residual');fullflags=build('full').ablation.copy();models={}
 for n in NAMES:
  if n=='no_residual':continue
  p=torch.load(path(n),weights_only=False,map_location='cpu');model=net().eval();model.load_state_dict({k.removeprefix('net.'):v for k,v in p['state_dict'].items()})
  models[n]=(model,torch.tensor(p['mean']),torch.tensor(p['std']))
 rows=[];teacher=0
 def observer(s):
  nonlocal teacher
  attrs=['last_acquisition','last_surrogate_mu','last_surrogate_sigma','last_router_prob','router_model','router_mean','router_std','ablation']
  saved={k:getattr(opt,k) for k in attrs};args,kwargs=opt.pool_context;pool=args[2];scale=s['archive_y'].std(1,keepdim=True,unbiased=False).clamp_min(1e-8)
  with torch.random.fork_rng(devices=[]):
   values=task.calfitness(torch.cat([s['baseline_cand'][:,1:2],pool],1));teacher+=148
   assert torch.isfinite(values).all()
   norm=values.std(1,unbiased=False).clamp_min(1e-8)
   try:
    for name in NAMES:
     opt.ablation=saved['ablation'] if name=='no_residual' else fullflags
     if name!='no_residual':opt.router_model,opt.router_mean,opt.router_std=models[name]
     idx=ROOPFOptimizer._select_by_acquisition(opt,*args,**kwargs)
     shortlist=pool.gather(1,idx.unsqueeze(-1).expand(-1,-1,10))
     combined=torch.cat([s['baseline_cand'][:,1:2],shortlist],1)
     ops=torch.cat([torch.full_like(idx[:,:1],-2),s['op_ids'][idx]],1)
     raw=ROOPFOptimizer._select_by_acquisition(opt,s['archive_x'],s['archive_y'],combined,torch.ones(combined.shape[:2]),s['problem'],s['remaining'],s['stagnation'],1,candidate_ops=ops,state_raw=s['state_raw'],fitness=s['fitness'],used=s['used'],use_learned_router=True,disable_online_proxy=None)
     chosen=opt.protected_index(opt.last_acquisition,opt.last_router_prob,scale,s['portfolio_seen'],s['portfolio_best_count'],name!='no_residual')
     point=combined.gather(1,chosen.unsqueeze(-1).expand(-1,-1,10))
     if name=='no_residual':
      actual=s['combined'].gather(1,s['extra_idx'].unsqueeze(-1).expand(-1,-1,10));assert torch.equal(point,actual)
     cf=torch.cat([values[:,:1],values[:,1:].gather(1,idx)],1)
     for b in range(4):
      a=float(values[b,0]);oracle=float(values[b].min());short=float(cf[b].min());pre=float(cf[b,raw[b,0]]);post=float(cf[b,chosen[b,0]])
      rows.append(dict(fid=q['fid'],batch=b,eval_before=opt.evalnum,model=name,anchor=a,pool_best=float(values[b,1:].min()),oracle=oracle,short_best=short,pre=pre,post=post,normalizer=float(norm[b]),shortlist_loss=short-oracle,rerank_loss=pre-short,gate_effect=post-pre,regret=post-oracle,short1=int(idx[b,0]),short2=int(idx[b,1]),raw_index=int(raw[b,0]),chosen_index=int(chosen[b,0]),incumbent=float(s['fitness'][b,0])))
   finally:
    for k,v in saved.items():setattr(opt,k,v)
 opt.supplement_gate_observer=observer if observe else None;r.set_seed(q['seed']+20000000)
 with torch.no_grad():_,trail,nfe,points=opt(q['pop'].clone(),task)
 assert nfe==300 and task.points==1200+teacher
 assert torch.isfinite(trail).all() and (trail[:,1:]<=trail[:,:-1]).all()
 return trail,points,torch.get_rng_state(),rows,teacher

def worker(i):
 p=OUT/f'case_{i:02d}.json'
 if p.exists():return json.loads(p.read_text())
 q=torch.load(OUT/'tasks.pt',weights_only=False)[i];a=run(q,False);b=run(q,True)
 assert all(torch.equal(a[k],b[k]) for k in [0,1,2]),'observer changed factual behavior'
 pd.DataFrame(b[3]).to_csv(OUT/f'case_{i:02d}.csv',index=False)
 np.savez_compressed(OUT/f'case_{i:02d}.npz',trail=b[0].numpy(),points=b[1].numpy())
 z=dict(fid=q['fid'],rows=len(b[3]),teacher_points=b[4],exact_parity=True);save(p,z);return z

def finish(results):
 d=pd.concat([pd.read_csv(OUT/f'case_{i:02d}.csv') for i in range(36)],ignore_index=True)
 assert len(d)==51840 and not d.duplicated(['fid','batch','eval_before','model']).any()
 assert np.allclose(d.shortlist_loss+d.rerank_loss+d.gate_effect,d.regret,rtol=1e-8,atol=1e-8)
 assert (d[['shortlist_loss','rerank_loss','regret']]>=-1e-10).all().all()
 stats={}
 for name,g in d.groupby('model'):
  blocked=(g.raw_index>0)&(g.chosen_index==0)
  stats[name]=dict(states=len(g),pool_has_better=int((g.pool_best<g.anchor).sum()),shortlist_has_better=int((g.short_best<g.anchor).sum()),shortlist_contains_oracle=int((g.short_best==g.oracle).sum()),good_proposal_blocked=int((blocked&(g.pre<g.anchor)).sum()),bad_proposal_prevented=int((blocked&(g.pre>g.anchor)).sum()),accepted=int((g.chosen_index>0).sum()),post_beats_anchor=int((g.post<g.anchor).sum()),post_improves_incumbent=int((g.post<g.incumbent-1e-12).sum()),**{'mean_normalized_'+c:float((g[c]/g.normalizer).mean()) for c in ['shortlist_loss','rerank_loss','gate_effect','regret']})
 protocol=json.loads((OUT/'protocol.json').read_text());assert protocol['script']==sha(__file__) and protocol['tasks']==sha(OUT/'tasks.pt')
 assert protocol['anchor']==sha(ROOT/'checkpoints/anchor_policy_d10.pt')
 for n,h in protocol['checkpoints'].items():assert sha(path(n))==h
 report=dict(stats=stats,budget=dict(main_trajectories=288,main_points=86400,teacher_points=sum(z['teacher_points'] for z in results),shared_states=6480,model_state_evaluations=51840),scope='Shared no-residual behavior states; local stage regret, not final optimization outcomes')
 save(OUT/'REPORT.json',report)
 lines=['# 相同状态下的候选选择分阶段诊断','',report['scope'],'','36个新生成实例，每实例4条轨迹。所有8种评分配置在完全相同的6480个后期状态上比较；不训练或调参。','','## 即时误差分解','','以下均为真值差除以该状态37候选真值标准差后的均值。总误差=短名单损失+二次排序损失+门控影响。门控影响可为负；不等同于最终搜索贡献。','','| 模型 | 短名单损失 | 二次排序损失 | 门控影响 | 总选择误差 |','|---|---:|---:|---:|---:|']
 for n,z in stats.items():lines.append('| '+n+' | '+' | '.join(f"{z['mean_normalized_'+c]:.6f}" for c in ['shortlist_loss','rerank_loss','gate_effect','regret'])+' |')
 lines+=['','## 候选保留与门控','','| 模型 | 池中存在优于anchor | 短名单仍有优于anchor | 短名单保留oracle | 拒绝好提案 | 阻止坏提案 | 最终优于anchor |','|---|---:|---:|---:|---:|---:|---:|']
 for n,z in stats.items():lines.append('| '+n+' | '+' | '.join(str(z[k]) for k in ['pool_has_better','shortlist_has_better','shortlist_contains_oracle','good_proposal_blocked','bad_proposal_prevented','post_beats_anchor'])+' |')
 lines+=['','oracle包含第二anchor与36个pool候选。好/坏提案按相对第二anchor的真实目标值判定。所有反事实配置共享无residual策略的历史与成功计数；不能据此断言它们完整部署时也有相同分布。教师真值仅用于观察，每个实例均通过点、轨迹与RNG精确一致性检查。','','```json',json.dumps(report['budget'],indent=2),'```']
 (ROOT/'docs/experiments/RESIDUAL_STAGE_RESULTS.zh-CN.md').write_text('\n'.join(lines)+'\n')
 dest=ROOT/'artifacts/residual_stage_audit';dest.mkdir(exist_ok=True);files=sorted(p for p in OUT.iterdir() if p.is_file());zp=dest/'data.zip'
 with zipfile.ZipFile(zp,'w',zipfile.ZIP_DEFLATED) as z:
  for p in files:z.write(p,p.name)
 hashes={p.name:sha(p) for p in files}
 with zipfile.ZipFile(zp) as z:
  assert z.testzip() is None
  for n,h in hashes.items():assert hashlib.sha256(z.read(n)).hexdigest()==h
 save(dest/'manifest.json',dict(file=zp.name,sha256=sha(zp),members=hashes));save(dest/'REPORT.json',report)
 (dest/'README.md').write_text('Restore data.zip into results/residual_stage_audit. Verify manifest SHA256. Frozen checkpoints come from residual_top_study. See docs/experiments/RESIDUAL_STAGE_PROTOCOL.md.\n')
 save(ROOT/'docs/experiments/RESIDUAL_STAGE_INTEGRITY.json',dict(passed=True,all36_cases_exact_parity=True,archived_files=len(files),checkpoints_unchanged=True))
def main():
 if not (OUT/'protocol.json').exists():setup()
 assert json.loads((OUT/'protocol.json').read_text())['script']==sha(__file__)
 results=[]
 with ProcessPoolExecutor(8,mp_context=multiprocessing.get_context('spawn')) as ex:
  fs=[ex.submit(worker,i) for i in range(36)]
  for j,f in enumerate(as_completed(fs)):results.append(f.result());print(j+1,'/36',flush=True)
 finish(results);print('STAGE AUDIT COMPLETE',flush=True)
if __name__=='__main__':main()
