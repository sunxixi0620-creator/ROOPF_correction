import os
os.environ.setdefault('OMP_NUM_THREADS','1')
import copy,json,multiprocessing,time
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import pandas as pd
import torch
from training_validation import ROOT,Task,source,sha
from supplementary_experiments import build
from residual_target_study import save,OUT as PREV
from roopf.supplement import RESIDUAL_FLAGS
import run_roopf as r
OUT=ROOT/'results/residual_decision_audit'
NAMES=['original']+[f'{t}_{s}' for t in ['incumbent','second'] for s in [20260928,20260929]]

def setup():
 OUT.mkdir(exist_ok=True);m=source();m.DEVICE=torch.device('cpu');tasks=[]
 for i,f0 in enumerate(sorted(m.TRAIN_FUNCTIONS,key=lambda f:f['fid'])):
  f=copy.deepcopy(f0);seed=74000000+100*i;r.set_seed(seed);m.gen_train_offset(10,f)
  pop=10*torch.rand(4,100,10,generator=torch.Generator().manual_seed(seed+10000000))-5
  tasks.append(dict(fid=f['fid'],seed=seed,params=f['params'],pop=pop))
 torch.save(tasks,OUT/'tasks.pt')
 save(OUT/'protocol.json',dict(script=sha(__file__),protocol=sha(ROOT/'docs/experiments/RESIDUAL_DECISION_PROTOCOL.md'),tasks=sha(OUT/'tasks.pt'),checkpoints={n:sha(path(n)) for n in NAMES},anchor=sha(ROOT/'checkpoints/anchor_policy_d10.pt')))
def path(n):return ROOT/'checkpoints/residual_selector_generated36_d10.pt' if n=='original' else PREV/(n+'.pt')
def run(q,name,diagnostic=False,disable_round=None):
 torch.set_num_threads(1);r.DEVICE=torch.device('cpu');m=source();f=copy.deepcopy(next(f for f in m.TRAIN_FUNCTIONS if f['fid']==q['fid']));f['params']=q['params'];task=Task(f,m);opt=build('full')
 p=torch.load(path(name),map_location='cpu',weights_only=False);opt.router_model.load_state_dict({k.removeprefix('net.'):v for k,v in p['state_dict'].items()});opt.router_mean.copy_(torch.tensor(p['mean']));opt.router_std.copy_(torch.tensor(p['std']))
 if disable_round is not None:
  original=opt._select_by_acquisition
  def select(*args,**kwargs):
   flags=opt.ablation.copy()
   try:
    if opt.evalnum==disable_round:opt.ablation.difference_update(RESIDUAL_FLAGS)
    return original(*args,**kwargs)
   finally:opt.ablation=flags
  opt._select_by_acquisition=select
 records=[];expected={};teacher=0
 def observe(s):
  nonlocal teacher
  chosen=s['combined'].gather(1,s['extra_idx'].unsqueeze(-1).expand(-1,-1,10))
  other=opt.without_residual_decision(s) if (opt.pool_router_active or s['combined_router_prob'] is not None) else chosen
  changed=(chosen!=other).any(2).view(-1)
  if changed.any():expected[opt.evalnum]=other.detach().clone()
  with torch.random.fork_rng(devices=[]):
   before=task.points;values=task.calfitness(torch.cat([s['combined'][:,:1],chosen,other],1));teacher+=task.points-before
  scale=s['archive_y'].std(1,unbiased=False).clamp_min(1e-8)
  for b in range(4):
   records.append(dict(fid=q['fid'],model=name,batch=b,eval_before=opt.evalnum,changed=int(changed[b]),accepted=int(s['extra_idx'][b,0]>=1),anchor=float(values[b,0]),selected=float(values[b,1]),counterfactual=float(values[b,2]),incumbent=float(s['fitness'][b,0]),scale=float(scale[b])))
 opt.supplement_gate_observer=observe if diagnostic else None
 r.set_seed(q['seed']+20000000)
 with torch.no_grad():_,trail,nfe,points=opt(q['pop'].clone(),task)
 assert nfe==300 and task.points==1200+teacher
 assert torch.isfinite(trail).all() and (trail[:,1:]<=trail[:,:-1]).all()
 return dict(trail=trail,points=points,rng=torch.get_rng_state(),records=records,expected=expected,teacher=teacher)
def worker(i,name):
 q=torch.load(OUT/'tasks.pt',weights_only=False)[i];dest=OUT/f'{i:02d}_{name}'
 if dest.with_suffix('.json').exists():return json.loads(dest.with_suffix('.json').read_text())
 a=run(q,name,True);pd.DataFrame(a['records']).to_csv(str(dest)+'_decisions.csv',index=False)
 epoch=min(a['expected']) if a['expected'] else None;rows=[];arrays={'factual_trail':a['trail'].numpy(),'factual_points':a['points'].numpy()}
 if epoch is not None:
  b=run(q,name,False,epoch);k=(epoch-100)//2
  assert torch.equal(a['points'][:,:2*k],b['points'][:,:2*k]);assert torch.equal(a['trail'][:,:k],b['trail'][:,:k])
  assert torch.equal(b['points'][:,2*k+1:2*k+2],a['expected'][epoch]),'replay not the counterfactual decision'
  arrays.update(replay_trail=b['trail'].numpy(),replay_points=b['points'].numpy())
  for j in range(4):rows.append(dict(fid=q['fid'],model=name,batch=j,disabled_round=epoch,factual=float(a['trail'][j,-1]),replay=float(b['trail'][j,-1])))
 np.savez_compressed(dest.with_suffix('.npz'),**arrays)
 result=dict(fid=q['fid'],model=name,teacher_points=a['teacher'],decision_rows=len(a['records']),replay=epoch is not None,rows=rows)
 save(dest.with_suffix('.json'),result);return result

def main():
 if not (OUT/'protocol.json').exists():setup()
 assert json.loads((OUT/'protocol.json').read_text())['script']==sha(__file__)
 q=torch.load(OUT/'tasks.pt',weights_only=False)[0]
 if not (OUT/'PARITY.json').exists():
  for name in NAMES:
   a=run(q,name);b=run(q,name,True);assert all(torch.equal(a[k],b[k]) for k in ['trail','points','rng'])
  save(OUT/'PARITY.json',dict(models=NAMES,exact_points_trail_rng=True))
 results=[]
 with ProcessPoolExecutor(8,mp_context=multiprocessing.get_context('spawn')) as ex:
  fs=[ex.submit(worker,i,n) for i in range(36) for n in NAMES]
  for j,f in enumerate(as_completed(fs)):
   results.append(f.result());print(j+1,'/180',flush=True)
 save(OUT/'COMPLETE',dict(cases=180,teacher_points=sum(z['teacher_points'] for z in results),factual_trajectories=720,replay_trajectories=4*sum(z['replay'] for z in results)))
if __name__=='__main__':main()
