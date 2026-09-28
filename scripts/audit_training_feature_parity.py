"""Bounded feature-contract probe. No training, tuning or model changes."""
import copy,json,sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from training_validation import ROOT,Task,source,sha
from residual_target_study import OUT as DATA,execute,save
from supplementary_experiments import build
import run_roopf as r
OUT=ROOT/'docs/revision/feature_contract_probe'

def main():
 OUT.mkdir(exist_ok=True);torch.set_num_threads(1);r.DEVICE=torch.device('cpu')
 tasks=torch.load(DATA/'tasks.pt',map_location='cpu',weights_only=False);allrows=[];report=[]
 for index in [0,24,48]:
  q=tasks[index];assert q['split']=='train'
  m=source();f=copy.deepcopy(next(f for f in m.TRAIN_FUNCTIONS if f['fid']==q['fid']));f['params']=q['params'];task=Task(f,m);opt=build('no_residual');opt.log_pool_candidates=True
  native=opt._select_by_acquisition;cache={};captured=[]
  def hook(*args,**kw):
   idx=native(*args,**kw);size=args[2].shape[1];score=opt.last_acquisition.clone()
   if size==36:cache['pool']=score;cache['idx']=idx.clone()
   elif size==38:
    cache['log']=score
    diff=(score[:,2:]-cache['pool']).detach().numpy().ravel()
    for d in diff:captured.append(dict(stage='pool',eval_before=opt.evalnum,difference=float(d)))
   elif size==3:
    logged=torch.cat([cache['log'][:,1:2],cache['log'][:,2:].gather(1,cache['idx'])],1)
    for d in (logged-score).detach().numpy().ravel():captured.append(dict(stage='combined',eval_before=opt.evalnum,difference=float(d)))
   return idx
  opt._select_by_acquisition=hook;r.set_seed(q['seed']+20000000)
  with torch.no_grad():_,trail,nfe,points=opt(q['pop'].clone(),task)
  rng=torch.get_rng_state();normal=execute(q,True)
  assert torch.equal(trail,normal[0]) and torch.equal(points,normal[1]) and torch.equal(rng,normal[3])
  assert nfe==300 and task.points==16400
  log=pd.DataFrame(opt.pool_candidate_samples)
  names=torch.load(ROOT/'checkpoints/residual_selector_generated36_d10.pt',map_location='cpu',weights_only=False)['feature_names']
  saved=np.load(DATA/f'case_{index:03d}.npz')['x'][:,names.index('surrogate_score')]
  observed=pd.to_numeric(log.surrogate_score).to_numpy(dtype=np.float32)
  historical_exact=bool(np.array_equal(saved,observed));historical_max=float(np.nanmax(np.abs(saved-observed)));historical_trail=bool(np.array_equal(np.load(DATA/f'case_{index:03d}.npz')['trail'],trail.numpy()))
  print('HISTORICAL',index,historical_exact,historical_max,historical_trail,flush=True)
  df=pd.DataFrame(captured)
  for stage,g in df.groupby('stage'):
   a=g.difference.abs();row=dict(task_index=index,fid=q['fid'],stage=stage,n=len(g),fraction_abs_difference_gt_1e_6=float((a>1e-6).mean()),mean_abs_difference=float(a.mean()),p95_abs_difference=float(a.quantile(.95)),max_abs_difference=float(a.max()),stored_training_score_exact=historical_exact,historical_max_score_difference=historical_max,historical_trail_exact=historical_trail,observation_parity_exact=True);report.append(row)
  df.insert(0,'fid',q['fid']);allrows+=df.to_dict('records')
 pd.DataFrame(report).to_csv(OUT/'summary.csv',index=False)
 pd.DataFrame(allrows).to_csv(OUT/'differences.csv.gz',index=False,compression='gzip')
 save(OUT/'manifest.json',dict(source=sha(__file__),tasks_sha256=sha(DATA/'tasks.pt'),cases=[0,24,48],scope='Three preselected training cases, same states; establishes feature-definition mismatch, not its causal performance impact',main_points=7200,teacher_points=91200,checkpoints={p.name:sha(p) for p in (ROOT/'checkpoints').glob('*.pt')}))
 print(pd.DataFrame(report).to_string(index=False))
if __name__=='__main__':main()
