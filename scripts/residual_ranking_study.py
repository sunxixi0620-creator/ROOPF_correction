import os
os.environ.setdefault('OMP_NUM_THREADS','1')
import copy,json,time,multiprocessing
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import pandas as pd
import torch
from torch import nn
from residual_target_study import ROOT,OUT as PREV,SEEDS,net,payload,execute,save,sha,source,Task
import run_roopf as r
OUT=ROOT/'results/residual_ranking_study'
NAMES=['original','no_residual']+[f'{t}_{s}' for t in ['second','ranking','magnitude'] for s in SEEDS]
def path(n):
 if n=='original':return ROOT/'checkpoints/residual_selector_generated36_d10.pt'
 return (PREV if n.startswith('second') else OUT)/(n+'.pt')
def load(split):
 tasks=torch.load(PREV/'tasks.pt',weights_only=False);ds=[np.load(PREV/f'case_{i:03d}.npz') for i,q in enumerate(tasks) if q['split']==split]
 return np.concatenate([d['x'].reshape(-1,38,d['x'].shape[-1]) for d in ds]),np.concatenate([d['fit'] for d in ds]).astype(np.float32)
def setup():
 OUT.mkdir(exist_ok=True)
 tasks=torch.load(PREV/'tasks.pt',weights_only=False)
 save(OUT/'protocol.json',dict(script=sha(__file__),protocol=sha(ROOT/'docs/experiments/RESIDUAL_RANKING_PROTOCOL.md'),data={f'case_{i:03d}.npz':sha(PREV/f'case_{i:03d}.npz') for i,q in enumerate(tasks) if q['split'] in ['train','validation']},checkpoints={n:sha(path(n)) for n in ['original']+[f'second_{s}' for s in SEEDS]},anchor=sha(ROOT/'checkpoints/anchor_policy_d10.pt')))
def train():
 tx,tf=load('train');vx,vf=load('validation');mean=tx.reshape(-1,tx.shape[-1]).mean(0);std=np.maximum(tx.reshape(-1,tx.shape[-1]).std(0),1e-6)
 dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu');torch.set_num_threads(4)
 x=torch.tensor((tx-mean)/std,device=dev);v=torch.tensor((vx-mean)/std,device=dev)
 y=torch.tensor(tf,device=dev);vy=torch.tensor(vf,device=dev);ij=torch.triu_indices(38,38,1,device=dev)
 def loss(model,xx,yy,objective):
  logits=model(xx.reshape(-1,xx.shape[-1])).reshape(-1,38)
  if objective=='ranking':
   sign=torch.sign(yy[:,ij[1]]-yy[:,ij[0]]);valid=sign!=0
   terms=nn.functional.softplus(-sign*(logits[:,ij[0]]-logits[:,ij[1]]))
   return (terms*valid).sum()/valid.sum().clamp_min(1)
  gain=(yy[:,1:2]-yy)/yy.std(1,unbiased=False,keepdim=True).clamp_min(1e-8);target=.5+.5*gain/(1+gain.abs())
  return nn.functional.mse_loss(torch.sigmoid(logits),target)
 for objective in ['ranking','magnitude']:
  for seed in SEEDS:
   tag=f'{objective}_{seed}';r.set_seed(seed);model=net().to(dev);optim=torch.optim.Adam(model.parameters(),lr=.001);best=float('inf');stale=0;hist=[]
   for epoch in range(1,41):
    model.train();total=0.;perm=torch.randperm(len(x),device=dev)
    for ids in perm.split(128):
     optim.zero_grad();l=loss(model,x[ids],y[ids],objective);assert torch.isfinite(l);l.backward();optim.step();total+=float(l)*len(ids)
    model.eval()
    with torch.no_grad():val=sum(float(loss(model,xx,yy,objective))*len(xx) for xx,yy in zip(v.split(128),vy.split(128)))/len(v)
    hist.append(dict(epoch=epoch,training_loss=total/len(x),validation_loss=val))
    if val<best-1e-5:
     best=val;stale=0;p=payload();p.update(state_dict={'net.'+k:z.detach().cpu() for k,z in model.state_dict().items()},mean=mean.tolist(),std=std.tolist(),target=objective,epoch=epoch,seed=seed);torch.save(p,OUT/(tag+'.pt'))
    else:stale+=1
    save(OUT/(tag+'_history.json'),hist);print('TRAIN',tag,epoch,val,flush=True)
    if stale>=6:break
 save(OUT/'SELECTION_FROZEN.json',{f'{t}_{s}':sha(path(f'{t}_{s}')) for t in ['ranking','magnitude'] for s in SEEDS})
def tasks():
 if (OUT/'tasks.pt').exists():return
 m=source();m.DEVICE=torch.device('cpu');qs=[]
 for i,f0 in enumerate(sorted(m.TRAIN_FUNCTIONS,key=lambda f:f['fid'])):
  f=copy.deepcopy(f0);seed=75000000+100*i;r.set_seed(seed);m.gen_train_offset(10,f)
  pop=10*torch.rand(4,100,10,generator=torch.Generator().manual_seed(seed+10000000))-5
  qs.append(dict(fid=f['fid'],seed=seed,params=f['params'],pop=pop))
 torch.save(qs,OUT/'tasks.pt')
def worker(i):
 q=torch.load(OUT/'tasks.pt',weights_only=False)[i];p=OUT/f'case_{i:02d}.json'
 if p.exists():return json.loads(p.read_text())
 # Teacher log and normal no-residual must yield identical search trajectories.
 teacher=execute(q,True);plain=execute(q,False)
 assert all(torch.equal(teacher[k],plain[k]) for k in [0,1,3])
 d=pd.DataFrame(teacher[2]);fit=d.candidate_fit.to_numpy().reshape(-1,38);cols=[]
 for name in payload()['feature_names']:
  if name.startswith('op_'):col=(d.op_name==name[3:]).to_numpy(dtype=np.float32)
  else:
   raw=pd.to_numeric(d[name.removesuffix('_missing')],errors='coerce').to_numpy(dtype=np.float32)
   col=(~np.isfinite(raw)).astype(np.float32) if name.endswith('_missing') else np.nan_to_num(raw,nan=0.,posinf=0.,neginf=0.)
  cols.append(col)
 np.savez_compressed(OUT/f'case_{i:02d}_teacher.npz',x=np.stack(cols,1),fit=fit,eval_before=d.eval_before.to_numpy())
 m=source();f=copy.deepcopy(next(f for f in m.TRAIN_FUNCTIONS if f['fid']==q['fid']));f['params']=q['params'];initial=Task(f,m).calfitness(q['pop']);best=initial.min(1).values;std=initial.std(1).clamp_min(1e-8);rows=[]
 for name in NAMES:
  trail=plain[0] if name=='no_residual' else execute(q,False,path(name),'full')[0]
  gain=((best-trail[:,-1])/std).clamp_min(0)
  for j in range(4):rows.append(dict(method=name,fid=q['fid'],seed_index=j,final=float(trail[j,-1]),bounded_gain=float(gain[j]/(1+gain[j])),nfe=300))
 save(p,rows);return rows

def main():
 if not (OUT/'protocol.json').exists():setup()
 assert json.loads((OUT/'protocol.json').read_text())['script']==sha(__file__)
 if not (OUT/'SELECTION_FROZEN.json').exists():train()
 tasks();rows=[]
 with ProcessPoolExecutor(8,mp_context=multiprocessing.get_context('spawn')) as ex:
  fs=[ex.submit(worker,i) for i in range(36)]
  for j,f in enumerate(as_completed(fs)):rows+=f.result();print('CONFIRM',j+1,36,flush=True)
 pd.DataFrame(rows).to_csv(OUT/'confirmation.csv',index=False);save(OUT/'COMPLETE',dict(main_trajectories=1296,main_points=388800,teacher_points=547200,initial_points=14400))
if __name__=='__main__':main()
