"""Fresh grouped residual target comparison; protocol in docs/experiments."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
import copy,json,sys,time,multiprocessing,hashlib,zipfile
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import pandas as pd
import torch
from torch import nn
from training_validation import ROOT,Task,source,sha
from supplementary_experiments import build
import run_roopf as r
OUT=ROOT/'results/residual_target_study'
SEEDS=[20260928,20260929]

def save(p,x):p.write_text(json.dumps(x,indent=2))
def payload():return torch.load(ROOT/'checkpoints/residual_selector_generated36_d10.pt',map_location='cpu',weights_only=False)
def net():
    p=payload();h=p['state_dict']['net.0.weight'].shape[0]
    return nn.Sequential(nn.Linear(len(p['feature_names']),h),nn.LayerNorm(h),nn.GELU(),nn.Linear(h,h),nn.GELU(),nn.Linear(h,1))
def setup():
    OUT.mkdir(exist_ok=True);m=source();m.DEVICE=torch.device('cpu');tasks=[]
    for split,base,n in [('train',71000000,2),('validation',72000000,1),('confirmation',73000000,1)]:
        for i,f0 in enumerate(sorted(m.TRAIN_FUNCTIONS,key=lambda f:f['fid'])):
            for instance in range(n):
                seed=base+100*i+instance;f=copy.deepcopy(f0);r.set_seed(seed);m.gen_train_offset(10,f)
                pop=10*torch.rand(4,100,10,generator=torch.Generator().manual_seed(seed+10000000))-5
                tasks.append(dict(split=split,fid=f['fid'],instance=instance,seed=seed,params=f['params'],pop=pop))
    torch.save(tasks,OUT/'tasks.pt')
    save(OUT/'protocol.json',dict(script=sha(__file__),protocol=sha(ROOT/'docs/experiments/RESIDUAL_TARGET_PROTOCOL.md'),tasks=sha(OUT/'tasks.pt'),original={p.name:sha(p) for p in (ROOT/'checkpoints').glob('*.pt')}))
def execute(q,teacher=False,ckpt=None,variant='no_residual'):
    r.DEVICE=torch.device('cpu');torch.set_num_threads(1)
    m=source();f=copy.deepcopy(next(f for f in m.TRAIN_FUNCTIONS if f['fid']==q['fid']));f['params']=q['params'];t=Task(f,m)
    opt=build(variant)
    if ckpt:
        p=torch.load(ckpt,map_location='cpu',weights_only=False);opt.router_model.load_state_dict({k.removeprefix('net.'):v for k,v in p['state_dict'].items()})
        opt.router_mean.copy_(torch.tensor(p['mean']));opt.router_std.copy_(torch.tensor(p['std']))
    opt.log_pool_candidates=teacher;r.set_seed(q['seed']+20000000)
    with torch.no_grad():_,trail,nfe,points=opt(q['pop'].clone(),t)
    assert nfe==300 and t.points==(16400 if teacher else 1200)
    assert torch.isfinite(trail).all() and (trail[:,1:]<=trail[:,:-1]).all()
    return trail,points,opt.pool_candidate_samples,torch.get_rng_state()
def collect(i):
    q=torch.load(OUT/'tasks.pt',weights_only=False)[i];p=OUT/f'case_{i:03d}.npz'
    if p.exists():return i
    trail,points,records,rng=execute(q,True);d=pd.DataFrame(records)
    fit=d.candidate_fit.to_numpy().reshape(-1,38)
    assert len(d)==15200 and np.isfinite(fit).all()
    assert np.array_equal(d.candidate_slot.to_numpy().reshape(-1,38),np.tile(np.arange(38),(400,1)))
    cols=[]
    for name in payload()['feature_names']:
        if name.startswith('op_'):v=(d.op_name==name[3:]).to_numpy(dtype=np.float32)
        else:
            raw=pd.to_numeric(d[name.removesuffix('_missing')],errors='coerce').to_numpy(dtype=np.float32)
            v=(~np.isfinite(raw)).astype(np.float32) if name.endswith('_missing') else np.nan_to_num(raw,nan=0.,posinf=0.,neginf=0.)
        cols.append(v)
    np.savez_compressed(p,x=np.stack(cols,1),incumbent=d.improved_best.to_numpy(dtype=np.float32),second=(fit<fit[:,1:2]).reshape(-1).astype(np.float32),fit=fit,eval_before=d.eval_before.to_numpy(),trail=trail.numpy())
    return i
def data(split):
    tasks=torch.load(OUT/'tasks.pt',weights_only=False);ds=[np.load(OUT/f'case_{i:03d}.npz') for i,q in enumerate(tasks) if q['split']==split]
    return {k:np.concatenate([d[k] for d in ds]) for k in ['x','incumbent','second','eval_before']}
def train():
    tr,va=data('train'),data('validation');mean=tr['x'].mean(0);std=np.maximum(tr['x'].std(0),1e-6)
    dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu');torch.set_num_threads(4)
    x=torch.tensor((tr['x']-mean)/std,device=dev);v=torch.tensor((va['x']-mean)/std,device=dev)
    for target in ['incumbent','second']:
        y=torch.tensor(tr[target],device=dev);vy=torch.tensor(va[target],device=dev)
        for seed in SEEDS:
            tag=f'{target}_{seed}';r.set_seed(seed);model=net().to(dev);optim=torch.optim.Adam(model.parameters(),lr=.001);best=float('inf');stale=0;hist=[]
            for epoch in range(1,61):
                model.train();total=0.;perm=torch.randperm(len(x),device=dev)
                for ids in perm.split(4096):
                    optim.zero_grad();loss=nn.functional.binary_cross_entropy_with_logits(model(x[ids]).view(-1),y[ids]);assert torch.isfinite(loss);loss.backward();optim.step();total+=float(loss)*len(ids)
                model.eval()
                with torch.no_grad():vl=sum(float(nn.functional.binary_cross_entropy_with_logits(model(z).view(-1),yy,reduction='sum')) for z,yy in zip(v.split(16384),vy.split(16384)))/len(v)
                hist.append(dict(epoch=epoch,train_bce=total/len(x),validation_bce=vl))
                if vl<best-1e-5:
                    best=vl;stale=0;p=payload();p.update(state_dict={'net.'+k:z.detach().cpu() for k,z in model.state_dict().items()},mean=mean.tolist(),std=std.tolist(),target=target,epoch=epoch,seed=seed);torch.save(p,OUT/(tag+'.pt'))
                else:stale+=1
                save(OUT/(tag+'_history.json'),hist)
                print('TRAIN',tag,epoch,round(vl,6),flush=True)
                if stale>=8:break
    save(OUT/'SELECTION_FROZEN.json',{f'{t}_{s}':sha(OUT/f'{t}_{s}.pt') for t in ['incumbent','second'] for s in SEEDS})
def confirm(i):
    q=torch.load(OUT/'tasks.pt',weights_only=False)[i];p=OUT/f'confirm_{i:03d}.json'
    if p.exists():return json.loads(p.read_text())
    rows=[]
    m=source();f=copy.deepcopy(next(f for f in m.TRAIN_FUNCTIONS if f['fid']==q['fid']));f['params']=q['params'];y=Task(f,m).calfitness(q['pop']);initial=y.min(1).values;std=y.std(1).clamp_min(1e-8)
    for name in ['original','no_residual']+[f'{t}_{s}' for t in ['incumbent','second'] for s in SEEDS]:
        trail,_,_,_=execute(q,False,None if name in ['original','no_residual'] else OUT/(name+'.pt'),'no_residual' if name=='no_residual' else 'full')
        gain=((initial-trail[:,-1])/std).clamp_min(0)
        for j in range(4):rows.append(dict(method=name,fid=q['fid'],seed_index=j,final=float(trail[j,-1]),bounded_gain=float(gain[j]/(1+gain[j])),nfe=300))
    save(p,rows);return rows

def main():
    start=time.time()
    if not (OUT/'protocol.json').exists():setup()
    assert json.loads((OUT/'protocol.json').read_text())['script']==sha(__file__)
    tasks=torch.load(OUT/'tasks.pt',weights_only=False)
    if not (OUT/'PARITY.json').exists():
        a=execute(tasks[0]);b=execute(tasks[0],True);assert all(torch.equal(a[k],b[k]) for k in [0,1,3]);save(OUT/'PARITY.json',dict(trail_points_rng_exact=True))
    with ProcessPoolExecutor(8,mp_context=multiprocessing.get_context('spawn')) as ex:
        fs=[ex.submit(collect,i) for i,q in enumerate(tasks) if q['split']!='confirmation']
        for j,f in enumerate(as_completed(fs)):f.result();print('COLLECT',j+1,len(fs),flush=True)
    if not (OUT/'SELECTION_FROZEN.json').exists():train()
    # Only open confirmation after every objective/seed has frozen selection.
    with ProcessPoolExecutor(8,mp_context=multiprocessing.get_context('spawn')) as ex:
        fs=[ex.submit(collect,i) for i,q in enumerate(tasks) if q['split']=='confirmation']
        for f in as_completed(fs):f.result()
        fs=[ex.submit(confirm,i) for i,q in enumerate(tasks) if q['split']=='confirmation'];rows=[]
        for j,f in enumerate(as_completed(fs)):rows+=f.result();print('CONFIRM',j+1,36,flush=True)
    pd.DataFrame(rows).to_csv(OUT/'confirmation.csv',index=False)
    save(OUT/'COMPLETE',dict(seconds=time.time()-start))
if __name__=='__main__':main()
