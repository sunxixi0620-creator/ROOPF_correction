"""Generated-instance gate controls and paired residual-label diagnostics."""
import argparse,copy,json,multiprocessing,sys,time
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import pandas as pd
import torch
from torch import nn
from training_validation import OUT,ROOT,Task,source,sha
from supplementary_experiments import build
import run_roopf as r

class ZeroGate(nn.Module):
    def forward(self,x):return torch.zeros(x.shape[0],6,device=x.device,dtype=x.dtype)

def run(q,mode,teacher=False):
    m=source();f=copy.deepcopy(next(f for f in m.TRAIN_FUNCTIONS if f['fid']==q['fid']));f['params']=q['params']
    task=Task(f,m);opt=build('full')
    if mode=='memory_only':opt.gate=ZeroGate()
    elif mode=='uniform':opt.gate=ZeroGate();opt.ablation.add('no_success_memory')
    opt.log_pool_candidates=teacher
    r.set_seed(69000000+100*sorted(f['fid'] for f in m.TRAIN_FUNCTIONS).index(q['fid'])+q['instance'])
    with torch.no_grad():_,trail,nfe,points=opt(q['pop'].clone(),task)
    assert nfe==300 and task.points==(16400 if teacher else 1200),(mode,task.points)
    assert torch.isfinite(trail).all() and (trail[:,1:]<=trail[:,:-1]).all()
    rows=[{'fid':q['fid'],'instance':q['instance'],'seed_index':i,'method':mode,'final':float(trail[i,-1]),'nfe':300} for i in range(4)]
    return rows,trail,points,opt.pool_candidate_samples,torch.get_rng_state()

def labels(records,path):
    d=pd.DataFrame(records);assert len(d)==15200
    # Each consecutive pool has exactly two anchor proposals followed by36slots.
    assert np.array_equal(d.candidate_slot.to_numpy().reshape(-1,38),np.tile(np.arange(38),(400,1)))
    fit=d.candidate_fit.to_numpy().reshape(-1,38);assert np.isfinite(fit).all()
    assert (d.is_baseline.to_numpy().reshape(-1,38)[:,:2]==1).all()
    p=d[d.is_baseline==0].copy();p['beats_second']= (fit[:,2:]<fit[:,1:2]).reshape(-1)
    p['beats_best_anchor']=(fit[:,2:]<fit[:,:2].min(1,keepdims=True)).reshape(-1)
    assert np.array_equal(p['beats_best_anchor'],p.beat_baseline.astype(bool))
    payload=torch.load(ROOT/'checkpoints/residual_selector_generated36_d10.pt',map_location='cpu',weights_only=False)
    cols=[]
    for name in payload['feature_names']:
        if name.startswith('op_'):col=(p.op_name==name[3:]).to_numpy(dtype=np.float32)
        else:
            raw=pd.to_numeric(p[name.removesuffix('_missing')],errors='coerce').to_numpy(dtype=np.float32)
            col=(~np.isfinite(raw)).astype(np.float32) if name.endswith('_missing') else np.nan_to_num(raw,nan=0.,posinf=0.,neginf=0.)
        cols.append(col)
    x=np.stack(cols,1);state=payload['state_dict'];hidden=state['net.0.weight'].shape[0]
    model=nn.Sequential(nn.Linear(x.shape[1],hidden),nn.LayerNorm(hidden),nn.GELU(),nn.Linear(hidden,hidden),nn.GELU(),nn.Linear(hidden,1)).eval()
    model.load_state_dict({k.removeprefix('net.'):v for k,v in state.items()})
    with torch.no_grad():
        q=torch.sigmoid(model((torch.tensor(x)-torch.tensor(payload['mean']))/torch.tensor(payload['std']).clamp_min(1e-6))).view(-1).numpy()
    np.savez_compressed(path,q=q,improved_best=p.improved_best.to_numpy(),beats_second=p.beats_second.to_numpy(),
        beats_best_anchor=p.beats_best_anchor.to_numpy(),eval_before=p.eval_before.to_numpy(),
        candidate_fit=p.candidate_fit.to_numpy(),second_fit=np.repeat(fit[:,1],36),best_before=p.fitness_best_before.to_numpy())

def worker(index):
    q=torch.load(OUT/'validation.pt',map_location='cpu',weights_only=False)[index]
    dest=OUT/'auxiliary';file=dest/f'case_{index:02d}.json'
    if file.exists():return json.loads(file.read_text())
    rows=[]
    for mode in ['full','memory_only','uniform']:
        rs,trail,points,log,rng=run(q,mode,teacher=mode=='full');rows+=rs
        np.savez_compressed(dest/f'case_{index:02d}_{mode}.npz',trail=trail.numpy(),points=points.numpy())
        if mode=='full':labels(log,dest/f'case_{index:02d}_labels.npz')
    file.write_text(json.dumps(rows,indent=2));return rows

def main():
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=6);a=p.parse_args()
    dest=OUT/'auxiliary';dest.mkdir(exist_ok=True)
    q=torch.load(OUT/'validation.pt',map_location='cpu',weights_only=False)[0]
    n=run(q,'full',False);t=run(q,'full',True)
    assert torch.equal(n[1],t[1]) and torch.equal(n[2],t[2]) and torch.equal(n[4],t[4])
    protocol={'source_sha256':sha(__file__),'dataset_sha256':sha(OUT/'validation.pt'),
        'teacher_parity':'exact points,trail,RNG for first fixed case; no target feedback',
        'methods':['full','memory_only','uniform'],'main_trajectories':864,'main_points':259200,
        'teacher_points':1094400,'scope':'generated instance-held-out development only'}
    (dest/'protocol.json').write_text(json.dumps(protocol,indent=2));rows=[]
    with ProcessPoolExecutor(a.workers,mp_context=multiprocessing.get_context('spawn')) as ex:
        fs=[ex.submit(worker,i) for i in range(72)]
        for i,f in enumerate(as_completed(fs)):
            rows+=f.result()
            if (i+1)%6==0:print(i+1,'/72',flush=True)
    pd.DataFrame(rows).to_csv(dest/'raw_results.csv',index=False)
    (dest/'COMPLETE').write_text(json.dumps(protocol))

if __name__=='__main__':main()
