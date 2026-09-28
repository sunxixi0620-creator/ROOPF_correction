"""Controlled retraining from recovered compatible architecture and loss.

Not a claim to recover the historical checkpoint's random seed/command.
"""
import argparse, importlib.util, json, time, hashlib, copy, sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from roopf.anchor_backbone import AnchorPolicyBackbone

def module(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

class TrainingProblem:
    def __init__(self,fun,mod):self.fun=fun;self.mod=mod;self.points=0
    def repaire(self,x):return x.clamp(-5,5)
    def calfitness(self,x):
        self.points+=x.shape[0]*x.shape[1]
        return self.mod.get_train_fitness(self.repaire(x),self.fun)
    def loss(self,pop,candidates,lam):
        a=self.calfitness(pop);b=self.calfitness(candidates)
        a=(a-a.mean(1,keepdim=True))/(a.std(1,keepdim=True)+1e-8)
        b=(b-b.mean(1,keepdim=True))/(b.std(1,keepdim=True)+1e-8)
        quality=(b.min(1).values-a.min(1).values).mean()
        diversity=-torch.cdist(candidates,candidates).mean()/(10*pop.shape[-1]**.5+1e-8)
        return quality+lam*diversity

def main():
    p=argparse.ArgumentParser();p.add_argument('--dim',type=int,required=True)
    p.add_argument('--order',choices=['curated','original','canonical'],required=True)
    p.add_argument('--seed',type=int,default=20271000);p.add_argument('--epochs',type=int,default=80)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    original=module('generated_original',ROOT/'artifacts/training_provenance/generated36_original.py')
    curated=module('generated_curated',ROOT/'artifacts/training_provenance/generated36_curated.py')
    source=curated if a.order=='curated' else original
    funs=[dict(f) for f in source.TRAIN_FUNCTIONS]
    if a.order=='canonical':funs.sort(key=lambda f:f['fid'])
    ids=sorted(f['fid'] for f in funs);assert len(ids)==len(set(ids))==36
    torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed)
    model=AnchorPolicyBackbone(a.dim,200,100).cuda().train()
    optim=torch.optim.Adam(model.parameters(),lr=.001)
    protocol={'dimension':a.dim,'order':a.order,'seed':a.seed,'epochs':a.epochs,
      'batch_size':64,'NFE_per_training_trajectory':300,'gradient_accumulation_functions':4,
      'loss':'separately standardized best candidate minus best parent; lambda=.5*(1-epoch/epochs) times negative mean candidate distance / bound diagonal',
      'lr':.001,'Adam_defaults':True,'clip_norm':10,'offset_refresh_epochs':[0,19,39,59,79],
      'checkpoint_selection':'best training loss only; never benchmark performance',
      'function_order':[f['fid'] for f in funs],
      'status':'reconstructed controlled recipe, not exact historical reproduction',
      'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    pp=a.output/'protocol.json'
    if pp.exists():assert json.loads(pp.read_text())==protocol
    else:pp.write_text(json.dumps(protocol,indent=2))
    start_epoch=0;history=[];best=float('inf');elapsed=0.;point_count=0
    statefile=a.output/'resume.pt'
    if statefile.exists():
        state=torch.load(statefile,map_location='cuda',weights_only=False)
        model.load_state_dict(state['model']);optim.load_state_dict(state['optimizer'])
        for f,param in zip(funs,state['function_params']):f['params']=param
        torch.set_rng_state(state['cpu_rng'].cpu());torch.cuda.set_rng_state_all([v.cpu() for v in state['cuda_rng']])
        start_epoch=state['epoch']+1;history=state['history'];best=state['best'];elapsed=state['seconds'];point_count=state['points']
    start=time.perf_counter()
    for epoch in range(start_epoch,a.epochs):
        # Stable per-function parameter and population streams isolate order from
        # assigning different random task instances to different function IDs.
        if epoch==0 or (epoch+1)%20==0:
            for f in funs:
                with torch.random.fork_rng(devices=[0]):
                    torch.manual_seed(a.seed+100000*epoch+ids.index(f['fid']))
                    source.gen_train_offset(a.dim,f)
        optim.zero_grad();total=0.;invalid=[]
        for i,f in enumerate(funs):
            with torch.random.fork_rng(devices=[0]):
                torch.manual_seed(a.seed+200000*epoch+1000*ids.index(f['fid']))
                pop=10*torch.rand((64,100,a.dim),device='cuda')-5
                task=TrainingProblem(f,source)
                _,_,_,candidates=model(pop,task)
                loss=task.loss(pop,candidates,.5*(1-epoch/a.epochs))/36
                assert torch.isfinite(loss), (epoch,f['fid'],'nonfinite loss')
                loss.backward()
            assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None), (epoch,f['fid'],'nonfinite gradient')
            total+=float(loss.detach());point_count+=task.points
            if (i+1)%4==0:
                torch.nn.utils.clip_grad_norm_(model.parameters(),10);optim.step();optim.zero_grad()
            del candidates,loss,task
        torch.cuda.synchronize()
        if total<best:
            best=total;torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()},a.output/'anchor.pt')
        history.append({'epoch':epoch,'loss':total,'best_training_loss':best,'seconds':elapsed+time.perf_counter()-start,'point_evaluations':point_count})
        (a.output/'history.json').write_text(json.dumps(history,indent=2))
        torch.save({'model':model.state_dict(),'optimizer':optim.state_dict(),'epoch':epoch,
            'function_params':[f['params'] for f in funs],'cpu_rng':torch.get_rng_state(),
            'cuda_rng':torch.cuda.get_rng_state_all(),'history':history,'best':best,
            'seconds':elapsed+time.perf_counter()-start,'points':point_count},statefile)
        print(a.dim,a.order,epoch+1,'/',a.epochs,'loss',total,'seconds',history[-1]['seconds'],flush=True)
    (a.output/'COMPLETE').write_text(json.dumps({'epochs':a.epochs,'seconds':history[-1]['seconds'],'point_evaluations':point_count,
        'peak_gpu_bytes':torch.cuda.max_memory_allocated(),'checkpoint_sha256':hashlib.sha256((a.output/'anchor.pt').read_bytes()).hexdigest()}))

if __name__=='__main__':main()
