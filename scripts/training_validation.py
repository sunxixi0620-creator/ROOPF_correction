"""Frozen-instance development validation and paired anchor continuation."""
import argparse, copy, hashlib, json, sys, time
from pathlib import Path
import numpy as np
import torch
from reconstruct_anchor_training import ROOT, module, TrainingProblem
from roopf.anchor_backbone import AnchorPolicyBackbone

OUT=ROOT/'results/training_validation'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def tree_to(x,device):
    if torch.is_tensor(x):return x.to(device)
    if isinstance(x,dict):return {k:tree_to(v,device) for k,v in x.items()}
    if isinstance(x,list):return [tree_to(v,device) for v in x]
    return x
def source():return module('validation_functions',ROOT/'artifacts/training_provenance/generated36_curated.py')

class Task(TrainingProblem):
    def getfunname(self):return self.fun['fid']
    def calfitness(self,x):
        self.points+=x.shape[0]*x.shape[1]
        y=self.fun['fun'](self.repaire(x),self.fun['params'])
        assert torch.isfinite(y).all(),self.fun['fid']
        return y

def prepare():
    OUT.mkdir(exist_ok=True);m=source();m.DEVICE=torch.device('cpu')
    for split,ps,ss in [('validation',65000000,66000000),('confirmation',67000000,68000000)]:
        path=OUT/(split+'.pt')
        if path.exists():continue
        tasks=[]
        for i,f0 in enumerate(sorted(m.TRAIN_FUNCTIONS,key=lambda f:f['fid'])):
            for inst in range(2):
                f=copy.deepcopy(f0);torch.manual_seed(ps+100*i+inst);m.gen_train_offset(10,f)
                pop=10*torch.rand(4,100,10,generator=torch.Generator().manual_seed(ss+100*i+inst))-5
                t=Task(f,m);y=t.calfitness(pop)
                tasks.append({'fid':f['fid'],'instance':inst,'params':f['params'],'pop':pop,
                              'initial_best':y.min(1).values,'initial_std':y.std(1).clamp_min(1e-8)})
        torch.save(tasks,path)
    (OUT/'dataset_manifest.json').write_text(json.dumps({s:sha(OUT/(s+'.pt')) for s in ['validation','confirmation']},indent=2))

def evaluate(model,split='validation'):
    device=next(model.parameters()).device;m=source();byfid={f['fid']:f for f in m.TRAIN_FUNCTIONS}
    tasks=torch.load(OUT/(split+'.pt'),weights_only=False,map_location='cpu');rows=[]
    was=model.training;model.eval()
    with torch.no_grad(),torch.random.fork_rng(devices=[0] if device.type=='cuda' else []):
        for q in tasks:
            f=copy.deepcopy(byfid[q['fid']]);f['params']=tree_to(q['params'],device)
            t=Task(f,m);pop=q['pop'].to(device);_,trail,nfe,_=model(pop,t)
            assert nfe==300 and t.points==1200
            assert torch.isfinite(trail).all() and (trail[:,1:]<=trail[:,:-1]).all()
            final=trail[:,-1].cpu();gain=(q['initial_best']-final)/q['initial_std']
            assert (gain>=-1e-5).all();gain=gain.clamp_min(0)
            for i in range(4):rows.append({'fid':q['fid'],'instance':q['instance'],'seed_index':i,
                'initial_best':float(q['initial_best'][i]),'initial_std':float(q['initial_std'][i]),
                'final':float(final[i]),'gain':float(gain[i]),'bounded_gain':float(gain[i]/(1+gain[i])), 'nfe':300})
    model.train(was)
    return {'score':float(np.mean([r['bounded_gain'] for r in rows])),
            'median_gain':float(np.median([r['gain'] for r in rows])),'rows':rows}

def train(arm,stop):
    assert stop==160,'Protocol fixes stop at160'
    out=OUT/arm;out.mkdir(exist_ok=True);device=torch.device('cuda')
    m=source();funs=[copy.deepcopy(f) for f in m.TRAIN_FUNCTIONS];ids=sorted(f['fid'] for f in funs)
    parent=ROOT/'results/retrain_d10_curated_20260928/resume.pt'
    protocol={'arm':arm,'start_epoch':80,'stop_epoch':160,'schedule_horizon':80,'seed':20271000,
              'parent_sha256':sha(parent),'dataset_sha256':sha(OUT/'validation.pt'),'source_sha256':sha(__file__)}
    pp=out/'protocol.json'
    if pp.exists():assert json.loads(pp.read_text())==protocol
    else:pp.write_text(json.dumps(protocol,indent=2))
    model=AnchorPolicyBackbone(10,200,100).to(device).train();optim=torch.optim.Adam(model.parameters(),lr=.001)
    statefile=out/'resume.pt';state=torch.load(statefile if statefile.exists() else parent,map_location=device,weights_only=False)
    model.load_state_dict(state['model']);optim.load_state_dict(state['optimizer'])
    for f,param in zip(funs,state['function_params']):f['params']=param
    torch.set_rng_state(state['cpu_rng'].cpu());torch.cuda.set_rng_state_all([v.cpu() for v in state['cuda_rng']])
    history=state.get('development_history',[]);best=state.get('best_validation',-float('inf'))
    start=time.perf_counter();start_epoch=state['epoch']+1;points=state.get('continuation_points',0)
    def validate(epoch):
        nonlocal best
        result=evaluate(model);result['epoch']=epoch
        (out/f'validation_{epoch:03d}.json').write_text(json.dumps(result,indent=2))
        torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()},out/f'epoch_{epoch:03d}.pt')
        if result['score']>best:
            best=result['score'];torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()},out/'selected.pt')
            (out/'selection.json').write_text(json.dumps({'epoch':epoch,'score':best,'criterion':'validation bounded gain only'},indent=2))
        print(arm,'validation',epoch,result['score'],flush=True)
    if not history:validate(80)
    for epoch in range(start_epoch,stop):
        if (epoch+1)%20==0:
            for f in funs:
                with torch.random.fork_rng(devices=[0]):
                    torch.manual_seed(20271000+100000*epoch+ids.index(f['fid']));m.gen_train_offset(10,f)
        model.train();optim.zero_grad();total=0.
        for i,f in enumerate(funs):
            with torch.random.fork_rng(devices=[0]):
                torch.manual_seed(20271000+200000*epoch+1000*ids.index(f['fid']))
                pop=10*torch.rand(64,100,10,device=device)-5;t=Task(f,m)
                _,_,_,candidates=model(pop,t)
                if arm=='legacy':loss=t.loss(pop,candidates,0.)/36
                else:
                    parent_y=t.calfitness(pop);child_y=t.calfitness(candidates)
                    scale=parent_y.std(1).detach().clamp_min(1e-8)
                    loss=((child_y.min(1).values-parent_y.min(1).values)/scale).mean()/36
                assert torch.isfinite(loss),(epoch,f['fid']);loss.backward()
            assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
            total+=float(loss.detach());points+=t.points
            if (i+1)%4==0:torch.nn.utils.clip_grad_norm_(model.parameters(),10);optim.step();optim.zero_grad()
            del candidates,loss,t
        if (epoch+1)%10==0:validate(epoch+1)
        history.append({'epoch':epoch+1,'loss':total,'seconds_this_session':time.perf_counter()-start,'continuation_points':points})
        (out/'history.json').write_text(json.dumps(history,indent=2))
        torch.save({'model':model.state_dict(),'optimizer':optim.state_dict(),'epoch':epoch,
            'function_params':[f['params'] for f in funs],'cpu_rng':torch.get_rng_state(),
            'cuda_rng':torch.cuda.get_rng_state_all(),'development_history':history,
            'best_validation':best,'continuation_points':points},statefile)
        print(arm,epoch+1,'/160',total,flush=True)
    (out/'COMPLETE').write_text(json.dumps({'epochs_completed':160,'additional_epochs':80,'training_points':points,'selected_sha256':sha(out/'selected.pt')}))

def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','train']);p.add_argument('--arm',choices=['legacy','shared_scale']);p.add_argument('--stop',type=int,default=160);a=p.parse_args()
    if a.mode=='prepare':prepare()
    else:train(a.arm,a.stop)

if __name__=='__main__':main()
