"""Paired, role-restricted source episodes and conditional-prior learning."""
import numpy as np
import torch
from torch import nn
from roopf.probabilistic_prior_v1 import NeuralPrior,BaseCovariance,condition,conditional_nll,normalize_context
from roopf.experiment_io import seed_for


class SourceModel(nn.Module):
    def __init__(self,dim,kind,seed,space):
        super().__init__();assert kind in ('N','M','PCA');self.kind=kind
        torch.manual_seed(seed_for('source_prior_init_v1',space,seed)%(2**31-1))
        self.prior=NeuralPrior(dim) if kind in ('N','M') else None
        self.base=BaseCovariance(dim)

    def forward(self,batch):
        xc,yc,xq,yq,pc,pq=batch
        zc,zq,_,_=normalize_context(yc,yq)
        if self.prior is not None:
            pc,pq=self.prior(xc),self.prior(xq)
            if self.kind=='M':pc=(pc[0],pc[1]*0);pq=(pq[0],pq[1]*0)
        mean,cov=condition(self.base,xc,zc,xq,pc,pq)
        nll=conditional_nll(mean,cov,zq)
        return nll,mean,cov,zq


class Episodes:
    """Every optimizer variant receives the same context/query indices per step."""
    def __init__(self,tasks,space,role,device):
        self.space=space;self.role=role;self.tasks=tasks;self.device=device
        assert tasks and all(t['role']==role for t in tasks)

    def description(self,task_index,n,repeat,seed,mode):
        t=self.tasks[task_index]
        rng=np.random.default_rng(seed_for('source_episode_v1',self.space,self.role,t['name'],n,repeat,seed,mode))
        permutation=rng.permutation(len(t['x']))
        context=permutation[:n] if mode=='uniform' else np.array(t['prefixes'][repeat%2][:n],dtype=np.int64)
        query=permutation[~np.isin(permutation,context)][:32]
        assert len(context)==n and len(set(context))==n and len(query)==32 and not set(query)&set(context)
        return task_index,context,query

    def batch(self,descriptions):
        rows=[]
        for i,c,q in descriptions:
            t=self.tasks[i]
            rows.append((t['x'][c],t['y'][c],t['x'][q],t['y'][q],
                         t['mean'][c],t['features'][c],t['mean'][q],t['features'][q]))
        tensors=[torch.stack([row[k] for row in rows]).to(self.device) for k in range(8)]
        return (*tensors[:4],(tensors[4],tensors[5]),(tensors[6],tensors[7]))

    def training(self,seed,step):
        assert self.role=='train'
        rng=np.random.default_rng(seed_for('source_training_batch_v1',self.space,seed,step))
        n=int(rng.choice([5,10,20,40]));ids=rng.integers(0,len(self.tasks),size=16)
        # Exactly half uniform, half search-prefix; all indices stateless on step.
        return self.batch([self.description(int(i),n,step*16+k,seed,
            'uniform' if k<8 else 'search') for k,i in enumerate(ids)])

    @torch.no_grad()
    def validation(self,model):
        assert self.role=='validation';was_training=model.training;model.eval();rows=[]
        for n in (5,10,20,40):
            desc=[self.description(i,n,r,0,'uniform' if r<4 else 'search')
                  for i in range(len(self.tasks)) for r in range(8)]
            for start in range(0,len(desc),32):
                subset=desc[start:start+32];batch=self.batch(subset)
                nll,mean,cov,y=model(batch)
                sd=cov.diagonal(dim1=-2,dim2=-1).clamp_min(0).sqrt()
                coverage=((y-mean).abs()<=1.6448536269514722*sd).double().mean(-1)
                best=y.max(-1).values;chosen=y.gather(-1,mean.argmax(-1,keepdim=True)).squeeze(-1)
                worst=y.min(-1).values;span=best-worst
                regret=torch.where(span>0,(best-chosen)/span.clamp_min(1e-12),torch.zeros_like(span))
                for k,(i,_,_) in enumerate(subset):
                    rows.append(dict(task=self.tasks[i]['name'],group=self.tasks[i]['group'],context=n,
                        nll=float(nll[k]),coverage90=float(coverage[k]),query_choice_regret=float(regret[k])))
        if was_training:model.train()
        # Exactly the same number of episodes for each source-validation task.
        return dict(nll=float(np.mean([r['nll'] for r in rows])),
            coverage90=float(np.mean([r['coverage90'] for r in rows])),
            query_choice_regret=float(np.mean([r['query_choice_regret'] for r in rows])),rows=rows)


def optimizer_for(model):
    return torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)


def update(model,optimizer,batch):
    model.train();optimizer.zero_grad(set_to_none=True)
    nll,_,_,_=model(batch);loss=nll.mean()
    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training NLL')
    loss.backward()
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0,error_if_nonfinite=True)
    optimizer.step()
    if not all(torch.isfinite(p).all() for p in model.parameters()):raise FloatingPointError('Nonfinite parameters')
    return float(loss.detach()),float(norm.detach())


def cpu_tree(value):
    if torch.is_tensor(value):return value.detach().cpu().clone()
    if isinstance(value,dict):return {k:cpu_tree(v) for k,v in value.items()}
    if isinstance(value,list):return [cpu_tree(v) for v in value]
    if isinstance(value,tuple):return tuple(cpu_tree(v) for v in value)
    return value
