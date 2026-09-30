"""Grouped long-horizon value learning; fixed-quota diagnostic, not deployment."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
from pathlib import Path
import sys,json,time,multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import torch
from torch import nn
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import region_potential as parent
from roopf.revision_tasks import ProceduralTask,population
from roopf.experiment_io import CaseStore,sha256,fingerprint,seed_for,write_json,save_torch
from scripts.unified_revision import bounded
RUN=ROOT/'results/long_value_20260930';OUT=ROOT/'docs/revision/long_value'
SOURCES=['scripts/long_value.py','docs/experiments/LONG_VALUE_PROTOCOL.md']
STEPS=(0,105);MENUS=('Iso','Pop','Center')


def setup(f,i,p,role='long_value_v1'):
    task=ProceduralTask(f,i,role,dim=20)
    x=population(f,i,role,count=4,dim=20)[p:p+1].clone()
    return task,x,seed_for(role,f,i,p,'policy')


def rollout(model,f,i,p,role='long_value_v1'):
    task,x,ps=setup(f,i,p,role);initial_x=x.clone();before=fingerprint(model.state_dict());torch.manual_seed(ps)
    with torch.no_grad():_,trail,nfe,points=model(x,task)
    assert nfe==model.MaxNFE and task.points==nfe and task.diagnostic_points==0
    assert torch.isfinite(trail).all() and before==fingerprint(model.state_dict())
    return dict(initial_x=initial_x,initial=task.initial_values,trail=trail,points=points,main_calls=nfe)


def candidates(state,base,step,f,i,p):
    best=state['x'][0,0];g=torch.Generator().manual_seed(seed_for('long_value_menu',f,i,p,step))
    iso=torch.randn(2,20,generator=g);indices=torch.randint(0,100,(2,2),generator=g)
    dirs=[iso,state['x'][0,indices[:,0]]-state['x'][0,indices[:,1]],torch.stack((-best,state['x'][0].mean(0)-best))]
    known=torch.cat((base['initial_x'][0],base['points'][0,:2*step],state['candidates'][0]),0)
    points=[];metadata=[]
    for m,d in enumerate(dirs):
        for j in range(2):
            norm=d[j].square().mean().sqrt()
            for radius in (.1,.2):
                raw=best+d[j]/norm.clamp_min(1e-12)*(10*radius);point=raw.clamp(-5,5)
                novelty=float(((known-point).square().mean(1).sqrt()/10).min())
                points.append(point);metadata.append(dict(menu=m,radius=radius,novelty=novelty,
                    eligible=bool(norm>1e-12 and novelty>=.05),clipped=bool(not torch.equal(raw,point))))
    return torch.stack(points),metadata


def features(state,points,metadata,task):
    # No Base continuation or branch label argument exists in this interface.
    x=state['x'][0];fit=state['fitness'][0];ax=state['ax'][0];ay=state['ay'][0]
    best=x[0];fy=fit[0];std=ay.std(unbiased=False).clamp_min(1e-8);centroid=x.mean(0)
    root=lambda v:v.square().mean(-1).sqrt()/10
    common=torch.tensor([state['remaining'],float(state['stagnation'][0]),len(ax)/256,
        float((fit.mean()-fy)/std),float(fit.std(unbiased=False)/std),float((fy-ay.mean())/std),
        float(root(x-best).square().mean().sqrt()),float(root(centroid-best))])
    rows=[]
    for point,meta in zip(points,metadata):
        distance=root(ax-point);indices=torch.argsort(distance,stable=True)[:5];ys=ay[indices]
        candidate=torch.tensor([float(root(point-best)),float(distance.min()),float(root(point-centroid)),
            float((point.abs()>=4.999).float().mean()),float((ys[0]-fy)/std),float((ys.mean()-fy)/std),
            float(ys.std(unbiased=False)/std),meta['novelty']])
        source=torch.zeros(3);source[meta['menu']]=1
        rows.append(torch.cat((common,candidate,(point-best)/10,source,torch.tensor([meta['radius'],float(meta['clipped'])]))))
    geo=torch.stack(rows);assert geo.shape==(12,41)
    model=parent.make();second=state['candidates'][:,int(state['chosen'][0,1]):int(state['chosen'][0,1])+1]
    before=(task.points,task.diagnostic_points)
    with torch.no_grad():
        joined=torch.cat((points[None],second),1)
        mu,sigma=model.surrogate.predict(state['ax'],state['ay'],joined,task,distance_scale=.03)
        score=model.scores(points[None],torch.full((1,12),1/6),state['ax'],state['ay'],task,state['remaining'],state['stagnation'])[0]
    assert before==(task.points,task.diagnostic_points)
    proxy=state['scores'][0,int(state['chosen'][0,1])]-score
    extra=torch.stack(((mu[0,:12]-fy)/std,sigma[0,:12]/std,proxy,(mu[0,:12]-mu[0,12])/std),1)
    fusion=torch.cat((geo,extra),1);assert fusion.shape==(12,45) and torch.isfinite(fusion).all()
    return geo,fusion,proxy


def verify():
    identity=json.loads((RUN/'identity.json').read_text())
    for p,h in identity['sources'].items():assert sha256(ROOT/p)==h
    parent.verify();assert identity['parent_identity']==sha256(parent.RUN/'identity.json')
    return identity


def contracts():
    path=RUN/'CONTRACTS.json'
    if path.exists():return
    role='long_value_contract';a=rollout(parent.make(114),0,0,0,role)
    model=parent.make(114);snap=parent.instrument(model,(2,));b=rollout(model,0,0,0,role);st=snap[2]
    model=parent.make(114);parent.instrument(model,(2,),dict(step=2,fingerprint=st['fingerprint'],replace=False));c=rollout(model,0,0,0,role)
    point=torch.full((20,),4.5);model=parent.make(114);parent.instrument(model,(2,),dict(step=2,fingerprint=st['fingerprint'],replace=True,point=point));d=rollout(model,0,0,0,role)
    assert torch.equal(a['points'],b['points']) and torch.equal(b['points'],c['points'])
    assert torch.equal(d['points'][:,:5],b['points'][:,:5]) and torch.equal(d['points'][0,5],point)
    pts,meta=candidates(st,b,2,0,0,0);task,_,_=setup(0,0,0,role);geo,fused,proxy=features(st,pts,meta,task)
    assert task.points==0 and task.diagnostic_points==0
    save_torch(RUN/'contract.pt',dict(base=a,hook=b,noop=c,forced=d,geo=geo,fused=fused,proxy=proxy))
    write_json(path,dict(calls=456,prefix_exact=True,noop_exact=True,feature_calls=0))


def collect(job):
    torch.set_num_threads(1);f,i,p=job;identity=verify();store=CaseStore(RUN/'cases',identity)
    spec=dict(fid=f,instance=i,population=p);key=f'f{f}_i{i}_p{p}'
    with store.lock(key):
        cached=store.load(key,spec)
        if cached is not None:return key
        model=parent.make();snap=parent.instrument(model,STEPS);base=rollout(model,f,i,p)
        init=base['initial'].min(1).values;scale=base['initial'].std(1);calls=600;prefix=0;states=[]
        for step in STEPS:
            st=snap[step];pts,meta=candidates(st,base,step,f,i,p);task,_,_=setup(f,i,p)
            geo,fused,proxy=features(st,pts,meta,task)
            feature_sha=fingerprint((geo,fused,proxy));labels=[];branches=[]
            for k,m in enumerate(meta):
                if m['eligible']:
                    model=parent.make();parent.instrument(model,(step,),dict(step=step,fingerprint=st['fingerprint'],replace=True,point=pts[k]))
                    b=rollout(model,f,i,p);calls+=600;prefix+=100+2*step
                    assert torch.equal(b['points'][:,:2*step+1],base['points'][:,:2*step+1])
                    assert torch.equal(b['points'][0,2*step+1],pts[k])
                else:b=base
                delta=float((bounded(init,b['trail'][:,-1],scale)-bounded(init,base['trail'][:,-1],scale))[0]);labels.append(delta)
                branches.append(dict(trail=b['trail'],points_sha=fingerprint(b['points']),calls=600 if m['eligible'] else 0))
            assert feature_sha==fingerprint((geo,fused,proxy))
            states.append(dict(step=step,snapshot=st,points=pts,metadata=meta,geo=geo,fusion=fused,proxy=proxy,
                eligible=torch.tensor([m['eligible'] for m in meta]),labels=torch.tensor(labels),branches=branches,feature_sha=feature_sha))
        store.save(key,spec,dict(base=base,states=states,calls=calls,redundant_prefix_calls=prefix,off_rollout_calls=0))
        return key


def subset(data,indices):return {k:v[indices] for k,v in data.items()}


def prepare_data():
    identity=verify();store=CaseStore(RUN/'cases',identity);rows=[];calls=0;prefix=0
    for f in range(36):
        for i in range(2):
            for p in range(4):
                v=store.load(f'f{f}_i{i}_p{p}',dict(fid=f,instance=i,population=p));assert v is not None
                calls+=v['calls'];prefix+=v['redundant_prefix_calls']
                for st in v['states']:
                    keys=[seed_for('long_value_tie',f,i,p,st['step'],k) for k in range(12)]
                    rows.append(dict(geo=st['geo'].numpy(),fusion=st['fusion'].numpy(),proxy=st['proxy'].numpy(),
                        eligible=st['eligible'].numpy(),y=st['labels'].numpy(),id=np.array([f,i,p,st['step']]),
                        candidate_ties=np.array(keys),state_tie=seed_for('long_value_state_tie',f,i,p,st['step'])))
    data={k:np.array([r[k] for r in rows]) for k in rows[0]}
    assert len(data['id'])==576 and data['eligible'].any(1).all()
    save_torch(RUN/'dataset.pt',data);folds=[]
    recipe=data['id'][:,0]//3
    for fold in range(3):
        test=[r for r in range(12) if r%3==fold];available=[r for r in range(12) if r not in test];val=available[-2:];train=available[:-2]
        parts={name:subset(data,np.flatnonzero(np.isin(recipe,rs))) for name,rs in [('train',train),('val',val),('test',test)]}
        assert [len(parts[k]['id']) for k in ('train','val','test')]==[288,96,192]
        save_torch(RUN/f'trainval_{fold}.pt',dict(train=parts['train'],val=parts['val']))
        save_torch(RUN/f'heldout_{fold}.pt',parts['test'])
        folds.append(dict(fold=fold,train=train,val=val,test=test,trainval_sha=sha256(RUN/f'trainval_{fold}.pt'),heldout_sha=sha256(RUN/f'heldout_{fold}.pt')))
    write_json(RUN/'DATA.json',dict(calls=calls,contract_calls=456,total_calls=calls+456,redundant_prefix_calls=prefix,
        states=576,trajectories=calls//600,folds=folds,dataset_sha=sha256(RUN/'dataset.pt')))


def net(dim):
    n=nn.Sequential(nn.Linear(dim,64),nn.SiLU(),nn.Linear(64,32),nn.SiLU(),nn.Linear(32,1))
    nn.init.zeros_(n[-1].weight);nn.init.zeros_(n[-1].bias);return n


def policy(scores,data,quota=.25):
    scores=np.where(data['eligible'],scores,-np.inf);n=len(scores);choice=[]
    for j in range(n):choice.append(int(np.lexsort((data['candidate_ties'][j],-scores[j]))[0]))
    choice=np.array(choice);best=scores[np.arange(n),choice]
    k=int(n*quota);order=np.lexsort((data['state_tie'],-best));mask=np.zeros(n,dtype=bool);mask[order[:k]]=True
    assert np.isfinite(best[mask]).all() and mask.sum()==k
    return dict(gain=np.where(mask,data['y'][np.arange(n),choice],0.),choice=choice,mask=mask)


def benchmark():
    path=RUN/'TRAINING_RESOURCES.json'
    if path.exists():return json.loads(path.read_text())
    times={}
    for device in ('cpu','cuda'):
        if device=='cuda' and not torch.cuda.is_available():continue
        torch.manual_seed(20261006);model=net(45).to(device);opt=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=.0001)
        x=torch.randn(256,45,device=device);y=torch.randn(256,1,device=device)
        for j in range(120):
            if j==20:
                if device=='cuda':torch.cuda.synchronize()
                start=time.perf_counter()
            opt.zero_grad();loss=(model(x)-y).square().mean();loss.backward();opt.step()
        if device=='cuda':torch.cuda.synchronize()
        times[device]=time.perf_counter()-start
        del model,opt,x,y
    chosen=min(times,key=times.get);result=dict(device=chosen,seconds_per_100_steps=times,synthetic_only=True,objective_calls=0)
    write_json(path,result);return result


def train(job):
    torch.set_num_threads(1);fold,kind,seed,device=job;verify();meta=json.loads((RUN/'DATA.json').read_text())['folds'][fold]
    assert sha256(RUN/f'trainval_{fold}.pt')==meta['trainval_sha']
    path=RUN/f'model_{fold}_{kind}_{seed}.pt';marker=RUN/f'model_{fold}_{kind}_{seed}.json'
    if marker.exists():
        assert sha256(path)==json.loads(marker.read_text())['sha'];return str(marker.name)
    payload=torch.load(RUN/f'trainval_{fold}.pt',weights_only=False);tr,va=payload['train'],payload['val']
    mask=tr['eligible'];raw=tr[kind][mask];mean=raw.mean(0);std=np.maximum(raw.std(0),.001)
    norm=lambda a:np.clip((a-mean)/std,-10,10)
    x=torch.tensor(norm(raw),device=device);y=torch.tensor(tr['y'][mask]*1000,device=device)[:,None]
    vx=torch.tensor(norm(va[kind]),device=device)
    torch.manual_seed(seed_for('long_value_training',fold,kind,seed));model=net(x.shape[1]).to(device)
    opt=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=.0001)
    g=torch.Generator().manual_seed(seed_for('long_value_batches',fold,kind,seed))
    best=-float('inf');bad=0;history=[];selected=0
    for epoch in range(201):
        if epoch:
            model.train();order=torch.randperm(len(x),generator=g).to(device)
            for ix in order.split(256):
                opt.zero_grad();loss=(model(x[ix])-y[ix]).square().mean();loss.backward();opt.step()
        if epoch%5==0:
            model.eval()
            with torch.no_grad():pred=model(vx).squeeze(-1).cpu().numpy()/1000
            value=float(policy(pred,va)['gain'].mean());mse=float(((pred[va['eligible']]-va['y'][va['eligible']])**2).mean())
            history.append(dict(epoch=epoch,validation_gain=value,validation_mse=mse))
            if value>best+1e-8:
                best=value;bad=0;selected=epoch
                save_torch(path,dict(state={k:v.detach().cpu() for k,v in model.state_dict().items()},mean=mean,std=std,
                    kind=kind,fold=fold,seed=seed,selected_epoch=epoch,validation_gain=value,trainval_sha=meta['trainval_sha']))
            else:bad+=1
            if bad>=4:break
    write_json(marker,dict(sha=sha256(path),fold=fold,kind=kind,seed=seed,epochs=epoch,selected_epoch=selected,history=history,device=device))
    return marker.name


def interval(delta):
    # 3 training seeds x36 configs x2instances x4populations x2 stages
    recipe=delta.reshape(3,12,3,2,4,2).mean((0,2,3,4,5));rng=np.random.default_rng(20261006)
    boot=recipe[rng.integers(0,12,(5000,12))].mean(1);lo,hi=np.quantile(boot,[.005,.995]);seeds=delta.mean((1,2,3,4))
    return dict(mean=float(delta.mean()),lower=float(lo),upper=float(hi),seed_means=seeds.tolist(),
        passed=bool(delta.mean()>=.0001 and lo>0 and (seeds>0).all()))


def evaluate():
    arrays={m:np.zeros((3,36,2,4,2)) for m in ('fusion','geo','proxy','random','constant','fusion_all','geo_all','proxy_all','fusion_random_candidates')}
    records=[]
    for fold in range(3):
        meta=json.loads((RUN/'DATA.json').read_text())['folds'][fold]
        assert sha256(RUN/f'heldout_{fold}.pt')==meta['heldout_sha']
        data=torch.load(RUN/f'heldout_{fold}.pt',weights_only=False)
        baseline={m:policy(scores,data) for m,scores in [('proxy',data['proxy']),('constant',np.zeros_like(data['proxy']))]}
        random_y=(data['y']*data['eligible']).sum(1)/data['eligible'].sum(1)
        for seed in range(3):
            outputs={m:v['gain'] for m,v in baseline.items()};outputs['random']=random_y*.25
            outputs['proxy_all']=policy(data['proxy'],data,1.)['gain']
            choices={m:dict(choice=v['choice'],mask=v['mask']) for m,v in baseline.items()}
            for kind in ('geo','fusion'):
                path=RUN/f'model_{fold}_{kind}_{seed}.pt';marker=json.loads(path.with_suffix('.json').read_text());assert sha256(path)==marker['sha']
                ck=torch.load(path,weights_only=False);model=net(data[kind].shape[-1]);model.load_state_dict(ck['state']);model.eval()
                x=torch.tensor(np.clip((data[kind]-ck['mean'])/ck['std'],-10,10))
                with torch.no_grad():pred=model(x).squeeze(-1).numpy()/1000
                decision=policy(pred,data);outputs[kind]=decision['gain'];outputs[kind+'_all']=policy(pred,data,1.)['gain']
                choices[kind]=dict(choice=decision['choice'],mask=decision['mask'],predictions=pred)
                if kind=='fusion':outputs['fusion_random_candidates']=np.where(decision['mask'],random_y,0.)
            for j,(f,i,p,step) in enumerate(data['id']):
                for m,v in outputs.items():arrays[m][seed,f,i,p,STEPS.index(int(step))]=v[j]
            records.append(dict(fold=fold,seed=seed,ids=data['id'],choices=choices,outputs=outputs))
    save_torch(RUN/'decisions.pt',records);save_torch(RUN/'effects.pt',arrays)
    effects={'Fusion-Base':interval(arrays['fusion'])}
    for m in ('proxy','random','constant','geo'):effects['Fusion-'+m]=interval(arrays['fusion']-arrays[m])
    primary=('Fusion-Base','Fusion-proxy','Fusion-random','Fusion-constant')
    models=[json.loads(p.read_text()) for p in sorted(RUN.glob('model_*.json'))];assert len(models)==18
    return dict(effects=effects,recognition_passed=all(effects[k]['passed'] for k in primary),fusion_passed=all(v['passed'] for v in effects.values()),
        mean_utilities={m:float(a.mean()) for m,a in arrays.items()},models=models,costs=json.loads((RUN/'DATA.json').read_text()),
        fixed_quota=.25,heldout_states=576,selected_per_seed=144,development_only=True,offline_quota_not_deployable=True)


def main():
    torch.set_num_threads(1);RUN.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
    if (RUN/'COMPLETE.json').exists():print('Already complete; no rerun.');return
    start=time.perf_counter()
    if not (RUN/'identity.json').exists():
        parent.verify();write_json(RUN/'identity.json',dict(version='long_value_v1',sources={p:sha256(ROOT/p) for p in SOURCES},parent_identity=sha256(parent.RUN/'identity.json')))
    verify();contracts();available=parent.parent.available_gib();workers=32 if available>=26 else 24 if available>=22 else 16;assert available>=17
    write_json(RUN/'RESOURCES.json',dict(workers=workers,available_gib=available,threads=1,rollout_device='cpu'))
    if not (RUN/'DATA.json').exists():
        with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn')) as pool:
            jobs=[(f,i,p) for f in range(36) for i in range(2) for p in range(4)]
            write_json(RUN/'STATUS.json',dict(status='collect',completed=0,total=288))
            for n,_ in enumerate(pool.map(collect,jobs),1):write_json(RUN/'STATUS.json',dict(status='collect',completed=n,total=288))
        prepare_data()
    resources=benchmark();device=resources['device'];jobs=[(fold,kind,s,device) for fold in range(3) for kind in ('geo','fusion') for s in range(3)]
    write_json(RUN/'STATUS.json',dict(status='training',device=device,total=18))
    if device=='cpu':
        with ProcessPoolExecutor(max_workers=9,mp_context=mp.get_context('spawn')) as pool:list(pool.map(train,jobs))
    else:
        for job in jobs:train(job)
    write_json(RUN/'MODELS_FROZEN.json',{p.name:sha256(p) for p in sorted(RUN.glob('model_*.pt'))})
    write_json(RUN/'STATUS.json',dict(status='heldout'))
    result=evaluate();result.update(seconds=time.perf_counter()-start,training_resources=resources,all_workers_joined=True)
    write_json(RUN/'COMPLETE.json',result);write_json(OUT/'RESULTS.json',result);write_json(RUN/'STATUS.json',dict(status='complete'))
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
