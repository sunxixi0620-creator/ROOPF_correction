"""Grouped conditional-prior diagnostic, not a new full-search ROOPF model."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
from pathlib import Path
import sys,json,time
import numpy as np
import torch
from torch import nn
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from roopf.revision_tasks import ProceduralTask
from roopf.online_portfolio import build_online
from roopf.experiment_io import sha256,fingerprint,seed_for,write_json,save_torch
RUN=ROOT/'results/fewshot_prior_20260930';OUT=ROOT/'docs/revision/fewshot_prior'
SOURCES=['scripts/fewshot_prior.py','docs/experiments/FEWSHOT_PRIOR_PROTOCOL.md','roopf/revision_tasks.py','roopf/online_portfolio.py','roopf/model.py']
COUNTS=(10,20,40)


class Prior(nn.Module):
    def __init__(self):
        super().__init__()
        self.pair=nn.Sequential(nn.Linear(62,64),nn.SiLU(),nn.Linear(64,32),nn.SiLU())
        self.attention=nn.Linear(32,1)
        self.correction=nn.Sequential(nn.Linear(52,64),nn.SiLU(),nn.Linear(64,1))
        nn.init.zeros_(self.attention.weight);nn.init.zeros_(self.attention.bias)
        nn.init.zeros_(self.correction[-1].weight);nn.init.zeros_(self.correction[-1].bias)
    def forward(self,cx,cy,qx):
        cx=cx/5;qx=qx/5
        c=cx[:,None].expand(-1,qx.shape[1],-1,-1);q=qx[:,:,None].expand(-1,-1,cx.shape[1],-1)
        delta=q-c;dist=delta.square().mean(-1,keepdim=True);y=cy[:,None,:,None].expand(-1,qx.shape[1],-1,-1)
        h=self.pair(torch.cat((c,q,delta,y,dist),-1))
        weight=torch.softmax(self.attention(h).squeeze(-1)-dist.squeeze(-1)/.5,-1)
        pooled=(h*weight[...,None]).sum(2)
        return (weight*cy[:,None]).sum(2)+self.correction(torch.cat((pooled,qx),-1)).squeeze(-1)


def net(fold,seed):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed_for('fewshot_prior_model',fold,seed));return Prior()


def verify():
    identity=json.loads((RUN/'identity.json').read_text())
    for p,h in identity['sources'].items():assert sha256(ROOT/p)==h
    return identity


def partitions(fold):
    test=[r for r in range(12) if r%3==fold];rest=[r for r in range(12) if r not in test]
    return dict(train=rest[:-2],val=rest[-2:],test=test)


def generate():
    if (RUN/'DATA.json').exists():
        meta=json.loads((RUN/'DATA.json').read_text())
        for p,h in meta['files'].items():assert sha256(RUN/p)==h
        return
    calls=0;files={}
    for fold in range(3):
        for split,recipes in partitions(fold).items():
            rows=[];size=32 if split=='train' else 16;queries=64 if split=='train' else 256
            role=f'fewshot_prior_v1_fold{fold}_{split}'
            for recipe in recipes:
                for scale in range(3):
                    f=recipe*3+scale
                    for i in range(size):
                        task=ProceduralTask(f,i,role,dim=20)
                        g=torch.Generator().manual_seed(seed_for('fewshot_points',role,f,i))
                        x=10*torch.rand(40+queries,20,generator=g)-5
                        y=task.calfitness(x[None])[0];calls+=len(x)
                        assert task.points==len(x) and task.diagnostic_points==0
                        perm=torch.randperm(len(x),generator=torch.Generator().manual_seed(seed_for('fewshot_shuffle',role,f,i)))
                        rows.append(dict(cx=x[:40],cy=y[:40],qx=x[40:],qy=y[40:],shuffle_cy=y[perm][:40],shuffle_qy=y[perm][40:],id=torch.tensor([f,i])))
            data={k:torch.stack([r[k] for r in rows]) for k in rows[0]}
            name=f'{fold}_{split}.pt';save_torch(RUN/name,data);files[name]=sha256(RUN/name)
            write_json(RUN/'STATUS.json',dict(status='data',fold=fold,split=split,calls=calls))
    assert calls==435456
    write_json(RUN/'DATA.json',dict(calls=calls,files=files,partitions={str(f):partitions(f) for f in range(3)}))


def standardized(data,n,kind='correct',index=None):
    get=lambda key:data[key] if index is None else data[key][index]
    prefix='shuffle_' if kind=='shuffled' else ''
    cy=get(prefix+'cy')[:,:n];mean=cy.mean(1,keepdim=True);std=cy.std(1,keepdim=True,unbiased=False).clamp_min(1e-6)
    return get('cx')[:,:n],(cy-mean)/std,get('qx'),(get(prefix+'qy')-mean)/std,mean,std


@torch.no_grad()
def predictions(model,data,n,device):
    result=[]
    for index in torch.arange(len(data['cx'])).split(16):
        cx,cy,qx,_,_,_=standardized(data,n,index=index)
        result.append(model(cx.to(device),cy.to(device),qx.to(device)).cpu())
    return torch.cat(result)


@torch.no_grad()
def validation(model,data,device):
    result=[]
    for n in COUNTS:
        pred=predictions(model,data,n,device);target=standardized(data,n)[3]
        result.append(float((pred-target).square().mean()))
    return float(np.mean(result))


def benchmark():
    path=RUN/'RESOURCES.json'
    if path.exists():return json.loads(path.read_text())
    times={}
    for dev in ('cpu','cuda'):
        if dev=='cuda' and not torch.cuda.is_available():continue
        model=net(0,0).to(dev);opt=torch.optim.Adam(model.parameters(),lr=.001)
        with torch.random.fork_rng(devices=[0] if dev=='cuda' else []):
            torch.manual_seed(20261009)
            cx=torch.randn(64,40,20,device=dev);cy=torch.randn(64,40,device=dev);qx=torch.randn(64,64,20,device=dev);qy=torch.randn(64,64,device=dev)
            for it in range(15):
                if it==5:
                    if dev=='cuda':torch.cuda.synchronize()
                    start=time.perf_counter()
                opt.zero_grad();loss=(model(cx,cy,qx)-qy).square().mean();loss.backward();opt.step()
            if dev=='cuda':torch.cuda.synchronize()
            times[dev]=(time.perf_counter()-start)/10
        del model,opt,cx,cy,qx,qy
        if dev=='cuda':torch.cuda.empty_cache()
    result=dict(seconds_per_training_step=times,device=min(times,key=times.get),threads=1,parameters=sum(p.numel() for p in net(0,0).parameters()),synthetic_only=True)
    write_json(path,result);return result


def train(fold,kind,seed,device):
    marker=RUN/f'model_{fold}_{kind}_{seed}.json';path=marker.with_suffix('.pt')
    if marker.exists():assert sha256(path)==json.loads(marker.read_text())['sha'];return
    verify();tr=torch.load(RUN/f'{fold}_train.pt',weights_only=False);va=torch.load(RUN/f'{fold}_val.pt',weights_only=False)
    tr={k:v.to(device) for k,v in tr.items()};model=net(fold,seed).to(device)
    opt=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=.0001)
    g=torch.Generator().manual_seed(seed_for('fewshot_training_batches',fold,seed))
    best=float('inf');selected=bad=0;history=[];start=time.perf_counter()
    for epoch in range(121):
        if epoch:
            model.train();order=torch.randperm(len(tr['cx']),generator=g)
            for index in order.split(64):
                n=COUNTS[int(torch.randint(0,3,(1,),generator=g))]
                cx,cy,qx,qy,_,_=standardized(tr,n,kind,index.to(device))
                opt.zero_grad();loss=(model(cx,cy,qx)-qy).square().mean();assert torch.isfinite(loss)
                loss.backward();opt.step()
        if epoch%5==0:
            model.eval();mse=validation(model,va,device);history.append(dict(epoch=epoch,validation_mse=mse))
            if mse<best-1e-6:
                best=mse;selected=epoch;bad=0
                save_torch(path,dict(state={k:v.detach().cpu() for k,v in model.state_dict().items()},fold=fold,kind=kind,seed=seed,epoch=epoch))
            else:bad+=1
            write_json(RUN/'STATUS.json',dict(status='training',fold=fold,kind=kind,seed=seed,epoch=epoch,best_epoch=selected))
            if bad>=4:break
    write_json(marker,dict(fold=fold,kind=kind,seed=seed,epochs=epoch,selected=selected,validation_mse=best,history=history,sha=sha256(path),seconds=time.perf_counter()-start))


def online_predictions(data):
    surrogate=build_online(20,600).surrogate;task=ProceduralTask(0,0,'fewshot_bounds_only',dim=20)
    result=[]
    for n in COUNTS:
        with torch.no_grad():mu,_=surrogate.predict(data['cx'][:,:n],data['cy'][:,:n],data['qx'],task)
        _,_,_,_,mean,std=standardized(data,n);result.append((mu-mean)/std)
    assert task.points==0 and task.diagnostic_points==0
    return torch.stack(result)


def score(pred,data):
    mse=[];u=[]
    for ci,n in enumerate(COUNTS):
        target=standardized(data,n)[3];mse.append((pred[ci]-target).square().mean(1))
        cy=standardized(data,n)[1];choice=pred[ci].argmin(1)
        improvement=(cy.min(1).values-target[torch.arange(len(target)),choice]).clamp_min(0)
        u.append(improvement/(1+improvement))
    return torch.stack(mse),torch.stack(u)


def effects_from(arrays):
    effects={};rng=np.random.default_rng(20261009);ix=rng.integers(0,12,(10000,12))
    for other in ('untrained','shuffled','online'):
        for metric in ('mse','utility'):
            correct=arrays['correct'][metric];control=arrays[other][metric]
            # arrays are seed x context x recipe; MSE has already been averaged within recipe.
            delta=(control-correct)/np.maximum(control,1e-8) if metric=='mse' else correct-control
            recipes=delta.mean((0,1));boot=recipes[ix].mean(1);lo,hi=np.quantile(boot,[.05/12,1-.05/12]);seeds=delta.mean((1,2))
            threshold=.05 if metric=='mse' else .001
            effects[f'{metric}:Correct-{other}']=dict(mean=float(delta.mean()),lower=float(lo),upper=float(hi),seed_means=seeds.tolist(),
                by_context=delta.mean((0,2)).tolist(),threshold=threshold,passed=bool(delta.mean()>=threshold and lo>0 and (seeds>0).all()))
    return effects


def evaluate(device):
    frozen=json.loads((RUN/'MODELS_FROZEN.json').read_text())
    for p,h in frozen.items():assert sha256(RUN/p)==h
    arrays={m:{metric:np.zeros((3,3,12)) for metric in ('mse','utility')} for m in ('correct','shuffled','untrained','online')}
    for fold in range(3):
        data=torch.load(RUN/f'{fold}_test.pt',weights_only=False);op=online_predictions(data);recipes=(data['id'][:,0]//3).numpy()
        for seed in range(3):
            preds={'online':op}
            for kind in ('correct','shuffled','untrained'):
                model=net(fold,seed).to(device).eval()
                if kind!='untrained':model.load_state_dict(torch.load(RUN/f'model_{fold}_{kind}_{seed}.pt',weights_only=False)['state'])
                preds[kind]=torch.stack([predictions(model,data,n,device) for n in COUNTS])
            metrics={}
            for m,pred in preds.items():
                assert torch.isfinite(pred).all();mse,u=score(pred,data);metrics[m]=dict(mse=mse,utility=u)
                for recipe in partitions(fold)['test']:
                    mask=recipes==recipe
                    arrays[m]['mse'][seed,:,recipe]=mse[:,mask].mean(1).numpy()
                    arrays[m]['utility'][seed,:,recipe]=u[:,mask].mean(1).numpy()
            save_torch(RUN/f'evaluation_{fold}_{seed}.pt',dict(predictions=preds,metrics=metrics,ids=data['id']))
    effects=effects_from(arrays);save_torch(RUN/'aggregates.pt',arrays)
    return dict(effects=effects,passed=all(s['passed'] for s in effects.values()),means={m:{k:float(a.mean()) for k,a in metrics.items()} for m,metrics in arrays.items()},
        models=[json.loads(p.read_text()) for p in sorted(RUN.glob('model_*.json'))],data=json.loads((RUN/'DATA.json').read_text()),
        prediction_and_one_step_only=True,development_only=True,online_controls_reused_not_independent_replicates=True)


def report(result):
    original=json.loads((ROOT/'docs/revision/long_value/VERIFICATION.json').read_text())['original_weights']
    for p,h in original.items():assert sha256(ROOT/p)==h
    rows=['# 少样本先验价值原型','',
        '六项预设门槛全部通过，可进入另行冻结的完整搜索融合实验。' if result['passed'] else '六项联合门槛未通过，不自动扩大模型或开启融合完整搜索。','',
        '| 主比较 | 平均差 | 99.1667%区间 | 三种子效应 | 通过 |','|---|---:|---|---|---|']
    for name,e in result['effects'].items():rows.append(f"| {name} | {e['mean']:+.6f} | [{e['lower']:+.6f},{e['upper']:+.6f}] | {e['seed_means']} | {e['passed']} |")
    rows+=['', 'MSE项为相对降低比例，utility项为有界单次选点效用差，两者不能混用；三个context数10/20/40等权，按12配方聚类。',
        f"模型参数{result['resources']['parameters']}，设备{result['resources']['device']}，最多120epoch，每5轮正确验证集MSE选模，连续4次未改善早停。18模型全部冻结后才读取测试标签。",
        f"源数据及诊断标签实际目标调用{result['data']['calls']:,}，训练和预测0新增调用，总计算耗时{result['seconds']/60:.2f}分钟。",
        '数据为现有生成配方的新实例；三个fold内配方隔离，外层训练集有重叠。没有外部泛化、概率校准或完整搜索终局的结论。',
        'Correct/Shuffled/Untrained使用同容量网络，Shuffled破坏源数据坐标与目标值关系。Online为仅拟合当前context的原在线ridge代理，所有方法共享候选池并使用预测均值选点。',
        '标签来自完整预采样候选池，成本全部计入离线诊断；不能称这些轨迹只花费11/21/41次调用。原解锁模型和原residual未修改。','',
        '| fold | kind | seed | 选中epoch | 实际epoch |','|---|---|---|---:|---:|']
    for m in result['models']:rows.append(f"| {m['fold']} | {m['kind']} | {m['seed']} | {m['selected']} | {m['epochs']} |")
    (OUT/'CONCLUSIONS.zh-CN.md').write_text('\n'.join(rows)+'\n')
    write_json(OUT/'VERIFICATION.json',dict(original_weights=original,identity=verify(),folds=3,models=18,test_tasks=576,
        source_files_checked=True,recipe_disjoint_per_model=True,all_models_frozen_before_test=True,zero_prediction_objective_calls=True))


def main():
    torch.set_num_threads(1);RUN.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
    if (RUN/'COMPLETE.json').exists():print('Already complete; no experiments relaunched.');return
    start=time.perf_counter()
    if not (RUN/'identity.json').exists():write_json(RUN/'identity.json',dict(version='fewshot_prior_v1',sources={p:sha256(ROOT/p) for p in SOURCES}))
    verify();generate();resources=benchmark();device=resources['device']
    for fold in range(3):
        for seed in range(3):
            for kind in ('correct','shuffled'):train(fold,kind,seed,device)
    write_json(RUN/'MODELS_FROZEN.json',{p.name:sha256(p) for p in sorted(RUN.glob('model_*.pt'))})
    assert len(json.loads((RUN/'MODELS_FROZEN.json').read_text()))==18
    write_json(RUN/'STATUS.json',dict(status='heldout'));result=evaluate(device)
    result.update(resources=resources,seconds=time.perf_counter()-start)
    write_json(RUN/'COMPLETE.json',result);write_json(OUT/'RESULTS.json',result);report(result)
    write_json(RUN/'STATUS.json',dict(status='complete'));print(json.dumps(result),flush=True)


if __name__=='__main__':main()
