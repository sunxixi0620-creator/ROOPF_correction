"""Post-hoc audit on archived predictions; zero additional objective evaluations."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
from pathlib import Path
import sys,json,time
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import fewshot_prior as parent
from roopf.experiment_io import sha256,seed_for,write_json,save_torch
RUN=ROOT/'results/radial_prior_audit_20260930';OUT=ROOT/'docs/revision/radial_prior_audit'
SOURCES=['scripts/radial_prior_audit.py','docs/experiments/PRIOR_STRUCTURE_PROTOCOL.md']


def radial(cx,cy,qx,kind):
    cx=cx.double()/5;qx=qx.double()/5;cy=cy.double()
    cr=cx.square().mean(-1,keepdim=True);qr=qx.square().mean(-1,keepdim=True)
    if kind=='fixed':return ((qr-cr.mean(1,keepdim=True))/cr.std(1,keepdim=True,unbiased=False).clamp_min(.01)).squeeze(-1).float()
    phi=lambda x,r:torch.cat((torch.ones_like(r),x,r),-1) if kind=='linear' else torch.cat((torch.ones_like(r),r),-1)
    a,b=phi(cx,cr),phi(qx,qr);d=a.shape[-1]
    ridge=torch.eye(d,dtype=a.dtype)*.01;ridge[0,0]=1e-6
    beta=torch.linalg.solve(a.transpose(1,2)@a+ridge,a.transpose(1,2)@cy[:,:,None])
    return (b@beta).squeeze(-1).float()


def main():
    torch.set_num_threads(1);RUN.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
    if (RUN/'COMPLETE.json').exists():print('Already complete; no rerun.');return
    start=time.perf_counter();parent.verify()
    identity=dict(version='radial_prior_audit_v1',sources={p:sha256(ROOT/p) for p in SOURCES},parent_identity=sha256(parent.RUN/'identity.json'),
        models=json.loads((parent.RUN/'MODELS_FROZEN.json').read_text()),data=json.loads((parent.RUN/'DATA.json').read_text())['files'])
    write_json(RUN/'identity.json',identity)
    for p,h in identity['models'].items():assert sha256(parent.RUN/p)==h
    for p,h in identity['data'].items():assert sha256(parent.RUN/p)==h
    names=('correct','permuted_context','fixed_radial','fitted_radial','radial_linear','online')
    arrays={m:{metric:np.zeros((3,3,12)) for metric in ('mse','utility')} for m in names};r2=np.zeros((3,3,12))
    device='cuda' if torch.cuda.is_available() else 'cpu'
    for fold in range(3):
        data=torch.load(parent.RUN/f'{fold}_test.pt',weights_only=False);recipes=data['id'][:,0]//3
        controls={m:[] for m in ('fixed_radial','fitted_radial','radial_linear')};permutations={}
        for n in parent.COUNTS:
            cx,cy,qx,_,_,_=parent.standardized(data,n)
            for m,k in [('fixed_radial','fixed'),('fitted_radial','fit'),('radial_linear','linear')]:controls[m].append(radial(cx,cy,qx,k))
            permutations[n]=torch.stack([torch.randperm(n,generator=torch.Generator().manual_seed(seed_for('radial_audit_context',fold,int(f),int(i),n))) for f,i in data['id']])
        for seed in range(3):
            rec=torch.load(parent.RUN/f'evaluation_{fold}_{seed}.pt',weights_only=False)
            preds={m:torch.stack(v) for m,v in controls.items()};preds.update(correct=rec['predictions']['correct'],online=rec['predictions']['online'])
            model=parent.net(fold,seed).to(device).eval();model.load_state_dict(torch.load(parent.RUN/f'model_{fold}_correct_{seed}.pt',weights_only=False)['state'])
            shuffled=[]
            for n in parent.COUNTS:
                cx,cy,qx,_,_,_=parent.standardized(data,n);cy=cy.gather(1,permutations[n]);batches=[]
                with torch.no_grad():
                    for ix in torch.arange(len(cx)).split(16):batches.append(model(cx[ix].to(device),cy[ix].to(device),qx[ix].to(device)).cpu())
                shuffled.append(torch.cat(batches))
            preds['permuted_context']=torch.stack(shuffled)
            scores={}
            for m,pred in preds.items():
                mse,u=parent.score(pred,data);scores[m]=dict(mse=mse,utility=u)
                for recipe in parent.partitions(fold)['test']:
                    mask=recipes==recipe
                    arrays[m]['mse'][seed,:,recipe]=mse[:,mask].mean(1).numpy()
                    arrays[m]['utility'][seed,:,recipe]=u[:,mask].mean(1).numpy()
            radius=(data['qx']/5).square().mean(-1);radius=radius-radius.mean(1,keepdim=True)
            pred=preds['correct'];centered=pred-pred.mean(2,keepdim=True)
            corr2=(centered*radius[None]).sum(2).square()/(centered.square().sum(2)*radius.square().sum(1)[None]).clamp_min(1e-12)
            for recipe in parent.partitions(fold)['test']:r2[seed,:,recipe]=corr2[:,recipes==recipe].mean(1).numpy()
            save_torch(RUN/f'fold{fold}_seed{seed}.pt',dict(predictions=preds,metrics=scores,radius_r2=corr2,context_permutations=permutations))
    save_torch(RUN/'aggregates.pt',dict(metrics=arrays,radius_r2=r2))
    result=dict(means={m:{k:float(a.mean()) for k,a in v.items()} for m,v in arrays.items()},
        by_context={m:{k:a.mean((0,2)).tolist() for k,a in v.items()} for m,v in arrays.items()},
        radial_explained_prediction_variance=float(r2.mean()),radial_explained_by_context=r2.mean((0,2)).tolist(),
        correct_minus_permuted_utility=float((arrays['correct']['utility']-arrays['permuted_context']['utility']).mean()),
        objective_calls=0,posthoc_only=True,seconds=time.perf_counter()-start,device=device)
    write_json(RUN/'COMPLETE.json',result);write_json(OUT/'RESULTS.json',result)
    rows=['# 已有预测先验的径向偏置审查','', '这是旧测试数据上的事后机制诊断，没有新增目标查询，不代替独立验证。','',
        '| 方法 | 标准化MSE | 单次选点效用 |','|---|---:|---:|']
    for m,v in result['means'].items():rows.append(f"| {m} | {v['mse']:.6f} | {v['utility']:.6f} |")
    rows+=['',f"候选预测方差中，常数+候选半径的解释率平均为{result['radial_explained_prediction_variance']:.2%}；它不是因果收益贡献率。",
        f"保持context的y数值不变，仅打乱其坐标对应关系，Correct效用相对打乱版本差为{result['correct_minus_permuted_utility']:+.6f}。",
        '径向+线性回归只使用已观测context，未使用候选标签拟合。所有方法在相同候选池按预测均值选点。',
        '不据此重调旧模型；下一项检验共享子空间、不同最优位置以及等半径候选池，区分可迁移方向结构与一般中心偏好。']
    (OUT/'CONCLUSIONS.zh-CN.md').write_text('\n'.join(rows)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
