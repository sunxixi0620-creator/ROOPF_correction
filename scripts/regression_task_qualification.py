"""Task qualification only. Does not run or rank fusion optimizers."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse,hashlib,io,json,math,time,warnings,zipfile,sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import pandas as pd
import requests,sklearn,scipy
from scipy.stats import qmc,spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'results/regression_qualification_20260930';OUT=ROOT/'docs/revision/regression_qualification'
DATA={'concrete':(165,'source'),'energy':(242,'source'),'airfoil':(291,'source'),'yacht':(243,'development'),'auto':(9,'development'),'wine_red':(186,'development')}
SOURCE=('scripts/regression_task_qualification.py','docs/experiments/REGRESSION_TASK_QUALIFICATION_PROTOCOL.md')
SEALED={'real_estate':477,'computer_hardware':29}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rng(*r):return np.random.default_rng(int(hashlib.sha256(json.dumps(r).encode()).hexdigest()[:15],16))
def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);temp=p.with_suffix('.tmp');temp.write_text(json.dumps(x,indent=2,allow_nan=False,default=lambda v:v.item() if isinstance(v,np.generic) else str(v)));temp.replace(p)
def prepare():
    RUN.mkdir(parents=True,exist_ok=True)
    if (RUN/'identity.json').exists():check();return
    provenance={};tasks=[]
    for name,(number,role) in DATA.items():
        url=f'https://archive.ics.uci.edu/static/public/{number}/data.csv' if name!='yacht' else 'https://archive.ics.uci.edu/static/public/243/yacht+hydrodynamics.zip'
        path=RUN/f'{name}.raw'
        if not path.exists():
            response=requests.get(url,timeout=40);response.raise_for_status();path.write_bytes(response.content)
        if name=='yacht':
            with zipfile.ZipFile(path) as z:
                member=next(n for n in z.namelist() if n.endswith('yacht_hydrodynamics.data'));df=pd.read_csv(io.BytesIO(z.read(member)),sep=r'\s+',header=None)
            features=list(df.columns[:-1]);target=df.columns[-1]
        else:
            df=pd.read_csv(path,na_values=['?'])
            if name=='energy':features=[f'X{i}' for i in range(1,9)];target='Y1'
            elif name=='auto':features=['cylinders','displacement','horsepower','weight','acceleration','model_year','origin'];target='mpg'
            elif name=='wine_red':
                assert set(df.color.unique())=={'red','white'};df=df[df.color=='red'];target='quality';features=[c for c in df if c not in ('quality','color')]
            else:features=list(df.columns[:-1]);target=df.columns[-1]
        before=len(df);df=df[features+[target]].apply(pd.to_numeric,errors='raise');missing=int(df.isna().any(axis=1).sum());df=df.dropna();duplicates=int(df.duplicated().sum());df=df.drop_duplicates()
        x=df[features].to_numpy(float);y=df[target].to_numpy(float)
        if name=='auto':x=np.column_stack([x[:,:-1]]+[x[:,-1]==k for k in (1,2,3)])
        assert np.isfinite(x).all() and np.isfinite(y).all();np.savez_compressed(RUN/f'{name}.npz',x=x,y=y)
        _,groups=np.unique(x,axis=0,return_inverse=True)
        for seed in range(3):
            order=rng('group_split',name,seed).permutation(int(groups.max())+1);selected=order[:int(.6*len(order))];mask=np.isin(groups,selected)
            tasks.append(dict(name=name,role=role,seed=seed,train=np.where(mask)[0].tolist(),val=np.where(~mask)[0].tolist()))
        provenance[name]=dict(url=url,raw_sha256=sha(path),data_sha256=sha(RUN/f'{name}.npz'),role=role,rows_before=before,missing_removed=missing,duplicates_removed=duplicates,rows=len(y),features=features,target=target,model_feature_count=x.shape[1],license='CC BY 4.0',uci_id=number)
    grid=qmc.Sobol(3,scramble=True,seed=270930).random_base2(7);np.save(RUN/'grid.npy',grid);write(RUN/'tasks.json',tasks);write(RUN/'DATA_PROVENANCE.json',provenance)
    write(RUN/'identity.json',dict(frozen=time.time(),source={p:sha(ROOT/p) for p in SOURCE},data=provenance,grid=sha(RUN/'grid.npy'),tasks=sha(RUN/'tasks.json'),
        sealed_not_downloaded=SEALED,environment=dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__,pandas=pd.__version__),
        checkpoints={p:sha(ROOT/p) for p in ('checkpoints/anchor_policy_d10.pt','checkpoints/residual_selector_generated36_d10.pt')}))
    print('Prepared six datasets, frozen identity, no SVR fits; confirmation not downloaded.',flush=True)
def check():
    z=json.loads((RUN/'identity.json').read_text())
    for p,h in {**z['source'],**z['checkpoints']}.items():assert sha(ROOT/p)==h,p
    for name,meta in z['data'].items():
        assert sha(RUN/f'{name}.raw')==meta['raw_sha256'] and sha(RUN/f'{name}.npz')==meta['data_sha256']
    assert sha(RUN/'grid.npy')==z['grid'] and sha(RUN/'tasks.json')==z['tasks']
    for name in SEALED:assert not list(RUN.glob(f'{name}*'))
    return z
def fit_task(t):
    name=f"{t['name']}_{t['seed']}";p=RUN/'tables'/f'{name}.npz';m=p.with_suffix('.json')
    if m.exists():assert sha(p)==json.loads(m.read_text())['sha256'];return
    d=np.load(RUN/f"{t['name']}.npz");a,b=np.array(t['train']),np.array(t['val']);x,y=d['x'],d['y'];assert not set(a)&set(b)
    assert not set(map(tuple,x[a]))&set(map(tuple,x[b]))
    scaler=StandardScaler().fit(x[a]);xt=scaler.transform(x[a]);xv=scaler.transform(x[b]);ym=y[a].mean();ys=y[a].std();assert ys>0
    yt=(y[a]-ym)/ys;yv=(y[b]-ym)/ys;grid=np.load(RUN/'grid.npy');loss=[];seconds=[];warn=[]
    def evaluate(point):
        u,v,w=point;model=SVR(C=10**(-2+5*u),gamma=10**(-4+5*v),epsilon=10**(-3+3*w),tol=.001,max_iter=-1)
        tick=time.perf_counter()
        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter('always');model.fit(xt,yt);pred=model.predict(xv)
        return float(np.sqrt(np.mean((pred-yv)**2))),time.perf_counter()-tick,len(records)
    for point in grid:
        value,elapsed,n=evaluate(point);loss.append(value);seconds.append(elapsed);warn.append(n)
    repeat=[evaluate(grid[i]) for i in (0,63,127)]
    p.parent.mkdir(exist_ok=True);np.savez_compressed(p,loss=loss,seconds=seconds,warnings=warn,repeat_loss=[z[0] for z in repeat],repeat_seconds=[z[1] for z in repeat],repeat_warnings=[z[2] for z in repeat],xmean=scaler.mean_,xscale=scaler.scale_,ymean=ym,yscale=ys)
    write(m,dict(sha256=sha(p),fits=131,task=name,seconds=sum(seconds)+sum(z[1] for z in repeat)))
def run(workers):
    check();ts=json.loads((RUN/'tasks.json').read_text());start=time.time()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        fs=[pool.submit(fit_task,t) for t in ts]
        for n,f in enumerate(as_completed(fs),1):f.result();print(f'tasks {n}/18, elapsed {time.time()-start:.1f}s',flush=True)
    write(RUN/'TIMING.json',dict(wall_seconds=time.time()-start,workers=workers,fits=2358))
def audit():
    identity=check();tables={};rows=[];repeats=[];warning_total=0;ts=json.loads((RUN/'tasks.json').read_text())
    for name,(_,role) in DATA.items():
        values=[]
        for s in range(3):
            p=RUN/'tables'/f'{name}_{s}.npz';assert sha(p)==json.loads(p.with_suffix('.json').read_text())['sha256'];z=np.load(p);v=z['loss'];assert len(v)==128 and np.isfinite(v).all()
            repeats.extend(abs(v[[0,63,127]]-z['repeat_loss']).tolist());warning_total+=int(z['warnings'].sum()+z['repeat_warnings'].sum())
            t=next(t for t in ts if t['name']==name and t['seed']==s);d=np.load(RUN/f'{name}.npz');a=t['train'];b=t['val'];assert not set(map(tuple,d['x'][a]))&set(map(tuple,d['x'][b]))
            scale=StandardScaler().fit(d['x'][a]);assert np.array_equal(scale.mean_,z['xmean']) and np.array_equal(scale.scale_,z['xscale'])
            assert np.isclose(d['y'][a].mean(),z['ymean']) and np.isclose(d['y'][a].std(),z['yscale'])
            values.append(v)
        values=np.array(values);tables[name]=values;good=values<=values.min(axis=1,keepdims=True)+.05;n=good.sum(1)
        prob=[1-math.comb(128-int(k),4)/math.comb(128,4) if k<=124 else 1. for k in n]
        corr=[spearmanr(values[i],values[j]).statistic for i,j in ((0,1),(0,2),(1,2))]
        jac=[(good[i]&good[j]).sum()/(good[i]|good[j]).sum() for i,j in ((0,1),(0,2),(1,2))]
        rows.append(dict(dataset=name,role=role,mean_good_fraction=float(good.mean()),random_four_success=float(np.mean(prob)),mean_split_spearman=float(np.mean(corr)),mean_good_jaccard=float(np.mean(jac)),mean_loss_spread_across_splits=float(values.std(0).mean()),mean_minimum=float(values.min(1).mean())))
    prior=np.mean([v.mean(0) for n,v in tables.items() if DATA[n][1]=='source'],axis=0);source_best=int(prior.argmin());dev={n:v.mean(0) for n,v in tables.items() if DATA[n][1]=='development'}
    good_dev=np.array([v<=v.min()+.05 for v in dev.values()]);coverage=good_dev.sum(0);max_cover=int(coverage.max())
    for row in rows:
        row['difficulty_pass']=bool(row['mean_good_fraction']<=.1 and row['random_four_success']<=.35)
        if row['role']=='development':
            v=dev[row['dataset']];row.update(source_spearman=float(spearmanr(prior,v).statistic),source_best_excess=float(v[source_best]-v.min()),source_best_meets_target=bool(v[source_best]<=v.min()+.05))
    gates=dict(source_difficulty=sum(r['difficulty_pass'] for r in rows if r['role']=='source')>=2,development_difficulty=sum(r['difficulty_pass'] for r in rows if r['role']=='development')>=2,
        heterogeneous=max_cover<=2,source_relation=sum(r['source_spearman']>=.2 for r in rows if r['role']=='development')>=2,
        stable_splits=sum(r['mean_split_spearman']>=.5 for r in rows if r['role']=='development')>=2,integrity=max(repeats)<=1e-10 and warning_total==0)
    result=dict(rows=rows,gates=gates,joint_pass=all(gates.values()),max_single_config_coverage=max_cover,covering_config_indices=np.where(coverage==max_cover)[0].tolist(),source_best_index=source_best,
        fits=2358,repeat_max_error=max(repeats),warnings=warning_total,confirmation_evaluations=0,optimizer_trajectories=0,identity_sha256=sha(RUN/'identity.json'))
    write(OUT/'RESULTS.json',result);write(OUT/'VERIFICATION.json',dict(source_and_data_hashes=True,group_separation=True,training_only_scaling=True,repeat_fits=54,repeat_max_error=max(repeats),warnings=warning_total,original_checkpoints_unchanged=True,confirmation_not_downloaded=True))
    np.savez_compressed(OUT/'source_prior.npz',prior=prior)
    lines=['# 跨数据集回归任务资格审查','',f"联合资格门槛：{'通过' if result['joint_pass'] else '未通过'}。本轮没有运行融合算法，不构成任何优化器胜负证据。",'',
        '|数据集|用途|好配置比例|随机4点成功率|跨划分排序相关|源先验相关|难度门槛|','|---|---|---:|---:|---:|---:|---|']
    for r in rows:lines.append(f"|{r['dataset']}|{r['role']}|{r['mean_good_fraction']:.1%}|{r['random_four_success']:.1%}|{r['mean_split_spearman']:.3f}|{r.get('source_spearman',float('nan')):.3f}|{'通过' if r['difficulty_pass'] else '未通过'}|")
    lines+=['','源行不计算源先验相关，避免自包含相关冒充迁移。好配置定义为距当前有限表最优不超过0.05个训练目标标准差的RMSE。','',f'单个配置最多同时满足 {max_cover}/3 个开发数据集；这是事后全表资格诊断，不向任何优化器提供。','', '逐项条件：']
    lines += [f"- {k}: {'通过' if v else '未通过'}" for k,v in gates.items()]
    lines += ['',f'实际拟合2358次（主表2304+重复54）；重复最大误差{max(repeats):.3g}；警告{warning_total}。原权重未变，确认候选数据集未下载、未评估。',
        '数据划分波动不等于同一目标的随机噪声。仅三个开发数据集，不进行总体显著性或跨领域泛化声明。',
        '所有条件通过只允许另行预登记开发比较；任一失败停止本集合准入，不调整网格/容差/划分，不按融合是否获胜替换数据集。']
    (OUT/'CONCLUSIONS.zh-CN.md').write_text('\n'.join(lines)+'\n');print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','run','audit']);p.add_argument('--workers',type=int,default=24);a=p.parse_args()
    if a.phase=='prepare':prepare()
    elif a.phase=='run':run(a.workers)
    else:audit()
