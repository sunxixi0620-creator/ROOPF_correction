"""Frozen real-data sample-efficiency pilot; policies never receive target tables."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'
import argparse, hashlib, itertools, json, time, warnings, sys, platform
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import scipy
from scipy.linalg import cho_factor, cho_solve
from scipy.special import ndtr
from scipy.spatial.distance import cdist
import sklearn
from sklearn.datasets import load_digits
from sklearn.svm import SVC
from sklearn.metrics import balanced_accuracy_score
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'results/real_hpo_prior_20260930'
OUT=ROOT/'docs/revision/real_hpo_prior'
METHODS=('A','O','F','Fixed','Bad','Random','GP')
SOURCES=('scripts/real_hpo_prior.py','docs/experiments/REAL_HPO_PRIOR_PROTOCOL.md')
AX=np.linspace(0,1,16)
X=np.array(list(itertools.product(AX,AX)))
D=cdist(X,X)/.25
K=.25**2*(1+np.sqrt(5)*D+5*D**2/3)*np.exp(-np.sqrt(5)*D)

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rng(*role):return np.random.default_rng(int(hashlib.sha256(json.dumps(role).encode()).hexdigest()[:15],16))
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(v,indent=2,allow_nan=False));tmp.replace(p)
def freeze():
    RUN.mkdir(parents=True,exist_ok=True)
    ident=RUN/'identity.json'
    if ident.exists():check();return
    d=load_digits();np.savez_compressed(RUN/'dataset.npz',x=d.data/16,y=d.target)
    pools={}
    for c in range(10):
        a=rng('images',c).permutation(np.where(d.target==c)[0]);n=len(a)//2
        pools[str(c)]={'source':a[:n].tolist(),'target':a[n:].tolist()}
    pairs=list(itertools.combinations(range(10),2));order=rng('pairs').permutation(len(pairs))
    tasks=[]
    for pos,i in enumerate(order):
        role='source' if pos<30 else 'target';pair=pairs[i]
        for split in range(2 if role=='source' else 3):
            train=[];val=[]
            for c in pair:
                a=rng('split',role,pair,split,c).permutation(pools[str(c)][role]);n=int(.6*len(a))
                train.extend(a[:n].tolist());val.extend(a[n:].tolist())
            tasks.append(dict(id=f'{role}_{pair[0]}{pair[1]}_{split}',role=role,pair=list(pair),split=split,train=train,val=val))
    write(RUN/'tasks.json',tasks)
    write(ident,dict(created=time.time(),sources={p:sha(ROOT/p) for p in SOURCES},dataset=sha(RUN/'dataset.npz'),tasks=sha(RUN/'tasks.json'),
        environment=dict(python=sys.version,sklearn=sklearn.__version__,numpy=np.__version__,scipy=scipy.__version__,platform=platform.platform()),
        original={p:sha(ROOT/p) for p in ('checkpoints/anchor_policy_d10.pt','checkpoints/residual_selector_generated36_d10.pt')}))
    print('Frozen protocol, source code, data and task split.',flush=True)

def check():
    z=json.loads((RUN/'identity.json').read_text())
    for p,h in z['sources'].items():assert sha(ROOT/p)==h,p
    for p,h in z['original'].items():assert sha(ROOT/p)==h,p
    assert sha(RUN/'dataset.npz')==z['dataset'] and sha(RUN/'tasks.json')==z['tasks']
    return z

def tasks():return json.loads((RUN/'tasks.json').read_text())
def table(task):
    path=RUN/'tables'/f"{task['id']}.npz";meta=path.with_suffix('.json')
    if meta.exists():assert sha(path)==json.loads(meta.read_text())['sha256'];return
    data=np.load(RUN/'dataset.npz');a,b=task['train'],task['val'];errors=[];elapsed=[]
    for u,v in X:
        start=time.perf_counter();model=SVC(C=10**(-3+6*u),gamma=10**(-5+6*v),class_weight='balanced',tol=.001,max_iter=-1)
        model.fit(data['x'][a],data['y'][a]);errors.append(1-balanced_accuracy_score(data['y'][b],model.predict(data['x'][b])))
        elapsed.append(time.perf_counter()-start)
    path.parent.mkdir(exist_ok=True);np.savez_compressed(path,error=errors,seconds=elapsed)
    write(meta,dict(sha256=sha(path),task=task['id'],fits=256,seconds=sum(elapsed)))

def tables(role,workers):
    check()
    if role=='target':
        m=json.loads((RUN/'prior.json').read_text());assert sha(RUN/'prior.npz')==m['sha256']
        assert (RUN/'CONTRACTS.json').exists()
    chosen=[t for t in tasks() if t['role']==role];start=time.time()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        fs=[pool.submit(table,t) for t in chosen]
        for n,f in enumerate(as_completed(fs),1):
            f.result()
            if n%5==0 or n==len(fs):print(f'{role} tables {n}/{len(fs)} elapsed {time.time()-start:.1f}s',flush=True)
    write(RUN/f'{role}_timing.json',dict(wall_seconds=time.time()-start,workers=workers,table_count=len(chosen)))
    if role=='source':
        vals=[np.load(RUN/'tables'/f"{t['id']}.npz")['error'] for t in chosen]
        p=np.mean(vals,axis=0);bad=p[rng('negative_prior').permutation(256)]
        np.savez_compressed(RUN/'prior.npz',p=p,bad=bad)
        write(RUN/'prior.json',dict(sha256=sha(RUN/'prior.npz'),frozen=time.time(),source_tables={t['id']:sha(RUN/'tables'/f"{t['id']}.npz") for t in chosen},neural_epochs=0))

def ordinary(indices,y,prior,weights):
    indices=np.asarray(indices);n=len(indices);cross=K[:,indices]
    inv=cho_solve(cho_factor(K[np.ix_(indices,indices)]+1e-4*np.eye(n),lower=True),np.eye(n))
    v=inv@np.ones(n);den=v.sum();P=inv-np.outer(v,v)/den
    residual=np.asarray(y)[:,None]-prior[indices,None]*np.asarray(weights)[None,:]
    loo=(P@residual)/np.diag(P)[:,None];scores=np.mean(loo**2,axis=0)
    j=int(np.argmin(scores));w=float(weights[j]);z=residual[:,j];mean=v@z/den
    mu=w*prior+mean+cross@(inv@(z-mean))
    var=np.diag(K)-np.sum((cross@inv)*cross,axis=1)+(1-cross@v)**2/den
    return mu,np.sqrt(np.maximum(var,1e-14)),w,scores,loo[:,j]

def choose(method,indices,observed,prior,order,force=None):
    """Only observed target labels enter this API; no table, optimum or threshold."""
    excluded=np.zeros(256,dtype=bool);excluded[indices]=True
    if method in ('A','Random'):
        ranking=np.argsort(prior['p'],kind='stable') if method=='A' else order
        return int(next(i for i in ranking if not excluded[i])),dict(alpha=None,warnings=0)
    warning_count=0
    if method=='GP':
        kernel=ConstantKernel(1,(.01,100))*Matern([.25,.25],(.05,2),nu=2.5)
        gp=GaussianProcessRegressor(kernel=kernel,alpha=1e-4,normalize_y=True,n_restarts_optimizer=0)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always');gp.fit(X[indices],observed);mu,sd=gp.predict(X,return_std=True)
        warning_count=len(caught);w=None;scores=[]
    else:
        weights=[0.] if method=='O' else ([1.] if method=='Fixed' else [0.,.25,.5,1.])
        if force is not None:weights=[float(force)]
        mu,sd,w,scores,_=ordinary(indices,observed,prior['bad' if method=='Bad' else 'p'],weights)
    delta=min(observed)-mu;z=delta/np.maximum(sd,1e-14)
    ei=delta*ndtr(z)+sd*np.exp(-.5*z*z)/np.sqrt(2*np.pi);ei[excluded]=-np.inf
    return int(np.argmax(ei)),dict(alpha=w,loo=np.asarray(scores).tolist(),warnings=warning_count)

def contracts():
    check();p=dict(np.load(RUN/'prior.npz'));idx=rng('contract').choice(256,9,replace=False);y=rng('contract_y').uniform(0,.5,9)
    _,_,_,_,loo=ordinary(idx,y,p['p'],[.5]);direct=[]
    for i in range(9):
        mask=np.arange(9)!=i;mu,*_=ordinary(idx[mask],y[mask],p['p'],[.5]);direct.append(y[i]-mu[idx[i]])
    assert np.allclose(loo,direct,atol=1e-10,rtol=1e-10)
    order=rng('contract_order').permutation(256)
    for w,method in [(0,'O'),(1,'Fixed')]:
        assert choose('F',idx.tolist(),y.tolist(),p,order,force=w)[0]==choose(method,idx.tolist(),y.tolist(),p,order)[0]
    fake=rng('poison').uniform(0,1,256);fake[idx]=y
    answer=choose('F',idx.tolist(),fake[idx].tolist(),p,order)[0];fake[~np.isin(np.arange(256),idx)]=1e9
    assert answer==choose('F',idx.tolist(),fake[idx].tolist(),p,order)[0]
    groups={r:[t for t in tasks() if t['role']==r] for r in ('source','target')}
    images={r:set(i for t in ts for i in t['train']+t['val']) for r,ts in groups.items()}
    pairs={r:set(tuple(t['pair']) for t in ts) for r,ts in groups.items()}
    assert not images['source']&images['target'] and not pairs['source']&pairs['target']
    assert all(not set(t['train'])&set(t['val']) for t in tasks())
    write(RUN/'CONTRACTS.json',dict(loo_max_error=float(np.max(np.abs(loo-direct))),forced_decisions=True,hidden_labels_invariant=True,image_and_pair_disjoint=True,extra_objective_fits=0))

def rollout(job):
    task,seed,method=job;name=f"{task['id']}_s{seed}_{method}";path=RUN/'traces'/f'{name}.json'
    if path.exists():return
    values=np.load(RUN/'tables'/f"{task['id']}.npz")['error'];prior=dict(np.load(RUN/'prior.npz'))
    order=rng('initial',task['id'],seed).permutation(256);ids=order[:4].tolist();ys=values[ids].tolist();decisions=[]
    start=time.perf_counter()
    while len(ids)<32:
        tick=time.perf_counter();i,info=choose(method,ids,ys,prior,order)
        assert i not in ids;info.update(seconds=time.perf_counter()-tick,nfe=len(ids)+1,index=i)
        # The oracle is accessed only after choose returns its committed index.
        ids.append(i);ys.append(float(values[i]));decisions.append(info)
    write(path,dict(task=task['id'],pair=task['pair'],split=task['split'],seed=seed,method=method,indices=ids,observed=ys,
        decisions=decisions,seconds=time.perf_counter()-start))

def search(workers):
    check();assert sha(RUN/'prior.npz')==json.loads((RUN/'prior.json').read_text())['sha256']
    jobs=[(t,s,m) for t in tasks() if t['role']=='target' for s in range(3) for m in METHODS]
    start=time.time()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        fs=[pool.submit(rollout,j) for j in jobs]
        for n,f in enumerate(as_completed(fs),1):
            f.result()
            if n%100==0 or n==len(fs):print(f'traces {n}/{len(fs)} elapsed {time.time()-start:.1f}s',flush=True)
    write(RUN/'search_timing.json',dict(wall_seconds=time.time()-start,workers=workers,trajectories=len(jobs),lookups=32*len(jobs)))

def verify_trace(path):
    z=json.loads(Path(path).read_text());y=np.load(RUN/'tables'/f"{z['task']}.npz")['error'];p=dict(np.load(RUN/'prior.npz'))
    order=rng('initial',z['task'],z['seed']).permutation(256)
    assert z['indices'][:4]==order[:4].tolist() and len(z['indices'])==len(set(z['indices']))==32
    assert np.array_equal(y[z['indices']],z['observed'])
    for n in range(4,32):
        i,info=choose(z['method'],z['indices'][:n],z['observed'][:n],p,order)
        old=z['decisions'][n-4];assert i==z['indices'][n] and info['alpha']==old['alpha']
        if 'loo' in info:assert np.allclose(info['loo'],old['loo'],rtol=1e-12,atol=1e-12)
    return 28

def verify(workers):
    check();contracts();meta=json.loads((RUN/'prior.json').read_text())
    assert sha(RUN/'prior.npz')==meta['sha256']
    for t in tasks():
        path=RUN/'tables'/f"{t['id']}.npz";assert sha(path)==json.loads(path.with_suffix('.json').read_text())['sha256']
    p=dict(np.load(RUN/'prior.npz'));source=[np.load(RUN/'tables'/f"{t['id']}.npz")['error'] for t in tasks() if t['role']=='source']
    assert np.array_equal(p['p'],np.mean(source,axis=0)) and np.array_equal(p['bad'],p['p'][rng('negative_prior').permutation(256)])
    traces=sorted((RUN/'traces').glob('*.json'));assert len(traces)==945
    with ProcessPoolExecutor(max_workers=workers) as pool:
        count=sum(pool.map(verify_trace,traces))
    write(OUT/'VERIFICATION.json',dict(verified=True,replayed_decisions=count,traces=len(traces),original_checkpoints_unchanged=True,extra_objective_fits=0,
        source_prior_recomputed=True,files={str(f.relative_to(RUN)):sha(f) for f in RUN.rglob('*') if f.is_file()}))
    print(f'Verified {count} decisions, no new objective evaluations.',flush=True)

def report():
    check();rows=[];curves={m:[] for m in METHODS};alphas=[]
    for path in sorted((RUN/'traces').glob('*.json')):
        z=json.loads(path.read_text());table_y=np.load(RUN/'tables'/f"{z['task']}.npz")['error']
        best=np.minimum.accumulate(z['observed']);regret=(best-table_y.min())/max(np.ptp(table_y),1e-12)
        hit=np.where(regret<=.05)[0]
        rows.append(dict(task=z['task'],pair=''.join(map(str,z['pair'])),split=z['split'],seed=z['seed'],method=z['method'],auc=float(regret[3:].mean()),
            hit=int(hit[0]+1) if len(hit) else 33,success=bool(len(hit)),final_error=float(best[-1]),initial_success=bool(np.any(regret[:4]<=.05)),
            seconds=z['seconds'],warnings=sum(d['warnings'] for d in z['decisions'])))
        curves[z['method']].append(regret.tolist())
        if z['method']=='F':alphas.extend(d['alpha'] for d in z['decisions'])
    assert len(rows)==945
    pairs=sorted(set(r['pair'] for r in rows));resamples=rng('bootstrap').integers(0,len(pairs),(10000,len(pairs)))
    contrasts=[]
    for metric,control,threshold in [('auc',c,.005) for c in ('A','O','GP','Bad')]+[('hit',c,1.) for c in ('A','O')]:
        differences=[];seeds=[]
        for pair in pairs:
            subset=[r for r in rows if r['pair']==pair]
            differences.append(np.mean([r[metric] for r in subset if r['method']==control])-np.mean([r[metric] for r in subset if r['method']=='F']))
        for s in range(3):
            subset=[r for r in rows if r['seed']==s]
            seeds.append(float(np.mean([r[metric] for r in subset if r['method']==control])-np.mean([r[metric] for r in subset if r['method']=='F'])))
        diff=np.array(differences);lo,hi=np.quantile(diff[resamples].mean(1),[.05/12,1-.05/12]);mean=float(diff.mean())
        contrasts.append(dict(metric=metric,control=control,improvement=mean,ci=[float(lo),float(hi)],threshold=threshold,seed_effects=seeds,
            passed=bool(mean>=threshold and lo>0 and min(seeds)>0)))
    methods={m:{k:float(np.mean([r[k] for r in rows if r['method']==m])) for k in ('auc','hit','success','final_error','initial_success','seconds','warnings')} for m in METHODS}
    result=dict(methods=methods,contrasts=contrasts,joint_pass=all(c['passed'] for c in contrasts),fits=26880,table_lookups=30240,trajectories=945,
        mean_alpha=float(np.mean(alphas)),alpha_frequencies={str(a):float(np.mean(np.array(alphas)==a)) for a in (0,.25,.5,1)},
        source_fit_seconds=sum(json.loads(p.read_text())['seconds'] for p in (RUN/'tables').glob('source*.json')),
        target_fit_seconds=sum(json.loads(p.read_text())['seconds'] for p in (RUN/'tables').glob('target*.json')),
        curves={m:np.mean(v,axis=0).tolist() for m,v in curves.items()},identity=sha(RUN/'identity.json'))
    write(OUT/'RESULTS.json',result);write(OUT/'rows.json',rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8,4.5))
    for m,c in result['curves'].items():ax.plot(np.arange(1,33),c,label=m)
    ax.set(xlabel='Objective evaluations',ylabel='Mean normalized simple regret',title='Digits HPO development pilot (one dataset)');ax.legend(ncol=4);fig.tight_layout()
    for suffix in ('png','pdf'):fig.savefig(OUT/f'curves.{suffix}',dpi=160)
    plt.close(fig)
    lines=['# 真实数据HPO先验试验：开发阶段结果','',f"联合门槛：{'通过' if result['joint_pass'] else '未通过'}。本轮不修改原ROOPF，不证明residual必要性。",'',
        'Digits数据上的二分类类别组合，源/目标图像与类别对均不重叠；类别和采集域共享，目标组合共享部分图像，因此不是15个独立数据集。',
        '源30类对×2划分，目标15类对×3划分；每任务256个C/gamma配置。离线先验仅为源误差均值，无神经网络训练轮次。',
        '945条配对轨迹，每条4点共同初始化、32次总评估；共26880次真实SVC训练、30240次缓存目标查询。全表最优仅供事后计算指标。','',
        '|方法|平均归一化遗憾AUC↓|达到目标的截断评估次数↓|32次内成功率|终局分类验证错误率↓|',
        '|---|---:|---:|---:|---:|']
    for m,v in methods.items():lines.append(f"|{m}|{v['auc']:.6f}|{v['hit']:.3f}|{v['success']:.1%}|{v['final_error']:.6f}|")
    lines+=['','A=固定离线排序；O=固定核纯在线EI；F=源均值+当前观测GP修正，LOO选择先验强度；Fixed=强度1加在线修正；Bad=乱序源先验；GP=在线优化核超参数的GP-EI；Random=随机。','',
        '|预设主比较（控制−F，正为F更好）|指标|差值|99.1667%描述性区间|门槛|','|---|---|---:|---|---|']
    for c in contrasts:lines.append(f"|{c['control']}−F|{c['metric']}|{c['improvement']:+.6f}|[{c['ci'][0]:+.6f},{c['ci'][1]:+.6f}]|{'通过' if c['passed'] else '未通过'}|")
    lines+=['',f"共同初始化已达到质量目标的比例：{methods['F']['initial_success']:.1%}；必须考虑任务天花板。F平均先验强度{result['mean_alpha']:.4f}，不等于离线贡献比例。",'',
        'bootstrap按15个类别对聚类，但类对共享图像，区间不支持独立数据集层面的显著性。只优化验证误差，无分类测试集泛化主张。',
        '全部主条件通过才值得预登记多数据集验证；任一失败则停止本原型，不改预算、网格或阈值制造成功。',
        '固定源先验优势不等于融合必要，融合优于纯在线也不等于优于纯离线。两端必须分别核对。']
    (OUT/'CONCLUSIONS.zh-CN.md').write_text('\n'.join(lines)+'\n');print(json.dumps(dict(joint_pass=result['joint_pass'],methods=methods,contrasts=contrasts),indent=2),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['freeze','source','contracts','target','search','verify','report']);parser.add_argument('--workers',type=int,default=24);a=parser.parse_args()
    if a.phase=='freeze':freeze()
    elif a.phase in ('source','target'):tables(a.phase,a.workers)
    elif a.phase=='contracts':contracts()
    elif a.phase=='search':search(a.workers)
    elif a.phase=='verify':verify(a.workers)
    else:report()
