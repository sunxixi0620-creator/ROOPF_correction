"""Read-only risk signal audit; all labels from completed paid histories."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import torch
from scipy.stats import rankdata
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import cold_start as st
from roopf.experiment_io import CaseStore,sha256,write_json
OUT=ROOT/'docs/revision/cold_start_failure_audit'
def load(f,i,s,m):
    j=(f,i,s,m);return CaseStore(st.RUN/'cases',st.verify()).load('_'.join(map(str,j)),dict(job=j))
def hit(y,goal,limit):
    h=np.flatnonzero(y[:limit]<=goal);return int(h[0]+1) if len(h) else limit+1

def audit(job):
    torch.set_num_threads(1);f,i,s=job;v=load(f,i,s,'W');x=v['x'].numpy();y=v['y'].numpy().astype(float);initial=y[:10];scale=max(initial.std(ddof=1),1e-8);goal=initial.min()-.5*scale
    p=st.base.Prior(f,s,'correct',v['x'][:10],v['y'][:10]);targets=(y-p.center)/p.scale
    gp=st.base.GP(x[:10],targets[:10]);online=[]
    for n in range(10,40):
        mu,_=gp.predict(x[n:n+1]);online.append(float(mu[0]));gp.append(x[n],targets[n])
    assert np.allclose(v['trace'][:30,5],targets[10:40],rtol=1e-12,atol=1e-12)
    peers={m:load(f,i,s if m=='F' else 0,m)['y'].numpy().astype(float) for m in ('O','S','F')}
    hits={m:hit(yy,goal,300) for m,yy in dict(W=y,**peers).items()};late={m:hit(yy,goal,600) for m,yy in dict(W=y,**peers).items()}
    r=dict(fid=f,recipe=f//3,instance=i,seed=s,hits300=hits,hits600=late,W_failure=hits['W']>300,S_only=hits['W']>300 and hits['S']<=300,
        deficit300={m:float((min(yy[:300])-goal)/scale) for m,yy in dict(W=y,**peers).items()},signals={})
    spread=lambda a:float(np.mean((a-a.mean(0))**2))
    for n in (20,40):
        before=np.r_[np.inf,np.minimum.accumulate(y[:n-1])];last=np.flatnonzero(y[:n]<before)[-1]+1
        fusion=v['trace'][:n-10,2];truth=targets[10:n];err=float(np.mean((fusion-truth)**2-(np.array(online[:n-10])-truth)**2))
        gain=float((initial.min()-min(y[:n]))/scale);ratio=spread(x[:n])/max(spread(x[:10]),1e-12)
        r['signals'][str(n)]=dict(at_risk=hit(y,goal,n)>n,low_gain=-gain,stagnation=n-last,low_spread=-ratio,fusion_error_excess=err)
    return r

def auc(y,score):
    y=np.array(y,dtype=bool);score=np.array(score);a=int(y.sum());b=len(y)-a
    if not a or not b:return None
    return float((rankdata(score)[y].sum()-a*(a+1)/2)/(a*b))
def main():
    st.verify();OUT.mkdir(parents=True,exist_ok=True)
    with ProcessPoolExecutor(max_workers=24) as ex:rows=list(ex.map(audit,[(f,i,s) for f in range(36) for i in range(2) for s in range(3)]))
    counts={}
    for w in (False,True):
        for s in (False,True):
            rr=[r for r in rows if (r['hits300']['W']<=300)==w and (r['hits300']['S']<=300)==s]
            counts[f'W{int(w)}_S{int(s)}']=dict(pairs=len(rr),tasks=len({(r['fid'],r['instance']) for r in rr}),recipes=len({r['recipe'] for r in rr}))
    failure=[r for r in rows if r['S_only']];signals=[];rng=np.random.default_rng(202609302);samples=rng.integers(0,12,(1000,12))
    for n in (20,40):
        rr=[r for r in rows if r['signals'][str(n)]['at_risk']]
        for outcome in ('W_failure','S_only'):
            yy=np.array([r[outcome] for r in rr]);recipe=np.array([r['recipe'] for r in rr]);byrecipe=[np.flatnonzero(recipe==g) for g in range(12)]
            for name in ('low_gain','stagnation','low_spread','fusion_error_excess'):
                sc=np.array([r['signals'][str(n)][name] for r in rr]);boots=[];loo=[]
                for sel in samples:
                    ix=np.concatenate([byrecipe[g] for g in sel]);v=auc(yy[ix],sc[ix])
                    if v is not None:boots.append(v)
                for g in range(12):
                    ix=recipe!=g;v=auc(yy[ix],sc[ix])
                    if v is not None:loo.append(v)
                signals.append(dict(nfe=n,outcome=outcome,signal=name,at_risk=len(rr),positives=int(yy.sum()),positive_tasks=len({(r['fid'],r['instance']) for r in rr if r[outcome]}),positive_recipes=len({r['recipe'] for r in rr if r[outcome]}),auc=auc(yy,sc),ci95=np.quantile(boots,[.025,.975]).tolist() if boots else None,loo_range=[min(loo),max(loo)] if loo else None,valid_bootstraps=len(boots)))
    failures=[dict(fid=r['fid'],instance=r['instance'],seed=r['seed'],hits300=r['hits300'],hits600=r['hits600'],deficit300=r['deficit300']) for r in failure]
    result=dict(counts=counts,S_only_failures=failures,signals=signals,extra_calls=0,neural_epochs=0,
        S_only_catchup600=sum(r['hits600']['W']<=600 for r in failure),S_only_O_success=sum(r['hits300']['O']<=300 for r in failure),S_only_F_success=sum(r['hits300']['F']<=300 for r in failure))
    write_json(OUT/'RESULTS.json',result);write_json(OUT/'rows.json',rows)
    paths=[st.RUN/'cases'/('_'.join(map(str,j))+ext) for j in st.jobs() if j[3]!='P' for ext in ('.pt','.json')]
    write_json(OUT/'IDENTITY.json',dict(parent_identity=sha256(st.RUN/'identity.json'),sources={p:sha256(ROOT/p) for p in ('scripts/audit_cold_start.py','docs/experiments/COLD_START_FAILURE_AUDIT.md')},inputs={str(p.relative_to(ROOT)):sha256(p) for p in paths},extra_calls=0))
    lines=['# 冷启动失败轨迹与提前信号审查','', '只读探索性分析：新增目标评估0次、训练0轮。不是新策略实验或独立确认。', '', '|300次结果|模型种子配对数|涉及任务实例|涉及配方|','|---|---:|---:|---:|']
    for k,v in counts.items():lines.append(f"|{k}|{v['pairs']}|{v['tasks']}|{v['recipes']}|")
    lines+=['','W1/S1表示达到目标，0表示未达到。同一实例的3个模型种子可能分属多组，任务列不能直接相加；S共享，不把它算作3个独立运行。', '',f"W失败而S成功共有{len(failure)}个模型种子配对；其中O在300次成功{result['S_only_O_success']}个，持续融合F成功{result['S_only_F_success']}个，W到600次追上{result['S_only_catchup600']}个。轨迹不同，不能归因于单一决策。", '', '|fid/实例/种子|W达到时间（上限600）|O/S/F达到时间（上限300）|W在300时距目标差/初始std|','|---|---:|---|---:|']
    for r in failures:lines.append(f"|{r['fid']}/{r['instance']}/{r['seed']}|{r['hits600']['W']}|{r['hits300']['O']}/{r['hits300']['S']}/{r['hits300']['F']}|{r['deficit300']['W']:+.4f}|")
    lines+=['','301或601是删失标记，不能当作实际成功时间。', '', '|时点|预测事件|固定风险信号|风险集/事件数|事件任务/配方数|AUC|95%探索区间|留一配方AUC范围|','|---|---|---|---|---|---:|---|---|']
    for r in signals:lines.append(f"|{r['nfe']}|{r['outcome']}|{r['signal']}|{r['at_risk']}/{r['positives']}|{r['positive_tasks']}/{r['positive_recipes']}|{r['auc']:.3f}|{r['ci95']}|{r['loo_range']}|")
    lines+=['','仅分析当时尚未达到目标的W轨迹，避免用已经成功来预测未来成功。low_gain低进展、stagnation停滞、low_spread低空间展开度、fusion_error_excess融合预测误差大于同历史纯在线预测误差。方向预先固定，没有反转AUC或搜索门槛。',
        '在线预测在W同一付费历史上逐点重建，只使用目标评估前已有数据；不表示纯在线策略会选择同样的点。信号计算没有使用S/O/F的未来结果；这些未来结果仅用于事后标签。',
        '留一配方只是既定信号的关联敏感性分析，没有训练模型，不能视为独立泛化检验。多个信号/时间/事件的区间没有多重比较校正，不能择优声称显著。先验失配、局部采样和任务困难仍可能混杂；不据此直接设计或宣称可靠门控。']
    (OUT/'CONCLUSIONS.zh-CN.md').write_text('\n'.join(lines)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='signals'},indent=2))
if __name__=='__main__':main()
