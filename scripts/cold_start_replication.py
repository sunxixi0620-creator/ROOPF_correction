"""Unchanged W/O/S replication on additional instances."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,time,argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import cold_start as st
from roopf.experiment_io import CaseStore,sha256,write_json
RUN=ROOT/'results/cold_start_replication_v1';OUT=ROOT/'docs/revision/cold_start_replication'
ROLE='cold_start_replication_v1'
SOURCES=('scripts/cold_start_replication.py','docs/experiments/COLD_START_REPLICATION_PROTOCOL.md')
def verify():
    st.verify();z=json.loads((RUN/'identity.json').read_text())
    for p,h in z['sources'].items():assert sha256(ROOT/p)==h
    assert sha256(st.RUN/'identity.json')==z['parent'];return z
def freeze():
    st.verify();RUN.mkdir(exist_ok=True)
    if (RUN/'identity.json').exists():return verify()
    write_json(RUN/'identity.json',dict(sources={p:sha256(ROOT/p) for p in SOURCES},parent=sha256(st.RUN/'identity.json'),time=time.time()))
def jobs():return [(f,i,s,m) for f in range(36) for i in range(8) for m in ('W','O','S') for s in (range(3) if m=='W' else [0])]
def collect(j):
    store=CaseStore(RUN/'cases',verify());name='_'.join(map(str,j))
    with store.lock(name):
        if store.load(name,dict(job=j)) is None:store.save(name,dict(job=j),st.rollout(*j,role=ROLE))
def run():
    start=time.time()
    with ProcessPoolExecutor(max_workers=24) as ex:
        fs=[ex.submit(collect,j) for j in jobs()]
        for n,f in enumerate(as_completed(fs),1):
            f.result()
            if n%144==0:print(f'{n}/1440, {time.time()-start:.1f}s',flush=True)
    write_json(RUN/'COMPLETE.json',dict(cases=1440,calls=864000,seconds=time.time()-start))
def report():
    store=CaseStore(RUN/'cases',verify());OUT.mkdir(parents=True,exist_ok=True)
    arrays={m:{'time':np.zeros((3,36,8,3)),'success':np.zeros((3,36,8,3)),'u':np.zeros((3,36,8,591))} for m in ('W','O','S')};calls=0
    for j in jobs():
        f,i,s,m=j;v=store.load('_'.join(map(str,j)),dict(job=j));assert v is not None;calls+=v['calls'];y=v['y'].numpy().astype(float);initial=y[:10];gain=np.maximum(0,initial.min()-np.minimum.accumulate(y)[9:])/max(initial.std(ddof=1),1e-8)
        ts=[];ss=[]
        for g in (.25,.5,1.):
            hit=np.flatnonzero(gain[:291]>=g);ts.append(int(hit[0]+10) if len(hit) else 301);ss.append(bool(len(hit)))
        for seed in ([s] if m=='W' else range(3)):
            arrays[m]['time'][seed,f,i]=ts;arrays[m]['success'][seed,f,i]=ss;arrays[m]['u'][seed,f,i]=gain/(1+gain)
    assert calls==864000;rng=np.random.default_rng(202609303);index=rng.integers(0,12,(10000,12))
    def est(d):
        groups=d.reshape(3,12,3,8).mean((0,2,3));lo,hi=np.quantile(groups[index].mean(1),[.0125,.9875]);return dict(mean=float(d.mean()),lower=float(lo),upper=float(hi),seeds=d.mean((1,2)).tolist())
    contrasts=[]
    for m in ('O','S'):
        a=est(arrays[m]['time'][...,1]-arrays['W']['time'][...,1]);b=est(arrays['W']['u'][...,-1]-arrays[m]['u'][...,-1]);c=est(arrays['W']['success'][...,1]-arrays[m]['success'][...,1]);passed=a['mean']>=5 and a['lower']>0 and min(a['seeds'])>0 and b['lower']>-.005 and c['mean']>=0
        contrasts.append(dict(control=m,time_saved=a,terminal_delta=b,attainment_delta=c,passed=bool(passed)))
    result=dict(joint_pass=all(c['passed'] for c in contrasts),contrasts=contrasts,calls=calls,epochs=0,means={m:{'time':a['time'].mean((0,1,2)).tolist(),'success':a['success'].mean((0,1,2)).tolist(),'terminal':float(a['u'][...,-1].mean())} for m,a in arrays.items()})
    write_json(OUT/'RESULTS.json',result);np.savez_compressed(OUT/'arrays.npz',**{m+'_'+k:v for m,a in arrays.items() for k,v in a.items()});print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['freeze','run','report']);a=p.parse_args();globals()[a.phase]()
