"""Exploratory source compatibility smoke, not a supported-environment benchmark."""
import os
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import sys,json,time,hashlib
ROOT=Path(__file__).resolve().parent
wheel=ROOT/'tunecontrol-0.1.0-py3-none-any.whl'
sys.path.insert(0,str(wheel))
import torch
import tunecontrol as tc
torch.set_num_threads(1)
records=[]
for family in ('cartpole','cascaded_tank'):
    p=tc.make(family);theta=p.bounds.mean(dim=0);values=[]
    for i in range(2):
        p.setup(run_seed=42);start=time.perf_counter();v,info=p.evaluate(theta);elapsed=time.perf_counter()-start
        assert torch.isfinite(v);values.append(float(v))
        records.append(dict(family=family,mode='deterministic',index=i,cost=float(v),seconds=elapsed,theta=theta.tolist(),trajectory_fields=list(info['trajectory'])))
    assert values[0]==values[1]
p=tc.CartPole(tc.CartPoleConfig(dim=2,noise=tc.CartPoleNoise()));theta=p.bounds.mean(dim=0);p.setup(run_seed=42)
vals=[]
for i in range(2):
    start=time.perf_counter();v,_=p.evaluate(theta);vals.append(float(v));records.append(dict(family='cartpole',mode='noisy',index=i,cost=float(v),seconds=time.perf_counter()-start))
p.setup(run_seed=42);start=time.perf_counter();v,_=p.evaluate(theta);assert float(v)==vals[0]
records.append(dict(family='cartpole',mode='noisy_replay',index=0,cost=float(v),seconds=time.perf_counter()-start))
result=dict(package='tunecontrol',version='0.1.0',wheel_sha256=hashlib.sha256(wheel.read_bytes()).hexdigest(),python=sys.version,torch=torch.__version__,official_python_requirement='>=3.12,<3.13',supported_environment=False,note='Unmodified pure-Python wheel loaded by path in existing Python 3.10 for exploratory compatibility only. No install or environment downgrade; formal Python 3.12 reproducibility pending.',objective_calls=7,optimizer_runs=0,records=records)
(ROOT/'TUNECONTROL_SMOKE.json').write_text(json.dumps(result,indent=2,allow_nan=False));print(json.dumps(result,indent=2))
