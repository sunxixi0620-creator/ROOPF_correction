"""Independent equations, isolation/recovery and numerical contracts (fake y only)."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import json
import sys
import tempfile
import time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from roopf.hpob_transfer_v1 import (fit,build_gp,Predictor,standardize,discordances,
    bootstrap_losses,rank_weights,taf_scores,propose,log_ei)
from roopf.hpob_grouped_oracle import PaidOracle
from roopf.experiment_io import write_json,sha256


def main():
    torch.set_num_threads(1);torch.manual_seed(9101)
    x=torch.rand(128,6,dtype=torch.double);y=-(x-.3).square().sum(1)+.1*torch.sin(17*x[:,0])
    z,_,_=standardize(y);model,info=fit(x,z,'fabricated_contract')
    predictor=Predictor(model);q=torch.rand(521,6,dtype=torch.double)
    mu,var=predictor.moments(q)
    with torch.no_grad():p=model.posterior(q)
    mean_error=float((mu-p.mean[:,0]).abs().max());var_error=float((var-p.variance[:,0]).abs().max())
    assert mean_error<1e-8 and var_error<1e-8,(mean_error,var_error)
    loo=predictor.loo_mean();loo_error=0.
    for i in (0,30,127):
        ix=[j for j in range(len(x)) if j!=i];m=build_gp(x[ix],z[ix])
        with torch.no_grad():
            for n,v in m.named_parameters():v.copy_(info['parameters'][n])
        m.eval();mean=Predictor(m).moments(x[i:i+1],False)[0]
        loo_error=max(loo_error,abs(float(mean[0]-loo[i])))
    assert loo_error<1e-8,loo_error
    rng=np.random.default_rng(9192);sy=rng.normal(size=(5,8));ty=rng.normal(size=8);tl=rng.normal(size=8)
    indices=rng.integers(0,8,(40,8));D=discordances(sy,tl,ty);fast=bootstrap_losses(D,indices)
    slow=np.empty_like(fast)
    for s,ix in enumerate(indices):
        for m in range(6):slow[m,s]=sum(D[m,i,j] for i in ix for j in ix)
    assert np.array_equal(fast,slow)
    weights,diag=rank_weights(sy,tl,ty,'test',8,horizon=8)
    assert np.array_equal(weights,np.array([0.,0.,0.,0.,0.,1.]))
    source=torch.tensor(sy,dtype=torch.double);le=torch.arange(8,dtype=torch.double)
    assert torch.equal(taf_scores(le,source,[0,1,2],weights),le)
    from unittest.mock import patch
    with patch('roopf.hpob_transfer_v1.rank_weights',return_value=(np.array([1.,0.,0.,0.,0.,0.]),{})):
        selected,zero=propose(x,list(range(5)),y[:5],torch.zeros(5,len(x),dtype=torch.double),'R','zero_contract')
        assert selected==5 and zero['score']==0. and zero['score_kind']=='raw_zero_acquisition'
    with tempfile.TemporaryDirectory(prefix='hpob_contract_') as d:
        d=Path(d);np.save(d/'labels.npy',y.numpy())
        o=PaidOracle(d/'labels.npy',d/'journal.json',budget=5)
        assert not o.paid;assert o.query(4)==float(y[4])
        for call in (lambda:o.query(4),lambda:o.query(-1),o.metrics):
            try:call();raise AssertionError('Expected oracle rejection')
            except ValueError:pass
        o.close()
        o=PaidOracle(d/'labels.npy',d/'journal.json',budget=5)
        assert len(o.paid)==1 and o.paid[0]['index']==4
        o.query(5);o.query(6);o.query(7);o.query(8);assert o.metrics()['terminal']>=0
        try:o.query(9);raise AssertionError('Budget should reject')
        except ValueError:pass
        o.close();assert len(json.loads((d/'journal.json').read_text()))==5
    # Run all four decision mechanisms on synthetic responses; reproducibility
    # and finite full-pool selection, independent of actual HPO exports.
    fake_source=torch.stack([-((x-c)**2).sum(1) for c in (.1,.2,.4,.6,.9)])
    decisions={}
    for method in ('O','P','A','R'):
        paid=list(range(5));prior=None;trace=[]
        for n in range(5,8):
            selected,record=propose(x,paid,y[paid],fake_source,method,('contract',method),prior)
            assert selected not in paid and record['paid_before']==n
            paid.append(selected);prior=record['parameters'];trace.append(record)
        decisions[method]=paid
    gpu={}
    if torch.cuda.is_available():
        m=build_gp(x,z)
        with torch.no_grad():
            for n,v in m.named_parameters():v.copy_(info['parameters'][n])
        m=m.cuda();m.eval();gp=Predictor(m)
        gm,gv=gp.moments(q.cuda());gpu=dict(mean_error=float((gm.cpu()-mu).abs().max()),variance_error=float((gv.cpu()-var).abs().max()))
        assert max(gpu.values())<1e-8,gpu
        # Cached source prediction throughput: same pre-trained GP, public X only.
        qbig=torch.rand(19972,6,dtype=torch.double);speed={}
        for device,pred in [('cpu',predictor),('cuda',gp)]:
            qb=qbig.to(device)
            if device=='cuda':torch.cuda.synchronize()
            start=time.perf_counter();pred.moments(qb,variance=False)
            if device=='cuda':torch.cuda.synchronize()
            speed[device]=time.perf_counter()-start
        gpu['cache_benchmark_seconds']=speed
    write_json(ROOT/'docs/revision/hpob_transfer_development_v1/CONTRACTS.json',dict(
        fake_only=True,hpob_target_calls=0,mean_error=mean_error,variance_error=var_error,
        loo_error=loo_error,bootstrap_matches_explicit_pairs=True,
        target_only_taf_matches_logei=True,zero_taf_canonical_tie=True,oracle_budget_duplicate_and_resume=True,
        decisions=decisions,gpu=gpu,contract_sha256=sha256(Path(__file__))))
    print('contracts passed',mean_error,var_error,loo_error,gpu,flush=True)


if __name__=='__main__':main()
