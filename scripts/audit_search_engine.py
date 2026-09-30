"""Zero-objective-query replay and finite candidate-pool audit."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
from scipy.special import ndtr
from scipy.spatial.distance import cdist

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import cold_start_components as study
from roopf.experiment_io import CaseStore, sha256, seed_for, write_json

OUT = ROOT / 'docs/research/search_engine_audit_20260930'
COUNTS = (20, 39, 100, 300)
METHODS = ('O', 'W', 'WA')


def forbidden(*args, **kwargs):
    raise RuntimeError('Objective evaluation forbidden during this audit')


def scores(q, x, y, prior, gp, active):
    pm = prior(q) if active else np.zeros(len(q))
    gm, sd = gp.predict(q)
    mu = pm + gm
    delta = (float(min(y)) - prior.center) / prior.scale - mu
    z = delta / np.maximum(sd, 1e-14)
    ei = delta * ndtr(z) + sd * np.exp(-z*z/2) / np.sqrt(2*np.pi)
    ei[cdist(q, x.numpy()).min(1) <= 1e-6] = -np.inf
    return ei, z, sd


@torch.no_grad()
def audit(job):
    torch.set_num_threads(1)
    study.ProceduralTask.calfitness = forbidden
    f, m = job
    j = (f, 0, 0, m)
    identity = study.verify()
    name = '_'.join(map(str, j))
    v = CaseStore(study.RUN / 'cases', identity).load(name, dict(job=j))
    assert v is not None
    xall, yall = v['x'], v['y']
    p = study.base.Prior(f, 0, {'O':'zero', 'W':'correct', 'WA':'analytic'}[m], xall[:10], yall[:10])
    assert p.identity == v['prior_identity']
    rows = []
    for n in COUNTS:
        x, y, step = xall[:n], yall[:n], n - 10
        active = m != 'O' and n < 40
        if not active:
            targets = (y.numpy().astype(float) - p.center) / p.scale
        else:
            initial = (y[:10].numpy().astype(float) - p.center) / p.scale - p(x[:10].numpy())
            targets = np.r_[initial, v['trace'][:step,5] - v['trace'][:step,1]]
        gp = study.base.GP(x.numpy(), targets)
        q = study.pool(x, y.numpy(), f, 0, study.ROLE, step)
        new = study.choose(q, x, y.numpy(), p, gp, active, m)
        old = v['trace'][step,:5]
        error = float(np.max(np.abs(np.asarray(new)-old)))
        assert new[0] == int(old[0]) and np.array_equal(q[new[0]], xall[n].numpy())
        assert np.allclose(new, old, rtol=1e-7, atol=1e-8), (job, n, new, old)
        ei, z, sd = scores(q, x, y.numpy(), p, gp, active)
        assert int(ei.argmax()) == new[0]
        rng = torch.Generator().manual_seed(seed_for('engine_audit_20260930',f,m,n))
        width = 10 * (.05 + .35 * (600-n)/600)
        extra = torch.cat((torch.rand(2048,20,generator=rng)*10-5,
                           (x[int(y.argmin())] + width*torch.randn(2048,20,generator=rng)).clamp(-5,5))).numpy()
        extra_ei = np.concatenate([scores(qpart, x, y.numpy(), p, gp, active)[0]
                                   for qpart in np.array_split(extra,16)])
        best = float(ei.max())
        expanded = max(best, float(extra_ei.max()))
        assert np.isfinite(best) and np.isfinite(expanded) and best > 0
        boundary = abs(q[64:]) >= 5
        rows.append(dict(fid=f, recipe=f//3, method=m, paid_count=n,
                         replay_error=error, chosen_local=bool(new[0]>=64),
                         local_bandwidth=width,
                         local_boundary_coordinate_fraction=float(boundary.mean()),
                         local_boundary_candidate_fraction=float(boundary.any(1).mean()),
                         original_best_ei=best, expanded_best_ei=expanded,
                         acquisition_ratio=expanded/best,
                         zero_ei_count=int((ei==0).sum()),negative_ei_count=int((ei<0).sum()),
                         nonfinite_ei_count=int((~np.isfinite(ei)).sum()),
                         max_tie_count=int(np.isclose(ei,best,rtol=1e-12,atol=0).sum()),
                         z_min=float(z.min()),z_max=float(z.max()),
                         sd_min=float(sd.min()),sd_max=float(sd.max()),sd_mean=float(sd.mean())))
    p.verify()
    return rows, {str((study.RUN/'cases'/f'{name}{suffix}').relative_to(ROOT)):
                  sha256(study.RUN/'cases'/f'{name}{suffix}') for suffix in ('.pt','.json')}


def run(workers):
    start=time.time()
    study.verify()
    OUT.mkdir(parents=True,exist_ok=True)
    sources=['scripts/audit_search_engine.py','docs/experiments/SEARCH_ENGINE_AUDIT_PROTOCOL.md',
             'scripts/cold_start.py','scripts/cold_start_components.py','scripts/frozen_prior_correction.py']
    identity=dict(sources={p:sha256(ROOT/p) for p in sources},
                  parent=sha256(study.RUN/'identity.json'),counts=COUNTS,methods=METHODS,
                  instances=[0],seeds=[0],configs=list(range(36)))
    write_json(OUT/'IDENTITY.json',identity)
    rows=[]; inputs={}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i,(r,h) in enumerate(ex.map(audit,[(f,m) for f in range(36) for m in METHODS]),1):
            rows.extend(r);inputs.update(h)
            if i%18==0:print(f'{i}/108 trajectories audited; {time.time()-start:.1f}s',flush=True)
    assert len(rows)==432
    def summarize(rr):
        ratios=np.array([r['acquisition_ratio'] for r in rr])
        return dict(states=len(rr), replay_max_error=max(r['replay_error'] for r in rr),
                    boundary_coordinates=float(np.mean([r['local_boundary_coordinate_fraction'] for r in rr])),
                    boundary_candidates=float(np.mean([r['local_boundary_candidate_fraction'] for r in rr])),
                    local_selection=float(np.mean([r['chosen_local'] for r in rr])),
                    best_ei_min=min(r['original_best_ei'] for r in rr),
                    zero_scores=sum(r['zero_ei_count'] for r in rr),
                    negative_scores=sum(r['negative_ei_count'] for r in rr),
                    nonfinite_scores=sum(r['nonfinite_ei_count'] for r in rr),
                    states_with_tied_max=sum(r['max_tie_count']>1 for r in rr),
                    acquisition_ratio_median=float(np.median(ratios)),
                    acquisition_ratio_p90=float(np.quantile(ratios,.9)),
                    acquisition_over_2x=float(np.mean(ratios>2)),
                    acquisition_over_1_1x=float(np.mean(ratios>1.1)))
    result=dict(overall=summarize(rows), groups={f'{m}_{n}':summarize([r for r in rows if r['method']==m and r['paid_count']==n]) for m in METHODS for n in COUNTS},
                objective_calls=0,training_epochs=0,workers=workers,seconds=time.time()-start,
                interpretation='Acquisition approximation and geometry only; no counterfactual objective outcomes.')
    write_json(OUT/'ROWS.json',rows);write_json(OUT/'INPUT_HASHES.json',inputs);write_json(OUT/'RESULTS.json',result)
    for p,h in identity['sources'].items():assert sha256(ROOT/p)==h
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=24)
    run(parser.parse_args().workers)
