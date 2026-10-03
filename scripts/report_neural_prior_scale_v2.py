"""Paired, source-only scale diagnostic with raw-unit density comparisons."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import json,math,sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import train_neural_prior_scale_v2 as s
from roopf.neural_prior_scale_v2 import StableSourceModel,normalize_with_reference
from roopf.experiment_io import CaseStore,write_json,sha256


def aggregate(rows):
    groups=sorted({r['group'] for r in rows});out={}
    for kind in ('N','M','PCA'):
        out[kind]={metric:dict(mean=float(np.mean([np.mean([r[metric] for r in rows if r['kind']==kind and r['group']==g]) for g in groups])),
            groups={g:float(np.mean([r[metric] for r in rows if r['kind']==kind and r['group']==g])) for g in groups})
            for metric in ('nll_raw','coverage90','query_choice_regret')}
    return out


def main():
    torch.set_num_threads(1);identity=s.verify();assert (s.RUN/'COMPLETE.json').exists()
    store=CaseStore(s.RUN/'models',identity);rows=[];fits=[];checks=[];metadata={}
    original=[dict(**r,nll_raw=r['nll']+math.log(r['normalization_scale']))
              for r in json.loads((s.AUDIT/'ROWS.json').read_text()) if r['seed']==0]
    for job in identity['jobs']:
        space,seed,kind,device=(job[k] for k in ('space','seed','kind','device'));name=f'{space}_{seed}_{kind}'
        v=store.load(name,job);assert v is not None
        assert v['best']['metrics']['nll']==min(h['nll'] for h in v['history'])
        _,val=s.parent.load_episodes(space,device)
        model=StableSourceModel(v['dim'],kind,seed,space,v['reference_variance']).to(device);model.load_state_dict(v['best']['state'])
        replay=val.validation(model);error=max(abs(a[k]-b[k]) for a,b in zip(replay['rows'],v['best']['metrics']['rows'])
                                             for k in ('nll','coverage90','query_choice_regret'))
        assert error<1e-7,(name,error)
        if space not in metadata:
            metadata[space]=[]
            for n in (5,10,20,40):
                for i,t in enumerate(val.tasks):
                    for repeat in range(8):
                        _,c,q=val.description(i,n,repeat,0,'uniform' if repeat<4 else 'search')
                        _,zq,_,scale=normalize_with_reference(t['y'][c],t['y'][q],v['reference_variance'])
                        metadata[space].append(dict(task=t['name'],context=n,repeat=repeat,normalization_scale=float(scale),
                            max_abs_standardized_query=float(zq.abs().max())))
        for r,meta in zip(replay['rows'],metadata[space]):
            assert r['task']==meta['task'] and r['context']==meta['context']
            rows.append(dict(**r,**{k:v for k,v in meta.items() if k not in ('task','context')},
                seed=seed,kind=kind,space=space,nll_raw=r['nll']+math.log(meta['normalization_scale'])))
        fits.append(dict(job=job,steps=v['steps'],best_step=v['best']['step'],seconds=v['seconds'],history=v['history']))
        checks.append(dict(job=job,max_error=error,checkpoint_sha256=sha256(s.RUN/'models'/f'{name}.pt')))
        print('replayed',name,device,'error',error,flush=True)
    assert len(rows)==len(original)==82*32*3 and len(fits)==45
    keys=lambda rr: {(r['space'],r['task'],r['context'],r['repeat'],r['kind']) for r in rr}
    assert keys(rows)==keys(original)
    v1=aggregate(original);v2=aggregate(rows)
    result=dict(scope='one-seed source-validation diagnostic; no new target experiments',v1_seed0=v1,v2_seed0=v2,
        version_differences={k:{m:v1[k][m]['mean']-v2[k][m]['mean'] for m in ('nll_raw','query_choice_regret')} for k in v2},
        contrasts={f'N_vs_{other}':{m:v2[other][m]['mean']-v2['N'][m]['mean'] for m in ('nll_raw','query_choice_regret')} for other in ('M','PCA')},
        max_standardized_query_v1=max(v['max_abs_standardized_query'] for v in json.loads((s.AUDIT/'SCALES.json').read_text())),
        max_standardized_query_v2=max(r['max_abs_standardized_query'] for r in rows),
        at_update_cap={k:sum(v['job']['kind']==k and v['steps']==20000 for v in fits) for k in v2},
        best_step_zero={k:sum(v['job']['kind']==k and v['best_step']==0 for v in fits) for k in v2},
        source_updates=sum(v['steps'] for v in fits),target_calls=0,confirmation_calls=0,
        same_device_replay_passed=True,maximum_replay_error=max(c['max_error'] for c in checks),
        identity_sha256=sha256(s.RUN/'identity.json'),online_evaluation_complete=False,neural_necessity_established=False)
    write_json(s.OUT/'RESULTS.json',result);write_json(s.OUT/'ROWS.json',rows);write_json(s.OUT/'FITS.json',fits)
    write_json(s.OUT/'VERIFICATION.json',dict(same_device_passed=True,cases=45,checks=checks))
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
