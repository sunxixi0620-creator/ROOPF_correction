"""Preserve failed v1 CPU check; replay on both devices and diagnose label scale.

This is a post-training audit. It changes no checkpoints or acceptance tolerance,
uses source validation only, and makes no new objective calls.
"""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import json,sys,time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import train_neural_prior_v1 as s
from roopf.neural_prior_training_v1 import SourceModel,Episodes
from roopf.experiment_io import CaseStore,write_json,sha256
OUT=ROOT/'docs/revision/neural_prior_training_v1_audit'


def main():
    torch.set_num_threads(1);identity=s.verify();start=time.perf_counter()
    assert (s.RUN/'TRAIN_COMPLETE.json').exists();OUT.mkdir(parents=True,exist_ok=True)
    store=CaseStore(s.RUN/'models',identity);fits=[];rows=[];scales=[];audits=[];hashes={}
    for space in sorted({t['space'] for t in s.tasks()},key=int):
        _,val=s.load_episodes(space,'cpu')
        gpu=Episodes(val.tasks,space,'validation','cuda');metadata=[]
        for n in (5,10,20,40):
            for i,t in enumerate(val.tasks):
                for repeat in range(8):
                    mode='uniform' if repeat<4 else 'search';_,c,q=val.description(i,n,repeat,0,mode)
                    yc,yq=t['y'][c],t['y'][q];sd=float(yc.std());scale=1. if sd<1e-8 else sd
                    record=dict(task=t['name'],space=space,group=t['group'],context=n,repeat=repeat,mode=mode,
                        raw_context_sd=sd,normalization_scale=scale,max_abs_standardized_query=float(((yq-yc.mean())/scale).abs().max()),
                        context_indices=c.tolist(),query_indices=q.tolist(),context_y=yc.tolist(),
                        query_min=float(yq.min()),query_max=float(yq.max()))
                    metadata.append(record);scales.append(record)
        for seed in (0,1,2):
            for kind in ('N','M','PCA'):
                job=(space,seed,kind);name=f'{space}_{seed}_{kind}';v=store.load(name,dict(job=job));assert v is not None
                assert v['best']['metrics']['nll']==min(h['nll'] for h in v['history'])
                assert 0<=v['best']['step']<=v['steps']<=20000 and len(v['history'])==v['steps']//100+1
                model=SourceModel(v['dim'],kind,seed,space);model.load_state_dict(v['best']['state'])
                replay_cpu=val.validation(model);replay_gpu=gpu.validation(model.to('cuda'))
                assert len(replay_cpu['rows'])==len(replay_gpu['rows'])==len(metadata)
                errors={device:{metric:[] for metric in ('nll','coverage90','query_choice_regret')} for device in ('cpu','cuda')}
                relative=[];failures=[]
                for index,(saved,rc,rg,meta) in enumerate(zip(v['best']['metrics']['rows'],replay_cpu['rows'],replay_gpu['rows'],metadata)):
                    for field in ('task','group','context'):assert saved[field]==rc[field]==rg[field]==meta[field]
                    for device,r in (('cpu',rc),('cuda',rg)):
                        for metric in errors[device]:
                            err=abs(saved[metric]-r[metric]);errors[device][metric].append(err)
                            if device=='cpu' and metric=='nll':relative.append(err/max(1.,abs(saved[metric])))
                            if err>=1e-7:failures.append(dict(episode=index,device=device,metric=metric,error=err,
                                recorded=saved[metric],replayed=r[metric],task=meta['task'],context=meta['context'],repeat=meta['repeat']))
                    rows.append(dict(**saved,space=space,seed=seed,kind=kind,repeat=meta['repeat'],mode=meta['mode'],
                        normalization_scale=meta['normalization_scale'],raw_context_sd=meta['raw_context_sd']))
                audit=dict(name=name,max_absolute_errors={d:{m:max(e) for m,e in a.items()} for d,a in errors.items()},
                    max_cpu_nll_scaled_error=max(relative),original_cpu_check_passed=not any(f['device']=='cpu' for f in failures),
                    same_cuda_check_passed=not any(f['device']=='cuda' for f in failures),failures=failures)
                audits.append(audit);hashes[name]=sha256(s.RUN/'models'/f'{name}.pt')
                fits.append(dict(name=name,space=space,seed=seed,kind=kind,dim=v['dim'],steps=v['steps'],best_step=v['best']['step'],
                    seconds=v['seconds'],initial={k:v['history'][0][k] for k in ('nll','coverage90','query_choice_regret')},
                    best={k:v['best']['metrics'][k] for k in ('nll','coverage90','query_choice_regret')},history=v['history']))
                print('audited',name,'cpu_abs_nll',audit['max_absolute_errors']['cpu']['nll'],
                      'cuda_abs_nll',audit['max_absolute_errors']['cuda']['nll'],flush=True)
    assert len(fits)==135 and len(scales)==82*32 and len(rows)==82*32*9
    groups=sorted({r['group'] for r in rows});assert len(groups)==8
    means={};group_values={};contexts={}
    for kind in ('N','M','PCA'):
        group_values[kind]={metric:[float(np.mean([r[metric] for r in rows if r['kind']==kind and r['group']==g]))
            for g in groups] for metric in ('nll','coverage90','query_choice_regret')}
        means[kind]={metric:float(np.mean(vals)) for metric,vals in group_values[kind].items()}
        contexts[kind]={str(n):{metric:float(np.mean([np.mean([r[metric] for r in rows if r['kind']==kind and r['group']==g and r['context']==n])
            for g in groups])) for metric in ('nll','coverage90','query_choice_regret')} for n in (5,10,20,40)}
    selected_rows=sorted(rows,key=lambda r:r['nll'],reverse=True)[:20]
    numerical=dict(original_cpu_absolute_tolerance=1e-7,
        original_cpu_check_passed=all(a['original_cpu_check_passed'] for a in audits),
        original_failed_models=sum(not a['original_cpu_check_passed'] for a in audits),
        same_device_cuda_replay_passed=all(a['same_cuda_check_passed'] for a in audits),
        same_device_max_absolute_error=max(e for a in audits for e in a['max_absolute_errors']['cuda'].values()),
        cpu_max_absolute_nll_error=max(a['max_absolute_errors']['cpu']['nll'] for a in audits),
        cpu_max_scaled_nll_error=max(a['max_cpu_nll_scaled_error'] for a in audits),
        cpu_max_coverage_error=max(a['max_absolute_errors']['cpu']['coverage90'] for a in audits),
        cpu_max_choice_regret_error=max(a['max_absolute_errors']['cpu']['query_choice_regret'] for a in audits))
    result=dict(scope='Source validation selected checkpoints; post-training diagnostic, not independent confirmation',
        means=means,group_values=group_values,groups=groups,by_context=contexts,numerical=numerical,
        fits=135,completed_updates=sum(v['steps'] for v in fits),
        at_update_cap={k:sum(v['kind']==k and v['steps']==20000 for v in fits) for k in means},
        best_step_zero={k:sum(v['kind']==k and v['best_step']==0 for v in fits) for k in means},
        scale_diagnostics=dict(episodes=len(scales),sd_between_fallback_and_1e4=sum(1e-8<=v['raw_context_sd']<1e-4 for v in scales),
            max_standardized_query=max(v['max_abs_standardized_query'] for v in scales),top_nll_rows=selected_rows),
        target_calls=0,confirmation_calls=0,neural_necessity_established=False,
        training_wall_seconds=json.loads((s.RUN/'TRAIN_COMPLETE.json').read_text())['seconds'],
        audit_seconds=time.perf_counter()-start,identity_sha256=sha256(s.RUN/'identity.json'),audit_source_sha256=sha256(Path(__file__)))
    write_json(OUT/'RESULTS.json',result);write_json(OUT/'NUMERICAL_REPLAYS.json',audits)
    write_json(OUT/'FITS.json',fits);write_json(OUT/'ROWS.json',rows);write_json(OUT/'SCALES.json',scales)
    write_json(OUT/'CHECKPOINTS.json',hashes)
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(5,3,figsize=(12,15));colors={'N':'tab:blue','M':'tab:orange','PCA':'tab:green'}
    for ax,space in zip(axes.flat,sorted({v['space'] for v in fits},key=int)):
        for v in fits:
            if v['space']==space:ax.plot([h['step'] for h in v['history']],[h['nll'] for h in v['history']],
                color=colors[v['kind']],alpha=.6,label=v['kind'] if v['seed']==0 else None)
        ax.set_title('Space '+space);ax.set_xlabel('Updates');ax.set_ylabel('Validation NLL');ax.set_yscale('symlog');ax.legend()
    fig.tight_layout()
    for ext in ('pdf','png'):fig.savefig(OUT/f'learning_curves.{ext}',dpi=150)
    print(json.dumps(dict(means=means,numerical=numerical),indent=2),flush=True)


if __name__=='__main__':main()
