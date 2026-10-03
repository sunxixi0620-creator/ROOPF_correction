"""Source-only learning diagnostics; selected validation scores are descriptive."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import json,sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import train_neural_prior_v1 as s
from roopf.neural_prior_training_v1 import SourceModel
from roopf.experiment_io import CaseStore,write_json,sha256


def main():
    torch.set_num_threads(1);identity=s.verify()
    assert (s.RUN/'TRAIN_COMPLETE.json').exists()
    store=CaseStore(s.RUN/'models',identity);fits=[];rows=[];max_error=0.;hashes={}
    spaces=sorted({t['space'] for t in s.tasks()},key=int)
    for space in spaces:
        _,validation=s.load_episodes(space,'cpu')
        for seed in (0,1,2):
            for kind in ('N','M','PCA'):
                job=(space,seed,kind);name=f'{space}_{seed}_{kind}'
                v=store.load(name,dict(job=job));assert v is not None
                assert v['target_calls']==v['confirmation_calls']==0
                assert 0<=v['best']['step']<=v['steps']<=20000
                assert v['steps']%100==0 and len(v['history'])==v['steps']//100+1
                assert v['best']['metrics']['nll']==min(h['nll'] for h in v['history'])
                assert not set(v['train_tasks'])&set(v['validation_tasks'])
                model=SourceModel(v['dim'],kind,seed,space);model.load_state_dict(v['best']['state'])
                replay=validation.validation(model)
                assert len(replay['rows'])==len(v['validation_tasks'])*32
                for observed,saved in zip(replay['rows'],v['best']['metrics']['rows']):
                    for key in ('task','group','context'):assert observed[key]==saved[key]
                    for metric in ('nll','coverage90','query_choice_regret'):
                        error=abs(observed[metric]-saved[metric]);max_error=max(max_error,error)
                        assert error<1e-7,(name,metric,error)
                rows.extend(dict(**r,space=space,seed=seed,kind=kind) for r in replay['rows'])
                fits.append(dict(name=name,space=space,seed=seed,kind=kind,dim=v['dim'],
                    steps=v['steps'],best_step=v['best']['step'],seconds=v['seconds'],device=v['device'],
                    initial={k:v['history'][0][k] for k in ('nll','coverage90','query_choice_regret')},
                    best={k:v['best']['metrics'][k] for k in ('nll','coverage90','query_choice_regret')},
                    network_parameters=v['network_parameters'],base_parameters=v['base_parameters'],history=v['history']))
                hashes[name]=sha256(s.RUN/'models'/f'{name}.pt')
                print('source checkpoint replay',name,'best',v['best']['step'],'/',v['steps'],flush=True)
    assert len(fits)==135 and len(rows)==82*32*9
    groups=sorted({r['group'] for r in rows});assert len(groups)==8
    metrics=('nll','coverage90','query_choice_regret');means={};group_values={};contexts={}
    for kind in ('N','M','PCA'):
        group_values[kind]={metric:[float(np.mean([r[metric] for r in rows if r['kind']==kind and r['group']==g]))
                                    for g in groups] for metric in metrics}
        means[kind]={metric:float(np.mean(vals)) for metric,vals in group_values[kind].items()}
        contexts[kind]={str(n):{metric:float(np.mean([
            np.mean([r[metric] for r in rows if r['kind']==kind and r['group']==g and r['context']==n])
            for g in groups])) for metric in metrics} for n in (5,10,20,40)}
    contrasts={}
    for other in ('M','PCA'):
        contrasts['N_vs_'+other]={metric:dict(
            mean=float(np.mean(np.asarray(group_values[other][metric])-np.asarray(group_values['N'][metric]))),
            by_group=(np.asarray(group_values[other][metric])-np.asarray(group_values['N'][metric])).tolist())
            for metric in ('nll','query_choice_regret')}
    recoveries={p.stem:json.loads(p.read_text()) for p in (s.RUN/'recovery').glob('*.json')}
    result=dict(scope='source validation used for checkpoint selection; descriptive only',
        means=means,contrasts=contrasts,by_context=contexts,groups=groups,group_values=group_values,
        jobs=135,completed_updates=sum(f['steps'] for f in fits),target_calls=0,confirmation_calls=0,
        untrained_checkpoint_counts={k:sum(f['kind']==k and f['best_step']==0 for f in fits) for k in ('N','M','PCA')},
        recovery_events=recoveries,cost_note='checkpointed successful compute only; interrupted compute may be missing',
        identity_sha256=sha256(s.RUN/'identity.json'),report_sha256=sha256(Path(__file__)),
        online_evaluation_complete=False,neural_necessity_established=False)
    write_json(s.OUT/'RESULTS.json',result);write_json(s.OUT/'FITS.json',fits);write_json(s.OUT/'ROWS.json',rows)
    write_json(s.OUT/'VERIFICATION.json',dict(passed=True,jobs=135,replayed_episodes=len(rows),
        maximum_absolute_metric_error=max_error,checkpoint_sha256=hashes,identity_sha256=sha256(s.RUN/'identity.json')))
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(5,3,figsize=(12,15))
    colors={'N':'tab:blue','M':'tab:orange','PCA':'tab:green'}
    for ax,space in zip(axes.flat,spaces):
        for f in fits:
            if f['space']!=space:continue
            ax.plot([h['step'] for h in f['history']],[h['nll'] for h in f['history']],
                color=colors[f['kind']],alpha=.65,label=f['kind'] if f['seed']==0 else None)
        ax.set_title('Space '+space);ax.set_xlabel('Optimizer updates');ax.set_ylabel('Source validation NLL')
        ax.set_yscale('symlog');ax.legend()
    fig.tight_layout()
    for ext in ('pdf','png'):fig.savefig(s.OUT/f'learning_curves.{ext}',dpi=150)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
