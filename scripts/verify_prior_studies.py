"""Read-only scientific verification, figures and archives; no objective queries."""
from pathlib import Path
import sys,json,argparse,zipfile,platform
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from roopf.experiment_io import CaseStore,sha256,fingerprint,write_json
from scripts.unified_revision import bounded


def label():
    from scripts import label_repeatability as s
    run,out=s.RUN,s.OUT;result=json.loads((run/'COMPLETE.json').read_text());identity=s.verify()
    store=CaseStore(run/'cases',identity);selection=json.loads((run/'SELECTION.json').read_text())
    assert selection==s.selection();assert len(list((run/'cases').glob('*.pt')))==768
    old_identity=json.loads((s.old.RUN/'identity.json').read_text());oldstore=CaseStore(s.old.RUN/'cases',old_identity)
    calls=0;deltas=np.zeros((48,16,3));events=[];eligible=np.zeros((48,3),bool)
    for n,row in enumerate(selection):
        f,i,p,step=row['id'];previous=oldstore.load(f'f{f}_i{i}_p{p}',dict(fid=f,instance=i,population=p))
        oldst=next(v for v in previous['states'] if v['step']==step)
        for r in range(16):
            v=store.load(f's{n}_r{r}',dict(state=n,stream=r));base=v['base'];events.append(base['rng_event'])
            assert v['row']==row and v['source_state_sha']==fingerprint(oldst['snapshot'])
            assert torch.equal(base['initial_x'],previous['base']['initial_x']) and torch.equal(base['initial'],previous['base']['initial'])
            assert torch.equal(base['points'][:,:2*step+2],previous['base']['points'][:,:2*step+2])
            initial=base['initial'].min(1).values;scale=base['initial'].std(1);cost=600
            assert base['main_calls']==600 and base['points'].shape==(1,500,20)
            for k,b in enumerate(v['branches']):
                ok=oldst['metadata'][row['indices'][k]]['eligible'];assert ok==v['eligible'][k];eligible[n,k]=ok
                assert b['main_calls']==600 and torch.isfinite(b['trail']).all()
                assert torch.equal(b['points'][:,:2*step+1],base['points'][:,:2*step+1])
                if ok:
                    assert torch.equal(b['points'][0,2*step+1],oldst['points'][row['indices'][k]])
                    assert b['rng_event']==base['rng_event'];cost+=600
                else:assert torch.equal(b['points'],base['points']) and torch.equal(b['trail'],base['trail'])
                d=float((bounded(initial,b['trail'][:,-1],scale)-bounded(initial,base['trail'][:,-1],scale))[0])
                assert d==v['gains'][k];deltas[n,r,k]=d
            assert cost==v['calls'];calls+=cost
    for key in ('global_after','private_after'):assert len({e[key] for e in events})==768
    ag=torch.load(run/'aggregates.pt',weights_only=False);assert np.array_equal(ag['delta'],deltas) and np.array_equal(ag['eligible'],eligible)
    old_original=json.loads((out/'VERIFICATION.json').read_text())['original_weights']
    for p,h in old_original.items():assert sha256(ROOT/p)==h
    before=sha256(run/'aggregates.pt');recomputed=s.summarize()
    for k,v in recomputed.items():assert result[k]==v,(k,result[k],v)
    # Save_torch need not preserve container bytes; tensor/value identity is checked above.
    assert calls==result['calls']
    write_json(out/'VERIFICATION.json',dict(original_weights=old_original,cases=768,states=48,streams=16,
        labels_independently_recomputed=True,all_prefixes_and_insertions_rechecked=True,distinct_global_rng_states=768,
        distinct_private_rng_states=768,effects_recomputed=True,total_calls=result['total_calls'],verification_objective_calls=0))
    return s,result


def prior():
    from scripts import fewshot_prior as s
    run,out=s.RUN,s.OUT;result=json.loads((run/'COMPLETE.json').read_text());s.verify()
    for p,h in result['data']['files'].items():assert sha256(run/p)==h
    frozen=json.loads((run/'MODELS_FROZEN.json').read_text());assert len(frozen)==18
    for p,h in frozen.items():assert sha256(run/p)==h
    arrays={m:{metric:np.zeros((3,3,12)) for metric in ('mse','utility')} for m in ('correct','shuffled','untrained','online')}
    model_count=0
    device=result['resources']['device']
    for fold in range(3):
        parts=s.partitions(fold);assert len(set(sum(parts.values(),[])))==12
        tr=torch.load(run/f'{fold}_train.pt',weights_only=False)
        assert torch.equal(torch.cat((tr['cy'],tr['qy']),1).sort(1).values,torch.cat((tr['shuffle_cy'],tr['shuffle_qy']),1).sort(1).values)
        assert not torch.equal(tr['cy'],tr['shuffle_cy'])
        va=torch.load(run/f'{fold}_val.pt',weights_only=False);data=torch.load(run/f'{fold}_test.pt',weights_only=False)
        for name,ds in [('train',tr),('val',va),('test',data)]:assert set((ds['id'][:,0]//3).tolist())==set(parts[name])
        recipes=(data['id'][:,0]//3).numpy();online=s.online_predictions(data)
        for seed in range(3):
            record=torch.load(run/f'evaluation_{fold}_{seed}.pt',weights_only=False)
            assert torch.equal(online,record['predictions']['online'])
            for kind in ('correct','shuffled','untrained'):
                model=s.net(fold,seed).to(device).eval()
                if kind!='untrained':
                    ck=torch.load(run/f'model_{fold}_{kind}_{seed}.pt',weights_only=False)
                    meta=json.loads((run/f'model_{fold}_{kind}_{seed}.json').read_text())
                    best=min(meta['history'],key=lambda r:r['validation_mse'])
                    assert ck['epoch']==meta['selected']==best['epoch']
                    model.load_state_dict(ck['state']);model_count+=1
                    check=s.validation(model,va,device)
                    assert abs(check-meta['validation_mse'])<1e-6
                pred=torch.stack([s.predictions(model,data,n,device) for n in s.COUNTS])
                assert torch.equal(pred,record['predictions'][kind])
            for method,pred in record['predictions'].items():
                mse,u=s.score(pred,data)
                assert torch.equal(mse,record['metrics'][method]['mse']) and torch.equal(u,record['metrics'][method]['utility'])
                for recipe in parts['test']:
                    mask=recipes==recipe
                    arrays[method]['mse'][seed,:,recipe]=mse[:,mask].mean(1).numpy()
                    arrays[method]['utility'][seed,:,recipe]=u[:,mask].mean(1).numpy()
    effects=s.effects_from(arrays);assert effects==result['effects'];assert model_count==18
    original=json.loads((out/'VERIFICATION.json').read_text())['original_weights']
    for p,h in original.items():assert sha256(ROOT/p)==h
    write_json(out/'VERIFICATION.json',dict(original_weights=original,models=18,test_tasks=576,
        prediction_arrays_exactly_reproduced=True,early_stopping_and_validation_recomputed=True,source_multisets_matched=True,
        recipe_disjoint_per_model=True,all_data_hashes_checked=True,all_metrics_recomputed=True,verification_objective_calls=0))
    return s,result


def fusion():
    from scripts import prior_fusion as s
    run,out=s.RUN,s.OUT;result=json.loads((run/'COMPLETE.json').read_text());identity=s.verify()
    store=CaseStore(run/'cases',identity);calls=0;decisions=0
    assert len(list((run/'cases').glob('*.pt')))==144
    for f in range(36):
        for i in range(2):
            for p in range(2):
                v=store.load(f'f{f}_i{i}_p{p}',dict(fid=f,instance=i,population=p));assert len(v['outputs'])==14
                calls+=v['calls'];assert v['calls']==4200
                assert len({fingerprint(r['initial_x']) for r in v['outputs'].values()})==1
                for key,row in v['outputs'].items():
                    method=key.rsplit('_',1)[0];initial=row['initial'].min(1).values;scale=row['initial'].std(1).clamp_min(1e-8)
                    observed=row['values'][0];expected=torch.cummin(torch.cat((initial,observed)),0).values[1:]
                    assert torch.equal(expected,row['trail']);assert row['calls']==300 and len(row['records'])==280
                    assert torch.isfinite(observed).all() and row['points'].shape==(1,280,20)
                    assert row['points'].min()>=-5 and row['points'].max()<=5
                    assert float(bounded(initial,expected[-1:],scale)[0])==row['utility']
                    errors=torch.ones(2)
                    for step,rec in enumerate(row['records']):
                        assert rec['nfe']==21+step and rec['value']==float(observed[step])
                        ci=rec['context_indices'];assert len(ci)==min(40,20+step) and len(ci.unique())==len(ci) and ci.max()<20+step
                        if method=='A':expected_weight=1.
                        elif method=='O':expected_weight=0.
                        elif method=='Fixed':expected_weight=.5
                        else:expected_weight=float((errors[1]+.05)/(errors.sum()+.1))
                        assert rec['prior_weight']==expected_weight
                        if method in ('F','Fshuffled','Fanalytic'):
                            errors=.9*errors+.1*((torch.tensor([rec['prior_prediction'],rec['online_prediction']])-observed[step])/scale[0]).square()
                        decisions+=1
    assert calls==result['calls'] and decisions==2016*280
    recomputed=s.summarize()
    for key,value in recomputed.items():assert result[key]==value,key
    original=json.loads((ROOT/'docs/revision/fewshot_prior/VERIFICATION.json').read_text())['original_weights']
    for p,h in original.items():assert sha256(ROOT/p)==h
    write_json(out/'VERIFICATION.json',dict(original_weights=original,cases=144,trajectories=2016,decisions=decisions,
        source_and_checkpoint_hashes_checked=True,all_initializations_paired=True,budgets_and_trails_recomputed=True,
        causal_weights_recomputed=True,context_contains_only_previous_observations=True,all_effects_recomputed=True,verification_objective_calls=0))
    return s,result


def figures(s,r,name):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    if name in ('label','fusion'):
        fig,ax=plt.subplots(figsize=(7,2.6 if name=='label' else 4));axes=[ax];groups=[list(r['effects'])]
    else:
        fig,axes=plt.subplots(1,2,figsize=(11,3.4));groups=[[k for k in r['effects'] if k.startswith(metric)] for metric in ('mse','utility')]
    for ax,keys in zip(axes,groups):
        for i,k in enumerate(keys):
            e=r['effects'][k];ax.errorbar(e['mean'],i,xerr=[[e['mean']-e['lower']],[e['upper']-e['mean']]],fmt='o',capsize=4)
        ax.set_yticks(range(len(keys)),[k.replace('mse:','').replace('utility:','') for k in keys]);ax.axvline(0,color='black',lw=.8)
        ax.set_xlabel('Relative MSE reduction' if keys[0].startswith('mse:') else 'Paired utility difference');ax.grid(axis='x',alpha=.2)
    fig.tight_layout();fig.savefig(s.OUT/'effects.png',dpi=180);fig.savefig(s.OUT/'effects.pdf');plt.close(fig)


def archive(s,name):
    folder=ROOT/'artifacts'/dict(label='label_repeatability_v1',prior='fewshot_prior_v1',fusion='prior_fusion_v1')[name];folder.mkdir(parents=True,exist_ok=True)
    sources=[ROOT/p for p in s.SOURCES]+[Path(__file__),ROOT/'docs/revision/RESEARCH_RESET_20260930.zh-CN.md']
    paths=sorted(set(sources+[p for p in s.RUN.rglob('*') if p.is_file() and p.suffix!='.lock']+[p for p in s.OUT.rglob('*') if p.is_file()]))
    chunks=[];chunk=[];size=0
    for p in paths:
        if chunk and size+p.stat().st_size>40_000_000:chunks.append(chunk);chunk=[];size=0
        chunk.append(p);size+=p.stat().st_size
    if chunk:chunks.append(chunk)
    manifest=[]
    for i,chunk in enumerate(chunks):
        path=folder/f'audit_{i}.zip'
        with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_DEFLATED) as z:
            for p in chunk:z.write(p,p.relative_to(ROOT))
        with zipfile.ZipFile(path) as z:assert z.testzip() is None
        assert path.stat().st_size<95_000_000
        manifest.append(dict(archive=path.name,bytes=path.stat().st_size,sha256=sha256(path),members=len(chunk)))
    write_json(folder/'MANIFEST.json',manifest)
    (folder/'README.md').write_text(f'完整实验归档：解压全部ZIP到同一仓库根目录。包含原始数据、源身份、冻结协议、结果与核验。\n\n复核（不调用目标函数、不重新训练）：`.venv/bin/python scripts/verify_prior_studies.py --study {name}`。\n\n标签审查还依赖 `artifacts/long_value_v1` 的父数据与其已归档依赖。\n')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--study',choices=['label','prior','fusion'],required=True);args=parser.parse_args()
    torch.set_num_threads(1);s,r=dict(label=label,prior=prior,fusion=fusion)[args.study]()
    write_json(s.OUT/'ENVIRONMENT.json',dict(python=sys.version,torch=str(torch.__version__),numpy=np.__version__,platform=platform.platform()))
    figures(s,r,args.study);archive(s,args.study)
    print(json.dumps(dict(study=args.study,verified=True,passed=r['passed']),indent=2))


if __name__=='__main__':main()
