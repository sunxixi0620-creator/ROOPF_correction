"""Verify the structural transfer diagnostics without new function evaluations."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
from pathlib import Path
import sys,json,zipfile,platform,argparse
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import shared_structure as s
from scripts import radial_prior_audit as r
from roopf.experiment_io import CaseStore,sha256,fingerprint,write_json


def verify_radial():
    result=json.loads((r.RUN/'COMPLETE.json').read_text());identity=json.loads((r.RUN/'identity.json').read_text())
    for p,h in identity['sources'].items():assert sha256(ROOT/p)==h
    for p,h in identity['models'].items():assert sha256(r.parent.RUN/p)==h
    for p,h in identity['data'].items():assert sha256(r.parent.RUN/p)==h
    saved=torch.load(r.RUN/'aggregates.pt',weights_only=False)
    for fold in range(3):
        data=torch.load(r.parent.RUN/f'{fold}_test.pt',weights_only=False);recipes=data['id'][:,0]//3
        for seed in range(3):
            rec=torch.load(r.RUN/f'fold{fold}_seed{seed}.pt',weights_only=False)
            previous=torch.load(r.parent.RUN/f'evaluation_{fold}_{seed}.pt',weights_only=False)
            assert torch.equal(previous['predictions']['correct'],rec['predictions']['correct'])
            device=result['device'];model=r.parent.net(fold,seed).to(device).eval()
            model.load_state_dict(torch.load(r.parent.RUN/f'model_{fold}_correct_{seed}.pt',weights_only=False)['state'])
            for ci,n in enumerate(r.parent.COUNTS):
                cx,cy,qx,_,_,_=r.parent.standardized(data,n)
                for name,kind in [('fixed_radial','fixed'),('fitted_radial','fit'),('radial_linear','linear')]:assert torch.equal(r.radial(cx,cy,qx,kind),rec['predictions'][name][ci])
                perm=rec['context_permutations'][n];assert torch.equal(perm.sort(1).values,torch.arange(n).expand(len(cx),-1))
                permcy=cy.gather(1,perm);pieces=[]
                with torch.no_grad():
                    for ix in torch.arange(len(cx)).split(16):pieces.append(model(cx[ix].to(device),permcy[ix].to(device),qx[ix].to(device)).cpu())
                assert torch.equal(torch.cat(pieces),rec['predictions']['permuted_context'][ci])
            for name,pred in rec['predictions'].items():
                mse,u=r.parent.score(pred,data)
                assert torch.equal(mse,rec['metrics'][name]['mse']) and torch.equal(u,rec['metrics'][name]['utility'])
                for recipe in r.parent.partitions(fold)['test']:
                    assert np.array_equal(mse[:,recipes==recipe].mean(1).numpy(),saved['metrics'][name]['mse'][seed,:,recipe])
                    assert np.array_equal(u[:,recipes==recipe].mean(1).numpy(),saved['metrics'][name]['utility'][seed,:,recipe])
    write_json(r.OUT/'VERIFICATION.json',dict(archived_predictions_and_metrics_recomputed=True,context_permutations_preserve_labels=True,
        permuted_context_predictions_exactly_reproduced=True,source_hashes_checked=True,objective_calls=0,posthoc_only=True))


def verify_alignment():
    from scripts import alignment_search as a
    from roopf.experiment_io import seed_for
    from scripts.unified_revision import bounded
    result=json.loads((a.RUN/'COMPLETE.json').read_text());identity=a.verify();store=CaseStore(a.RUN/'cases',identity)
    assert len(list((a.RUN/'cases').glob('*.pt')))==96
    checked=0;calls=0
    for group in range(12):
        for t in range(4):
            for mi in range(2):
                v=store.load(f'g{group}_t{t}_m{mi}',dict(group=group,task=t,mismatch=bool(mi)));calls+=v['calls'];assert v['calls']==4200
                assert len({fingerprint(o['initial_x']) for o in v['outputs'].values()})==1
                for key,row in v['outputs'].items():
                    method=key.rsplit('_',1)[0];initial=row['initial'].min();scale=row['initial'].std()
                    trail=torch.cummin(torch.cat((initial[None],row['values'])),0).values[1:]
                    assert torch.equal(trail,row['trail']);assert row['calls']==300 and len(row['records'])==280
                    assert float(bounded(initial,trail[-1],scale))==row['utility']
                    for j,rec in enumerate(row['records']):
                        expected=1. if method=='T' else 0. if method=='O' else .5 if method=='Fixed' else float(rec['loo_t']<rec['loo_o'])
                        assert rec['weight']==expected and rec['nfe']==21+j and rec['value']==float(row['values'][j]);checked+=1
                    if method!='S':continue
                    seed=int(key.rsplit('_',1)[1]);B=torch.load(a.parent.RUN/f'model_g{group}_s{seed}.pt',weights_only=False)['B']
                    for step in (0,79,279):
                        x=torch.cat((row['initial_x'],row['points'][:step]));y=torch.cat((row['initial'],row['values'][:step]));rec=row['records'][step]
                        tm=a.fit(a.features(x,B),y);om=a.fit(a.features(x),y)
                        assert float(tm['loo'])==rec['loo_t'] and float(om['loo'])==rec['loo_o']
                        gen=torch.Generator().manual_seed(seed_for('alignment_search_v1','pool',group,t,step));remaining=(300-20-step)/300
                        pool=torch.cat((10*torch.rand(64,20,generator=gen,dtype=torch.float64)-5,
                            (x[y.argmin()]+10*(.05+.35*remaining)*torch.randn(64,20,generator=gen,dtype=torch.float64)).clamp(-5,5)))
                        tp=a.predict(tm,a.features(pool,B));op=a.predict(om,a.features(pool));w=rec['weight'];score=w*tp+(1-w)*op
                        score[torch.cdist(pool,x).min(1).values<1e-8]=torch.inf;index=int(score.argmin())
                        assert torch.equal(pool[index],row['points'][step]) and float(tp[index])==rec['pred_t'] and float(op[index])==rec['pred_o']
    assert calls==result['calls'] and checked==1344*280
    recomputed=a.summarize()
    for k,value in recomputed.items():assert result[k]==value,k
    original=json.loads((s.OUT/'VERIFICATION.json').read_text())['original_weights']
    for p,h in original.items():assert sha256(ROOT/p)==h
    write_json(a.OUT/'VERIFICATION.json',dict(original_weights=original,cases=96,trajectories=1344,decisions=checked,
        causal_selectors_checked=True,selected_states_refitted_and_reselected=864,all_metrics_recomputed=True,
        initialization_and_budget_verified=True,objective_calls=0))
    return a,result


def verify_shared():
    result=json.loads((s.RUN/'COMPLETE.json').read_text());identity=s.verify();sources=CaseStore(s.RUN/'source',identity)
    models=json.loads((s.RUN/'MODELS_FROZEN.json').read_text());assert len(models)==36
    for p,h in models.items():assert sha256(s.RUN/p)==h
    for group in range(12):
        for seed in range(3):
            v=sources.load(f'g{group}_s{seed}',dict(group=group,seed=seed));model=s.fit_subspace(v['x'],v['y'])
            assert torch.equal(model['B'],v['model']['B']) and torch.equal(model['H'],v['model']['H'])
            assert v['calls']==5120
    target=CaseStore(s.RUN/'target',identity);bounds=type('Bounds',(),{'fun':dict(xlb=-5.,xub=5.)})()
    from roopf.online_portfolio import build_online
    for group in range(12):
        for mi in range(2):
            v=target.load(f'g{group}_m{mi}',dict(group=group,mismatch=bool(mi)));assert v['calls']==32*552
            x,y=v['x'],v['y'];assert x.shape==(32,552,20) and y.shape==(32,552)
            assert torch.allclose(x[:,296:].square().sum(-1),torch.full((32,256),20.25,dtype=x.dtype),atol=1e-12)
            if mi:
                other=target.load(f'g{group}_m0',dict(group=group,mismatch=False));assert torch.equal(other['x'],x)
            for n in s.COUNTS:
                cx=x[:,:n];mean=y[:,:n].mean(1,keepdim=True);std=y[:,:n].std(1,keepdim=True,unbiased=False).clamp_min(1e-8);cy=(y[:,:n]-mean)/std;qx=x[:,40:]
                shared=dict(RadialLinear=s.predict_ridge(cx,cy,qx,'radial'),OnlineFull=s.predict_ridge(cx,cy,qx,'full'))
                with torch.no_grad():mu,_=build_online(20,300).surrogate.predict(cx,y[:,:n],qx,bounds)
                shared['OnlineRidge']=(mu-mean)/std
                for seed in range(3):
                    from roopf.experiment_io import seed_for
                    g=torch.Generator().manual_seed(seed_for('shared_random_subspace',group,seed));random=torch.linalg.qr(torch.randn(20,3,generator=g,dtype=torch.float64)).Q
                    a=torch.load(s.RUN/f'model_g{group}_s{seed}.pt',weights_only=False)['B'];b=torch.load(s.RUN/f'model_g{(group+1)%12}_s{seed}.pt',weights_only=False)['B']
                    preds=dict(shared)
                    for name,B in [('Learned',a),('Wrong',b),('Random',random)]:preds[name]=s.predict_ridge(cx,cy,qx,'subspace',B)
                    for name,pred in preds.items():assert torch.equal(pred,v['predictions'][f'{name}_{n}_{seed}'])
    recalculated=s.summarize()
    for k,v in recalculated.items():assert result[k]==v,k
    original=json.loads((ROOT/'docs/revision/prior_fusion/VERIFICATION.json').read_text())['original_weights']
    for p,h in original.items():assert sha256(ROOT/p)==h
    write_json(s.OUT/'VERIFICATION.json',dict(original_weights=original,models=36,source_refits_exact=True,
        all_predictions_recomputed=True,metrics_and_intervals_recomputed=True,matched_mismatch_coordinates_paired=True,
        shell_radii_verified=True,source_only_representation_fit=True,objective_calls=0))


def archive(study,name):
    folder=ROOT/'artifacts'/name;folder.mkdir(parents=True,exist_ok=True)
    paths=sorted(set([ROOT/p for p in study.SOURCES]+[Path(__file__)]+[p for p in study.RUN.rglob('*') if p.is_file() and p.suffix!='.lock']+[p for p in study.OUT.rglob('*') if p.is_file()]))
    chunks=[];chunk=[];size=0
    for p in paths:
        if chunk and size+p.stat().st_size>40000000:chunks.append(chunk);chunk=[];size=0
        chunk.append(p);size+=p.stat().st_size
    if chunk:chunks.append(chunk)
    manifest=[]
    for j,chunk in enumerate(chunks):
        path=folder/f'audit_{j}.zip'
        with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_DEFLATED) as z:
            for p in chunk:z.write(p,p.relative_to(ROOT))
        with zipfile.ZipFile(path) as z:assert z.testzip() is None
        assert path.stat().st_size<95000000
        manifest.append(dict(archive=path.name,bytes=path.stat().st_size,sha256=sha256(path),members=len(chunk)))
    write_json(folder/'MANIFEST.json',manifest)
    (folder/'README.md').write_text('完整归档解压到仓库根目录；radial诊断依赖fewshot_prior_v1父数据。复核两项研究（0目标查询）：`.venv/bin/python scripts/verify_structure.py`。\n')


def figures():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    a=torch.load(s.RUN/'aggregates.pt',weights_only=False)
    fig,axes=plt.subplots(1,2,figsize=(10,3.7))
    for mi,ax in enumerate(axes):
        for method in ('Learned','RadialLinear','OnlineFull','OnlineRidge'):
            values=a[method]['regret'][mi,1].mean((0,2));ax.plot(s.COUNTS,values,marker='o',label=method)
        ax.set_title('Matched source' if mi==0 else 'Misaligned source');ax.set_xlabel('Observed target points');ax.set_ylabel('Normalized candidate regret (sphere)');ax.grid(alpha=.25)
    axes[1].legend(fontsize=8);fig.tight_layout();fig.savefig(s.OUT/'effects.png',dpi=180);fig.savefig(s.OUT/'effects.pdf');plt.close(fig)


def alignment_figures(a,result):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,3.7))
    for mi,ax in enumerate(axes):
        for method in ('T','O','S','RadialS'):
            ax.plot([20,40,100,200,300],result['mean_curves'][method][mi],marker='o',label=method)
        ax.set_title('Matched source' if mi==0 else 'Misaligned source');ax.set_xlabel('Objective evaluations');ax.set_ylabel('Bounded improvement');ax.grid(alpha=.25)
    axes[1].legend();fig.tight_layout();fig.savefig(a.OUT/'curves.png',dpi=180);fig.savefig(a.OUT/'curves.pdf');plt.close(fig)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--alignment',action='store_true');args=parser.parse_args()
    torch.set_num_threads(1)
    if args.alignment:
        a,result=verify_alignment()
        write_json(a.OUT/'ENVIRONMENT.json',dict(python=sys.version,torch=str(torch.__version__),numpy=np.__version__,platform=platform.platform()))
        alignment_figures(a,result)
        archive(a,'alignment_search_v1');print('Alignment study verified; no objective queries; archive checked.',flush=True);return
    verify_radial();verify_shared();figures()
    for study,name in [(r,'radial_prior_audit_v1'),(s,'shared_structure_v1')]:
        write_json(study.OUT/'ENVIRONMENT.json',dict(python=sys.version,torch=str(torch.__version__),numpy=np.__version__,platform=platform.platform()))
        archive(study,name)
    print('Both studies verified; no objective queries; archives checked.',flush=True)


if __name__=='__main__':main()
