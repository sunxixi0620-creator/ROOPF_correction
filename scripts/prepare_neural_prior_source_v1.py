"""Response-blind source holdout and source-only PCA caches, no neural fitting."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import argparse,hashlib,json,sys,time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import hpob_transfer_development_v1 as parent
from roopf.hpob_transfer_v1 import build_gp,Predictor
from roopf.probabilistic_prior_v1 import pca_directions,pca_prior
from roopf.experiment_io import CaseStore,write_json,sha256,seed_for
RUN=ROOT/'results/neural_prior_source_v1';OUT=ROOT/'docs/revision/neural_prior_source_v1'
PROTOCOL=ROOT/'docs/experiments/NEURAL_PRIOR_SOURCE_STAGE_V1.json'
SOURCES=('scripts/prepare_neural_prior_source_v1.py','scripts/check_probabilistic_prior_v1.py',
         'roopf/probabilistic_prior_v1.py','docs/experiments/NEURAL_PRIOR_SOURCE_STAGE_V1.json')


def split():
    parent.verify();RUN.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
    tasks=parent.tasks('source');groups=sorted({t['group'] for t in tasks},
        key=lambda g:hashlib.sha256(('hpob_neural_source_valid_v1|'+g).encode()).hexdigest())
    spaces={s:{t['group'] for t in tasks if t['space']==s} for s in sorted({t['space'] for t in tasks},key=int)}
    selected=[];uncovered=set(spaces)
    def admissible(g):return all(len(gs-set(selected)-{g})>=3 for gs in spaces.values())
    while uncovered:
        eligible=[g for g in groups if g not in selected and admissible(g)]
        assert eligible,'Cannot cover validation spaces'
        chosen=max(eligible,key=lambda g:sum(g in spaces[s] for s in uncovered))
        assert any(chosen in spaces[s] for s in uncovered)
        selected.append(chosen);uncovered={s for s in uncovered if chosen not in spaces[s]}
    for g in groups:
        if len(selected)==8:break
        if g not in selected and admissible(g):selected.append(g)
    assert len(selected)==8
    counts={s:dict(train=len(gs-set(selected)),validation=len(gs&set(selected))) for s,gs in spaces.items()}
    assert all(c['train']>=3 and c['validation']>=1 for c in counts.values())
    result=dict(parent_split_sha256=sha256(parent.SPLIT),validation_groups=selected,
        train_groups=[g for g in groups if g not in selected],spaces=counts,
        tasks=[dict(**t,source_role='validation' if t['group'] in selected else 'train') for t in tasks],
        response_blind=True,source_training_updates=0,target_calls=0,confirmation_calls=0)
    path=OUT/'SOURCE_SPLIT.json'
    if path.exists():assert json.loads(path.read_text())==result
    else:write_json(path,result)
    print('source groups',len(groups)-8,'train, 8 validation;',counts,flush=True)


def freeze():
    split()
    contracts=json.loads((OUT/'CONTRACTS.json').read_text())
    assert contracts['source_sha256']==sha256(ROOT/'roopf/probabilistic_prior_v1.py')
    assert contracts['test_sha256']==sha256(ROOT/'scripts/check_probabilistic_prior_v1.py')
    path=RUN/'identity.json'
    if not path.exists():write_json(path,dict(parent=sha256(parent.RUN/'identity.json'),
        split=sha256(OUT/'SOURCE_SPLIT.json'),sources={p:sha256(ROOT/p) for p in SOURCES},time=time.time()))
    verify();write_json(OUT/'IDENTITY.json',json.loads(path.read_text()))


def verify():
    parent.verify();z=json.loads((RUN/'identity.json').read_text())
    assert z['parent']==sha256(parent.RUN/'identity.json') and z['split']==sha256(OUT/'SOURCE_SPLIT.json')
    for p,h in z['sources'].items():assert sha256(ROOT/p)==h,p
    return z


def cache(device):
    torch.set_num_threads(1);identity=verify();store=CaseStore(RUN/'pca',identity)
    split_info=json.loads((OUT/'SOURCE_SPLIT.json').read_text());tasks=split_info['tasks'];start=time.perf_counter();rows=[]
    parent_store=CaseStore(parent.RUN/'models',parent.verify())
    exports=json.loads((parent.RUN/'EXPORTS.json').read_text())
    for space in sorted(split_info['spaces'],key=int):
        task_rows=[t for t in tasks if t['space']==space];train=[t for t in task_rows if t['source_role']=='train']
        case=dict(space=space)
        with store.lock(space):
            saved=store.load(space,case)
            if saved is not None:
                rows.append(saved['summary']);continue
            points={};data_hashes={}
            for t in task_rows:
                # Strictly restricted to the source_data allowlist. Query/target
                # labels are never read; even source y is unnecessary for PCA.
                name=parent.key(t);path=parent.RUN/'source_data'/f'{name}.npz'
                h=sha256(path);assert h==exports['files'][str(path.relative_to(parent.RUN))]
                with np.load(path) as data:points[name]=torch.from_numpy(data['x'].copy())
                data_hashes[name]=h
            pool=torch.cat([points[parent.key(t)] for t in train]).numpy();pool=np.unique(pool,axis=0)
            order=np.random.default_rng(seed_for('hpob_pca_reference_v1',space)).permutation(len(pool))[:2048]
            reference=torch.from_numpy(pool[order]);queries=torch.cat([points[parent.key(t)] for t in task_rows])
            combined=torch.cat((reference,queries)).to(device);predictions=[];model_hashes={}
            for t in train:
                original={k:v for k,v in t.items() if k!='source_role'};name=parent.key(t)
                v=parent_store.load(name,dict(task=original));assert v is not None
                model=build_gp(v['x'],v['z'])
                with torch.no_grad():
                    for n,p in model.named_parameters():p.copy_(v['info']['parameters'][n])
                model.to(device);model.eval()
                predictions.append(Predictor(model).moments(combined,False)[0].cpu())
                model_hashes[name]=sha256(parent.RUN/'models'/f'{name}.pt')
            values=torch.stack(predictions)
            directions,diagnosis=pca_directions(values[:,:len(reference)],8)
            task_priors={};offset=len(reference)
            for t in task_rows:
                name=parent.key(t);n=len(points[name]);mean,features=pca_prior(values[:,offset:offset+n],directions)
                task_priors[name]=dict(mean=mean,features=features,x=points[name],source_role=t['source_role'])
                offset+=n
            summary=dict(space=space,training_tasks=len(train),validation_tasks=len(task_rows)-len(train),
                reference_points=len(reference),**diagnosis)
            store.save(space,case,dict(summary=summary,priors=task_priors,directions=directions,
                reference_x=reference,source_predictions=values[:,:len(reference)],
                source_order=[parent.key(t) for t in train],source_model_hashes=model_hashes,
                input_hashes=data_hashes,device=device,target_calls=0))
            rows.append(summary);print('PCA',space,'rank',diagnosis['active_rank'],'train',len(train),'validation',len(task_rows)-len(train),flush=True)
    write_json(OUT/'PCA_PREPARATION.json',dict(spaces=rows,seconds=time.perf_counter()-start,
        device=device,neural_updates=0,additional_labels=0,target_calls=0,confirmation_calls=0))


def assess_parent():
    verify()
    results_path=parent.OUT/'RESULTS.json';checks_path=parent.OUT/'VERIFICATION.json'
    if not results_path.exists() or not checks_path.exists():
        print('Parent full results/verification still pending; no neural training authorized by entry gate.',flush=True);return
    z=json.loads(results_path.read_text());checks=json.loads(checks_path.read_text())
    assert checks['passed'] and checks['cases']==1320 and checks['calls']==138600
    assert z['identity_sha256']==sha256(parent.RUN/'identity.json')
    decision=dict(parent_results_sha256=sha256(results_path),parent_verification_sha256=sha256(checks_path),
        entry_gate_passed=bool(z['neural_entry_gate']),observed_primary=z['contrasts']['R_vs_O']['cold_auc'],
        observed_terminal=z['contrasts']['R_vs_O']['terminal'],
        real_neural_updates=0,target_stage_started=False,
        next_action='freeze episode generator and training sources, then train source priors' if z['neural_entry_gate'] else
          'stop before neural fitting; diagnose parent transfer failure using existing trajectories only')
    write_json(OUT/'ENTRY_DECISION.json',decision);print(json.dumps(decision,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['split','freeze','cache','assess_parent'])
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda');a=p.parse_args()
    cache(a.device) if a.phase=='cache' else globals()[a.phase]()
