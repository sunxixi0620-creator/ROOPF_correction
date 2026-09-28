"""Audit completed development study identities, provenance and cost accounting."""
import json,zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from training_validation import OUT,ROOT,sha

def main():
    assert (OUT/'COMPLETE').exists()
    r=json.loads((OUT/'REPORT.json').read_text());assert r['counts']['validation_runs']==5184
    assert r['counts']['confirmation_runs']==1152 and r['counts']['auxiliary_runs']==864
    assert r['counts']['anchor_extra_training_points']==221184000
    dataset=json.loads((OUT/'dataset_manifest.json').read_text())
    for name,h in dataset.items():assert sha(OUT/(name+'.pt'))==h
    assert dataset['validation']!=dataset['confirmation']
    initial=[]
    for arm in ['legacy','shared_scale']:
        root=OUT/arm;protocol=json.loads((root/'protocol.json').read_text());assert protocol['source_sha256']==sha(ROOT/'scripts/training_validation.py')
        assert protocol['parent_sha256']==sha(ROOT/'results/retrain_d10_curated_20260928/resume.pt')
        history=json.loads((root/'history.json').read_text());assert [x['epoch'] for x in history]==list(range(81,161))
        assert history[-1]['continuation_points']==110592000
        vals=[json.loads(p.read_text()) for p in sorted(root.glob('validation_*.json'))]
        assert [x['epoch'] for x in vals]==list(range(80,161,10))
        selected=json.loads((root/'selection.json').read_text());expected=max(vals,key=lambda x:x['score'])
        assert selected['epoch']==expected['epoch'] and selected['score']==expected['score']
        a=torch.load(root/'selected.pt',map_location='cpu',weights_only=True)
        b=torch.load(root/f"epoch_{selected['epoch']:03d}.pt",map_location='cpu',weights_only=True)
        assert set(a)==set(b) and all(torch.equal(a[k],b[k]) for k in a)
        for v in vals:
            d=pd.DataFrame(v['rows']);assert len(d)==288 and not d.duplicated(['fid','instance','seed_index']).any()
            assert (d.nfe==300).all() and np.isfinite(d.final).all()
            raw=(d.initial_best-d.final)/d.initial_std
            assert (raw>=-1e-5).all()
            assert np.allclose(d.gain,np.maximum(raw,0),atol=1e-6)
            assert np.isclose(v['score'],d.bounded_gain.mean())
        initial.append(vals[0])
    assert initial[0]==initial[1], 'Continuation arms do not share starting outcomes'
    confirm=pd.read_csv(OUT/'confirmation_raw.csv');aux=pd.read_csv(OUT/'auxiliary/raw_results.csv')
    for d in [confirm,aux]:
        assert not d.duplicated(['method','fid','instance','seed_index']).any()
        assert (d.nfe==300).all() and np.isfinite(d.final).all()
    for path in (OUT/'auxiliary').glob('case_*_labels.npz'):
        with np.load(path) as x:
            assert len(x['q'])==14400 and np.isfinite(x['q']).all()
            assert ((x['q']>=0)&(x['q']<=1)).all()
            assert np.array_equal(x['beats_second'],x['candidate_fit']<x['second_fit'])
    artifact=ROOT/'artifacts/training_validation';manifest=json.loads((artifact/'manifest.json').read_text());paths=set()
    for part in manifest:
        assert sha(artifact/part['archive'])==part['sha256']
        with zipfile.ZipFile(artifact/part['archive']) as z:assert z.testzip() is None
        for path,info in part['files'].items():
            assert path not in paths;paths.add(path);assert sha(ROOT/path)==info['sha256']
    assert {str(p.relative_to(ROOT)) for p in OUT.rglob('*') if p.is_file()}==paths
    frozen={'anchor_policy_d10.pt':'a265d448ad4a2d191eed23dafa36e03e9475a7b95e824bb04d9a88715031a06d',
        'residual_selector_generated36_d10.pt':'42bac7c6792b6980ad149eff08ce4b2d93ce3ffba51d3d869dc0d1625b3a9d01'}
    for name,h in frozen.items():assert sha(ROOT/'checkpoints'/name)==h
    result={'all_checks_passed':True,'formal_trajectory_executions':7200,'formal_main_nfe':2160000,
        'teacher_points':1094400,'offline_extra_training_points':221184000,'original_checkpoints_unchanged':True,
        'archived_files':len(paths),'scope':'Single continuation random stream; repeated controls are not additional independent samples.'}
    (ROOT/'docs/experiments/TRAINING_VALIDATION_INTEGRITY.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':main()
