"""Audit PCA source membership, prediction reconstruction and PSD without y."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import prepare_neural_prior_source_v1 as s
from roopf.hpob_transfer_v1 import build_gp,Predictor
from roopf.probabilistic_prior_v1 import pca_prior
from roopf.experiment_io import CaseStore,sha256,write_json


def main():
    torch.set_num_threads(1);identity=s.verify();split=json.loads((s.OUT/'SOURCE_SPLIT.json').read_text())
    models=CaseStore(s.parent.RUN/'models',s.parent.verify());pca=CaseStore(s.RUN/'pca',identity)
    errors=[];points=0;rows=[]
    for space in sorted(split['spaces'],key=int):
        saved=pca.load(space,dict(space=space));assert saved is not None
        allowed=[t for t in split['tasks'] if t['space']==space and t['source_role']=='train']
        assert set(saved['source_order'])=={s.parent.key(t) for t in allowed}
        query=[];expected=[]
        for name,v in saved['priors'].items():
            indices=torch.linspace(0,len(v['x'])-1,5).long()
            query.append(v['x'][indices]);expected.append((v['mean'][indices],v['features'][indices]))
            assert v['source_role']==next(t['source_role'] for t in split['tasks'] if s.parent.key(t)==name)
        query=torch.cat(query);predictions=[]
        for name in saved['source_order']:
            t=next(t for t in allowed if s.parent.key(t)==name);original={k:v for k,v in t.items() if k!='source_role'}
            v=models.load(name,dict(task=original))
            assert sha256(s.parent.RUN/'models'/f'{name}.pt')==saved['source_model_hashes'][name]
            model=build_gp(v['x'],v['z'])
            with torch.no_grad():
                for k,p in model.named_parameters():p.copy_(v['info']['parameters'][k])
            model.eval();predictions.append(Predictor(model).moments(query,False)[0])
        mean,features=pca_prior(torch.stack(predictions),saved['directions'])
        mean_error=float((mean-torch.cat([v[0] for v in expected])).abs().max())
        feature_error=float((features-torch.cat([v[1] for v in expected])).abs().max())
        errors.extend((mean_error,feature_error));points+=len(query)*len(allowed)
        assert max(mean_error,feature_error)<1e-7,(space,mean_error,feature_error)
        cov=features[:20]@features[:20].T
        assert torch.linalg.eigvalsh(cov).min()>-1e-10
        rows.append(dict(space=space,mean_error=mean_error,feature_error=feature_error,active_rank=saved['summary']['active_rank']))
    result=dict(spaces=15,source_point_replays=points,max_error=max(errors),rows=rows,
        tolerance=1e-7,passed=True,validation_gp_excluded=True,source_y_read=False,target_calls=0,
        confirmation_calls=0,neural_updates=0)
    write_json(s.OUT/'PCA_VERIFICATION.json',result);print('PCA verified',points,max(errors),flush=True)


if __name__=='__main__':main()
