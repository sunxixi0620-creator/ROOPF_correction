"""Identify historical configuration for four non-replaying archived cases; never overwrite data."""
import copy,json
import numpy as np
import pandas as pd
import torch
from residual_target_study import ROOT,OUT,execute,payload,sha
DEST=ROOT/'docs/revision/feature_contract_probe'
if __name__=='__main__':
 oldpath=ROOT/'docs/revision/feature_contract_probe/recovered_legacy_tasks.pt';qs=torch.load(oldpath,weights_only=False,map_location='cpu');rows=[]
 for i in [24,25,26,27]:
  q=copy.deepcopy(qs[i]);q['seed']-=18000000
  t=execute(q,True);d=pd.DataFrame(t[2]);cols=[]
  for name in payload()['feature_names']:
   if name.startswith('op_'):col=(d.op_name==name[3:]).to_numpy(dtype=np.float32)
   else:
    raw=pd.to_numeric(d[name.removesuffix('_missing')],errors='coerce').to_numpy(dtype=np.float32)
    col=(~np.isfinite(raw)).astype(np.float32) if name.endswith('_missing') else np.nan_to_num(raw,nan=0.,posinf=0.,neginf=0.)
   cols.append(col)
  saved=np.load(OUT/f'case_{i:03d}.npz');fit=d.candidate_fit.to_numpy().reshape(-1,38)
  z=dict(index=i,fid=q['fid'],legacy_trace_exact=bool(np.array_equal(saved['trail'],t[0].numpy())),legacy_all_features_exact=bool(np.array_equal(saved['x'],np.stack(cols,1))),legacy_teacher_values_exact=bool(np.array_equal(saved['fit'],fit)),legacy_labels_exact=bool(np.array_equal(saved['incumbent'],d.improved_best.to_numpy()) and np.array_equal(saved['second'],(fit<fit[:,1:2]).reshape(-1))))
  rows.append(z);print(z,flush=True)
 (DEST/'legacy_case_recovery.json').write_text(json.dumps(dict(script=sha(__file__),legacy_tasks_sha256=sha(oldpath),population_seed_offset=1000000,policy_seed_offset=2000000,cases=rows,scope='Exact recovery identifies old preflight seed configuration; original cached files preserved',main_points=4800,teacher_points=60800),indent=2))
