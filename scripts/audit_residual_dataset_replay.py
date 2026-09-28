"""Read-only replay audit of stored residual behavior trajectories."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
import multiprocessing,json
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import torch
from residual_target_study import OUT,execute,ROOT,sha
DEST=ROOT/'docs/revision/feature_contract_probe'
def worker(i):
 q=torch.load(OUT/'tasks.pt',weights_only=False,map_location='cpu')[i]
 a=execute(q,False);stored=np.load(OUT/f'case_{i:03d}.npz')['trail'];current=a[0].numpy()
 equal=np.array_equal(stored,current);out=dict(index=i,fid=q['fid'],split=q['split'],exact=bool(equal),max_abs_difference=float(np.abs(stored-current).max()),initial_55_rounds_exact=bool(np.array_equal(stored[:,:55],current[:,:55])),endpoint_exact=bool(np.array_equal(stored[:,-1],current[:,-1])))
 if not equal:
  t=execute(q,True);out['current_teacher_plain_exact']=bool(torch.equal(a[0],t[0]) and torch.equal(a[1],t[1]) and torch.equal(a[3],t[3]));out['stored_teacher_replay_exact']=bool(np.array_equal(stored,t[0].numpy()))
  legacy=dict(q);legacy['seed']=q['seed']-18000000
  old=execute(legacy,False);out['old_policy_seed_exact']=bool(np.array_equal(stored,old[0].numpy()))
 return out
if __name__=='__main__':
 rows=[]
 with ProcessPoolExecutor(8,mp_context=multiprocessing.get_context('spawn')) as ex:
  fs=[ex.submit(worker,i) for i in range(144)]
  for j,f in enumerate(as_completed(fs)):
   rows.append(f.result())
   if (j+1)%24==0:print(j+1,'/144',flush=True)
 rows.sort(key=lambda z:z['index']);(DEST/'dataset_replay.json').write_text(json.dumps(dict(script=sha(__file__),tasks_sha256=sha(OUT/'tasks.pt'),cases=rows),indent=2))
 print(json.dumps(dict(exact=sum(r['exact'] for r in rows),total=len(rows),mismatches=[r for r in rows if not r['exact']]),indent=2))
