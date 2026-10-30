"""Independent postrun audit of exact paired coordinates, means, API and removal counters."""
import csv,json,gzip,hashlib,multiprocessing as mp
from pathlib import Path
import numpy as np
from roopf.policy_api_r10 import validate_response,wire_bytes
from roopf.current_evidence_agent import request_context_digest
from experiment_r19_ablation import E,ROOT,ARMS,binding

def inspect(task):
 g,f,b=task;folder=E/'results'/f'{g}_f{f:02d}'/f'b{b:02d}'
 r={a:json.loads((folder/(a+'.json')).read_text()) for a in ARMS};data={};api=[]
 with np.load(folder/'REFERENCES.npz') as z:base=z['original_points'];archive=z['archived_full_points']
 for a in ARMS:
  q=r[a];assert q['objective_rng_boundaries_matched']==201 and q['native_fe']+q['cached_fe']==3000
  with np.load(folder/(a+'.npz')) as z:
   values=z['values'];pts=z['executed_points'];ref=z['reference_points'];assert values.shape==(10,300) and pts.shape==(10,100,2,10)
   assert np.array_equal(pts[:,:,0],ref[:,:,0]);assert q['coordinate_changes_vs_original']==int((pts!=base).any(-1).sum());assert q['coordinate_changes_vs_archived_full']==int((pts!=archive).any(-1).sum())
   data[a]=(values.min(1).astype(float).tolist(),pts.copy())
  with gzip.open(folder/(a+'_DECISIONS.json.gz'),'rt') as f:dec=json.load(f)
  if a=='no_agent':assert not dec and not q['agent_stats']
  else:
   assert len(dec)==45
   for d in dec:
    if not d['request_event']:continue
    identity=d['api_request_sha256'];record=json.loads((E/'api_journal'/(identity+'.json')).read_text());assert hashlib.sha256(wire_bytes(record['request'])).hexdigest()==identity
    assert record['execution_context']==d['execution_context'] and record['config_ids']==d['configs'] and record['decision_context_sha256']==request_context_digest(d['api_trajectories'],d['execution_context'])
    payload=json.loads(record['request']['messages'][-1]['content'].split('INPUT:\n',1)[1])
    if record['status'] in ('ok','partial'):
     decoded=validate_response(record['response'],payload);assert all(decoded[k]==record[k] for k in decoded)
    api.append((identity,a,len(record.get('valid_trajectory_ids',[]))))
  if a=='llm_no_anchor':assert q['route_counts']['anchor_forward']==0
  if a=='llm_no_online_generators':assert not any(q['route_counts']['generators'].values())
 return {'g':g,'f':f,'b':b,'values':{a:data[a][0] for a in ARMS},'changed':{a:int((data[a][1]!=data['llm_full'][1]).any(-1).sum()) for a in ARMS},'api':api}

def main():
 tasks=[(g,f,b) for g,n in [('BBOB',24),('CEC-style',6),('UAV',5)] for f in range(1,n+1) for b in range(10)]
 with mp.get_context('spawn').Pool(24) as pool:rows=list(pool.imap_unordered(inspect,tasks,chunksize=1))
 assert len(rows)==350
 ids=set();coverage={};measure={}
 for x in rows:
  for identity,a,n in x['api']:
   assert identity not in ids;ids.add(identity);k=(x['g'],x['f'],a);coverage[k]=coverage.get(k,0)+n
  for a in ARMS:measure.setdefault((x['g'],x['f'],a),[]).append((x['b'],x['values'][a]))
 assert len(ids)==2100 and len(coverage)==105 and all(v>0 for v in coverage.values())
 stat=list(csv.DictReader((E/'FUNCTION_STATISTICS.csv').open(encoding='utf-8-sig')));assert len(stat)==140
 for r in stat:
  rows=sorted(measure[(r['group'],int(r['fid']),r['arm'])]);x=np.array([v for _,values in rows for v in values]);assert len(x)==100
  assert np.isclose(x.mean(),float(r['mean']),rtol=1e-13,atol=0) and np.isclose(x.std(ddof=0),float(r['population_std']),rtol=1e-13,atol=0)
 result=dict(status='complete_independently_verified',functions=35,arms=4,trajectories=14000,api_records=2100,valid_executed_api_coverage=105,per_function_means_and_population_std_recomputed=True,actual_points_checked=True,original_and_archived_point_differences_recomputed=True,binding=binding())
 (E/'INDEPENDENT_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
