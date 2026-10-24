import collections,gzip,json,math,statistics
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
P=Path('/home/skq/repos/roopf_agent/ROOPF/experiments/r09_evidence_fusion')
paths=[]
for path in (P/'results').glob('*/b??/llm_live.json'):
 s=json.loads(path.read_text())['agent_stats']
 if s.get('anchor_variant_rounds') or s.get('local_scaled_rounds'):paths.append(path.with_name('llm_live_DECISIONS.json.gz'))
def read(path):
 c=collections.Counter();ent=[];dist=[]
 with gzip.open(path,'rt') as stream:rows=json.load(stream)
 for row in rows:
  for current,action in zip(row['current'],row['configs']):
   g=action.split('@')[0];slot=current['selected_index']
   if current['extra_origin']=='anchor_variant':
    c['anchor_variant_rounds']+=1;c['anchor_variant_selected']+=int(slot==37);c['anchor_variant_raw_argmin']+=int(min(range(38),key=lambda i:current['acquisition'][i])==37)
    ent.append(current['anchor']['entropy']);dist.append(current['anchor']['normalized_distance'])
   if g in ('local_fine','local_wide','joint_sharp_wide','joint_soft_fine'):
    c['local_scaled_rounds']+=1;c['local_adjusted_selected']+=int(slot in (22,23,24,28,29,30))
 return c,ent,dist
count=collections.Counter();entropy=[];distance=[]
with ThreadPoolExecutor(max_workers=4) as pool:
 futures=[pool.submit(read,p) for p in paths]
 for i,f in enumerate(as_completed(futures),1):
  c,e,d=f.result();count.update(c);entropy+=e;distance+=d
  if i%25==0:print('Read adjusted batches',i,'/',len(paths),flush=True)
train=collections.Counter()
for path in (P/'collection').glob('*/CASES.json.gz'):
 with gzip.open(path,'rt') as stream:cases=json.load(stream)
 for case in cases:
  if case['function'] in json.loads((P/'offline_memory.json').read_text())['heldout_functions']:continue
  a=case['candidate_diagnostics']['anchor_sharp'];b=case['candidate_diagnostics']['anchor_soft']
  train['paired_training_cases']+=1;train['sharp_variant_selected']+=int(a['selected_index']==37);train['soft_variant_selected']+=int(b['selected_index']==37)
  train['sharp_soft_extra_objective_distinct']+=int(a['values'][37]!=b['values'][37])
  train['sharp_soft_selected_index_distinct']+=int(a['selected_index']!=b['selected_index'])
result=dict(source='read-only actuator diagnosis on all relevant llm_live rounds; no new objective/API calls',adjusted_batches_read=len(paths),counts=dict(count),anchor_entropy=dict(min=min(entropy),median=statistics.median(entropy),max=max(entropy),uniform100_entropy=math.log(100)),variant_distance_from_original_competing_anchor=dict(min=min(distance),median=statistics.median(distance),max=max(distance)),paired_training_diagnostics=dict(train))
Path('/tmp/r09_actuator_diagnosis.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)
