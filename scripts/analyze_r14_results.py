from pathlib import Path
import json,csv,hashlib,collections,datetime,gzip
import numpy as np
R=Path(__file__).resolve().parents[1];E=R/'experiments/r14_online_entry_timing'
final=json.loads((E/'FINAL_RESULT.json').read_text());assert final['status']=='complete_live_timing_measurement_verified'
rows=list(csv.DictReader((E/'FUNCTION_STATISTICS_AND_COMPARISONS.csv').open(encoding='utf-8-sig')))
look={(r['group'],int(r['fid']),r['arm']):r for r in rows if r['reference']=='frozen_final'}
receipts=[];computed={}
for folder in sorted(E.glob('results/*/b??')):
 for arm in ['fe_210','fe_100','fe_130','fe_150','fe_180','fe_230','fe_260']:
  receipt=json.loads((folder/(arm+'.json')).read_text());receipts.append(receipt)
  with np.load(folder/(arm+'.npz')) as d:
   v=d['values'];assert v.shape==(10,300) and np.isfinite(v).all()
   computed.setdefault((receipt['group'],receipt['fid'],arm),[]).append((receipt['batch'],v.min(1).astype(float)))
assert len(receipts)==2450 and len(computed)==245
for k,values in computed.items():
 arr=np.concatenate([v for b,v in sorted(values)]);assert arr.shape==(100,)
 row=look[k]
 assert float(row['mean'])==float(arr.mean())
 assert float(row['population_std'])==float(arr.std(ddof=0))
 assert float(row['variance'])==float(arr.var(ddof=0))
rank_sums=collections.Counter()
for group,count in [('BBOB',24),('CEC-style',6),('UAV',5)]:
 for fid in range(1,count+1):
  means={a:float(look[group,fid,a]['mean']) for a in final['actual_api']}
  for a,x in means.items():
   rank=sum(v<x for v in means.values())+(sum(v==x for v in means.values())+1)/2
   if (group,fid)!=('BBOB',20):rank_sums[a]+=rank
for r in final['parameter_rankings']:
 if r['group']=='primary_34_functions':assert r['average_rank']==rank_sums[r['arm']]/34
api=[];coverage=collections.defaultdict(collections.Counter);errors=collections.Counter()
for p in sorted((E/'api_journal').glob('*.json')):
 r=json.loads(p.read_text());c=r['execution_context'];key=(c['group'],c['fid'],c['arm'])
 cnt=coverage[key];cnt['attempts']+=1;cnt[r['status']]+=1;cnt['valid_rows']+=len(r['valid_trajectory_ids']);cnt['fallback_rows']+=len(r['row_errors'])
 usage=r.get('response',{}).get('usage',{})
 api.append(dict(request_sha256=p.stem,group=c['group'],fid=c['fid'],batch=c['batch'],arm=c['arm'],status=r['status'],
   valid_rows=len(r['valid_trajectory_ids']),fallback_rows=len(r['row_errors']),prompt_tokens=usage.get('prompt_tokens',0),
   completion_tokens=usage.get('completion_tokens',0),provider_credit=usage.get('credit'),journal_sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
 if r['status']=='failed':errors[r.get('error') or 'all_action_rows_invalid']+=1
assert len(api)==4900 and len(coverage)==245
assert all(c['attempts']==20 and c['valid_rows']>0 for c in coverage.values())
def writecsv(p,rows):
 with p.open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
writecsv(E/'API_REQUEST_AUDIT.csv',api)
writecsv(E/'API_FUNCTION_COVERAGE.csv',[dict(group=g,fid=f,arm=a,**dict(v)) for (g,f,a),v in sorted(coverage.items())])
metrics=['native_fe','cached_fe','coordinate_changes_vs_frozen_final','coordinate_changes_vs_original','actual_online_evaluated_points']
totals={a:{k:sum(r[k] for r in receipts if r['arm']==a) for k in metrics} for a in final['actual_api']}
assert sum(v['native_fe']+v['cached_fe'] for v in totals.values())==7350000
uav=[]
for fid in range(1,6):
 r=look['UAV',fid,'fe_100'];uav.append({'fid':fid,'final_mean':float(r['reference_mean']),'fe100_mean':float(r['mean']),'relative_mean_reduction_percent':100*(float(r['reference_mean'])-float(r['mean']))/float(r['reference_mean'])})
for p,h in json.loads((E/'RUNTIME_ENTRY_BINDING.json').read_text())['sha256'].items():assert hashlib.sha256((R/p).read_bytes()).hexdigest()==h,p
record={'status':'all245_raw_function_distributions_and_primary_ranks_independently_recomputed','trajectory_count':24500,
 'condition_batches':2450,'functions_per_condition':35,'actual_api_requests':4900,'minimum_valid_action_rows_per_function_and_condition':min(c['valid_rows'] for c in coverage.values()),
 'frozen_parent_files':len(json.loads((E/'FROZEN_PARENT.json').read_text())['sha256']),'source_bindings_unchanged':True,
 'execution_metrics':totals,'api_totals':{k:sum(c.get(k,0) for c in final['actual_api'].values()) for k in ['attempts','ok','partial','failed','valid_rows','fallback_rows','prompt_tokens','completion_tokens']},
 'provider_credit_units':final['provider_credit_units'],'responses_with_credit':final['responses_with_credit'],'currency_inferred':False,
 'api_failure_causes':dict(errors),'uav_fe100_improvement':uav,'recommended_uniform_threshold':210,
 'best_exploratory_by_class':{'BBOB_primary23':210,'CEC-style':150,'UAV':100,'BBOB20_separate':130},
 'limitations':['Descriptive average ranks and paired W/T/L, not significance tests','LLM calls fixed atFE220/240; control windows unchanged','36.65% fully failed API responses; fallback and live-return mix preserved','Threshold selection on existing suite is exploratory; no final code changed']}
(E/'INDEPENDENT_RESULT_ANALYSIS.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(record,ensure_ascii=False))
