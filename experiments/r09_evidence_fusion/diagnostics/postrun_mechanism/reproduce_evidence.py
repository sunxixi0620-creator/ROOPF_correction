import collections,gzip,json,numpy as np
from pathlib import Path
P=Path('/home/skq/repos/roopf_agent/ROOPF/experiments/r09_evidence_fusion')
m=json.loads((P/'offline_memory.json').read_text());cases=m['cases'];names=m['allowed_configs'];byid={r['id']:r for r in cases}
f=np.array([r['features'] for r in cases]);sc=np.maximum(f.std(0),.05);d=((f[:,None,:]-f[None,:,:])/sc)**2;d=d.mean(-1)
same=np.array([[a['function']==b['function'] for b in cases] for a in cases]);d[same]=np.inf;lofo=d.min(1);q95=np.quantile(lofo,.95)
lookup={};api=collections.Counter();contracts=collections.Counter();invalid_pairs=collections.Counter()
for path in (P/'api_journal').glob('*.json'):
 r=json.loads(path.read_text());payload=json.loads(r['request']['messages'][-1]['content'].split('INPUT:\n',1)[1]);lookup[path.stem]=(r,payload)
 if r.get('error')=='invalid_factor_pair':
  try:
   rows=json.loads(r['response']['choices'][0]['message']['content'])['actions']
   for x in rows:
    if isinstance(x,list) and len(x)==3 and (type(x[1])!=int or type(x[2])!=int or not(0<=x[1]<=7 and 0<=x[2]<=2) or (x[1]==1 and x[2]!=1)):invalid_pairs[str(x[1:])]+=1
  except Exception:pass
c=collections.Counter();chosen_stats=collections.defaultdict(collections.Counter);distances=[];testfeatures=[];examples=[];anchor=collections.Counter();epoch=collections.Counter();visible=collections.Counter();offlinefusion=collections.Counter()
with gzip.open(P/'publication/API_EXECUTED_DECISIONS.jsonl.gz','rt') as stream:
 for i,line in enumerate(stream,1):
  r=json.loads(line);z=r['decision']
  for current,action in zip(z['current'],z['configs']):
   if current['extra_origin']=='anchor_variant':
    anchor['sampled_variant_rounds']+=1;anchor['variant_selected']+=int(current['selected_index']==37);anchor['variant_raw_acquisition_argmin']+=int(np.argmin(current['acquisition'])==37)
  if not z['request_event']:continue
  record,payload=lookup[z['api_request_sha256']];status=record['status'];api[status]+=1
  for idx,(item,action,compact) in enumerate(zip(z['api_trajectories'],z['configs'],payload['trajectories'])):
   c['all_trajectory_requests']+=1;epoch[str(z['step'])]+=1
   retrieved=item['offline_cases'];near=[a['distance'] for a in retrieved];distances.append(near[0]);testfeatures.append(item['current']['features']);c['nearest_beyond_train_lofo_p95']+=int(near[0]>q95)
   c['cross_epoch_nearest']+=int(not retrieved[0]['id'].endswith(':d'+str(z['step'])))
   weights=np.array([1/(1+x) for x in near]);weights/=weights.sum()
   A=np.array([[x['advantages5'][n] for n in names] for x in retrieved]);B=np.array([[x['advantages10'][n] for n in names] for x in retrieved]);v=weights@A;v10=weights@B;rounded=np.round(v,5)
   c['all_22_rounded_estimates_tie']+=int(np.ptp(rounded)==0)
   if status!='ok':continue
   c['valid_trajectory_choices']+=1;g=action.split('@')[0];chosen=names.index(action);s=chosen_stats[g]
   s['n']+=1;s['offline_gain5_positive']+=int(v[chosen]>0);s['offline_gain5_zero']+=int(v[chosen]==0);s['offline_gain5_negative']+=int(v[chosen]<0)
   s['shown']+=int(chosen in [a[0] for a in compact['offline']]);s['strictly_worse_than_best_offline5']+=int(v[chosen]<v.max());s['all_estimates_rounded_tie']+=int(np.ptp(rounded)==0)
   shown=[a[0] for a in compact['offline']];visible['chosen_shown']+=int(chosen in shown);visible['chosen_not_shown']+=int(chosen not in shown)
   if g!='release':
    triplet=[names.index(n) for n in [g,g+'@online',g+'@residual_boost']]
    offlinefusion['valid_nonrelease_choices']+=1;offlinefusion['same_generation_lambda_estimates_rounded_tie']+=int(np.ptp(rounded[triplet])==0)
    if '@' not in action:
     offlinefusion['default_lambda_choices']+=1;offlinefusion['default_lambda_when_other_lambda_better_offline5']+=int(max(v[triplet[1]],v[triplet[2]])>v[triplet[0]])
     offlinefusion['both_alternative_lambdas_visible']+=int(all(t in shown for t in triplet))
   if action=='release' and len(examples)<4:examples.append(dict(group=r['group'],fid=r['fid'],batch=r['batch'],step=z['step'],trajectory=idx,chosen_gain5=float(v[chosen]),best_gain5=float(v.max()),shown=shown))
  if i%200==0:print('Read records',i,flush=True)
T=np.array(testfeatures);A=np.array([[r['advantages']['5'][n] for n in names] for r in cases]);B=np.array([[r['advantages']['10'][n] for n in names] for r in cases]);dists=np.array(distances)
result=dict(source='read-only R09 posthoc descriptive diagnosis; no new API/objective evaluations',training=dict(cases=len(cases),all22_action_labels_equal5=int((np.ptp(A,axis=1)==0).sum()),all22_action_labels_equal10=int((np.ptp(B,axis=1)==0).sum()),anchor_sharp_soft_identical5=int((A[:,names.index('anchor_sharp')]==A[:,names.index('anchor_soft')]).sum()),anchor_sharp_soft_identical10=int((B[:,names.index('anchor_sharp')]==B[:,names.index('anchor_soft')]).sum()),feature_std=f.std(0).tolist(),lofo_nearest_distance_quantiles=np.quantile(lofo,[0,.5,.9,.95,1]).tolist()),counts=dict(c),api_request_events=dict(api),selected_action_offline_evidence={k:dict(v) for k,v in chosen_stats.items()},visibility=dict(visible),fusion_evidence=dict(offlinefusion),sampled_generation_outcomes=dict(anchor),invalid_factor_rows=dict(invalid_pairs),runtime_nearest_distance_quantiles=np.quantile(dists,[0,.5,.9,.95,1]).tolist(),runtime_feature_min=T.min(0).tolist(),runtime_feature_max=T.max(0).tolist(),training_feature_min=f.min(0).tolist(),training_feature_max=f.max(0).tolist(),runtime_feature_outside_training_range=((T<f.min(0))|(T>f.max(0))).sum(0).tolist(),release_examples=examples)
Path('/tmp/r09_deep_diagnosis.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)
