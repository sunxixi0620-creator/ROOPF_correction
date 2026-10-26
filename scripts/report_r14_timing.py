"""Summarize complete or partial paired R14 measurements; independently audit actual API execution."""
from pathlib import Path
import csv,gzip,json,hashlib,collections,sys
from decimal import Decimal
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
ARMS=['fe_210','fe_100','fe_130','fe_150','fe_180','fe_230','fe_260']


def read(p):return json.loads(Path(p).read_text())
def csvwrite(p,rows):
    if not rows:return
    with Path(p).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def report(exp,verify=False):
    from roopf.policy_api_r10 import validate_response,wire_bytes
    from roopf.current_evidence_agent import request_context_digest,digest
    exp=Path(exp);summaries=[];function_rows=[];trajectory_rows=[];api_counts={a:collections.Counter() for a in ARMS};consumed=set();credit=Decimal(0);credit_rows=0
    journals={}
    for p in (exp/'api_journal').glob('*.json'):
        r=read(p);journals[p.stem]=r
        assert hashlib.sha256(wire_bytes(r['request'])).hexdigest()==p.stem
        payload=json.loads(r['request']['messages'][-1]['content'].split('INPUT:\n',1)[1])
        assert payload['execution_context_sha256']==digest(r['execution_context'])
        if r['status'] in ('ok','partial'):
            decoded=validate_response(r['response'],payload)
            assert all(decoded[k]==r[k] for k in decoded)
        arm=r['execution_context']['arm'];c=api_counts[arm];c['attempts']+=1;c[r['status']]+=1
        c['valid_rows']+=len(r.get('valid_trajectory_ids',[]));c['fallback_rows']+=len(r.get('row_errors',{}))
        u=r.get('response',{}).get('usage',{});c['prompt_tokens']+=u.get('prompt_tokens',0);c['completion_tokens']+=u.get('completion_tokens',0)
        if isinstance(u.get('credit'),(int,float)):credit+=Decimal(str(u['credit']));credit_rows+=1
    valid_coverage={};receipts=0
    for group,count in [('BBOB',24),('CEC-style',6),('UAV',5)]:
        for fid in range(1,count+1):
            per={a:[] for a in ARMS};refs={'frozen_final':[],'frozen_original':[]};coordinates={a:collections.Counter() for a in ARMS}
            for batch in range(10):
                folder=exp/'results'/f'{group}_f{fid:02d}'/f'b{batch:02d}'
                if not (folder/'REFERENCES.npz').exists():continue
                with np.load(folder/'REFERENCES.npz') as data:
                    frozen=data['frozen_final_values'].min(1).astype(float);original=data['original_values'].min(1).astype(float)
                    refs['frozen_final'].append((batch,frozen));refs['frozen_original'].append((batch,original))
                for arm in ARMS:
                    receipt=folder/(arm+'.json')
                    if not receipt.exists():continue
                    receipts+=1;r=read(receipt)
                    with np.load(folder/(arm+'.npz')) as data:values=data['values'].min(1).astype(float);points=data['executed_points'].copy()
                    per[arm].append((batch,values));coordinates[arm].update({k:r[k] for k in ['coordinate_changes_vs_frozen_final','coordinate_changes_vs_original','native_fe','cached_fe','actual_online_evaluated_points']})
                    if verify:
                        with gzip.open(folder/(arm+'_DECISIONS.json.gz'),'rt') as f:decisions=json.load(f)
                        assert len(decisions)==45
                        requests=[d for d in decisions if d['request_event']];assert [d['step'] for d in requests]==[5,15]
                        for d in decisions:
                            for i,c in enumerate(d['current']):
                                assert np.array_equal(points[i,55+d['step'],0],c['protected_point'])
                                assert np.array_equal(points[i,55+d['step'],1],c['candidates'][c['selected_index']])
                        for d in requests:
                            identity=d['api_request_sha256'];assert identity not in consumed
                            e=journals[identity];assert e['execution_context']==d['execution_context'] and e['config_ids']==d['configs']
                            assert e['decision_context_sha256']==request_context_digest(d['api_trajectories'],d['execution_context'])
                            assert d['api_row_validation']['valid_trajectory_ids']==e['valid_trajectory_ids']
                            for cont in decisions[d['step']+1:d['step']+5]:
                                assert cont['configs']==e['config_ids'] and cont['applied_generation']==[n.split('@')[0] for n in e['config_ids']]
                            consumed.add(identity);key=(group,fid,arm);valid_coverage[key]=valid_coverage.get(key,0)+len(e['valid_trajectory_ids'])
            complete={a:np.concatenate([v for b,v in sorted(rows)]) for a,rows in per.items() if len(rows)==10}
            complete_refs={a:np.concatenate([v for b,v in sorted(rows)]) for a,rows in refs.items() if len(rows)==10}
            for arm,values in complete.items():
                for reference,base in [*complete_refs.items(),*([('contemporary_210',complete['fe_210'])] if 'fe_210' in complete else [])]:
                    row=dict(group=group,fid=fid,arm=arm,threshold=int(arm[3:]),reference=reference,n=100,
                        mean=float(values.mean()),population_std=float(values.std(ddof=0)),variance=float(values.var(ddof=0)),reference_mean=float(base.mean()),
                        mean_difference=float(values.mean()-base.mean()),wins=int((values<base).sum()),ties=int((values==base).sum()),losses=int((values>base).sum()),
                        primary=not(group=='BBOB' and fid==20),**dict(coordinates[arm]))
                    function_rows.append(row)
                    for i,(x,y) in enumerate(zip(values,base)):trajectory_rows.append(dict(group=group,fid=fid,arm=arm,reference=reference,trajectory=i,value=float(x),reference_value=float(y),difference=float(x-y)))
    for arm in ARMS:
        for group in ['BBOB','BBOB20_separate','CEC-style','UAV','primary_34_functions']:
            for reference in ['frozen_final','contemporary_210','frozen_original']:
                rows=[r for r in function_rows if r['arm']==arm and r['reference']==reference and
                    ((group=='primary_34_functions' and r['primary']) or (group=='BBOB20_separate' and r['group']=='BBOB' and r['fid']==20) or
                     (group==r['group'] and not(group=='BBOB' and r['fid']==20)))]
                if rows:summaries.append(dict(arm=arm,threshold=int(arm[3:]),group=group,reference=reference,functions=len(rows),
                    function_wins=sum(r['mean_difference']<0 for r in rows),function_ties=sum(r['mean_difference']==0 for r in rows),function_losses=sum(r['mean_difference']>0 for r in rows),
                    trajectory_wins=sum(r['wins'] for r in rows),trajectory_ties=sum(r['ties'] for r in rows),trajectory_losses=sum(r['losses'] for r in rows)))
    csvwrite(exp/'FUNCTION_STATISTICS_AND_COMPARISONS.csv',function_rows);csvwrite(exp/'PAIRED_TRAJECTORIES.csv',trajectory_rows);csvwrite(exp/'SUMMARY.csv',summaries)
    rankings=[]
    groups=['BBOB','CEC-style','UAV','primary_34_functions','BBOB20_separate']
    for group in groups:
        rank_sums=collections.Counter();rank_counts=collections.Counter();best_counts=collections.Counter()
        keys={(r['group'],r['fid']) for r in function_rows if r['reference']=='frozen_final' and
            ((group=='primary_34_functions' and r['primary']) or (group=='BBOB20_separate' and r['group']=='BBOB' and r['fid']==20) or
             (group==r['group'] and not(group=='BBOB' and r['fid']==20)))}
        for g,f in keys:
            rows=[r for r in function_rows if r['reference']=='frozen_final' and r['group']==g and r['fid']==f]
            if len(rows)!=7:continue
            for r in rows:
                less=sum(v['mean']<r['mean'] for v in rows);equal=sum(v['mean']==r['mean'] for v in rows)
                rank_sums[r['arm']]+=less+(equal+1)/2;rank_counts[r['arm']]+=1
                best_counts[r['arm']]+=int(less==0)
        for arm in ARMS:
            if rank_counts[arm]:rankings.append(dict(group=group,arm=arm,threshold=int(arm[3:]),functions=rank_counts[arm],
                average_rank=rank_sums[arm]/rank_counts[arm],best_or_tied_best_functions=best_counts[arm]))
    csvwrite(exp/'PARAMETER_RANKINGS.csv',rankings)
    execution_complete=receipts==2450
    if verify and execution_complete:assert consumed==set(journals) and len(journals)==4900
    valid_all=verify and all(valid_coverage.get((g,f,a),0)>0 for g,n in [('BBOB',24),('CEC-style',6),('UAV',5)] for f in range(1,n+1) for a in ARMS)
    if verify:
        for p,h in read(exp/'FROZEN_PARENT.json')['sha256'].items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h,p
    status='complete_live_timing_measurement_verified' if execution_complete and valid_all else ('execution_complete_api_validation_incomplete' if execution_complete else 'partial')
    result=dict(status=status,completed_arm_batches=receipts,expected_arm_batches=2450,matched_actual_api_records=len(consumed),expected_actual_requests=4900,
        actual_api={k:dict(v) for k,v in api_counts.items()},provider_credit_units=str(credit),responses_with_credit=credit_rows,currency_inferred=False,
        summaries=summaries,parameter_rankings=rankings,not_significance_test=True,bbob20_separate=True)
    write=exp/('FINAL_RESULT.json' if execution_complete else 'PARTIAL_RESULT.json');write.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    return result

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--verify',action='store_true');a=p.parse_args()
    print(json.dumps(report(ROOT/'experiments/r14_online_entry_timing',a.verify)))
