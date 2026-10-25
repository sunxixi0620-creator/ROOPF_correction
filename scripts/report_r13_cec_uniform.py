"""Read-only numerical/API/physical-removal verification of completed R13 runs."""
from __future__ import annotations
import collections
import csv
from decimal import Decimal
import gzip
import hashlib
import json
from pathlib import Path
import tarfile
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
ARMS=('frozen_baseline','no_agent','llm_full','llm_no_anchor','llm_no_online_generators')
LIVE=ARMS[2:]
LABELS={'frozen_baseline':'冻结基线','no_agent':'移除Agent','llm_full':'Full',
        'llm_no_anchor':'移除Anchor','llm_no_online_generators':'移除在线生成算子'}


def readj(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def csv_write(p,rows):
    with Path(p).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def wtl(a,b):return dict(wins=int((a<b).sum()),ties=int((a==b).sum()),losses=int((a>b).sum()))


def api_audit(directory):
    from roopf.policy_api_r10 import validate_response,wire_bytes
    from roopf.current_evidence_agent import request_context_digest,digest
    journals={};usage={a:collections.Counter() for a in LIVE};credits={a:Decimal(0) for a in LIVE}
    credit_n=collections.Counter();request_rows=[];consumed=set();coverage=[]
    memory_digest=digest(readj(directory/'offline_memory.json'))
    for p in sorted((directory/'api_journal').glob('*.json')):
        e=readj(p);assert e['status']!='pending','Unresolved API outcome'
        assert hashlib.sha256(wire_bytes(e['request'])).hexdigest()==p.stem
        payload=json.loads(e['request']['messages'][-1]['content'].split('INPUT:\n',1)[1])
        assert payload['memory_sha256']==memory_digest
        assert payload['execution_context_sha256']==digest(e['execution_context'])
        assert payload['decision_context_sha256']==e['decision_context_sha256']
        assert e['execution_context']['group']=='CEC-style'
        arm=e['execution_context']['arm'];assert arm in LIVE
        if e['status'] in ('ok','partial'):
            decoded=validate_response(e['response'],payload)
            assert all(decoded[k]==e[k] for k in decoded)
        else:assert e['config_ids']==['matched38']*10
        assert e['decision_context_sha256'] not in journals
        journals[e['decision_context_sha256']]=(p.stem,e,payload)
        count=usage[arm];count['attempts']+=1;count[e['status']]+=1
        count['valid_rows']+=len(e['valid_trajectory_ids']);count['fallback_rows']+=len(e['row_errors'])
        u=e.get('response',{}).get('usage',{}) or {}
        count['prompt_tokens']+=u.get('prompt_tokens',0) or 0
        count['completion_tokens']+=u.get('completion_tokens',0) or 0
        credit=u.get('credit')
        if type(credit) in (int,float):credits[arm]+=Decimal(str(credit));credit_n[arm]+=1
        request_rows.append(dict(request_id=p.stem,fid=e['execution_context']['fid'],batch=e['execution_context']['batch'],
            arm=arm,status=e['status'],valid_rows=len(e['valid_trajectory_ids']),fallback_rows=len(e['row_errors']),
            prompt_tokens=u.get('prompt_tokens',0) or 0,completion_tokens=u.get('completion_tokens',0) or 0,
            provider_credit=credit,elapsed_seconds=e['elapsed_seconds'],queue_seconds=e.get('queue_seconds'),
            error_type=e.get('error_type'),error=e.get('error')))
    for fid in range(1,7):
        for arm in LIVE:
            c=collections.Counter();valid_trajectories=set()
            for batch in range(10):
                with gzip.open(directory/'results'/f'CEC-style_f{fid:02d}'/f'b{batch:02d}'/(arm+'_DECISIONS.json.gz'),'rt') as f:records=json.load(f)
                events=[r for r in records if r['request_event']];assert len(records)==45 and len(events)==2
                for r in events:
                    key=request_context_digest(r['api_trajectories'],r['execution_context'])
                    identity,e,payload=journals[key]
                    assert identity==r['api_request_sha256'] and r['execution_context']==e['execution_context']
                    assert r['configs']==e['config_ids'] and r['api_row_validation']['valid_trajectory_ids']==e['valid_trajectory_ids']
                    assert r['step'] in (5,15) and r['applied_generation']==['matched38']*10
                    assert payload['step']==r['step']
                    for t in r['api_trajectories']:
                        i=int(t['trajectory_id'])
                        assert r['current'][i]['generation']==t['current']
                        assert r['decision_features'][i]==t['decision_features']
                        assert r['current_options'][i]==t['current_options']
                    for later in records[r['step']+1:r['step']+5]:
                        assert later['configs']==e['config_ids']
                        assert later['applied_generation']==[n.split('@')[0] for n in e['config_ids']]
                    c['attempts']+=1;c[e['status']]+=1;c['valid_rows']+=len(e['valid_trajectory_ids'])
                    c['fallback_rows']+=len(e['row_errors'])
                    c['singleton_allowed']+=sum(len(t['eligible_configs'])==1 for t in r['api_trajectories'])
                    c['nondefault_actions']+=sum(n!='matched38' for n in r['configs'])
                    valid_trajectories.update((batch,int(i)) for i in e['valid_trajectory_ids'])
                    consumed.add(identity)
            assert c['attempts']==20
            coverage.append(dict(fid=fid,arm=arm,**dict(c),trajectories_with_valid_api=len(valid_trajectories),cec_structural_exclusion=False))
    assert len(journals)==len(consumed)==360
    assert all(v['attempts']==120 for v in usage.values())
    per_arm={arm:dict(**dict(v),provider_credit_sum=str(credits[arm]),responses_with_credit=credit_n[arm]) for arm,v in usage.items()}
    totals=collections.Counter()
    for v in usage.values():totals.update(v)
    result=dict(status='all_360_actual_requests_mapped_to_executed_decisions',attempts=360,
        valid_responses=totals['ok']+totals['partial'],fully_valid=totals['ok'],partial=totals['partial'],failures=totals['failed'],
        valid_rows=totals['valid_rows'],fallback_rows=totals['fallback_rows'],prompt_tokens=totals['prompt_tokens'],completion_tokens=totals['completion_tokens'],
        provider_credit_sum=str(sum(credits.values(),Decimal(0))),responses_with_credit=sum(credit_n.values()),currency='not inferred',
        per_arm=per_arm,function_arm_coverage=coverage,
        every_function_arm_has_valid_api=all(r['valid_rows']>0 for r in coverage),generation_delay_and_actual_coordinates_verified=True)
    write(directory/'API_EXECUTION_AUDIT.json',result);csv_write(directory/'API_REQUESTS.csv',request_rows)
    return result


def report(run,verify=False):
    directory=run.EXP;binding=run.binding();assert readj(directory/'CODE_BINDING.json')==binding
    for p,expected in readj(directory/'FROZEN_PARENT.json')['sha256'].items():assert sha(ROOT/p)==expected,p
    parent=ROOT/'experiments/r10_current_candidate_fusion'
    statistics=[];pairs=[];comparisons=[];totals={a:collections.Counter() for a in ARMS}
    routes={a:collections.Counter() for a in ARMS};coordinate={a:collections.Counter() for a in ARMS}
    for fid in range(1,7):
        folder=directory/'results'/f'CEC-style_f{fid:02d}'
        arrays={a:[] for a in ARMS};point_arrays={a:[] for a in ARMS}
        for batch in range(10):
            target=folder/f'b{batch:02d}';assert readj(target/'BATCH_COMPLETE.json')['binding']==binding
            frozen=readj(parent/'results'/folder.name/f'b{batch:02d}/llm_live.json')
            with gzip.open(target/'OBJECTIVE_RNG_SCHEDULE.pkl.gz','rb') as f:
                import pickle
                schedule=pickle.load(f)
            assert len(schedule)==201
            for arm in ARMS:
                receipt=readj(target/(arm+'.json'));assert receipt['binding']==binding and receipt['nfe']==300 and receipt['n']==10
                assert receipt['native_fe']+receipt['cached_fe']==3000
                assert receipt['final_rng_matches_frozen_original'] and receipt['objective_rng_boundaries_matched']==201
                assert receipt['objective_name']==f'cecf{fid}'
                for key in ('paired_initial_sha256','paired_solver_rng_sha256'):assert receipt[key]==frozen[key]
                data=np.load(target/(arm+'.npz'));values=data['values'];points=data['executed_points'];refs=data['reference_points']
                assert values.shape==(10,300) and points.shape==refs.shape==(10,100,2,10) and np.isfinite(values).all()
                assert np.array_equal(points[:,:,0],refs[:,:,0])
                with gzip.open(target/(arm+'_DECISIONS.json.gz'),'rt') as f:records=json.load(f)
                c=receipt['route_counts'];routes[arm].update({k:v for k,v in c.items() if k!='generators'})
                routes[arm].update({'generator:'+k:v for k,v in c['generators'].items()})
                if arm=='frozen_baseline':
                    assert not records and c['residual_forward']==0 and receipt['cec_structural_exclusion']
                    assert np.array_equal(values,np.load(parent/'results'/folder.name/f'b{batch:02d}/off.npz')['values'])
                    assert np.array_equal(points,refs)
                else:
                    assert c['residual_forward']>0 and not receipt['cec_structural_exclusion']
                    if arm=='no_agent':assert not records and not receipt['agent_present']
                    else:
                        assert len(records)==45 and c['prepare_generation']==45
                        for k,r in enumerate(records):
                            for i,current in enumerate(r['current']):
                                assert current['generation']['nfe_before']==210+2*k
                                assert np.array_equal(points[i,55+k,0],np.asarray(current['protected_point'],dtype=np.float32))
                                assert np.array_equal(points[i,55+k,1],np.asarray(current['candidates'][current['selected_index']],dtype=np.float32))
                                assert current['residual_support'] is not None
                                if arm=='llm_no_online_generators':assert current['candidate_count'] in (1,2) and all(op==-2 for op in current['candidate_ops'])
                                else:assert current['candidate_count']==38
                                if arm=='llm_no_anchor':assert current['anchor']['enabled'] is False
                        for key in ('changed_points','fusion_changed_points','selected_adjusted_candidates','unsupported_queries','valid_rows','invalid_rows','api_reused'):
                            totals[arm][key]+=receipt['agent_stats'].get(key,0)
                    if arm=='llm_no_online_generators':assert not any(c['generators'].values())
                    elif arm=='llm_no_anchor':assert c['anchor_forward']==0 and c['anchor_temperature_proposal']==0
                    else:assert all(v>=100 for v in c['generators'].values())
                totals[arm]['native_fe']+=receipt['native_fe'];totals[arm]['cached_fe']+=receipt['cached_fe']
                arrays[arm].append(values);point_arrays[arm].append(points)
        arrays={a:np.concatenate(v) for a,v in arrays.items()};point_arrays={a:np.concatenate(v) for a,v in point_arrays.items()}
        finals={a:v.min(1).astype(float) for a,v in arrays.items()}
        for arm in ARMS:
            np.savez_compressed(folder/(arm+'_FULL100.npz'),values=arrays[arm])
            delta=(point_arrays[arm]!=point_arrays['frozen_baseline']).any(-1)
            delta_full=(point_arrays[arm]!=point_arrays['llm_full']).any(-1)
            changes=dict(coordinates_changed_vs_baseline=int(delta.sum()),protected_changed_vs_baseline=int(delta[:,:,0].sum()),
                         competitor_changed_vs_baseline=int(delta[:,:,1].sum()),coordinates_changed_vs_full=int(delta_full.sum()))
            coordinate[arm].update(changes)
            statistics.append(dict(group='CEC-style',fid=fid,arm=arm,label=LABELS[arm],n=100,
                mean=float(finals[arm].mean()),population_std=float(finals[arm].std(ddof=0)),variance=float(finals[arm].var(ddof=0)),
                **{'vs_baseline_'+k:v for k,v in wtl(finals[arm],finals['frozen_baseline']).items()},**changes))
        for other in ARMS:
            if other=='llm_full':continue
            result=wtl(finals['llm_full'],finals[other])
            comparisons.append(dict(fid=fid,reference=other,label=LABELS[other],full_mean=float(finals['llm_full'].mean()),
                reference_mean=float(finals[other].mean()),**result))
            for i,(a,b) in enumerate(zip(finals['llm_full'],finals[other])):
                pairs.append(dict(fid=fid,reference=other,trajectory=i,full=float(a),alternative=float(b),result='胜' if a<b else '负' if a>b else '平'))
        write(folder/'COMPLETE.json',dict(status='complete',fid=fid,functions=1,arms=list(ARMS),trajectories_per_arm=100,binding=binding))
    assert all(v['native_fe']+v['cached_fe']==180000 for v in totals.values())
    api=api_audit(directory)
    summary=[]
    for other in ARMS:
        if other=='llm_full':continue
        rows=[r for r in comparisons if r['reference']==other]
        summary.append(dict(comparison='Full vs '+LABELS[other],reference=other,
            function_wins=sum(r['full_mean']<r['reference_mean'] for r in rows),
            function_ties=sum(r['full_mean']==r['reference_mean'] for r in rows),
            function_losses=sum(r['full_mean']>r['reference_mean'] for r in rows),
            trajectory_wins=sum(r['wins'] for r in rows),trajectory_ties=sum(r['ties'] for r in rows),trajectory_losses=sum(r['losses'] for r in rows)))
    csv_write(directory/'FUNCTION_STATISTICS.csv',statistics);csv_write(directory/'FUNCTION_COMPARISONS.csv',comparisons)
    csv_write(directory/'PAIRED_TRAJECTORIES.csv',pairs);csv_write(directory/'SUMMARY.csv',summary)
    status='complete_live_cec_uniform_measurement_verified' if api['every_function_arm_has_valid_api'] else 'complete_execution_but_live_validation_incomplete'
    result=dict(status=status,scope='CEC-style6 only; explicit user protocol change',functions=6,arms=list(ARMS),trajectories_per_arm=600,
        total_trajectories=3000,nfe_including_initial=300,comparisons=summary,api=api,
        evaluation_counts={a:dict(v) for a,v in totals.items()},route_counts={a:dict(v) for a,v in routes.items()},
        actual_coordinate_changes={a:dict(v) for a,v in coordinate.items()},paired_arm_batches=300,paired_objective_rng_boundaries=60300,
        baseline_all_values_equal_frozen=True,previous_sources_and_results_unchanged=True,
        analysis_script_sha256=sha(Path(__file__)),code_binding_sha256=sha(directory/'CODE_BINDING.json'),
        limitations=['The local six CEC-style functions are not a complete official CEC suite.',
            'Frozen original baseline is CPU; other conditions share the CUDA proxy. CUDA numerical differences are included in the comparison.',
            'Full offline memory and thresholds held fixed for all strict component removals; no ablation-specific relabeling.',
            'Default-only allowed sets and valid default responses are possible; no forced intervention.',
            'Function and paired trajectory win counts are descriptive, not significance tests.',
            'No amount or currency is inferred from token usage; credit is reported only when supplied.'])
    write(directory/('INDEPENDENT_VERIFICATION.json' if verify else 'FINAL_RESULT.json'),result)
    return result


def publish(directory):
    """Package independently named scientific records without changing old rounds."""
    directory=Path(directory);out=directory/'publication';out.mkdir(exist_ok=True)
    manifest={}
    for fid in range(1,7):
        folder=directory/'results'/f'CEC-style_f{fid:02d}'
        target=out/(folder.name+'_MEASUREMENTS.tar.xz')
        with tarfile.open(target,'w:xz') as tar:
            for p in sorted(folder.rglob('*')):
                if p.is_file():tar.add(p,arcname=str(p.relative_to(directory)))
        assert target.stat().st_size<95*1024**2,'Split archive before publishing to GitHub'
        manifest[target.name]=dict(sha256=sha(target),bytes=target.stat().st_size)
    target=out/'ACTUAL_API_JOURNAL.tar.xz'
    with tarfile.open(target,'w:xz') as tar:
        for p in sorted((directory/'api_journal').glob('*.json')):tar.add(p,arcname='api_journal/'+p.name)
    manifest[target.name]=dict(sha256=sha(target),bytes=target.stat().st_size)
    write(out/'MANIFEST.json',manifest)
    return manifest
