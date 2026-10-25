"""Read-only full-record and physical module-removal audit of R12."""
import collections
import csv
import gzip
import hashlib
import json
import pickle
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
PARENT=ROOT/'experiments/r10_current_candidate_fusion'


def write_csv(path,rows):
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)


def wtl(a,b):return dict(wins=int((a<b).sum()),ties=int((a==b).sum()),losses=int((a>b).sum()))


def report(run,verify=False):
    from report_r10 import audit_api
    from experiment_r12_strict_ablations import GENERATORS
    p=run.EXP;identity=run.binding();assert json.loads((p/'CODE_BINDING.json').read_text())==identity
    for name,expected in json.loads((p/'FROZEN_PARENT.json').read_text())['sha256'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==expected,name
    api=audit_api(run,write=True)
    rows=[];trajectories=[];totals={arm:collections.Counter() for arm in run.settings()['arms']}
    coordinate_counts={arm:collections.Counter() for arm in run.settings()['arms']}
    for group,fid in run.legacy.TASKS:
        folder=p/'results'/f'{group}_f{fid:02d}';parent=PARENT/'results'/folder.name
        assert json.loads((folder/'COMPLETE.json').read_text())['binding']==identity
        full=np.load(parent/'llm_live_FULL100.npz')['values'].min(1).astype(float)
        for arm in run.settings()['arms']:
            data=np.load(folder/(arm+'_FULL100.npz'))['values'];assert data.shape==(100,300) and np.isfinite(data).all()
            final=data.min(1).astype(float)
            row=dict(group=group,fid=fid,arm=arm,primary=not(group=='BBOB' and fid==20),full_mean=float(full.mean()),full_std=float(full.std(ddof=0)),alternative_mean=float(final.mean()),alternative_std=float(final.std(ddof=0)),**wtl(full,final))
            for ref,label in [('off','original'),('matched38','matched38')]:
                reference=np.load(parent/(ref+'_FULL100.npz'))['values'].min(1).astype(float)
                row[label+'_mean']=float(reference.mean());row[label+'_std']=float(reference.std(ddof=0))
                row.update({'alternative_vs_'+label+'_'+k:v for k,v in wtl(final,reference).items()})
            rows.append(row)
            for i,(a,b) in enumerate(zip(full,final)):trajectories.append(dict(group=group,fid=fid,arm=arm,trajectory=i,primary=row['primary'],full=float(a),alternative=float(b),result='win' if a<b else 'loss' if a>b else 'tie'))
            for batch in range(10):
                receipt=json.loads((folder/f'b{batch:02d}/{arm}.json').read_text())
                frozen=json.loads((parent/f'b{batch:02d}/llm_live.json').read_text())
                assert receipt['binding']==identity and receipt['native_fe']+receipt['cached_fe']==3000
                assert receipt['final_rng_matches_frozen_original'] and receipt['objective_rng_boundaries_matched']==201
                for key in ('paired_initial_sha256','paired_solver_rng_sha256'):assert receipt[key]==frozen[key]
                for key in ('native_fe','cached_fe'):totals[arm][key]+=receipt[key]
                with gzip.open(folder/f'b{batch:02d}/{arm}_DECISIONS.json.gz','rt') as stream:records=json.load(stream)
                points=np.load(folder/f'b{batch:02d}/{arm}.npz')['executed_points']
                with gzip.open(folder/f'b{batch:02d}/OBJECTIVE_RNG_SCHEDULE.pkl.gz','rb') as stream:schedule=pickle.load(stream)
                assert len(schedule)==201 and points.shape==(10,100,2,10)
                if arm=='no_agent':
                    assert receipt['agent_present'] is False and not records and not receipt['agent_stats']
                    assert receipt['generator_calls']=={name:100 for name in GENERATORS}
                else:
                    assert receipt['agent_present'] is True and not any(receipt['generator_calls'].values())
                    assert len(records)==(0 if group=='CEC-style' else 45)
                    for record in records:
                        for i,current in enumerate(record['current']):
                            assert current['candidate_count'] in (1,2) and all(op==-2 for op in current['candidate_ops'])
                            assert np.array_equal(points[i,55+record['step'],1],np.asarray(current['candidates'][current['selected_index']],dtype=np.float32))
                    for key in ('changed_points','fusion_changed_points','selected_adjusted_candidates'):totals[arm][key]+=receipt['agent_stats'].get(key,0)
                if group!='CEC-style':
                    with gzip.open(parent/f'b{batch:02d}/llm_live_DECISIONS.json.gz','rt') as stream:old=json.load(stream)
                    for record in old:
                        for i,current in enumerate(record['current']):
                            for slot,point in [(0,current['protected_point']),(1,current['candidates'][current['selected_index']])]:
                                coordinate_counts[arm]['compared']+=1
                                coordinate_counts[arm]['different']+=int(not np.array_equal(points[i,55+record['step'],slot],np.asarray(point,dtype=np.float32)))
    summary=[]
    for arm in run.settings()['arms']:
        assert totals[arm]['native_fe']+totals[arm]['cached_fe']==1050000
        for scope in ('primary34','BBOB23','CEC6','UAV5','BBOB20_secondary'):
            selected=[r for r in rows if r['arm']==arm and (r['primary'] if scope=='primary34' else r['group']=='BBOB' and r['fid']!=20 if scope=='BBOB23' else r['group']=='CEC-style' if scope=='CEC6' else r['group']=='UAV' if scope=='UAV5' else r['group']=='BBOB' and r['fid']==20)]
            summary.append(dict(arm=arm,scope=scope,function_wins=sum(r['full_mean']<r['alternative_mean'] for r in selected),function_ties=sum(r['full_mean']==r['alternative_mean'] for r in selected),function_losses=sum(r['full_mean']>r['alternative_mean'] for r in selected),trajectory_wins=sum(r['wins'] for r in selected),trajectory_ties=sum(r['ties'] for r in selected),trajectory_losses=sum(r['losses'] for r in selected)))
    write_csv(p/'FUNCTION_COMPARISON.csv',rows);write_csv(p/'TRAJECTORY_COMPARISON.csv',trajectories)
    write_csv(p/'SUMMARY.csv',summary);write_csv(p/'BBOB20_SEPARATE.csv',[r for r in rows if not r['primary']])
    result=dict(status='full_strict_component_ablations_verified',functions=35,arms=run.settings()['arms'],trajectories=7000,
        comparisons=summary,api=api,totals={k:dict(v) for k,v in totals.items()},active_window_coordinate_comparison={k:dict(v) for k,v in coordinate_counts.items()},
        no_agent=dict(controller_present=False,llm_calls=0,retrieval_calls=0,agent_decision_records=0,online_generators_retained=True),
        no_online_generators=dict(generator_calls=0,native_extra_calls=0,competing_candidates='1 original Anchor, optionally 1 actual Anchor variant; no padding',agent_retained=True,online_proxy_retained=True),
        paired_initial_and_final_rng_batches=700,paired_objective_rng_boundaries=140700,
        parent_full='frozen completed real-LLM R10 results',
        limitations=['Full offline cases held fixed; not relabeled for changed generator availability.',
            'Local-step profiles are inactive aliases in no-online-generators arm and explicitly disclosed to LLM.',
            'Removed generator RNG draws are skipped using the recorded reference objective-boundary cursor; no removed generator is run.',
            'CEC structural lock retained; no Agent calls in CEC.', 'BBOB20 secondary only; function counts are not significance tests.',
            'No-Agent is the original ROOPF core on the same CUDA proxy backend as Full; CPU frozen original is separately reported.'])
    run.legacy.write_json(p/('INDEPENDENT_VERIFICATION.json' if verify else 'FINAL_RESULT.json'),result)
    return result
