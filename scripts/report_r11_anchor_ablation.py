"""Audit full R11 records and compare with frozen live R10 Full, without new calls."""
import collections
import csv
import gzip
import hashlib
import json
from pathlib import Path
import tarfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / 'experiments/r10_current_candidate_fusion'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def wtl(a, b):
    return dict(wins=int((a < b).sum()), ties=int((a == b).sum()), losses=int((a > b).sum()))


def report(run, verify=False):
    from report_r10 import audit_api
    directory = run.EXP
    assert run.binding() == json.loads((directory/'CODE_BINDING.json').read_text())
    sources = json.loads((directory/'FROZEN_PARENT.json').read_text())
    for name, expected in sources['sha256'].items():
        assert sha(ROOT/name) == expected, name
    audit = audit_api(run, write=True)
    rows, trajectory_rows, comparisons = [], [], []
    totals = collections.Counter()
    for group, fid in run.legacy.TASKS:
        folder=directory/'results'/f'{group}_f{fid:02d}'
        parent=PARENT/'results'/folder.name
        assert json.loads((folder/'COMPLETE.json').read_text())['binding']==run.binding()
        full_data=np.load(parent/'llm_live_FULL100.npz')['values']
        alternative=np.load(folder/'llm_no_anchor_FULL100.npz')['values']
        assert alternative.shape==(100,300) and np.isfinite(alternative).all()
        full=full_data.min(1).astype(float); other=alternative.min(1).astype(float)
        primary=not(group=='BBOB' and fid==20)
        row=dict(group=group,fid=fid,primary=primary,full_mean=float(full.mean()),full_std=float(full.std(ddof=0)),
            no_anchor_mean=float(other.mean()),no_anchor_std=float(other.std(ddof=0)),**wtl(full,other))
        for arm,label in [('off','original'),('matched38','matched38')]:
            ref=np.load(parent/f'{arm}_FULL100.npz')['values'].min(1).astype(float)
            row[label+'_mean']=float(ref.mean());row[label+'_std']=float(ref.std(ddof=0))
            row.update({'no_anchor_vs_'+label+'_'+k:v for k,v in wtl(other,ref).items()})
        rows.append(row)
        for i,(a,b) in enumerate(zip(full,other)):
            trajectory_rows.append(dict(group=group,fid=fid,trajectory=i,primary=primary,full=float(a),no_anchor=float(b),
                result='win' if a<b else 'loss' if a>b else 'tie'))
        totals['new_native_fe']+=sum(json.loads(p.read_text())['native_fe'] for p in folder.glob('b??/llm_no_anchor.json'))
        totals['new_cached_fe']+=sum(json.loads(p.read_text())['cached_fe'] for p in folder.glob('b??/llm_no_anchor.json'))
        for batch in range(10):
            receipt=json.loads((folder/f'b{batch:02d}/llm_no_anchor.json').read_text())
            old=json.loads((parent/f'b{batch:02d}/llm_live.json').read_text())
            assert receipt['binding']==run.binding() and receipt['native_fe']+receipt['cached_fe']==3000
            for key in ('paired_initial_sha256','paired_solver_rng_sha256'): assert receipt[key]==old[key]
            data=np.load(folder/f'b{batch:02d}/llm_no_anchor.npz')
            points=data['executed_points'];references=data['native_references']
            assert points.shape==(10,100,2,10) and np.array_equal(points[:,:,0],references[:,:,0])
            if group=='CEC-style':assert np.array_equal(points,references)
            with gzip.open(folder/f'b{batch:02d}/llm_no_anchor_DECISIONS.json.gz','rt') as stream: decisions=json.load(stream)
            assert len(decisions)==(0 if group=='CEC-style' else 45)
            if group!='CEC-style':
                with gzip.open(parent/f'b{batch:02d}/llm_live_DECISIONS.json.gz','rt') as stream: previous=json.load(stream)
                assert len(previous)==len(decisions)==45
                for record,old_record in zip(decisions,previous):
                    assert record['step']==old_record['step']
                    iteration=55+record['step']
                    for i,current in enumerate(record['current']):
                        assert current['anchor']['enabled'] is False
                        assert current['extra_origin']=='native_extra'
                        assert len(current['candidates'])==38
                        assert np.array_equal(points[i,iteration,1],np.asarray(current['candidates'][current['selected_index']],dtype=np.float32))
                        earlier=old_record['current'][i]
                        for slot,coordinate in [(0,earlier['protected_point']),(1,earlier['candidates'][earlier['selected_index']])]:
                            totals['r10_active_window_points_compared']+=1
                            totals['r10_active_window_points_different']+=int(not np.array_equal(points[i,iteration,slot],np.asarray(coordinate,dtype=np.float32)))
                for key in ('changed_points','fusion_changed_points','selected_adjusted_candidates','nondefault_fusion_rounds'):
                    totals[key]+=receipt['agent_stats'].get(key,0)
    assert len(rows)==35 and totals['new_native_fe']+totals['new_cached_fe']==1050000
    for scope in ('primary34','BBOB23','CEC6','UAV5','BBOB20_secondary'):
        selected=[r for r in rows if (r['primary'] if scope=='primary34' else r['group']=='BBOB' and r['fid']!=20 if scope=='BBOB23' else r['group']=='CEC-style' if scope=='CEC6' else r['group']=='UAV' if scope=='UAV5' else r['group']=='BBOB' and r['fid']==20)]
        comparisons.append(dict(scope=scope,function_wins=sum(r['full_mean']<r['no_anchor_mean'] for r in selected),
            function_ties=sum(r['full_mean']==r['no_anchor_mean'] for r in selected),function_losses=sum(r['full_mean']>r['no_anchor_mean'] for r in selected),
            trajectory_wins=sum(r['wins'] for r in selected),trajectory_ties=sum(r['ties'] for r in selected),trajectory_losses=sum(r['losses'] for r in selected)))
    write_csv(directory/'FUNCTION_COMPARISON.csv',rows)
    write_csv(directory/'TRAJECTORY_COMPARISON.csv',trajectory_rows)
    write_csv(directory/'SUMMARY.csv',comparisons)
    write_csv(directory/'BBOB20_SEPARATE.csv',[r for r in rows if not r['primary']])
    result=dict(status='full_live_no_anchor_records_verified',functions=35,trajectories=3500,nfe=300,
        paired_batches_verified=350,anchor_network_present=False,reference_rounds_verified=35000,
        parent_full='Frozen R10 actual live results reused as reference; no new Full calls',
        comparisons=comparisons,api=audit,totals=dict(totals),
        changed_points_definition='same-pool original selector; active_window_points_different is coordinate comparison to frozen Full, 90 points per non-CEC trajectory only',
        limitations=['Native trust-region substitutes keep 38 competing slots; this is Anchor-to-native replacement, not a retrained no-Anchor optimum.',
            'Original R10 offline memory is held fixed; Full-case returns may not transfer to this ablation.',
            'Anchor-temperature profiles are execution aliases with no temperature effect; LLM is explicitly informed.',
            'CEC structured lock retained on native references; no LLM intervention in CEC.',
            'BBOB20 secondary only; original batch-coupled stochastic semantics.',
            'Function and paired trajectory counts are not significance tests.'])
    run.legacy.write_json(directory/('INDEPENDENT_VERIFICATION.json' if verify else 'FINAL_RESULT.json'),result)
    return result
