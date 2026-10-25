"""Verify R13 published archives without local results/ or paid API calls."""
from pathlib import Path
from collections import Counter
import csv
import gzip
import hashlib
import io
import json
import sys
import tarfile
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from roopf.policy_api_r10 import validate_response,wire_bytes
from roopf.current_evidence_agent import request_context_digest

EXP=ROOT/'experiments/r13_cec_uniform'
ARMS=['frozen_baseline','no_agent','llm_full','llm_no_anchor','llm_no_online_generators']


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    publication=EXP/'publication';manifest=json.loads((publication/'MANIFEST.json').read_text())
    for name,v in manifest['files'].items():assert sha(publication/name)==v['sha256']
    with (EXP/'FUNCTION_STATISTICS.csv').open(encoding='utf-8-sig',newline='') as f:
        stats={(int(r['fid']),r['arm']):r for r in csv.DictReader(f)}
    journals={}
    with tarfile.open(publication/'ACTUAL_API_JOURNAL.tar.xz') as tar:
        for member in tar.getmembers():
            e=json.load(tar.extractfile(member));identity=Path(member.name).stem
            assert hashlib.sha256(wire_bytes(e['request'])).hexdigest()==identity
            assert e['status']!='pending'
            payload=json.loads(e['request']['messages'][-1]['content'].split('INPUT:\n',1)[1])
            if e['status'] in ('ok','partial'):
                decoded=validate_response(e['response'],payload)
                assert all(decoded[k]==e[k] for k in decoded)
            journals[identity]=e
    assert len(journals)==360
    consumed=set();records=0;receipts=0;points_checked=0
    for fid in range(1,7):
        name=f'CEC-style_f{fid:02d}';points={};values={a:[] for a in ARMS}
        with tarfile.open(publication/f'EVIDENCE_{name}.tar.xz') as tar:
            for batch in range(10):
                for arm in ARMS:
                    base=f'results/{name}/b{batch:02d}/{arm}'
                    r=json.load(tar.extractfile(base+'.json'))
                    assert r['native_fe']+r['cached_fe']==3000 and r['objective_rng_boundaries_matched']==201
                    assert r['objective_name']==f'cecf{fid}'
                    assert r['final_rng_matches_frozen_original']
                    data=np.load(io.BytesIO(tar.extractfile(base+'.npz').read()))
                    assert data['values'].shape==(10,300) and np.isfinite(data['values']).all()
                    assert data['executed_points'].shape==(10,100,2,10)
                    assert np.array_equal(data['executed_points'][:,:,0],data['reference_points'][:,:,0])
                    points[batch,arm]=data['executed_points'];values[arm].append(data['values']);receipts+=1
            for arm in ARMS:
                data=np.load(io.BytesIO(tar.extractfile(f'results/{name}/{arm}_FULL100.npz').read()))['values']
                assert np.array_equal(data,np.concatenate(values[arm]));final=data.min(1).astype(float);r=stats[fid,arm]
                assert float(final.mean())==float(r['mean']) and float(final.var(ddof=0))==float(r['variance'])
                assert float(final.std(ddof=0))==float(r['population_std'])
        last={};counts=Counter()
        with gzip.open(publication/f'EXECUTED_DECISIONS_{name}.jsonl.gz','rt') as f:
            for line in f:
                wrapper=json.loads(line);r=wrapper['decision'];arm=wrapper['arm'];batch=int(wrapper['batch'][1:]);k=r['step']
                key=(batch,arm);counts[key]+=1
                if k in (5,15):last[key]=r['configs']
                expected=last[key] if k in (5,6,7,8,9,15,16,17,18,19) else ['release']*10
                assert r['configs']==expected
                applied=['matched38']*10 if k in (5,15) else ([n.split('@')[0] for n in last[key]] if k in (6,7,8,9,16,17,18,19) else ['release']*10)
                assert r['applied_generation']==applied
                target=points[batch,arm][:,55+k]
                if k in (5,15):
                    identity=r['api_request_sha256'];e=journals[identity]
                    assert r['execution_context']==e['execution_context']
                    assert request_context_digest(r['api_trajectories'],r['execution_context'])==e['decision_context_sha256']
                    assert r['configs']==e['config_ids'];consumed.add(identity)
                    actual=np.asarray([[c['protected_point'],c['candidates'][c['selected_index']]] for c in r['current']],dtype=np.float32)
                else:
                    actual=np.stack([np.asarray(r['protected_reference_points'],dtype=np.float32),np.asarray(r['executed_competing_points'],dtype=np.float32)],axis=1)
                assert np.array_equal(target,actual)
                points_checked+=20;records+=1
        assert len(counts)==30 and all(v==45 for v in counts.values())
    assert receipts==300 and records==8100 and points_checked==162000 and consumed==set(journals)
    result=dict(status='published_measurements_and_actual_api_verified',archives_verified=len(manifest['files']),
        functions=6,conditions=5,trajectory_records=3000,arm_batch_receipts=receipts,
        active_batch_decision_records=records,active_executed_points_checked=points_checked,
        actual_api_requests_matched=len(consumed),new_api_calls=0,unused_pool_predictions_not_required=True,
        script_sha256=sha(Path(__file__)))
    (EXP/'PUBLICATION_VERIFICATION.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()
