"""Engineering checks; these never substitute for full live-API validation."""
import gzip
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import secrets
import sys
import tempfile
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'scripts'))
from hardware_backend import serve, write_json


def main():
    from roopf.policy_api_r10 import validate_response
    from roopf.current_evidence_agent import ACTION_NAMES, compact
    def response(actions,finish='stop'):
        return dict(choices=[dict(finish_reason=finish,message=dict(content=json.dumps(dict(actions=actions))))])
    payload=dict(action_names=ACTION_NAMES,trajectories=[dict(trajectory_id=str(i),allowed=ACTION_NAMES) for i in range(22)])
    out=validate_response(response({str(i):n for i,n in enumerate(ACTION_NAMES)}),payload)
    assert out['config_ids']==ACTION_NAMES and out['status']=='ok'
    partial=validate_response(response({str(i):n if i!=3 else 'invalid' for i,n in enumerate(ACTION_NAMES)}),payload)
    assert partial['status']=='partial' and len(partial['valid_trajectory_ids'])==21 and partial['config_ids'][3]=='matched38'
    bad=[response({'0':'matched38'},'length'),response({'unknown':'matched38'}),
         dict(choices=[dict(finish_reason='stop',message=dict(content='{"actions":{"0":"matched38","0":"release"}}'))])]
    for r in bad:
        try:validate_response(r,payload)
        except ValueError:pass
        else:raise AssertionError('Ambiguous/incomplete output accepted')
    assert compact(1e-12)!=0
    p=ROOT/'experiments/r10_current_candidate_fusion/engineering_checks'
    p.mkdir(exist_ok=True)
    raw=Path(tempfile.mkdtemp(prefix='attempt_',dir=p))
    checks=dict(status='running',actual_api_calls=0,full_live_validation=False,contract_partial_rows_verified=True,
                duplicate_keys_rejected=True,small_gain_precision_preserved=True)
    context=mp.get_context('spawn')
    with tempfile.TemporaryDirectory(prefix='r10-check-') as t:
        ready,child=context.Pipe(duplex=False);address=str(Path(t)/'gpu.sock');key=secrets.token_bytes(32)
        process=context.Process(target=serve,args=(address,key,'0',child,str(raw/'GPU_STATS.json'),2))
        process.start();child.close()
        try:
            assert ready.poll(60)
            meta=ready.recv();assert meta['status']=='ready',meta
            import numpy as np
            import torch
            from roopf.current_evidence_agent import CurrentEvidenceAgent,PROFILES,POLICY_ID
            from run_r10_hardware import configure
            run,collection=configure(str(p.parent),dict(device='cuda',gpu_min_batch=2),address,key)
            for fid in (9,20):
                run.legacy.build(original=True)
                torch.set_rng_state(torch.from_numpy(np.frombuffer((run.legacy.SNAP/f'BBOB_f{fid-1:02d}/FULL_RNG.bin').read_bytes(),dtype=np.uint8).copy()))
                problem=run.legacy.problem_for('BBOB',fid);population=problem.genRandomPop((10,100,10));start=torch.get_rng_state().clone()
                calls,_=run.storage.full_calls('BBOB',fid)
                model=run.legacy.build();agent=CurrentEvidenceAgent(None,raw/'no_api',mode='collect',arm='matched38')
                records=[];agent.active_record=records.append;model.fusion_agent=agent
                objective=run.legacy.Counted(problem,calls[0],stochastic=fid==20)
                torch.set_rng_state(start)
                with torch.no_grad():model(population.clone(),objective)
                frozen=np.load(ROOT/f'experiments/r07_hardware_accelerated_v1/results/BBOB_f{fid:02d}/b00/matched38.npz')['values']
                assert np.array_equal(torch.cat(objective.values,1).numpy(),frozen),f'Frozen matched38 differs f{fid}'
                for record in records:
                    assert record['selected']==[[r['fusion'][1][1]] for r in record['current_options']] if record['step'] in range(5,10) or record['step'] in range(15,20) else True
                print(f'Exact matched38 GPU restoration and current-pool previews passed: BBOB{fid}',flush=True)
            reference,_=collection.episode(0,0,{})
            prefix=reference['records'][5]['current'][0]['generation']['nfe_before']
            for name in ACTION_NAMES:
                branch,_=collection.episode(0,0,{5:{0:name}})
                records=branch['records']
                assert np.array_equal(branch['values'][:,:prefix],reference['values'][:,:prefix])
                assert records[5]['applied_generation']==['matched38']
                assert records[5]['configs']==[name]
                for step in range(6,10):
                    assert records[step]['applied_generation']==[name.split('@')[0]]
                if name!='release':
                    multiplier=int(PROFILES[name]['residual_multiplier'])
                    assert records[5]['selected'][0][0]==records[5]['current_options'][0]['fusion'][multiplier][1]
                assert records[10]['applied_generation']==['release']
                assert records[5]['policy_id']==POLICY_ID
                print('Paired prefix and delayed generation verified: '+name,flush=True)
            r=reference['records'][5]
            raw_payload=dict(trajectories=[dict(trajectory_id='0',current=r['current'][0]['generation'],
                decision_features=r['decision_features'][0],current_options=r['current_options'][0],
                evidence=dict(enabled=True,supported=False,neighbors=[],rows=[],allowed=['matched38']),eligible_configs=['matched38'])])
            probe=CurrentEvidenceAgent(None,raw/'compat',mode='collect')
            probe.execution_context=dict(engineering_compatibility=True)
            probe.steps=5
            (p/'COMPATIBILITY_PAYLOAD.json').write_text(json.dumps(probe.compact_payload(raw_payload),indent=2))
            previous=json.loads((p.parent/'PRE_MIGRATION.json').read_text())
            assert all(hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==sha for n,sha in previous['previous_records_and_sources_sha256'].items())
            checks.update(status='engineering_checks_passed',frozen_matched38_batches=['BBOB9_b00','BBOB20_b00'],
                all22_temporal_prefix_checks=True,previous_sources_and_records_unchanged=True)
        except BaseException:
            checks.update(status='failed',traceback=traceback.format_exc());raise
        finally:
            write_json(raw/'CHECKS.json',checks)
            if process.is_alive() and Path(address).exists():
                from multiprocessing.connection import Client
                with Client(address,family='AF_UNIX',authkey=key) as connection:connection.send(dict(op='shutdown'))
            process.join(timeout=5)
            if process.is_alive():process.terminate();process.join(timeout=5)
    write_json(p/'CHECKS.json',checks)


if __name__=='__main__':main()
