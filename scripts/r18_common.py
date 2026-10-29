"""Immutable inputs and identity for the prespecified R15 factorial experiment."""
from pathlib import Path
import hashlib,json,os,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
EXP=ROOT/'experiments/r18_contract_recovery'
PROTOCOL=json.loads((EXP/'PROTOCOL.json').read_text())
CONFIGS={c['config_id']:c for c in PROTOCOL['configs']}
RUNTIME=json.loads((ROOT/'experiments/r14_online_entry_timing/HARDWARE_RUNTIME.json').read_text())
SOURCES=('roopf/scheduled_evidence_agent_r18.py','roopf/policy_api_r18.py','scripts/r18_common.py','scripts/collect_r18.py',
         'scripts/experiment_r18_schedule.py','scripts/check_r18_schedule.py','scripts/report_r18_schedule.py','experiments/r18_contract_recovery/variant_model.py','scripts/publish_r18_records.py','scripts/audit_r18_parallel.py','scripts/summarize_r18_crossround.py','scripts/run_r18_pipeline.py')
RUN=None

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def sha_bytes(b):return hashlib.sha256(b).hexdigest()
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    q=p.with_suffix(p.suffix+'.pending');q.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n');q.replace(p)
def validate_frozen():
    for name,h in read(EXP/'PLAN_BINDING.json')['sha256'].items():assert sha(EXP/name)==h,name
    for name,h in read(EXP/'BASELINE_SNAPSHOT.json')['sha256'].items():assert sha(ROOT/name)==h,name
    registry=read(ROOT/'experiments/r14_online_entry_timing/FROZEN_PARENT.json')
    for name,h in registry['sha256'].items():assert sha(ROOT/name)==h,name
    return True

def binding():
    return dict(plan_sha256=sha(EXP/'PLAN_BINDING.json'),frozen_snapshot_sha256=sha(EXP/'BASELINE_SNAPSHOT.json'),
                code_sha256={p:sha(ROOT/p) for p in SOURCES},hardware_sha256=digest(RUNTIME))

def schedule(cid):
    from roopf.scheduled_evidence_agent import ACTION_NAMES,FEATURES,POLICY_ID,PROFILES
    return dict(CONFIGS[cid],online_admission_start_fe=200,agent_generation_and_factual_history_start_fe=200,policy_version=POLICY_ID,action_menu_hash=digest(PROFILES),feature_schema_hash=digest(FEATURES),
        training_function_hash=sha(ROOT/'experiments/r02_fusion_agent/training_assets/generated36_curated.py'),
        seed_and_split_hash=digest(dict(seed_formula='2026102400+1000*seed_index+function_index',
            heldout=sorted(['comp_global','comp_rugged','dirridge_noisy','lowrank_tanh','plateau_sharp','ring_single','wavepacket_local']),
            radii=PROTOCOL['offline']['initial_radii'])),backend_signature=digest(RUNTIME))

def configure(address=None,authkey=None):
    global RUN
    import experiment_r13_cec_uniform as parent
    RUN=parent.configure(str(EXP),RUNTIME,address,authkey)
    RUN.settings=lambda:dict(api=PROTOCOL['fixed_conditions']['api'],workers=24,max_new_api_attempts=19600,offline_native_fe_cap=3000000)
    return RUN

def freeze():
    validate_frozen();value=binding();p=EXP/'CODE_BINDING.json'
    if p.exists():assert read(p)==value,'R15 source changed after execution began'
    else:write(p,value)
    return value
