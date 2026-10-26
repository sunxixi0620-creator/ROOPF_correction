import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import argparse
import gzip
import fcntl
import json
import multiprocessing
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT))
import experiment_r04 as generated
import experiment_r10_full as run
import r15_common as common
from roopf.scheduled_evidence_agent import ScheduledEvidenceAgent, FEATURES, POLICY_ID, PROFILES, digest

EXP, legacy = common.EXP, run.legacy
CID = None
SCHEDULE = None

def configure(cid):
    global EXP,CID,SCHEDULE
    CID=cid;EXP=common.EXP/'cases'/cid;SCHEDULE=common.schedule(cid)
    EXP.mkdir(parents=True,exist_ok=True)
    return SCHEDULE



def freeze_collection():
    identity = dict(binding=common.binding(),schedule=SCHEDULE,schedule_signature=digest(SCHEDULE))
    path = EXP / 'COLLECTION_BINDING.json'
    if path.exists() and json.loads(path.read_text()) != identity:
        raise RuntimeError('Collection source changed; preserve previous attempt')
    if not path.exists():
        legacy.write_json(path, identity)
    return identity


def reserve_native(amount):
    path=EXP/'NATIVE_BUDGET.json'
    with (EXP/'NATIVE_BUDGET.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        budget=common.read(path) if path.exists() else dict(native_fe=0,reserved_fe=0,cap=3000000)
        if budget['native_fe']+budget['reserved_fe']+amount>budget['cap']:
            raise RuntimeError('Prespecified offline native FE cap would be exceeded')
        budget['reserved_fe']+=amount;common.write(path,budget)

def settle_native(reserved,actual):
    assert 0<=actual<=reserved
    path=EXP/'NATIVE_BUDGET.json'
    with (EXP/'NATIVE_BUDGET.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        budget=common.read(path);budget['reserved_fe']-=reserved;budget['native_fe']+=actual
        assert budget['reserved_fe']>=0 and budget['native_fe']<=budget['cap']
        common.write(path,budget)

def episode(function_index, seed_index, forced, cache=None):
    reserve_native(300)
    model, problem, population, rng, seed, function = generated.episode(function_index, seed_index)
    radii = (1.0, 1.0, .3, .3, .1, .1, .03, .03)
    radius = radii[seed_index]
    population = problem.repaire(population[:, :1] + radius * (population - population[:, :1]))
    controller = ScheduledEvidenceAgent(None, EXP / 'collection_no_api', schedule=SCHEDULE, mode='collect')
    controller.forced = forced
    records = []
    controller.active_record = records.append
    model.fusion_agent = controller
    measured = legacy.Counted(problem, cache, stochastic=True)
    rng_boundaries=[]
    original_fitness=measured.calfitness
    def record_fitness(x):
        rng_boundaries.append(common.sha_bytes(torch.get_rng_state().numpy().tobytes()))
        return original_fitness(x)
    measured.calfitness=record_fitness
    torch.set_rng_state(rng)
    with torch.no_grad():
        _, trail, nfe, executed = model(population.clone(), measured)
    values = torch.cat(measured.values, 1).numpy()
    if nfe != 300 or values.shape != (1, 300) or len(records) != 45:
        raise RuntimeError('Offline episode incomplete')
    if not np.array_equal(values.min(1), trail[:, -1].numpy()):
        raise RuntimeError('Offline best trace mismatch')
    settle_native(300,measured.native_fe)
    return dict(executed_points=executed.numpy(),objective_rng_boundaries=rng_boundaries,records=records, values=values, calls=measured.new_calls, seed=seed,
        function=function, native_fe=measured.native_fe, cached_fe=measured.cached_fe,
        initial_population=population.numpy(), initial_rng=rng.numpy()), problem


def collect(task):
    cid,function_index, seed_index = task
    configure(cid)
    identity = freeze_collection()
    folder = EXP / 'collection' / f'f{function_index:02d}_s{seed_index:02d}'
    folder.mkdir(parents=True, exist_ok=True)
    marker = folder / 'COMPLETE.json'
    if marker.exists():
        receipt = json.loads(marker.read_text())
        if receipt['binding'] != identity:
            raise RuntimeError('Offline receipt binding mismatch')
        return receipt
    names = list(PROFILES)
    prior_profile = names[1 + (function_index + seed_index) % (len(names) - 1)]
    cases, native_total, diagnostic_total = [], 0, 0
    decision_steps=(SCHEDULE['first_call_active_step'],SCHEDULE['second_call_active_step'])
    cached_total=0
    for ordinal,decision_step in enumerate(decision_steps):
        prefix_actions = {} if ordinal == 0 else {decision_steps[0]: {0: prior_profile}}
        reference, problem = episode(function_index, seed_index, prefix_actions)
        native_total += reference['native_fe']
        cached_total += reference['cached_fe']
        run.storage.save_pickle(folder / f'd{decision_step:02d}_reference.pkl.gz', reference)
        cache = run.storage.merged_calls({}, reference['calls'])
        current = reference['records'][decision_step]['current'][0]
        context = current['generation']
        decision_features = reference['records'][decision_step]['decision_features'][0]
        prefix = context['nfe_before']
        scale = float(np.std(reference['values'][0, :prefix]))
        scale = max(scale, 1e-8)
        case = dict(id=f"{reference['function']}:s{seed_index}:d{decision_step}",
            function=reference['function'], seed_index=seed_index, step=decision_step,
            prefix_profile='matched38' if ordinal == 0 else prior_profile,
            reference_profile='matched38',
            features=decision_features, generation_context_sha256=digest(context), initial_radius=(1.,1.,.3,.3,.1,.1,.03,.03)[seed_index],
            scale=scale, nfe_before=prefix, advantages={'5': {}, '10': {}},
            candidate_diagnostics={}, policy_id=POLICY_ID,decision_ordinal=ordinal,
            schedule_signature=digest(SCHEDULE),call_fe_pair=[SCHEDULE['first_call_fe'],SCHEDULE['second_call_fe']])
        for name in names:
            if name == 'matched38':
                branch = reference
            else:
                forced = dict(prefix_actions)
                forced[decision_step] = {0: name}
                branch, problem = episode(function_index, seed_index, forced, cache)
                native_total += branch['native_fe']
                cached_total += branch['cached_fe']
                cache = run.storage.merged_calls(cache, branch['calls'])
                run.storage.save_pickle(folder / f'd{decision_step:02d}_{name}.pkl.gz', branch)
            if not np.array_equal(branch['values'][:, :prefix], reference['values'][:, :prefix]):
                raise RuntimeError('Counterfactual prefix differs')
            assert np.array_equal(branch['executed_points'][:,:prefix-100],reference['executed_points'][:,:prefix-100]),'Counterfactual prefix coordinates differ'
            assert branch['objective_rng_boundaries']==reference['objective_rng_boundaries'],'Counterfactual objective RNG differs'
            assert branch['records'][decision_step]['current'][0]['candidates']==current['candidates'],'Counterfactual predecision pool differs'
            assert branch['initial_population'].tobytes()==reference['initial_population'].tobytes() and branch['initial_rng'].tobytes()==reference['initial_rng'].tobytes()
            branch_context = branch['records'][decision_step]['current'][0]['generation']
            if digest(branch_context) != case['generation_context_sha256']:
                raise RuntimeError('Counterfactual pre-generation state differs')
            for horizon in (5, 10):
                stop = prefix + 2 * horizon
                reference_best = float(reference['values'][0, :stop].min())
                branch_best = float(branch['values'][0, :stop].min())
                case['advantages'][str(horizon)][name] = (reference_best - branch_best) / scale
            row = branch['records'][decision_step]['current'][0]
            candidates = torch.tensor(row['candidates'], dtype=torch.float32).unsqueeze(0)
            observations = []
            reserve_native(candidates.size(1))
            with torch.random.fork_rng(devices=[]), torch.no_grad():
                torch.manual_seed(reference['seed'] + 100000 + decision_step)
                for slot in range(candidates.size(1)):
                    observations.append(float(problem.calfitness(candidates[:, slot:slot + 1]).reshape(-1)[0]))
            diagnostic_total += candidates.size(1)
            settle_native(candidates.size(1),candidates.size(1))
            selected_value = observations[row['selected_index']]
            case['candidate_diagnostics'][name] = dict(values=observations,
                selected_index=row['selected_index'], sampled_pool_best=min(observations),
                sampled_selection_gap=selected_value - min(observations),
                anchor=row['anchor'], extra_origin=row['extra_origin'],
                never_provided_to_runtime=True, stochastic_snapshot_not_causal=True)
        cases.append(case)
    with gzip.open(folder / 'CASES.json.gz', 'wt') as stream:
        json.dump(cases, stream, allow_nan=False)
    receipt = dict(status='complete', config_id=CID,cached_fe=cached_total,logical_branch_episodes=44,logical_branch_fe=13200, function=reference['function'], seed_index=seed_index,
        cases=len(cases), native_fe=native_total, diagnostic_native_fe=diagnostic_total,
        binding=identity, real_api_calls=0)
    legacy.write_json(marker, receipt)
    return receipt


def coverage_radii(training, heldout):
    from roopf.current_evidence_agent import HISTORY_FEATURES
    features = np.asarray([r['features'] for r in training], dtype=float)
    scale = np.maximum(features.std(0), .05)
    radii = {}
    for name in ('full', 'no_online'):
        mask = np.ones(len(FEATURES), dtype=bool)
        if name == 'no_online':
            mask[list(HISTORY_FEATURES)] = False
        for step in (SCHEDULE['first_call_active_step'],SCHEDULE['second_call_active_step']):
            third = []
            for query in heldout:
                if query['step'] != step:
                    continue
                distances = np.mean(((features[:,mask]-np.asarray(query['features'])[mask])/scale[mask])**2,axis=1)
                by_function = {}
                for i, row in enumerate(training):
                    if row['step'] == step:
                        by_function[row['function']] = min(by_function.get(row['function'], float('inf')), float(distances[i]))
                third.append(sorted(by_function.values())[2])
            radii[name+':'+str(step)] = max(1e-8, float(np.quantile(third, .95)))
    return radii


def memory(cid):
    configure(cid)
    cases, receipts = [], []
    identity = freeze_collection()
    for function_index in range(36):
        for seed_index in range(8):
            folder = EXP / 'collection' / f'f{function_index:02d}_s{seed_index:02d}'
            receipt = json.loads((folder / 'COMPLETE.json').read_text())
            if receipt['binding'] != identity:
                raise RuntimeError('Mixed-policy collection')
            receipts.append(receipt)
            with gzip.open(folder / 'CASES.json.gz', 'rt') as stream:
                cases.extend(json.load(stream))
    training = [row for row in cases if row['function'] not in legacy.HELDOUT]
    heldout = [row for row in cases if row['function'] in legacy.HELDOUT]
    native_fe = sum(row['native_fe'] + row['diagnostic_native_fe'] for row in receipts)
    budget=common.read(EXP/'NATIVE_BUDGET.json')
    assert budget['native_fe']==native_fe and budget['reserved_fe']==0,'Offline budget ledger mismatch'
    if native_fe > common.PROTOCOL['offline']['native_fe_cap_per_configuration']:
        raise RuntimeError('Offline objective budget exceeded')
    compact = [{key: row[key] for key in ('id', 'function', 'features', 'advantages', 'policy_id', 'prefix_profile', 'reference_profile', 'step', 'initial_radius','decision_ordinal','nfe_before','schedule_signature','call_fe_pair')}
        for row in training]
    result = dict(policy_id=POLICY_ID, cases=compact, feature_names=FEATURES,
        allowed_configs=list(PROFILES), heldout_functions=sorted(legacy.HELDOUT),
        schedule=SCHEDULE,schedule_signature=digest(SCHEDULE),action_duration=SCHEDULE['fusion_rounds'],generation_rounds=SCHEDULE['generation_rounds'], generation_delay_rounds=1, candidate_count=38, no_test_suite_labels=True, default_profile='matched38',
        advantage_reference='matched38',
        policy_screening=False, collection_binding=identity)
    result['coverage_radii'] = coverage_radii(training, heldout)
    assert len(training)==464 and len(heldout)==112
    legacy.write_json(EXP/'offline_memory.json', result)
    with gzip.open(EXP / 'HELDOUT_CASES.json.gz', 'wt') as stream:
        json.dump(heldout, stream, allow_nan=False)
    summary = dict(config_id=CID,schedule_signature=digest(SCHEDULE),memory_sha256=common.sha(EXP/'offline_memory.json'),
        cached_fe=sum(r['cached_fe'] for r in receipts),logical_branch_episodes=12672,diagnostic_native_fe=sum(r['diagnostic_native_fe'] for r in receipts),status='collected_not_performance_validated', episodes=len(receipts),
        training_cases=len(training), heldout_cases=len(heldout), native_fe=native_fe,
        policy_screening=False, retained_profiles=list(PROFILES))
    legacy.write_json(EXP / 'COLLECTION_SUMMARY.json', summary)
    print(json.dumps(summary), flush=True)


def worker(task):
    try:return collect(task)
    except BaseException:
        import time,traceback
        common.write(common.EXP/'failures'/f'offline_{task[0]}_f{task[1]}_s{task[2]}_{time.time_ns()}.json',dict(task=task,traceback=traceback.format_exc()))
        raise
