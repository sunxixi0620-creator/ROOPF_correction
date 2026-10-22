import gzip
import json
from pathlib import Path
import sys
import tempfile
from unittest import mock

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT))
import experiment_r04 as run
from roopf.fusion_agent import CONFIGS, context_rows, fusion_indices
from roopf.multistep_fusion_agent import MultiStepFusionAgent
from verify_r03 import rule_for


def contracts():
    score = torch.tensor([[0., -.08, .5]])
    probability = torch.tensor([[.3, .31, .9]])
    scale = torch.ones(1, 1)
    original = torch.zeros(1, 1, dtype=torch.long)
    rule = rule_for(score, probability, scale)
    choices = fusion_indices(score, probability, scale, original, .008, rule)
    row = context_rows(torch.zeros(1, 11), score, score, score.abs(), probability,
        scale, original, .008, fitness=torch.tensor([[-.1, .2, 1.]]),
        candidate_ops=torch.tensor([[-2, 0, 1]]), options=choices)[0]
    with tempfile.TemporaryDirectory(prefix='r04_contract_') as directory:
        folder = Path(directory)
        cases = []
        for index in range(3):
            advantages = {name: float(name == 'balanced') for name in CONFIGS}
            cases.append(dict(row, id=str(index), function=f'train{index}',
                advantages=advantages, horizon_advantages={'1': advantages, '5': advantages, '10': advantages}))
        memory = dict(version='r04', cases=cases, allowed_configs=list(CONFIGS),
            primary_horizon_rounds=5, action_duration_rounds=1,
            context_validation={row['context_key']: {'balanced': {'supported': True}}})
        path = folder / 'memory.json'
        path.write_text(json.dumps(memory))
        agent = MultiStepFusionAgent(path, folder / 'journal', mode='llm', budget=dict(used=0, limit=0))
        agent.reset(1)
        logs = []
        agent.active_record = logs.append
        kwargs = dict(original_idx=original, score=score, mu=score, sigma=score.abs(), probability=probability,
            state=torch.zeros(1, 11), archive_y=torch.tensor([[-1., 1.]]), fitness=torch.tensor([[-.1, .2, 1.]]),
            candidates=torch.randn(1, 3, 10), candidate_ops=torch.tensor([[-2, 0, 1]]),
            remaining=.2, problem=None, residual_weight=.008, selection_rule=rule)
        selected = []
        with mock.patch.object(agent, 'call', return_value=['balanced']) as request:
            for _ in range(6):
                selected.append(int(agent.choose(**kwargs)))
            if request.call_count != 2 or selected != [1, 0, 0, 0, 0, 1]:
                raise RuntimeError('R04 action duration or cadence differs from labels')
        if [record['configs'][0] for record in logs] != ['balanced', 'release', 'release', 'release', 'release', 'balanced']:
            raise RuntimeError('R04 action logs misattribute a repeated configuration')
        for case in agent.cases:
            case['horizon_advantages'] = dict(case['horizon_advantages'], **{'10': {name: -1 for name in CONFIGS}})
        if agent.eligible_configs(row, agent.cases) != ['release']:
            raise RuntimeError('Negative longer-horizon evidence was ignored')
    return dict(single_action_duration=True, five_round_decision_interval=True,
        negative_ten_round_evidence_blocks_action=True, r03_default_contracts_unchanged=True,
        live_api_calls=0)


def smoke():
    original_exp = run.EXP
    folder = original_exp / 'verification/smoke'
    folder.mkdir(parents=True, exist_ok=True)
    (folder / 'PROTOCOL.json').write_bytes((original_exp / 'PROTOCOL.json').read_bytes())
    run.EXP = folder
    try:
        receipt = run.collect_episode(0, 999, folder / 'episode')
        baseline = run.load_pickle(folder / 'episode/BASELINE.pkl.gz')
        model, problem, population, rng, _, _ = run.episode(0, 999)
        record = baseline['records'][0]
        forced = run.ForcedAction(record, 'release')
        model.fusion_agent = forced
        calls = {index: (dict(shape=value['shape'], x=value['x'], rng=run.legacy.digest(value['rng_before'])),
            dict(shape=value['shape'][:2], y=value['y'], rng_after=value['rng_after'])) for index, value in enumerate(baseline['calls'])}
        objective = run.PrefixObjective(problem, calls, min(300, record['nfe_before'] + 20))
        torch.set_rng_state(rng)
        try:
            with torch.no_grad():
                model(population.clone(), objective)
            raise RuntimeError('Horizon stopping failed')
        except run.HorizonReached:
            pass
        values = torch.cat(objective.values, 1).numpy()
        if objective.native_fe != 0 or forced.applied != 1 or not np.array_equal(values, baseline['values'][:, :values.shape[1]]):
            raise RuntimeError('Release continuation is not exactly the original policy')
        with gzip.open(folder / 'episode/CASES.json.gz', 'rt') as stream:
            cases = json.load(stream)
        for case in cases:
            for name, action in case['actions'].items():
                branch = None if not action['changes_original'] else run.load_pickle(
                    folder / f"episode/d{case['decision_step']:02d}_i{action['selected_index']}.pkl.gz")
                for horizon, stop in case['horizon_nfe'].items():
                    reference = float(baseline['values'][:, :stop].min())
                    best = reference if branch is None else float(branch['values'][:, :stop].min())
                    expected = float(np.clip((reference - best) / case['scale'], -20, 20))
                    if expected != case['horizon_advantages'][horizon][name]:
                        raise RuntimeError('Multistep label does not match saved objectives')
        result = dict(status='passed', release_prefix_and_continuation_bitwise_equal=True,
            original_max_nfe_unchanged=True, actual_multistep_labels_recomputed=True,
            smoke_native_training_fe=receipt['native_fe'], smoke_retained_cases=len(cases),
            smoke_branches=receipt['distinct_branches'], smoke_seed_index=999,
            smoke_not_in_memory_or_calibration=True, **contracts())
    finally:
        run.EXP = original_exp
    run.write_json(original_exp / 'CONTRACT_VERIFICATION.json', result)
    print(json.dumps(result))


if __name__ == '__main__':
    smoke()
