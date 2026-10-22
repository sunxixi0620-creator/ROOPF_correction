import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from unittest import mock

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from roopf.fusion_agent import CONFIGS, FusionAgent, context_rows, fusion_indices
from roopf.fusion_rules import protected_indices


def rule_for(score, probability, scale, **overrides):
    batch = score.size(0)
    zeros = torch.zeros(batch)
    settings = dict(ablation={'residual_safety_gate'}, portfolio_seen=zeros,
        portfolio_best_count=zeros, portfolio_admit_count=zeros, portfolio_no_admit=zeros,
        stagnation=zeros, remaining=.2, local_proxy_friendly=True, hpo=False, embedded=False,
        selected_ops=torch.zeros(batch, score.size(1) - 1, dtype=torch.long), op_penalty=None)
    settings.update(overrides)
    return lambda acquisition, proposed, margin=.10, soft_margin=.04: protected_indices(
        acquisition, probability, scale, proposed, 1, margin=margin, soft_margin=soft_margin, **settings)


def verify():
    torch.manual_seed(73)
    score = torch.randn(10000, 3)
    probability = torch.rand(10000, 3)
    scale = torch.rand(10000, 1) + .1
    rule = rule_for(score, probability, scale)
    original = rule(score, score.argmin(1, keepdim=True))
    choices = fusion_indices(score, probability, scale, original, .008, rule)
    assert torch.equal(choices['release'], original)
    assert not ((original > 0) & (choices['balanced'] == 0)).any()
    assert not ((original == 0) & (choices['cautious'] > 0)).any()
    transformed = fusion_indices(score * 7 + 3, probability, scale * 7, original, .008,
        rule_for(score * 7 + 3, probability, scale * 7))
    assert all(torch.equal(choices[name], transformed[name]) for name in CONFIGS)
    close_score = torch.tensor([[0., -.2, -.2]])
    close_prob = torch.tensor([[.1, .2, .9]])
    close_rule = rule_for(close_score, close_prob, torch.ones(1, 1))
    reweighted = fusion_indices(close_score, close_prob, torch.ones(1, 1), torch.tensor([[1]]), .008, close_rule)
    assert reweighted['proxy'].item() == 1 and reweighted['residual'].item() == 2
    sample_score = torch.tensor([[0., -.05, .5]])
    sample_prob = torch.tensor([[.1, .3, .8]])
    sample_rule = rule_for(sample_score, sample_prob, torch.ones(1, 1))
    assert int(sample_rule(sample_score, torch.tensor([[1]]))) == 1
    zeros = torch.zeros(1, 1, dtype=torch.long)
    penalty_rule = rule_for(sample_score, sample_prob, torch.ones(1, 1), op_penalty=torch.ones(1, 1))
    assert int(penalty_rule(sample_score, torch.tensor([[1]]), margin=.01)) == 0
    score = torch.tensor([[0., -.08, .5]])
    probability = torch.tensor([[.3, .31, .9]])
    scale = torch.ones(1, 1)
    rule = rule_for(score, probability, scale)
    choices = fusion_indices(score, probability, scale, zeros, .008, rule)
    rows = context_rows(torch.zeros(1, 11), score, score, score.abs(), probability, scale, zeros, .008,
        fitness=torch.tensor([[-.1, .2, 1.]]), candidate_ops=torch.tensor([[-2, 0, 1]]), options=choices)
    assert choices['balanced'].item() == 1
    with tempfile.TemporaryDirectory(prefix='r03_contract_') as directory:
        path = Path(directory)
        cases = [dict(rows[0], id=str(index), function=f'train{index}',
            advantages={name: float(name == 'balanced') for name in CONFIGS}) for index in range(3)]
        memory = dict(version='r03', cases=cases, allowed_configs=list(CONFIGS),
            context_validation={rows[0]['context_key']: {'balanced': {'supported': True}}})
        (path / 'memory.json').write_text(json.dumps(memory))
        agent = FusionAgent(path / 'memory.json', path / 'journal', budget=dict(used=0, limit=1))
        records = []
        agent.active_record = records.append
        agent.reset(1)
        assert agent.active_record == records.append
        assert agent.feedback_summary(0)['mean_proxy_error'] is None
        assert agent.feedback_summary(0)['mean_residual_error'] is None
        assert len({case['function'] for case in agent.retrieve(rows[0])}) == 3
        with mock.patch.object(agent, 'call', return_value=['balanced']) as request:
            chosen = agent.choose(original_idx=zeros, score=score, mu=score, sigma=score.abs(),
                probability=probability, state=torch.zeros(1, 11), archive_y=torch.tensor([[-1., 1.]]),
                fitness=torch.tensor([[-.1, .2, 1.]]), candidates=torch.randn(1, 3, 10),
                candidate_ops=torch.tensor([[-2, 0, 1]]), remaining=.2, problem=None,
                residual_weight=.008, selection_rule=rule)
            assert request.call_count == 1 and chosen.item() == 1
        assert len(records) == 1 and records[0]['selected'] == [[1]]
        agent.observe(torch.tensor([[-.2, -.3]]), torch.tensor([[-.2, -.25]]),
            torch.tensor([[float('nan'), .8]]), torch.tensor([-.1]), torch.tensor([[-2, 0]]), torch.tensor([[-1., 1.]]))
        assert agent.feedback[0]['residual_n'] == 1 and len(agent.history[0]) == 1
        assert agent.history[0][0]['config'] == 'balanced'
        assert agent.feedback_summary(0)['mean_residual_error'] == .04
        payload = dict(trajectories=[dict(trajectory_id='7', eligible_configs=['release'])])
        def transport(*args, **kwargs):
            request_file = next((path / 'journal').glob('*.request.payload'))
            body = dict(choices=[dict(finish_reason='stop', message=dict(content='{"config_ids":{"7":"residual"}}'))])
            Path(str(request_file) + '.response.raw').write_text(json.dumps(body))
            return SimpleNamespace(returncode=0, stdout='', stderr='')
        with mock.patch('roopf.fusion_agent.subprocess.check_output', side_effect=lambda args, **kwargs: args[-1]), mock.patch('roopf.fusion_agent.subprocess.run', side_effect=transport):
            try:
                agent.call(payload)
                raise AssertionError('Unsupported action accepted')
            except ValueError:
                pass
        assert agent.budget['used'] == 1
        assert list((path / 'journal').glob('*.json'))
        contracts = [
            ('valid', 'stop', '{"config_ids":{"7":"release"}}', True),
            ('wrong_ids', 'stop', '{"config_ids":{"0":"release"}}', False),
            ('array', 'stop', '{"config_ids":["release"]}', False),
            ('extra', 'stop', '{"config_ids":{"7":"release"},"explanation":"x"}', False),
            ('truncated', 'length', '{"config_ids":{"7":"release"}}', False),
            ('empty', 'stop', '', False),
        ]
        agent.budget['limit'] = len(contracts) + 1
        for label, finish_reason, content, valid in contracts:
            def contract_transport(*args, **kwargs):
                request_file = Path(kwargs['args'][-1]) if 'args' in kwargs else Path(args[0][-1])
                body = dict(choices=[dict(finish_reason=finish_reason, message=dict(content=content))])
                Path(str(request_file) + '.response.raw').write_text(json.dumps(body))
                return SimpleNamespace(returncode=0, stdout='', stderr='')
            with mock.patch('roopf.fusion_agent.subprocess.check_output', side_effect=lambda args, **kwargs: args[-1]), mock.patch('roopf.fusion_agent.subprocess.run', side_effect=contract_transport):
                try:
                    result = agent.call(dict(payload, contract_case=label))
                    assert valid and result == ['release']
                except ValueError:
                    assert not valid
        for context in agent.memory['context_validation'].values():
            context.clear()
        agent.reset(1)
        with mock.patch.object(agent, 'call', side_effect=AssertionError('Unsupported action must not call API')) as request:
            chosen = agent.choose(original_idx=zeros, score=score, mu=score, sigma=score.abs(),
                probability=probability, state=torch.zeros(1, 11), archive_y=torch.tensor([[-1., 1.]]),
                fitness=torch.tensor([[-.1, .2, 1.]]), candidates=torch.randn(1, 3, 10),
                candidate_ops=torch.tensor([[-2, 0, 1]]), remaining=.2, problem=None,
                residual_weight=.008, selection_rule=rule)
            assert chosen.item() == 0 and request.call_count == 0
    result = dict(status='passed', randomized_gate_cases=10000, residual_or_regression=True,
        relaxed_threshold_monotonic=True, strict_threshold_monotonic=True,
        scale_invariance=True, operation_penalty_preserved=True, callback_survives_reset=True,
        labelled_evidence_and_actions=True, unknown_errors_are_null=True,
        distinct_function_retrieval=True, competing_slot_credit_recorded=True,
        unsupported_api_action_rejected=True, unsupported_state_skips_api=True,
        api_contract_cases=7, truncated_output_rejected=True,
        reweighting_can_change_portfolio_ranking=True,
        real_api_calls=0, real_objective_evaluations=0)
    target = ROOT / 'experiments/r03_evidence_fusion/CONTRACT_VERIFICATION.json'
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    verify()
