import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from roopf.reviewed_fusion_agent import ReviewedFusionAgent
from roopf.fusion_rules import protected_indices

EXP = ROOT / 'experiments/r06_event_reviewed_fusion'


def main():
    agent = ReviewedFusionAgent(EXP / 'BOOTSTRAP.json', EXP / 'contract_no_api', mode='llm')
    agent.reset(1)
    payloads, logs = [], []
    def stub(payload):
        payloads.append(payload)
        return ['adaptive_fusion' if 'adaptive_fusion' in item['eligible_configs'] else 'release' for item in payload['trajectories']]
    agent.call, agent.active_record = stub, logs.append
    candidates = torch.arange(370, dtype=torch.float32).reshape(1, 37, 10)
    probability = torch.full((1, 37), .1)
    probability[0, 2] = .9
    online = torch.ones(1, 37)
    online[0, 0], online[0, 1] = 0, -.3
    archive = torch.tensor([[0.0, 1.0, 2.0]])
    scale = archive.std(1, keepdim=True, unbiased=False)
    signal = ((probability - probability.mean(1, keepdim=True)) / probability.std(1, keepdim=True, unbiased=False)).clamp(-2.5, 2.5)
    score = online - .008 * scale * signal
    state = torch.zeros(1, 11)
    state[0, 6], state[0, 7] = .3, .7
    operations = torch.zeros(1, 37, dtype=torch.long)
    operations[0, 0] = -2
    def selector(acquisition, proposed, margin=.1, soft_margin=.04, online_only=False):
        return protected_indices(acquisition, probability, scale, proposed, 1,
            ablation=set() if online_only else {'residual_safety_gate'},
            portfolio_seen=torch.zeros(1), portfolio_best_count=torch.zeros(1),
            portfolio_admit_count=torch.zeros(1), portfolio_no_admit=torch.zeros(1),
            stagnation=torch.zeros(1), remaining=.3, local_proxy_friendly=True,
            hpo=False, embedded=False, selected_ops=operations[:, 1:], op_penalty=None,
            margin=margin, soft_margin=soft_margin)
    for step in range(16):
        agent.choose(original_idx=torch.zeros(1, 1, dtype=torch.long), score=score, mu=online,
            sigma=torch.full((1, 37), .1), probability=probability, state=state,
            archive_y=archive, fitness=torch.tensor([[0.0, 1.0, 2.0]]),
            candidates=candidates, candidate_ops=operations, remaining=.3,
            problem=None, residual_weight=.008, selection_rule=selector)
        agent.observe(torch.tensor([[-.1, -.2]]), torch.tensor([[0.0, -.1]]),
            torch.tensor([[.1, .8]]), torch.tensor([0.0]), torch.tensor([[-2, 1]]), archive)
    epochs = [row['step'] for row in logs if row['request_event']]
    if epochs != [5, 10] or payloads[1]['request_kind'] != 'review':
        raise RuntimeError('Event and feedback review schedule broken')
    review = payloads[1]['trajectories'][0]['observed_feedback']
    if not review['completed_reviews'] or review['adaptation']['config_incumbent_utility']['adaptive_fusion']['n'] < 1:
        raise RuntimeError('Action feedback not consumed at next model decision')
    if len(logs[5]['feedback'][0]['competing_slot_recent']) != 5:
        raise RuntimeError('Feedback snapshot changed after observation')
    probe = ReviewedFusionAgent(EXP / 'BOOTSTRAP.json', EXP / 'contract_no_api', mode='collect')
    probe.reset(1)
    probe.pending, probe.steps = ['adaptive_fusion'], 1
    probe.observe(torch.tensor([[.3, .2]]), torch.tensor([[.3, .2]]), torch.tensor([[.1, .8]]),
        torch.tensor([0.0]), torch.tensor([[-2, 1]]), archive)
    if probe.config_returns[0]['adaptive_fusion']['mean'] != -.001:
        raise RuntimeError('Reward still favors beating Anchor without improving incumbent')
    before = probe.adaptation(0)['online_margin']
    probe.reliability[0]['residual_success'] = 0.0
    if probe.adaptation(0)['online_margin'] != before:
        raise RuntimeError('Online-only threshold depends on Residual')
    compact = json.dumps(payloads[0])
    if len(compact.encode()) > 12000 or 'old_shortlist_horizon_advantages' in compact:
        raise RuntimeError('Payload not compact/current-policy aligned')
    result = dict(status='passed', mocked_unit_test_only=True, real_api_calls=0,
        event_epochs=epochs, observed_action_feedback_reused=True,
        incumbent_aligned_reward=True, recent_rank_window=16,
        immutable_pre_action_feedback=True, online_only_independence=True,
        current37_policy_id_enforced=True, compact_single_trajectory_bytes=len(compact.encode()))
    (EXP / 'CONTRACT_VERIFICATION.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
