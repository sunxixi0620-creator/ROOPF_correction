import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from roopf.adaptive_fusion_agent import AdaptiveFusionAgent
from roopf.fusion_rules import protected_indices


def main():
    folder = ROOT / 'experiments/r05_adaptive_evidence_fusion'
    agent = AdaptiveFusionAgent(ROOT / 'experiments/r04_multiseed_horizon/offline_memory.json',
                                folder / 'contract_no_api', mode='retrieval')
    agent.reset(1)
    before = agent.adaptation(0)
    agent.pending = ['adaptive_fusion']
    agent.observe(torch.tensor([[1.0, 0.0]]), torch.tensor([[0.0, 1.0]]),
        torch.tensor([[0.1, 0.9]]), torch.tensor([0.5]), torch.tensor([[-2, 1]]),
        torch.tensor([[0.0, 1.0, 2.0]]))
    after = agent.adaptation(0)
    assert after['residual_multiplier'] > before['residual_multiplier']
    assert after['margin'] != before['margin']
    assert after['config_observed_relative_to_protected_anchor']['adaptive_fusion']['n'] == 1
    online_margin = after['online_margin']
    agent.reliability[0]['residual_success'] = 0.0
    assert agent.adaptation(0)['online_margin'] == online_margin
    snapshot = agent.feedback_summary(0)
    agent.history[0].append(dict(future=True))
    assert len(snapshot['competing_slot_recent']) == 1
    acquisition = torch.tensor([[0.0, -0.07]])
    probability = torch.tensor([[0.1, 0.9]])
    common = dict(scale=torch.ones(1, 1), proposed=torch.ones(1, 1, dtype=torch.long), base_count=1,
        portfolio_seen=torch.zeros(1), portfolio_best_count=torch.zeros(1), portfolio_admit_count=torch.zeros(1),
        portfolio_no_admit=torch.zeros(1), stagnation=torch.zeros(1), remaining=0.1,
        local_proxy_friendly=True, hpo=False, embedded=False,
        selected_ops=torch.ones(1, 1, dtype=torch.long), op_penalty=None)
    offline = protected_indices(acquisition, probability, ablation={'residual_safety_gate'}, **common)
    online = protected_indices(acquisition, probability, ablation=set(), **common)
    assert int(offline[0, 0]) == 1 and int(online[0, 0]) == 0
    result = dict(status='passed', feedback_changes_weight=True, feedback_changes_margin=True,
        configuration_returns_updated=True, immutable_feedback_snapshot=True,
        pure_online_disables_residual_acceptance=True, online_margin_independent_of_residual=True, api_calls=0,
        full_integration_and_original_equivalence='validated by full runner, not this contract check')
    (folder / 'CONTRACT_VERIFICATION.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


if __name__ == '__main__':
    main()
