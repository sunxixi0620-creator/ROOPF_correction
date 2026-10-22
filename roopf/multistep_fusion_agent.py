from __future__ import annotations

import numpy as np

from roopf.fusion_agent import FusionAgent, SYSTEM


MULTISTEP_SYSTEM = SYSTEM.replace(
    '离线案例包含当时证据、动作和单步收益；不是终局保证。',
    '离线案例包含当时证据和配对反事实收益：在同一状态只执行一次该融合动作，随后回到原策略。'
    'advantages是后续5轮最优值的归一化改善，horizon_advantages包含1、5、10轮结果。'
    '应结合5轮与10轮结果，而非只看当下；这不是终局保证。'
)


class MultiStepFusionAgent(FusionAgent):
    def __init__(self, memory_path, journal, mode='retrieval', budget=None):
        super().__init__(memory_path, journal, mode, budget, memory_version='r04',
            request_interval=5, system_prompt=MULTISTEP_SYSTEM)
        if self.memory.get('primary_horizon_rounds') != 5 or self.memory.get('action_duration_rounds') != 1:
            raise ValueError('R04 requires one-action, five-round rollout evidence')

    def eligible_configs(self, row, cases):
        supported = super().eligible_configs(row, cases)
        result = ['release']
        for name in supported:
            if name == 'release':
                continue
            changed = [case for case in cases if case['actions'][name]['changes_original']]
            if len(changed) >= 2 and np.mean([case['horizon_advantages']['10'][name] for case in changed]) >= 0:
                result.append(name)
        return result

    def choose(self, **kwargs):
        selected = super().choose(**kwargs)
        self.ids = ['release'] * len(self.ids)
        return selected
