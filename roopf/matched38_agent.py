"""R08: the complete joint LLM controller on a matched38 default policy.

The historical R07 implementation is retained for exact baseline reproduction.
This migration changes the decision default, failure fallback and offline labels;
it does not remove any Anchor/local action or change scoring coefficients.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from roopf.joint_generation_agent import JointGenerationAgent, FEATURES, PROFILES, digest

POLICY_ID = 'r08_matched38_default_full_joint_agent_v1'
SYSTEM = '''Choose one joint generation configuration for each supplied trajectory. Minimize the objective under a fixed evaluation budget.
Return only {"config_ids":{"trajectory_id":"eligible_config"}}. No explanation or other fields.
The default policy is matched38: the original protected Anchor, the original competing Anchor, 36 original-family candidates, and one independent native trust-region proposal. It uses the coordinated numerical selector during the existing five-round action windows.
The same Agent controls the competing Anchor temperature and the local trust-region/coordinate step scale for five active rounds. The protected Anchor and original competing Anchor remain available.
matched38 preserves this default generation and coordinated selection. release deliberately chooses the original selection. All original sharp/soft Anchor, fine/wide local and joint configurations remain available. Outside action windows the original release schedule is retained.
Temperature changes neural feature-weight concentration, not guaranteed spatial locality. Local scale changes only the final three trust-region and coordinate candidates, with the original centers and repair.
Offline advantages are paired five/ten-round returns relative to matched38 under this exact policy, not returns relative to release. Historical online feedback and five-round reviews are selection-biased, not causal advantage estimates. Missing evidence is unknown.
Prefer measured offline evidence and current observed reliability. When no change is supported, choose matched38. Never invent current unevaluated objective values or confidence probabilities.'''


class Matched38Agent(JointGenerationAgent):
    def __init__(self, memory_path, journal, mode='llm', budget=None, arm='joint'):
        self.memory = json.loads(Path(memory_path).read_text()) if memory_path else dict(
            policy_id=POLICY_ID, cases=[], allowed_configs=list(PROFILES))
        if self.memory.get('policy_id') != POLICY_ID:
            raise ValueError('R08 requires cases relabeled against the matched38 default')
        self.cases = self.memory['cases']
        self.memory_hash = digest(self.memory)
        self.features = np.asarray([case['features'] for case in self.cases], dtype=float).reshape(-1, len(FEATURES))
        self.feature_scale = np.maximum(self.features.std(0), 0.05) if self.cases else np.ones(len(FEATURES))
        self.journal = Path(journal)
        self.journal.mkdir(parents=True, exist_ok=True)
        self.mode, self.arm = mode, arm
        self.budget = budget if budget is not None else dict(used=0, limit=600)
        self.system_prompt, self.active_record = SYSTEM, None
        self.forced = {}
        self.capture_candidates = False
        self.resource_exhausted = False
        self.reset(0)

    def allowed(self):
        names = super().allowed()
        # Preserve the action set, but make an evidence tie select the new default.
        return ['matched38'] + [name for name in names if name != 'matched38']

    def call(self, payload):
        payload = dict(payload, policy_id=POLICY_ID, default_profile='matched38',
                       offline_advantage_reference='matched38', failure_fallback='matched38')
        try:
            return super().call(payload)
        except Exception:
            # The parent call has already journaled the attempt and enforced the
            # global cap. A failed request is not retried or counted as valid.
            self.stats['failures'] += 1
            return ['matched38'] * len(payload['trajectories'])

    def prepare_generation(self, model, population, fitness, archive_y, state_raw, problem, remaining):
        forced = self.forced
        if self.mode == 'collect':
            self.forced = {step: {index: 'matched38' for index in range(population.size(0))}
                           for step in (5, 15)}
            for step, actions in forced.items():
                self.forced.setdefault(step, {}).update({int(index): name for index, name in actions.items()})
        try:
            return super().prepare_generation(model, population, fitness, archive_y, state_raw, problem, remaining)
        finally:
            self.forced = forced

    def choose(self, **kwargs):
        callback = self.active_record
        records = []
        self.active_record = records.append
        try:
            selected = super().choose(**kwargs)
        finally:
            self.active_record = callback
        record = records[0]
        record.update(policy_id=POLICY_ID, default_profile='matched38', failure_fallback='matched38')
        if callback:
            callback(record)
        return selected
