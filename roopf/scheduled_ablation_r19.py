"""Scheduled full-module ablations; R15 S01_L08 cases remain frozen.

The LLM support evidence was learned with the full model and is explicitly
disclosed to the model for ablated candidate pools.
"""
import copy
import torch
from roopf.scheduled_evidence_agent_r18 import POLICY_ID, ACTION_NAMES, digest, request_policy
from roopf.scheduled_evidence_agent_r18 import ScheduledEvidenceAgent
from roopf.no_anchor_ablation import NoAnchorAgent
from roopf.strict_component_ablations import NoOnlineGeneratorsAgent


class ScheduledNoAnchor(ScheduledEvidenceAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.system_prompt += ('\nThe pretrained Anchor is removed. The two reference positions are native '
            'trust-region proposals. Anchor temperature has no effect. Offline evidence was collected '
            'with the Full Anchor and is not re-estimated for this ablation.')

    def prepare_generation(self, *args, **kwargs):
        super().prepare_generation(*args, **kwargs)
        self.stats['requested_anchor_temperature_aliases'] = self.stats.get('requested_anchor_temperature_aliases', 0) + self.stats['anchor_variant_rounds']
        self.stats['anchor_variant_rounds'] = 0

    def origin(self, index, slot, operation):
        if slot == 0:
            return 'native_reference_competitor'
        return super().origin(index, slot, operation)

    make_extra = NoAnchorAgent.make_extra

    def compact_payload(self, payload):
        result = super().compact_payload(payload)
        result['ablation'] = dict(pretrained_anchor_present=False,
            reference_replacement='native trust_region slots1,2; extra slot0; unchanged RNG',
            anchor_temperature_effective=False, offline_memory='frozen Full cases; not relabeled')
        result['state_names'][5] = 'recent_native_reference_gain'
        for item in result['trajectories']:
            history = item['measured_history']
            if history:
                for observation in history['observations']:
                    if 'anchor_gain' in observation:
                        observation['native_reference_gain'] = observation.pop('anchor_gain')
        return result


class ScheduledNoOnline(ScheduledEvidenceAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.system_prompt += ('\nAll six online generators and the extra native proposal are removed. '
            'Only the original Anchor and an optional Anchor temperature variant can be evaluated. '
            'Local-step changes are inactive aliases. Offline evidence comes from Full and was not relabeled.')

    def prepare_generation(self, *args, **kwargs):
        super().prepare_generation(*args, **kwargs)
        self.stats['local_scaled_rounds'] = 0

    origin = NoOnlineGeneratorsAgent.origin
    make_extra = NoOnlineGeneratorsAgent.make_extra
    numerical_options = NoOnlineGeneratorsAgent.numerical_options
    _numeric_choose = NoOnlineGeneratorsAgent._numeric_choose

    def compact_payload(self, payload):
        result = super().compact_payload(payload)
        result['ablation'] = dict(online_generators_present=False, extra_native_proposal_present=False,
            local_scale_effective=False, actual_competing_candidates=[t['current_options']['candidate_count'] for t in payload['trajectories']],
            offline_memory='frozen Full cases; not relabeled')
        return result

    def choose(self, **kwargs):
        options, vectors, _ = self.numerical_options(kwargs)
        self.request_event, self.api_trajectories = False, []
        if self.steps in self.decision_steps and self.arm != 'matched38':
            trajectories = []
            for i, vector in enumerate(vectors):
                e = self.evidence(vector, self.steps)
                self.stats['unsupported_queries'] += int(not e['supported'])
                trajectories.append(dict(trajectory_id=str(i), current=copy.deepcopy(self.generation_context[i]),
                    decision_features=vector, current_options=options[i], evidence=e, eligible_configs=e['allowed']))
            self.ids = ['matched38'] * len(vectors)
            if self.mode == 'llm':
                self.request_event = True
                self.api_trajectories = trajectories
                result = request_policy(self, self.compact_payload(dict(trajectories=trajectories)))
                self.ids = result['config_ids']
                self.api_row_validation = result
                self.stats['failures'] += int(result['status'] == 'failed')
                self.stats['partial_responses'] += int(result['status'] == 'partial')
                self.stats['valid_rows'] += len(result['valid_trajectory_ids'])
                self.stats['invalid_rows'] += len(vectors) - len(result['valid_trajectory_ids'])
            elif self.mode == 'retrieval':
                for i, t in enumerate(trajectories):
                    table = {r[0]: r[1] for r in t['evidence']['rows']}
                    self.ids[i] = max(t['eligible_configs'], key=lambda name: table.get(name, 0.))
            if self.steps in self.forced:
                for i, name in self.forced[self.steps].items():
                    if name not in ACTION_NAMES:
                        raise ValueError('Unknown forced R10 action')
                    self.ids[int(i)] = name
            self.plans = list(self.ids)
            for i,name in enumerate(self.ids):
                self.inflight[i] = dict(step=self.steps, config=name, start_best=float(self.pre_best[i]), scale=float(self.scale[i]))
        callback = self.active_record
        self.active_record = None
        try:
            selected = self._numeric_choose(**kwargs)
        finally:
            self.active_record = callback
        record = self.last_record
        record.update(policy_id=POLICY_ID, api_row_validation=copy.deepcopy(self.api_row_validation),
            applied_generation=list(self.applied_generation), planned_generation=[n.split('@')[0] for n in self.plans],
            current_options=options, decision_features=vectors,
            schedule_signature=digest(self.schedule), config_id=self.schedule['config_id'], temporal_semantics=f'fusion now{self.duration} rounds, generation next{self.duration - 1} rounds')
        if callback:
            callback(record)
        return selected

