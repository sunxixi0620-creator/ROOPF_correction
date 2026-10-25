"""Frozen R10 with its pretrained Anchor replaced by the native trust-region operator.

The replacement fills the same two reference slots; it is not a new optimizer.
It runs in an RNG fork because the original evaluation-mode Anchor uses no RNG.
The offline memory is deliberately NOT retrained for this module ablation.
"""
import torch

from roopf.model import ROOPFOptimizer
from roopf.current_evidence_agent import CurrentEvidenceAgent, SYSTEM


class NoAnchorOptimizer(ROOPFOptimizer):
    def _candidate_pool(self, x, fitness, problem, state, state_raw, success_logits):
        self._no_anchor_state = state_raw
        return super()._candidate_pool(x, fitness, problem, state, state_raw, success_logits)

    @torch.no_grad()
    def _baseline_candidates(self, x, fitness, problem, rest):
        assert self.baseline_model is None, 'Pretrained Anchor must be absent'
        before = torch.get_rng_state().clone()
        control = self._generation_control
        with torch.random.fork_rng(devices=[]):
            self._generation_control = None
            try:
                # Slot0 is reserved for the existing extra native proposal.
                candidates = problem.repaire(self._trust_region(x, problem, self._no_anchor_state)[:, 1:1 + rest])
            finally:
                self._generation_control = control
        assert torch.equal(before, torch.get_rng_state()), 'Replacement consumed solver RNG'
        self.no_anchor_references.append(candidates.detach().cpu().clone())
        if control is not None:
            control.make_extra(self, x, fitness, problem, candidates)
        return candidates


def install(model):
    assert type(model) is ROOPFOptimizer
    model.__class__ = NoAnchorOptimizer
    model.baseline_model = None
    model.no_anchor_references = []
    return model


class NoAnchorAgent(CurrentEvidenceAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.system_prompt = SYSTEM.replace('Protected Anchor', 'Protected native reference').replace(
            'Temperature changes weight concentration, not search radius.',
            'The pretrained Anchor is REMOVED. Both original Anchor slots now use native trust-region proposals. '
            'All Anchor temperature changes are inactive; their action names remain as aliases. '
            'Offline returns were collected with the full Anchor model and are not re-estimated returns for this ablation.')

    def prepare_generation(self, *args, **kwargs):
        super().prepare_generation(*args, **kwargs)
        self.stats['requested_anchor_temperature_aliases'] = self.stats.get('requested_anchor_temperature_aliases', 0) + self.stats['anchor_variant_rounds']
        self.stats['anchor_variant_rounds'] = 0

    def origin(self, index, slot, operation):
        if slot == 0:
            return 'native_reference_competitor'
        return super().origin(index, slot, operation)

    def make_extra(self, model, population, fitness, problem, baseline):
        assert model.baseline_model is None
        self.protected_point = baseline[:, :1].detach().clone()
        self.original_anchor = baseline[:, 1:2].detach().clone()  # historical field: now native
        with torch.random.fork_rng(devices=[]):
            control = model._generation_control
            model._generation_control = None
            try:
                self.extra = problem.repaire(model._trust_region(population, problem, self.state_raw)[:, :1])
            finally:
                model._generation_control = control
        self.extra_ops = torch.full((population.size(0), 1), model.op_names.index('trust_region'), dtype=torch.long)
        self.extra_origin = ['native_extra'] * population.size(0)
        self.anchor_diagnostics = [dict(enabled=False, replacement='native_trust_region',
            temperature_effective=False, requested_temperature=float(self.temperature[i]),
            entropy=None, repaired=None, normalized_distance=None) for i in range(population.size(0))]

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
