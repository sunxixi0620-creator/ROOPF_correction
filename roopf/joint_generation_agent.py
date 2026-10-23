from __future__ import annotations

import copy
import fcntl
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from roopf.fusion_agent import FusionAgent
from roopf.model import _get_bounds


POLICY_ID = 'r07_anchor_temperature_local_scale_joint_v1'
PROFILES = {
    'release': dict(temperature=1.0, local_scale=1.0, select='original'),
    'matched38': dict(temperature=1.0, local_scale=1.0, select='coordinated'),
    'anchor_sharp': dict(temperature=0.8, local_scale=1.0, select='coordinated'),
    'anchor_soft': dict(temperature=1.25, local_scale=1.0, select='coordinated'),
    'local_fine': dict(temperature=1.0, local_scale=0.5, select='coordinated'),
    'local_wide': dict(temperature=1.0, local_scale=1.5, select='coordinated'),
    'joint_sharp_wide': dict(temperature=0.8, local_scale=1.5, select='coordinated'),
    'joint_soft_fine': dict(temperature=1.25, local_scale=0.5, select='coordinated'),
}
FEATURES = ('remaining', 'stagnation', 'elite_spread', 'population_spread',
            'elite_population_ratio', 'recent_anchor_gain', 'recent_extra_gain', 'recent_proxy_error')
SYSTEM = '''Choose one joint generation configuration for each supplied trajectory. Minimize the objective under a fixed evaluation budget.
Return only {"config_ids":{"trajectory_id":"eligible_config"}}. No explanation or other fields.
The same Agent controls the competing Anchor proposal and local operator generation for five active rounds. The protected Anchor and original competing Anchor remain available.
Temperature controls neural feature-weight concentration, not guaranteed spatial locality. Local scale changes the final three trust-region and coordinate candidates, retaining their centers and random directions.
release uses original generation and selection. matched38 is an equal-candidate-count native-generator control. Other profiles use the same coordinated numerical selector.
Offline advantages are paired five/ten-round returns of this exact policy, not guarantees. Past online progress is selected-state evidence, not a causal attribution. Missing data is unknown, not poor performance.
Prefer configurations supported by offline cases and current measured reliability. Distance/diversity alone does not imply utility. Do not invent current unevaluated objectives or confidence probabilities.'''


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


class JointGenerationAgent:
    expand_full_pool = True

    def __init__(self, memory_path, journal, mode='llm', budget=None, arm='joint'):
        self.memory = json.loads(Path(memory_path).read_text()) if memory_path else dict(
            policy_id=POLICY_ID, cases=[], allowed_configs=list(PROFILES))
        if self.memory.get('policy_id') != POLICY_ID:
            raise ValueError('Joint generation requires current-policy cases')
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
        self.reset(0)

    def reset(self, batch_size):
        self.steps = 0
        self.ids = ['release'] * batch_size
        self.history = [[] for _ in range(batch_size)]
        self.reviews = [[] for _ in range(batch_size)]
        self.inflight = [None] * batch_size
        self.pending = None
        self.stats = dict(decisions=0, changed=0, changed_points=0, requests=0, failures=0,
            nonrelease=0, opportunities=0, no_supported_action=0,
            anchor_variant_rounds=0, local_scaled_rounds=0, selected_adjusted_candidates=0)
        self.extra = None
        self.resource_exhausted = False
        self.last_selected = []
        self.previous_generation = [{} for _ in range(batch_size)]

    def source_statistics(self, index):
        result = {}
        for item in self.history[index][-16:]:
            source = result.setdefault(item['origin'], dict(n=0, retained=0, marginal_gain=0.0,
                proxy_error_sum=0.0, proxy_error_n=0, residual_success=2.0, residual_total=4))
            source['n'] += 1
            source['retained'] += int(item['retained'])
            source['marginal_gain'] += item['marginal_gain']
            if item['proxy_error'] is not None:
                source['proxy_error_sum'] += item['proxy_error']
                source['proxy_error_n'] += 1
            if item['residual_rank_correct'] is not None:
                source['residual_success'] += item['residual_rank_correct']
                source['residual_total'] += 1
        return result

    def call(self, payload):
        with (self.journal / 'API.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            self.budget['used'] = len(list(self.journal.glob('*.json')))
            try:
                return FusionAgent.call(self, payload)
            except RuntimeError as error:
                if str(error) == '本轮API次数上限':
                    self.resource_exhausted = True
                raise

    def allowed(self):
        if self.arm == 'matched38':
            return ['matched38']
        if self.arm == 'anchor':
            return ['release', 'matched38', 'anchor_sharp', 'anchor_soft']
        if self.arm == 'local':
            return ['release', 'matched38', 'local_fine', 'local_wide']
        return list(PROFILES)

    def origin(self, index, slot, operation):
        if slot == 37:
            if self.extra_origin[index] == 'anchor_variant':
                return 'anchor_variant_T' + str(float(self.temperature[index]))
            return 'native_extra'
        if slot == 0:
            return 'original_anchor'
        name = 'operator_' + str(int(operation))
        if slot in (22, 23, 24, 28, 29, 30) and float(self.local_scale[index]) != 1:
            name += '_scale' + str(float(self.local_scale[index]))
        return name

    def retrieve(self, features):
        if not self.cases:
            return []
        distances = np.mean(((self.features - features) / self.feature_scale) ** 2, axis=1)
        result, seen = [], set()
        for position in np.argsort(distances, kind='stable'):
            case = self.cases[int(position)]
            if case['function'] in seen:
                continue
            result.append(dict(id=case['id'], function=case['function'], distance=float(distances[position]),
                advantages5=case['advantages']['5'], advantages10=case['advantages']['10']))
            seen.add(case['function'])
            if len(result) == 3:
                break
        return result

    def prepare_generation(self, model, population, fitness, archive_y, state_raw, problem, remaining):
        lower, upper = _get_bounds(problem, population)
        self.width = (upper - lower).clamp_min(1e-8)
        self.state_raw = state_raw
        self.pre_best = fitness[:, 0].detach().clone()
        self.scale = archive_y.std(1, unbiased=False).clamp_min(1e-8)
        self.worst = fitness[:, -1].detach().clone()
        self.pre_fitness = fitness.detach().clone()
        normalized = population / self.width
        elite_count = max(2, population.size(1) // 5)
        elite = normalized[:, :elite_count].std(1, unbiased=False).mean(1)
        spread = normalized.std(1, unbiased=False).mean(1)
        self.generation_context = []
        for index in range(population.size(0)):
            recent = self.history[index][-16:]
            anchor_gain = sum(row['anchor_gain'] for row in recent) / max(1, len(recent))
            extra_gain = sum(row['marginal_gain'] for row in recent) / max(1, len(recent))
            errors = [row['proxy_error'] for row in recent if row['proxy_error'] is not None]
            vector = [float(remaining), float(state_raw[index, 8]), float(elite[index]), float(spread[index]),
                float(elite[index] / spread[index].clamp_min(1e-8)), anchor_gain, extra_gain,
                sum(errors) / max(1, len(errors))]
            self.generation_context.append(dict(features=vector, named=dict(zip(FEATURES, vector)),
                feedback_samples=len(recent), proxy_error_samples=len(errors),
                source_statistics=self.source_statistics(index),
                previous_generation=copy.deepcopy(self.previous_generation[index]),
                previous_actions=copy.deepcopy(recent[-3:]), reviews=copy.deepcopy(self.reviews[index][-2:]),
                no_current_objective=True, nfe_before=int(round(300 * (1 - remaining)))))
        self.request_event = False
        self.api_trajectories = []
        if self.steps not in (5, 6, 7, 8, 9, 15, 16, 17, 18, 19):
            self.ids = ['release'] * len(self.ids)
        if self.arm == 'matched38' and self.steps in (5, 15):
            self.ids = ['matched38'] * len(self.ids)
        if self.steps in (5, 15) and self.arm != 'matched38':
            self.ids = ['release'] * len(self.ids)
            trajectories = []
            for index, row in enumerate(self.generation_context):
                cases = self.retrieve(row['features'])
                trajectories.append(dict(trajectory_id=str(index), current=row,
                    offline_cases=cases, eligible_configs=self.allowed()))
            if self.mode == 'llm':
                self.request_event = True
                self.api_trajectories = copy.deepcopy(trajectories)
                try:
                    self.ids = self.call(dict(policy_id=POLICY_ID, profiles=PROFILES,
                        request_kind='generation' if self.steps == 5 else 'review_generation',
                        action_duration=5, trajectories=trajectories))
                except Exception:
                    self.stats['failures'] += 1
            elif self.mode == 'retrieval':
                for index, item in enumerate(trajectories):
                    estimates = {name: sum(case['advantages5'][name] / (1 + case['distance'])
                        for case in item['offline_cases']) for name in item['eligible_configs']}
                    self.ids[index] = max(estimates, key=estimates.get)
        if self.steps in self.forced:
            for index, name in self.forced[self.steps].items():
                if name not in PROFILES:
                    raise ValueError('Unknown forced profile')
                self.ids[int(index)] = name
        for index, name in enumerate(self.ids):
            if self.steps in (5, 15) and name != 'release':
                self.inflight[index] = dict(step=self.steps, config=name,
                    start_best=float(self.pre_best[index]), scale=float(self.scale[index]))
        self.temperature = population.new_tensor([PROFILES[name]['temperature'] for name in self.ids])
        self.local_scale = population.new_tensor([PROFILES[name]['local_scale'] for name in self.ids])
        self.stats['anchor_variant_rounds'] += int((self.temperature != 1).sum())
        self.stats['local_scaled_rounds'] += int((self.local_scale != 1).sum())

    def scale_displacement(self, displacement, operator):
        if operator not in ('trust_region', 'coordinate_pattern'):
            return displacement
        multiplier = torch.ones_like(displacement[:, :, :1])
        multiplier[:, 3:] = self.local_scale.reshape(-1, 1, 1)
        return displacement * multiplier

    def make_extra(self, model, population, fitness, problem, baseline):
        self.protected_point = baseline[:, :1].detach().clone()
        self.original_anchor = baseline[:, 1:2].detach().clone()
        with torch.random.fork_rng(devices=[]):
            controller = model._generation_control
            model._generation_control = None
            try:
                native = problem.repaire(model._trust_region(population, problem, self.state_raw)[:, :1])
            finally:
                model._generation_control = controller
            variant, details = model.baseline_model.generator.competing_proposal(
                population, problem, fitness, self.temperature)
        changed = self.temperature != 1
        self.extra = torch.where(changed.reshape(-1, 1, 1), variant, native)
        self.extra_ops = torch.where(changed, torch.full_like(changed, -2, dtype=torch.long),
            torch.full_like(changed, model.op_names.index('trust_region'), dtype=torch.long)).unsqueeze(1)
        self.extra_origin = ['anchor_variant' if flag else 'native_extra' for flag in changed.tolist()]
        self.anchor_diagnostics = [dict(temperature=float(self.temperature[index]),
            entropy=float(details['entropy'][index]), repaired=bool(details['repaired'][index]),
            normalized_distance=float(torch.norm((variant[index] - baseline[index, 1:2]) / self.width) /
                math.sqrt(population.size(-1)))) for index in range(population.size(0))]

    def extend_candidates(self, points, operations):
        if self.extra is None:
            raise RuntimeError('Generation did not produce extra candidate')
        expanded = torch.cat([points, self.extra], 1)
        normalized = expanded / self.width
        pair_distance = torch.cdist(normalized, normalized) / math.sqrt(points.size(-1))
        diagonal = torch.eye(expanded.size(1), device=points.device, dtype=torch.bool).unsqueeze(0)
        pair_distance = pair_distance.masked_fill(diagonal, float('inf'))
        self.previous_generation = [dict(near_duplicate_fraction=float((pair_distance[index].amin(1) < 1e-6).float().mean()),
            extra_origin=self.extra_origin[index], anchor=self.anchor_diagnostics[index],
            local_scale=float(self.local_scale[index])) for index in range(points.size(0))]
        if not torch.isfinite(expanded).all():
            raise RuntimeError('Non-finite generated candidate; do not evaluate or silently repair')
        return expanded, torch.cat([operations, self.extra_ops], 1)

    def choose(self, **kwargs):
        points, original = kwargs['candidates'], kwargs['original_idx']
        if points.size(1) != 38:
            raise RuntimeError('Joint controller requires38 competing candidates')
        score, probability = kwargs['score'], kwargs['probability']
        scale = self.scale.unsqueeze(1)
        signal = torch.zeros_like(score) if probability is None else (
            (probability - probability.mean(1, keepdim=True)) /
            probability.std(1, keepdim=True, unbiased=False).clamp_min(1e-6)).clamp(-2.5, 2.5)
        online = score + kwargs['residual_weight'] * scale * signal
        supported = torch.ones_like(score)
        errors = []
        reliability = torch.full_like(score, 0.5)
        source_error = torch.zeros_like(score)
        for index, recent in enumerate(self.history):
            window = recent[-16:]
            error = [row['proxy_error'] for row in window if row['proxy_error'] is not None]
            errors.append(min(2.0, sum(error) / max(1, len(error))))
            sources = self.source_statistics(index)
            for slot in range(points.size(1)):
                source_name = self.origin(index, slot, kwargs['candidate_ops'][index, slot])
                evidence = sources.get(source_name)
                if evidence:
                    reliability[index, slot] = evidence['residual_success'] / evidence['residual_total']
                    source_error[index, slot] = min(2.0, evidence['proxy_error_sum'] / max(1, evidence['proxy_error_n']))
            if self.extra_origin[index] == 'anchor_variant':
                supported[index, -1] = 0
            if float(self.local_scale[index]) != 1:
                supported[index, 1 + 3 * 6 + 3:1 + 4 * 6] = 0
                supported[index, 1 + 4 * 6 + 3:1 + 5 * 6] = 0
        calibrated_signal = torch.zeros_like(signal)
        if probability is not None:
            count = supported.sum(1, keepdim=True).clamp_min(1)
            mean = (probability * supported).sum(1, keepdim=True) / count
            deviation = (((probability - mean) ** 2 * supported).sum(1, keepdim=True) / count).sqrt().clamp_min(1e-6)
            calibrated_signal = ((probability - mean) / deviation).clamp(-2.5, 2.5) * supported
        acquisition = online - kwargs['residual_weight'] * scale * calibrated_signal * (2 * reliability)
        distance = torch.norm((points - self.protected_point) / self.width, dim=-1) / math.sqrt(points.size(-1))
        redundancy = (0.01 - distance).clamp_min(0) / 0.01
        acquisition = acquisition + 0.02 * scale * redundancy + 0.02 * scale * source_error
        risk_margin = 0.06 + 0.02 * score.new_tensor(errors).unsqueeze(1)
        proposed = acquisition.argmin(1, keepdim=True)
        selected = kwargs['selection_rule'](acquisition, proposed, margin=risk_margin,
            soft_margin=risk_margin, online_only=True)
        for index, name in enumerate(self.ids):
            if name == 'release':
                selected[index] = original[index]
        self.pending = list(self.ids)
        self.last_selected = selected[:, 0].tolist()
        for index, slot in enumerate(self.last_selected):
            adjusted = (slot == 37 and self.extra_origin[index] == 'anchor_variant') or (
                slot in (22, 23, 24, 28, 29, 30) and float(self.local_scale[index]) != 1)
            self.stats['selected_adjusted_candidates'] += int(adjusted)
        before = points.gather(1, original.unsqueeze(-1).expand(-1, -1, points.size(-1)))
        after = points.gather(1, selected.unsqueeze(-1).expand(-1, -1, points.size(-1)))
        self.stats['decisions'] += len(self.ids)
        self.stats['changed'] += int((selected != original).sum())
        self.stats['changed_points'] += int((before != after).any(-1).sum())
        self.stats['nonrelease'] += sum(name != 'release' for name in self.ids)
        self.stats['opportunities'] += len(self.ids)
        rows = []
        for index in range(len(self.ids)):
            row = dict(generation=copy.deepcopy(self.generation_context[index]), candidate_count=38,
                candidates=points[index].tolist(), original_index=int(original[index, 0]),
                selected_index=int(selected[index, 0]), mu=kwargs['mu'][index].tolist(),
                sigma=kwargs['sigma'][index].tolist(), acquisition=acquisition[index].tolist(),
                residual_support=None if probability is None else probability[index].tolist(),
                residual_eligible=supported[index].tolist(), candidate_ops=kwargs['candidate_ops'][index].tolist(),
                extra_origin=self.extra_origin[index], anchor=self.anchor_diagnostics[index],
                protected_point=self.protected_point[index, 0].tolist(),
                predicted_marginal_gain=((torch.minimum(self.pre_best[index], self.protected_mu[index, 0]) -
                    kwargs['mu'][index]).clamp_min(0) / self.scale[index]).tolist(),
                protected_slot_unchanged=True, numerical_selection='empirical-risk margin plus redundancy penalty; not calibrated EI')
            row['context_sha256'] = digest(row)
            rows.append(row)
        record = dict(step=self.steps, configs=list(self.ids), current=rows, selected=selected.tolist(),
            original=original.tolist(), request_event=self.request_event,
            api_trajectories=copy.deepcopy(self.api_trajectories), policy_id=POLICY_ID)
        if self.active_record:
            self.active_record(record)
        self.last_record = record
        self.steps += 1
        return selected

    def observe(self, observed, mu, probability, previous_best, selected_ops, archive_y):
        if self.pending is None:
            return
        for index, name in enumerate(self.pending):
            anchor, competitor = map(float, observed[index, [0, -1]])
            before, scale = float(previous_best[index]), float(self.scale[index])
            marginal = max(0.0, min(before, anchor) - competitor) / scale
            joint_fit = torch.cat([self.pre_fitness[index], observed[index]])
            retained = bool((torch.sort(joint_fit).indices[:self.pre_fitness.size(1)] == joint_fit.numel() - 1).any())
            origin = self.origin(index, self.last_selected[index], selected_ops[index, -1])
            proxy_error = None if mu is None or not math.isfinite(float(mu[index, -1])) else abs(competitor - float(mu[index, -1])) / scale
            rank_correct = None
            selected_support = self.last_record['current'][index]['residual_eligible'][self.last_selected[index]]
            if probability is not None and selected_support and competitor != anchor:
                prediction = float(probability[index, -1] - probability[index, 0])
                if math.isfinite(prediction):
                    rank_correct = 0.5 if prediction == 0 else float((prediction > 0) == (competitor < anchor))
            item = dict(step=self.steps - 1, config=name, origin=origin,
                anchor_gain=max(0.0, before - anchor) / scale, marginal_gain=marginal,
                target_minus_previous_best=(competitor - before) / scale,
                retained=retained, proxy_error=proxy_error, residual_rank_correct=rank_correct,
                selection_biased=True, causal_advantage=False)
            self.history[index].append(item)
            self.history[index] = self.history[index][-32:]
            action = self.inflight[index]
            if action and self.steps - action['step'] >= 5:
                progress = max(0.0, action['start_best'] - min(before, anchor, competitor)) / action['scale']
                self.reviews[index].append(dict(**action, observed_progress5=progress,
                    includes_anchor_and_other_actions=True, causal_advantage=False))
                self.inflight[index] = None
        self.last_record['observed_feedback'] = copy.deepcopy([items[-1] for items in self.history])
        self.last_record['completed_reviews_after'] = copy.deepcopy(self.reviews)
        self.pending = None
