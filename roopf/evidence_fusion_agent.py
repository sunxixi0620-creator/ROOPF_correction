"""R09: slow policy control of generation and measured offline/online fusion.

Frozen R08 implementations stay untouched. The inherited generator and feedback
consume only already evaluated data; the numeric selector evaluates no candidates.
"""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import numpy as np
import torch

from roopf.joint_generation_agent import JointGenerationAgent, FEATURES, PROFILES as GENERATORS, digest
from roopf.model import _get_bounds
from roopf.policy_api import DEFAULT_API, request_policy

POLICY_ID = 'r09_factorized_generation_residual_fusion_v1'
GENERATION_NAMES = ['matched38', 'release'] + [name for name in GENERATORS if name not in ('matched38', 'release')]
FUSION_NAMES = ['online', 'calibrated', 'residual_boost']
FUSION_MULTIPLIERS = [0.0, 1.0, 2.0]
PROFILES = {name: dict(GENERATORS[name], generation=name, fusion='calibrated', residual_multiplier=1.0)
            for name in GENERATION_NAMES}
for generation in GENERATION_NAMES:
    if generation != 'release':
        for fusion, multiplier in (('online', 0.0), ('residual_boost', 2.0)):
            PROFILES[generation + '@' + fusion] = dict(GENERATORS[generation], generation=generation,
                fusion=fusion, residual_multiplier=multiplier)
ACTION_NAMES = list(PROFILES)
ACTION_LOOKUP = {(GENERATION_NAMES.index(v['generation']), FUSION_NAMES.index(v['fusion'])): k
                 for k, v in PROFILES.items()}
SYSTEM = """Control two bounded policy factors for each ROOPF trajectory over the next five active rounds. Minimize objective.
Return ONLY JSON, e.g. {"actions":[["0",0,1],["1",0,0]]}, one row per supplied trajectory; ids are strings, factors are integers; no prose.
Generation controls competing Anchor temperature/local step size; a protected Anchor is always retained. Temperature is not a spatial radius.
Fusion 0 uses online acquisition without residual scoring; 1 is matched38 calibrated residual fusion; 2 doubles that supported residual term. Structural protections and gates stay fixed. Original-selection release allows only fusion 1.
The online proxy predicts current candidates. The frozen Residual is relative ranking support, not a calibrated probability of success. Changed-distribution proposals are excluded from Residual scoring.
State and last_fusion describe measured history BEFORE generating this round, not future candidate objectives. Source errors are from selected observations, with selection bias. Missing/null is unknown.
Offline rows are paired 5/10-round advantages versus matched38 under this exact 22-action policy; positive is better. Rows summarize the best and worst actions under distance-weighted neighboring cases, not all actions. All allowed actions remain available. Source ids and distances support auditing, not causal certainty.
Choose BOTH generation and fusion from evidence. For proxy/Residual disagreement use measured ranking reliability and prediction error, not invented objectives. Unknown evidence defaults to generation 0/fusion 1. Return every id exactly once."""


def rounded(value):
    if isinstance(value, dict):
        return {key: rounded(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [rounded(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return round(float(value), 5)
    return value


def request_context_digest(trajectories, execution_context):
    return digest(dict(trajectories=trajectories, execution_context_sha256=digest(execution_context)))


class EvidenceFusionAgent(JointGenerationAgent):
    def __init__(self, memory_path, journal, mode='llm', budget=None, arm='joint', api_config=None):
        self.memory = json.loads(Path(memory_path).read_text()) if memory_path else dict(policy_id=POLICY_ID, cases=[])
        if self.memory.get('policy_id') != POLICY_ID:
            raise ValueError('R09 requires independently collected 22-action fusion cases')
        self.cases = self.memory['cases']
        self.memory_hash = digest(self.memory)
        self.features = np.asarray([case['features'] for case in self.cases], dtype=float).reshape(-1, len(FEATURES))
        self.feature_scale = np.maximum(self.features.std(0), 0.05) if self.cases else np.ones(len(FEATURES))
        self.journal = Path(journal)
        self.journal.mkdir(parents=True, exist_ok=True)
        self.mode, self.arm = mode, arm
        self.budget = budget if budget is not None else dict(used=0, limit=600)
        self.api_config = dict(DEFAULT_API, **(api_config or {}))
        if not 1 <= self.api_config['concurrency'] <= 8:
            raise ValueError('API concurrency outside frozen bounds')
        self.system_prompt, self.active_record = SYSTEM, None
        self.forced = {}
        self.execution_context = None
        self.capture_candidates = False
        self.reset(0)

    def reset(self, batch_size):
        super().reset(batch_size)
        self.api_identity = None
        self.uncertain_api_outcome = False
        self.stats.update(fusion_changed_points=0, nondefault_fusion_rounds=0, api_reused=0)

    def allowed(self):
        if self.arm == 'matched38':
            return ['matched38']
        if self.arm == 'anchor':
            return ['matched38', 'release', 'anchor_sharp', 'anchor_soft']
        if self.arm == 'local':
            return ['matched38', 'release', 'local_fine', 'local_wide']
        return list(PROFILES)

    def compact_payload(self, payload):
        trajectories = []
        for item in payload['trajectories']:
            context = item['current']
            cases = item['offline_cases']
            weights = np.asarray([1 / (1 + c['distance']) for c in cases], dtype=float)
            if len(weights):
                weights /= weights.sum()
            evidence = []
            for action in item['eligible_configs']:
                gains5 = [c['advantages5'][action] for c in cases]
                gains10 = [c['advantages10'][action] for c in cases]
                evidence.append([ACTION_NAMES.index(action), float(np.dot(weights, gains5)),
                    float(np.dot(weights, gains10)), sum(g > 0 for g in gains5), len(gains5)])
            ranked = sorted(evidence, key=lambda row: (row[1], row[0]))
            keep = {0} | {row[0] for row in ranked[:3] + ranked[-3:]}
            previous = context['previous_generation']
            anchor = previous.get('anchor', {})
            sources = []
            for name, row in sorted(context['source_statistics'].items()):
                sources.append([name, row['n'], row['retained'], row['marginal_gain'] / row['n'],
                    None if not row['proxy_error_n'] else row['proxy_error_sum'] / row['proxy_error_n'],
                    row['proxy_error_n'], row['residual_success'] / row['residual_total'], row['residual_total'] - 4])
            trajectories.append(dict(trajectory_id=item['trajectory_id'], state=context['features'],
                samples=[context['feedback_samples'], context['proxy_error_samples']],
                last_fusion=context['previous_generation'].get('fusion'), sources=sources,
                last_generation=dict(near_duplicate_fraction=previous.get('near_duplicate_fraction'),
                    anchor_entropy=anchor.get('entropy'), anchor_distance=anchor.get('normalized_distance'),
                    anchor_repaired=anchor.get('repaired'), local_scale=previous.get('local_scale')),
                reviews=[[r['step'], ACTION_NAMES.index(r['config']), r['observed_progress5']]
                         for r in context['reviews']],
                neighbors=[[c['id'], c['distance']] for c in cases],
                offline=[row for row in evidence if row[0] in keep],
                allowed=[ACTION_NAMES.index(name) for name in item['eligible_configs']]))
        return rounded(dict(policy_id=POLICY_ID, default_profile='matched38', failure_fallback='matched38',
            decision_context_sha256=request_context_digest(payload['trajectories'], self.execution_context),
            execution_context_sha256=digest(self.execution_context), memory_sha256=self.memory_hash,
            state_names=FEATURES, source_columns=['origin', 'n', 'retained', 'mean_marginal_gain', 'mean_proxy_error', 'proxy_error_n', 'smoothed_rank_accuracy', 'rank_n'],
            offline_columns=['action_code', 'weighted_gain5', 'weighted_gain10', 'positive_neighbors5', 'neighbors'],
            generation_menu=[[name, GENERATORS[name]['temperature'], GENERATORS[name]['local_scale']]
                             for name in GENERATION_NAMES],
            fusion_menu=[[name, m] for name, m in zip(FUSION_NAMES, FUSION_MULTIPLIERS)],
            action_codes=[[GENERATION_NAMES.index(v['generation']), FUSION_NAMES.index(v['fusion'])]
                          for v in PROFILES.values()],
            request_kind=payload['request_kind'], action_duration=5, trajectories=trajectories))

    @staticmethod
    def decode(response, payload):
        choice = response['choices'][0]
        if choice.get('finish_reason') != 'stop':
            raise ValueError('output_truncated_or_incomplete')
        content = choice['message'].get('content')
        if not isinstance(content, str):
            raise ValueError('missing_final_content')
        # Never mine reasoning_content, repair truncation or silently invent rows.
        parsed = json.loads(content)
        if not isinstance(parsed, dict) or set(parsed) != {'actions'} or not isinstance(parsed['actions'], list):
            raise ValueError('invalid_action_schema')
        expected = {r['trajectory_id']: r for r in payload['trajectories']}
        decisions = {}
        for row in parsed['actions']:
            if not isinstance(row, list) or len(row) != 3:
                raise ValueError('invalid_action_row')
            identity, generation, fusion = row
            if not isinstance(identity, str) or identity not in expected or identity in decisions:
                raise ValueError('duplicate_or_unknown_trajectory')
            if type(generation) is not int or type(fusion) is not int or (generation, fusion) not in ACTION_LOOKUP:
                raise ValueError('invalid_factor_pair')
            name = ACTION_LOOKUP[generation, fusion]
            if ACTION_NAMES.index(name) not in expected[identity]['allowed']:
                raise ValueError('action_not_eligible')
            decisions[identity] = name
        if set(decisions) != set(expected):
            raise ValueError('missing_trajectory')
        return [decisions[row['trajectory_id']] for row in payload['trajectories']]

    def call(self, payload):
        self.api_identity = None
        try:
            return request_policy(self, self.compact_payload(payload), self.decode)
        except Exception:
            self.stats['failures'] += 1
            return ['matched38'] * len(payload['trajectories'])

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
            self.ids = ['matched38'] * len(self.ids)
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
        residual_term = kwargs['residual_weight'] * scale * calibrated_signal * (2 * reliability)
        multiplier = score.new_tensor([PROFILES[name]['residual_multiplier'] for name in self.ids]).unsqueeze(1)
        acquisition = online - multiplier * residual_term
        distance = torch.norm((points - self.protected_point) / self.width, dim=-1) / math.sqrt(points.size(-1))
        redundancy = (0.01 - distance).clamp_min(0) / 0.01
        acquisition = acquisition + 0.02 * scale * redundancy + 0.02 * scale * source_error
        risk_margin = 0.06 + 0.02 * score.new_tensor(errors).unsqueeze(1)
        default_acquisition = online - residual_term + 0.02 * scale * redundancy + 0.02 * scale * source_error
        default_selected = kwargs['selection_rule'](default_acquisition, default_acquisition.argmin(1, keepdim=True),
            margin=risk_margin, soft_margin=risk_margin, online_only=True)
        proposed = acquisition.argmin(1, keepdim=True)
        selected = kwargs['selection_rule'](acquisition, proposed, margin=risk_margin,
            soft_margin=risk_margin, online_only=True)
        for index, name in enumerate(self.ids):
            if name == 'release':
                selected[index] = original[index]
        same_pool_reference = points.gather(1, default_selected.unsqueeze(-1).expand(-1, -1, points.size(-1)))
        chosen_points = points.gather(1, selected.unsqueeze(-1).expand(-1, -1, points.size(-1)))
        self.stats['fusion_changed_points'] += int(((same_pool_reference != chosen_points).any(-1) & (multiplier != 1)).sum())
        self.stats['nondefault_fusion_rounds'] += int((multiplier != 1).sum())
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
        for index in range(len(self.ids)):
            self.previous_generation[index]['fusion'] = dict(
                proxy_residual_conflict=None if probability is None else bool(online[index].argmin() != calibrated_signal[index].argmax()),
                residual_spread=None if probability is None else float(probability[index].std(unbiased=False)),
                mean_sigma=float(kwargs['sigma'][index].mean() / self.scale[index]),
                selected_proxy_gap=float((online[index, self.last_selected[index]] - online[index].min()) / self.scale[index]),
                residual_multiplier=float(multiplier[index, 0]),
                observed_at_step=self.steps, current_candidate_objectives_known=False)
        record = dict(step=self.steps, configs=list(self.ids), current=rows, selected=selected.tolist(),
            original=original.tolist(), request_event=self.request_event,
            api_trajectories=copy.deepcopy(self.api_trajectories), policy_id=POLICY_ID,
            api_request_sha256=self.api_identity if self.request_event else None,
            execution_context=copy.deepcopy(self.execution_context),
            default_profile='matched38', failure_fallback='matched38',
            fusion_multipliers=multiplier[:, 0].tolist(),
            same_generated_pool_matched_fusion_selected=default_selected.tolist())
        if self.active_record:
            self.active_record(record)
        self.last_record = record
        self.steps += 1
        return selected
