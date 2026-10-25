"""R10: current-pool fusion with delayed generation and supported case evidence.

The unchanged R09 numerical selector remains the matched38 foundation. At steps
5/15 fusion starts immediately; generation starts next round through steps9/19.
Every offline branch uses these same temporal semantics.
"""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import numpy as np
import torch

from roopf.evidence_fusion_agent import EvidenceFusionAgent, ACTION_NAMES, PROFILES, GENERATION_NAMES, digest, request_context_digest
from roopf.joint_generation_agent import FEATURES as BASE_FEATURES
from roopf.policy_api_r10 import DEFAULT_API, request_policy

POLICY_ID = 'r10_current_pool_fusion_delayed_generation_v1'
FEATURES = list(BASE_FEATURES) + ['current_proxy_residual_conflict', 'current_fusion_choice_disagreement',
    'current_mean_sigma', 'current_residual_spread', 'current_online_gap', 'current_proxy_residual_rank_correlation',
    'selected_residual_rank_accuracy', 'selected_rank_sample_fraction']
HISTORY_FEATURES = (5, 6, 7, 14, 15)
SYSTEM = '''Choose one allowed action string per trajectory. Return only {"actions":{"0":"matched38","1":"local_wide@online"}} with every supplied id exactly once. No prose or reasoning.
Minimize objective. matched38 is the default and failure fallback. All action names have one canonical meaning, with no numeric action ids.
The current pool has already been generated: fusion is applied NOW and for the next four rounds; the chosen generation applies only to the NEXT four rounds. Current-pool comparisons do not preview future generation or prove five-round policy equivalence.
Actions without @ use calibrated residual fusion, @online uses multiplier0, @residual_boost uses multiplier2. release uses original selection. Protected Anchor, evaluation budget and structural gates remain fixed. Temperature changes weight concentration, not search radius.
Current options show actual numerical choices under each fusion strength, using model predictions only; no unevaluated objective truth is available. Residual scores are relative ranking support, not success probabilities. sigma and score gaps are not calibrated confidence bounds.
All22 actions have offline evidence rows when offline input is enabled. Gains are paired five/ten-round returns versus matched38; dispersion is neighboring-case variation, not statistical confidence. Neighbor coverage may be insufficient. Only allowed actions can execute. Do not choose a change merely to increase change counts.
When offline input is absent, use current comparisons and available measured history cautiously. Online selected-observation feedback is biased; observed progress is not causal action advantage. Missing/null means unknown. Prefer matched38 when evidence is absent or contradictory.'''


def compact(value):
    if isinstance(value, dict):
        return {k: compact(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [compact(v) for v in value]
    if isinstance(value, (float, np.floating)):
        if not math.isfinite(value):
            raise ValueError('Non-finite policy input')
        return float(format(float(value), '.7g'))
    return value


def average_ranks(value):
    _, inverse, counts = torch.unique(value, sorted=True, return_inverse=True, return_counts=True)
    middle = counts.cumsum(0).to(value.dtype) - (counts.to(value.dtype) + 1) / 2
    return middle[inverse]


class CurrentEvidenceAgent(EvidenceFusionAgent):
    def __init__(self, memory_path, journal, mode='llm', budget=None, arm='joint', api_config=None):
        super().__init__(None, journal, mode, budget, arm, dict(DEFAULT_API, **(api_config or {})))
        self.memory = json.loads(Path(memory_path).read_text()) if memory_path else dict(policy_id=POLICY_ID, cases=[], coverage_radii={})
        if self.memory['policy_id'] != POLICY_ID:
            raise ValueError('R10 requires independently collected delayed-generation cases')
        self.cases = self.memory['cases']
        self.memory_hash = digest(self.memory)
        self.features = np.asarray([r['features'] for r in self.cases], dtype=float).reshape(-1, len(FEATURES))
        self.feature_scale = np.maximum(self.features.std(0), .05) if self.cases else np.ones(len(FEATURES))
        self.system_prompt = SYSTEM

    def reset(self, batch_size):
        super().reset(batch_size)
        self.plans = ['matched38'] * batch_size
        self.applied_generation = ['release'] * batch_size
        self.stats.update(valid_rows=0, invalid_rows=0, partial_responses=0, unsupported_queries=0)
        self.api_row_validation = None

    def allowed(self):
        return ['matched38'] if self.arm == 'matched38' else list(ACTION_NAMES)

    def retrieve(self, features):
        # Frozen preparation asks for its old8-dimensional retrieval, but R10
        # retrieval occurs only after current predictions add the other features.
        return []

    def prepare_generation(self, *args, **kwargs):
        # Use frozen state/geometry preparation, but never call its pre-pool LLM.
        mode, arm, forced = self.mode, self.arm, self.forced
        self.ids = list(self.plans)
        self.mode, self.arm, self.forced = 'collect', 'matched38', {}
        try:
            super().prepare_generation(*args, **kwargs)
        finally:
            self.mode, self.arm, self.forced = mode, arm, forced
        self.applied_generation = [name.split('@')[0] for name in self.ids]
        self.api_identity, self.api_row_validation = None, None

    def numerical_options(self, kwargs):
        """Pure previews of the three fusion strengths; no RNG or objectives."""
        points, score, p = kwargs['candidates'], kwargs['score'], kwargs['probability']
        scale = self.scale.unsqueeze(1)
        signal = torch.zeros_like(score) if p is None else ((p - p.mean(1, keepdim=True)) / p.std(1, keepdim=True, unbiased=False).clamp_min(1e-6)).clamp(-2.5, 2.5)
        online = score + kwargs['residual_weight'] * scale * signal
        supported, reliability, source_error = torch.ones_like(score), torch.full_like(score, .5), torch.zeros_like(score)
        errors = []
        for i, recent in enumerate(self.history):
            error = [r['proxy_error'] for r in recent[-16:] if r['proxy_error'] is not None]
            errors.append(min(2., sum(error) / max(1, len(error))))
            sources = self.source_statistics(i)
            for slot in range(38):
                row = sources.get(self.origin(i, slot, kwargs['candidate_ops'][i, slot]))
                if row:
                    reliability[i, slot] = row['residual_success'] / row['residual_total']
                    source_error[i, slot] = min(2., row['proxy_error_sum'] / max(1, row['proxy_error_n']))
            if self.extra_origin[i] == 'anchor_variant':
                supported[i, -1] = 0
            if float(self.local_scale[i]) != 1:
                supported[i, [22, 23, 24, 28, 29, 30]] = 0
        calibrated = torch.zeros_like(score)
        if p is not None:
            count = supported.sum(1, keepdim=True).clamp_min(1)
            mean = (p * supported).sum(1, keepdim=True) / count
            dev = (((p - mean) ** 2 * supported).sum(1, keepdim=True) / count).sqrt().clamp_min(1e-6)
            calibrated = ((p - mean) / dev).clamp(-2.5, 2.5) * supported
        residual = kwargs['residual_weight'] * scale * calibrated * (2 * reliability)
        distance = torch.norm((points - self.protected_point) / self.width, dim=-1) / math.sqrt(points.size(-1))
        redundancy = (.01 - distance).clamp_min(0) / .01
        margin = .06 + .02 * score.new_tensor(errors).unsqueeze(1)
        indices, acquisitions, raw = [], [], []
        for multiplier in (0., 1., 2.):
            acq = online - multiplier * residual
            acq = acq + .02 * scale * redundancy + .02 * scale * source_error
            proposed = acq.argmin(1, keepdim=True)
            indices.append(kwargs['selection_rule'](acq, proposed, margin=margin, soft_margin=margin, online_only=True))
            acquisitions.append(acq)
            raw.append(proposed)
        summaries, features = [], []
        for i in range(points.size(0)):
            reference = points[i, int(indices[1][i, 0])]
            rows = []
            for k in range(3):
                slot = int(indices[k][i, 0])
                rows.append([k, slot, int(raw[k][i, 0]),
                    float((kwargs['mu'][i, slot] - self.pre_best[i]) / self.scale[i]),
                    float(kwargs['sigma'][i, slot] / self.scale[i]),
                    float((acquisitions[k][i, slot] - acquisitions[k][i, 0]) / self.scale[i]),
                    bool((points[i, slot] != reference).any())])
            q = sorted(online[i].tolist())
            correlation, conflict, spread = 0., 0., 0.
            if p is not None:
                conflict = float(online[i].argmin() != calibrated[i].argmax())
                spread = float(p[i].std(unbiased=False))
                a, b = average_ranks(-online[i]), average_ranks(p[i])
                denominator = a.std(unbiased=False)*b.std(unbiased=False)
                correlation = 0. if float(denominator) == 0 else float(((a-a.mean())*(b-b.mean())).mean() / denominator)
            ranked = [r['residual_rank_correct'] for r in self.history[i][-16:] if r['residual_rank_correct'] is not None]
            vector = list(self.generation_context[i]['features']) + [conflict,
                sum(r[-1] for r in rows) / 3, float(kwargs['sigma'][i].mean() / self.scale[i]), spread,
                (q[1] - q[0]) / float(self.scale[i]), correlation,
                sum(ranked) / len(ranked) if ranked else .5, len(ranked) / 16]
            features.append(vector)
            summaries.append(dict(applied_generation=self.applied_generation[i], fusion=rows,
                original_selection=int(kwargs['original_idx'][i, 0]), current_objectives_known=False))
        return summaries, features, indices

    def evidence(self, vector, step):
        if self.arm == 'no_offline' or not self.cases:
            return dict(enabled=self.arm != 'no_offline', supported=False, neighbors=[], rows=[], allowed=list(ACTION_NAMES) if self.arm == 'no_offline' else ['matched38'])
        mask = np.ones(len(FEATURES), dtype=bool)
        if self.arm == 'no_online':
            mask[list(HISTORY_FEATURES)] = False
        distances = np.mean(((self.features[:, mask] - np.asarray(vector)[mask]) / self.feature_scale[mask]) ** 2, axis=1)
        positions, functions = [], set()
        for pos in np.argsort(distances, kind='stable'):
            row = self.cases[int(pos)]
            if row['step'] != step or row['function'] in functions:
                continue
            positions.append(int(pos)); functions.add(row['function'])
            if len(positions) == 3:
                break
        radius_key = ('no_online' if self.arm == 'no_online' else 'full') + ':' + str(step)
        radius = self.memory['coverage_radii'][radius_key]
        neighbors = [dict(id=self.cases[p]['id'], distance=float(distances[p])) for p in positions]
        supported = len(positions) == 3 and float(distances[positions[-1]]) <= radius
        weights = np.asarray([1 / (1 + distances[p]) for p in positions]); weights /= weights.sum()
        rows, allowed = [], ['matched38']
        for name in ACTION_NAMES:
            a = np.array([self.cases[p]['advantages']['5'][name] for p in positions])
            b = np.array([self.cases[p]['advantages']['10'][name] for p in positions])
            m5, m10 = float(weights @ a), float(weights @ b)
            std = float(np.sqrt(weights @ ((a-m5)**2)))
            positive = int((a > 0).sum())
            admissible = name == 'matched38' or (supported and positive >= 2 and m5 > max(0., .25 * std) and m10 >= 0)
            if admissible and name != 'matched38':
                allowed.append(name)
            rows.append([name, m5, m10, std, positive, len(positions), admissible])
        return dict(enabled=True, supported=supported, radius=radius, neighbors=neighbors, rows=rows, allowed=allowed)

    def compact_payload(self, payload):
        items = []
        for item in payload['trajectories']:
            current = item['current']
            history = None if self.arm == 'no_online' else dict(
                sources=current['source_statistics'], samples=current['feedback_samples'],
                # Do not send observed_progress5 as an action reward.
                observations=current['previous_actions'])
            vector = list(item['decision_features'])
            if self.arm == 'no_online':
                for i in HISTORY_FEATURES:
                    vector[i] = None
            e = item['evidence']
            items.append(dict(trajectory_id=item['trajectory_id'], state=vector,
                current_options=item['current_options'], measured_history=history,
                offline=None if self.arm == 'no_offline' else {k:v for k,v in e.items() if k != 'allowed'},
                allowed=item['eligible_configs']))
        return compact(dict(policy_id=POLICY_ID, default_profile='matched38', failure_fallback='matched38',
            decision_context_sha256=request_context_digest(payload['trajectories'], self.execution_context),
            execution_context_sha256=digest(self.execution_context), memory_sha256=self.memory_hash,
            action_names=list(ACTION_NAMES), generation_menu={k:[PROFILES[k]['temperature'],PROFILES[k]['local_scale']] for k in GENERATION_NAMES},
            state_names=FEATURES, offline_columns=['action','gain5','gain10','dispersion5','positive_functions','functions','admissible'],
            fusion_columns=['multiplier','selected_slot','ungated_slot','predicted_mu_minus_best','sigma','score_gap_to_anchor','changes_point_vs_calibrated'],
            current_fusion_rounds=5, future_generation_rounds=4, step=self.steps, trajectories=items))

    def choose(self, **kwargs):
        options, vectors, _ = self.numerical_options(kwargs)
        self.request_event, self.api_trajectories = False, []
        if self.steps in (5, 15) and self.arm != 'matched38':
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
            selected = super().choose(**kwargs)
        finally:
            self.active_record = callback
        record = self.last_record
        record.update(policy_id=POLICY_ID, api_row_validation=copy.deepcopy(self.api_row_validation),
            applied_generation=list(self.applied_generation), planned_generation=[n.split('@')[0] for n in self.plans],
            current_options=options, decision_features=vectors,
            temporal_semantics='fusion now5 rounds, generation next4 rounds')
        if callback:
            callback(record)
        return selected
