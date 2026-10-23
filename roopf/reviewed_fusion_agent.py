from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from roopf.adaptive_fusion_agent import AdaptiveFusionAgent
from roopf.fusion_agent import FusionAgent


POLICY_ID = 'r06_full37_incumbent_one_action_v1'
CONFIGS = ('release', 'adaptive_fusion', 'online_only')
SYSTEM = '''Select a protected ROOPF evidence-fusion configuration. Minimize the objective.
Return ONLY {"config_ids":{"trajectory_id":"eligible_config"}}, using exactly the supplied IDs. No reasoning, explanation, Markdown or other keys.
release preserves the original point. adaptive_fusion uses feedback-adapted Residual weight and gates. online_only removes Residual from scoring and acceptance, retaining structural protections and the separate protected Anchor.
All37 candidates remain available to the program. actions are precomputed protected results, not proposals requiring your arithmetic. Negative predicted_minus_best means predicted improvement; uncertainty is not observed error; offline support is not a calibrated success probability.
offline_cases use this exact37-candidate, one-action policy. Their paired5/10-round advantages are against release, not guaranteed current outcomes. Feedback only uses past evaluated points, is selection-biased, and is not causal counterfactual evidence. Prefer supported positive multi-round evidence; release remains valid when evidence is adverse or insufficient.
On review, use the previous intervention's observed incumbent and5-round progress, not just the candidate ranking. Never infer unobserved current objective values.'''


class ReviewedFusionAgent(AdaptiveFusionAgent):
    def __init__(self, memory_path, journal, mode='llm', budget=None):
        path = Path(memory_path)
        self.memory = json.loads(path.read_text())
        if self.memory.get('version') != 'r06' or self.memory.get('policy_id') != POLICY_ID:
            raise ValueError('R06 requires matching full37, incumbent-aligned memory')
        self.memory_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        self.cases = self.memory['cases']
        self.features = np.asarray([case['fusion_features'] for case in self.cases], dtype=float).reshape(-1, 7)
        self.feature_scale = np.maximum(self.features.std(0), .1) if self.cases else np.ones(7)
        self.allowed = list(CONFIGS)
        self.journal = Path(journal)
        self.journal.mkdir(parents=True, exist_ok=True)
        self.mode, self.budget = mode, budget if budget is not None else dict(used=0, limit=600)
        self.request_interval, self.system_prompt, self.active_record = 1, SYSTEM, None
        self.forced = {}
        self.reset(0)

    def reset(self, batch_size):
        super().reset(batch_size)
        self.rank_windows = [dict(proxy=[], residual=[]) for _ in range(batch_size)]
        self.horizon_returns = [{name: dict(n=0, mean=0.0) for name in CONFIGS} for _ in range(batch_size)]
        self.interventions = [[] for _ in range(batch_size)]
        self.review_due = [[] for _ in range(batch_size)]
        self.batch_requests, self.last_request = 0, -100
        self.last_rows = []

    def adaptation(self, index):
        result = super().adaptation(index)
        result['config_incumbent_utility'] = result.pop('config_observed_relative_to_protected_anchor')
        result['horizon5_observed_progress'] = copy.deepcopy(self.horizon_returns[index])
        result['rank_window_limit'] = 16
        result['reward_definition'] = 'competing-point incumbent gain / archive std minus0.001 evaluation cost; not causal'
        return result

    def retrieve(self, row):
        if not self.cases:
            return []
        vector = np.asarray(row['fusion_features'])
        distances = (((self.features - vector) / self.feature_scale) ** 2).mean(1)
        result, functions = [], set()
        for position in np.argsort(distances, kind='stable'):
            case = self.cases[position]
            if case['function'] in functions or case['context_key'] != row['context_key']:
                continue
            result.append(dict(id=case['id'], function=case['function'], distance=round(float(distances[position]), 6),
                paired_advantages5=case['horizon_advantages']['5'], paired_advantages10=case['horizon_advantages']['10'],
                action_changes=case['action_changes'], policy_id=POLICY_ID))
            functions.add(case['function'])
            if len(result) == 3:
                break
        return result

    def feedback_summary(self, index):
        return dict(adaptation=self.adaptation(index),
            competing_slot_recent=copy.deepcopy(self.history[index][-8:]),
            completed_reviews=copy.deepcopy(self.review_due[index]), selection_biased=True)

    @staticmethod
    def compact(row):
        actions = {}
        for name, action in row['actions'].items():
            selected = action['selected_index']
            actions[name] = dict(index=selected, changes_original=action['changes_original'],
                acquisition=round(action['selected_acquisition'], 6),
                predicted_minus_best=round(row['mu_minus_incumbent'][selected], 6),
                uncertainty=round(row['sigma'][selected], 6),
                offline_support=None if row['probability'] is None else row['probability'][selected])
        return dict(context_sha256=row['context_sha256'], candidate_count=row['candidate_count'],
            remaining=row['state_named']['remaining_budget'], stagnation=row['state_named']['stagnation'],
            fusion_features=row['fusion_features'], actions=actions,
            adaptation=row['adaptation'])

    def choose(self, **kwargs):
        mode, callback = self.mode, self.active_record
        captured = []
        self.mode, self.active_record = 'off', captured.append
        try:
            original = super().choose(**kwargs)
        finally:
            self.mode, self.active_record = mode, callback
        record = captured[0]
        step = record['step']
        rows, eligible = record['current'], record['eligible_configs']
        scales = kwargs['archive_y'].std(1, unbiased=False).clamp_min(1e-8)
        for index, row in enumerate(rows):
            row['scale'] = float(scales[index])
            row['nfe_before'] = int(round(300 * (1 - kwargs['remaining'])))
            row['context_sha256'] = hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()
        self.last_rows = copy.deepcopy(rows)
        due = [index for index, items in enumerate(self.review_due) if items]
        meaningful = [index for index, row in enumerate(rows) if len(eligible[index]) > 1 and
            (row['fusion_features'][0] or abs(row['adaptation']['proxy_rank_reliability'] -
             row['adaptation']['residual_rank_reliability']) >= .10 or row['state_named']['stagnation'] >= .04)]
        request = mode in ('llm', 'retrieval') and 5 <= step <= 35 and self.batch_requests < 2 and (
            bool(due) or (self.batch_requests == 0 and step <= 25 and bool(meaningful)))
        request_indices = (due if due else [index for index in range(len(rows)) if len(eligible[index]) > 1]) if request else []
        feedback_before = [self.feedback_summary(index) for index in range(len(rows))]
        trajectories = [dict(trajectory_id=str(index), current=self.compact(rows[index]),
            offline_cases=self.retrieve(rows[index]), eligible_configs=eligible[index],
            observed_feedback=feedback_before[index]) for index in request_indices]
        ids = ['release'] * len(rows)
        request_kind = 'review' if due and request else 'event' if request else 'none'
        if request:
            self.batch_requests += 1
            self.last_request = step
            if mode == 'llm':
                try:
                    response = self.call(dict(policy_id=POLICY_ID, request_kind=request_kind,
                        trajectory_count=len(trajectories), trajectories=trajectories))
                    if len(response) != len(trajectories) or any(name not in item['eligible_configs'] for name, item in zip(response, trajectories)):
                        raise ValueError('Invalid configuration response')
                    for item, name in zip(trajectories, response):
                        ids[int(item['trajectory_id'])] = name
                except Exception:
                    self.stats['failures'] += 1
            else:
                for item in trajectories:
                    index = int(item['trajectory_id'])
                    evidence = item['offline_cases']
                    estimates = {name: sum(case['paired_advantages5'][name] / (1 + case['distance']) for case in evidence)
                        for name in item['eligible_configs']}
                    ids[index] = max(estimates, key=estimates.get)
            for index in due:
                self.review_due[index] = []
        for index, forced in self.forced.get(step, {}).items():
            ids[int(index)] = forced
        selected = original.clone()
        for index, name in enumerate(ids):
            selected[index, 0] = rows[index]['actions'][name]['selected_index']
        points = kwargs['candidates']
        before = torch.gather(points, 1, original.unsqueeze(-1).expand(-1, -1, points.size(-1)))
        after = torch.gather(points, 1, selected.unsqueeze(-1).expand(-1, -1, points.size(-1)))
        self.stats['changed'] += int((selected != original).sum())
        self.stats['changed_points'] += int((after != before).any(-1).sum())
        self.stats['nonrelease'] += sum(name != 'release' for name in ids)
        record.update(configs=ids, selected=selected.tolist(), current=rows,
            feedback=feedback_before,
            request_event=bool(request and mode == 'llm'), request_kind=request_kind,
            api_trajectories=copy.deepcopy(trajectories) if request and mode == 'llm' else [],
            reviewed_rows=due if request else [], review_backlog=copy.deepcopy(self.review_due))
        if callback is not None:
            callback(copy.deepcopy(record))
        self.pending = list(ids)
        return selected

    def observe(self, observed, mu, probability, previous_best, selected_ops, archive_y):
        pending = None if self.pending is None else list(self.pending)
        FusionAgent.observe(self, observed, mu, probability, previous_best, selected_ops, archive_y)
        if pending is None:
            return
        step = self.steps - 1
        scale = archive_y.std(1, unbiased=False).clamp_min(1e-8)
        for index, name in enumerate(pending):
            actual = float(observed[index, -1] - observed[index, 0])
            if actual != 0:
                for source, predictions, direction in (('proxy', mu, 1), ('residual', probability, -1)):
                    if predictions is None:
                        continue
                    difference = direction * float(predictions[index, -1] - predictions[index, 0])
                    if not math.isfinite(difference):
                        continue
                    correct = .5 if difference == 0 else float((difference < 0) == (actual < 0))
                    window = (self.rank_windows[index][source] + [correct])[-16:]
                    self.rank_windows[index][source] = window
                    self.reliability[index][source + '_success'] = 2 + sum(window)
                    self.reliability[index][source + '_total'] = 4 + len(window)
            gain = max(0.0, float(previous_best[index] - observed[index, -1])) / float(scale[index])
            reward = min(20.0, gain) - .001
            estimate = self.config_returns[index][name]
            estimate['n'] += 1
            estimate['mean'] += (reward - estimate['mean']) / estimate['n']
            if name != 'release':
                self.interventions[index].append(dict(config=name, step=step,
                    start_best=float(previous_best[index]), scale=float(scale[index]), immediate_gain=gain))
            remaining = []
            best = min(float(previous_best[index]), float(observed[index].min()))
            for action in self.interventions[index]:
                if step - action['step'] >= 4:
                    progress = max(0.0, action['start_best'] - best) / action['scale']
                    estimate = self.horizon_returns[index][action['config']]
                    estimate['n'] += 1
                    estimate['mean'] += (progress - estimate['mean']) / estimate['n']
                    self.review_due[index].append(dict(**action, horizon_rounds=5,
                        observed_best_progress=progress, selection_biased=True, causal_advantage=False))
                else:
                    remaining.append(action)
            self.interventions[index] = remaining
