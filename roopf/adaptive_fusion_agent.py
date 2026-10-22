from __future__ import annotations

import copy
import math

import numpy as np
import torch

from roopf.fusion_agent import FusionAgent, context_rows, plain


POLICY = '''你是ROOPF的证据融合配置控制器，目标最小化。只输出严格JSON config_ids映射，键必须覆盖所有输入trajectory_id，值只能来自该轨迹eligible_configs。
release严格执行原策略；adaptive_fusion使用由已评价点的配对排序可靠性更新的Residual倍率和接受门槛；online_only在候选评分与接受判断中均不使用Residual，但保留Anchor和其他结构保护。
候选0为竞争Anchor，其余为完整36候选池；另有独立保护Anchor槽。只选择配置，不重新计算数值。
proxy_acquisition越小越好，sigma是预测不确定性而非真实误差。probability是冻结离线支持分数，不是校准成功概率。当前候选真实目标不可见。
actions已给出每种配置最终受保护候选和评分。adaptation为仅使用之前真实评价的配对排序可靠性、配置收益描述统计和当前权重门槛；反馈有选择偏差，不是因果收益保证。
offline_cases来自旧三候选动作空间，只提供情境参考，不能把其动作收益当作当前完整候选池配置的反事实收益。证据不足选择release。
没有参数训练、真实值预知或无偏可靠性保证。不要输出解释、Markdown或额外字段。'''


class AdaptiveFusionAgent(FusionAgent):
    expand_full_pool = True

    def __init__(self, memory_path, journal, mode='llm', budget=None):
        super().__init__(memory_path, journal, mode, budget, memory_version='r04',
                         request_interval=23, system_prompt=POLICY)
        self.allowed = ['release', 'adaptive_fusion', 'online_only']

    def reset(self, batch_size):
        super().reset(batch_size)
        self.reliability = [dict(proxy_success=2.0, proxy_total=4.0,
            residual_success=2.0, residual_total=4.0) for _ in range(batch_size)]
        self.config_returns = [{name: dict(n=0, mean=0.0) for name in
            ('release', 'adaptive_fusion', 'online_only')} for _ in range(batch_size)]

    def adaptation(self, index):
        evidence = self.reliability[index]
        proxy = evidence['proxy_success'] / evidence['proxy_total']
        residual = evidence['residual_success'] / evidence['residual_total']
        multiplier = min(2.0, max(0.5, residual / max(proxy, 1e-8)))
        adverse = max(0.0, -self.config_returns[index]['adaptive_fusion']['mean'])
        risk = 1.0 - min(proxy, residual)
        online_adverse = max(0.0, -self.config_returns[index]['online_only']['mean'])
        return dict(proxy_rank_reliability=proxy, residual_rank_reliability=residual,
            residual_multiplier=multiplier, margin=min(0.16, 0.06 + 0.10 * risk + 0.02 * adverse),
            soft_margin=min(0.08, 0.03 + 0.05 * risk + 0.01 * adverse),
            online_margin=min(0.16, 0.06 + 0.10 * (1.0 - proxy) + 0.02 * online_adverse),
            online_soft_margin=min(0.08, 0.03 + 0.05 * (1.0 - proxy) + 0.01 * online_adverse),
            pair_counts=dict(proxy=evidence['proxy_total'] - 4, residual=evidence['residual_total'] - 4),
            config_observed_relative_to_protected_anchor=copy.deepcopy(self.config_returns[index]),
            selection_biased=True, causal_advantages=False)

    def retrieve(self, row):
        vector = np.asarray(row['fusion_features'])
        distances = (((self.features - vector) / self.feature_scale) ** 2).mean(1)
        selected, functions = [], set()
        for position in np.argsort(distances, kind='stable'):
            case = self.cases[position]
            if case['function'] in functions or case['context_key'] != row['context_key']:
                continue
            selected.append(dict(id=case['id'], function=case['function'],
                context_key=case['context_key'], fusion_features=case['fusion_features'],
                distance=round(float(distances[position]), 6),
                old_shortlist_horizon_advantages=case['horizon_advantages'],
                scope='context reference only; different candidate and action space'))
            functions.add(case['function'])
            if len(selected) == 3:
                break
        return selected

    def feedback_summary(self, index):
        return copy.deepcopy(super().feedback_summary(index))

    def choose(self, *, original_idx, score, mu, sigma, probability, state, archive_y,
               fitness, candidates, candidate_ops, remaining, problem, residual_weight,
               selection_rule=None):
        if selection_rule is None:
            raise ValueError('A shared protected selector is required')
        scale = archive_y.std(1, keepdim=True, unbiased=False).clamp_min(1e-8)
        signal = torch.zeros_like(score) if probability is None else ((probability - probability.mean(1, keepdim=True)) /
            probability.std(1, keepdim=True, unbiased=False).clamp_min(1e-6)).clamp(-2.5, 2.5)
        online = score + residual_weight * scale * signal
        adaptations = [self.adaptation(index) for index in range(score.size(0))]
        multipliers = score.new_tensor([row['residual_multiplier'] for row in adaptations]).view(-1, 1)
        margins = score.new_tensor([row['margin'] for row in adaptations]).view(-1, 1)
        soft = score.new_tensor([row['soft_margin'] for row in adaptations]).view(-1, 1)
        online_margins = score.new_tensor([row['online_margin'] for row in adaptations]).view(-1, 1)
        online_soft = score.new_tensor([row['online_soft_margin'] for row in adaptations]).view(-1, 1)
        fused = online - residual_weight * multipliers * scale * signal
        options = dict(release=original_idx.clone(),
            adaptive_fusion=selection_rule(fused, fused.argmin(1, keepdim=True), margin=margins, soft_margin=soft),
            online_only=selection_rule(online, online.argmin(1, keepdim=True), margin=online_margins,
                                       soft_margin=online_soft, online_only=True))
        rows = context_rows(state, score, mu, sigma, probability, scale, original_idx,
            residual_weight, fitness=fitness, candidate_ops=candidate_ops)
        eligible, cases = [], []
        for index, row in enumerate(rows):
            row.pop('state')
            row.pop('score')
            row['candidate_count'] = candidates.size(1)
            row['adaptation'] = adaptations[index]
            row['actions'] = {}
            names = ['release']
            for name, indices in options.items():
                selected_index = int(indices[index, 0])
                changes = bool((candidates[index, selected_index] != candidates[index, int(original_idx[index, 0])]).any())
                action_scores = score if name == 'release' else fused if name == 'adaptive_fusion' else online
                row['actions'][name] = dict(selected_index=selected_index, changes_original=changes,
                    selected_acquisition=float(action_scores[index, selected_index] / scale[index, 0]),
                    residual_in_score=name != 'online_only', residual_in_gate=name != 'online_only')
                if name != 'release' and changes:
                    names.append(name)
            eligible.append(names)
            cases.append(self.retrieve(row))
        ids = ['release'] * len(rows)
        trajectories = []
        if self.steps % self.request_interval == 0:
            trajectories = [dict(trajectory_id=str(index), current=row, offline_cases=cases[index],
                eligible_configs=eligible[index], observed_feedback=self.feedback_summary(index))
                for index, row in enumerate(rows) if len(eligible[index]) > 1]
            if trajectories and self.mode == 'llm':
                payload = dict(trajectory_count=len(trajectories),
                    allowed_configs={name: dict(description=name) for name in self.allowed}, trajectories=trajectories)
                try:
                    response = self.call(payload)
                    if len(response) != len(trajectories) or any(name not in item['eligible_configs'] for name, item in zip(response, trajectories)):
                        raise ValueError('Invalid configuration response')
                    for item, name in zip(trajectories, response):
                        ids[int(item['trajectory_id'])] = name
                except Exception:
                    self.stats['failures'] += 1
            elif trajectories and self.mode == 'retrieval':
                for item in trajectories:
                    index = int(item['trajectory_id'])
                    preferred = 'adaptive_fusion' if adaptations[index]['residual_rank_reliability'] >= adaptations[index]['proxy_rank_reliability'] else 'online_only'
                    ids[index] = preferred if preferred in eligible[index] else 'release'
        selected = original_idx.clone()
        for index, name in enumerate(ids):
            selected[index] = options[name][index]
        original_points = torch.gather(candidates, 1, original_idx.unsqueeze(-1).expand(-1, -1, candidates.size(-1)))
        selected_points = torch.gather(candidates, 1, selected.unsqueeze(-1).expand(-1, -1, candidates.size(-1)))
        self.stats['decisions'] += len(rows)
        self.stats['changed'] += int((selected != original_idx).sum())
        self.stats['changed_points'] += int((selected_points != original_points).any(-1).sum())
        self.stats['opportunities'] += sum(len(names) > 1 for names in eligible)
        self.stats['no_supported_action'] += sum(len(names) == 1 for names in eligible)
        self.stats['nonrelease'] += sum(name != 'release' for name in ids)
        if self.active_record is not None:
            self.active_record(copy.deepcopy(dict(step=self.steps, configs=ids, original=original_idx.tolist(),
                selected=selected.tolist(), eligible_configs=eligible, current=rows,
                retrieved_case_ids=[[case['id'] for case in found] for found in cases],
                feedback=[self.feedback_summary(index) for index in range(len(rows))])))
        self.pending = list(ids)
        self.steps += 1
        return selected

    def observe(self, observed, mu, probability, previous_best, selected_ops, archive_y):
        pending = None if self.pending is None else list(self.pending)
        super().observe(observed, mu, probability, previous_best, selected_ops, archive_y)
        if pending is None or observed.size(1) < 2:
            return
        scale = archive_y.std(1, unbiased=False).clamp_min(1e-8)
        for index, name in enumerate(pending):
            actual = float(observed[index, -1] - observed[index, 0])
            if actual != 0.0:
                for source, prediction, lower_better in (('proxy', mu, True), ('residual', probability, False)):
                    if prediction is None:
                        continue
                    difference = float(prediction[index, -1] - prediction[index, 0])
                    if not math.isfinite(difference):
                        continue
                    aligned = difference if lower_better else -difference
                    correct = 0.5 if aligned == 0.0 else float((aligned < 0) == (actual < 0))
                    self.reliability[index][source + '_success'] += correct
                    self.reliability[index][source + '_total'] += 1.0
            reward = math.tanh(-actual / float(scale[index]))
            estimate = self.config_returns[index][name]
            estimate['n'] += 1
            estimate['mean'] += (reward - estimate['mean']) / estimate['n']
