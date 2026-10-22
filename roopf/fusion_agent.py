from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

from roopf.fusion_rules import protected_indices


CONFIGS = {
    'release': dict(residual_multiplier=1.0, margin=0.10, soft_margin=0.04),
    'proxy': dict(residual_multiplier=0.0, margin=0.10, soft_margin=0.04),
    'residual': dict(residual_multiplier=2.0, margin=0.10, soft_margin=0.04),
    'cautious': dict(residual_multiplier=1.0, margin=0.16, soft_margin=0.08),
    'balanced': dict(residual_multiplier=1.0, margin=0.06, soft_margin=0.03),
}

STATE_NAMES = ('best_z', 'spread_z', 'top_gap', 'diversity', 'elite_diversity',
               'centroid_shift', 'remaining_budget', 'used_budget', 'stagnation',
               'population_log_size', 'dimension_divided_by_100')

SYSTEM = '''你是ROOPF的离线在线证据融合控制器。每条轨迹只能从其eligible_configs选择配置。
目标最小化。候选0是竞争Anchor，另一个保护Anchor槽始终由程序保留。
proxy_acquisition包含在线代理和结构保护项，越小越好；probability为冻结离线Residual的改进支持，越大越好。
sigma是在线代理不确定性，不是真实误差。mu_minus_incumbent是预测目标减去已知最优目标，负值表示预测改善。
各目标量除以已评价档案的目标标准差；mu字段相对候选0。state_named明确每项含义。
actions已由程序计算每种配置的融合评分、选中候选及是否改变原选择，不要自行重算。
proxy去掉Residual评分项，residual将其加倍；其他保护条件不变。谨慎/宽松配置只调整接受门槛。
离线案例包含当时证据、动作和单步收益；不是终局保证。只用训练/验证函数构建，不含当前候选真实值。
反馈仅来自实际选中点，有选择偏差，不是所有候选的无偏可靠性估计。null表示未知而非零误差。
优先根据在线/离线冲突、历史同类案例及实际反馈决策，证据不足选release。不要推理文本。
输出严格JSON：{"config_ids":{"0":"release","1":"balanced"}}。
config_ids必须是映射，键恰好覆盖输入的全部trajectory_id；每个值只能是allowed_configs中的配置名。不要解释、Markdown或额外字段。'''


def plain(tensor):
    return tensor.detach().cpu().double().numpy()


def fusion_indices(score, probability, scale, original_idx, residual_weight, selection_rule=None):
    batch = score.size(0)
    result = {'release': original_idx.clone()}
    signal = torch.zeros_like(score)
    if probability is not None:
        signal = ((probability - probability.mean(1, keepdim=True)) /
                  probability.std(1, keepdim=True, unbiased=False).clamp_min(1e-6)).clamp(-2.5, 2.5)
    raw_proxy = score + residual_weight * scale * signal
    for name, config in CONFIGS.items():
        if name == 'release':
            continue
        fused = raw_proxy - residual_weight * config['residual_multiplier'] * scale * signal
        selected = fused.argmin(1, keepdim=True)
        if selection_rule is not None:
            result[name] = selection_rule(fused, selected, margin=config['margin'], soft_margin=config['soft_margin'])
        else:
            counts = torch.zeros(batch, device=score.device, dtype=score.dtype)
            result[name] = protected_indices(
                fused, probability, scale, selected, 1, ablation={'residual_safety_gate'},
                portfolio_seen=counts, portfolio_best_count=counts, portfolio_admit_count=counts,
                portfolio_no_admit=counts, stagnation=counts, remaining=1.0,
                local_proxy_friendly=True, hpo=False, embedded=False,
                selected_ops=torch.zeros((batch, score.size(1) - 1), dtype=torch.long, device=score.device),
                op_penalty=None, margin=config['margin'], soft_margin=config['soft_margin'],
            )
    return result


def context_rows(state, score, mu, sigma, probability, scale, original_idx, residual_weight,
                 fitness=None, candidate_ops=None, options=None):
    values = dict(state=plain(state), score=plain(score / scale),
                  mu=plain((mu - mu[:, :1]) / scale), sigma=plain(sigma / scale))
    probabilities = plain(probability) if probability is not None else None
    signal = torch.zeros_like(score) if probability is None else ((probability - probability.mean(1, keepdim=True)) /
        probability.std(1, keepdim=True, unbiased=False).clamp_min(1e-6)).clamp(-2.5, 2.5)
    proxy = score + residual_weight * scale * signal
    rows = []
    for index in range(state.size(0)):
        row = {name: np.round(value[index], 6).tolist() for name, value in values.items()}
        row['probability'] = np.round(probabilities[index], 6).tolist() if probabilities is not None else None
        row['original_index'] = int(original_idx[index, 0])
        row['residual_weight'] = float(residual_weight)
        row['state_named'] = dict(zip(STATE_NAMES, row['state']))
        row['proxy_acquisition'] = np.round(plain(proxy / scale)[index], 6).tolist()
        row['mu_minus_incumbent'] = None if fitness is None else np.round(plain((mu - fitness.amin(1, keepdim=True)) / scale)[index], 6).tolist()
        row['candidate_ops'] = None if candidate_ops is None else candidate_ops[index].tolist()
        row['fusion_features'] = [
            float(proxy[index].argmin() != (probability[index].argmax() if probability is not None else proxy[index].argmin())),
            float((proxy[index, 0] - proxy[index].min()) / scale[index, 0]),
            float((sigma[index].max() - sigma[index].min()) / scale[index, 0]),
            float(probability[index].std(unbiased=False)) if probability is not None else 0.0,
            float(original_idx[index, 0] > 0), float(state[index, 6]), float(state[index, 8]),
        ]
        row['context_key'] = ('conflict' if row['fusion_features'][0] else 'agree') + ('_end' if row['state'][6] < 0.15 else '_late')
        if options is not None:
            row['actions'] = {}
            for name, chosen in options.items():
                fused = proxy - residual_weight * CONFIGS[name]['residual_multiplier'] * scale * signal
                row['actions'][name] = dict(selected_index=int(chosen[index, 0]),
                    changes_original=bool(chosen[index, 0] != original_idx[index, 0]),
                    fused_acquisition=np.round(plain(fused / scale)[index], 6).tolist())
        rows.append(row)
    return rows


class FusionAgent:
    def __init__(self, memory_path, journal, mode='llm', budget=None, *,
                 memory_version='r03', request_interval=23, system_prompt=SYSTEM):
        self.memory = json.loads(Path(memory_path).read_text())
        self.memory_hash = hashlib.sha256(Path(memory_path).read_bytes()).hexdigest()
        self.cases = self.memory['cases']
        if self.memory.get('version') != memory_version:
            raise ValueError('Memory version does not match the requested evidence semantics')
        if request_interval < 1:
            raise ValueError('Request interval must be positive')
        self.request_interval = request_interval
        self.system_prompt = system_prompt
        self.features = np.asarray([case['fusion_features'] for case in self.cases], dtype=float)
        self.feature_scale = np.maximum(self.features.std(0), 0.1)
        self.allowed = self.memory['allowed_configs']
        self.journal = Path(journal)
        self.journal.mkdir(parents=True, exist_ok=True)
        self.mode = mode
        self.budget = budget if budget is not None else {'used': 0, 'limit': 96}
        self.active_record = None
        self.reset(0)

    def reset(self, batch_size):
        self.feedback = [dict(n=0, residual_n=0, proxy_error=0.0, residual_error=0.0, improvements=0, portfolio=0) for _ in range(batch_size)]
        self.steps = 0
        self.ids = ['release'] * batch_size
        self.history = [[] for _ in range(batch_size)]
        self.pending = None
        self.stats = dict(decisions=0, changed=0, requests=0, failures=0, nonrelease=0,
                          opportunities=0, changed_points=0, no_supported_action=0)

    def observe(self, observed, mu, probability, previous_best, selected_ops, archive_y):
        scale = archive_y.std(1, unbiased=False).clamp_min(1e-8)
        for index, feedback in enumerate(self.feedback):
            for slot in range(observed.size(1)):
                value = float(observed[index, slot])
                improved = value < float(previous_best[index])
                feedback['improvements'] += int(improved)
                feedback['portfolio'] += int(selected_ops[index, slot] >= 0)
                if mu is not None and math.isfinite(float(mu[index, slot])):
                    feedback['proxy_error'] += min(20.0, abs(value - float(mu[index, slot])) / float(scale[index]))
                    feedback['n'] += 1
                if probability is not None and math.isfinite(float(probability[index, slot])):
                    feedback['residual_error'] += (float(probability[index, slot]) - float(improved)) ** 2
                    feedback['residual_n'] += 1
            if self.pending is not None:
                slot = observed.size(1) - 1
                self.history[index].append(dict(config=self.pending[index],
                    origin='portfolio' if int(selected_ops[index, slot]) >= 0 else 'anchor',
                    improvement=float((previous_best[index] - observed[index, slot]).clamp_min(0) / scale[index]),
                    proxy_error=None if mu is None or not math.isfinite(float(mu[index, slot])) else
                        float((observed[index, slot] - mu[index, slot]).abs() / scale[index]),
                    residual_error=None if probability is None or not math.isfinite(float(probability[index, slot])) else
                        (float(probability[index, slot]) - float(observed[index, slot] < previous_best[index])) ** 2))
                self.history[index] = self.history[index][-32:]
        self.pending = None

    def feedback_summary(self, index):
        feedback = dict(self.feedback[index])
        feedback['mean_proxy_error'] = None if not feedback['n'] else round(feedback['proxy_error'] / feedback['n'], 5)
        feedback['mean_residual_error'] = None if not feedback['residual_n'] else round(feedback['residual_error'] / feedback['residual_n'], 5)
        feedback.pop('proxy_error')
        feedback.pop('residual_error')
        feedback['competing_slot_recent'] = self.history[index]
        feedback['selection_biased'] = True
        return feedback

    def retrieve(self, row):
        vector = np.asarray(row['fusion_features'])
        distances = (((self.features - vector) / self.feature_scale) ** 2).mean(1)
        indices, functions = [], set()
        for index in np.argsort(distances, kind='stable'):
            case = self.cases[index]
            if case['function'] in functions or case['context_key'] != row['context_key']:
                continue
            if not any(case['actions'][name]['changes_original'] for name in self.allowed):
                continue
            indices.append(index)
            functions.add(case['function'])
            if len(indices) == 3:
                break
        return [dict(self.cases[index], distance=round(float(distances[index]), 6)) for index in indices]

    def eligible_configs(self, row, cases):
        eligible = ['release']
        validation = self.memory.get('context_validation', {}).get(row['context_key'], {})
        for name in self.allowed:
            if name == 'release' or not row['actions'][name]['changes_original']:
                continue
            support = validation.get(name, {})
            local = [case['advantages'][name] for case in cases if case['actions'][name]['changes_original']]
            if support.get('supported', False) and local and np.mean(local) > 0:
                eligible.append(name)
        return eligible

    def call(self, payload):
        data = json.dumps(dict(model='deepseek-v4-flash', temperature=0, max_tokens=4096,
                               response_format={'type': 'json_object'},
                               thinking={'type': 'disabled'},
                               messages=[dict(role='system', content=self.system_prompt), dict(role='user', content=json.dumps(payload))])).encode()
        identity = hashlib.sha256(data).hexdigest()
        target = self.journal / (identity + '.json')
        if target.exists():
            result = json.loads(target.read_text())
            if result['status'] != 'ok':
                raise RuntimeError('已记录失败或结果未知，不自动重复付费')
            return result['config_ids']
        if self.budget['used'] >= self.budget['limit']:
            raise RuntimeError('本轮API次数上限')
        record = dict(status='pending', request_sha256=identity, request=json.loads(data), memory_sha256=self.memory_hash)
        target.write_text(json.dumps(record, ensure_ascii=False, indent=2))
        self.budget['used'] += 1
        self.stats['requests'] += 1
        started = time.monotonic()
        try:
            request_file = self.journal / (identity + '.request.payload')
            request_file.write_bytes(data)
            script = Path(__file__).resolve().parents[1] / 'scripts/worldcodes_request.ps1'
            windows_script = subprocess.check_output(['wslpath', '-w', str(script)], text=True).strip()
            windows_request = subprocess.check_output(['wslpath', '-w', str(request_file)], text=True).strip()
            response = subprocess.run(['/mnt/c/WINDOWS/System32/WindowsPowerShell/v1.0/powershell.exe',
                        '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', windows_script,
                        '-RequestFile', windows_request], capture_output=True, encoding='utf-8', timeout=45)
            if response.returncode:
                raise RuntimeError(response.stderr.strip()[:300])
            raw_response = Path(str(request_file) + '.response.raw').read_text(encoding='utf-8-sig')
            response_data = json.loads(raw_response)
            record['transport'] = 'Windows system proxy via PowerShell; WSL direct DNS failed'
            record['response'] = response_data
            if response_data['choices'][0].get('finish_reason') != 'stop':
                raise ValueError('Incomplete API output; no automatic paid retry')
            content = response_data['choices'][0]['message']['content'].strip()
            if content.startswith('```'):
                content = content.split('\n', 1)[1].rsplit('```', 1)[0]
            parsed = json.loads(content)
            if not isinstance(parsed, dict) or set(parsed) != {'config_ids'}:
                raise ValueError('Expected only config_ids')
            ids = parsed['config_ids']
            if isinstance(ids, dict):
                expected = [item['trajectory_id'] for item in payload['trajectories']]
                if set(ids) != set(expected):
                    raise ValueError('trajectory_id集合不匹配')
                ids = [ids[index] for index in expected]
            else:
                raise ValueError('config_ids must be an object')
            if not all(isinstance(name, str) and name in item['eligible_configs'] for name, item in zip(ids, payload['trajectories'])):
                raise ValueError('Unsupported config for trajectory')
            record.update(status='ok', config_ids=ids, response=response_data)
            return ids
        except Exception as error:
            record.update(status='failed', error_type=type(error).__name__, error=str(error)[:300])
            raise
        finally:
            record['elapsed_seconds'] = time.monotonic() - started
            target.write_text(json.dumps(record, ensure_ascii=False, indent=2))

    def choose(self, *, original_idx, score, mu, sigma, probability, state, archive_y,
               fitness, candidates, candidate_ops, remaining, problem, residual_weight,
               selection_rule=None):
        if selection_rule is None:
            raise ValueError('R03 requires the original shared protected selection rule')
        scale = archive_y.std(1, keepdim=True, unbiased=False).clamp_min(1e-8)
        options = fusion_indices(score, probability, scale, original_idx, residual_weight, selection_rule)
        rows = context_rows(state, score, mu, sigma, probability, scale, original_idx, residual_weight,
                            fitness=fitness, candidate_ops=candidate_ops, options=options)
        retrieved = [self.retrieve(row) for row in rows]
        eligible = [self.eligible_configs(row, cases) for row, cases in zip(rows, retrieved)]
        if self.steps % self.request_interval == 0:
            trajectories = []
            for index, row in enumerate(rows):
                if len(eligible[index]) > 1:
                    trajectories.append(dict(trajectory_id=str(index), current=row, offline_cases=retrieved[index],
                        eligible_configs=eligible[index], observed_feedback=self.feedback_summary(index)))
            payload = dict(trajectory_count=len(trajectories), allowed_configs={name: CONFIGS[name] for name in self.allowed}, trajectories=trajectories)
            self.ids = ['release'] * len(rows)
            if self.mode == 'llm' and trajectories:
                try:
                    response_ids = self.call(payload)
                    if len(response_ids) != len(trajectories) or any(name not in item['eligible_configs'] for name, item in zip(response_ids, trajectories)):
                        raise ValueError('Invalid live or cached action')
                    for item, name in zip(trajectories, response_ids):
                        self.ids[int(item['trajectory_id'])] = name
                except Exception:
                    self.ids = ['release'] * len(rows)
                    self.stats['failures'] += 1
            elif self.mode == 'retrieval':
                for item in trajectories:
                    self.ids[int(item['trajectory_id'])] = max(item['eligible_configs'],
                        key=lambda name: sum(case['advantages'][name] / (1 + case['distance']) for case in item['offline_cases']))
        selected = original_idx.clone()
        for index, name in enumerate(self.ids):
            if name not in eligible[index]:
                self.ids[index] = name = 'release'
            selected[index] = options[name][index]
        original_points = torch.gather(candidates, 1, original_idx.unsqueeze(-1).expand(-1, -1, candidates.size(-1)))
        selected_points = torch.gather(candidates, 1, selected.unsqueeze(-1).expand(-1, -1, candidates.size(-1)))
        self.stats['decisions'] += len(rows)
        self.stats['changed'] += int((selected != original_idx).sum())
        self.stats['changed_points'] += int((selected_points != original_points).any(-1).sum())
        self.stats['opportunities'] += sum(any(int(chosen[index, 0]) != int(original_idx[index, 0]) for chosen in options.values()) for index in range(len(rows)))
        self.stats['no_supported_action'] += sum(len(names) == 1 for names in eligible)
        self.stats['nonrelease'] += sum(name != 'release' for name in self.ids)
        if self.active_record is not None:
            self.active_record(dict(step=self.steps, configs=list(self.ids), original=original_idx.tolist(),
                selected=selected.tolist(), eligible_configs=eligible, current=rows,
                retrieved_case_ids=[[case['id'] for case in cases] for cases in retrieved],
                feedback=[self.feedback_summary(index) for index in range(len(rows))]))
        self.pending = list(self.ids)
        self.steps += 1
        return selected
