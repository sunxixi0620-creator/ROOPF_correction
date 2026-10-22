from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import subprocess
import time

import numpy as np
import torch


CONFIGS = {
    'release': dict(residual_multiplier=1.0, margin=0.10, soft_margin=0.04),
    'proxy': dict(residual_multiplier=0.0, margin=0.10, soft_margin=0.04),
    'residual': dict(residual_multiplier=2.0, margin=0.10, soft_margin=0.04),
    'cautious': dict(residual_multiplier=1.0, margin=0.16, soft_margin=0.08),
    'balanced': dict(residual_multiplier=1.0, margin=0.06, soft_margin=0.03),
}

SYSTEM = '''你是ROOPF的离线在线证据融合控制器。只能从allowed_configs选择配置。
输入包含已验证的训练函数案例、当前11维状态、Anchor候选与在线Proxy及冻结Residual证据、已完成评价的反馈。
候选0是竞争Anchor；保护Anchor槽由程序单独保留。目标是最小化昂贵目标；当前未评价候选没有真实标签。
离线案例的效益仅为该历史状态下一步效益，不代表终局保证。证据不足选release。
probability是冻结Residual改进支持，sigma是在线Proxy不确定性；它们均不是已知真实结果。
输出严格JSON：{"config_ids":{"0":"release","1":"balanced"}}。
config_ids必须是映射，键恰好覆盖输入的全部trajectory_id；每个值只能是allowed_configs中的配置名。不要解释、Markdown或额外字段。'''


def plain(tensor):
    return tensor.detach().cpu().double().numpy()


def fusion_indices(score, probability, scale, original_idx, residual_weight):
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
        gain = fused[:, :1] - torch.gather(fused, 1, selected)
        support = torch.gather(signal, 1, selected) > torch.maximum(signal[:, :1] + 0.55, torch.ones_like(gain))
        if probability is None or config['residual_multiplier'] == 0:
            support = torch.zeros_like(support)
        allows = (gain > config['margin'] * scale) | (support & (gain > config['soft_margin'] * scale))
        result[name] = torch.where((selected > 0) & allows, selected, torch.zeros((batch, 1), dtype=torch.long, device=score.device))
    return result


def context_rows(state, score, mu, sigma, probability, scale, original_idx, residual_weight):
    values = dict(state=plain(state), score=plain(score / scale),
                  mu=plain((mu - mu[:, :1]) / scale), sigma=plain(sigma / scale))
    probabilities = plain(probability) if probability is not None else None
    rows = []
    for index in range(state.size(0)):
        row = {name: np.round(value[index], 6).tolist() for name, value in values.items()}
        row['probability'] = np.round(probabilities[index], 6).tolist() if probabilities is not None else None
        row['original_index'] = int(original_idx[index, 0])
        row['residual_weight'] = float(residual_weight)
        rows.append(row)
    return rows


class FusionAgent:
    def __init__(self, memory_path, journal, mode='llm', budget=None):
        self.memory = json.loads(Path(memory_path).read_text())
        self.memory_hash = hashlib.sha256(Path(memory_path).read_bytes()).hexdigest()
        self.cases = self.memory['cases']
        self.features = np.asarray([case['state'] for case in self.cases], dtype=float)
        self.feature_scale = np.maximum(self.features.std(0), 0.1)
        self.allowed = self.memory['allowed_configs']
        self.journal = Path(journal)
        self.journal.mkdir(parents=True, exist_ok=True)
        self.mode = mode
        self.budget = budget if budget is not None else {'used': 0, 'limit': 96}
        self.reset(0)

    def reset(self, batch_size):
        self.feedback = [dict(n=0, proxy_error=0.0, residual_error=0.0, improvements=0, portfolio=0) for _ in range(batch_size)]
        self.steps = 0
        self.ids = ['release'] * batch_size
        self.stats = dict(decisions=0, changed=0, requests=0, failures=0, nonrelease=0)
        self.active_record = None

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

    def retrieve(self, state):
        vector = np.asarray(state)
        distances = (((self.features - vector) / self.feature_scale) ** 2).mean(1)
        indices = np.argsort(distances, kind='stable')[:3]
        return [dict(id=self.cases[index]['id'], distance=round(float(distances[index]), 4),
                     state=self.cases[index]['state'], advantages=self.cases[index]['advantages']) for index in indices]

    def call(self, payload):
        data = json.dumps(dict(model='deepseek-v4-flash', temperature=0, max_tokens=2048,
                               response_format={'type': 'json_object'},
                               thinking={'type': 'disabled'},
                               messages=[dict(role='system', content=SYSTEM), dict(role='user', content=SYSTEM + '\nSelect one allowed config per trajectory_id. Return config_ids as an object keyed by every supplied trajectory_id, never as an array.\nINPUT:\n' + json.dumps(payload))])).encode()
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
            content = response_data['choices'][0]['message']['content'].strip()
            if content.startswith('```'):
                content = content.split('\n', 1)[1].rsplit('```', 1)[0]
            parsed = json.loads(content)
            assert set(parsed) == {'config_ids'}
            ids = parsed['config_ids']
            if isinstance(ids, dict):
                expected = [str(index) for index in range(len(payload['trajectories']))]
                assert set(ids) == set(expected), 'trajectory_id集合不匹配'
                ids = [ids[index] for index in expected]
            assert isinstance(ids, list) and len(ids) == len(payload['trajectories'])
            assert all(isinstance(name, str) and name in self.allowed for name in ids)
            record.update(status='ok', config_ids=ids, response=response_data)
            return ids
        except Exception as error:
            record.update(status='failed', error_type=type(error).__name__, error=str(error)[:300])
            raise
        finally:
            record['elapsed_seconds'] = time.monotonic() - started
            target.write_text(json.dumps(record, ensure_ascii=False, indent=2))

    def choose(self, *, original_idx, score, mu, sigma, probability, state, archive_y,
               fitness, candidates, candidate_ops, remaining, problem, residual_weight):
        scale = archive_y.std(1, keepdim=True, unbiased=False).clamp_min(1e-8)
        options = fusion_indices(score, probability, scale, original_idx, residual_weight)
        rows = context_rows(state, score, mu, sigma, probability, scale, original_idx, residual_weight)
        if self.steps % 23 == 0:
            trajectories = []
            for index, row in enumerate(rows):
                cases = self.retrieve(row['state'])
                feedback = dict(self.feedback[index])
                denominator = max(feedback['n'], 1)
                feedback['mean_proxy_error'] = round(feedback.pop('proxy_error') / denominator, 5)
                feedback['mean_residual_error'] = round(feedback.pop('residual_error') / denominator, 5)
                trajectories.append(dict(trajectory_id=str(index), current=row, offline_cases=cases, observed_feedback=feedback))
            payload = dict(trajectory_count=len(trajectories), allowed_configs={name: CONFIGS[name] for name in self.allowed}, trajectories=trajectories)
            if self.mode == 'llm':
                try:
                    self.ids = self.call(payload)
                except Exception:
                    self.ids = ['release'] * len(rows)
                    self.stats['failures'] += 1
            elif self.mode == 'retrieval':
                self.ids = [max(self.allowed, key=lambda name: sum(case['advantages'][name] / (1 + case['distance']) for case in item['offline_cases'])) for item in trajectories]
            else:
                self.ids = ['release'] * len(rows)
        selected = original_idx.clone()
        for index, name in enumerate(self.ids):
            selected[index] = options[name][index]
        self.stats['decisions'] += len(rows)
        self.stats['changed'] += int((selected != original_idx).sum())
        self.stats['nonrelease'] += sum(name != 'release' for name in self.ids)
        if self.active_record is not None:
            self.active_record(dict(step=self.steps, configs=list(self.ids), original=original_idx.tolist(), selected=selected.tolist(), feedback=[dict(row) for row in self.feedback]))
        self.steps += 1
        return selected
