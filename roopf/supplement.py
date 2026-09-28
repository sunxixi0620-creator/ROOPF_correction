"""Instrumentation and explicitly named ablations of the frozen final ROOPF.

No objective is queried by ordinary observers. Optional counterfactual calls use
an independent diagnostic counter and never update the optimizer's state.
"""
from collections import defaultdict
from contextlib import contextmanager
import time

import torch

from .model import ROOPFOptimizer, EPS, _ensure_fitness_2d, _sort_pop, _repair


RESIDUAL_FLAGS = {'learned_router', 'residual_safety_gate', 'protected_residual_selector',
                  'pool_rank_router', 'router_veto', 'weak_router_veto'}
VARIANTS = ('full', 'anchor_only', 'no_residual', 'score_only', 'no_proxy')


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class SupplementOptimizer(ROOPFOptimizer):
    def configure(self, variant='full', warmup=0.70, diagnostic_stride=0):
        if variant not in VARIANTS:
            raise ValueError(variant)
        self.variant = variant
        self.ablation.discard('structured_system_lock')
        if variant == 'no_residual':
            self.ablation.difference_update(RESIDUAL_FLAGS)
        if variant == 'score_only':
            self.ablation.discard('baseline_safe_selector')
        if variant == 'no_proxy':
            self.ablation.add('no_surrogate')
        # Use the original literal at the default boundary to preserve exactness.
        self.intervention_remaining = 0.30 if warmup == 0.70 else round(1.0 - warmup, 12)
        self.diagnostic_stride = diagnostic_stride
        self.supplement_gate_observer = self.observe_gate
        self.log_candidates = True
        self.log_pool_candidates = False
        self.timings = defaultdict(float)
        self.decisions = []
        self.pool_context = None
        self.pool_router_active = False
        self.selection_call = 0
        return self

    @contextmanager
    def measured(self, name):
        sync()
        start = time.perf_counter()
        try:
            yield
        finally:
            sync()
            self.timings[name] += time.perf_counter() - start

    def _candidate_pool(self, *args, **kwargs):
        with self.measured('candidate_pool_seconds'):
            return super()._candidate_pool(*args, **kwargs)

    def _baseline_candidates(self, *args, **kwargs):
        with self.measured('anchor_seconds'):
            return super()._baseline_candidates(*args, **kwargs)

    def _select_by_acquisition(self, *args, **kwargs):
        with self.measured('ranking_seconds'):
            if self.variant == 'no_proxy':
                # The release's structural score adds stochastic tie-breaking.
                # Isolate it so disabling a proxy cannot shift candidate RNG.
                devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
                with torch.random.fork_rng(devices=devices):
                    torch.manual_seed(490000 + self.evalnum * 10 + self.selection_call)
                    result = super()._select_by_acquisition(*args, **kwargs)
            else:
                result = super()._select_by_acquisition(*args, **kwargs)
        if args[2].shape[1] == self.num_ops * self.pool_per_op:
            self.pool_context = (args, kwargs)
            self.pool_router_active = self.last_router_prob is not None
        self.selection_call = (self.selection_call + 1) % 2
        return result

    @staticmethod
    def protected_index(score, prob, scale, seen, best_count, with_residual=True):
        """Exact default unlocked gate, for a single competing anchor at index 0."""
        chosen = score.topk(k=1, dim=1, largest=False).indices
        chosen_score = score.gather(1, chosen)
        portfolio = chosen >= 1
        safe = portfolio & (chosen_score < score[:, :1] - .10 * scale)
        if with_residual and prob is not None:
            std = prob.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-6)
            mean = prob.mean(dim=1, keepdim=True)
            base_signal = (prob[:, :1] - mean) / std
            chosen_signal = (prob.gather(1, chosen) - mean) / std
            veto = portfolio & (chosen_signal + .35 < base_signal) & (chosen_signal < -.25)
            veto &= (seen.view(-1, 1) >= 8) & (best_count.view(-1, 1) <= 0)
            support = (~portfolio) | (chosen_signal > base_signal + .55) | (chosen_signal > 1.)
            rescue = portfolio & support & (chosen_score < score[:, :1] - .04 * scale)
            safe = (safe & ~veto) | rescue
        return torch.where(safe, chosen, torch.zeros_like(chosen))

    def without_residual_decision(self, state):
        """Recompute both shortlist and gate at the same state; no true calls."""
        saved_flags = self.ablation.copy()
        names = ('last_acquisition', 'last_surrogate_mu', 'last_surrogate_sigma', 'last_router_prob')
        saved = {name: getattr(self, name) for name in names}
        args, kwargs = self.pool_context
        devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
        try:
            self.ablation.difference_update(RESIDUAL_FLAGS)
            with torch.random.fork_rng(devices=devices):
                idx = super()._select_by_acquisition(*args, **kwargs)
                pool = args[2]
                shortlist = pool.gather(1, idx.unsqueeze(-1).expand(-1, -1, pool.shape[-1]))
                combined = torch.cat([state['baseline_cand'][:, 1:2], shortlist], dim=1)
                ops = torch.cat([torch.full_like(idx[:, :1], -2), state['op_ids'][idx]], dim=1)
                super()._select_by_acquisition(state['archive_x'], state['archive_y'], combined,
                    torch.ones(combined.shape[:2], device=combined.device), state['problem'],
                    state['remaining'], state['stagnation'], 1, candidate_ops=ops,
                    state_raw=state['state_raw'], fitness=state['fitness'], used=state['used'])
                chosen = self.protected_index(self.last_acquisition, None,
                    state['archive_y'].std(dim=1, keepdim=True, unbiased=False).clamp_min(EPS),
                    state['portfolio_seen'], state['portfolio_best_count'], False)
                return combined.gather(1, chosen.unsqueeze(-1).expand(-1, -1, combined.shape[-1]))
        finally:
            self.ablation = saved_flags
            for name, value in saved.items():
                setattr(self, name, value)

    def observe_gate(self, s):
        with self.measured('observer_seconds'):
            score, prob, chosen = s['combined_score'], s['combined_router_prob'], s['extra_idx']
            raw = s['proposed_extra_idx']
            scale = s['archive_y'].std(dim=1, keepdim=True, unbiased=False).clamp_min(EPS)
            if self.variant != 'score_only':
                expected = self.protected_index(score, prob, scale, s['portfolio_seen'],
                    s['portfolio_best_count'], 'residual_safety_gate' in self.ablation)
                assert torch.equal(expected, chosen), 'Gate observer does not match deployed gate'
            selected_point = s['combined'].gather(1, chosen.unsqueeze(-1).expand(-1, -1, s['x'].shape[-1]))
            compare = self.variant == 'full' and self.diagnostic_stride > 0 and len(s['all_candidates']) % self.diagnostic_stride == 0
            no_residual_point = None
            if compare:
                no_residual_point = self.without_residual_decision(s) if (self.pool_router_active or prob is not None) else selected_point
            diagnostic_values = None
            if compare:
                proposed_point = s['combined'].gather(1, raw.unsqueeze(-1).expand(-1, -1, s['x'].shape[-1]))
                diagnostic_values = s['problem'].diagnostic_fitness(torch.cat([s['combined'][:, :1], proposed_point], dim=1))
            for bi in range(score.shape[0]):
                admitted = int(chosen[bi, 0]) >= 1
                proposed = int(raw[bi, 0]) >= 1
                reason = 'score_only' if self.variant == 'score_only' else ('accepted' if admitted else ('anchor_ranked_first' if not proposed else 'protected_rejection'))
                row = {'step': len(s['all_candidates']), 'batch': bi, 'eval_before': self.evalnum,
                    'proposed_portfolio': int(proposed), 'accepted_portfolio': int(admitted),
                    'reason': reason, 'pool_residual_active': int(self.pool_router_active),
                    'gate_residual_active': int(prob is not None),
                    'archive_scale': float(scale[bi, 0]), 'anchor_score': float(score[bi, 0]),
                    'proposed_score': float(score[bi, int(raw[bi, 0])]),
                    'diagnostic_sampled': int(compare), 'residual_decision_changed': '',
                    'diagnostic_anchor_value': '', 'diagnostic_proposed_value': ''}
                if compare:
                    row['residual_decision_changed'] = int(not torch.equal(selected_point[bi], no_residual_point[bi]))
                    row['diagnostic_anchor_value'] = float(diagnostic_values[bi, 0])
                    row['diagnostic_proposed_value'] = float(diagnostic_values[bi, 1])
                self.decisions.append(row)

    @torch.no_grad()
    def forward(self, x, problem):
        self.timings.clear()
        self.decisions.clear()
        self.selection_call = 0
        if self.variant != 'anchor_only':
            return super().forward(x, problem)
        # Same anchor, repair, sorting, update and objective-call order; no unused
        # portfolio/proxy computation, so anchor runtime is meaningful.
        self.candidate_samples = []
        self.evalnum = 0
        fitness = _ensure_fitness_2d(problem.calfitness(x))
        self.evalnum += x.shape[1]
        x, fitness = _sort_pop(_repair(problem, x), fitness)
        trails, points = [], []
        while self.evalnum < self.MaxNFE:
            rest = min(self.k_nums, self.MaxNFE - self.evalnum)
            cand = self._baseline_candidates(x, fitness, problem, rest)
            values = torch.cat([_ensure_fitness_2d(problem.calfitness(cand[:, i:i+1])) for i in range(rest)], dim=1)
            for bi in range(x.shape[0]):
                for ci in range(rest):
                    self.candidate_samples.append({'step': len(trails), 'batch': bi, 'candidate_slot': ci,
                        'eval_before': self.evalnum, 'is_baseline': 1, 'candidate_fit': float(values[bi, ci]),
                        'fitness_best_before': float(fitness[bi, 0]),
                        'improved_best': int(values[bi, ci] < fitness[bi, 0] - 1e-12),
                        'admitted': int(values[bi, ci] < fitness[bi, -1] - 1e-12)})
            self.evalnum += rest
            points.append(cand)
            x, fitness = _sort_pop(torch.cat([x, cand], 1), torch.cat([fitness, values], 1))
            x, fitness = x[:, :self.popsize], fitness[:, :self.popsize]
            trails.append(fitness[:, :1])
        self.trail = torch.cat(trails, 1)
        self.all_k_candidates = torch.cat(points, 1)
        return x, self.trail, self.evalnum, self.all_k_candidates
