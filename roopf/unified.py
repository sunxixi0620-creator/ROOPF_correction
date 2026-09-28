"""One explicit research candidate. No benchmark-name-dependent decisions.

Formula and fixed constants are preregistered in UNIFIED_REVISION_PROTOCOL.md.
This does not implement or imply a formal safety/non-degradation guarantee.
"""
import math
import time
import torch
from .model import ROOPFOptimizer, EPS, _sort_pop, _get_bounds, _normalize_x
from .residual_features import residual_features

VARIANTS = ('full', 'no_residual', 'score_only', 'late_full', 'late_no_residual', 'anchor_only')


class UnifiedOptimizer(ROOPFOptimizer):
    def configure_unified(self, variant='full'):
        if variant not in VARIANTS:
            raise ValueError(variant)
        self.variant = variant
        # These flags affect reusable candidate generation, which must be blind
        # to benchmark labels. The inherited task-dependent forward is not used.
        self.ablation.difference_update({'local_op_boost', 'productive_op_router'})
        self.teacher_observer = None
        self.input_observer = None
        self.decisions = []
        return self

    def inputs_and_scores(self, x, fitness, archive_x, archive_y, candidates,
                          ops, prior, problem, raw, remaining, used, stagnation):
        mu, sigma = self.surrogate.predict(archive_x, archive_y, candidates, problem,
                                           distance_scale=.03)
        scale = archive_y.std(1, keepdim=True, unbiased=False).clamp_min(EPS)
        center = archive_y.mean(1, keepdim=True)
        lb, ub = _get_bounds(problem, x)
        cx = _normalize_x(candidates, lb, ub)
        ax = _normalize_x(archive_x, lb, ub)
        # The actual best archive point, independent of archive insertion order.
        best_idx = archive_y.argmin(1)
        best = ax[torch.arange(len(x), device=x.device), best_idx].unsqueeze(1)
        trust = (1-remaining)**2*(1-.7*stagnation[:, None]) * torch.norm(cx-best, dim=-1)/math.sqrt(x.shape[-1])
        novelty = (.15+.35*remaining+.35*stagnation[:, None])*torch.cdist(cx, ax).amin(2)
        score = (mu-center)/scale - .3125*sigma/scale + trust - novelty - .03*prior.clamp_min(EPS).log()
        features = residual_features(self.router_feature_names, self.op_names,
            candidate_ops=ops, state_raw=raw, fitness=fitness, score=score,
            mu=mu, sigma=sigma, remaining=remaining, used=used, stagnation=stagnation)
        normalized = (features-self.router_mean.view(1, 1, -1))/self.router_std.view(1, 1, -1)
        if self.input_observer is not None:
            self.input_observer(features.detach().clone(), normalized.detach().clone(), ops.detach().clone(), self.evalnum)
        active = self.variant in ('full', 'late_full', 'score_only')
        if active:
            prob = self.router_model(normalized.reshape(-1, normalized.shape[-1])).reshape(score.shape)
            prob = torch.sigmoid(prob)
            signal = ((prob-prob.mean(1, keepdim=True))/prob.std(1, keepdim=True, unbiased=False).clamp_min(1e-6)).clamp(-2.5, 2.5)
            corrected = score-.008*signal
        else:
            corrected = score
        return features, mu, scale, score, corrected

    @torch.no_grad()
    def forward(self, x, problem):
        self.evalnum = 0
        self.decisions = []
        self.teacher_points = 0
        fitness = problem.calfitness(x)
        self.evalnum += x.shape[1]
        x, fitness = _sort_pop(problem.repaire(x), fitness)
        archive_x, archive_y = x, fitness
        best_before = fitness[:, 0]
        no_improve = torch.zeros_like(best_before)
        memory = x.new_zeros((len(x), self.num_ops))
        errors, trails, all_points = [], [], []
        portfolio_seen = torch.zeros_like(best_before)
        for step in range((self.MaxNFE-self.popsize+1)//2):
            rest = min(2, self.MaxNFE-self.evalnum)
            if rest <= 0:
                break
            remaining = (self.MaxNFE-self.evalnum)/self.MaxNFE
            used = self.evalnum/self.MaxNFE
            stagnation = (no_improve/max(1., (self.MaxNFE-self.popsize)/2)).clamp(0, 1)
            anchor = self._baseline_candidates(x, fitness, problem, rest)
            if self.variant == 'anchor_only' or rest == 1:
                chosen = anchor
                chosen_ops = torch.full((len(x), rest), -2, dtype=torch.long, device=x.device)
            else:
                state, raw = self.state_encoder(x, fitness, problem, remaining, used, stagnation)
                pool, ids, priors = self._candidate_pool(x, fitness, problem, state, raw, memory)
                candidates = torch.cat((anchor, pool), 1)
                ops = torch.cat((torch.full((len(x), 2), -2, dtype=torch.long, device=x.device),
                                 ids[None].expand(len(x), -1)), 1)
                prior = torch.cat((torch.full((len(x), 2), 1/self.num_ops, device=x.device), priors), 1)
                features, mu, scale, base_score, score = self.inputs_and_scores(x, fitness, archive_x,
                    archive_y, candidates, ops, prior, problem, raw, remaining, used, stagnation)
                # Single score definition; the first anchor retains its slot.
                proposed = score[:, 1:].argmin(1, keepdim=True)+1
                no_residual = base_score[:, 1:].argmin(1, keepdim=True)+1
                if errors:
                    error = torch.cat(errors[-8:], 1).mean(1, keepdim=True)
                else:
                    error = torch.full_like(scale, 1.)
                margin = .10+.25*error
                eligible = not self.variant.startswith('late_') or used >= .70
                ready = len(errors) >= 4  # eight previously paid prediction outcomes
                raw_gap = score[:, 1:2]-score.gather(1, proposed)
                accepted = (proposed >= 2) & eligible
                if self.variant != 'score_only':
                    accepted &= ready & (raw_gap > margin)
                index = torch.where(accepted, proposed, torch.ones_like(proposed))
                nores_gap = base_score[:, 1:2]-base_score.gather(1, no_residual)
                nores_accept = (no_residual >= 2) & eligible
                if self.variant != 'score_only':
                    nores_accept &= ready & (nores_gap > margin)
                nores_index = torch.where(nores_accept, no_residual, torch.ones_like(no_residual))
                chosen_idx = torch.cat((torch.zeros_like(index), index), 1)
                chosen = candidates.gather(1, chosen_idx[:, :, None].expand(-1, -1, x.shape[-1]))
                chosen_ops = ops.gather(1, chosen_idx)
                chosen_mu = mu.gather(1, chosen_idx)
                if self.teacher_observer is not None:
                    # Pure offline diagnostics: no teacher result is fed back.
                    truth = problem.diagnostic_fitness(candidates)
                    self.teacher_points += candidates.shape[0]*candidates.shape[1]
                    self.teacher_observer(features.detach().cpu(), truth.detach().cpu(),
                        fitness[:, 0].detach().cpu(), self.evalnum)
                for bi in range(len(x)):
                    self.decisions.append(dict(step=step, batch=bi, eval_before=self.evalnum,
                        eligible=bool(eligible), evidence_ready=bool(ready),
                        proposed=int(proposed[bi, 0]), selected=int(index[bi, 0]),
                        accepted=bool(accepted[bi, 0]), margin=float(margin[bi, 0]),
                        residual_active=self.variant in ('full', 'late_full', 'score_only'),
                        residual_changed=bool(index[bi, 0] != nores_index[bi, 0]),
                        portfolio_seen=int(portfolio_seen[bi])))
            values = torch.cat([problem.calfitness(chosen[:, i:i+1]) for i in range(rest)], 1)
            self.evalnum += rest
            if self.variant != 'anchor_only' and rest == 2:
                errors.append(((values-chosen_mu).abs()/scale).clamp_max(10))
            portfolio_seen += (chosen_ops >= 0).sum(1)
            reward = torch.where(values < fitness[:, :1]-1e-12, 1.,
                       torch.where(values < fitness[:, -1:]-1e-12, .25, -.05))
            memory *= .86
            for i in range(rest):
                mask = chosen_ops[:, i] >= 0
                memory[torch.arange(len(x), device=x.device)[mask], chosen_ops[mask, i]] += reward[mask, i]
            memory.clamp_(-2, 3)
            archive_x = torch.cat((archive_x, chosen), 1)
            archive_y = torch.cat((archive_y, values), 1)
            if archive_x.shape[1] > 256:
                bx, by = _sort_pop(archive_x, archive_y)
                archive_x = torch.cat((bx[:, :100], archive_x[:, -128:]), 1)
                archive_y = torch.cat((by[:, :100], archive_y[:, -128:]), 1)
            x, fitness = _sort_pop(torch.cat((x, chosen), 1), torch.cat((fitness, values), 1))
            x, fitness = x[:, :self.popsize], fitness[:, :self.popsize]
            no_improve = torch.where(fitness[:, 0] < best_before-1e-12, 0., no_improve+1)
            best_before = fitness[:, 0]
            trails.append(best_before[:, None]); all_points.append(chosen)
        self.trail = torch.cat(trails, 1)
        self.all_k_candidates = torch.cat(all_points, 1)
        return x, self.trail, self.evalnum, self.all_k_candidates


def build_unified(anchor, residual, variant='full', *, device='cpu', dim=10, budget=300):
    device = torch.device(device)
    with torch.random.fork_rng(devices=[device.index or 0] if device.type == 'cuda' else []):
        torch.manual_seed(20260630)
        opt = UnifiedOptimizer(dim=dim, hidden_dim=200, popSize=100, max_nfe=budget,
            k_nums=2, pool_per_op=6, surrogate_members=5, ablation='roopf',
            baseline_ckpt=str(anchor), router_ckpt=str(residual), router_weight=.008)
    return opt.to(device).eval().configure_unified(variant)
