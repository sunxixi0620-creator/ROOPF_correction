"""Strict R10 component ablations. Generated once; frozen before live runs.

No-online-generators removes all six runtime generators and the extra native
proposal. Only original Anchor candidates and an actual Anchor temperature
variant can exist; there is no padding to the Full candidate count.
The frozen objective-boundary RNG schedule preserves common random numbers
without calling the removed generators, including BBOB20 stochastic semantics.
"""
import copy
import math
import torch
import numpy as np

from roopf.model import (ROOPFOptimizer,EPS,_ensure_fitness_2d,_sort_pop,_repair,
    _is_structured_system_problem,_is_embedded_high_dim_problem,
    _is_local_proxy_friendly_problem,_is_hpo_problem)
from roopf.fusion_rules import protected_indices
from roopf.current_evidence_agent import CurrentEvidenceAgent,SYSTEM,average_ranks,POLICY_ID
from roopf.evidence_fusion_agent import PROFILES,ACTION_NAMES,digest,request_context_digest
from roopf.policy_api_r10 import request_policy


class NoOnlineGeneratorsOptimizer(ROOPFOptimizer):
    def _anchor_round(self,x,fitness,problem,rest,archive_x,archive_y,state_raw,
                      remaining,stagnation,used,seen,best_count,admit_count,no_admit):
        baseline=self._baseline_candidates(x,fitness,problem,rest)
        assert baseline is not None and baseline.shape[1]==rest
        operations=torch.full((x.size(0),rest),-2,dtype=torch.long)
        control=self._generation_control
        if control is None:
            return baseline,operations,None,None,None,None
        assert rest==2 and not _is_structured_system_problem(problem)
        points=baseline
        if bool((control.temperature!=1).any()):
            points=torch.cat([baseline,control.extra],1)
        candidate_ops=torch.full(points.shape[:2],-2,dtype=torch.long)
        with torch.random.fork_rng(devices=[]):
            self._select_by_acquisition(archive_x,archive_y,points,
                torch.ones(points.shape[:2]),problem,remaining,stagnation,1,
                candidate_ops=candidate_ops,state_raw=state_raw,fitness=fitness,used=used,
                use_learned_router=True)
        control.protected_mu=self.last_surrogate_mu[:,:1].clone()
        probability=self.last_router_prob
        scale=archive_y.std(1,keepdim=True,unbiased=False).clamp_min(EPS)
        signal=torch.zeros_like(self.last_acquisition)
        if probability is not None:
            signal=((probability-probability.mean(1,keepdim=True))/probability.std(1,keepdim=True,unbiased=False).clamp_min(1e-6)).clamp(-2.5,2.5)
        online=self.last_acquisition+self.last_residual_weight*scale*signal
        competing=points[:,1:];ops=candidate_ops[:,1:]
        p=None if probability is None else probability[:,1:]
        combined_signal=torch.zeros_like(online[:,1:])
        if p is not None:
            combined_signal=((p-p.mean(1,keepdim=True))/p.std(1,keepdim=True,unbiased=False).clamp_min(1e-6)).clamp(-2.5,2.5)
        score=online[:,1:]-self.last_residual_weight*scale*combined_signal
        def selection_rule(acquisition,proposed,margin=.10,soft_margin=.04,online_only=False):
            ablation=set(self.ablation)-({'residual_safety_gate','router_veto','weak_router_veto'} if online_only else set())
            return protected_indices(acquisition,p,scale,proposed,1,ablation=ablation,
                portfolio_seen=seen,portfolio_best_count=best_count,portfolio_admit_count=admit_count,
                portfolio_no_admit=no_admit,stagnation=stagnation,remaining=remaining,
                local_proxy_friendly=_is_local_proxy_friendly_problem(problem),hpo=_is_hpo_problem(problem),
                embedded=_is_embedded_high_dim_problem(problem),selected_ops=ops[:,1:],op_penalty=None,
                margin=margin,soft_margin=soft_margin)
        original=torch.zeros((x.size(0),1),dtype=torch.long)
        selected=control.choose(original_idx=original,score=score,mu=self.last_surrogate_mu[:,1:],
            sigma=self.last_surrogate_sigma[:,1:],probability=p,state=state_raw,archive_y=archive_y,
            fitness=fitness,candidates=competing,candidate_ops=ops,remaining=remaining,problem=problem,
            residual_weight=self.last_residual_weight,selection_rule=selection_rule)
        chosen=competing.gather(1,selected.unsqueeze(-1).expand(-1,-1,x.size(-1)))
        def selected_value(value):
            return torch.cat([value[:,:1],value[:,1:].gather(1,selected)],1)
        control.previous_generation=[dict(online_generators_enabled=False,competing_candidates=competing.size(1),
            extra_origin=control.extra_origin[i],anchor=control.anchor_diagnostics[i],local_scale_effective=False)
            for i in range(x.size(0))]
        return torch.cat([baseline[:,:1],chosen],1),operations,selected_value(self.last_acquisition),selected_value(self.last_surrogate_mu),selected_value(self.last_surrogate_sigma),None if probability is None else selected_value(probability)


    def forward(self, x: torch.Tensor, problem):
        self.evalnum = 0
        self._generation_control = None
        self.trail = None
        self.all_k_candidates = []
        self.gate_history = []
        self.teacher_history = []
        self.fused_history = []
        self.candidate_samples = []
        self.pool_candidate_samples = []
        self.agentic_history = []
        self._reset_origin_stats()
        if self.fusion_agent is not None:
            self.fusion_agent.reset(x.size(0))

        fitness = _ensure_fitness_2d(problem.calfitness(x))
        self.evalnum += x.size(1)
        x, fitness = _sort_pop(_repair(problem, x), fitness)

        archive_x = x
        archive_y = fitness
        best_prev = fitness[:, 0]
        no_improve = torch.zeros_like(best_prev)
        portfolio_no_admit = torch.zeros_like(best_prev)
        portfolio_seen = torch.zeros_like(best_prev)
        portfolio_admit_count = torch.zeros_like(best_prev)
        portfolio_best_count = torch.zeros_like(best_prev)
        portfolio_catastrophic_lock = torch.zeros_like(best_prev, dtype=torch.bool)
        op_seen = torch.zeros((x.size(0), self.num_ops), device=x.device, dtype=x.dtype)
        op_admit = torch.zeros_like(op_seen)
        op_best = torch.zeros_like(op_seen)
        success_logits = torch.zeros((x.size(0), self.num_ops), device=x.device, dtype=x.dtype)
        all_candidates = []

        while self.evalnum < self.MaxNFE:
            rest = min(self.k_nums, self.MaxNFE - self.evalnum)
            if rest <= 0:
                break
            remaining = max(0.0, float(self.MaxNFE - self.evalnum) / float(self.MaxNFE))
            used = min(1.0, float(self.evalnum) / float(self.MaxNFE))
            max_steps = max(1.0, float(self.MaxNFE - self.popsize) / float(self.k_nums))
            stagnation = (no_improve / max_steps).clamp(0.0, 1.0)

            state, state_raw = self.state_encoder(x, fitness, problem, remaining, used, stagnation)
            if self.fusion_agent is not None and hasattr(self.fusion_agent, 'prepare_generation'):
                active = remaining <= 0.30 and rest > 1 and not _is_structured_system_problem(problem)
                self._generation_control = self.fusion_agent if active else None
                if active:
                    self.fusion_agent.prepare_generation(self, x, fitness, archive_y, state_raw, problem, remaining)
            # Preserve the original stream at every objective boundary, without
            # executing any removed generator or creating dummy candidates.
            iteration=(self.evalnum-self.popsize)//self.k_nums
            torch.set_rng_state(self.objective_rng_schedule[1+2*iteration])
            op_prior=x.new_zeros((x.size(0),self.num_ops*self.pool_per_op))
            cand,selected_ops,cand_score,cand_mu,cand_sigma,cand_router_prob = self._anchor_round(
                x,fitness,problem,rest,archive_x,archive_y,state_raw,remaining,stagnation,used,
                portfolio_seen,portfolio_best_count,portfolio_admit_count,portfolio_no_admit)
            fits = []
            for i in range(rest):
                fits.append(_ensure_fitness_2d(problem.calfitness(cand[:, i : i + 1])))
            cand_fit = torch.cat(fits, dim=1)
            self.evalnum += rest
            all_candidates.append(cand)

            threshold = fitness[:, -1].view(-1, 1)
            improved_best = cand_fit < (best_prev.view(-1, 1) - 1e-12)
            admitted = cand_fit < (threshold - 1e-12)
            self._record_origin_stats(selected_ops, improved_best, admitted)
            if self.fusion_agent is not None:
                self.fusion_agent.observe(
                    cand_fit, cand_mu, cand_router_prob, best_prev,
                    selected_ops, archive_y,
                )
            if self.log_candidates:
                self._record_candidate_samples(
                    step_idx=len(all_candidates) - 1,
                    eval_before=self.evalnum - rest,
                    used=used,
                    remaining=remaining,
                    stagnation=stagnation,
                    selected_ops=selected_ops,
                    cand_fit=cand_fit,
                    admitted=admitted,
                    improved_best=improved_best,
                    best_before=best_prev,
                    worst_before=threshold.view(-1),
                    state_raw=state_raw,
                    score=cand_score,
                    mu=cand_mu,
                    sigma=cand_sigma,
                    router_prob=cand_router_prob,
                )
            if "adaptive_portfolio_lock" in self.ablation or "adaptive_proxy_disable" in self.ablation:
                portfolio_mask = selected_ops >= 0
                portfolio_eval = portfolio_mask.any(dim=1)
                portfolio_success = (portfolio_mask & admitted).any(dim=1)
                portfolio_no_admit = torch.where(
                    portfolio_success,
                    torch.zeros_like(portfolio_no_admit),
                    torch.where(portfolio_eval, portfolio_no_admit + 1.0, portfolio_no_admit),
                )
            if (
                "aggressive_takeover" in self.ablation
                or "admitted_no_best_lock" in self.ablation
                or "adaptive_proxy_disable" in self.ablation
            ):
                portfolio_mask = selected_ops >= 0
                portfolio_seen = portfolio_seen + portfolio_mask.sum(dim=1).to(portfolio_seen.dtype)
                portfolio_admit_count = portfolio_admit_count + (portfolio_mask & admitted).sum(dim=1).to(portfolio_admit_count.dtype)
                portfolio_best_count = portfolio_best_count + (portfolio_mask & improved_best).sum(dim=1).to(portfolio_best_count.dtype)
            if (
                self.agentic_controller is not None
                or ("highdim_adaptive_op_router" in self.ablation and _is_embedded_high_dim_problem(problem))
                or (
                    "adaptive_proxy_disable" in self.ablation
                    and not _is_local_proxy_friendly_problem(problem)
                    and not _is_structured_system_problem(problem)
                    and not _is_embedded_high_dim_problem(problem)
                )
            ):
                batch = torch.arange(x.size(0), device=x.device)
                for ci in range(rest):
                    valid_ops = selected_ops[:, ci] >= 0
                    if valid_ops.any():
                        idx = selected_ops[valid_ops, ci]
                        op_seen[batch[valid_ops], idx] += 1.0
                        op_admit[batch[valid_ops], idx] += admitted[valid_ops, ci].to(op_admit.dtype)
                        op_best[batch[valid_ops], idx] += improved_best[valid_ops, ci].to(op_best.dtype)
            if (
                (
                    "dynamic_highdim_op_adapter" in self.ablation
                    or "validated_highdim_adapter" in self.ablation
                    or "gated_highdim_adapter" in self.ablation
                    or "gated_rescue_highdim_adapter" in self.ablation
                )
                and _is_embedded_high_dim_problem(problem)
                and self.agentic_controller is None
            ):
                batch = torch.arange(x.size(0), device=x.device)
                for ci in range(rest):
                    valid_ops = selected_ops[:, ci] >= 0
                    if valid_ops.any():
                        idx = selected_ops[valid_ops, ci]
                        op_seen[batch[valid_ops], idx] += 1.0
                        op_admit[batch[valid_ops], idx] += admitted[valid_ops, ci].to(op_admit.dtype)
                        op_best[batch[valid_ops], idx] += improved_best[valid_ops, ci].to(op_best.dtype)
            if "no_success_memory" not in self.ablation:
                success_logits = 0.86 * success_logits
                for ci in range(rest):
                    reward = torch.where(
                        improved_best[:, ci],
                        torch.full_like(best_prev, 1.0),
                        torch.where(admitted[:, ci], torch.full_like(best_prev, 0.25), torch.full_like(best_prev, -0.05)),
                    )
                    batch = torch.arange(x.size(0), device=x.device)
                    valid_ops = selected_ops[:, ci] >= 0
                    if valid_ops.any():
                        success_logits[batch[valid_ops], selected_ops[valid_ops, ci]] += reward[valid_ops]
                success_logits = success_logits.clamp(-2.0, 3.0)

            archive_x = torch.cat([archive_x, cand], dim=1)
            archive_y = torch.cat([archive_y, cand_fit], dim=1)
            if archive_x.size(1) > 256:
                keep_best = min(self.popsize, 128)
                best_x, best_y = _sort_pop(archive_x, archive_y)
                recent_x = archive_x[:, -128:]
                recent_y = archive_y[:, -128:]
                archive_x = torch.cat([best_x[:, :keep_best], recent_x], dim=1)
                archive_y = torch.cat([best_y[:, :keep_best], recent_y], dim=1)

            ext_x = torch.cat([x, cand], dim=1)
            ext_y = torch.cat([fitness, cand_fit], dim=1)
            x, fitness = _sort_pop(ext_x, ext_y)
            x = x[:, : self.popsize]
            fitness = fitness[:, : self.popsize]

            best_now = fitness[:, 0]
            improved = best_now < (best_prev - 1e-12)
            no_improve = torch.where(improved, torch.zeros_like(no_improve), no_improve + 1.0)
            best_prev = torch.minimum(best_prev, best_now)
            trail_step = best_now.view(-1, 1)
            self.trail = trail_step if self.trail is None else torch.cat([self.trail, trail_step], dim=1)

            gate_snapshot = torch.softmax(success_logits, dim=1).detach()
            self.gate_history.append(gate_snapshot)
            self.teacher_history.append(op_prior.view(x.size(0), self.num_ops, self.pool_per_op).mean(dim=2).detach())
            self.fused_history.append(gate_snapshot)

        if all_candidates:
            self.all_k_candidates = torch.cat(all_candidates, dim=1)
        else:
            self.all_k_candidates = torch.empty((x.size(0), 0, x.size(2)), device=x.device, dtype=x.dtype)
        if self.trail is None:
            self.trail = fitness[:, :1]

        self.distill_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        if self.gate_history:
            self.gate_history = torch.stack(self.gate_history, dim=1)
            self.teacher_history = torch.stack(self.teacher_history, dim=1)
            self.fused_history = torch.stack(self.fused_history, dim=1)
        else:
            empty = torch.empty((x.size(0), 0, self.num_ops), device=x.device, dtype=x.dtype)
            self.gate_history = empty
            self.teacher_history = empty
            self.fused_history = empty
        if self.agentic_history:
            self.agentic_history = torch.stack(self.agentic_history, dim=1)
        else:
            self.agentic_history = torch.empty(
                (x.size(0), 0, self.num_ops), device=x.device, dtype=x.dtype
            )
        return x, self.trail, self.evalnum, self.all_k_candidates

class NoOnlineGeneratorsAgent(CurrentEvidenceAgent):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.system_prompt=SYSTEM.replace('All22 actions have offline evidence rows when offline input is enabled.',
            'ALL online generating operators are REMOVED, including the extra native proposal. Only original Anchor and its temperature variant can be evaluated. Local-step changes are inactive aliases. Offline cases are frozen Full cases, not relabeled for this ablation. All22 actions have offline evidence rows when offline input is enabled.')

    def prepare_generation(self,*args,**kwargs):
        super().prepare_generation(*args,**kwargs)
        self.stats['local_scaled_rounds']=0

    def origin(self,index,slot,operation):
        return 'anchor_variant' if slot==1 and self.extra_origin[index]=='anchor_variant' else 'original_anchor'

    def make_extra(self,model,population,fitness,problem,baseline):
        self.protected_point=baseline[:,:1].detach().clone()
        self.original_anchor=baseline[:,1:2].detach().clone()
        with torch.random.fork_rng(devices=[]):
            variant,details=model.baseline_model.generator.competing_proposal(population,problem,fitness,self.temperature)
        changed=self.temperature!=1
        self.extra=torch.where(changed[:,None,None],variant,baseline[:,1:2])
        self.extra_ops=torch.full((population.size(0),1),-2,dtype=torch.long)
        self.extra_origin=['anchor_variant' if flag else 'anchor_alias' for flag in changed.tolist()]
        self.anchor_diagnostics=[dict(temperature=float(self.temperature[i]),entropy=float(details['entropy'][i]),
            repaired=bool(details['repaired'][i]),normalized_distance=float(torch.norm((variant[i]-baseline[i,1:2])/self.width)/math.sqrt(population.size(-1)))) for i in range(population.size(0))]

    def compact_payload(self,payload):
        result=super().compact_payload(payload)
        result['ablation']=dict(online_generators_present=False,extra_native_proposal_present=False,
            local_scale_effective=False,actual_competing_candidates=[t['current_options']['candidate_count'] for t in payload['trajectories']],
            offline_memory='frozen Full cases; not relabeled')
        return result

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
            for slot in range(points.size(1)):
                row = sources.get(self.origin(i, slot, kwargs['candidate_ops'][i, slot]))
                if row:
                    reliability[i, slot] = row['residual_success'] / row['residual_total']
                    source_error[i, slot] = min(2., row['proxy_error_sum'] / max(1, row['proxy_error_n']))
            if self.extra_origin[i] == 'anchor_variant':
                supported[i, -1] = 0
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
                (q[1] - q[0] if len(q)>1 else 0.) / float(self.scale[i]), correlation,
                sum(ranked) / len(ranked) if ranked else .5, len(ranked) / 16]
            features.append(vector)
            summaries.append(dict(applied_generation=self.applied_generation[i], fusion=rows,
                candidate_count=points.size(1), original_selection=int(kwargs['original_idx'][i, 0]), current_objectives_known=False))
        return summaries, features, indices

    def _numeric_choose(self, **kwargs):
        points, original = kwargs['candidates'], kwargs['original_idx']
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
            adjusted = (slot == points.size(1)-1 and self.extra_origin[index] == 'anchor_variant') or (
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
            row = dict(generation=copy.deepcopy(self.generation_context[index]), candidate_count=points.size(1),
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
            selected = self._numeric_choose(**kwargs)
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
