"""Frozen-model coordinate adapters for controlled experiments, not retrained policies."""
import torch
from .model import ROOPFOptimizer, _get_bounds, _repair


class CoordinateOptimizer(ROOPFOptimizer):
    mode = 'legacy'

    @torch.no_grad()
    def _baseline_candidates(self, x, fitness, problem, rest):
        original = super()._baseline_candidates(x, fitness, problem, rest)
        if self.mode not in {'centroid_slot', 'relative_slot'} or rest < 2:
            return original
        elite = x[:, :max(2, x.shape[1] // 5)]
        center = elite.mean(dim=1, keepdim=True)
        if self.mode == 'centroid_slot':
            replacement = center
        else:
            lb, ub = _get_bounds(problem, x)
            width = (ub-lb).view(1,1,-1)
            scale = torch.maximum(elite.std(dim=1, keepdim=True, unbiased=False), width*1e-6)
            normalized = (x-center)/scale

            class LocalDomain:
                def repaire(self, candidate):
                    return torch.maximum(torch.minimum(candidate,(ub.view(1,1,-1)-center)/scale),(lb.view(1,1,-1)-center)/scale)

            # Same frozen model and fitness values. Only coordinates change.
            proposed = self.baseline_model.generator(normalized, LocalDomain(), fitness)
            replacement = center + scale*proposed[:,1:2]
        result = original.clone()
        result[:,1:2] = _repair(problem,replacement)
        return result
