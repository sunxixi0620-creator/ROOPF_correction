"""Budget-aware multi-agent control for ROOPF operator scheduling."""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import torch
import torch.nn as nn


EPS = 1e-8


class BudgetAwareAgenticController(nn.Module):
    """Fuse exploration, exploitation, recovery, and safety agent votes.

    The controller only reweights the cheap candidate pool. It never calls the
    true objective, so enabling it does not change ROOPF's evaluation budget.
    """

    def __init__(
        self,
        operator_names: Sequence[str],
        warmup_evaluations: int = 6,
        minimum_multiplier: float = 0.55,
        maximum_multiplier: float = 1.80,
    ) -> None:
        super().__init__()
        expected = [
            "amortized_elite",
            "current_to_pbest_archive",
            "covariance_elite",
            "trust_region",
            "coordinate_pattern",
            "opposition_restart",
        ]
        if list(operator_names) != expected:
            raise ValueError("The agentic controller expects the released six-operator ROOPF bank.")
        self.operator_names = list(operator_names)
        self.warmup_evaluations = int(warmup_evaluations)
        self.minimum_multiplier = float(minimum_multiplier)
        self.maximum_multiplier = float(maximum_multiplier)

        self.register_buffer(
            "exploration_vote",
            torch.tensor([0.85, 1.00, 1.35, 0.90, 1.10, 1.80], dtype=torch.float32),
        )
        self.register_buffer(
            "exploitation_vote",
            torch.tensor([1.20, 1.45, 1.05, 1.40, 1.20, 0.45], dtype=torch.float32),
        )
        self.register_buffer(
            "recovery_vote",
            torch.tensor([0.75, 0.95, 1.20, 1.25, 1.05, 2.00], dtype=torch.float32),
        )

    def forward(
        self,
        candidate_prior: torch.Tensor,
        candidate_ops: torch.Tensor,
        op_seen: torch.Tensor,
        op_admitted: torch.Tensor,
        op_best: torch.Tensor,
        remaining: float,
        stagnation: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if candidate_ops.dim() == 1:
            candidate_ops = candidate_ops.view(1, -1).expand(candidate_prior.size(0), -1)
        candidate_ops = candidate_ops.to(device=candidate_prior.device, dtype=torch.long)
        dtype = candidate_prior.dtype
        device = candidate_prior.device

        remaining_tensor = torch.full(
            (candidate_prior.size(0), 1),
            float(remaining),
            device=device,
            dtype=dtype,
        )
        stagnation_tensor = stagnation.view(-1, 1).to(device=device, dtype=dtype).clamp(0.0, 1.0)
        explore_weight = 0.15 + 0.85 * remaining_tensor
        exploit_weight = 0.25 + 0.75 * (1.0 - remaining_tensor)
        recovery_weight = 1.50 * stagnation_tensor
        vote_total = (explore_weight + exploit_weight + recovery_weight).clamp_min(EPS)

        exploration = self.exploration_vote.to(device=device, dtype=dtype).view(1, -1)
        exploitation = self.exploitation_vote.to(device=device, dtype=dtype).view(1, -1)
        recovery = self.recovery_vote.to(device=device, dtype=dtype).view(1, -1)
        role_vote = (
            explore_weight * exploration
            + exploit_weight * exploitation
            + recovery_weight * recovery
        ) / vote_total

        evidence = op_seen / (op_seen + float(self.warmup_evaluations))
        admitted_rate = (op_admitted + 1.0) / (op_seen + 2.0)
        best_rate = (op_best + 0.5) / (op_seen + 2.0)
        empirical_quality = 0.35 * admitted_rate + 0.65 * (2.0 * best_rate)
        empirical_vote = 1.0 + evidence * (empirical_quality - 0.75)

        total_seen = op_seen.sum(dim=1, keepdim=True)
        uncertainty_bonus = torch.sqrt(torch.log(total_seen + 2.0) / (op_seen + 1.0))
        uncertainty_vote = 1.0 + 0.12 * remaining_tensor * uncertainty_bonus.clamp(max=2.0)

        proven_unproductive = (
            (op_seen >= float(self.warmup_evaluations))
            & (op_admitted <= 0.0)
            & (op_best <= 0.0)
        )
        safety_vote = torch.where(
            proven_unproductive,
            torch.full_like(op_seen, 0.70),
            torch.ones_like(op_seen),
        )

        operator_multiplier = (role_vote * empirical_vote * uncertainty_vote * safety_vote).clamp(
            self.minimum_multiplier,
            self.maximum_multiplier,
        )
        candidate_multiplier = torch.gather(operator_multiplier, 1, candidate_ops)
        adapted_prior = candidate_prior * candidate_multiplier
        adapted_prior = adapted_prior / adapted_prior.sum(dim=1, keepdim=True).clamp_min(EPS)

        diagnostics = {
            "operator_weights": operator_multiplier.detach(),
            "role_vote": role_vote.detach(),
            "empirical_vote": empirical_vote.detach(),
            "safety_vote": safety_vote.detach(),
        }
        return adapted_prior, diagnostics
