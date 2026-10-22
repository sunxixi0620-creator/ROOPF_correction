from __future__ import annotations

import torch


def protected_indices(acquisition, probability, scale, proposed, base_count, *,
                      ablation, portfolio_seen, portfolio_best_count,
                      portfolio_admit_count, portfolio_no_admit, stagnation,
                      remaining, local_proxy_friendly, hpo, embedded,
                      selected_ops, op_penalty, margin=0.10, soft_margin=0.04):
    base_acq, base_idx = acquisition[:, :base_count].min(dim=1, keepdim=True)
    chosen_acq = torch.gather(acquisition, 1, proposed)
    chosen_is_portfolio = proposed >= base_count
    safe = chosen_is_portfolio & (chosen_acq < base_acq - margin * scale)
    if "residual_safety_gate" in ablation and probability is not None:
        base_prob = probability[:, :base_count].max(dim=1, keepdim=True).values
        chosen_prob = torch.gather(probability, 1, proposed)
        prob_std = probability.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-6)
        prob_mean = probability.mean(dim=1, keepdim=True)
        base_signal = (base_prob - prob_mean) / prob_std
        chosen_signal = (chosen_prob - prob_mean) / prob_std
        residual_blocks = chosen_is_portfolio & ((chosen_signal + 0.35 < base_signal) & (chosen_signal < -0.25))
        residual_blocks = residual_blocks & (portfolio_seen.view(-1, 1) >= 8.0) & (portfolio_best_count.view(-1, 1) <= 0.0)
        residual_supports = (~chosen_is_portfolio) | ((chosen_signal > base_signal + 0.55) | (chosen_signal > 1.00))
        residual_rescue = chosen_is_portfolio & residual_supports & (chosen_acq < base_acq - soft_margin * scale)
        safe = (safe & (~residual_blocks)) | residual_rescue
        if "adaptive_proxy_disable" in ablation and not local_proxy_friendly:
            safe = safe & (~(chosen_is_portfolio & (chosen_acq < -1.0e3)))
    if "router_veto" in ablation and probability is not None:
        base_prob = probability[:, :base_count].max(dim=1, keepdim=True).values
        chosen_prob = torch.gather(probability, 1, proposed)
        safe = safe & ((~chosen_is_portfolio) | (chosen_prob > base_prob + 0.03))
    if "weak_router_veto" in ablation and probability is not None:
        base_prob = probability[:, :base_count].max(dim=1, keepdim=True).values
        chosen_prob = torch.gather(probability, 1, proposed)
        router_blocks = chosen_is_portfolio & (chosen_prob < 0.35) & (chosen_prob + 0.05 < base_prob)
        safe = safe & (~router_blocks)
    if "admitted_no_best_lock" in ablation and not hpo and not embedded:
        lock = chosen_is_portfolio & (portfolio_best_count.view(-1, 1) <= 0.0)
        lock = lock & (portfolio_seen.view(-1, 1) >= 12.0) & (portfolio_admit_count.view(-1, 1) >= 3.0)
        safe = safe & (~(lock & torch.full_like(lock, float(remaining) < 0.20, dtype=torch.bool)))
    if "aggressive_takeover" in ablation and not hpo and not embedded:
        stag = stagnation.view(-1, 1).to(device=acquisition.device, dtype=acquisition.dtype)
        late = torch.full(stag.shape, float(remaining) < 0.18, device=acquisition.device, dtype=torch.bool)
        admit_rate = (portfolio_admit_count / portfolio_seen.clamp_min(1.0)).view(-1, 1)
        has_signal = (portfolio_seen.view(-1, 1) >= 6.0) & (admit_rate > 0.12)
        takeover = chosen_is_portfolio & has_signal & (late | (stag > 0.04)) & (chosen_acq < base_acq + 0.05 * scale)
        safe = safe | takeover
    result = torch.where(safe, proposed, base_idx)
    if "adaptive_portfolio_lock" in ablation:
        result = torch.where(portfolio_no_admit.view(-1, 1) >= 8.0, base_idx, result)
    if op_penalty is not None:
        chosen_portfolio = result >= base_count
        port_idx = (result - base_count).clamp_min(0)
        chosen_op = torch.gather(selected_ops, 1, port_idx)
        penalized = torch.gather(op_penalty > 0.0, 1, chosen_op.clamp_min(0))
        result = torch.where(chosen_portfolio & penalized, base_idx, result)
    return result
