"""Single feature definition shared by deployed inference and new training data."""
import torch


def residual_features(feature_names, op_names, *, candidate_ops, state_raw, fitness,
                      score, mu, sigma, used, remaining, stagnation):
    b, m = candidate_ops.shape
    device = candidate_ops.device
    dtype = score.dtype
    fitness = fitness.squeeze(-1) if fitness.ndim == 3 and fitness.shape[-1] == 1 else fitness
    state_names = {
        "state_best_z": 0,
        "state_spread_z": 1,
        "state_top_gap": 2,
        "state_diversity": 3,
        "state_elite_diversity": 4,
        "state_centroid_shift": 5,
        "state_remaining": 6,
        "state_used": 7,
        "state_stagnation": 8,
        "state_archive_pressure": 9,
        "state_dim_scaled": 10,
    }
    op_name_to_id = {"baseline": -2, **{name: i for i, name in enumerate(op_names)}}
    cols = []
    missing_cache = {}
    for name in feature_names:
        if name.endswith("_missing"):
            base = name[: -len("_missing")]
            if base not in missing_cache:
                value = torch.zeros((b, m), device=device, dtype=dtype)
                missing = torch.zeros_like(value)
            else:
                missing = missing_cache[base]
            cols.append(missing)
            continue
        if name == "used":
            value = torch.full((b, m), float(used), device=device, dtype=dtype)
        elif name == "remaining":
            value = torch.full((b, m), float(remaining), device=device, dtype=dtype)
        elif name == "stagnation":
            value = stagnation.view(b, 1).to(device=device, dtype=dtype).expand(b, m)
        elif name == "is_baseline":
            value = (candidate_ops < 0).to(dtype)
        elif name == "surrogate_score":
            value = score
        elif name == "surrogate_mu":
            value = mu
        elif name == "surrogate_sigma":
            value = sigma
        elif name == "fitness_best_before":
            value = fitness[:, :1].to(dtype).expand(b, m)
        elif name == "fitness_worst_before":
            value = fitness[:, -1:].to(dtype).expand(b, m)
        elif name in state_names:
            value = state_raw[:, state_names[name] : state_names[name] + 1].to(dtype).expand(b, m)
        elif name.startswith("op_"):
            op_name = name[len("op_") :]
            op_id = op_name_to_id.get(op_name, -999)
            value = (candidate_ops == op_id).to(dtype)
        else:
            value = torch.zeros((b, m), device=device, dtype=dtype)
        missing = torch.isnan(value) | torch.isinf(value)
        missing_cache[name] = missing.to(dtype)
        value = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
        cols.append(value)
    features = torch.stack(cols, dim=-1)
    return features
