import math
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from .residual_features import residual_features


EPS = 1e-8


def _ensure_fitness_2d(fitness: torch.Tensor) -> torch.Tensor:
    if fitness.dim() == 3 and fitness.size(-1) == 1:
        fitness = fitness.squeeze(-1)
    return fitness


def _to_bound_tensor(value, x: torch.Tensor) -> torch.Tensor:
    if torch.is_tensor(value):
        out = value.to(device=x.device, dtype=x.dtype)
    else:
        out = torch.tensor(value, device=x.device, dtype=x.dtype)
    if out.dim() == 0:
        out = out.view(1).repeat(x.size(-1))
    return out


def _get_bounds(problem, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    fun = getattr(problem, "fun", None)
    if isinstance(fun, dict):
        lb = _to_bound_tensor(fun.get("xlb", -5.0), x)
        ub = _to_bound_tensor(fun.get("xub", 5.0), x)
    else:
        lb = torch.full((x.size(-1),), -5.0, device=x.device, dtype=x.dtype)
        ub = torch.full((x.size(-1),), 5.0, device=x.device, dtype=x.dtype)
    return lb, ub


def _repair(problem, x: torch.Tensor) -> torch.Tensor:
    if hasattr(problem, "repaire"):
        return problem.repaire(x)
    lb, ub = _get_bounds(problem, x)
    return torch.max(torch.min(x, ub.view(1, 1, -1)), lb.view(1, 1, -1))


def _is_structured_system_problem(problem) -> bool:
    if not hasattr(problem, "getfunname"):
        return False
    return str(problem.getfunname()).lower().startswith("cecf")


def _is_hpo_problem(problem) -> bool:
    if not hasattr(problem, "getfunname"):
        return False
    return str(problem.getfunname()).lower().startswith("hpof")


def _is_uav_problem(problem) -> bool:
    if not hasattr(problem, "getfunname"):
        return False
    return str(problem.getfunname()).lower().startswith("uavf")


def _is_local_proxy_friendly_problem(problem) -> bool:
    return _is_hpo_problem(problem) or _is_uav_problem(problem)


def _is_embedded_high_dim_problem(problem) -> bool:
    high_dim = int(getattr(problem, "high_dim", 0) or 0)
    low_dim = int(getattr(problem, "low_dim", getattr(problem, "dim", 0)) or 0)
    return high_dim > 0 and low_dim > 0 and high_dim > low_dim


def _sort_pop(x: torch.Tensor, fitness: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    fitness = _ensure_fitness_2d(fitness)
    fit, idx = torch.sort(fitness, dim=1)
    pop = torch.gather(x, 1, idx.unsqueeze(-1).expand(-1, -1, x.size(-1)))
    return pop, fit


def _normalize_x(x: torch.Tensor, lb: torch.Tensor, ub: torch.Tensor) -> torch.Tensor:
    rng = (ub - lb).clamp_min(EPS).view(1, 1, -1)
    return 2.0 * (x - lb.view(1, 1, -1)) / rng - 1.0


def _gather_population(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    idx = idx.long().clamp(min=0, max=x.size(1) - 1)
    return torch.gather(x, 1, idx.unsqueeze(-1).expand(-1, -1, x.size(-1)))


def _fitness_features(fitness: torch.Tensor) -> torch.Tensor:
    fitness = _ensure_fitness_2d(fitness)
    b, n = fitness.shape
    mean = fitness.mean(dim=1, keepdim=True)
    std = fitness.std(dim=1, keepdim=True, unbiased=False).clamp_min(EPS)
    z = (fitness - mean) / std
    pref = torch.softmax(-z, dim=1)
    if n == 1:
        rank = torch.ones((1, 1), device=fitness.device, dtype=fitness.dtype)
    else:
        rank = torch.linspace(1.0, 0.0, n, device=fitness.device, dtype=fitness.dtype).view(1, n)
    rank = rank.expand(b, n)
    return torch.stack([pref, -z, rank], dim=-1)


class CheapStateEncoder(nn.Module):
    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.raw_dim = 11
        self.net = nn.Sequential(
            nn.Linear(self.raw_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

    def forward(
        self,
        x: torch.Tensor,
        fitness: torch.Tensor,
        problem,
        remaining: float,
        used: float,
        stagnation: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        fitness = _ensure_fitness_2d(fitness)
        b, n, d = x.shape
        fit_mean = fitness.mean(dim=1, keepdim=True)
        fit_std = fitness.std(dim=1, keepdim=True, unbiased=False).clamp_min(EPS)
        z = (fitness - fit_mean) / fit_std
        lb, ub = _get_bounds(problem, x)
        scale = torch.norm((ub - lb).clamp_min(EPS), p=2).clamp_min(EPS)

        best_z = z[:, 0]
        spread_z = z[:, -1] - z[:, 0]
        top_gap = z[:, min(1, n - 1)] - z[:, 0] if n > 1 else torch.zeros_like(best_z)
        diversity = torch.cdist(x, x).mean(dim=(1, 2)) / scale if n > 1 else torch.zeros_like(best_z)
        elite_n = max(2, min(n, n // 5))
        elite = x[:, :elite_n]
        elite_div = torch.cdist(elite, elite).mean(dim=(1, 2)) / scale if elite_n > 1 else torch.zeros_like(best_z)
        centroid = x.mean(dim=1)
        center = 0.5 * (lb + ub).view(1, -1)
        centroid_shift = torch.norm(centroid - center, p=2, dim=1) / scale

        rem = torch.full((b,), float(remaining), device=x.device, dtype=x.dtype)
        usd = torch.full((b,), float(used), device=x.device, dtype=x.dtype)
        stag = stagnation.to(device=x.device, dtype=x.dtype).view(-1)
        archive_pressure = torch.log1p(torch.full((b,), float(n), device=x.device, dtype=x.dtype)) / 6.0

        raw = torch.stack(
            [
                best_z,
                spread_z,
                top_gap,
                diversity,
                elite_div,
                centroid_shift,
                rem,
                usd,
                stag,
                archive_pressure,
                torch.full_like(best_z, float(d) / 100.0),
            ],
            dim=1,
        )
        raw = torch.nan_to_num(raw, nan=0.0, posinf=10.0, neginf=-10.0)
        return self.net(raw), raw


class AmortizedEliteGenerator(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, pool_per_op: int):
        super().__init__()
        self.pool_per_op = pool_per_op
        self.parent_net = nn.Sequential(
            nn.Linear(dim + 3 + hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, pool_per_op),
        )
        self.step_logit = nn.Parameter(torch.full((pool_per_op,), -2.0))
        self.mix_logit = nn.Parameter(torch.zeros((pool_per_op,)))

    def forward(
        self,
        x: torch.Tensor,
        fitness: torch.Tensor,
        state: torch.Tensor,
        problem,
    ) -> torch.Tensor:
        b, n, d = x.shape
        fit_feat = _fitness_features(fitness)
        state_each = state.unsqueeze(1).expand(b, n, state.size(-1))
        feats = torch.cat([x, fit_feat, state_each], dim=-1)
        parent_logits = self.parent_net(feats)
        weights = torch.softmax(parent_logits, dim=1)
        anchors = torch.einsum("bnk,bnd->bkd", weights, x)

        lb, ub = _get_bounds(problem, x)
        elite_n = max(2, min(n, n // 5))
        elite_std = x[:, :elite_n].std(dim=1, keepdim=True, unbiased=False).clamp_min(EPS)
        rng = (ub - lb).view(1, 1, d).clamp_min(EPS)
        sigma = torch.sigmoid(self.step_logit).view(1, self.pool_per_op, 1)
        noise = torch.randn_like(anchors) * (0.35 * sigma * elite_std + 0.005 * rng)
        best = x[:, :1].expand_as(anchors)
        mix = torch.sigmoid(self.mix_logit).view(1, self.pool_per_op, 1)
        return mix * anchors + (1.0 - mix) * best + noise


class RidgeFeatureEnsemble(nn.Module):
    def __init__(self, dim: int, members: int = 5, rff: int = 32, ridge: float = 1e-3):
        super().__init__()
        self.dim = dim
        self.members = members
        self.rff = max(4, int(rff))
        self.ridge = ridge
        freq = torch.randn(members, self.rff, dim)
        phase = 2.0 * math.pi * torch.rand(members, self.rff)
        self.register_buffer("freq", freq)
        self.register_buffer("phase", phase)

    def _features(self, x: torch.Tensor, member: int) -> torch.Tensor:
        proj = x @ self.freq[member].to(device=x.device, dtype=x.dtype).transpose(0, 1)
        proj = proj + self.phase[member].to(device=x.device, dtype=x.dtype).view(1, -1)
        rff = math.sqrt(2.0 / float(self.rff)) * torch.cos(proj)
        return torch.cat([torch.ones_like(x[:, :1]), x, x * x, rff], dim=1)

    @torch.no_grad()
    def predict(
        self,
        archive_x: torch.Tensor,
        archive_y: torch.Tensor,
        candidates: torch.Tensor,
        problem,
        distance_scale: float = 0.15,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        archive_y = _ensure_fitness_2d(archive_y)
        b, n, d = archive_x.shape
        m = candidates.size(1)
        lb, ub = _get_bounds(problem, archive_x)
        ax = _normalize_x(archive_x, lb, ub).detach()
        cx = _normalize_x(candidates, lb, ub).detach()

        preds: List[torch.Tensor] = []
        eye_cache: Dict[int, torch.Tensor] = {}
        for member in range(self.members):
            member_preds = []
            for bi in range(b):
                x_b = ax[bi]
                y_b = archive_y[bi]
                y_mean = y_b.mean()
                y_std = y_b.std(unbiased=False).clamp_min(EPS)
                yz = (y_b - y_mean) / y_std
                phi = self._features(x_b, member)
                phi_c = self._features(cx[bi], member)
                p = phi.size(1)
                if p not in eye_cache:
                    eye_cache[p] = torch.eye(p, device=phi.device, dtype=phi.dtype)
                lhs = phi.transpose(0, 1) @ phi + self.ridge * eye_cache[p]
                rhs = phi.transpose(0, 1) @ yz.view(-1, 1)
                try:
                    beta = torch.linalg.solve(lhs, rhs)
                except RuntimeError:
                    beta = torch.linalg.pinv(lhs) @ rhs
                member_preds.append((phi_c @ beta).view(m) * y_std + y_mean)
            preds.append(torch.stack(member_preds, dim=0))

        pred_stack = torch.stack(preds, dim=0)
        mu = pred_stack.mean(dim=0)
        sigma = pred_stack.std(dim=0, unbiased=False)

        if distance_scale > 0.0:
            dist = torch.cdist(cx, ax).amin(dim=2)
            sigma = sigma + float(distance_scale) * archive_y.std(dim=1, keepdim=True, unbiased=False).clamp_min(EPS) * dist
        return mu, sigma.clamp_min(EPS)


class ROOPFOptimizer(nn.Module):
    """Reliable Offline-Online Proxy Fusion optimizer."""

    def __init__(
        self,
        dim: int = 10,
        hidden_dim: int = 200,
        popSize: int = 100,
        max_nfe: int = 300,
        k_nums: int = 2,
        pool_per_op: int = 6,
        surrogate_members: int = 5,
        ablation: str = "",
        baseline_ckpt: str = "",
        log_candidates: bool = False,
        log_pool_candidates: bool = False,
        router_ckpt: str = "",
        router_weight: float = 0.10,
        policy_gate_better_ckpt: str = "",
        policy_gate_worse_ckpt: str = "",
    ):
        super().__init__()
        self.popsize = popSize
        self.k_nums = k_nums
        self.MaxNFE = max_nfe
        self.pool_per_op = pool_per_op
        self.op_names = [
            "amortized_elite",
            "current_to_pbest_archive",
            "covariance_elite",
            "trust_region",
            "coordinate_pattern",
            "opposition_restart",
        ]
        self.num_ops = len(self.op_names)
        self.ablation = {
            part.strip().lower()
            for part in str(ablation or "").replace("+", ",").split(",")
            if part.strip()
        }
        if "roopf" not in self.ablation:
            raise ValueError("This release exposes only the final ROOPF configuration.")
        self.ablation.update(
            {
                "baseline_guard",
                "baseline_backbone",
                "baseline_safe_selector",
                "structured_system_lock",
                "learned_router",
                "protected_residual_selector",
                "local_op_boost",
                "bbob_guarded_proxy",
                "residual_safety_gate",
                "weak_uncertainty",
            }
        )

        self.state_encoder = CheapStateEncoder(hidden_dim=hidden_dim)
        self.neural_generator = AmortizedEliteGenerator(dim=dim, hidden_dim=hidden_dim, pool_per_op=pool_per_op)
        self.surrogate = RidgeFeatureEnsemble(
            dim=dim,
            members=surrogate_members,
            rff=min(64, max(16, 4 * dim)),
            ridge=2e-3,
        )
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim + self.num_ops, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.num_ops),
        )
        self.log_kappa = nn.Parameter(torch.tensor(math.log(1.25)))
        self.de_f_logit = nn.Parameter(torch.tensor(0.0))
        self.trust_logit = nn.Parameter(torch.tensor(-1.5))
        self.cov_logit = nn.Parameter(torch.tensor(-1.0))
        self.coord_logit = nn.Parameter(torch.tensor(-1.5))
        self.restart_logit = nn.Parameter(torch.tensor(-2.0))

        self.generator = SimpleNamespace(op_names=self.op_names, num_ops=self.num_ops)
        self.baseline_model = None
        if "baseline_guard" in self.ablation:
            state = torch.load(baseline_ckpt, map_location="cpu") if baseline_ckpt else None
            has_v_proj = isinstance(state, dict) and any(str(k).endswith("v_proj.weight") for k in state.keys())
            if not has_v_proj:
                raise ValueError("The released ROOPF anchor checkpoint has an incompatible architecture.")
            from roopf.anchor_backbone import AnchorPolicyBackbone

            self.baseline_model = AnchorPolicyBackbone(dim=dim, hidden_dim=hidden_dim, popSize=popSize)
            if baseline_ckpt:
                self.baseline_model.load_state_dict(state, strict=True)
            self.baseline_model.eval()
            for param in self.baseline_model.parameters():
                param.requires_grad_(False)
        self.distill_alpha = 0.0
        self.distill_loss = torch.tensor(0.0)
        self.gate_history = []
        self.teacher_history = []
        self.fused_history = []
        self.last_acquisition = None
        self.last_surrogate_mu = None
        self.last_surrogate_sigma = None
        self.last_router_prob = None
        self.origin_stats = {}
        self.candidate_samples = []
        self.pool_candidate_samples = []
        self.log_candidates = bool(log_candidates)
        self.log_pool_candidates = bool(log_pool_candidates)
        self.router_model = None
        self.router_feature_names: List[str] = []
        self.router_weight = float(router_weight)
        self.policy_gate_better = None
        self.policy_gate_worse = None
        self.policy_gate_feature_names: List[str] = []
        if "learned_router" in self.ablation and router_ckpt:
            cpu_rng_state = torch.random.get_rng_state()
            cuda_rng_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            payload = torch.load(router_ckpt, map_location="cpu")
            state_dict = payload["state_dict"]
            in_dim = int(state_dict["net.0.weight"].shape[1])
            hidden = int(state_dict["net.0.weight"].shape[0])
            try:
                self.router_model = nn.Sequential(
                    nn.Linear(in_dim, hidden),
                    nn.LayerNorm(hidden),
                    nn.GELU(),
                    nn.Linear(hidden, hidden),
                    nn.GELU(),
                    nn.Linear(hidden, 1),
                )
                stripped_state = {str(k).replace("net.", "", 1): v for k, v in state_dict.items()}
                self.router_model.load_state_dict(stripped_state, strict=True)
                self.router_model.eval()
                for param in self.router_model.parameters():
                    param.requires_grad_(False)
            finally:
                torch.random.set_rng_state(cpu_rng_state)
                if cuda_rng_states is not None:
                    torch.cuda.set_rng_state_all(cuda_rng_states)
            self.router_feature_names = list(payload["feature_names"])
            self.register_buffer("router_mean", torch.tensor(payload["mean"], dtype=torch.float32))
            self.register_buffer("router_std", torch.tensor(payload["std"], dtype=torch.float32).clamp_min(EPS))
        if (
            ("gated_highdim_adapter" in self.ablation or "gated_rescue_highdim_adapter" in self.ablation)
            and policy_gate_better_ckpt
            and policy_gate_worse_ckpt
        ):
            cpu_rng_state = torch.random.get_rng_state()
            cuda_rng_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            try:
                better_payload = torch.load(policy_gate_better_ckpt, map_location="cpu")
                worse_payload = torch.load(policy_gate_worse_ckpt, map_location="cpu")
                self.policy_gate_better = self._build_policy_gate(better_payload)
                self.policy_gate_worse = self._build_policy_gate(worse_payload)
                self.policy_gate_feature_names = list(better_payload["feature_names"])
                self.register_buffer("policy_gate_better_mean", torch.tensor(better_payload["mean"], dtype=torch.float32))
                self.register_buffer(
                    "policy_gate_better_std",
                    torch.tensor(better_payload["std"], dtype=torch.float32).clamp_min(EPS),
                )
                self.register_buffer("policy_gate_worse_mean", torch.tensor(worse_payload["mean"], dtype=torch.float32))
                self.register_buffer(
                    "policy_gate_worse_std",
                    torch.tensor(worse_payload["std"], dtype=torch.float32).clamp_min(EPS),
                )
            finally:
                torch.random.set_rng_state(cpu_rng_state)
                if cuda_rng_states is not None:
                    torch.cuda.set_rng_state_all(cuda_rng_states)

    def _build_policy_gate(self, payload):
        state_dict = payload["state_dict"]
        in_dim = int(state_dict["net.0.weight"].shape[1])
        hidden = int(state_dict["net.0.weight"].shape[0])
        model = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        stripped_state = {str(k).replace("net.", "", 1): v for k, v in state_dict.items()}
        model.load_state_dict(stripped_state, strict=True)
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)
        return model

    def _reset_origin_stats(self) -> None:
        self.origin_stats = {
            "evals_total": 0,
            "baseline_evals": 0,
            "portfolio_evals": 0,
            "baseline_best_improvements": 0,
            "portfolio_best_improvements": 0,
            "baseline_admitted": 0,
            "portfolio_admitted": 0,
            "policy_gate_seen": 0,
            "policy_gate_allowed": 0,
            "policy_gate_rescue_allowed": 0,
            "op_evals": {name: 0 for name in self.op_names},
            "op_best_improvements": {name: 0 for name in self.op_names},
            "op_admitted": {name: 0 for name in self.op_names},
        }

    def _record_origin_stats(
        self,
        selected_ops: torch.Tensor,
        improved_best: torch.Tensor,
        admitted: torch.Tensor,
    ) -> None:
        ops = selected_ops.detach().cpu()
        best = improved_best.detach().cpu()
        adm = admitted.detach().cpu()
        self.origin_stats["evals_total"] += int(ops.numel())

        baseline_mask = ops < 0
        portfolio_mask = ops >= 0
        self.origin_stats["baseline_evals"] += int(baseline_mask.sum().item())
        self.origin_stats["portfolio_evals"] += int(portfolio_mask.sum().item())
        self.origin_stats["baseline_best_improvements"] += int((baseline_mask & best).sum().item())
        self.origin_stats["portfolio_best_improvements"] += int((portfolio_mask & best).sum().item())
        self.origin_stats["baseline_admitted"] += int((baseline_mask & adm).sum().item())
        self.origin_stats["portfolio_admitted"] += int((portfolio_mask & adm).sum().item())

        for op_idx, name in enumerate(self.op_names):
            op_mask = ops == op_idx
            self.origin_stats["op_evals"][name] += int(op_mask.sum().item())
            self.origin_stats["op_best_improvements"][name] += int((op_mask & best).sum().item())
            self.origin_stats["op_admitted"][name] += int((op_mask & adm).sum().item())

    def _record_candidate_samples(
        self,
        step_idx: int,
        eval_before: int,
        used: float,
        remaining: float,
        stagnation: torch.Tensor,
        selected_ops: torch.Tensor,
        cand_fit: torch.Tensor,
        admitted: torch.Tensor,
        improved_best: torch.Tensor,
        best_before: torch.Tensor,
        worst_before: torch.Tensor,
        state_raw: torch.Tensor,
        score: Optional[torch.Tensor] = None,
        mu: Optional[torch.Tensor] = None,
        sigma: Optional[torch.Tensor] = None,
        router_prob: Optional[torch.Tensor] = None,
    ) -> None:
        b, rest = selected_ops.shape
        ops = selected_ops.detach().cpu()
        fit = cand_fit.detach().cpu()
        adm = admitted.detach().cpu()
        imp = improved_best.detach().cpu()
        stag = stagnation.detach().cpu()
        best = best_before.detach().cpu()
        worst = worst_before.detach().cpu()
        raw = state_raw.detach().cpu()
        score_cpu = score.detach().cpu() if score is not None else None
        mu_cpu = mu.detach().cpu() if mu is not None else None
        sigma_cpu = sigma.detach().cpu() if sigma is not None else None
        router_cpu = router_prob.detach().cpu() if router_prob is not None else None
        raw_names = [
            "state_best_z",
            "state_spread_z",
            "state_top_gap",
            "state_diversity",
            "state_elite_diversity",
            "state_centroid_shift",
            "state_remaining",
            "state_used",
            "state_stagnation",
            "state_archive_pressure",
            "state_dim_scaled",
        ]
        for bi in range(b):
            for ci in range(rest):
                op_id = int(ops[bi, ci].item())
                row = {
                    "step": int(step_idx),
                    "batch": int(bi),
                    "candidate_slot": int(ci),
                    "eval_before": int(eval_before),
                    "used": float(used),
                    "remaining": float(remaining),
                    "stagnation": float(stag[bi].item()),
                    "op_id": op_id,
                    "op_name": "baseline" if op_id < 0 else self.op_names[op_id],
                    "is_baseline": int(op_id < 0),
                    "fitness_best_before": float(best[bi].item()),
                    "fitness_worst_before": float(worst[bi].item()),
                    "candidate_fit": float(fit[bi, ci].item()),
                    "admitted": int(bool(adm[bi, ci].item())),
                    "improved_best": int(bool(imp[bi, ci].item())),
                    "surrogate_score": "",
                    "surrogate_mu": "",
                    "surrogate_sigma": "",
                    "router_prob": "",
                }
                if score_cpu is not None:
                    row["surrogate_score"] = float(score_cpu[bi, ci].item())
                if mu_cpu is not None:
                    row["surrogate_mu"] = float(mu_cpu[bi, ci].item())
                if sigma_cpu is not None:
                    row["surrogate_sigma"] = float(sigma_cpu[bi, ci].item())
                if router_cpu is not None and not torch.isnan(router_cpu[bi, ci]):
                    row["router_prob"] = float(router_cpu[bi, ci].item())
                for ri, name in enumerate(raw_names):
                    row[name] = float(raw[bi, ri].item())
                self.candidate_samples.append(row)

    def _record_pool_candidate_samples(
        self,
        step_idx: int,
        eval_before: int,
        used: float,
        remaining: float,
        stagnation: torch.Tensor,
        baseline_cand: Optional[torch.Tensor],
        pool: torch.Tensor,
        pool_ops: torch.Tensor,
        baseline_score: Optional[torch.Tensor],
        baseline_mu: Optional[torch.Tensor],
        baseline_sigma: Optional[torch.Tensor],
        pool_score: Optional[torch.Tensor],
        pool_mu: Optional[torch.Tensor],
        pool_sigma: Optional[torch.Tensor],
        problem,
        fitness: torch.Tensor,
        state_raw: torch.Tensor,
    ) -> None:
        if baseline_cand is None:
            teacher_cand = pool
            teacher_ops = pool_ops.view(1, -1).expand(pool.size(0), -1)
            base_count = 0
        else:
            teacher_cand = torch.cat([baseline_cand, pool], dim=1)
            base_ops = torch.full((pool.size(0), baseline_cand.size(1)), -2, device=pool.device, dtype=torch.long)
            teacher_ops = torch.cat([base_ops, pool_ops.view(1, -1).expand(pool.size(0), -1)], dim=1)
            base_count = baseline_cand.size(1)

        teacher_fit = _ensure_fitness_2d(problem.calfitness(teacher_cand))
        threshold = fitness[:, -1].view(-1, 1)
        best_before = fitness[:, 0].view(-1, 1)
        admitted = teacher_fit < (threshold - 1e-12)
        improved_best = teacher_fit < (best_before - 1e-12)
        if base_count > 0:
            baseline_best = teacher_fit[:, :base_count].amin(dim=1, keepdim=True)
        else:
            baseline_best = torch.full_like(best_before, float("nan"))
        beat_baseline = teacher_fit < baseline_best
        rank = torch.argsort(torch.argsort(teacher_fit, dim=1), dim=1)

        b, m = teacher_fit.shape
        nan_block = torch.full((b, base_count), float("nan"), device=pool.device, dtype=pool.dtype)
        if base_count > 0:
            base_score = baseline_score if baseline_score is not None else nan_block
            base_mu = baseline_mu if baseline_mu is not None else nan_block
            base_sigma = baseline_sigma if baseline_sigma is not None else nan_block
            score = torch.cat([base_score, pool_score], dim=1) if pool_score is not None else base_score
            mu = torch.cat([base_mu, pool_mu], dim=1) if pool_mu is not None else base_mu
            sigma = torch.cat([base_sigma, pool_sigma], dim=1) if pool_sigma is not None else base_sigma
        else:
            score = pool_score
            mu = pool_mu
            sigma = pool_sigma
        if score is None:
            score = torch.full_like(teacher_fit, float("nan"))
        if mu is None:
            mu = torch.full_like(teacher_fit, float("nan"))
        if sigma is None:
            sigma = torch.full_like(teacher_fit, float("nan"))

        ops = teacher_ops.detach().cpu()
        fit = teacher_fit.detach().cpu()
        adm = admitted.detach().cpu()
        imp = improved_best.detach().cpu()
        beat = beat_baseline.detach().cpu()
        rank_cpu = rank.detach().cpu()
        score_cpu = score.detach().cpu()
        mu_cpu = mu.detach().cpu()
        sigma_cpu = sigma.detach().cpu()
        stag = stagnation.detach().cpu()
        best = fitness[:, 0].detach().cpu()
        worst = fitness[:, -1].detach().cpu()
        raw = state_raw.detach().cpu()
        raw_names = [
            "state_best_z",
            "state_spread_z",
            "state_top_gap",
            "state_diversity",
            "state_elite_diversity",
            "state_centroid_shift",
            "state_remaining",
            "state_used",
            "state_stagnation",
            "state_archive_pressure",
            "state_dim_scaled",
        ]
        for bi in range(b):
            for ci in range(m):
                op_id = int(ops[bi, ci].item())
                row = {
                    "step": int(step_idx),
                    "batch": int(bi),
                    "candidate_slot": int(ci),
                    "eval_before": int(eval_before),
                    "used": float(used),
                    "remaining": float(remaining),
                    "stagnation": float(stag[bi].item()),
                    "op_id": op_id,
                    "op_name": "baseline" if op_id < 0 else self.op_names[op_id],
                    "is_baseline": int(op_id < 0),
                    "fitness_best_before": float(best[bi].item()),
                    "fitness_worst_before": float(worst[bi].item()),
                    "candidate_fit": float(fit[bi, ci].item()),
                    "admitted": int(bool(adm[bi, ci].item())),
                    "improved_best": int(bool(imp[bi, ci].item())),
                    "baseline_candidate_fit": float(baseline_best.detach().cpu()[bi, 0].item()),
                    "candidate_minus_baseline": float((fit[bi, ci] - baseline_best.detach().cpu()[bi, 0]).item()),
                    "beat_baseline": int(bool(beat[bi, ci].item())),
                    "rank_in_pool": int(rank_cpu[bi, ci].item()),
                    "pool_size": int(m),
                    "surrogate_score": float(score_cpu[bi, ci].item()) if not torch.isnan(score_cpu[bi, ci]) else "",
                    "surrogate_mu": float(mu_cpu[bi, ci].item()) if not torch.isnan(mu_cpu[bi, ci]) else "",
                    "surrogate_sigma": float(sigma_cpu[bi, ci].item()) if not torch.isnan(sigma_cpu[bi, ci]) else "",
                }
                for ri, name in enumerate(raw_names):
                    row[name] = float(raw[bi, ri].item())
                self.pool_candidate_samples.append(row)

    def _router_probability(
        self,
        candidate_ops: torch.Tensor,
        state_raw: torch.Tensor,
        fitness: torch.Tensor,
        score: torch.Tensor,
        mu: torch.Tensor,
        sigma: torch.Tensor,
        used: float,
        remaining: float,
        stagnation: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        if self.router_model is None:
            return None
        b, m = candidate_ops.shape
        device, dtype = candidate_ops.device, score.dtype
        features = residual_features(self.router_feature_names, self.op_names,
            candidate_ops=candidate_ops, state_raw=state_raw, fitness=fitness,
            score=score, mu=mu, sigma=sigma, used=used,
            remaining=remaining, stagnation=stagnation)
        mean = self.router_mean.to(device=device, dtype=dtype).view(1, 1, -1)
        std = self.router_std.to(device=device, dtype=dtype).view(1, 1, -1)
        normalized = (features - mean) / std
        observer = getattr(self, "residual_input_observer", None)
        if observer is not None:
            observer(features.detach().clone(), normalized.detach().clone(),
                     candidate_ops.detach().clone(), self.evalnum)
        features = normalized
        logits = self.router_model.to(device)(features.view(b * m, -1)).view(b, m)
        return torch.sigmoid(logits)

    def _state_gate(self, state: torch.Tensor, success_logits: torch.Tensor) -> torch.Tensor:
        if "no_success_memory" in self.ablation:
            success_logits = torch.zeros_like(success_logits)
        logits = self.gate(torch.cat([state, success_logits], dim=1))
        weights = torch.softmax(logits + 0.35 * success_logits, dim=1)
        if self.training:
            weights = 0.92 * weights + 0.08 / self.num_ops
        return weights

    def _current_to_pbest(self, x: torch.Tensor, problem, state_raw: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        p_ratio = 0.08 + 0.22 * state_raw[:, 6].clamp(0.0, 1.0)
        out = []
        f = 0.15 + 0.75 * torch.sigmoid(self.de_f_logit)
        for bi in range(b):
            p_count = max(2, min(n, int(math.ceil(float(p_ratio[bi].item()) * n))))
            cur = x[bi, torch.randint(0, n, (self.pool_per_op,), device=x.device)]
            pb = x[bi, torch.randint(0, p_count, (self.pool_per_op,), device=x.device)]
            r1 = x[bi, torch.randint(0, n, (self.pool_per_op,), device=x.device)]
            r2 = x[bi, torch.randint(0, n, (self.pool_per_op,), device=x.device)]
            out.append(cur + f * (pb - cur) + f * (r1 - r2))
        return torch.stack(out, dim=0)

    def _covariance_elite(self, x: torch.Tensor, problem, state_raw: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        elite_n = max(3, min(n, n // 5))
        elite = x[:, :elite_n]
        mean = elite.mean(dim=1, keepdim=True)
        centered = elite - mean
        base_idx = torch.randint(0, elite_n, (b, self.pool_per_op), device=x.device)
        base = _gather_population(elite, base_idx)
        ranks = torch.arange(1, elite_n + 1, device=x.device, dtype=x.dtype)
        weights = torch.log(torch.tensor(float(elite_n) + 0.5, device=x.device, dtype=x.dtype)) - torch.log(ranks)
        weights = weights.clamp_min(0.0)
        weights = weights / weights.sum().clamp_min(EPS)
        coeff = torch.randn((b, self.pool_per_op, elite_n), device=x.device, dtype=x.dtype)
        coeff = coeff * torch.sqrt(weights.view(1, 1, -1))
        rank_noise = torch.einsum("bke,bed->bkd", coeff, centered)
        diag_noise = torch.randn_like(base) * centered.std(dim=1, keepdim=True, unbiased=False).clamp_min(EPS)
        step = 0.03 + 0.35 * torch.sigmoid(self.cov_logit)
        rem = state_raw[:, 6].view(b, 1, 1).clamp(0.0, 1.0)
        return base + step * (0.6 + rem) * (0.25 * rank_noise + 0.75 * diag_noise)

    def _trust_region(self, x: torch.Tensor, problem, state_raw: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        lb, ub = _get_bounds(problem, x)
        rng = (ub - lb).view(1, 1, d).clamp_min(EPS)
        elite_n = max(2, min(n, n // 5))
        elite_std = x[:, :elite_n].std(dim=1, keepdim=True, unbiased=False).clamp_min(EPS)
        sigma = 0.003 + 0.16 * torch.sigmoid(self.trust_logit)
        if "local_op_boost" in self.ablation and _is_local_proxy_friendly_problem(problem):
            sigma = torch.clamp(1.65 * sigma, max=0.18)
        rem = state_raw[:, 6].view(b, 1, 1).clamp(0.0, 1.0)
        stag = state_raw[:, 8].view(b, 1, 1).clamp(0.0, 1.0)
        expand = 0.75 + 1.25 * stag
        if "local_op_boost" in self.ablation and _is_local_proxy_friendly_problem(problem):
            expand = 0.55 + 0.55 * rem + 0.55 * stag
        base = x[:, :1].expand(b, self.pool_per_op, d)
        return base + torch.randn_like(base) * expand * (sigma * (0.2 + rem) * elite_std + 0.0015 * rng)

    def _coordinate_pattern(self, x: torch.Tensor, problem, state_raw: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        lb, ub = _get_bounds(problem, x)
        rng = (ub - lb).view(1, 1, d).clamp_min(EPS)
        elite_n = max(2, min(n, n // 5))
        idx = torch.randint(0, elite_n, (b, self.pool_per_op), device=x.device)
        base = _gather_population(x[:, :elite_n], idx)
        step = 0.002 + 0.08 * torch.sigmoid(self.coord_logit)
        coord_rate = min(0.65, max(1.0 / max(float(d), 1.0), 0.25))
        if "local_op_boost" in self.ablation and _is_local_proxy_friendly_problem(problem):
            step = torch.clamp(1.55 * step, max=0.09)
            coord_rate = min(0.45, max(2.0 / max(float(d), 1.0), 0.20))
        mask = torch.rand((b, self.pool_per_op, d), device=x.device, dtype=x.dtype) < coord_rate
        forced = torch.randint(0, d, (b, self.pool_per_op, 1), device=x.device)
        mask.scatter_(2, forced, True)
        sign = torch.where(torch.rand_like(base) < 0.5, -torch.ones_like(base), torch.ones_like(base))
        rem = state_raw[:, 6].view(b, 1, 1).clamp(0.0, 1.0)
        return base + sign * mask.to(x.dtype) * step * (0.25 + rem) * rng

    def _opposition_restart(self, x: torch.Tensor, problem, state_raw: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        lb, ub = _get_bounds(problem, x)
        lbv = lb.view(1, 1, d)
        ubv = ub.view(1, 1, d)
        center = 0.5 * (lbv + ubv)
        idx = torch.randint(0, n, (b, self.pool_per_op), device=x.device)
        parent = _gather_population(x, idx)
        rng = (ubv - lbv).clamp_min(EPS)
        noise = (0.005 + 0.20 * torch.sigmoid(self.restart_logit)) * torch.randn_like(parent) * rng
        return center + (center - parent) + noise

    def _candidate_pool(
        self,
        x: torch.Tensor,
        fitness: torch.Tensor,
        problem,
        state: torch.Tensor,
        state_raw: torch.Tensor,
        success_logits: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        weights = self._state_gate(state, success_logits)
        op_cands = [
            self.neural_generator(x, fitness, state, problem),
            self._current_to_pbest(x, problem, state_raw),
            self._covariance_elite(x, problem, state_raw),
            self._trust_region(x, problem, state_raw),
            self._coordinate_pattern(x, problem, state_raw),
            self._opposition_restart(x, problem, state_raw),
        ]
        stacked = torch.stack(op_cands, dim=1)
        stacked = _repair(problem, stacked.view(x.size(0), -1, x.size(-1)))
        op_ids = torch.arange(self.num_ops, device=x.device).repeat_interleave(self.pool_per_op)
        prior = weights.repeat_interleave(self.pool_per_op, dim=1)
        if "no_neural" in self.ablation:
            neural_mask = op_ids.view(1, -1) == 0
            prior = torch.where(neural_mask, torch.full_like(prior, EPS), prior)
            prior = prior / prior.sum(dim=1, keepdim=True).clamp_min(EPS)
        if (
            "productive_op_router" in self.ablation
            and not _is_hpo_problem(problem)
            and not _is_embedded_high_dim_problem(problem)
        ):
            op = op_ids.view(1, -1)
            restart_mask = op == self.op_names.index("opposition_restart")
            prior = torch.where(restart_mask, prior * 0.02, prior)
            prior = prior / prior.sum(dim=1, keepdim=True).clamp_min(EPS)
        if "local_op_boost" in self.ablation and _is_local_proxy_friendly_problem(problem):
            op = op_ids.view(1, -1)
            multiplier = torch.ones_like(prior)
            trust_idx = self.op_names.index("trust_region")
            coord_idx = self.op_names.index("coordinate_pattern")
            pbest_idx = self.op_names.index("current_to_pbest_archive")
            restart_idx = self.op_names.index("opposition_restart")
            multiplier = torch.where(op == trust_idx, torch.full_like(multiplier, 2.30), multiplier)
            multiplier = torch.where(op == coord_idx, torch.full_like(multiplier, 1.85), multiplier)
            multiplier = torch.where(op == pbest_idx, torch.full_like(multiplier, 1.25), multiplier)
            multiplier = torch.where(op == restart_idx, torch.full_like(multiplier, 0.35), multiplier)
            prior = prior * multiplier
            prior = prior / prior.sum(dim=1, keepdim=True).clamp_min(EPS)
        return stacked, op_ids, prior

    def _dynamic_highdim_prior(
        self,
        op_prior: torch.Tensor,
        op_ids: torch.Tensor,
        op_seen: torch.Tensor,
        op_admit: torch.Tensor,
        op_best: torch.Tensor,
        remaining: float,
        stagnation: torch.Tensor,
    ) -> torch.Tensor:
        seen = op_seen.clamp_min(1.0)
        best_rate = op_best / seen

        score = torch.where(op_best > 0.0, best_rate, torch.zeros_like(best_rate))
        warm = (op_seen.sum(dim=1, keepdim=True) >= 48.0).to(op_prior.dtype)
        rem = float(remaining)
        strength = (0.02 + 0.04 * (1.0 - rem)) * warm
        multiplier = torch.exp(strength * score).clamp(1.0, 1.12)
        candidate_multiplier = torch.gather(multiplier, 1, op_ids.view(1, -1).expand(op_prior.size(0), -1))
        prior = op_prior * candidate_multiplier
        return prior / prior.sum(dim=1, keepdim=True).clamp_min(EPS)

    def _trajectory_highdim_prior(
        self,
        op_prior: torch.Tensor,
        op_ids: torch.Tensor,
        state_raw: torch.Tensor,
        remaining: float,
        stagnation: torch.Tensor,
    ) -> torch.Tensor:
        """Conservative high-dimensional adapter derived from trajectory labels.

        The trajectory dataset showed future improvements concentrating in late,
        low-stagnation phases where trust-region candidates were already useful.
        This adapter only nudges operator priors under that state pattern; it does
        not alter the final protected selector.
        """
        if state_raw is None:
            return op_prior

        spread_z = state_raw[:, 1].view(-1, 1)
        centroid_shift = state_raw[:, 5].view(-1, 1)
        used = state_raw[:, 7].view(-1, 1)
        stag = stagnation.view(-1, 1).to(device=op_prior.device, dtype=op_prior.dtype)
        rem = torch.full_like(used, float(remaining))

        late = used > 0.66
        still_budget = rem > 0.03
        not_stalled = stag < 0.18
        moved_center = centroid_shift > 0.025
        not_too_rugged = spread_z < 4.75
        gate = (late & still_budget & not_stalled & moved_center & not_too_rugged).to(op_prior.dtype)

        op_multiplier = torch.ones((op_prior.size(0), self.num_ops), device=op_prior.device, dtype=op_prior.dtype)
        trust_idx = self.op_names.index("trust_region")
        pbest_idx = self.op_names.index("current_to_pbest_archive")
        cov_idx = self.op_names.index("covariance_elite")
        restart_idx = self.op_names.index("opposition_restart")

        # Small multipliers are intentional: f20/f24 are path-sensitive, and
        # aggressive high-dimensional adaptation can lose even when offline
        # diagnostics look good.
        strength = 0.015 + 0.025 * used.clamp(0.0, 1.0)
        op_multiplier[:, trust_idx : trust_idx + 1] = 1.0 + gate * strength
        op_multiplier[:, pbest_idx : pbest_idx + 1] = 1.0 + gate * (0.45 * strength)
        op_multiplier[:, cov_idx : cov_idx + 1] = 1.0 + gate * (0.35 * strength)
        op_multiplier[:, restart_idx : restart_idx + 1] = 1.0 - gate * (0.50 * strength)

        candidate_multiplier = torch.gather(op_multiplier, 1, op_ids.view(1, -1).expand(op_prior.size(0), -1))
        prior = op_prior * candidate_multiplier.clamp(0.96, 1.04)
        return prior / prior.sum(dim=1, keepdim=True).clamp_min(EPS)

    def _validated_highdim_prior(
        self,
        op_prior: torch.Tensor,
        op_ids: torch.Tensor,
        op_seen: torch.Tensor,
        op_best: torch.Tensor,
        state_raw: torch.Tensor,
        remaining: float,
        stagnation: torch.Tensor,
    ) -> torch.Tensor:
        if state_raw is None:
            return op_prior

        seen = op_seen.clamp_min(1.0)
        best_rate = op_best / seen
        used = state_raw[:, 7].view(-1, 1)
        stag = stagnation.view(-1, 1).to(device=op_prior.device, dtype=op_prior.dtype)
        rem = torch.full_like(used, float(remaining))

        # Activate only after the run itself has validated an operator. This is
        # This is deliberately strict because f11/f20 punish early prior drift.
        global_gate = ((used > 0.58) & (rem > 0.03) & (stag < 0.25)).to(op_prior.dtype)
        warm = (op_seen >= 24.0).to(op_prior.dtype)
        strong = ((op_best >= 6.0) & (best_rate >= 0.20)).to(op_prior.dtype)
        weak = ((op_seen >= 48.0) & (op_best <= 0.0)).to(op_prior.dtype)

        op_multiplier = torch.ones((op_prior.size(0), self.num_ops), device=op_prior.device, dtype=op_prior.dtype)
        strength = (0.008 + 0.018 * used.clamp(0.0, 1.0)) * global_gate
        boost = warm * strong * strength * best_rate.clamp(0.0, 1.0)
        penalty = weak * (0.50 * strength)
        op_multiplier = op_multiplier + boost - penalty

        # Avoid over-amplifying restart evidence; its high-dim successes are sparse
        # and trajectory labels showed it is usually a late recovery move.
        restart_idx = self.op_names.index("opposition_restart")
        op_multiplier[:, restart_idx : restart_idx + 1] = torch.minimum(
            op_multiplier[:, restart_idx : restart_idx + 1],
            torch.ones_like(op_multiplier[:, restart_idx : restart_idx + 1]),
        )

        candidate_multiplier = torch.gather(op_multiplier, 1, op_ids.view(1, -1).expand(op_prior.size(0), -1))
        prior = op_prior * candidate_multiplier.clamp(0.985, 1.025)
        return prior / prior.sum(dim=1, keepdim=True).clamp_min(EPS)

    def _policy_gate_features(
        self,
        feature_names: List[str],
        state_raw: torch.Tensor,
        fitness: torch.Tensor,
        used: float,
        remaining: float,
        stagnation: torch.Tensor,
        step_idx: int,
        eval_before: int,
    ) -> torch.Tensor:
        b = state_raw.size(0)
        fit = _ensure_fitness_2d(fitness)
        values = {
            "used": torch.full((b,), float(used), device=state_raw.device, dtype=state_raw.dtype),
            "remaining": torch.full((b,), float(remaining), device=state_raw.device, dtype=state_raw.dtype),
            "stagnation": stagnation.view(-1).to(device=state_raw.device, dtype=state_raw.dtype),
            "fitness_best_before": fit[:, 0].to(device=state_raw.device, dtype=state_raw.dtype),
            "fitness_worst_before": fit[:, -1].to(device=state_raw.device, dtype=state_raw.dtype),
            "state_best_z": state_raw[:, 0],
            "state_spread_z": state_raw[:, 1],
            "state_top_gap": state_raw[:, 2],
            "state_diversity": state_raw[:, 3],
            "state_elite_diversity": state_raw[:, 4],
            "state_centroid_shift": state_raw[:, 5],
            "state_remaining": state_raw[:, 6],
            "state_used": state_raw[:, 7],
            "state_stagnation": state_raw[:, 8],
            "state_archive_pressure": state_raw[:, 9],
            "state_dim_scaled": state_raw[:, 10],
            "step": torch.full((b,), float(step_idx), device=state_raw.device, dtype=state_raw.dtype),
            "eval_before": torch.full((b,), float(eval_before), device=state_raw.device, dtype=state_raw.dtype),
        }
        return torch.stack([values.get(name, torch.zeros((b,), device=state_raw.device, dtype=state_raw.dtype)) for name in feature_names], dim=1)

    def _policy_gate_prob(
        self,
        gate,
        features: torch.Tensor,
        mean: torch.Tensor,
        std: torch.Tensor,
    ) -> torch.Tensor:
        if gate is None:
            return torch.zeros((features.size(0),), device=features.device, dtype=features.dtype)
        x = (features - mean.to(device=features.device, dtype=features.dtype).view(1, -1)) / std.to(
            device=features.device,
            dtype=features.dtype,
        ).view(1, -1).clamp_min(EPS)
        logits = gate.to(features.device)(x).view(-1)
        return torch.sigmoid(logits)

    def _gated_highdim_prior(
        self,
        op_prior: torch.Tensor,
        op_ids: torch.Tensor,
        op_seen: torch.Tensor,
        op_best: torch.Tensor,
        state_raw: torch.Tensor,
        fitness: torch.Tensor,
        used: float,
        remaining: float,
        stagnation: torch.Tensor,
        step_idx: int,
        eval_before: int,
    ) -> torch.Tensor:
        if self.policy_gate_better is None or self.policy_gate_worse is None or not self.policy_gate_feature_names:
            return op_prior
        features = self._policy_gate_features(
            self.policy_gate_feature_names,
            state_raw=state_raw,
            fitness=fitness,
            used=used,
            remaining=remaining,
            stagnation=stagnation,
            step_idx=step_idx,
            eval_before=eval_before,
        )
        better_prob = self._policy_gate_prob(
            self.policy_gate_better,
            features,
            self.policy_gate_better_mean,
            self.policy_gate_better_std,
        )
        worse_prob = self._policy_gate_prob(
            self.policy_gate_worse,
            features,
            self.policy_gate_worse_mean,
            self.policy_gate_worse_std,
        )
        allow = ((better_prob > 0.45) & (worse_prob < 0.20)).view(-1, 1)
        if self.origin_stats:
            self.origin_stats["policy_gate_seen"] = self.origin_stats.get("policy_gate_seen", 0) + int(allow.numel())
            self.origin_stats["policy_gate_allowed"] = self.origin_stats.get("policy_gate_allowed", 0) + int(allow.sum().item())
        adapted = self._validated_highdim_prior(
            op_prior=op_prior,
            op_ids=op_ids,
            op_seen=op_seen,
            op_best=op_best,
            state_raw=state_raw,
            remaining=remaining,
            stagnation=stagnation,
        )
        return torch.where(allow, adapted, op_prior)

    def _gated_rescue_highdim_prior(
        self,
        op_prior: torch.Tensor,
        op_ids: torch.Tensor,
        op_seen: torch.Tensor,
        op_best: torch.Tensor,
        state_raw: torch.Tensor,
        fitness: torch.Tensor,
        used: float,
        remaining: float,
        stagnation: torch.Tensor,
        step_idx: int,
        eval_before: int,
    ) -> torch.Tensor:
        if self.policy_gate_better is None or self.policy_gate_worse is None or not self.policy_gate_feature_names:
            return op_prior
        features = self._policy_gate_features(
            self.policy_gate_feature_names,
            state_raw=state_raw,
            fitness=fitness,
            used=used,
            remaining=remaining,
            stagnation=stagnation,
            step_idx=step_idx,
            eval_before=eval_before,
        )
        better_prob = self._policy_gate_prob(
            self.policy_gate_better,
            features,
            self.policy_gate_better_mean,
            self.policy_gate_better_std,
        )
        worse_prob = self._policy_gate_prob(
            self.policy_gate_worse,
            features,
            self.policy_gate_worse_mean,
            self.policy_gate_worse_std,
        )
        gate_allow = ((better_prob > 0.45) & (worse_prob < 0.20)).view(-1, 1)

        seen_total = op_seen.sum(dim=1, keepdim=True)
        best_total = op_best.sum(dim=1, keepdim=True)
        total_best_rate = best_total / seen_total.clamp_min(1.0)
        trust_idx = self.op_names.index("trust_region")
        trust_seen = op_seen[:, trust_idx : trust_idx + 1]
        trust_best_rate = op_best[:, trust_idx : trust_idx + 1] / trust_seen.clamp_min(1.0)
        warm = seen_total >= 16.0
        productive_trust = ((trust_seen >= 48.0) & (trust_best_rate > 0.35)) | (total_best_rate > 0.30)
        rescue_allow = gate_allow & warm & (~productive_trust)

        if self.origin_stats:
            self.origin_stats["policy_gate_seen"] = self.origin_stats.get("policy_gate_seen", 0) + int(gate_allow.numel())
            self.origin_stats["policy_gate_allowed"] = self.origin_stats.get("policy_gate_allowed", 0) + int(gate_allow.sum().item())
            self.origin_stats["policy_gate_rescue_allowed"] = self.origin_stats.get("policy_gate_rescue_allowed", 0) + int(rescue_allow.sum().item())

        op_multiplier = torch.ones((op_prior.size(0), self.num_ops), device=op_prior.device, dtype=op_prior.dtype)
        restart_idx = self.op_names.index("opposition_restart")
        pbest_idx = self.op_names.index("current_to_pbest_archive")
        cov_idx = self.op_names.index("covariance_elite")
        coord_idx = self.op_names.index("coordinate_pattern")
        neural_idx = self.op_names.index("amortized_elite")

        op_multiplier[:, restart_idx : restart_idx + 1] = 1.45
        op_multiplier[:, trust_idx : trust_idx + 1] = 1.12
        op_multiplier[:, pbest_idx : pbest_idx + 1] = 1.08
        op_multiplier[:, cov_idx : cov_idx + 1] = 0.92
        op_multiplier[:, coord_idx : coord_idx + 1] = 0.92
        op_multiplier[:, neural_idx : neural_idx + 1] = 0.85

        candidate_multiplier = torch.gather(op_multiplier, 1, op_ids.view(1, -1).expand(op_prior.size(0), -1))
        adapted = op_prior * candidate_multiplier
        adapted = adapted / adapted.sum(dim=1, keepdim=True).clamp_min(EPS)
        return torch.where(rescue_allow, adapted, op_prior)

    def _fixed_highdim_probe_prior(self, op_prior: torch.Tensor, op_ids: torch.Tensor, mode: str) -> torch.Tensor:
        op_multiplier = torch.ones((op_prior.size(0), self.num_ops), device=op_prior.device, dtype=op_prior.dtype)
        neural_idx = self.op_names.index("amortized_elite")
        pbest_idx = self.op_names.index("current_to_pbest_archive")
        cov_idx = self.op_names.index("covariance_elite")
        trust_idx = self.op_names.index("trust_region")
        coord_idx = self.op_names.index("coordinate_pattern")
        restart_idx = self.op_names.index("opposition_restart")

        if mode == "restart":
            op_multiplier[:, neural_idx : neural_idx + 1] = 0.65
            op_multiplier[:, pbest_idx : pbest_idx + 1] = 1.10
            op_multiplier[:, cov_idx : cov_idx + 1] = 0.80
            op_multiplier[:, trust_idx : trust_idx + 1] = 1.20
            op_multiplier[:, coord_idx : coord_idx + 1] = 0.85
            op_multiplier[:, restart_idx : restart_idx + 1] = 2.50
        elif mode == "trust":
            op_multiplier[:, neural_idx : neural_idx + 1] = 0.70
            op_multiplier[:, pbest_idx : pbest_idx + 1] = 1.25
            op_multiplier[:, cov_idx : cov_idx + 1] = 1.10
            op_multiplier[:, trust_idx : trust_idx + 1] = 2.00
            op_multiplier[:, coord_idx : coord_idx + 1] = 0.90
            op_multiplier[:, restart_idx : restart_idx + 1] = 0.40
        elif mode == "mixed":
            op_multiplier[:, neural_idx : neural_idx + 1] = 0.70
            op_multiplier[:, pbest_idx : pbest_idx + 1] = 1.20
            op_multiplier[:, cov_idx : cov_idx + 1] = 1.80
            op_multiplier[:, trust_idx : trust_idx + 1] = 1.10
            op_multiplier[:, coord_idx : coord_idx + 1] = 1.40
            op_multiplier[:, restart_idx : restart_idx + 1] = 0.60

        candidate_multiplier = torch.gather(op_multiplier, 1, op_ids.view(1, -1).expand(op_prior.size(0), -1))
        prior = op_prior * candidate_multiplier
        return prior / prior.sum(dim=1, keepdim=True).clamp_min(EPS)

    def _select_by_acquisition(
        self,
        archive_x: torch.Tensor,
        archive_y: torch.Tensor,
        pool: torch.Tensor,
        op_prior: torch.Tensor,
        problem,
        remaining: float,
        stagnation: torch.Tensor,
        k: int,
        op_penalty: Optional[torch.Tensor] = None,
        candidate_ops: Optional[torch.Tensor] = None,
        state_raw: Optional[torch.Tensor] = None,
        fitness: Optional[torch.Tensor] = None,
        used: Optional[float] = None,
        use_learned_router: bool = True,
        disable_online_proxy: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        b, m, d = pool.shape
        self.last_router_prob = None
        lb, ub = _get_bounds(problem, pool)
        px = _normalize_x(pool.detach(), lb, ub)
        ax = _normalize_x(archive_x.detach(), lb, ub)
        dist_to_archive = torch.cdist(px, ax).amin(dim=2)
        best = ax[:, :1]
        dist_to_best = torch.norm(px - best, p=2, dim=2) / math.sqrt(float(d))

        rem = float(remaining)
        stag = stagnation.view(b, 1).to(device=pool.device, dtype=pool.dtype)
        trust_penalty = (1.0 - rem) ** 2 * (1.0 - 0.7 * stag) * dist_to_best
        novelty_bonus = (0.15 + 0.35 * rem + 0.35 * stag) * dist_to_archive
        log_prior_bonus = 0.03 * torch.log(op_prior.clamp_min(EPS))
        if "local_op_boost" in self.ablation and _is_local_proxy_friendly_problem(problem):
            trust_penalty = 0.55 * trust_penalty
            novelty_bonus = 0.65 * novelty_bonus
            log_prior_bonus = 0.06 * torch.log(op_prior.clamp_min(EPS))

        if "no_surrogate" in self.ablation:
            mu = torch.zeros((b, m), device=pool.device, dtype=pool.dtype)
            sigma = torch.zeros_like(mu)
            acquisition = trust_penalty - novelty_bonus - log_prior_bonus
            acquisition = acquisition + 1e-4 * torch.rand_like(acquisition)
        else:
            distance_scale = 0.15
            if "weak_uncertainty" in self.ablation:
                distance_scale = 0.03
            if "no_distance_uncertainty" in self.ablation:
                distance_scale = 0.0
            mu, sigma = self.surrogate.predict(
                archive_x.detach(),
                archive_y.detach(),
                pool.detach(),
                problem,
                distance_scale=distance_scale,
            )
            kappa = torch.exp(self.log_kappa).clamp(0.15, 4.0)
            if "mean_only" in self.ablation or "no_uncertainty" in self.ablation:
                sigma_term = torch.zeros_like(sigma)
            else:
                sigma_term = sigma
            if "weak_uncertainty" in self.ablation:
                sigma_term = 0.25 * sigma_term
            if "no_novelty" in self.ablation:
                novelty_bonus = torch.zeros_like(novelty_bonus)
            if "no_trust_penalty" in self.ablation:
                trust_penalty = torch.zeros_like(trust_penalty)
            proxy_acquisition = mu - kappa * sigma_term + trust_penalty - novelty_bonus - log_prior_bonus
            if (
                "bbob_guarded_proxy" in self.ablation
                and not _is_local_proxy_friendly_problem(problem)
                and not _is_structured_system_problem(problem)
                and not _is_embedded_high_dim_problem(problem)
            ):
                structural_acquisition = trust_penalty - novelty_bonus - log_prior_bonus
                archive_scale = archive_y.std(dim=1, keepdim=True, unbiased=False).clamp_min(EPS)
                proxy_scale = proxy_acquisition.std(dim=1, keepdim=True, unbiased=False).clamp_min(EPS)
                struct_scale = structural_acquisition.std(dim=1, keepdim=True, unbiased=False).clamp_min(EPS)
                proxy_z = (proxy_acquisition - proxy_acquisition.amin(dim=1, keepdim=True)) / proxy_scale
                struct_z = (structural_acquisition - structural_acquisition.amin(dim=1, keepdim=True)) / struct_scale
                conflict = (struct_z - proxy_z).clamp_min(0.0).clamp_max(3.0)
                weak_proxy = (proxy_z / 0.75).clamp(0.0, 1.0)
                acquisition = proxy_acquisition + 0.020 * archive_scale * conflict * weak_proxy
            else:
                acquisition = proxy_acquisition
            if disable_online_proxy is not None:
                structural_acquisition = trust_penalty - novelty_bonus - log_prior_bonus
                disable_mask = disable_online_proxy.to(device=pool.device, dtype=torch.bool).view(b, 1)
                acquisition = torch.where(disable_mask, structural_acquisition, acquisition)

        if "no_neural" in self.ablation:
            neural_mask = (torch.arange(m, device=pool.device) // self.pool_per_op).view(1, -1) == 0
            acquisition = acquisition.masked_fill(neural_mask, float("inf"))
        if (
            "productive_op_router" in self.ablation
            and not _is_hpo_problem(problem)
            and not _is_embedded_high_dim_problem(problem)
            and m == self.num_ops * self.pool_per_op
        ):
            op = (torch.arange(m, device=pool.device) // self.pool_per_op).view(1, -1)
            restart_idx = self.op_names.index("opposition_restart")
            restart_penalty = 0.08 * archive_y.std(dim=1, keepdim=True, unbiased=False).clamp_min(EPS)
            acquisition = acquisition + torch.where(
                op == restart_idx,
                restart_penalty.expand_as(acquisition),
                torch.zeros_like(acquisition),
            )
        if op_penalty is not None and m == self.num_ops * self.pool_per_op:
            op = (torch.arange(m, device=pool.device) // self.pool_per_op).view(1, -1)
            acquisition = acquisition + torch.gather(op_penalty, 1, op.expand(pool.size(0), -1))
        if "local_op_boost" in self.ablation and _is_local_proxy_friendly_problem(problem) and m == self.num_ops * self.pool_per_op:
            op = (torch.arange(m, device=pool.device) // self.pool_per_op).view(1, -1)
            trust_idx = self.op_names.index("trust_region")
            coord_idx = self.op_names.index("coordinate_pattern")
            pbest_idx = self.op_names.index("current_to_pbest_archive")
            scale = archive_y.std(dim=1, keepdim=True, unbiased=False).clamp_min(EPS)
            local_bonus = torch.zeros_like(acquisition)
            local_bonus = torch.where(op == trust_idx, 0.030 * scale.expand_as(local_bonus), local_bonus)
            local_bonus = torch.where(op == coord_idx, 0.020 * scale.expand_as(local_bonus), local_bonus)
            local_bonus = torch.where(op == pbest_idx, 0.010 * scale.expand_as(local_bonus), local_bonus)
            acquisition = acquisition - local_bonus
        if (
            "learned_router" in self.ablation
            and use_learned_router
            and self.router_model is not None
            and candidate_ops is not None
            and state_raw is not None
            and fitness is not None
            and used is not None
            and not _is_hpo_problem(problem)
            and not _is_structured_system_problem(problem)
            and (not _is_embedded_high_dim_problem(problem) or "highdim_residual_router" in self.ablation)
        ):
            prob = self._router_probability(
                candidate_ops=candidate_ops,
                state_raw=state_raw,
                fitness=fitness,
                score=acquisition.detach(),
                mu=mu.detach(),
                sigma=sigma.detach(),
                used=float(used),
                remaining=remaining,
                stagnation=stagnation,
            )
            if prob is not None:
                self.last_router_prob = prob.detach()
                if "router_veto" not in self.ablation and "weak_router_veto" not in self.ablation:
                    scale = archive_y.std(dim=1, keepdim=True, unbiased=False).clamp_min(EPS)
                    if "residual_safety_gate" in self.ablation:
                        prob_std = prob.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-6)
                        residual_signal = ((prob - prob.mean(dim=1, keepdim=True)) / prob_std).clamp(-2.5, 2.5)
                        residual_weight = self.router_weight if self.router_weight > 0 else 0.004
                        if (
                            "bbob_guarded_proxy" in self.ablation
                            and not _is_local_proxy_friendly_problem(problem)
                            and not _is_structured_system_problem(problem)
                            and not _is_embedded_high_dim_problem(problem)
                        ):
                            residual_weight = self.router_weight if self.router_weight > 0 else 0.004
                        acquisition = acquisition - residual_weight * scale * residual_signal
                    else:
                        acquisition = acquisition - self.router_weight * scale * prob

        selected = torch.topk(acquisition, k=k, dim=1, largest=False).indices
        self.last_acquisition = acquisition.detach()
        self.last_surrogate_mu = mu.detach()
        self.last_surrogate_sigma = sigma.detach()
        return selected

    @torch.no_grad()
    def _baseline_candidates(self, x: torch.Tensor, fitness: torch.Tensor, problem, rest: int) -> Optional[torch.Tensor]:
        if self.baseline_model is None:
            return None
        was_training = self.baseline_model.training
        self.baseline_model.eval()
        try:
            cand = self.baseline_model.generator(x, problem, fitness)
        finally:
            self.baseline_model.train(was_training)
        return _repair(problem, cand[:, :rest, :].to(device=x.device, dtype=x.dtype))

    def forward(self, x: torch.Tensor, problem):
        self.evalnum = 0
        self.trail = None
        self.all_k_candidates = []
        self.gate_history = []
        self.teacher_history = []
        self.fused_history = []
        self.candidate_samples = []
        self.pool_candidate_samples = []
        self._reset_origin_stats()

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
            pool, op_ids, op_prior = self._candidate_pool(x, fitness, problem, state, state_raw, success_logits)
            if "dynamic_highdim_op_adapter" in self.ablation and _is_embedded_high_dim_problem(problem):
                op_prior = self._dynamic_highdim_prior(
                    op_prior=op_prior,
                    op_ids=op_ids,
                    op_seen=op_seen,
                    op_admit=op_admit,
                    op_best=op_best,
                    remaining=remaining,
                    stagnation=stagnation,
                )
            if "trajectory_highdim_adapter" in self.ablation and _is_embedded_high_dim_problem(problem):
                op_prior = self._trajectory_highdim_prior(
                    op_prior=op_prior,
                    op_ids=op_ids,
                    state_raw=state_raw,
                    remaining=remaining,
                    stagnation=stagnation,
                )
            if "validated_highdim_adapter" in self.ablation and _is_embedded_high_dim_problem(problem):
                op_prior = self._validated_highdim_prior(
                    op_prior=op_prior,
                    op_ids=op_ids,
                    op_seen=op_seen,
                    op_best=op_best,
                    state_raw=state_raw,
                    remaining=remaining,
                    stagnation=stagnation,
                )
            if "gated_highdim_adapter" in self.ablation and _is_embedded_high_dim_problem(problem):
                op_prior = self._gated_highdim_prior(
                    op_prior=op_prior,
                    op_ids=op_ids,
                    op_seen=op_seen,
                    op_best=op_best,
                    state_raw=state_raw,
                    fitness=fitness,
                    used=used,
                    remaining=remaining,
                    stagnation=stagnation,
                    step_idx=len(all_candidates),
                    eval_before=self.evalnum,
                )
            if "gated_rescue_highdim_adapter" in self.ablation and _is_embedded_high_dim_problem(problem):
                op_prior = self._gated_rescue_highdim_prior(
                    op_prior=op_prior,
                    op_ids=op_ids,
                    op_seen=op_seen,
                    op_best=op_best,
                    state_raw=state_raw,
                    fitness=fitness,
                    used=used,
                    remaining=remaining,
                    stagnation=stagnation,
                    step_idx=len(all_candidates),
                    eval_before=self.evalnum,
                )
            if "highdim_restart_probe" in self.ablation and _is_embedded_high_dim_problem(problem):
                op_prior = self._fixed_highdim_probe_prior(op_prior, op_ids, mode="restart")
            if "highdim_trust_probe" in self.ablation and _is_embedded_high_dim_problem(problem):
                op_prior = self._fixed_highdim_probe_prior(op_prior, op_ids, mode="trust")
            if "highdim_mixed_probe" in self.ablation and _is_embedded_high_dim_problem(problem):
                op_prior = self._fixed_highdim_probe_prior(op_prior, op_ids, mode="mixed")
            op_penalty = None
            if "highdim_adaptive_op_router" in self.ablation and _is_embedded_high_dim_problem(problem):
                admit_rate = op_admit / op_seen.clamp_min(1.0)
                no_admit = (op_seen >= 12.0) & (op_admit <= 0.0)
                weak_admit = (op_seen >= 24.0) & (admit_rate < 0.05)
                scale = 0.12 * archive_y.std(dim=1, keepdim=True, unbiased=False).clamp_min(EPS)
                op_penalty = torch.where(no_admit | weak_admit, scale.expand_as(op_seen), torch.zeros_like(op_seen))
            if (
                "adaptive_proxy_disable" in self.ablation
                and not _is_local_proxy_friendly_problem(problem)
                and not _is_structured_system_problem(problem)
                and not _is_embedded_high_dim_problem(problem)
            ):
                admit_rate = op_admit / op_seen.clamp_min(1.0)
                no_admit = (op_seen >= 12.0) & (op_admit <= 0.0)
                weak_admit = (op_seen >= 24.0) & (admit_rate < 0.05) & (op_best <= 0.0)
                scale = 0.18 * archive_y.std(dim=1, keepdim=True, unbiased=False).clamp_min(EPS)
                adaptive_penalty = torch.where(no_admit | weak_admit, scale.expand_as(op_seen), torch.zeros_like(op_seen))
                op_penalty = adaptive_penalty if op_penalty is None else op_penalty + adaptive_penalty
            selected = self._select_by_acquisition(
                archive_x,
                archive_y,
                pool,
                op_prior,
                problem,
                remaining,
                stagnation,
                rest,
                op_penalty=op_penalty,
                candidate_ops=op_ids.view(1, -1).expand(x.size(0), -1),
                state_raw=state_raw,
                fitness=fitness,
                used=used,
                use_learned_router=(
                    "pool_rank_router" in self.ablation
                    or "protected_residual_selector" in self.ablation
                    or "highdim_residual_router" in self.ablation
                ),
                disable_online_proxy=None,
            )
            pool_score = self.last_acquisition
            pool_mu = self.last_surrogate_mu
            pool_sigma = self.last_surrogate_sigma
            selected_pool_cand = torch.gather(pool, 1, selected.unsqueeze(-1).expand(-1, -1, x.size(-1)))
            selected_ops = op_ids[selected]
            selected_score = torch.gather(pool_score, 1, selected) if pool_score is not None else None
            selected_mu = torch.gather(pool_mu, 1, selected) if pool_mu is not None else None
            selected_sigma = torch.gather(pool_sigma, 1, selected) if pool_sigma is not None else None

            baseline_cand = self._baseline_candidates(x, fitness, problem, rest)
            if self.log_pool_candidates:
                pool_log_score = pool_score
                pool_log_mu = pool_mu
                pool_log_sigma = pool_sigma
                baseline_log_score = None
                baseline_log_mu = None
                baseline_log_sigma = None
                if baseline_cand is not None:
                    log_baseline = baseline_cand[:, :rest]
                    log_pool = torch.cat([log_baseline, pool], dim=1)
                    log_prior = torch.ones((x.size(0), log_pool.size(1)), device=x.device, dtype=x.dtype)
                    self._select_by_acquisition(
                        archive_x,
                        archive_y,
                        log_pool,
                        log_prior,
                        problem,
                        remaining,
                        stagnation,
                        1,
                        candidate_ops=torch.cat(
                            [
                                torch.full((x.size(0), log_baseline.size(1)), -2, device=x.device, dtype=torch.long),
                                op_ids.view(1, -1).expand(x.size(0), -1),
                            ],
                            dim=1,
                        ),
                        state_raw=state_raw,
                        fitness=fitness,
                        used=used,
                        use_learned_router=False,
                    )
                    log_score = self.last_acquisition
                    log_mu = self.last_surrogate_mu
                    log_sigma = self.last_surrogate_sigma
                    baseline_log_score = log_score[:, : log_baseline.size(1)]
                    baseline_log_mu = log_mu[:, : log_baseline.size(1)]
                    baseline_log_sigma = log_sigma[:, : log_baseline.size(1)]
                    pool_log_score = log_score[:, log_baseline.size(1) :]
                    pool_log_mu = log_mu[:, log_baseline.size(1) :]
                    pool_log_sigma = log_sigma[:, log_baseline.size(1) :]
                self._record_pool_candidate_samples(
                    step_idx=len(all_candidates),
                    eval_before=self.evalnum,
                    used=used,
                    remaining=remaining,
                    stagnation=stagnation,
                    baseline_cand=baseline_cand[:, :rest] if baseline_cand is not None else None,
                    pool=pool,
                    pool_ops=op_ids,
                    baseline_score=baseline_log_score,
                    baseline_mu=baseline_log_mu,
                    baseline_sigma=baseline_log_sigma,
                    pool_score=pool_log_score,
                    pool_mu=pool_log_mu,
                    pool_sigma=pool_log_sigma,
                    problem=problem,
                    fitness=fitness,
                    state_raw=state_raw,
                )
            cand_score = selected_score
            cand_mu = selected_mu
            cand_sigma = selected_sigma
            cand_router_prob = None
            if baseline_cand is not None and rest > 0:
                if "baseline_only" in self.ablation:
                    cand = baseline_cand[:, :rest]
                    selected_ops = torch.full((x.size(0), rest), -2, device=x.device, dtype=torch.long)
                    cand_score = None
                    cand_mu = None
                    cand_sigma = None
                    cand_router_prob = None
                elif "baseline_backbone" in self.ablation and remaining > getattr(self, "intervention_remaining", 0.30):
                    cand = baseline_cand[:, :rest]
                    selected_ops = torch.full((x.size(0), rest), -2, device=x.device, dtype=torch.long)
                    cand_score = None
                    cand_mu = None
                    cand_sigma = None
                    cand_router_prob = None
                else:
                    cand_parts = [baseline_cand[:, :1]]
                    op_parts = [torch.full((x.size(0), 1), -2, device=x.device, dtype=torch.long)]
                    nan_score = torch.full((x.size(0), 1), float("nan"), device=x.device, dtype=x.dtype)
                    score_parts = [nan_score]
                    mu_parts = [nan_score]
                    sigma_parts = [nan_score]
                    router_parts = [nan_score]
                    if rest > 1:
                        combined = torch.cat([baseline_cand[:, 1:rest], selected_pool_cand], dim=1)
                        prior_extra = torch.ones((x.size(0), combined.size(1)), device=x.device, dtype=x.dtype)
                        combined_ops = torch.cat(
                            [
                                torch.full((x.size(0), max(rest - 1, 0)), -2, device=x.device, dtype=torch.long),
                                selected_ops,
                            ],
                            dim=1,
                        )
                        extra_idx = self._select_by_acquisition(
                            archive_x,
                            archive_y,
                            combined,
                            prior_extra,
                            problem,
                            remaining,
                            stagnation,
                            rest - 1,
                            candidate_ops=combined_ops,
                            state_raw=state_raw,
                            fitness=fitness,
                            used=used,
                            use_learned_router=(
                                "residual_safety_gate" in self.ablation
                                or "protected_residual_selector" not in self.ablation
                            ),
                            disable_online_proxy=None,
                        )
                        combined_score = self.last_acquisition
                        combined_mu = self.last_surrogate_mu
                        combined_sigma = self.last_surrogate_sigma
                        combined_router_prob = self.last_router_prob
                        proposed_extra_idx = extra_idx
                        if "baseline_safe_selector" in self.ablation:
                            base_count = max(rest - 1, 0)
                            if base_count > 0 and self.last_acquisition is not None:
                                acq = self.last_acquisition
                                base_acq, base_idx = acq[:, :base_count].min(dim=1, keepdim=True)
                                if "structured_system_lock" in self.ablation and _is_structured_system_problem(problem):
                                    extra_idx = base_idx
                                else:
                                    chosen_acq = torch.gather(acq, 1, extra_idx)
                                    chosen_is_portfolio = extra_idx >= base_count
                                    chosen_sigma = (
                                        torch.gather(self.last_surrogate_sigma, 1, extra_idx)
                                        if self.last_surrogate_sigma is not None
                                        else torch.zeros_like(chosen_acq)
                                    )
                                    margin = 0.10 * archive_y.std(dim=1, keepdim=True, unbiased=False).clamp_min(EPS)
                                    safe = chosen_is_portfolio & (chosen_acq < base_acq - margin)
                                    if "residual_safety_gate" in self.ablation and self.last_router_prob is not None:
                                        router_prob = self.last_router_prob
                                        base_prob = router_prob[:, :base_count].max(dim=1, keepdim=True).values
                                        chosen_prob = torch.gather(router_prob, 1, extra_idx)
                                        prob_std = router_prob.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-6)
                                        prob_mean = router_prob.mean(dim=1, keepdim=True)
                                        base_signal = (base_prob - prob_mean) / prob_std
                                        chosen_signal = (chosen_prob - prob_mean) / prob_std
                                        residual_blocks = chosen_is_portfolio & (
                                            (chosen_signal + 0.35 < base_signal) & (chosen_signal < -0.25)
                                        )
                                        enough_portfolio_evidence = portfolio_seen.view(-1, 1) >= 8.0
                                        no_portfolio_best = portfolio_best_count.view(-1, 1) <= 0.0
                                        residual_blocks = residual_blocks & enough_portfolio_evidence & no_portfolio_best
                                        residual_supports = (~chosen_is_portfolio) | (
                                            (chosen_signal > base_signal + 0.55) | (chosen_signal > 1.00)
                                        )
                                        soft_margin = 0.04 * archive_y.std(dim=1, keepdim=True, unbiased=False).clamp_min(EPS)
                                        residual_rescue = chosen_is_portfolio & residual_supports & (chosen_acq < base_acq - soft_margin)
                                        safe = (safe & (~residual_blocks)) | residual_rescue
                                        if "adaptive_proxy_disable" in self.ablation and not _is_local_proxy_friendly_problem(problem):
                                            proxy_exploded = chosen_is_portfolio & (chosen_acq < -1.0e3)
                                            safe = safe & (~proxy_exploded)
                                    if "router_veto" in self.ablation and self.last_router_prob is not None:
                                        router_prob = self.last_router_prob
                                        base_prob = router_prob[:, :base_count].max(dim=1, keepdim=True).values
                                        chosen_prob = torch.gather(router_prob, 1, extra_idx)
                                        router_allows = (~chosen_is_portfolio) | (chosen_prob > base_prob + 0.03)
                                        safe = safe & router_allows
                                    if "weak_router_veto" in self.ablation and self.last_router_prob is not None:
                                        router_prob = self.last_router_prob
                                        base_prob = router_prob[:, :base_count].max(dim=1, keepdim=True).values
                                        chosen_prob = torch.gather(router_prob, 1, extra_idx)
                                        router_blocks = chosen_is_portfolio & (chosen_prob < 0.35) & (chosen_prob + 0.05 < base_prob)
                                        safe = safe & (~router_blocks)
                                    if "admitted_no_best_lock" in self.ablation and not _is_hpo_problem(problem) and not _is_embedded_high_dim_problem(problem):
                                        no_best = portfolio_best_count.view(-1, 1) <= 0.0
                                        enough_seen = portfolio_seen.view(-1, 1) >= 12.0
                                        enough_admitted = portfolio_admit_count.view(-1, 1) >= 3.0
                                        late = torch.full_like(no_best, float(remaining) < 0.20, dtype=torch.bool)
                                        lock = chosen_is_portfolio & no_best & enough_seen & enough_admitted & late
                                        safe = safe & (~lock)
                                    if "aggressive_takeover" in self.ablation and not _is_hpo_problem(problem) and not _is_embedded_high_dim_problem(problem):
                                        stag = stagnation.view(-1, 1).to(device=x.device, dtype=x.dtype)
                                        late = torch.full(stag.shape, float(remaining) < 0.18, device=x.device, dtype=torch.bool)
                                        stalled = stag > 0.04
                                        admit_rate = (portfolio_admit_count / portfolio_seen.clamp_min(1.0)).view(-1, 1)
                                        has_portfolio_signal = (portfolio_seen.view(-1, 1) >= 6.0) & (admit_rate > 0.12)
                                        relaxed_margin = 0.05 * archive_y.std(dim=1, keepdim=True, unbiased=False).clamp_min(EPS)
                                        takeover = chosen_is_portfolio & has_portfolio_signal & (late | stalled) & (chosen_acq < base_acq + relaxed_margin)
                                        safe = safe | takeover
                                    extra_idx = torch.where(safe, extra_idx, base_idx)
                                    if "adaptive_portfolio_lock" in self.ablation:
                                        locked = portfolio_no_admit.view(-1, 1) >= 8.0
                                        extra_idx = torch.where(locked, base_idx, extra_idx)
                                    if op_penalty is not None:
                                        chosen_portfolio = extra_idx >= base_count
                                        chosen_port_idx = (extra_idx - base_count).clamp_min(0)
                                        chosen_port_op = torch.gather(selected_ops, 1, chosen_port_idx)
                                        op_is_penalized = torch.gather(op_penalty > 0.0, 1, chosen_port_op.clamp_min(0))
                                        extra_idx = torch.where(chosen_portfolio & op_is_penalized, base_idx, extra_idx)
                        observer = getattr(self, "supplement_gate_observer", None)
                        if observer is not None:
                            observer(locals())
                        extra = torch.gather(combined, 1, extra_idx.unsqueeze(-1).expand(-1, -1, x.size(-1)))
                        cand_parts.append(extra)
                        score_parts.append(torch.gather(combined_score, 1, extra_idx) if combined_score is not None else nan_score)
                        mu_parts.append(torch.gather(combined_mu, 1, extra_idx) if combined_mu is not None else nan_score)
                        sigma_parts.append(torch.gather(combined_sigma, 1, extra_idx) if combined_sigma is not None else nan_score)
                        router_parts.append(torch.gather(combined_router_prob, 1, extra_idx) if combined_router_prob is not None else nan_score)
                        extra_ops = torch.where(
                            extra_idx < max(rest - 1, 0),
                            torch.full_like(extra_idx, -2),
                            torch.gather(selected_ops, 1, (extra_idx - max(rest - 1, 0)).clamp_min(0)),
                        )
                        op_parts.append(extra_ops)
                    cand = torch.cat(cand_parts, dim=1)
                    selected_ops = torch.cat(op_parts, dim=1)
                    cand_score = torch.cat(score_parts, dim=1)
                    cand_mu = torch.cat(mu_parts, dim=1)
                    cand_sigma = torch.cat(sigma_parts, dim=1)
                    cand_router_prob = torch.cat(router_parts, dim=1)
            else:
                cand = selected_pool_cand

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
                ("highdim_adaptive_op_router" in self.ablation and _is_embedded_high_dim_problem(problem))
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
        return x, self.trail, self.evalnum, self.all_k_candidates
