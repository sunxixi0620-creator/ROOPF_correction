"""Matched online-only control; constructs no anchor and loads no checkpoint."""
import math
import torch
from torch import nn
from .model import (ROOPFOptimizer, CheapStateEncoder, AmortizedEliteGenerator,
                    RidgeFeatureEnsemble, EPS, _sort_pop, _get_bounds, _normalize_x)
from .experiment_io import seed_for


class OnlinePortfolio(ROOPFOptimizer):
    def __init__(self, dim=10, budget=300):
        # Do not call ROOPFOptimizer.__init__: it mandates loading an anchor.
        nn.Module.__init__(self)
        self.dim, self.popsize, self.MaxNFE = dim, 100, budget
        self.pool_per_op, self.num_ops = 6, 6
        self.op_names = ['amortized_elite', 'current_to_pbest_archive', 'covariance_elite',
                        'trust_region', 'coordinate_pattern', 'opposition_restart']
        self.ablation = set()
        # Exact same construction order/seed as the untrained shared F modules.
        self.state_encoder = CheapStateEncoder(hidden_dim=200)
        self.neural_generator = AmortizedEliteGenerator(dim=dim, hidden_dim=200, pool_per_op=6)
        self.surrogate = RidgeFeatureEnsemble(dim=dim, members=5,
            rff=min(64, max(16, 4*dim)), ridge=2e-3)
        self.gate = nn.Sequential(nn.Linear(206, 200), nn.LayerNorm(200), nn.GELU(), nn.Linear(200, 6))
        self.log_kappa = nn.Parameter(torch.tensor(math.log(1.25)))
        self.de_f_logit = nn.Parameter(torch.tensor(0.))
        self.trust_logit = nn.Parameter(torch.tensor(-1.5))
        self.cov_logit = nn.Parameter(torch.tensor(-1.))
        self.coord_logit = nn.Parameter(torch.tensor(-1.5))
        self.restart_logit = nn.Parameter(torch.tensor(-2.))
        self.baseline_model = None
        self.router_model = None

    def scores(self, candidates, prior, archive_x, archive_y, problem, remaining, stagnation):
        mu, sigma = self.surrogate.predict(archive_x, archive_y, candidates, problem, distance_scale=.03)
        scale = archive_y.std(1, keepdim=True, unbiased=False).clamp_min(EPS)
        center = archive_y.mean(1, keepdim=True)
        lb, ub = _get_bounds(problem, candidates)
        cx, ax = _normalize_x(candidates, lb, ub), _normalize_x(archive_x, lb, ub)
        best_idx = archive_y.argmin(1)
        best = ax[torch.arange(len(ax), device=ax.device), best_idx].unsqueeze(1)
        trust = (1-remaining)**2*(1-.7*stagnation[:, None])*torch.norm(cx-best, dim=-1)/math.sqrt(self.dim)
        novelty = (.15+.35*remaining+.35*stagnation[:, None])*torch.cdist(cx, ax).amin(2)
        return (mu-center)/scale-.3125*sigma/scale+trust-novelty-.03*prior.clamp_min(EPS).log()

    @torch.no_grad()
    def forward(self, x, problem):
        self.evalnum = 0
        self.teacher_points = 0
        self.decisions = []
        fitness = problem.calfitness(x)
        self.evalnum += x.shape[1]
        x, fitness = _sort_pop(problem.repaire(x), fitness)
        archive_x, archive_y = x, fitness
        best_before = fitness[:, 0]
        no_improve = torch.zeros_like(best_before)
        memory = x.new_zeros((len(x), 6))
        trails, points = [], []
        # Separate proposal stream does not change the common36 pool RNG stream.
        extra_rng = torch.Generator(device=x.device).manual_seed(
            seed_for('online_only_uniform_slots', torch.initial_seed(), self.dim))
        for step in range((self.MaxNFE-self.popsize+1)//2):
            rest = min(2, self.MaxNFE-self.evalnum)
            remaining = (self.MaxNFE-self.evalnum)/self.MaxNFE
            used = self.evalnum/self.MaxNFE
            stagnation = (no_improve/max(1., (self.MaxNFE-self.popsize)/2)).clamp(0, 1)
            state, raw = self.state_encoder(x, fitness, problem, remaining, used, stagnation)
            pool, ids, priors = self._candidate_pool(x, fitness, problem, state, raw, memory)
            lb, ub = _get_bounds(problem, x)
            extra = lb+(ub-lb)*torch.rand((len(x), 2, self.dim), generator=extra_rng, device=x.device)
            candidates = torch.cat((extra, pool), 1)
            ops = torch.cat((torch.full((len(x), 2), -1, dtype=torch.long, device=x.device),
                             ids[None].expand(len(x), -1)), 1)
            prior = torch.cat((torch.full((len(x), 2), 1/6, device=x.device), priors), 1)
            score = self.scores(candidates, prior, archive_x, archive_y, problem, remaining, stagnation)
            indices = torch.argsort(score, dim=1, stable=True)[:, :rest]
            chosen = candidates.gather(1, indices[:, :, None].expand(-1, -1, self.dim))
            chosen_ops = ops.gather(1, indices)
            for bi in range(len(x)):
                self.decisions.append(dict(batch=bi, step=step, eval_before=self.evalnum,
                    selected=indices[bi].tolist(), chosen_ops=chosen_ops[bi].tolist(),
                    candidate_count=38, anchor_calls=0, residual_calls=0))
            values = torch.cat([problem.calfitness(chosen[:, i:i+1]) for i in range(rest)], 1)
            self.evalnum += rest
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
            x, fitness = x[:, :100], fitness[:, :100]
            no_improve = torch.where(fitness[:, 0] < best_before-1e-12, 0., no_improve+1)
            best_before = fitness[:, 0]
            trails.append(best_before[:, None])
            points.append(chosen)
        self.trail = torch.cat(trails, 1)
        self.all_k_candidates = torch.cat(points, 1)
        return x, self.trail, self.evalnum, self.all_k_candidates


def build_online(dim=10, budget=300):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(20260630)
        model = OnlinePortfolio(dim, budget)
    return model.eval()
