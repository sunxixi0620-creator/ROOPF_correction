"""Optional anchor proposals with all evaluation slots ranked jointly.

The forward path copies the frozen OnlinePortfolio control, changing only the
source/identity of its two extra proposals. Anchor-free mode is a parity fixture.
"""
import torch
from .online_portfolio import OnlinePortfolio
from .anchor_backbone import AnchorPolicyBackbone
from .model import _sort_pop, _get_bounds
from .experiment_io import seed_for


class FreeAllocation(OnlinePortfolio):
    def __init__(self, anchor, dim=10, budget=300):
        super().__init__(dim, budget)
        if anchor is not None:
            self.baseline_model = AnchorPolicyBackbone(dim, 200, 100).eval().requires_grad_(False)
            self.baseline_model.load_state_dict(torch.load(anchor, map_location='cpu', weights_only=False), strict=True)

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
            if self.baseline_model is None:
                extra = lb+(ub-lb)*torch.rand((len(x), 2, self.dim), generator=extra_rng, device=x.device)
            else:
                extra = self._baseline_candidates(x, fitness, problem, 2)
            candidates = torch.cat((extra, pool), 1)
            ops = torch.cat((torch.full((len(x), 2), -2 if self.baseline_model is not None else -1, dtype=torch.long, device=x.device),
                             ids[None].expand(len(x), -1)), 1)
            prior = torch.cat((torch.full((len(x), 2), 1/6, device=x.device), priors), 1)
            score = self.scores(candidates, prior, archive_x, archive_y, problem, remaining, stagnation)
            indices = torch.argsort(score, dim=1, stable=True)[:, :rest]
            chosen = candidates.gather(1, indices[:, :, None].expand(-1, -1, self.dim))
            chosen_ops = ops.gather(1, indices)
            for bi in range(len(x)):
                self.decisions.append(dict(batch=bi, step=step, eval_before=self.evalnum,
                    selected=indices[bi].tolist(), chosen_ops=chosen_ops[bi].tolist(),
                    candidate_count=38, anchor_calls=int(self.baseline_model is not None), residual_calls=0))
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


def build_free(anchor, dim=10, budget=300):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(20260630)
        model = FreeAllocation(anchor, dim, budget)
    return model.eval()
