"""Experimental online-context-conditioned offline proposal module.

No truth from unevaluated points enters this module. Training teachers live in
the experiment driver, never in the search implementation.
"""
import torch
from torch import nn
from .anchor_backbone import AnchorPolicyBackbone
from .online_portfolio import OnlinePortfolio


class ComplementaryProposal(nn.Module):
    def __init__(self, anchor, dim=20):
        super().__init__()
        self.backbone = AnchorPolicyBackbone(dim, 200, 100)
        self.backbone.load_state_dict(torch.load(anchor, map_location='cpu', weights_only=False))
        self.context = nn.Sequential(nn.Linear(2*dim+4, 200), nn.GELU(), nn.Linear(200, 2*dim))
        nn.init.zeros_(self.context[-1].weight)
        nn.init.zeros_(self.context[-1].bias)
        self.dim = dim

    def forward(self, x, fitness, intent, intent_scores, remaining, stagnation, problem):
        base = self.backbone.generator(x, problem, fitness)
        rem = remaining if torch.is_tensor(remaining) else x.new_full((len(x),), remaining)
        features = torch.cat(((intent-x[:, :1]).flatten(1)/10,
            intent_scores.clamp(-10, 10),
            rem[:, None], stagnation[:, None]), 1)
        correction = .5*torch.tanh(self.context(features)).reshape(-1, 2, self.dim)
        return problem.repaire(base+correction)


class ContextPortfolio(OnlinePortfolio):
    """Same O proposal stream/pool/ranker; substitutes only its two extra points.

    Intent is O's original choice BEFORE substitution, using surrogate scores.
    The second scoring call uses the identical archive and no new observations.
    """
    def __init__(self, proposal=None, anchor=None, dim=20, budget=600, capture=False):
        super().__init__(dim, budget)
        self.proposal = proposal
        self.old_anchor = None
        if anchor is not None:
            self.old_anchor = AnchorPolicyBackbone(dim, 200, 100).eval().requires_grad_(False)
            self.old_anchor.load_state_dict(torch.load(anchor, map_location='cpu', weights_only=False))
        self.capture = capture
        self.snapshots = []
        self.extra_selected = 0

    def _candidate_pool(self, x, fitness, problem, state, raw, memory):
        self.current_x, self.current_fitness = x, fitness
        return super()._candidate_pool(x, fitness, problem, state, raw, memory)

    def scores(self, candidates, prior, archive_x, archive_y, problem, remaining, stagnation):
        score = super().scores(candidates, prior, archive_x, archive_y, problem, remaining, stagnation)
        order = torch.argsort(score, dim=1, stable=True)[:, :2]
        intent = candidates.gather(1, order[:, :, None].expand(-1, -1, self.dim)).clone()
        intent_score = score.gather(1, order).clone()
        step = (self.evalnum-100)//2
        if self.capture and step in (0, 35, 70, 105, 140, 175, 210, 245):
            self.snapshots.append(dict(x=self.current_x.clone(), fitness=self.current_fitness.clone(),
                intent=intent, intent_scores=intent_score, remaining=remaining,
                stagnation=stagnation.clone(), step=step))
        if self.proposal is not None:
            candidates[:, :2] = self.proposal(self.current_x, self.current_fitness,
                intent, intent_score, remaining, stagnation, problem)
        elif self.old_anchor is not None:
            candidates[:, :2] = self.old_anchor.generator(self.current_x, problem, self.current_fitness)
        else:
            return score
        score = super().scores(candidates, prior, archive_x, archive_y, problem, remaining, stagnation)
        self.extra_selected += int((torch.argsort(score, dim=1, stable=True)[:, :2] < 2).sum())
        return score


def build_context(proposal=None, anchor=None, dim=20, budget=600, capture=False):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(20260630)
        model = ContextPortfolio(proposal, anchor, dim, budget, capture)
    return model.eval()
