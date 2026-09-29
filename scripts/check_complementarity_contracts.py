"""Checks matching and absence of offline access before comparative evaluation."""
import sys
from pathlib import Path
from unittest.mock import patch
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from roopf.online_portfolio import build_online
from roopf.unified import build_unified
from roopf.anchor_backbone import AnchorPolicyBackbone
from roopf.model import _sort_pop
from roopf.revision_tasks import ProceduralTask, population
from roopf.experiment_io import write_json


def main():
    torch.set_num_threads(1)
    residual = ROOT/'checkpoints/residual_selector_generated36_d10.pt'
    checks = []
    points = 0
    for dim in (10, 20):
        anchor = (ROOT/'checkpoints/anchor_policy_d10.pt' if dim == 10 else
                  ROOT/'artifacts/unified_external_v1/anchor20_0.pt')
        fusion = build_unified(anchor, residual, 'no_residual', dim=dim, budget=112)
        with patch('torch.load', side_effect=AssertionError('O attempted checkpoint load')):
            online = build_online(dim, 112)
        assert online.baseline_model is None and online.router_model is None
        for name in ('state_encoder', 'neural_generator', 'surrogate', 'gate'):
            a, b = getattr(online, name).state_dict(), getattr(fusion, name).state_dict()
            assert a.keys() == b.keys()
            assert all(torch.equal(a[k], b[k]) for k in a)
        task = ProceduralTask(5, 0, 'complementarity_contract', dim=dim)
        pop = population(5, 0, 'complementarity_contract', count=2, dim=dim)
        fit = task.calfitness(pop)
        x, fit = _sort_pop(pop, fit)
        with torch.no_grad():
            state, raw = fusion.state_encoder(x, fit, task, .5, .5, torch.zeros(2))
            torch.manual_seed(271)
            pool, ids, prior = fusion._candidate_pool(x, fit, task, state, raw, torch.zeros(2, 6))
            torch.manual_seed(271)
            opool, oids, oprior = online._candidate_pool(x, fit, task, state, raw, torch.zeros(2, 6))
            assert torch.equal(pool, opool) and torch.equal(ids, oids) and torch.equal(prior, oprior)
            ops = ids[None].expand(2, -1)
            reference = fusion.inputs_and_scores(x, fit, x, fit, pool, ops, prior,
                task, raw, .5, .5, torch.zeros(2))[4]
            actual = online.scores(pool, prior, x, fit, task, .5, torch.zeros(2))
            assert torch.equal(reference, actual), float((reference-actual).abs().max())
        points += task.points
        # O budget, renamed-task invariance, no hidden loading/teacher queries.
        saved = None
        for rename in (False, True):
            task = ProceduralTask(5, 0, 'complementarity_contract', dim=dim)
            if rename:
                task.fun['fid'] = 'CEC_HPO_arbitrary_name'
            with patch('torch.load', side_effect=AssertionError('O attempted checkpoint load')):
                model = build_online(dim, 301)
                torch.manual_seed(691)
                _, trail, nfe, candidates = model(pop.clone(), task)
            assert nfe == 301 and task.points == 602 and task.diagnostic_points == 0
            assert candidates.shape == (2, 201, dim)
            assert (trail[:, 1:] <= trail[:, :-1]).all()
            if saved is not None:
                assert torch.equal(saved[0], trail) and torch.equal(saved[1], candidates)
            saved = (trail, candidates)
            points += task.points
        # A is numerically the same anchor path used in the controlled ablation.
        a = AnchorPolicyBackbone(dim, 200, 100).eval()
        a.MaxNFE = 112
        a.load_state_dict(torch.load(anchor, map_location='cpu', weights_only=False))
        a.requires_grad_(False)
        reference_a = build_unified(anchor, residual, 'anchor_only', dim=dim, budget=112)
        values = []
        for model in (a, reference_a):
            task = ProceduralTask(5, 0, 'complementarity_contract', dim=dim)
            torch.manual_seed(763)
            with torch.no_grad():
                _, trail, nfe, candidates = model(pop.clone(), task)
            assert nfe == 112 and task.points == 224
            values.append((trail, candidates)); points += task.points
        assert all(torch.equal(p, q) for p, q in zip(*values))
        # Logging full-pool truth cannot change the main path.
        values = []
        for teacher in (False, True):
            task = ProceduralTask(5, 0, 'complementarity_contract', dim=dim)
            model = build_unified(anchor, residual, 'no_residual', dim=dim, budget=112)
            if teacher:
                model.teacher_observer = lambda *args: None
            torch.manual_seed(993)
            _, trail, nfe, candidates = model(pop.clone(), task)
            values.append((trail, candidates, torch.get_rng_state()))
            assert task.diagnostic_points == (456 if teacher else 0)
            points += task.points + task.diagnostic_points
        assert all(torch.equal(p, q) for p, q in zip(*values))
        checks.append(dict(dimension=dim, matching_fixed_modules=True,
            matching_pool_and_score=True, online_no_checkpoint_access=True,
            online_budget_and_name_invariance=True, anchor_path_parity=True,
            teacher_path_and_rng_parity=True))
    out = ROOT/'docs/revision/complementarity'
    out.mkdir(parents=True, exist_ok=True)
    write_json(out/'CONTRACTS.json', dict(passed=True, checks=checks,
        objective_calls_including_teacher=points, comparative_performance_evaluated=False))
    print('PASS', points, 'contract objective calls; no comparative performance study')


if __name__ == '__main__':
    main()
