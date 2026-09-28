"""Behavioral checks against frozen code, not just the refactored implementation."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'scripts')]
import run_roopf as r
from roopf.factory import build_final
from roopf.experiment_io import CaseStore, fingerprint, seed_for, sha256, write_json
from supplementary_experiments import problem_for


def frozen_module(name, path, replace=None):
    source = subprocess.check_output(['git', 'show', 'f127b7b:' + path], cwd=ROOT).decode()
    if replace:
        source = source.replace(*replace)
    mod = types.ModuleType(name)
    mod.__package__ = 'roopf'
    mod.__file__ = str(ROOT/path)
    sys.modules[name] = mod
    exec(compile(source, str(ROOT/path), 'exec'), mod.__dict__)
    return mod


class Contracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        r.DEVICE = torch.device('cpu')
        for name in ['roopf.benchmarks.bbobfunctions', 'roopf.benchmarks.cecfunctions',
                     'roopf.benchmarks.utils']:
            sys.modules[name].DEVICE = r.DEVICE
        cls.old = frozen_module('roopf._frozen_model', 'roopf/model.py')
        cls.old_supp = frozen_module('roopf._frozen_supplement', 'roopf/supplement.py',
                                    ('from .model import', 'from ._frozen_model import'))
        cls.results = []

    @classmethod
    def tearDownClass(cls):
        out = ROOT/'results/revision_contract_checks'
        out.mkdir(exist_ok=True)
        write_json(out/'behavior.json', cls.results)

    def test_frozen_behavior_and_entry(self):
        for case in ['bbob_f1_fixed', 'bbob_f9_fixed', 'cec_f3_shifted']:
            for variant in ['full', 'no_residual', 'score_only']:
                problem, _ = problem_for(case)
                pop = problem.genRandomPop((2, 100, 10))
                r.set_seed(20260630)
                old = self.old_supp.SupplementOptimizer(dim=10, hidden_dim=200,
                    popSize=100, max_nfe=300, k_nums=2, pool_per_op=6, surrogate_members=5,
                    ablation='roopf', baseline_ckpt=str(ROOT/'checkpoints/anchor_policy_d10.pt'),
                    router_ckpt=str(ROOT/'checkpoints/residual_selector_generated36_d10.pt'),
                    router_weight=.008).eval().configure(variant)
                new = build_final(variant)
                for k, v in old.state_dict().items():
                    self.assertTrue(torch.equal(v, new.state_dict()[k]), k)
                results = []
                for model in [old, new]:
                    problem, _ = problem_for(case)
                    r.set_seed(20281101)
                    with torch.no_grad():
                        _, trail, nfe, points = model(pop.clone(), problem)
                    self.assertEqual(nfe, 300)
                    self.assertEqual(problem.actual, 300)
                    results.append((trail, points, torch.get_rng_state()))
                self.assertTrue(all(torch.equal(a, b) for a, b in zip(*results)), (case, variant))
                self.results.append({'case': case, 'variant': variant, 'nfe': 300,
                                     'trail_points_rng_exact': True})
        args = types.SimpleNamespace(profile='final_unlocked', dimension=10,
                                     budget=300, population_size=100)
        cli = r.build_optimizer(args)
        direct = build_final()
        self.assertEqual(cli.ablation, direct.ablation)
        self.assertEqual(cli.intervention_remaining, direct.intervention_remaining)
        self.assertTrue(all(torch.equal(v, direct.state_dict()[k]) for k, v in cli.state_dict().items()))

    def test_exact_feature_logging_no_interference(self):
        problem, _ = problem_for('bbob_f9_fixed')
        pop = problem.genRandomPop((2, 100, 10))
        captures = []
        model = build_final()
        def observe(raw, normalized, ops, nfe):
            expected = (raw - model.router_mean.view(1, 1, -1)) / model.router_std.view(1, 1, -1)
            self.assertTrue(torch.equal(expected, normalized))
            # Round trip through the actual training artifact representation.
            self.assertTrue(torch.equal(torch.from_numpy(raw.cpu().numpy().copy()), raw.cpu()))
            captures.append((raw, normalized, ops, nfe))
        results = []
        for enabled in [False, True]:
            model.residual_input_observer = observe if enabled else None
            problem, _ = problem_for('bbob_f9_fixed')
            r.set_seed(20281103)
            with torch.no_grad():
                _, trail, nfe, points = model(pop.clone(), problem)
            results.append((trail.clone(), points.clone(), torch.get_rng_state()))
            self.assertEqual(problem.actual, 300)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(*results)))
        self.assertEqual({x[0].shape[1] for x in captures}, {36, 3})
        self.results.append({'exact_input_events': len(captures), 'logging_no_interference': True})

    def test_cache_rejects_stale_and_corrupt(self):
        with tempfile.TemporaryDirectory() as name:
            store = CaseStore(name, {'code': 'abc', 'weights': 'xyz'})
            identity = {'seed': seed_for('training', 'population', 0), 'pop': torch.ones(2, 3)}
            with store.lock('case_0'):
                self.assertIsNone(store.load('case_0', identity))
                store.save('case_0', identity, {'x': torch.ones(3)})
                self.assertTrue(torch.equal(store.load('case_0', identity)['x'], torch.ones(3)))
                with self.assertRaises(ValueError):
                    store.load('case_0', {**identity, 'seed': 0})
                (Path(name)/'case_0.pt').write_bytes(b'corrupt')
                with self.assertRaises(ValueError):
                    store.load('case_0', identity)
            with self.assertRaises(ValueError):
                CaseStore(name, {'code': 'different'})
            self.assertNotEqual(seed_for('training', 'policy', 0), identity['seed'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
