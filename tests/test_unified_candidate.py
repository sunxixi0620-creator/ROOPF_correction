import sys
import unittest
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from roopf.unified import build_unified
from roopf.revision_tasks import ProceduralTask, population
from roopf.experiment_io import fingerprint


class CandidateContract(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def build(self, variant='full', budget=300):
        return build_unified(ROOT/'checkpoints/anchor_policy_d10.pt',
            ROOT/'checkpoints/residual_selector_generated36_d10.pt', variant, budget=budget)

    def test_label_blind_and_teacher_parity(self):
        pop = population(7, 0, 'contract', count=2)
        outputs = []
        captures = []
        for label, teacher in [('plain', False), ('cecf1', False), ('hpof1', True)]:
            task = ProceduralTask(7, 0, 'contract')
            task.fun['fid'] = label
            opt = self.build()
            if teacher:
                def observe(features, truth, incumbent, nfe):
                    self.assertEqual(features.shape, (2, 38, 27))
                    self.assertEqual(truth.shape, (2, 38))
                    captures.append((features, truth, incumbent, nfe))
                opt.teacher_observer = observe
                actual_inputs = []
                handle = opt.router_model.register_forward_pre_hook(lambda model, args: actual_inputs.append(args[0].clone()))
                def input_observer(raw, normalized, ops, nfe):
                    self.assertTrue(torch.equal((raw-opt.router_mean)/opt.router_std, normalized))
                    input_observer.pending = normalized
                opt.input_observer = input_observer
                def verify(model, args, output):
                    self.assertTrue(torch.equal(input_observer.pending.reshape(-1, 27), args[0]))
                handle2 = opt.router_model.register_forward_hook(verify)
            torch.manual_seed(19273)
            result = opt(pop.clone(), task)
            self.assertEqual(task.points, 600)
            self.assertEqual(task.diagnostic_points, 7600 if teacher else 0)
            self.assertEqual(result[2], 300)
            self.assertTrue((result[1][:, 1:] <= result[1][:, :-1]).all())
            self.assertFalse(any(row['accepted'] for row in opt.decisions if row['eval_before'] < 108))
            outputs.append((result[1], result[3], torch.get_rng_state(), fingerprint(opt.decisions)))
        for output in outputs[1:]:
            self.assertTrue(all(torch.equal(a, b) for a, b in zip(outputs[0][:3], output[:3])))
            self.assertEqual(outputs[0][3], output[3])
        self.assertEqual(len(captures), 100)

    def test_variants_and_odd_budget(self):
        for variant in ['full', 'no_residual', 'score_only', 'late_full', 'late_no_residual', 'anchor_only']:
            task = ProceduralTask(1, 0, 'contract')
            model = self.build(variant, budget=111)
            result = model(population(1, 0, 'contract', count=1), task)
            self.assertEqual(result[2], 111)
            self.assertEqual(task.points, 111)
            if variant.startswith('late_'):
                self.assertFalse(any(row['accepted'] for row in model.decisions if row['eval_before']/111 < .70))
            if variant in ('no_residual', 'late_no_residual'):
                self.assertFalse(any(row['residual_active'] or row['residual_changed'] for row in model.decisions))

    def test_generator_is_batch_independent_and_differentiable(self):
        for fid in range(36):
            task = ProceduralTask(fid, 0, 'contract')
            x = population(fid, 0, 'contract', count=2).requires_grad_(True)
            y = task.value(x)
            self.assertTrue(torch.allclose(y[0], task.value(x[:1])[0], atol=1e-6, rtol=1e-6))
            y.sum().backward()
            self.assertTrue(torch.isfinite(x.grad).all())


if __name__ == '__main__':
    unittest.main(verbosity=2)
