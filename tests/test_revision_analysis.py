"""Verify paired analysis direction and stop decisions on known synthetic data."""
import sys
import tempfile
import unittest
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'scripts')]
from unified_revision import summarize
from roopf.unified import VARIANTS


class Analysis(unittest.TestCase):
    def test_constant_pairwise_effect(self):
        rows = []
        effects = dict(anchor_only=0, full=.02, no_residual=.01, score_only=-.03,
                       late_full=.012, late_no_residual=.009)
        for seed in range(3):
            for fid in range(36):
                for instance in range(2):
                    for population in range(4):
                        for method in VARIANTS:
                            rows.append(dict(model_seed=seed, fid=fid, instance=instance,
                                population=population, method=method, utility=.2+effects[method],
                                early_utility=.1, accepted=0, early_accepted=0, residual_changed=0))
        with tempfile.TemporaryDirectory() as name:
            result = summarize(Path(name), rows)
        row = result['comparisons']['full_vs_anchor_only']
        self.assertAlmostEqual(row['mean'], .02)
        self.assertTrue(all(abs(v-.02) < 1e-10 for v in row['ci95']))
        self.assertTrue(result['residual_independent_gain'])
        self.assertTrue(result['protection_supported'])
        self.assertEqual(result['surviving_candidate'], 'full')


if __name__ == '__main__':
    unittest.main(verbosity=2)
