import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import argparse
import fcntl
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT))

import experiment_r04_full as run
from roopf.adaptive_fusion_agent import AdaptiveFusionAgent

EXP = ROOT / 'experiments/r05_adaptive_evidence_fusion'
run.EXP = EXP
run.JOURNAL = EXP / 'api_journal'
run.MultiStepFusionAgent = AdaptiveFusionAgent
run.SOURCES = tuple(run.SOURCES) + ('roopf/adaptive_fusion_agent.py',
    'scripts/experiment_r05_full.py', 'scripts/verify_r05.py', 'scripts/finalize_r05_full.py')


class FullAdaptiveAgent(AdaptiveFusionAgent):
    def call(self, payload):
        run.JOURNAL.mkdir(parents=True, exist_ok=True)
        with (run.JOURNAL / 'API.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            self.budget['used'] = len(list(run.JOURNAL.glob('*.json'))) + len(list((EXP / 'setup_attempt_v1/api_journal').glob('*.json')))
            self.budget['limit'] = run.settings()['max_new_api_attempts']
            try:
                return super().call(payload)
            except RuntimeError as error:
                if str(error) == '本轮API次数上限':
                    self.resource_exhausted = True
                raise


run.FullLiveAgent = FullAdaptiveAgent


def __getattr__(name):
    return getattr(run, name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['health', 'run'])
    arguments = parser.parse_args()
    if arguments.phase == 'health':
        run.health()
    else:
        run.run_all()
