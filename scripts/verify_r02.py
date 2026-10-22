import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '1'

import csv
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from unittest import mock
import zlib
import pickle

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from roopf.fusion_agent import CONFIGS, FusionAgent, fusion_indices

EXP = ROOT / 'experiments/r02_fusion_agent'


def main():
    snapshot = EXP / 'baseline_52ec388'
    identities = json.loads((snapshot / 'SNAPSHOT.json').read_text())
    for entry in identities['files']:
        assert hashlib.sha256((snapshot / entry['path']).read_bytes()).hexdigest() == entry['sha256']
    binding = json.loads((EXP / 'CODE_BINDING.json').read_text())
    for name, sha in binding.items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == sha
    memory = json.loads((EXP / 'offline_memory.json').read_text())
    assert memory['test_data_used'] is False
    assert len({case['function'] for case in memory['cases']}) == 29
    assert not ({case['function'] for case in memory['cases']} & set(memory['heldout_functions']))
    assert all(memory['validation_mean_one_step_advantage'][name] >= 0 for name in memory['allowed_configs'])

    score = torch.tensor([[0.2, 0.1, 0.05]])
    probability = torch.tensor([[0.1, 0.3, 0.8]])
    scale = torch.ones((1, 1))
    original = torch.zeros((1, 1), dtype=torch.long)
    options = fusion_indices(score, probability, scale, original, .008)
    shifted = fusion_indices(score * 7 + 3, probability, scale * 7, original, .008)
    assert torch.equal(options['release'], original)
    assert all(torch.equal(options[name], shifted[name]) for name in CONFIGS)
    assert all(0 <= int(value) < 3 for value in options.values())
    assert int(options['balanced']) == 2

    with tempfile.TemporaryDirectory(prefix='r02_invalid_output_contract_') as directory:
        agent = FusionAgent(EXP / 'offline_memory.json', directory, mode='llm', budget=dict(used=0, limit=1))
        agent.reset(1)
        def fake_transport(*args, **kwargs):
            request = next(Path(directory).glob('*.request.payload'))
            body = dict(model='synthetic_contract_only', choices=[dict(message=dict(content='{"config_ids":["invented"]}'))])
            Path(str(request) + '.response.raw').write_text(json.dumps(body))
            return SimpleNamespace(returncode=0, stdout='', stderr='')
        with mock.patch('roopf.fusion_agent.subprocess.check_output', side_effect=lambda args, **kwargs: args[-1]), mock.patch('roopf.fusion_agent.subprocess.run', side_effect=fake_transport):
            selected = agent.choose(original_idx=original, score=score, mu=score, sigma=score.abs(), probability=probability,
                state=torch.zeros((1, 11)), archive_y=torch.tensor([[0., 1., 2.]]), fitness=torch.tensor([[0., 1., 2.]]),
                candidates=torch.zeros((1, 3, 10)), candidate_ops=torch.tensor([[-2, 0, 1]]), remaining=.3, problem=None, residual_weight=.008)
        assert torch.equal(selected, original) and agent.stats['failures'] == 1

    verified = []
    native = 0
    for group, count in [('BBOB', 24), ('CEC-style', 6), ('UAV', 5)]:
        for fid in range(1, count + 1):
            name = f'{group}_f{fid:02d}'
            folder = EXP / 'results' / name
            data = pickle.loads(zlib.decompress((snapshot / 'docs/revision/zip_replication820_results' / name / 'VALUES.bin').read_bytes()))
            shape, raw = data['full']
            reference = np.frombuffer(raw, dtype=np.float32).reshape(shape)
            assert reference.shape == (100, 300)
            stored = np.load(folder / 'archived_baseline.npz')['values']
            assert np.array_equal(stored, reference[:10])
            off = np.load(folder / 'off.npz')['values']
            assert np.array_equal(off, stored)
            paired_ids = set()
            for mode in ['off', 'retrieval', 'llm_live']:
                receipt_path = folder / f'{mode}.json'
                if not receipt_path.exists():
                    continue
                receipt = json.loads(receipt_path.read_text())
                values = np.load(folder / f'{mode}.npz')['values']
                assert values.shape == (10, 300) and np.isfinite(values).all()
                assert receipt['nfe'] == 300
                assert np.isclose(values.min(1).astype(float).mean(), receipt['mean'], rtol=1e-14, atol=1e-14)
                paired_ids.add(receipt['paired_initial_sha256'])
                native += receipt['native_fe']
                if mode == 'llm_live':
                    assert receipt['agent_stats']['failures'] < 2
                    assert receipt['native_fe'] + receipt['cached_fe'] == 3000
                if mode == 'off':
                    assert receipt['native_fe'] == 0
            assert len(paired_ids) == 1
            verified.append(name)
    record = dict(status='passed', snapshot_files=len(identities['files']), paired_functions=len(verified),
                  original_and_off_bitwise_equal=True, exact_nfe=300, candidate_scale_invariance=True,
                  invalid_api_output_falls_back=True, mock_contract_api_calls=0,
                  no_test_tasks_in_offline_memory=True, native_evaluations_completed=native,
                  source_commit=identities['source_commit'], code_binding=binding)
    (EXP / 'VERIFICATION.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record))


if __name__ == '__main__':
    main()
