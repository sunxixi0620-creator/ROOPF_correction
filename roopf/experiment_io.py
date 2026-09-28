"""Content-addressed experiment identity and atomic, verified case artifacts."""
from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import torch


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def canonical(value):
    if torch.is_tensor(value):
        value = value.detach().cpu().contiguous().numpy()
    if isinstance(value, np.ndarray):
        return {'dtype': str(value.dtype), 'shape': list(value.shape),
                'sha256': hashlib.sha256(value.tobytes()).hexdigest()}
    if isinstance(value, dict):
        return {str(k): canonical(v) for k, v in sorted(value.items(), key=lambda p: str(p[0]))}
    if isinstance(value, (list, tuple)):
        return [canonical(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def fingerprint(value):
    return hashlib.sha256(json.dumps(canonical(value), sort_keys=True,
                         allow_nan=False, separators=(',', ':')).encode()).hexdigest()


def seed_for(*identity):
    # Random streams have explicit roles; no additive offsets or shared global RNG.
    return int(fingerprint(identity)[:15], 16)


@contextlib.contextmanager
def atomic_path(destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.' + destination.name + '.', dir=destination.parent)
    os.close(fd)
    temporary = Path(name)
    try:
        yield temporary
        with temporary.open('rb') as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def write_json(path, value):
    with atomic_path(path) as temporary:
        temporary.write_text(json.dumps(canonical(value), indent=2, allow_nan=False) + '\n')


def save_torch(path, value):
    with atomic_path(path) as temporary:
        torch.save(value, temporary)


class CaseStore:
    """A verified manifest is the commit marker, never mere file existence."""
    def __init__(self, root, run_identity):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.identity = canonical(run_identity)
        path = self.root / 'identity.json'
        with self.lock('_identity'):
            if path.exists():
                if json.loads(path.read_text()) != self.identity:
                    raise ValueError('Run identity mismatch; use a new run directory')
            else:
                write_json(path, self.identity)

    @contextlib.contextmanager
    def lock(self, case_id):
        if not case_id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in case_id):
            raise ValueError('Unsafe case id')
        with (self.root / (case_id + '.lock')).open('a') as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            yield

    def load(self, case_id, case_identity):
        manifest = self.root / (case_id + '.json')
        data = self.root / (case_id + '.pt')
        if not manifest.exists():
            return None  # An uncommitted data file is never reused.
        meta = json.loads(manifest.read_text())
        expected = fingerprint({'run': self.identity, 'case': case_identity})
        if meta['identity_sha256'] != expected:
            raise ValueError(f'{case_id}: configuration/seed/source mismatch')
        if not data.exists() or sha256(data) != meta['data_sha256']:
            raise ValueError(f'{case_id}: corrupt or incomplete artifact')
        return torch.load(data, map_location='cpu', weights_only=False)

    def save(self, case_id, case_identity, value):
        # Caller holds lock across load, computation, and commit.
        data = self.root / (case_id + '.pt')
        save_torch(data, value)
        write_json(self.root / (case_id + '.json'), {
            'identity_sha256': fingerprint({'run': self.identity, 'case': case_identity}),
            'case': canonical(case_identity), 'data_sha256': sha256(data)})
