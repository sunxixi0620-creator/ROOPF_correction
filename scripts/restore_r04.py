import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / 'experiments/r04_multiseed_horizon'


def restore(write=False):
    manifest = json.loads((EXP / 'ARCHIVE_MANIFEST.json').read_text())
    expected = {(row['archive'], row['path']): row for row in manifest['members']}
    seen, written, existing = set(), 0, 0
    for shard in manifest['archives']:
        archive_path = EXP / shard['path']
        raw = archive_path.read_bytes()
        if len(raw) != shard['bytes'] or hashlib.sha256(raw).hexdigest() != shard['sha256']:
            raise RuntimeError('Archive checksum mismatch')
        with tarfile.open(archive_path, 'r:xz') as archive:
            for member in archive:
                key = shard['path'], member.name
                entry = expected.get(key)
                relative = PurePosixPath(member.name)
                if entry is None or key in seen or not member.isfile() or relative.is_absolute() or '..' in relative.parts:
                    raise RuntimeError('Unexpected, duplicate, or unsafe archive member')
                if relative.parts[0] not in {'collection', 'confirmatory', 'verification'}:
                    raise RuntimeError('Archive member outside raw-evidence trees')
                destination = EXP.joinpath(*relative.parts)
                if not destination.resolve().is_relative_to(EXP.resolve()):
                    raise RuntimeError('Archive path escapes experiment directory')
                data = archive.extractfile(member).read()
                if len(data) != entry['bytes'] or hashlib.sha256(data).hexdigest() != entry['sha256']:
                    raise RuntimeError('Archive member checksum mismatch')
                if destination.exists():
                    if destination.read_bytes() != data:
                        raise RuntimeError('Existing local evidence differs; refusing to overwrite')
                    existing += 1
                elif write:
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    with destination.open('xb') as stream:
                        stream.write(data)
                    written += 1
                seen.add(key)
    if set(expected) != seen:
        raise RuntimeError('Archive manifest members missing')
    result = dict(status='verified', members=len(seen), already_present=existing,
        restored=written, read_only=not write, existing_files_never_overwritten=True)
    print(json.dumps(result))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore', action='store_true')
    restore(parser.parse_args().restore)
