import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / 'experiments/r04_full_llm_benchmark'


def restore(write=False):
    manifest = json.loads((EXP / 'ARCHIVE_MANIFEST.json').read_text())
    expected = {(row['archive'], row['path']): row for row in manifest['members']}
    seen, existing, restored = set(), 0, 0
    for shard in manifest['archives']:
        path = EXP / shard['path']
        if hashlib.sha256(path.read_bytes()).hexdigest() != shard['sha256']:
            raise RuntimeError('Archive checksum mismatch')
        with tarfile.open(path, 'r:xz') as archive:
            for member in archive:
                key = shard['path'], member.name
                relative = PurePosixPath(member.name)
                row = expected.get(key)
                if row is None or key in seen or not member.isfile() or relative.is_absolute() or '..' in relative.parts:
                    raise RuntimeError('Unexpected or unsafe raw archive member')
                if relative.parts[0] not in {'results', 'api_journal', 'setup_attempt_v1'}:
                    raise RuntimeError('Raw member outside expected evidence tree')
                destination = EXP.joinpath(*relative.parts)
                if not destination.resolve().is_relative_to(EXP.resolve()):
                    raise RuntimeError('Raw member escapes experiment')
                content = archive.extractfile(member).read()
                if len(content) != row['bytes'] or hashlib.sha256(content).hexdigest() != row['sha256']:
                    raise RuntimeError('Raw member checksum mismatch')
                if destination.exists():
                    if destination.read_bytes() != content:
                        raise RuntimeError('Existing evidence differs; refusing overwrite')
                    existing += 1
                elif write:
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    with destination.open('xb') as stream:
                        stream.write(content)
                    restored += 1
                seen.add(key)
    if set(expected) != seen:
        raise RuntimeError('Archive manifest incomplete')
    result = dict(status='verified', raw_members=len(seen), already_present=existing,
        restored=restored, read_only=not write, existing_files_never_overwritten=True)
    print(json.dumps(result))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore', action='store_true')
    restore(parser.parse_args().restore)
