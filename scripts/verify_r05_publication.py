import gzip
import hashlib
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / 'experiments/r05_adaptive_evidence_fusion'


def main():
    line = (Path.home() / '.config/roopf_agent/worldcodes.env').read_text().strip()
    if not line.startswith('WORLDCODES_API_KEY='):
        raise RuntimeError('Unexpected credential format; refusing publication without leak check')
    secret = line.split('=', 1)[1].strip().encode()
    if len(secret) < 12:
        raise RuntimeError('Unexpected credential length')
    checked = 0
    def inspect(content, name):
        nonlocal checked
        if secret in content:
            raise RuntimeError('Credential detected in publication artifact: ' + name)
        if name.endswith('.gz'):
            if secret in gzip.decompress(content):
                raise RuntimeError('Credential detected inside compressed artifact: ' + name)
        checked += 1
    for path in EXP.iterdir():
        if not path.is_file():
            continue
        if path.name.endswith('.tar.xz'):
            with tarfile.open(path, 'r:xz') as archive:
                for member in archive:
                    if not member.isfile() or member.name.endswith('.env'):
                        raise RuntimeError('Unexpected publication member')
                    inspect(archive.extractfile(member).read(), member.name)
        else:
            inspect(path.read_bytes(), path.name)
    for name in ('roopf/model.py', 'roopf/adaptive_fusion_agent.py', 'scripts/experiment_r05_full.py',
                 'scripts/finalize_r05_full.py', 'scripts/verify_r05.py',
                 'scripts/analyze_r05_full.py', 'scripts/verify_r05_publication.py'):
        inspect((ROOT / name).read_bytes(), name)
    result = dict(status='passed', checked_artifacts=checked, credential_found=False,
                  credential_value_never_logged=True)
    (EXP / 'PUBLICATION_SAFETY.json').write_text(json.dumps(result, indent=2) + '\n')
    manifest = [dict(path=path.name, bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        for path in sorted(EXP.iterdir()) if path.is_file() and path.name not in {'MANIFEST.json', 'COMPLETE.json'}]
    (EXP / 'MANIFEST.json').write_text(json.dumps(manifest, indent=2) + '\n')
    complete = json.loads((EXP / 'COMPLETE.json').read_text())
    complete.update(public_artifacts=len(manifest), publication_safety_passed=True,
        manifest_sha256=hashlib.sha256((EXP / 'MANIFEST.json').read_bytes()).hexdigest())
    (EXP / 'COMPLETE.json').write_text(json.dumps(complete, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
