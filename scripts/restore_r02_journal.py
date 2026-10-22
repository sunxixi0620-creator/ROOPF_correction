from pathlib import Path
import hashlib
import json
import tarfile


def restore(experiment):
    root = Path(experiment).resolve()
    archive_path = root / 'API_AND_DECISION_JOURNAL.tar.xz'
    manifest = json.loads((root / 'JOURNAL_MANIFEST.json').read_text())
    with tarfile.open(archive_path, 'r:xz') as archive:
        for entry in manifest['files']:
            destination = (root / entry['path']).resolve()
            assert destination.is_relative_to(root / 'results'), '归档成员超出结果目录'
            data = archive.extractfile(entry['path']).read()
            assert hashlib.sha256(data).hexdigest() == entry['sha256']
            if destination.exists():
                assert destination.read_bytes() == data, '已有回执与归档不一致，禁止覆盖'
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(data)
    return len(manifest['files'])


if __name__ == '__main__':
    experiment = Path(__file__).resolve().parents[1] / 'experiments/r02_fusion_agent'
    print('Verified journal files:', restore(experiment))
