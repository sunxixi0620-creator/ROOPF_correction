"""Archive all completed v1 source fits and their preserved numerical audit."""
import hashlib,json,sys,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import train_neural_prior_v1 as study
from roopf.experiment_io import sha256,write_json


def main():
    study.verify();audit=ROOT/'docs/revision/neural_prior_training_v1_audit'
    assert json.loads((study.RUN/'TRAIN_COMPLETE.json').read_text())['jobs']==135
    result=json.loads((audit/'RESULTS.json').read_text())
    assert result['numerical']['same_device_cuda_replay_passed']
    files=sorted({p for root in (study.RUN,study.OUT,audit) for p in root.rglob('*')
                  if p.is_file() and p.suffix!='.lock'}|{ROOT/p for p in study.SOURCES}|
                 {Path(__file__),ROOT/'scripts/audit_neural_prior_training_v1.py',ROOT/'scripts/report_neural_prior_training_v1.py'})
    manifest={str(p.relative_to(ROOT)):sha256(p) for p in files}
    dest=ROOT/'artifacts/neural_prior_training_v1';dest.mkdir(parents=True,exist_ok=True)
    batches=[];current=[];size=0
    for p in files:
        if current and size+p.stat().st_size>50_000_000:batches.append(current);current=[];size=0
        current.append(p);size+=p.stat().st_size
    if current:batches.append(current)
    archives={}
    for index,batch in enumerate(batches,1):
        path=dest/f'source_training.part{index:03d}.zip'
        with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
            for p in batch:archive.write(p,str(p.relative_to(ROOT)))
        assert path.stat().st_size<100_000_000
        with zipfile.ZipFile(path) as archive:
            assert archive.testzip() is None
            for name in archive.namelist():assert hashlib.sha256(archive.read(name)).hexdigest()==manifest[name]
        archives[path.name]=sha256(path);print('verified archive',index,'/',len(batches),path.stat().st_size,flush=True)
    write_json(dest/'MANIFEST.json',dict(files=manifest,archives=archives,
        parent='artifacts/hpob_transfer_and_prior_v1',fits=135,source_updates=result['completed_updates'],
        numerical_status=result['numerical'],target_calls=0,confirmation_calls=0,
        restore='Restore parent dependencies first, then extract all ZIPs at repository root.',
        note='Original failed CPU absolute check is preserved. CUDA replay is a separate exact reproduction, not a relaxation of that check.'))


if __name__=='__main__':main()
