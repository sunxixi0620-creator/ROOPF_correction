"""Archive completed transfer evidence and frozen source preparation only."""
import hashlib,json,sys,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import hpob_transfer_development_v1 as parent
from scripts import prepare_neural_prior_source_v1 as source
from scripts import train_neural_prior_v1 as training
from roopf.experiment_io import sha256,write_json


def main():
    parent.verify();source.verify();training.verify()
    assert json.loads((parent.OUT/'VERIFICATION.json').read_text())['passed']
    dest=ROOT/'artifacts/hpob_transfer_and_prior_v1';dest.mkdir(parents=True,exist_ok=True)
    files=[]
    # Complete paid journals/cases suffice for the original full-pool replay.
    # Omit redundant recovery files and oracle tables containing unpaid labels.
    for directory in ('caches','cases','journals','models','public','source_data'):
        files.extend(p for p in (parent.RUN/directory).rglob('*') if p.is_file() and p.suffix!='.lock')
    files.extend(p for p in parent.RUN.iterdir() if p.is_file() and p.suffix in ('.json','.log'))
    for root in (parent.OUT,source.RUN,source.OUT):
        files.extend(p for p in root.rglob('*') if p.is_file() and p.suffix!='.lock')
    files.extend(ROOT/p for p in (*parent.SOURCES,*source.SOURCES,*training.SOURCES))
    files.extend(training.OUT/p for p in ('CONTRACTS.json','IDENTITY.json'))
    files.extend([training.RUN/'identity.json',Path(__file__),
        ROOT/'scripts/report_neural_prior_training_v1.py',ROOT/'scripts/run_neural_prior_pipeline_v1.py',
        ROOT/'scripts/report_hpob_transfer_development_v1.py',ROOT/'scripts/check_hpob_transfer_cache_v1.py',
        ROOT/'scripts/check_neural_prior_pca_v1.py'])
    files=sorted(set(files));manifest={str(p.relative_to(ROOT)):sha256(p) for p in files}
    batches=[];current=[];size=0
    for p in files:
        if current and size+p.stat().st_size>50_000_000:batches.append(current);current=[];size=0
        current.append(p);size+=p.stat().st_size
    if current:batches.append(current)
    archives={}
    for index,batch in enumerate(batches,1):
        path=dest/f'evidence.part{index:03d}.zip'
        with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
            for p in batch:archive.write(p,str(p.relative_to(ROOT)))
        assert path.stat().st_size<100_000_000
        with zipfile.ZipFile(path) as archive:
            assert archive.testzip() is None
            for name in archive.namelist():assert hashlib.sha256(archive.read(name)).hexdigest()==manifest[name]
        archives[path.name]=sha256(path);print('verified archive',index,'/',len(batches),path.stat().st_size,flush=True)
    write_json(dest/'MANIFEST.json',dict(files=manifest,archives=archives,
        transfer_trajectories=1320,target_calls_recorded=138600,additional_target_calls=0,
        includes_neural_fitted_models=False,confirmation_labels_included=False,
        restore='Extract all ZIPs at repository root; upstream data/identities retain their existing dependencies.',
        omitted='Redundant progress snapshots, unpaid development oracle tables, in-progress neural fits',
        recheck_commands=['.venv/bin/python scripts/report_hpob_transfer_development_v1.py',
                          '.venv/bin/python scripts/check_neural_prior_pca_v1.py']))
    print('archived',len(files),'files',len(archives),'parts',flush=True)


if __name__=='__main__':main()
