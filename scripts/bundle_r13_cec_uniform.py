"""Publish complete measurements/API decisions; retain unused pool diagnostics locally."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import gzip
import hashlib
import json
import shutil
import tarfile

ROOT=Path(__file__).resolve().parents[1]
EXP=ROOT/'experiments/r13_cec_uniform'
PARENT=ROOT/'experiments/r10_current_candidate_fusion/results'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    result=json.loads((EXP/'INDEPENDENT_VERIFICATION.json').read_text())
    assert result['status']=='complete_live_cec_uniform_measurement_verified'
    publication=EXP/'publication'
    # A full diagnostic archive, if prepared, is kept independently and never discarded.
    if publication.exists():
        assert not (EXP/'raw_full_archives').exists()
        publication.rename(EXP/'raw_full_archives')
    publication.mkdir()
    key=(Path.home()/'.config/roopf_agent/worldcodes.env').read_text().strip().split('=',1)[1].encode()
    journals=sorted((EXP/'api_journal').glob('*.json'));assert len(journals)==360
    for p in journals:assert key not in p.read_bytes(),'Credential in API evidence'
    with tarfile.open(publication/'ACTUAL_API_JOURNAL.tar.xz','w:xz',preset=3) as tar:
        for p in journals:tar.add(p,arcname='api_journal/'+p.name)

    def pack(folder):
        raw={}
        target=publication/f'EVIDENCE_{folder.name}.tar.xz'
        with tarfile.open(target,'w:xz',preset=3) as tar:
            paths=[folder/'COMPLETE.json',*sorted(folder.glob('*_FULL100.npz')),
                   *sorted(folder.glob('b??/*.npz')),*sorted(folder.glob('b??/*.json')),
                   *sorted(folder.glob('b??/*_NATIVE_CALLS.pkl.gz')),
                   *sorted(folder.glob('b??/OBJECTIVE_RNG_SCHEDULE.pkl.gz'))]
            for p in paths:
                raw[str(p.relative_to(EXP))]=sha(p);assert key not in p.read_bytes(),'Credential in measurement evidence'
                tar.add(p,arcname=str(p.relative_to(EXP)))
            for p in sorted((PARENT/folder.name).glob('b??/PAIRED_START.pkl.gz')):
                tar.add(p,arcname='frozen_r10/'+str(p.relative_to(PARENT)))
        count=0
        with gzip.open(publication/f'EXECUTED_DECISIONS_{folder.name}.jsonl.gz','wt') as output:
            for p in sorted(folder.glob('b??/*_DECISIONS.json.gz')):
                raw[str(p.relative_to(EXP))]=sha(p)
                with gzip.open(p,'rt') as f:records=json.load(f)
                for r in records:
                    if r['request_event']:saved=r
                    else:
                        saved={k:r[k] for k in ['step','configs','selected','original','applied_generation','planned_generation','execution_context','policy_id','temporal_semantics']}
                        saved['executed_competing_points']=[c['candidates'][c['selected_index']] for c in r['current']]
                        saved['protected_reference_points']=[c['protected_point'] for c in r['current']]
                        saved['candidate_counts']=[c['candidate_count'] for c in r['current']]
                        saved['raw_record_sha256']=hashlib.sha256(json.dumps(r,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
                    output.write(json.dumps(dict(function=folder.name,batch=p.parent.name,arm=p.name.split('_DECISIONS')[0],decision=saved),separators=(',',':'),allow_nan=False)+'\n');count+=1
        assert count==1350
        return dict(function=folder.name,decision_records=count,raw_files_sha256=raw)
    folders=sorted((EXP/'results').iterdir());assert len(folders)==6
    with ThreadPoolExecutor(max_workers=6) as pool:records=list(pool.map(pack,folders))
    files={p.name:dict(sha256=sha(p),bytes=p.stat().st_size) for p in publication.iterdir() if p.is_file()}
    assert all(v['bytes']<95*1024**2 for v in files.values())
    manifest=dict(status='complete_cec_uniform_scientific_evidence',files=files,functions=records,
        actual_api_journals=360,all_request_rounds_retained=True,all_executed_points_and_objective_values_retained=True,
        full_unused_candidate_pool_predictions_retained_locally=True,raw_full_archive_manifest='raw_full_archives/MANIFEST.json',
        script_sha256=sha(Path(__file__)))
    (publication/'MANIFEST.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'publication_files':len(files),'bytes':sum(v['bytes'] for v in files.values()),'active_batch_records':sum(v['decision_records'] for v in records),'credential_scan_passed':True}))


if __name__=='__main__':main()
