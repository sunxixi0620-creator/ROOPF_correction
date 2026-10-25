"""Post-run publication of R11 evidence; never imported by the frozen solver."""
from concurrent.futures import ThreadPoolExecutor
import csv
import gzip
import hashlib
import json
from pathlib import Path
import tarfile
import io
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
EXP=ROOT/'experiments/r11_anchor_ablation'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    verification=json.loads((EXP/'INDEPENDENT_VERIFICATION.json').read_text())
    assert verification['status']=='full_live_no_anchor_records_verified'
    publication=EXP/'publication'
    publication.mkdir(exist_ok=False)
    key=(Path.home()/'.config/roopf_agent/worldcodes.env').read_text().strip().split('=',1)[1].encode()
    journals=sorted((EXP/'api_journal').glob('*.json'))
    assert len(journals)==580
    for path in journals:assert key not in path.read_bytes()
    with tarfile.open(publication/'API_JOURNAL.tar.xz','w:xz',preset=3) as archive:
        for path in journals:archive.add(path,arcname=str(path.relative_to(EXP)))
    folders=sorted((EXP/'results').iterdir())
    assert len(folders)==35
    with tarfile.open(publication/'EVALUATION_AND_EXECUTED_POINTS.tar.xz','w:xz',preset=3) as archive:
        for folder in folders:
            for path in [folder/'COMPLETE.json',folder/'llm_no_anchor_FULL100.npz',
                         *sorted(folder.glob('b??/llm_no_anchor.json')),*sorted(folder.glob('b??/llm_no_anchor.npz'))]:
                archive.add(path,arcname=str(path.relative_to(EXP)))
            parent=ROOT/'experiments/r10_current_candidate_fusion/results'/folder.name
            for path in [*sorted(parent.glob('*_FULL100.npz')),*sorted(parent.glob('b??/PAIRED_START.pkl.gz'))]:
                archive.add(path,arcname='frozen_r10/'+str(path.relative_to(parent.parent)))
            if not folder.name.startswith('CEC-style'):
                paired_points=[]
                for path in sorted(parent.glob('b??/llm_live_DECISIONS.json.gz')):
                    with gzip.open(path,'rt') as stream: records=json.load(stream)
                    assert len(records)==45
                    paired_points.append(np.asarray([[[c['protected_point'],c['candidates'][c['selected_index']]]
                        for c in record['current']] for record in records],dtype=np.float32).transpose(1,0,2,3))
                data=io.BytesIO()
                np.savez_compressed(data,points=np.concatenate(paired_points),iteration_indices=np.arange(55,100))
                content=data.getvalue();info=tarfile.TarInfo('frozen_r10/'+folder.name+'/Full_active_executed_points.npz')
                info.size=len(content);archive.addfile(info,io.BytesIO(content))

    def pack_function(folder):
        raw={}
        native=publication/f'NATIVE_OBJECTIVES_{folder.name}.tar.xz'
        with tarfile.open(native,'w:xz',preset=3) as archive:
            for path in sorted(folder.glob('b??/*_NATIVE_CALLS.pkl.gz')):
                raw[str(path.relative_to(EXP))]=sha(path)
                archive.add(path,arcname=str(path.relative_to(EXP)))
        records=0
        decision_path=publication/f'EXECUTED_DECISIONS_{folder.name}.jsonl.gz'
        with gzip.open(decision_path,'wt') as output:
            for path in sorted(folder.glob('b??/*_DECISIONS.json.gz')):
                raw[str(path.relative_to(EXP))]=sha(path)
                with gzip.open(path,'rt') as stream:decisions=json.load(stream)
                for record in decisions:
                    if record['request_event']:
                        saved=record
                    else:
                        saved={k:record[k] for k in ('step','configs','selected','original','applied_generation',
                            'planned_generation','execution_context','policy_id','temporal_semantics')}
                        saved['executed_competing_points']=[r['candidates'][r['selected_index']] for r in record['current']]
                        saved['native_reference_points']=[r['protected_point'] for r in record['current']]
                        saved['raw_record_sha256']=hashlib.sha256(json.dumps(record,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
                    output.write(json.dumps(dict(function=folder.name,batch=path.parent.name,decision=saved),separators=(',',':'),allow_nan=False)+'\n')
                    records+=1
        return dict(function=folder.name,decision_records=records,raw_files_sha256=raw)

    with ThreadPoolExecutor(max_workers=4) as pool: manifests=list(pool.map(pack_function,folders))
    files={path.name:sha(path) for path in publication.iterdir() if path.is_file()}
    assert all(path.stat().st_size<95*1024**2 for path in publication.iterdir() if path.is_file())
    manifest=dict(status='complete_live_no_anchor_evidence_bundle',sha256=files,functions=manifests,
        raw_unused_candidate_predictions_retained_locally=True,source_script_sha256=sha(Path(__file__)))
    (publication/'MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
    previous=list(csv.DictReader((ROOT/'analyses/r10_requested_ablations_20261025/SUMMARY.csv').open()))
    added=[dict(comparison='Full_vs_llm_no_anchor',**row) for row in csv.DictReader((EXP/'SUMMARY.csv').open())]
    with (EXP/'ALL_ABLATION_SUMMARY.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(previous[0]),lineterminator='\n')
        writer.writeheader();writer.writerows(added+previous)
    print(json.dumps(dict(files=len(files),total_bytes=sum(p.stat().st_size for p in publication.iterdir()),api_journals=len(journals),functions=len(folders))))


if __name__=='__main__':main()
