"""Post-run records for the actual whole-module ablations, not input ablations."""
from concurrent.futures import ThreadPoolExecutor
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
import pickle
import tarfile
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
EXP=ROOT/'experiments/r12_strict_component_ablations'
PARENT=ROOT/'experiments/r10_current_candidate_fusion/results'
ARMS=['no_agent','llm_no_online_generators']


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def csv_rows(path):
    with path.open() as stream:return list(csv.DictReader(stream))


def write_csv(path,rows):
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)


def main():
    result=json.loads((EXP/'INDEPENDENT_VERIFICATION.json').read_text())
    assert result['status']=='full_strict_component_ablations_verified'
    publication=EXP/'publication';publication.mkdir(exist_ok=False)
    key=(Path.home()/'.config/roopf_agent/worldcodes.env').read_text().strip().split('=',1)[1].encode()
    journals=sorted((EXP/'api_journal').glob('*.json'));assert len(journals)==580
    for path in journals:assert key not in path.read_bytes()
    with tarfile.open(publication/'API_JOURNAL.tar.xz','w:xz',preset=3) as archive:
        for path in journals:archive.add(path,arcname=str(path.relative_to(EXP)))

    def function(folder):
        raw={};rng_hashes={}
        artifact=publication/f'EVIDENCE_{folder.name}.tar.xz'
        with tarfile.open(artifact,'w:xz',preset=3) as archive:
            for path in [folder/'COMPLETE.json',*sorted(folder.glob('*_FULL100.npz')),
                *sorted(folder.glob('b??/*.npz')),*sorted(folder.glob('b??/*.json')),
                *sorted(folder.glob('b??/*_NATIVE_CALLS.pkl.gz'))]:
                raw[str(path.relative_to(EXP))]=sha(path)
                archive.add(path,arcname=str(path.relative_to(EXP)))
            for path in sorted(folder.glob('b??/OBJECTIVE_RNG_SCHEDULE.pkl.gz')):
                raw[str(path.relative_to(EXP))]=sha(path)
                with gzip.open(path,'rb') as stream:states=pickle.load(stream)
                assert len(states)==201
                rng_hashes[path.parent.name]=[hashlib.sha256(state.numpy().tobytes()).hexdigest() for state in states]
            content=json.dumps(dict(raw_schedule_sha256={k:v for k,v in raw.items() if k.endswith('OBJECTIVE_RNG_SCHEDULE.pkl.gz')},objective_rng_hashes=rng_hashes)).encode()
            entry=tarfile.TarInfo('results/'+folder.name+'/OBJECTIVE_RNG_HASHES.json');entry.size=len(content);archive.addfile(entry,io.BytesIO(content))
            reference={arm:np.load(PARENT/folder.name/(arm+'_FULL100.npz'))['values'].min(1) for arm in ['llm_live','off','matched38']}
            memory=io.BytesIO();np.savez_compressed(memory,**reference);content=memory.getvalue()
            entry=tarfile.TarInfo('frozen_r10/'+folder.name+'/FINAL_VALUES.npz');entry.size=len(content);archive.addfile(entry,io.BytesIO(content))
        decisions=publication/f'EXECUTED_DECISIONS_{folder.name}.jsonl.gz';count=0
        with gzip.open(decisions,'wt') as output:
            for path in sorted(folder.glob('b??/*_DECISIONS.json.gz')):
                raw[str(path.relative_to(EXP))]=sha(path)
                with gzip.open(path,'rt') as stream:records=json.load(stream)
                if path.name.startswith('no_agent'):assert not records
                for record in records:
                    if record['request_event']:saved=record
                    else:
                        saved={k:record[k] for k in ['step','configs','selected','original','applied_generation','planned_generation','execution_context','policy_id','temporal_semantics']}
                        saved['executed_competing_points']=[r['candidates'][r['selected_index']] for r in record['current']]
                        saved['protected_anchor_points']=[r['protected_point'] for r in record['current']]
                        saved['candidate_counts']=[r['candidate_count'] for r in record['current']]
                        saved['raw_record_sha256']=hashlib.sha256(json.dumps(record,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
                    output.write(json.dumps(dict(function=folder.name,batch=path.parent.name,decision=saved),separators=(',',':'),allow_nan=False)+'\n');count+=1
        return dict(function=folder.name,decision_records=count,raw_files_sha256=raw)

    folders=sorted((EXP/'results').iterdir());assert len(folders)==35
    with ThreadPoolExecutor(max_workers=4) as pool:records=list(pool.map(function,folders))
    files={path.name:sha(path) for path in publication.iterdir() if path.is_file()}
    assert all(path.stat().st_size<95*1024**2 for path in publication.iterdir() if path.is_file())
    (publication/'MANIFEST.json').write_text(json.dumps(dict(status='strict_whole_module_evidence_complete',sha256=files,functions=records,
        raw_rng_states_and_unused_candidate_predictions_retained_locally=True,script_sha256=sha(Path(__file__))),indent=2)+'\n')
    anchor=[dict(arm='llm_no_anchor',**row) for row in csv_rows(ROOT/'experiments/r11_anchor_ablation/SUMMARY.csv')]
    write_csv(EXP/'WHOLE_MODULE_ABLATION_SUMMARY.csv',anchor+csv_rows(EXP/'SUMMARY.csv'))
    anchor_functions=[dict(group=row['group'],fid=row['fid'],arm='llm_no_anchor',primary=row['primary'],full_mean=row['full_mean'],full_std=row['full_std'],alternative_mean=row['no_anchor_mean'],alternative_std=row['no_anchor_std'],wins=row['wins'],ties=row['ties'],losses=row['losses']) for row in csv_rows(ROOT/'experiments/r11_anchor_ablation/FUNCTION_COMPARISON.csv')]
    keys=list(anchor_functions[0]);current=[{k:row[k] for k in keys} for row in csv_rows(EXP/'FUNCTION_COMPARISON.csv')]
    write_csv(EXP/'WHOLE_MODULE_FUNCTION_COMPARISONS.csv',anchor_functions+current)
    print(json.dumps(dict(files=len(files),total_bytes=sum(path.stat().st_size for path in publication.iterdir()),functions=35,arms=2,api_journals=len(journals))))


if __name__=='__main__':main()
