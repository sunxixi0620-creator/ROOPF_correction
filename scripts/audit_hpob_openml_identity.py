"""Fetch public dataset metadata only; no raw data or optimization labels."""
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import sys
import requests
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from roopf.experiment_io import write_json,sha256


def main():
    manifest=json.loads((ROOT/'docs/research/hpob_preparation/DEVELOPMENT_MANIFEST.json').read_text())
    source=set().union(*(set(t['official_sources']) for t in manifest['tasks']))
    heldout={t['dataset'] for t in manifest['tasks']}|set().union(*map(set,manifest['inferred_test_ids'].values()))
    root=ROOT/'results/hpob_source_preparation/openml_tasks'
    root.mkdir(exist_ok=True)
    def fetch(k):
        path=root/f'{k}.json'
        if not path.exists():
            r=requests.get(f'https://www.openml.org/api/v1/json/task/{k}',timeout=(10,40))
            r.raise_for_status()
            value=r.json()
            write_json(path,value)
        d=json.loads(path.read_text())['task']
        assert d['task_id']==k
        data=next(v['data_set'] for v in d['input'] if v['name']=='source_data')
        return dict(id=k,name=d['task_name'],raw_dataset_id=data['data_set_id'],
                    sha256=sha256(path),source=k in source,heldout=k in heldout)
    rows=[];errors={}
    with ThreadPoolExecutor(max_workers=6) as ex:
        fs={ex.submit(fetch,k):k for k in sorted(source|heldout,key=int)}
        for f in as_completed(fs):
            try:rows.append(f.result())
            except Exception as e:errors[fs[f]]=repr(e)
    aliases=[]
    for key in ('raw_dataset_id',):
        groups={}
        for r in rows:
            value=r[key]
            if value:groups.setdefault(value.lower(),[]).append(r)
        for value,group in groups.items():
            if len(group)>1 and any(r['source'] for r in group) and any(r['heldout'] for r in group):
                aliases.append(dict(key=key,value=value,datasets=group))
    heldout_raw={r['raw_dataset_id'] for r in rows if r['heldout']}
    mapping={r['id']:r['raw_dataset_id'] for r in rows}
    tasks=[]
    for task in manifest['tasks']:
        copy=dict(task)
        copy['raw_dataset_id']=mapping.get(task['dataset'])
        copy['raw_disjoint_sources']=[k for k in task['official_sources'] if k in mapping and mapping[k] not in heldout_raw]
        copy['eligible_raw_disjoint']=bool(task['eligible_strict'] and len(copy['raw_disjoint_sources'])>=5 and not errors)
        tasks.append(copy)
    write_json(ROOT/'docs/research/hpob_preparation/RAW_DATASET_MANIFEST.json',
        dict(tasks=tasks,eligible=sum(t['eligible_raw_disjoint'] for t in tasks),
             unique_raw_dataset_ids=len(set(mapping.values())), unresolved_ids=errors,
             parent_sha256=sha256(ROOT/'docs/research/hpob_preparation/DEVELOPMENT_MANIFEST.json')))
    write_json(ROOT/'docs/research/hpob_preparation/OPENML_IDENTITY_AUDIT.json',
        dict(metadata=sorted(rows,key=lambda r:int(r['id'])),errors=errors,exact_aliases=aliases,
             target_response_queries=0,raw_datasets_downloaded=False,
             limitation='HPO-B keys resolve as OpenML task IDs. Raw dataset-ID grouping does not rule out derived datasets with different IDs.'))
    print(json.dumps(dict(metadata=len(rows),errors=errors,aliases=aliases),indent=2))


if __name__=='__main__':main()
