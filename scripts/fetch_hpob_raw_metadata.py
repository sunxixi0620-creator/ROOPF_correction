"""Fetch raw dataset descriptions after resolving the OpenML task namespace."""
import json,requests,concurrent.futures
from pathlib import Path
root=Path(__file__).resolve().parents[1];m=json.load(open(root/'docs/research/hpob_preparation/OPENML_IDENTITY_AUDIT.json'))
ids=sorted({r['raw_dataset_id'] for r in m['metadata']},key=int);dest=root/'results/hpob_source_preparation/raw_dataset_metadata';dest.mkdir(exist_ok=True)
def get(k):
 p=dest/f'{k}.json'
 if not p.exists():
  r=requests.get(f'https://www.openml.org/api/v1/json/data/{k}',timeout=(10,40));r.raise_for_status();p.write_text(json.dumps(r.json(),indent=2))
 d=json.loads(p.read_text())['data_set_description'];return {'id':k,'name':d['name'],'version':d.get('version'),'md5':d.get('md5_checksum')}
with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:rows=list(ex.map(get,ids))
(root/'docs/research/hpob_preparation/RAW_DATASET_METADATA.json').write_text(json.dumps(rows,indent=2));print(json.dumps(rows,indent=2))
