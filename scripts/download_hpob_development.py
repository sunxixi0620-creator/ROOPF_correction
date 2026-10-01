"""Reproduce the pinned, allowlisted source/development downloads only."""
import requests,json,hashlib,time,concurrent.futures
from pathlib import Path
root=Path(__file__).resolve().parents[1]/'results/hpob_source_preparation'
rev=json.loads((root/'hf_metadata.json').read_text())['sha']; url=f'https://huggingface.co/datasets/Sebastianpinar/hpob/resolve/{rev}/'
rows=requests.get(f'https://huggingface.co/api/datasets/Sebastianpinar/hpob/tree/{rev}',timeout=30).json()
(root/'hf_tree.json').write_text(json.dumps(rows,indent=2))
allowed={'meta-train-dataset.json','meta-validation-dataset.json','bo-initializations.json','README.md'}
def get(row):
 name=row['path']
 p=root/'data'/name;p.parent.mkdir(exist_ok=True)
 if not p.exists():
  r=requests.get(url+name,stream=True,timeout=(20,120));r.raise_for_status()
  with p.with_suffix('.partial').open('wb') as f:
   for b in r.iter_content(1048576):f.write(b)
  p.with_suffix('.partial').replace(p)
 h=hashlib.sha256(p.read_bytes()).hexdigest();assert p.stat().st_size==row['size']
 if 'lfs' in row:assert h==row['lfs']['oid']
 print(name,p.stat().st_size,h,flush=True)
 return {'name':name,'bytes':row['size'],'sha256':h,'url':url+name}
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:r=list(ex.map(get,[r for r in rows if r['path'] in allowed]))
(root/'DOWNLOAD.json').write_text(json.dumps({'revision':rev,'files':r,'test_response_downloaded':False},indent=2))
