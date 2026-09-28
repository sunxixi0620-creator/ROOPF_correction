"""Archive only completed stages, with <=40MB uncompressed per ZIP and SHA256."""
import argparse,hashlib,json,shutil,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
NAMES=['remaining_operators_20260928','remaining_replay_20260928','remaining_uav_20260928',
 'remaining_gp_ei_20260928','remaining_surr_rlde_20260928','random_shortlist_20260928',
 'retrain_d20_canonical_20260928','retrain_d10_original_20260928','retrain_d10_curated_20260928',
 'native_residual_logs_20260928','retrain_residual_d20_20260928','native20_eval_20260928',
 'order_original_eval_20260928','order_curated_eval_20260928','serial_external_timing_20260928']

def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--require-all',action='store_true');a=p.parse_args()
    out=ROOT/'artifacts/remaining_20260928';out.mkdir(exist_ok=True)
    manifest_path=out/'manifest.json';manifest=json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    for name in NAMES:
        src=ROOT/'results'/name
        if not (src/'COMPLETE').exists():
            assert not a.require_all,('Incomplete',name)
            continue
        if name in manifest:continue
        paths=sorted(p for p in src.rglob('*') if p.is_file());log=ROOT/'results'/(name+'.log')
        if log.exists():paths.append(log)
        groups=[];group=[];size=0
        for path in paths:
            n=path.stat().st_size
            if group and size+n>40_000_000:groups.append(group);group=[];size=0
            assert n<40_000_000,(path,n)
            group.append(path);size+=n
        if group:groups.append(group)
        records=[]
        for i,group in enumerate(groups,1):
            target=out/f'{name}.part{i:03d}.zip';files={}
            with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
                for path in group:
                    rel=str(path.relative_to(ROOT));z.write(path,rel)
                    files[rel]={'bytes':path.stat().st_size,'sha256':digest(path)}
            with zipfile.ZipFile(target) as z:assert z.testzip() is None
            assert target.stat().st_size<45_000_000
            records.append({'archive':target.name,'bytes':target.stat().st_size,'sha256':digest(target),'files':files})
        manifest[name]=records;manifest_path.write_text(json.dumps(manifest,indent=2))
        summary=out/'summaries'/name;summary.mkdir(parents=True,exist_ok=True)
        for filename in ['protocol.json','history.json','metrics.json','counts.csv','raw_results.csv','COMPLETE','REPORT.md']:
            path=src/filename
            if path.exists():shutil.copy2(path,summary/filename)
        print(name,len(paths),sum(r['bytes'] for r in records),flush=True)
    (out/'README.md').write_text('''# Remaining supplementary evidence

Extract **every** ZIP part of a stage at repository root; each part is an independent valid ZIP containing original `results/` paths. No binary concatenation is needed. `manifest.json` records every file and ZIP SHA256; CRC validation was performed after archive creation. Files are grouped by size, never selected by performance.

`summaries/` contains browsable protocols, tables and training curves. Paired tables and final interpretation are under `docs/experiments/`. Reconstructed training checkpoints are supplementary controls, not replacements for the frozen main checkpoints.

Stages appear only after their completion markers exist. The final archive command uses `--require-all` so a missing stage is an error.
''')

if __name__=='__main__':main()
