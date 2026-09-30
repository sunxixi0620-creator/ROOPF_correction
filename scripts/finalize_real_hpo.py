"""Independent metric checks, cost accounting, and a checksummed pilot archive."""
from pathlib import Path
import csv, hashlib, json, zipfile
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'results/real_hpo_prior_20260930'
OUT=ROOT/'docs/revision/real_hpo_prior'
ART=ROOT/'artifacts/real_hpo_prior_v1'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    results=json.loads((OUT/'RESULTS.json').read_text());records=[]
    for p in sorted((RUN/'traces').glob('*.json')):
        z=json.loads(p.read_text());tab=np.load(RUN/'tables'/f"{z['task']}.npz")
        y=tab['error'];ids=z['indices'];regret=(np.minimum.accumulate(y[ids])-min(y))/max(max(y)-min(y),1e-12)
        hits=[i+1 for i,r in enumerate(regret) if r<=.05]
        records.append(dict(task=z['task'],pair=''.join(map(str,z['pair'])),seed=z['seed'],method=z['method'],auc=float(sum(regret[3:])/29),
            hit=min(hits) if hits else 33,success=bool(hits),final_error=float(min(y[ids])),
            measured_objective_seconds=float(sum(tab['seconds'][ids])),policy_seconds=z['seconds'],
            simulated_total_seconds=float(sum(tab['seconds'][ids]))+z['seconds']))
    assert len(records)==945
    for m,row in results['methods'].items():
        subset=[r for r in records if r['method']==m]
        for k in ('auc','hit','success','final_error'):assert np.isclose(np.mean([r[k] for r in subset]),row[k],rtol=1e-12,atol=1e-12)
    pairs=sorted(set(r['pair'] for r in records));g=np.random.default_rng(int(hashlib.sha256(json.dumps(('bootstrap',)).encode()).hexdigest()[:15],16))
    boot=g.integers(0,len(pairs),(10000,len(pairs)))
    for c in results['contrasts']:
        grouped=[]
        for p in pairs:
            v=[r for r in records if r['pair']==p]
            grouped.append(np.mean([r[c['metric']] for r in v if r['method']==c['control']])-np.mean([r[c['metric']] for r in v if r['method']=='F']))
        assert np.isclose(np.mean(grouped),c['improvement'],rtol=1e-12,atol=1e-12)
        interval=np.quantile(np.array(grouped)[boot].mean(1),[.05/12,1-.05/12])
        assert np.allclose(interval,c['ci'],rtol=1e-12,atol=1e-12)
        assert c['passed']==bool(c['improvement']>=c['threshold'] and interval[0]>0 and min(c['seed_effects'])>0)
    assert results['joint_pass']==all(c['passed'] for c in results['contrasts'])
    with (OUT/'trajectory_costs.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    cost={m:{k:float(np.mean([r[k] for r in records if r['method']==m])) for k in ('measured_objective_seconds','policy_seconds','simulated_total_seconds')} for m in results['methods']}
    (OUT/'COSTS.json').write_text(json.dumps(dict(methods=cost,source_fits=15360,target_table_fits=11520,table_lookups=30240,
        source_cumulative_fit_seconds=results['source_fit_seconds'],target_cumulative_fit_seconds=results['target_fit_seconds'],
        timing_caveat='Objective times measured under parallel table collection; sums are reconstructed per-trajectory costs, not serial end-to-end benchmarks.'),indent=2))
    (OUT/'METRIC_VERIFICATION.json').write_text(json.dumps(dict(independent_metric_recompute=True,bootstrap_intervals_recomputed=True,trajectories=945,extra_objective_fits=0),indent=2))
    ART.mkdir(parents=True,exist_ok=True)
    files=sorted([p for folder in (RUN,OUT) for p in folder.rglob('*') if p.is_file()]+[
        ROOT/'scripts/real_hpo_prior.py',Path(__file__),ROOT/'docs/experiments/REAL_HPO_PRIOR_PROTOCOL.md'])
    manifest={str(p.relative_to(ROOT)):sha(p) for p in files}
    archive=ART/'real_hpo_prior_v1.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in files:z.write(p,str(p.relative_to(ROOT)))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for p,h in manifest.items():assert hashlib.sha256(z.read(p)).hexdigest()==h
    (ART/'MANIFEST.json').write_text(json.dumps(dict(archive_sha256=sha(archive),files=manifest),indent=2))
    print(json.dumps(dict(archive_bytes=archive.stat().st_size,files=len(files),verified=True),indent=2))

if __name__=='__main__':main()
