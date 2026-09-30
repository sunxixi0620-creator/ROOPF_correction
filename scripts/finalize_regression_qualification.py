"""Independent task-screen metric verification, figures and raw data archive."""
from pathlib import Path
import hashlib,json,zipfile
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'results/regression_qualification_20260930';OUT=ROOT/'docs/revision/regression_qualification';ART=ROOT/'artifacts/regression_qualification_v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def correlation(a,b):return float(np.corrcoef(pd.Series(a).rank(),pd.Series(b).rank())[0,1])
def main():
    result=json.loads((OUT/'RESULTS.json').read_text());identity=json.loads((RUN/'identity.json').read_text());assert identity['data']==json.loads((RUN/'DATA_PROVENANCE.json').read_text())
    tables={n:np.array([np.load(RUN/'tables'/f'{n}_{s}.npz')['loss'] for s in range(3)]) for n in identity['data']}
    p=np.mean([v.mean(0) for n,v in tables.items() if identity['data'][n]['role']=='source'],axis=0)
    assert np.array_equal(np.load(OUT/'source_prior.npz')['prior'],p)
    assert identity['data']['energy']['features']==[f'X{i}' for i in range(1,9)] and identity['data']['energy']['target']=='Y1'
    gates={};coverage=[]
    for row in result['rows']:
        v=tables[row['dataset']];good=v<=v.min(1)[:,None]+.05;n=good.sum(1)
        probs=[1-float(np.prod([(128-int(k)-i)/(128-i) for i in range(4)])) if k<=124 else 1. for k in n]
        assert np.isclose(good.mean(),row['mean_good_fraction'],atol=1e-14)
        assert np.isclose(np.mean(probs),row['random_four_success'],atol=1e-14)
        assert np.isclose(np.mean([correlation(v[i],v[j]) for i,j in ((0,1),(0,2),(1,2))]),row['mean_split_spearman'],atol=1e-14)
        assert row['difficulty_pass']==bool(good.mean()<=.1 and np.mean(probs)<=.35)
        if row['role']=='development':
            mean=v.mean(0);coverage.append(mean<=mean.min()+.05)
            assert np.isclose(correlation(p,mean),row['source_spearman'],atol=1e-14)
    max_cover=int(np.array(coverage).sum(0).max());assert max_cover==result['max_single_config_coverage']
    for role in ('source','development'):gates[role+'_difficulty']=sum(r['difficulty_pass'] for r in result['rows'] if r['role']==role)>=2
    gates['heterogeneous']=max_cover<=2;gates['source_relation']=sum(r.get('source_spearman',-1)>=.2 for r in result['rows'])>=2
    gates['stable_splits']=sum(r['mean_split_spearman']>=.5 for r in result['rows'] if r['role']=='development')>=2
    fits=sum(json.loads(p.read_text())['fits'] for p in (RUN/'tables').glob('*.json'));assert fits==2358
    repeat_errors=[];warnings=0
    for f in (RUN/'tables').glob('*.npz'):
        z=np.load(f);repeat_errors.extend(abs(z['loss'][[0,63,127]]-z['repeat_loss']));warnings+=int(z['warnings'].sum()+z['repeat_warnings'].sum())
    gates['integrity']=max(repeat_errors)<=1e-10 and warnings==0
    assert gates==result['gates'] and all(gates.values())==result['joint_pass']
    (OUT/'INDEPENDENT_VERIFICATION.json').write_text(json.dumps(dict(metrics_recomputed=True,all_gates_recomputed=True,source_prior_recomputed=True,fit_count=fits,extra_fits=0),indent=2))
    report=OUT/'CONCLUSIONS.zh-CN.md';report.write_text(report.read_text().replace('|nan|','|—|'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows=result['rows'];fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,key,threshold,title in zip(axes,['mean_good_fraction','random_four_success'],[.1,.35],['Near-optimal configuration fraction','Random-four success probability']):
        ax.barh([r['dataset'] for r in rows],[r[key] for r in rows],color=['#537da0' if r['role']=='source' else '#c5814d' for r in rows]);ax.axvline(threshold,color='black',linestyle='--',label='Screening threshold');ax.set_title(title);ax.set_xlim(0,1);ax.legend(fontsize=8)
    fig.suptitle('SVR task qualification: fixed source/development collection');fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(OUT/f'qualification.{ext}',dpi=150)
    plt.close(fig)
    ART.mkdir(parents=True,exist_ok=True)
    files=sorted([p for folder in (RUN,OUT) for p in folder.rglob('*') if p.is_file()]+[Path(__file__),ROOT/'scripts/regression_task_qualification.py',ROOT/'docs/experiments/REGRESSION_TASK_QUALIFICATION_PROTOCOL.md'])
    manifest={str(p.relative_to(ROOT)):sha(p) for p in files};archive=ART/'regression_qualification_v1.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in files:z.write(p,str(p.relative_to(ROOT)))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for name,h in manifest.items():assert hashlib.sha256(z.read(name)).hexdigest()==h
    (ART/'MANIFEST.json').write_text(json.dumps(dict(archive_sha256=sha(archive),files=manifest),indent=2))
    print(dict(files=len(files),archive_bytes=archive.stat().st_size,verified=True))
if __name__=='__main__':main()
