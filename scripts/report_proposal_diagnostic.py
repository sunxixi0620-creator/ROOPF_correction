"""Report and archive frozen-state diagnostics; no objective calls."""
from pathlib import Path
import json,sys,zipfile
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from roopf.experiment_io import sha256,write_json
run=ROOT/'results/proposal_diagnostic_20260930'
out=ROOT/'docs/revision/proposal_diagnostic'
df=pd.read_csv(out/'states.csv')
fig,axes=plt.subplots(1,3,figsize=(13,3.5),layout='constrained')
for behavior,g in df.groupby('behavior'):
 m=g.groupby('step').mean(numeric_only=True)
 axes[0].plot(100+2*m.index,m.potential_new,marker='o',label=behavior+' states')
 axes[1].plot(100+2*m.index,100*m.opportunity,marker='o',label=behavior+' states')
 r=g[g.opportunity>0].groupby('step').realized.mean()
 axes[2].plot(100+2*r.index,100*r,marker='o',label=behavior+' states')
for ax,title in zip(axes,['Proposal potential (state-normalized)','Opportunity rate (%)','Opportunity realization (%)']):
 ax.set_title(title);ax.set_xlabel('Paid evaluations before decision');ax.grid(alpha=.2);ax.legend()
fig.savefig(out/'diagnostic.png',dpi=180);fig.savefig(out/'diagnostic.pdf');plt.close(fig)
identity=json.loads((run/'identity.json').read_text())
for p,h in identity['sources'].items():assert sha256(ROOT/p)==h
for f in (run/'cases').glob('*.json'):
 if f.name=='identity.json':continue
 assert sha256(f.with_suffix('.pt'))==json.loads(f.read_text())['data_sha256']
assert len(list((run/'cases').glob('*.pt')))==288
artifact=ROOT/'artifacts/proposal_diagnostic_v1'
artifact.mkdir(parents=True,exist_ok=True)
archive=artifact/'diagnostic.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for p in sorted(run.rglob('*')):
  if p.is_file() and p.suffix!='.lock':z.write(p,str(p.relative_to(ROOT)))
 for p in identity['sources']:z.write(ROOT/p,p)
 z.write(__file__,'scripts/report_proposal_diagnostic.py')
 for p in sorted(out.glob('*')):
  if p.is_file():z.write(p,str(p.relative_to(ROOT)))
with zipfile.ZipFile(archive) as z:assert z.testzip() is None
write_json(artifact/'MANIFEST.json',dict(sha256=sha256(archive),bytes=archive.stat().st_size,
 cases_verified=288,rows=13824,zip_crc_passed=True))
(artifact/'README.md').write_text('# Same-state diagnosis v1\n\nExtract at repository root. Parent trajectories/checkpoints are in complementary_proposal_v1.\nNo new training. All288 replays match exactly; teachers run after replay.\nFailed-attempt cost bounds are recorded separately. See docs/revision/proposal_diagnostic/CONCLUSIONS.zh-CN.md.\n')
print('Verified288 cases and archive',archive.stat().st_size)
