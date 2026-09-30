"""Reporting/archive only; never calls an objective or selects a model."""
import json
from pathlib import Path
import sys
import zipfile
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from roopf.experiment_io import sha256,write_json


def main():
    run=ROOT/'results/complementary_proposal_20260930'
    out=ROOT/'docs/revision/complementary_proposal'
    result=json.loads((run/'COMPLETE.json').read_text())
    assert result['all_workers_joined']
    for name,digest in json.loads((run/'identity.json').read_text())['sources'].items():
        assert sha256(ROOT/name)==digest
    # Audit every committed case, including all unplotted raw trajectories.
    cases=0
    for folder in ('data','search'):
        for manifest in sorted((run/folder).glob('*.json')):
            if manifest.name=='identity.json':continue
            meta=json.loads(manifest.read_text())
            assert sha256(manifest.with_suffix('.pt'))==meta['data_sha256']
            cases+=1
    assert cases==144+(720 if result['search'] else 0)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    def panel(ax,labels,effects,title):
        for i,e in enumerate(effects):
            ax.plot([e['lower'],e['upper']],[i,i],color='#28649a',lw=2)
            ax.plot(e['mean'],i,'o',color='#28649a')
        ax.axvline(0,color='gray',linestyle='--',lw=1)
        ax.set_yticks(range(len(labels)),labels)
        ax.set_title(title)
        ax.set_xlabel('Difference in bounded utility')
        ax.grid(axis='x',alpha=.2)
    panel(axes[0],['New - old anchor'],[result['proposal']],
          'Proposal potential on O states (95% CI)')
    if result['search']:
        panel(axes[1],['New - O','New - Old','New - A'],
              [result['search'][m] for m in ('O','Old','A')],
              'Paid search (97.5%; A descriptive 95%)')
    else:
        axes[1].text(.5,.5,'Search not opened: proposal gate failed',ha='center',transform=axes[1].transAxes)
    fig.savefig(out/'effects.png',dpi=180)
    fig.savefig(out/'effects.pdf')
    plt.close(fig)
    dest=ROOT/'artifacts/complementary_proposal_v1'
    dest.mkdir(parents=True,exist_ok=True)
    groups={}
    for path in sorted(run.rglob('*')):
        if not path.is_file() or path.suffix=='.lock':continue
        rel=path.relative_to(run)
        if rel.parts[0]=='data' and path.stem!='identity':
            group='data_'+path.name.split('_')[0]
        elif rel.parts[0]=='search' and path.stem!='identity':
            group='search_'+path.name.split('_')[0]
        else:group='metadata_models_sources'
        groups.setdefault(group,[]).append(path)
    manifest={}
    for group,paths in groups.items():
        target=dest/(group+'.zip')
        with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
            for path in paths:z.write(path,str(path.relative_to(run)))
            if group=='metadata_models_sources':
                z.write(__file__,'reporting_source/report_complementary_proposal.py')
        with zipfile.ZipFile(target) as z:assert z.testzip() is None
        assert target.stat().st_size<95_000_000
        manifest[target.name]=dict(sha256=sha256(target),bytes=target.stat().st_size,files=len(paths))
    write_json(dest/'MANIFEST.json',manifest)
    write_json(out/'PACKAGE_VERIFICATION.json',dict(cases_verified=cases,
        payloads=len(manifest),zip_crc_passed=True,sha256_passed=True,
        total_bytes=sum(v['bytes'] for v in manifest.values()),
        report_source_sha256=sha256(__file__),result_sha256=sha256(run/'COMPLETE.json')))
    (dest/'README.md').write_text('# Contextual proposal pilot v1\n\n'
        'Extract every ZIP into `results/complementary_proposal_20260930/`.\n'
        'Includes source/protocol snapshot, selected/last models, optimizer states, histories,\n'
        'behavior states, teacher thresholds, and all required paid search trajectories.\n'
        'Every payload has SHA256 in MANIFEST.json; each case has its own verified manifest.\n'
        'Shared procedural families: development evidence, not independent external validation.\n'
        'See [conclusions](../../docs/revision/complementary_proposal/CONCLUSIONS.zh-CN.md)\n'
        'and [reproduction](../../docs/revision/complementary_proposal/REPRODUCE.md).\n')
    print(json.dumps(manifest,indent=2))


if __name__=='__main__':main()
