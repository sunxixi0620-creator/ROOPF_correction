"""Task-specific final values and complete win/tie/loss for external expansion."""
import argparse
import csv
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
from supplementary_experiments import write_csv


def main():
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);a=p.parse_args();root=a.directory
    assert (root/'COMPLETE').exists()
    rows=list(csv.DictReader((root/'raw_results.csv').open()))
    assert len(rows)==len({(x['suite'],x['fid'],x['instance'],x['method'],x['seed']) for x in rows})
    assert all(int(x['actual_nfe'])==300 for x in rows)
    d=defaultdict(list)
    for x in rows:d[(x['suite'],int(x['fid']),int(x['instance']),x['method'])].append(float(x['final']))
    comparisons=[];ranks=[]
    methods=['full','anchor_only','cma_es','de','random']
    for suite,fid,instance in sorted({k[:3] for k in d}):
        arrays=[np.array(d[(suite,fid,instance,m)]) for m in methods]
        assert all(len(x)==10 for x in arrays)
        means=[x.mean() for x in arrays];rr=rankdata(means)
        for m,x,rank in zip(methods,arrays,rr):
            ranks.append({'suite':suite,'fid':fid,'instance':instance,'method':m,'mean':x.mean(),'std':x.std(ddof=1),'rank':rank})
        for m,x in zip(methods[1:],arrays[1:]):
            comparisons.append({'suite':suite,'fid':fid,'instance':instance,'comparator':m,
                'full_mean':means[0],'comparator_mean':x.mean(),'full_minus_comparator':means[0]-x.mean()})
    write_csv(root/'per_case_methods.csv',ranks);write_csv(root/'comparisons.csv',comparisons)
    aggregates=[]
    for suite in ('coco','cec2017'):
        for m in methods[1:]:
            vals=[x['full_minus_comparator'] for x in comparisons if x['suite']==suite and x['comparator']==m]
            aggregates.append({'suite':suite,'comparator':m,'cases':len(vals),'wins':sum(x<0 for x in vals),'ties':sum(x==0 for x in vals),'losses':sum(x>0 for x in vals)})
    write_csv(root/'group_summary.csv',aggregates)
    lines=['# Frozen post-development benchmark expansion','',
        f'{len(rows)} trajectories; 300 true objective evaluations each; ten repetitions per method and case.', '',
        'COCO uses new BBOB instances 101/102. opfunu uses its versioned hybrid/composition classes. This is not a claim of entire unseen primitive families or official CEC competition conformance.', '',
        '| Suite | Comparator | Cases | ROOPF mean W/T/L |','|---|---|---:|---|']
    for x in aggregates:lines.append(f"| {x['suite']} | {x['comparator']} | {x['cases']} | {x['wins']}/{x['ties']}/{x['losses']} |")
    lines+=['','The strong improvement over the frozen anchor coexists with frequent losses to CMA-ES. No claim of broad baseline superiority is supported by these counts. Runtime records are parallel-execution diagnostics, not uncontended latency measurements. Per-case means and deviations are in per_case_methods.csv; no averages of unnormalized objectives across functions are used.']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,suite in zip(axes,('coco','cec2017')):
        values=[np.mean([x['rank'] for x in ranks if x['suite']==suite and x['method']==m]) for m in methods]
        ax.bar(methods,values);ax.set_title(suite);ax.set_ylabel('Mean case rank (lower is better)');ax.tick_params(axis='x',rotation=30)
    fig.tight_layout();fig.savefig(root/'mean_ranks.pdf');fig.savefig(root/'mean_ranks.png',dpi=170)
    print(json.dumps(aggregates),flush=True)

if __name__=='__main__':main()
