"""Complete-case paired summaries; no outcome-driven case selection."""
import csv,json
from pathlib import Path
from collections import defaultdict
import numpy as np
from supplementary_experiments import ROOT,write_csv

OUT=ROOT/'docs/experiments/remaining_tables';OUT.mkdir(exist_ok=True)
def read_csv(p):return list(csv.DictReader(p.open()))
def paired(a,b):
    assert set(a)==set(b)
    diff=np.array([float(a[s])-float(b[s]) for s in sorted(a)])
    rng=np.random.default_rng(20260928);boot=diff[rng.integers(0,len(diff),(5000,len(diff)))].mean(1)
    return dict(n=len(diff),difference=diff.mean(),ci_low=np.quantile(boot,.025),ci_high=np.quantile(boot,.975),
        wins=int((diff<0).sum()),ties=int((diff==0).sum()),losses=int((diff>0).sum()))

def operators():
    root=ROOT/'results/remaining_operators_20260928'
    if not (root/'COMPLETE').exists():return
    rows=json.loads((root/'all_rows.json').read_text());assert len(rows)==2340
    d=defaultdict(dict)
    for x in rows:d[x['case'],x['mode']][x['seed']]=x['final']
    table=[]
    for case in sorted({x['case'] for x in rows}):
        for mode in sorted({x['mode'] for x in rows}-{'full'}):
            table.append(dict(case=case,comparator=mode,full_mean=np.mean(list(d[case,'full'].values())),comparator_mean=np.mean(list(d[case,mode].values())),**paired(d[case,'full'],d[case,mode])))
    write_csv(OUT/'operators_paired.csv',table)
    origins=defaultdict(lambda:[0,0,0])
    for x in rows:
        if x['mode']!='full':continue
        for op,counts in x['origin_stats']['op_evals'].items():
            v=origins[x['case'],op];v[0]+=counts;v[1]+=x['origin_stats']['op_admitted'][op];v[2]+=x['origin_stats']['op_best_improvements'][op]
    write_csv(OUT/'operator_origins.csv',[dict(case=k[0],operator=k[1],evaluations=v[0],admitted=v[1],incumbent_improvements=v[2]) for k,v in origins.items()])

def replay():
    root=ROOT/'results/remaining_replay_20260928'
    if not (root/'COMPLETE').exists():return
    rows=json.loads((root/'all_rows.json').read_text());assert len(rows)==1260
    base={(r['case'],r['seed']):r for r in rows if r['force_at']<0};groups=defaultdict(list);pools=[]
    for x in rows:
        if x['force_at']<0:
            for p in x['pool_rows']:pools.append(dict(case=x['case'],seed=x['seed'],**p))
        else:groups[x['case'],x['force_at'],x['force_kind']].append(x)
    table=[]
    for k,rs in sorted(groups.items()):
        assert len(rs)==30 and all(r['prefix_exact'] for r in rs)
        a={r['seed']:r['final'] for r in rs};b={r['seed']:base[k[0],r['seed']]['final'] for r in rs}
        changed=[r for r in rs if r['intervention']['changed']]
        effects=np.array([r['final']-base[k[0],r['seed']]['final'] for r in changed])
        table.append(dict(case=k[0],intervention_at=k[1],intervention=k[2],changed=len(changed),
          changed_better=int((effects<0).sum()),changed_tied=int((effects==0).sum()),changed_worse=int((effects>0).sum()),**paired(a,b)))
    write_csv(OUT/'single_intervention_replay.csv',table);write_csv(OUT/'same_pool_proxy.csv',pools)
    optroot=ROOT/'results/remaining_operators_20260928'
    if (optroot/'COMPLETE').exists():
        for case,seed in base:
            stem=f'{case}_s{seed}_full_-1_anchor.npz'
            aa=np.load(root/stem);bb=np.load(optroot/stem)
            assert np.array_equal(aa['points'],bb['points']) and np.array_equal(aa['trail'],bb['trail'])
        (OUT/'diagnostic_parity.json').write_text(json.dumps({'full_trajectories_exact_with_and_without_pool_truth':len(base)}))

def uav():
    root=ROOT/'results/remaining_uav_20260928'
    if not (root/'COMPLETE').exists():return
    rows=read_csv(root/'raw_results.csv');assert len(rows)==1050
    d=defaultdict(list)
    for r in rows:d[r['fid'],r['method']].append(r)
    table=[];comparison=[]
    for (fid,method),rs in sorted(d.items()):
        vals=np.array([float(r['final']) for r in rs]);assert len(vals)==30
        table.append(dict(fid=fid,method=method,mean=vals.mean(),std=vals.std(ddof=1),
            feasible=sum(int(r['geometrically_feasible']) for r in rs),runs=len(rs),
            mean_clearance=np.mean([float(r['min_clearance']) for r in rs])))
        if method!='full':comparison.append(dict(fid=fid,comparator=method,**paired({r['seed']:r['final'] for r in d[fid,'full']},{r['seed']:r['final'] for r in rs})))
    write_csv(OUT/'uav_methods.csv',table);write_csv(OUT/'uav_paired.csv',comparison)

def external(method):
    root=ROOT/f'results/remaining_{method}_20260928'
    if not (root/'COMPLETE').exists():return
    gp=read_csv(root/'raw_results.csv');old=read_csv(ROOT/'results/independent_stage3_20260928/raw_results.csv')
    d=defaultdict(dict)
    for r in gp+old:d[r['suite'],r['fid'],r['instance'],r['method']][r['seed']]=r['final']
    table=[]
    for suite,fid,inst in sorted({k[:3] for k in d}):
        table.append(dict(suite=suite,fid=fid,instance=inst,comparator=method,**paired(d[suite,fid,inst,'full'],d[suite,fid,inst,method])))
    write_csv(OUT/(method+'_paired.csv'),table)

def retraining():
    for name in ['native20_eval_20260928','order_original_eval_20260928','order_curated_eval_20260928']:
        root=ROOT/'results'/name
        if not (root/'COMPLETE').exists():continue
        rows=read_csv(root/'raw_results.csv');d=defaultdict(dict)
        for r in rows:d[r['budget'],r['fid'],r['instance'],r['method']][r['seed']]=r['final']
        table=[]
        for budget,fid,inst in sorted({k[:3] for k in d}):
            for m in sorted({k[-1] for k in d}-{'full'}):
                table.append(dict(budget=budget,fid=fid,instance=inst,comparator=m,**paired(d[budget,fid,inst,'full'],d[budget,fid,inst,m])))
        write_csv(OUT/(name+'_paired.csv'),table)
    roots=[ROOT/'results'/f'order_{m}_eval_20260928' for m in ['original','curated']]
    if all((p/'COMPLETE').exists() for p in roots):
        d=defaultdict(dict)
        for order,p in zip(['original','curated'],roots):
            for r in read_csv(p/'raw_results.csv'):d[r['fid'],r['instance'],r['method'],order][r['seed']]=r['final']
        table=[dict(fid=f,instance=i,method=m,**paired(d[f,i,m,'curated'],d[f,i,m,'original'])) for f,i,m in sorted({k[:3] for k in d})]
        write_csv(OUT/'training_order_paired.csv',table)

def main():
    operators();replay();uav();external('gp_ei');external('surr_rlde');retraining()
    print('Updated',str(OUT))

if __name__=='__main__':main()
