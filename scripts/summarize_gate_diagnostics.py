"""Aggregate logged same-state counterfactual diagnostics without causal overclaim."""
import argparse,csv,json
from collections import defaultdict
from pathlib import Path
from supplementary_experiments import write_csv


def main():
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);a=p.parse_args();root=a.directory
    assert (root/'COMPLETE').exists()
    groups=defaultdict(lambda:defaultdict(float))
    for directory in root.iterdir():
        if not directory.is_dir() or not (directory/'COMPLETE').exists():continue
        metadata=json.loads((directory/'rows.json').read_text())[0]
        group='bbob' if metadata['case'].startswith('bbob') else 'cec_shifted'
        key=(group,float(metadata['warmup']))
        with (directory/'decisions.csv').open() as f:
            for row in csv.DictReader(f):
                out=groups[key];out['gate_opportunities']+=1
                out['accepted']+=int(row['accepted_portfolio'])
                out['residual_active']+=int(row['pool_residual_active'])
                if not int(row['diagnostic_sampled']):continue
                out['sampled']+=1;out['residual_changed']+=int(row['residual_decision_changed'])
                if not int(row['proposed_portfolio']):continue
                out['sampled_portfolio_proposals']+=1
                delta=float(row['diagnostic_proposed_value'])-float(row['diagnostic_anchor_value'])
                accepted=int(row['accepted_portfolio'])
                if accepted:
                    out['sampled_accepted']+=1
                    out['accepted_better']+=int(delta<0);out['accepted_equal']+=int(delta==0);out['accepted_worse']+=int(delta>0)
                else:
                    out['sampled_rejected']+=1
                    out['rejected_better']+=int(delta<0);out['rejected_equal']+=int(delta==0);out['rejected_worse']+=int(delta>0)
    names=['gate_opportunities','accepted','residual_active','sampled','residual_changed','sampled_portfolio_proposals','sampled_accepted','accepted_better','accepted_equal','accepted_worse','sampled_rejected','rejected_better','rejected_equal','rejected_worse']
    rows=[{'group':g,'warmup':w,**{k:int(x[k]) for k in names}} for (g,w),x in sorted(groups.items())]
    write_csv(root/'gate_counterfactual_summary.csv',rows)
    print(json.dumps(rows),flush=True)

if __name__=='__main__':main()
