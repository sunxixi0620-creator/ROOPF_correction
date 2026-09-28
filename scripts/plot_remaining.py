"""Standalone publication-editable PDF figures; all cases retained."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import rankdata
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];TABLE=ROOT/'docs/experiments/remaining_tables';OUT=ROOT/'docs/experiments/remaining_figures';OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'ps.fonttype':42})
LABELS={'full':'ROOPF','anchor_only':'Anchor only','cma_es':'CMA-ES','de':'DE',
 'random':'Random','gp_ei':'GP-EI','surr_rlde':'Surr-RLDE','score_only':'Score only',
 'no_residual':'No residual','full_transferred_residual':'10-D residual transfer'}
def save(fig,name):
    fig.tight_layout();fig.savefig(OUT/(name+'.pdf'));fig.savefig(OUT/(name+'.png'),dpi=180);plt.close(fig)

def external():
    paths=[ROOT/f'results/{n}/raw_results.csv' for n in ['independent_stage3_20260928','remaining_gp_ei_20260928','remaining_surr_rlde_20260928']]
    if not all(p.exists() for p in paths):return
    df=pd.concat([pd.read_csv(p) for p in paths]);means=df.groupby(['suite','fid','instance','method']).final.mean().unstack('method')
    methods=['full','anchor_only','cma_es','de','random','gp_ei','surr_rlde'];ranks=means[methods].apply(lambda row:pd.Series(rankdata(row),index=methods),axis=1)
    ranks.to_csv(TABLE/'external_seven_method_case_ranks.csv')
    fig,axes=plt.subplots(1,2,figsize=(11,3.8))
    for ax,suite in zip(axes,['coco','cec2017']):
        v=ranks.loc[suite].mean();ax.bar([LABELS[m] for m in methods],v,color=['#be4444']+['#48789b']*6);ax.set_title(suite+' (10-D, 300 NFE)');ax.set_ylabel('Mean case rank (lower is better)');ax.set_ylim(0,7);ax.tick_params(axis='x',rotation=35)
    save(fig,'external_seven_methods')

def mechanism():
    p=TABLE/'same_pool_proxy.csv'
    if not p.exists():return
    x=pd.read_csv(p);cases=sorted(x.case.unique());fig,axes=plt.subplots(1,2,figsize=(11,3.8))
    for ax,col,title in zip(axes,['spearman_mu','spearman_score'],['Proxy mean vs. true pool values','Final score vs. true pool values']):
        ax.boxplot([x[x.case==c][col].dropna().to_numpy() for c in cases],tick_labels=cases,showfliers=False)
        ax.axhline(0,color='gray',lw=1);ax.set_ylim(-1,1);ax.set_ylabel('Spearman correlation');ax.set_title(title);ax.tick_params(axis='x',rotation=35)
    save(fig,'same_pool_rank_diagnostics')
    r=pd.read_csv(TABLE/'single_intervention_replay.csv');r['suite']=r.case.str.split('_').str[0]
    grouped=r.groupby(['suite','intervention'])[['changed_better','changed_tied','changed_worse']].sum()
    fig,ax=plt.subplots(figsize=(8,3.8));bottom=np.zeros(len(grouped))
    for col,label,color in [('changed_better','Forced branch better','#4c9471'),('changed_tied','Same endpoint','#a5a5a5'),('changed_worse','Forced branch worse','#b95858')]:
        v=grouped[col].to_numpy();ax.bar(range(len(v)),v,bottom=bottom,label=label,color=color);bottom+=v
    ax.set_xticks(range(len(grouped)),[f'{a}: force {b}' for a,b in grouped.index]);ax.set_ylabel('Changed single interventions');ax.legend();save(fig,'single_intervention_effects')

def uav():
    p=TABLE/'uav_methods.csv'
    if not p.exists():return
    x=pd.read_csv(p);fig,axes=plt.subplots(1,2,figsize=(11,3.8));methods=['full','anchor_only','score_only','cma_es','de']
    for method in methods:
        d=x[x.method==method].sort_values('fid');axes[0].plot(d.fid,d['mean'],marker='o',label=LABELS[method]);axes[1].plot(d.fid,d.feasible/d.runs,marker='o',label=LABELS[method])
    axes[0].set_ylabel('Mean soft-penalty objective');axes[1].set_ylabel('Geometrically feasible fraction');axes[1].set_ylim(-.03,1.03)
    for ax in axes:ax.set_xlabel('UAV proxy scenario');ax.set_xticks(range(1,6))
    axes[0].legend(fontsize=8);save(fig,'uav_objective_and_geometry')

def native():
    p=TABLE/'native20_eval_20260928_paired.csv'
    if not p.exists():return
    x=pd.read_csv(p);fig,axes=plt.subplots(1,2,figsize=(11,3.8))
    for ax,budget in zip(axes,[300,600]):
        d=x[x.budget==budget];methods=list(d.comparator.unique());left=np.zeros(len(methods))
        for label,fun,color in [('Full better',lambda v:v<0,'#4c9471'),('Tie',lambda v:v==0,'#a5a5a5'),('Full worse',lambda v:v>0,'#b95858')]:
            counts=np.array([fun(d[d.comparator==m].difference).sum() for m in methods]);ax.barh([LABELS[m] for m in methods],counts,left=left,label=label,color=color);left+=counts
        ax.set_title(f'Native20-D, {budget} NFE');ax.set_xlabel('Number of case means');ax.set_xlim(0,48)
    axes[0].legend(fontsize=8);save(fig,'native20_comparisons')

if __name__=='__main__':external();mechanism();uav();native()
