"""Plot already-completed results; no objective evaluations or model selection."""
import json
from pathlib import Path
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from roopf.experiment_io import sha256,write_json
OUT=ROOT/'docs/revision/free_allocation/development'
data=json.loads((OUT/'COMPARISONS.json').read_text())
fig,axes=plt.subplots(1,3,figsize=(12,3.5))
for ax,(condition,effects) in zip(axes,data.items()):
    for row,name in enumerate(('Free_vs_A','Free_vs_O','Free_vs_F')):
        v=effects[name];lo,hi=v['ci_adjusted'];mean=v['mean']
        ax.plot([lo,hi],[row,row],color='tab:blue')
        ax.scatter([mean],[row],color='tab:blue')
    ax.axvline(0,color='black',linewidth=1)
    ax.axvline(.005,color='gray',linestyle=':',linewidth=1)
    ax.set_yticks([0,1,2],['Free - A','Free - O','Free - F'])
    ax.invert_yaxis();ax.set_title(condition);ax.set_xlabel('Paired bounded-utility difference')
    ax.grid(axis='x',alpha=.2)
fig.suptitle('Development diagnosis; adjusted intervals; dotted line = practical threshold',fontsize=10)
fig.tight_layout();fig.savefig(OUT/'allocation_effects.png',dpi=180);fig.savefig(OUT/'allocation_effects.pdf');plt.close(fig)
curves=pd.read_csv(OUT/'curves.csv')
fig,axes=plt.subplots(1,3,figsize=(12,3.5))
for ax,(d,b) in zip(axes,((10,300),(20,300),(20,600))):
    part=curves[(curves.dim==d)&(curves.budget==b)]
    for name in ('A','O','F','Free'):
        a=part[part.method==name];ax.plot(a.nfe,a.utility,label=name)
    ax.set_title(f'{d}D, {b} evaluations');ax.set_xlabel('Consumed NFE');ax.set_ylabel('Mean bounded improvement')
    ax.legend();ax.grid(alpha=.2)
fig.suptitle('Development tasks, post-initialization curves; descriptive means',fontsize=10)
fig.tight_layout();fig.savefig(OUT/'allocation_curves.png',dpi=180);fig.savefig(OUT/'allocation_curves.pdf');plt.close(fig)
write_json(OUT/'PLOT_IDENTITY.json',dict(objective_calls=0,source_sha256=sha256(__file__),
    comparisons_sha256=sha256(OUT/'COMPARISONS.json'),curves_sha256=sha256(OUT/'curves.csv')))
