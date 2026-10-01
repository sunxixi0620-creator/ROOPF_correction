"""Freeze a response-blind raw-group split of the two permitted HPO-B tables."""
import json,hashlib,math,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from roopf.experiment_io import write_json,sha256
OUT=ROOT/'docs/research/hpob_preparation'
DATA=ROOT/'results/hpob_source_preparation/data'


def main():
    audit=json.loads((OUT/'OPENML_IDENTITY_AUDIT.json').read_text())
    assert not audit['errors']
    mapping={r['id']:r['raw_dataset_id'] for r in audit['metadata']}
    def group(raw):
        if raw in ('29','40509'):return 'credit_approval'
        if raw in ('1552','1553','1554'):return 'autouniv_au7'
        return 'openml_'+raw
    candidates={}
    for filename in ('meta-train-dataset.json','meta-validation-dataset.json'):
        tables=json.loads((DATA/filename).read_text())
        for space,table in tables.items():
            for tid,task in table.items():
                key=(space,group(mapping[tid]))
                # Choose representative by public ID, never by outcomes/coverage.
                if key in candidates and int(candidates[key]['task_id'])<int(tid):continue
                x=np.asarray(task['X'],dtype=np.float64)
                unique,indices=np.unique(x,axis=0,return_index=True)
                ok=bool(x.ndim==2 and 1<=x.shape[1]<=64 and np.isfinite(x).all() and
                        x.min()>=-1e-6 and x.max()<=1+1e-6 and len(unique)>=105)
                candidates[key]=dict(space=space,group=key[1],task_id=tid,raw_dataset_id=mapping[tid],
                    original_file=filename,dim=x.shape[1],points=len(x),unique_points=len(unique),
                    eligible_inputs=ok,canonical_indices_sha256=hashlib.sha256(np.sort(indices).astype('<i8').tobytes()).hexdigest())
    groups=sorted({g for _,g in candidates},key=lambda g:hashlib.sha256(('hpob_raw_group_v1|'+g).encode()).hexdigest())
    n=len(groups);a=math.floor(.6*n);b=a+math.floor(.2*n)
    roles={g:('source' if k<a else 'development' if k<b else 'confirmation') for k,g in enumerate(groups)}
    sources={}
    for (space,g),task in candidates.items():
        if roles[g]=='source' and task['eligible_inputs']:sources.setdefault(space,[]).append(g)
    rows=[]
    for (space,g),task in sorted(candidates.items()):
        task.update(role=roles[g],source_groups=sorted(sources.get(space,[])),
                    eligible=task['eligible_inputs'] and len(sources.get(space,[]))>=5)
        rows.append(task)
    counts={role:sum(t['eligible'] and t['role']==role for t in rows) for role in ('source','development','confirmation')}
    write_json(OUT/'RAW_GROUPED_SPLIT.json',dict(groups=roles,group_counts={r:list(roles.values()).count(r) for r in counts},
        eligible_tasks=counts,eligible_spaces={r:sorted({t['space'] for t in rows if t['eligible'] and t['role']==r},key=int) for r in counts},
        tasks=rows,source_sha256=sha256(Path(__file__)),protocol_sha256=sha256(ROOT/'docs/experiments/HPOB_RAW_GROUPED_PROTOCOL.md'),
        metadata_sha256=sha256(OUT/'OPENML_IDENTITY_AUDIT.json'),response_statistics_computed=False,
        optimization_calls=0,official_test_file_downloaded=False))
    print(json.dumps(dict(groups=n,role_groups={r:list(roles.values()).count(r) for r in counts},eligible=counts),indent=2))


if __name__=='__main__':main()
