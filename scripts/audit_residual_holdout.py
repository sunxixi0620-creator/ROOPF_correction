"""GPU inference audit of the frozen residual on recovered held-out-function logs.

Uses all available pool rows from the seven residual holdouts, not the historical
1.2M sampled split. Never described as a whole-pipeline independent test.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    metrics=json.loads((ROOT/'artifacts/training_provenance/residual_training_metrics.json').read_text())
    holdout=set(metrics['holdout_fids'])
    checkpoint=ROOT/'checkpoints/residual_selector_generated36_d10.pt'
    payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
    state=payload['state_dict'];features=payload['feature_names']
    hidden=state['net.0.weight'].shape[0]
    model=nn.Sequential(nn.Linear(len(features),hidden),nn.LayerNorm(hidden),nn.GELU(),nn.Linear(hidden,hidden),nn.GELU(),nn.Linear(hidden,1))
    model.load_state_dict({k.removeprefix('net.'):v for k,v in state.items()})
    device='cuda' if torch.cuda.is_available() else 'cpu';model.to(device).eval()
    mean=torch.tensor(payload['mean'],device=device).reshape(1,-1)
    std=torch.tensor(payload['std'],device=device).reshape(1,-1).clamp_min(1e-6)
    source=Path('/home/skq/repos/xixi/main-parameter22/saferank_neurgo_v58_retrained_offline/research_v41_surrogate_portfolio/results_v58_generated36_logs')
    paths=sorted(source.glob('*pool_candidates.csv'))
    assert len(paths)==2
    source_hashes={}
    for path in paths:
        h=hashlib.sha256()
        with path.open('rb') as f:
            for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
        source_hashes[str(path)]=h.hexdigest()
    protocol={'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        'source_hashes':source_hashes,'holdout_fids':sorted(holdout),'device':device,
        'feature_names':features,'selection':'All portfolio rows on seven residual-held-out functions; not exact original sampled validation subset.',
        'scope':'Anchor may have seen these functions. Incumbent-improvement labels, not candidate-vs-anchor treatment effects.'}
    (args.output/'protocol.json').write_text(json.dumps(protocol,indent=2))
    values=[];labels=[];function_ids=[];read=0;start=time.perf_counter()
    for path in paths:
        for df in pd.read_csv(path,chunksize=100000):
            read+=len(df)
            identifiers=df['target'].astype(str)+':'+df['fid'].astype(str)
            df=df.loc[identifiers.isin(holdout)&(df['is_baseline']==0)].copy()
            if df.empty:continue
            columns=[]
            for name in features:
                if name.startswith('op_'):
                    arr=(df['op_name']==name[3:]).to_numpy(dtype=np.float32)
                else:
                    key=name.removesuffix('_missing')
                    raw=pd.to_numeric(df[key],errors='coerce').to_numpy(dtype=np.float32)
                    arr=(~np.isfinite(raw)).astype(np.float32) if name.endswith('_missing') else np.nan_to_num(raw,nan=0.,posinf=0.,neginf=0.)
                columns.append(arr)
            x=np.stack(columns,axis=1)
            pred=[]
            with torch.no_grad():
                for i in range(0,len(x),16384):
                    t=torch.tensor(x[i:i+16384],device=device)
                    pred.append(torch.sigmoid(model((t-mean)/std)).reshape(-1).cpu().numpy())
            values.append(np.concatenate(pred));labels.append(df['improved_best'].to_numpy(dtype=np.float32))
            function_ids.extend((df['target'].astype(str)+':'+df['fid'].astype(str)).tolist())
        print(json.dumps({'file':path.name,'rows_read':read,'selected':sum(len(x) for x in labels)}),flush=True)
    prob=np.concatenate(values);y=np.concatenate(labels);fids=np.array(function_ids)
    bins=[]
    for i in range(10):
        mask=(prob>=i/10)&(prob<((i+1)/10) if i<9 else prob<=1)
        if mask.any():bins.append({'lower':i/10,'upper':(i+1)/10,'n':int(mask.sum()),'mean_predicted':float(prob[mask].mean()),'observed_positive_rate':float(y[mask].mean())})
    order=np.argsort(-prob);base=float(y.mean())
    result={'n_pool_rows_read':read,'n_holdout_rows':len(y),'positive_rate':base,
        'brier':float(np.mean((prob-y)**2)),
        'constant_empirical_rate_brier':float(np.mean((base-y)**2)),
        'ece_10_equal_width_bins':sum(b['n']*abs(b['mean_predicted']-b['observed_positive_rate']) for b in bins)/len(y),
        'precision_top1_percent':float(y[order[:max(1,len(y)//100)]].mean()),
        'mean_probability':float(prob.mean()),'pipeline_seconds':time.perf_counter()-start,'bins':bins,
        'per_function':[{ 'fid':fid,'n':int((fids==fid).sum()),'positive_rate':float(y[fids==fid].mean()),'brier':float(((prob[fids==fid]-y[fids==fid])**2).mean())} for fid in sorted(holdout)]}
    (args.output/'metrics.json').write_text(json.dumps(result,indent=2))
    (args.output/'COMPLETE').write_text('Frozen residual holdout inference audit complete; no training performed.\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('bins','per_function')}),flush=True)


if __name__=='__main__':main()
