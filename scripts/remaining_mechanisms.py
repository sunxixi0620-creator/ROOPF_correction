"""Predeclared fixed-width operator and single-intervention replay diagnostics."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import json
import hashlib
from pathlib import Path
import numpy as np
import torch
from scipy.stats import spearmanr
from supplementary_experiments import build, problem_for, write_csv, ROOT
from roopf.supplement import SupplementOptimizer
import run_roopf as r

CASES = ['bbob_f9_fixed','bbob_f11_fixed','bbob_f15_fixed',
         'cec_f1_shifted','cec_f3_shifted','cec_f6_shifted']

class MechanismOptimizer(SupplementOptimizer):
    def _candidate_pool(self, *args, **kwargs):
        pool, ops, prior = super()._candidate_pool(*args, **kwargs)
        mode = self.pool_mode
        if mode == 'full':
            return pool, ops, prior
        chosen = int(mode.split('_')[-1])
        # Always consume the full method's original RNG calls first. Additional
        # draws have an isolated stream; surviving original candidates stay fixed.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.replay_seed + 1000*self.evalnum)
            if mode.startswith('drop'):
                keep = torch.where(ops != chosen)[0]
                extra, _, extra_prior = super()._candidate_pool(*args, **kwargs)
                donors = [i for i in range(6) if i != chosen]
                idx = torch.tensor([donors[j % 5]*6 + j//5 for j in range(6)])
                pool = torch.cat([pool[:,keep], extra[:,idx]],1)
                prior = torch.cat([prior[:,keep], extra_prior[:,idx]],1)
                ops = torch.cat([ops[keep], ops[idx]])
            else:
                parts = [pool[:,chosen*6:(chosen+1)*6]]
                for _ in range(5):
                    extra, _, _ = super()._candidate_pool(*args, **kwargs)
                    parts.append(extra[:,chosen*6:(chosen+1)*6])
                pool = torch.cat(parts,1)
                ops = torch.full((36,),chosen,dtype=torch.long)
                prior = torch.ones(pool.shape[:2],dtype=pool.dtype)
        assert pool.shape[1] == 36
        return pool, ops, prior/prior.sum(1,keepdim=True)

    def observe_gate(self, s):
        super().observe_gate(s)
        if self.evalnum in (220,260) and self.collect_pool:
            args,_ = self.pool_context
            pool=args[2]
            truth=s['problem'].diagnostic_fitness(pool)[0].cpu().numpy()
            score, mu, sigma = self.pool_prediction
            denom=max(float(np.std(truth)),1e-8)
            self.pool_rows.append({'eval_before':self.evalnum,
                'spearman_mu':float(spearmanr(mu,truth).statistic),
                'spearman_score':float(spearmanr(score,truth).statistic),
                'rmse_over_pool_std':float(np.sqrt(np.mean((mu-truth)**2))/denom),
                'chosen_truth':float(truth[np.argmin(score)]),
                'random_expected_truth':float(np.mean(truth)),
                'pool_best_truth':float(np.min(truth))})
        if self.evalnum == self.force_at:
            before=int(s['extra_idx'][0,0])
            after=0 if self.force_kind=='anchor' else int(s['proposed_extra_idx'][0,0])
            self.intervention={'eval_before':self.evalnum,'before_index':before,
                'after_index':after,'changed':before!=after}
            s['extra_idx'].fill_(after)

    def _select_by_acquisition(self,*args,**kwargs):
        result=super()._select_by_acquisition(*args,**kwargs)
        if args[2].shape[1]==36:
            self.pool_prediction=tuple(v[0].detach().cpu().numpy().copy() for v in
                (self.last_acquisition,self.last_surrogate_mu,self.last_surrogate_sigma))
        return result

def run(case, seed, mode, force_at=-1, force_kind='anchor', collect=False):
    p,_=problem_for(case)
    opt=build();opt.__class__=MechanismOptimizer
    opt.supplement_gate_observer=opt.observe_gate
    opt.pool_mode=mode;opt.replay_seed=seed;opt.force_at=force_at
    opt.force_kind=force_kind;opt.collect_pool=collect;opt.pool_rows=[];opt.intervention=None
    pop=p.fun['xlb']+(p.fun['xub']-p.fun['xlb'])*torch.rand((1,100,10),generator=torch.Generator().manual_seed(seed))
    r.set_seed(seed+770000)
    with torch.no_grad(): _,trail,nfe,points=opt(pop,p)
    assert nfe==p.actual==300 and torch.isfinite(trail).all()
    row={'case':case,'seed':seed,'mode':mode,'force_at':force_at,'force_kind':force_kind,
        'final':float(trail[0,-1]),'actual_nfe':p.actual,'diagnostic_nfe':p.diagnostic_actual,
        'intervention':opt.intervention,'origin_stats':opt.origin_stats,'pool_rows':opt.pool_rows}
    return row,points.cpu().numpy(),trail.cpu().numpy()

def worker(job):
    case,seed,stage,out=job;out=Path(out);rows=[]
    modes=['full']+[f'drop_{i}' for i in range(6)]+[f'only_{i}' for i in range(6)] if stage=='operators' else ['full']
    settings=[(m,-1,'anchor') for m in modes]
    if stage=='replay': settings += [('full',n,k) for n in (210,240,270) for k in ('anchor','proposal')]
    baseline=None
    for mode,at,kind in settings:
        stem=f'{case}_s{seed}_{mode}_{at}_{kind}';target=out/(stem+'.json')
        if target.exists():
            row=json.loads(target.read_text());points=np.load(out/(stem+'.npz'))['points']
        else:
            row,points,trail=run(case,seed,mode,at,kind,stage=='replay' and at<0)
            if at>=0:
                assert np.array_equal(points[:,:at-100],baseline[:,:at-100]), 'Branch prefix diverged'
                row['prefix_exact']=True
            np.savez_compressed(out/(stem+'.npz'),points=points,trail=trail)
            target.write_text(json.dumps(row,indent=2))
        if at<0 and mode=='full': baseline=points
        rows.append(row)
    return rows

def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['operators','replay'],required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=8)
    p.add_argument('--seeds',type=int,default=30);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    protocol={'stage':a.stage,'cases':CASES,'seeds':list(range(20269000,20269000+a.seeds)),
      'width':36,'nfe':300,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    path=a.output/'protocol.json'
    if path.exists():assert json.loads(path.read_text())==protocol
    else:path.write_text(json.dumps(protocol,indent=2))
    jobs=[(c,s,a.stage,str(a.output)) for c in CASES for s in protocol['seeds']]
    rows=[]
    with ProcessPoolExecutor(a.workers,mp_context=multiprocessing.get_context('spawn')) as ex:
        futures=[ex.submit(worker,j) for j in jobs]
        for i,f in enumerate(as_completed(futures)):
            rows.extend(f.result())
            if (i+1)%10==0:print(a.stage,i+1,'/',len(jobs),flush=True)
    (a.output/'all_rows.json').write_text(json.dumps(rows,indent=2));(a.output/'COMPLETE').write_text(str(len(rows)))

if __name__=='__main__':main()
