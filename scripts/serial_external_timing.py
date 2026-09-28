"""Serial external-baseline latency at identical objective/budget boundaries."""
import json,hashlib
from pathlib import Path
from independent_benchmarks import run as classical
from remaining_gp_ei import run as gp
from remaining_surr_rlde import run as learned
from supplementary_experiments import ROOT,write_csv

def main():
    assert (ROOT/'results/remaining_training_pipeline_COMPLETE').exists(), 'Finish our training/evaluation jobs before serial timing'
    out=ROOT/'results/serial_external_timing_20260928';out.mkdir(exist_ok=True)
    cases=[('coco',9,101),('coco',11,101),('cec2017',10,1),('cec2017',20,1)]
    methods=['full','anchor_only','cma_es','de','gp_ei','surr_rlde']
    protocol={'cases':cases,'methods':methods,'seeds':list(range(20275000,20275003)),'nfe':300,
        'execution':'single CPU process,1 BLAS/OMP/MKL thread; run after our remaining training/evaluation jobs finish',
        'boundary':'online search/evaluation; checkpoint loading and objective construction excluded; GP fitting included',
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2));rows=[]
    for suite,fid,inst in cases:
        for seed in protocol['seeds']:
            for m in methods:
                if m=='gp_ei':row,_=gp(suite,fid,inst,seed)
                elif m=='surr_rlde':row,_=learned(suite,fid,inst,seed)
                else:row,_=classical(suite,fid,inst,m,seed)
                rows.append({'suite':suite,'fid':fid,'instance':inst,'seed':seed,'method':m,'seconds':row['seconds'],'actual_nfe':row['actual_nfe'],'final':row['final']})
            print(suite,fid,seed,flush=True)
    write_csv(out/'raw_results.csv',rows);(out/'COMPLETE').write_text(str(len(rows)))

if __name__=='__main__':main()
