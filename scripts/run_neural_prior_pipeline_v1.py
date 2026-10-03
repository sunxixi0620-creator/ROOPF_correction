"""Source contexts, paired training, and source diagnostics; never target BO."""
import fcntl,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from roopf.experiment_io import write_json,sha256
RUN=ROOT/'results/neural_prior_training_v1';PY=ROOT/'.venv/bin/python'


def main():
    RUN.mkdir(parents=True,exist_ok=True)
    with (RUN/'pipeline.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        stages=[('freeze',['scripts/train_neural_prior_v1.py','freeze']),
                ('contexts',['scripts/train_neural_prior_v1.py','contexts','--workers','16']),
                ('training',['scripts/train_neural_prior_v1.py','training','--workers','3','--device','cuda']),
                ('report',['scripts/report_neural_prior_training_v1.py'])]
        hashes={p:sha256(ROOT/p) for p in ('scripts/run_neural_prior_pipeline_v1.py','scripts/report_neural_prior_training_v1.py')}
        for phase,args in stages:
            print('START',phase,time.strftime('%Y-%m-%d %H:%M:%S'),flush=True)
            write_json(RUN/'PIPELINE_STATUS.json',dict(phase=phase,state='running',time=time.time(),sources=hashes))
            with (RUN/(phase+'.log')).open('a') as log:
                p=subprocess.run([str(PY),'-u',*args],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            if p.returncode:
                write_json(RUN/'PIPELINE_STATUS.json',dict(phase=phase,state='failed',returncode=p.returncode,time=time.time(),sources=hashes))
                print('STOP',phase,'exit',p.returncode,flush=True);return p.returncode
            print('DONE',phase,time.strftime('%Y-%m-%d %H:%M:%S'),flush=True)
        write_json(RUN/'PIPELINE_STATUS.json',dict(phase='complete',state='complete',time=time.time(),sources=hashes,
            target_calls=0,confirmation_calls=0))
    return 0


if __name__=='__main__':sys.exit(main())
