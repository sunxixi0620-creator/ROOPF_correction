"""Run bounded v2 source diagnostic; all later stages stop on errors."""
import fcntl,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from roopf.experiment_io import write_json
RUN=ROOT/'results/neural_prior_scale_v2';PY=ROOT/'.venv/bin/python'


def main():
    RUN.mkdir(parents=True,exist_ok=True)
    with (RUN/'pipeline.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for phase,args in [('freeze',['scripts/train_neural_prior_scale_v2.py','freeze']),
                           ('training',['scripts/train_neural_prior_scale_v2.py','run']),
                           ('report',['scripts/report_neural_prior_scale_v2.py'])]:
            write_json(RUN/'PIPELINE_STATUS.json',dict(phase=phase,state='running',time=time.time()))
            with (RUN/(phase+'.log')).open('a') as log:
                p=subprocess.run([str(PY),'-u',*args],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            if p.returncode:
                write_json(RUN/'PIPELINE_STATUS.json',dict(phase=phase,state='failed',returncode=p.returncode,time=time.time()));return p.returncode
        write_json(RUN/'PIPELINE_STATUS.json',dict(phase='complete',state='complete',time=time.time(),target_calls=0,confirmation_calls=0))
    return 0


if __name__=='__main__':sys.exit(main())
