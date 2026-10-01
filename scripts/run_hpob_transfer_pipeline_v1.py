"""Dependency-checked unattended pipeline; errors stop later stages."""
import fcntl
import json
import subprocess
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'results/hpob_transfer_development_v1'
PY=ROOT/'.venv/bin/python'


def main():
    RUN.mkdir(exist_ok=True)
    with (RUN/'pipeline.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        stages=[('freeze',['scripts/hpob_transfer_development_v1.py','freeze']),
                ('sources',['scripts/hpob_transfer_development_v1.py','sources','--workers','16']),
                ('caches',['scripts/hpob_transfer_development_v1.py','caches','--workers','2','--device','cuda']),
                ('cache_audit',['scripts/check_hpob_transfer_cache_v1.py']),
                ('run',['scripts/hpob_transfer_development_v1.py','run','--workers','16']),
                ('report',['scripts/report_hpob_transfer_development_v1.py'])]
        for name,args in stages:
            print('START',name,time.strftime('%Y-%m-%d %H:%M:%S'),flush=True)
            # Keep prior logs on an explicit resume. Source/case manifests provide
            # identity-checked reuse; journals make paid observations durable.
            with (RUN/(name+'.log')).open('a') as log:
                result=subprocess.run([str(PY),'-u',*args],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            if result.returncode:
                print('STOP',name,'exit',result.returncode,flush=True)
                return result.returncode
            print('DONE',name,time.strftime('%Y-%m-%d %H:%M:%S'),flush=True)
    return 0


if __name__=='__main__':sys.exit(main())
