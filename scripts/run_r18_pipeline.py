"""Durable orchestration: checks, matched cases, live suite, audit, comparison, archive."""
import os,subprocess,sys,time,traceback
from pathlib import Path
import r18_common as c

def run(script,*args):
    print('START',script,*args,flush=True)
    subprocess.run([sys.executable,'-u','scripts/'+script,*args],cwd=c.ROOT,check=True)

def main():
    pid=int(sys.argv[1])
    while True:
        try:os.kill(pid,0)
        except ProcessLookupError:break
        time.sleep(10)
    assert c.read(c.EXP/'engineering_checks/RESULT.json')['status']=='engineering_passed'
    assert c.read(c.EXP/'REMOTE_IMPLEMENTATION_VERIFICATION.json')['status']=='remote_ref_verified'
    run('experiment_r18_schedule.py','run')
    run('experiment_r18_schedule.py','verify')
    run('audit_r18_parallel.py')
    run('summarize_r18_crossround.py')
    run('publish_r18_records.py','--final')
    c.write(c.EXP/'PIPELINE_STATUS.json',dict(status='complete_with_cross_round_summary',finished=time.strftime('%Y-%m-%dT%H:%M:%S%z')))
if __name__=='__main__':
    try:main()
    except BaseException:
        c.write(c.EXP/f'PIPELINE_FAILURE_{time.time_ns()}.json',dict(status='incomplete',traceback=traceback.format_exc()))
        raise
