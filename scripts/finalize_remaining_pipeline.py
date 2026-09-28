"""Monitored session tail: serial timing, summaries and complete archives."""
import os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
while not (ROOT/'results/remaining_training_pipeline_COMPLETE').exists():
    time.sleep(15)
env=os.environ.copy();env.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='')
commands=[['scripts/serial_external_timing.py'],['scripts/summarize_remaining.py'],['scripts/plot_remaining.py'],['scripts/write_remaining_report.py'],['scripts/archive_remaining.py','--require-all']]
for args in commands:
    print('START',args,flush=True)
    subprocess.run([str(ROOT/'.venv/bin/python'),'-u',*args],cwd=ROOT,env=env,check=True)
    print('DONE',args,flush=True)
(ROOT/'results/remaining_finalization_COMPLETE').write_text('Serial timing, summaries, final report and complete archives finished.\n')
