"""Execute R19, independent validation and archival sequentially."""
import subprocess,sys,json,time,traceback
from pathlib import Path
E=Path('experiments/r19_s01l08_component_ablations')
try:
 for script,args in [('experiment_r19_ablation.py',[]),('audit_r19_independent.py',[]),('publish_r19.py',['--final'])]:
  print('START',script,flush=True);subprocess.run([sys.executable,'-u','scripts/'+script,*args],check=True)
 (E/'PIPELINE_STATUS.json').write_text(json.dumps(dict(status='complete_with_independent_audit_and_remote_archive',finished=time.strftime('%Y-%m-%dT%H:%M:%S%z')))+'\n')
except BaseException:
 (E/f'PIPELINE_FAILURE_{time.time_ns()}.json').write_text(json.dumps(dict(status='partial_or_failed',traceback=traceback.format_exc()),indent=2)+'\n');raise
