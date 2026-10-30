"""Fail closed on a severe live API outage; preserve unknown pending outcomes."""
import json,os,signal,time
from pathlib import Path
E=Path('experiments/r19_s01l08_component_ablations')
while True:
 try:p=json.loads((E/'PIPELINE_PID.json').read_text())['pid'];os.kill(p,0)
 except (OSError,FileNotFoundError):break
 counts={'ok':0,'partial':0,'failed':0,'pending':0}
 for f in (E/'api_journal').glob('*.json'):
  try:counts[json.loads(f.read_text())['status']]+=1
  except (FileNotFoundError,json.JSONDecodeError):pass
 done=counts['ok']+counts['partial']+counts['failed']
 if done>=100 and (counts['ok']+counts['partial'])/done<.2:
  assert os.getpgid(p)==p;os.killpg(p,signal.SIGTERM)
  value=dict(status='aborted_api_contract_outage',counts=counts,time=time.strftime('%Y-%m-%dT%H:%M:%S%z'),reason='Less than 20% of at least 100 completed requests produced valid rows; no retries or parser changes.')
  (E/'ABORTED_API_OUTAGE.json').write_text(json.dumps(value,indent=2)+'\n');(E/'EXECUTION_STATUS.json').write_text(json.dumps(value,indent=2)+'\n');break
 time.sleep(15)
