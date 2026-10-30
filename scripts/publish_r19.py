"""Publish R19 code and scientific records using a private index and secret scan."""
from pathlib import Path
import sys,json,os,tempfile,tarfile
sys.path[:0]=[str(Path(__file__).resolve().parents[1]/'scripts')]
import publish_r18_records as transport
from experiment_r19_ablation import E,ROOT,SOURCES
BRANCH='experiment/r19-s01l08-ablations-20261030';BASE='baseline/r19-s01l08-ablations-20261030'
git=transport.git

def publish(final=False):
 paths=[ROOT/x for x in SOURCES]+[ROOT/'scripts/publish_r19.py']+[p for p in E.iterdir() if p.is_file() and p.suffix in ('.json','.csv','.xlsx','.log') and p.name not in ('EXECUTION_STATUS.json','PROGRESS.json')]
 if final:
  archive=E/'scientific_records';archive.mkdir(exist_ok=True)
  for group in ['BBOB','CEC-style','UAV']:
   for arm in ['llm_full','llm_no_anchor','llm_no_online_generators','no_agent','REFERENCES']:
    target=archive/f'{group}_{arm}.tar.gz'
    with tarfile.open(target,'w:gz',compresslevel=6) as tar:
     for p in E.glob(f'results/{group}_f*/b??/{arm}*'):
      if p.is_file():tar.add(p,arcname=str(p.relative_to(E)))
    paths.append(target)
  for arm in ['llm_full','llm_no_anchor','llm_no_online_generators']:
   target=archive/f'{arm}_api_journal.tar.gz'
   with tarfile.open(target,'w:gz',compresslevel=6) as tar:
    for p in (E/'api_journal').glob('*.json'):
     if json.loads(p.read_text())['execution_context']['arm']==arm:tar.add(p,arcname='api_journal/'+p.name)
   paths.append(target)
 credentials=Path.home()/'.config/roopf_agent/worldcodes.env';secrets=[]
 if credentials.exists():
  for line in credentials.read_text().splitlines():
   if '=' in line and 'KEY' in line.split('=',1)[0]:
    val=line.split('=',1)[1].strip().strip('\"\'')
    if len(val)>12:secrets.append(val.encode())
 paths=list(dict.fromkeys(paths))
 for p in paths:
  assert p.is_file() and p.stat().st_size<95*1024**2,p
  assert not any(s in p.read_bytes() for s in secrets),'Credential detected'
 if final:
  for p in (E/'api_journal').glob('*.json'):assert not any(s in p.read_bytes() for s in secrets)
 parent=git('rev-parse','refs/heads/'+BRANCH)
 with tempfile.TemporaryDirectory(prefix='r19_git_') as t:
  env=dict(os.environ,GIT_INDEX_FILE=str(Path(t)/'index'));git('read-tree',parent,env=env);git('add','-f','--',*[str(p.relative_to(ROOT)) for p in paths],env=env)
  tree=git('write-tree',env=env);commit=git('commit-tree',tree,'-p',parent,'-m','R19: '+('archive verified ablation results' if final else 'implement selected schedule component ablations'),env=env)
  git('update-ref','refs/heads/'+BRANCH,commit,parent)
 git('-c','core.sshCommand='+transport.SSH,'push',transport.REMOTE,'refs/heads/'+BRANCH+':refs/heads/'+BRANCH)
 assert git('-c','core.sshCommand='+transport.SSH,'ls-remote',transport.REMOTE,'refs/heads/'+BRANCH).split()[0]==commit
 (E/('REMOTE_RESULT_VERIFICATION.json' if final else 'REMOTE_IMPLEMENTATION_VERIFICATION.json')).write_text(json.dumps(dict(status='remote_ref_verified',commit=commit,branch=BRANCH,final=final),indent=2)+'\n')
 print(json.dumps(dict(event='published',commit=commit,final=final)),flush=True)
if __name__=='__main__':publish('--final' in sys.argv)
