"""Publish only explicit R17 source and scientific records using an isolated Git index."""
from pathlib import Path
import argparse,gzip,hashlib,json,os,subprocess,tarfile,tempfile
import r17_common as common
BRANCH='experiment/r17-fe200-seven-schedules-20261028'
SSH='ssh -i /home/skq/.ssh/id_ed25519_roopf_correction -o IdentitiesOnly=yes -o ConnectTimeout=15 -o UserKnownHostsFile=/tmp/r14_github_known_hosts -o HostKeyAlias=github.com -o StrictHostKeyChecking=yes -o "ProxyCommand=nc -X connect -x 127.0.0.1:7890 %h %p" -p 443'
REMOTE='ssh://git@ssh.github.com:443/sunxixi0620-creator/ROOPF_correction.git'

def git(*args,env=None):return subprocess.check_output(['git',*args],cwd=common.ROOT,env=env,text=True).strip()
def publish(final=False):
    common.validate_frozen();exp=common.EXP
    paths=[common.ROOT/p for p in common.SOURCES]
    paths += [p for p in exp.iterdir() if p.is_file() and p.suffix in ('.json','.csv','.txt','.xlsx') and p.name not in ('EXECUTION_STATUS.json','PROGRESS.json')]
    if final:
        result=common.read(exp/'FINAL_RESULT.json')
        # Keep unsuccessful experiments as reported; never relabel completion or suppress failures.
        archive=exp/'scientific_records';archive.mkdir(exist_ok=True)
        for cid in common.CONFIGS:
            with tarfile.open(archive/(cid+'_matched_cases_and_api.tar.gz'),'w:gz',compresslevel=6) as t:
                case=exp/'cases'/cid
                for name in ('offline_memory.json','HELDOUT_CASES.json.gz','COLLECTION_BINDING.json','COLLECTION_SUMMARY.json','NATIVE_BUDGET.json'):
                    t.add(case/name,arcname='cases/'+cid+'/'+name,recursive=False)
                for p in sorted((exp/'api_journal').glob('*.json')):
                    if common.read(p)['execution_context']['arm']==cid:t.add(p,arcname='api_journal/'+p.name,recursive=False)
            paths.append(archive/(cid+'_matched_cases_and_api.tar.gz'))
    # Exact secret check; credential contents are never printed or staged.
    keyfile=Path.home()/'.config/roopf_agent/worldcodes.env'
    secrets=[]
    if keyfile.exists():
        for line in keyfile.read_text().splitlines():
            if '=' in line and 'KEY' in line.split('=',1)[0]:
                value=line.split('=',1)[1].strip().strip('\"\'')
                if len(value)>12:secrets.append(value.encode())
    paths=list(dict.fromkeys(paths))
    for p in paths:
        assert p.is_file() and p.stat().st_size<95*1024**2,p
        raw=p.read_bytes()
        assert not any(secret in raw for secret in secrets),'Credential detected in publish list'
    # Also check uncompressed journal contents before adding archives.
    if final:
        for p in (exp/'api_journal').glob('*.json'):
            assert not any(secret in p.read_bytes() for secret in secrets),'Credential detected in journal'
    parent=git('rev-parse','refs/heads/'+BRANCH)
    with tempfile.TemporaryDirectory(prefix='r15_git_') as temp:
        env=dict(os.environ,GIT_INDEX_FILE=str(Path(temp)/'index'))
        git('read-tree',parent,env=env)
        git('add','-f','--',*[str(p.relative_to(common.ROOT)) for p in paths],env=env)
        tree=git('write-tree',env=env)
        if tree==git('rev-parse',parent+'^{tree}'):commit=parent
        else:
            message='R17: archive full matched LLM schedule factorial results' if final else 'R17: implement isolated matched schedule factorial pipeline'
            commit=git('commit-tree',tree,'-p',parent,'-m',message,env=env)
            git('update-ref','refs/heads/'+BRANCH,commit,parent)
    git('-c','core.sshCommand='+SSH,'push',REMOTE,'refs/heads/'+BRANCH+':refs/heads/'+BRANCH)
    remote=git('-c','core.sshCommand='+SSH,'ls-remote',REMOTE,'refs/heads/'+BRANCH).split()[0]
    assert remote==commit
    common.write(exp/('REMOTE_RESULT_VERIFICATION.json' if final else 'REMOTE_IMPLEMENTATION_VERIFICATION.json'),
        dict(status='remote_ref_verified',commit=commit,branch=BRANCH,paths=[str(p.relative_to(common.ROOT)) for p in paths],final=final,
             final_release_unchanged=git('rev-parse','HEAD')==common.PROTOCOL['parent_commit']))
    print(json.dumps(dict(event='remote_records_published',commit=commit,final=final)),flush=True)
    return commit

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--final',action='store_true');args=parser.parse_args();publish(args.final)
