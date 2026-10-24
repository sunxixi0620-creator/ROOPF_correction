"""Bounded, journaled policy requests. No retries, credential logging or RNG use."""
from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import time
import urllib.error
import urllib.request

DEFAULT_API = dict(model='deepseek-v4-flash', endpoint='https://worldcodes.online/v1/chat/completions',
                   transport='proxy_http', proxy='http://127.0.0.1:7890', concurrency=2,
                   timeout_seconds=90, max_tokens=8192, max_request_bytes=48000)


def wire_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def atomic_json(path, value):
    temporary = path.with_suffix('.writing')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
    temporary.replace(path)


@contextlib.contextmanager
def request_slot(journal, concurrency):
    while True:
        for index in range(concurrency):
            lock = (journal / f'API.slot{index}.lock').open('a')
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                lock.close()
                continue
            try:
                yield
            finally:
                lock.close()
            return
        time.sleep(0.05)


def send_request(data, config, request_path):
    if config['transport'] == 'windows':
        # The old script has a fixed 35s timeout, so R09 owns its longer-lived bridge.
        script = Path(__file__).resolve().parents[1] / 'scripts/worldcodes_request_r09.ps1'
        request_path.write_bytes(data)
        windows = [subprocess.check_output(['wslpath', '-w', str(p)], text=True).strip()
                   for p in (script, request_path)]
        result = subprocess.run(['/mnt/c/WINDOWS/System32/WindowsPowerShell/v1.0/powershell.exe',
            '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', windows[0], '-RequestFile', windows[1],
            '-TimeoutSeconds', str(config['timeout_seconds'])], capture_output=True, encoding='utf-8',
            timeout=config['timeout_seconds'] + 10)
        if result.returncode:
            raise RuntimeError(result.stderr.strip()[:300])
        return json.loads(Path(str(request_path) + '.response.raw').read_text(encoding='utf-8-sig'))
    if config['transport'] != 'proxy_http':
        raise ValueError('Unknown API transport')
    credential = (Path.home() / '.config/roopf_agent/worldcodes.env').read_text().strip()
    prefix = 'WORLDCODES_API_KEY='
    if not credential.startswith(prefix) or not credential[len(prefix):]:
        raise ValueError('Invalid credential file format')
    key = credential[len(prefix):]
    opener = urllib.request.build_opener(urllib.request.ProxyHandler(
        {'http': config['proxy'], 'https': config['proxy']}))
    request = urllib.request.Request(config['endpoint'], data=data, method='POST', headers={
        'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json', 'User-Agent': 'ROOPF-R09'})
    try:
        with opener.open(request, timeout=config['timeout_seconds']) as response:
            return json.loads(response.read().decode('utf-8-sig'))
    except urllib.error.HTTPError as error:
        # Preserve status, but never copy response/request headers or credentials.
        raise RuntimeError(f'http_{error.code}') from None
    except urllib.error.URLError as error:
        raise RuntimeError('connection_or_timeout: ' + str(error.reason).replace(key, '<redacted>')[:180]) from None


def request_policy(agent, payload, decode):
    config = agent.api_config
    request = dict(model=config['model'], temperature=0, max_tokens=config['max_tokens'],
                   response_format={'type': 'json_object'}, thinking={'type': 'disabled'},
                   messages=[dict(role='system', content=agent.system_prompt),
                             dict(role='user', content='INPUT:\n' + json.dumps(payload, separators=(',', ':'), allow_nan=False))])
    data = wire_bytes(request)
    identity = hashlib.sha256(data).hexdigest()
    agent.api_identity = identity
    target = agent.journal / (identity + '.json')
    if len(data) > config['max_request_bytes']:
        raise ValueError('Compact request exceeds frozen request-size limit')
    with (agent.journal / 'API.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if target.exists():
            previous = json.loads(target.read_text())
            if previous['request'] != request:
                raise RuntimeError('Request identity collision')
            if previous['status'] != 'ok':
                if previous['status'] == 'pending':
                    agent.uncertain_api_outcome = True
                raise RuntimeError('Recorded failed/unknown request; no automatic paid retry')
            agent.stats['api_reused'] += 1
            return decode(previous['response'], payload)
        if len(list(agent.journal.glob('*.json'))) >= agent.budget['limit']:
            agent.resource_exhausted = True
            raise RuntimeError('本轮API次数上限')
        record = dict(status='pending', request_sha256=identity, request=request,
                      memory_sha256=agent.memory_hash, decision_context_sha256=payload['decision_context_sha256'],
                      execution_context=agent.execution_context,
                      transport=config['transport'], request_bytes=len(data), automatic_retry=False)
        atomic_json(target, record)
        agent.stats['requests'] += 1
    started = time.monotonic()
    try:
        with request_slot(agent.journal, config['concurrency']):
            record['queue_seconds'] = time.monotonic() - started
            response = send_request(data, config, agent.journal / (identity + '.request.payload'))
        record['response'] = response
        ids = decode(response, payload)
        record.update(status='ok', config_ids=ids)
        return ids
    except Exception as error:
        record.update(status='failed', error_type=type(error).__name__, error=str(error)[:300])
        raise
    finally:
        record['elapsed_seconds'] = time.monotonic() - started
        atomic_json(target, record)
