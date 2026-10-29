"""R10 final-content action contract, partial row validation and bounded live calls.

No retries or reasoning extraction. Historical API code remains frozen.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import time

from roopf.policy_api import DEFAULT_API as OLD_API, atomic_json, request_slot, send_request, wire_bytes

DEFAULT_API = dict(OLD_API, max_tokens=1024, max_request_bytes=96000)


def validate_response(response, payload):
    choice = response['choices'][0]
    if choice.get('finish_reason') != 'stop':
        raise ValueError('output_truncated_or_incomplete')
    content = choice['message'].get('content')
    if not isinstance(content, str):
        raise ValueError('missing_final_content')
    parsed = json.loads(content)
    if not isinstance(parsed, dict) or set(parsed) != {'actions'} or not isinstance(parsed['actions'], dict):
        raise ValueError('invalid_action_schema')
    expected = {row['trajectory_id']: row for row in payload['trajectories']}
    # Duplicate JSON keys are rejected before per-row validation; no last-key wins.
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('duplicate_json_key')
            result[key] = value
        return result
    parsed = json.loads(content, object_pairs_hook=unique_object)
    if set(parsed['actions']) - set(expected):
        raise ValueError('unknown_trajectory')
    ids, valid, errors = [], [], {}
    for identity, row in expected.items():
        value = parsed['actions'].get(identity)
        if not isinstance(value, str) or value not in payload['action_names']:
            errors[identity] = 'missing_or_unknown_action'
        elif value not in row['allowed']:
            errors[identity] = 'unsupported_action'
        else:
            valid.append(identity)
        ids.append(value if identity in valid else 'matched38')
    return dict(config_ids=ids, valid_trajectory_ids=valid, row_errors=errors,
                status='ok' if not errors else ('partial' if valid else 'failed'))


def request_policy(agent, payload):
    config = agent.api_config
    request = dict(model=config['model'], temperature=0, max_tokens=config['max_tokens'],
                   response_format={'type': 'json_object'}, thinking={'type': 'disabled'},
                   enable_thinking=False,
                   messages=[dict(role='system', content=agent.system_prompt),
                             dict(role='user', content=agent.system_prompt + '\nReturn exactly a JSON object with only the key actions; map every trajectory_id to a permitted action string. No other keys or prose.\nINPUT:\n' + json.dumps(payload, separators=(',', ':'), allow_nan=False))])
    data = wire_bytes(request)
    identity = hashlib.sha256(data).hexdigest()
    agent.api_identity = identity
    target = agent.journal / (identity + '.json')
    if len(data) > config['max_request_bytes']:
        raise ValueError('Request exceeds frozen size limit')
    with (agent.journal / 'API.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if target.exists():
            prior = json.loads(target.read_text())
            if prior['request'] != request:
                raise RuntimeError('Request identity collision')
            if prior['status'] == 'pending':
                agent.uncertain_api_outcome = True
                raise RuntimeError('Unknown prior outcome; no automatic retry')
            agent.stats['api_reused'] += 1
            if prior['status'] in ('ok', 'partial'):
                return validate_response(prior['response'], payload)
            return dict(config_ids=['matched38'] * len(payload['trajectories']), valid_trajectory_ids=[],
                        row_errors={r['trajectory_id']: 'recorded_failure' for r in payload['trajectories']}, status='failed')
        if len(list(agent.journal.glob('*.json'))) >= agent.budget['limit']:
            agent.resource_exhausted = True
            raise RuntimeError('API cap reached')
        record = dict(status='pending', request_sha256=identity, request=request,
                      memory_sha256=agent.memory_hash, decision_context_sha256=payload['decision_context_sha256'],
                      execution_context=agent.execution_context, transport=config['transport'],
                      request_bytes=len(data), automatic_retry=False)
        atomic_json(target, record)
        agent.stats['requests'] += 1
    started = time.monotonic()
    try:
        with request_slot(agent.journal, config['concurrency']):
            record['queue_seconds'] = time.monotonic() - started
            response = send_request(data, config, agent.journal / (identity + '.request.payload'))
        record['response'] = response
        decoded = validate_response(response, payload)
        record.update(decoded)
        return decoded
    except Exception as error:
        record.update(status='failed', error_type=type(error).__name__, error=str(error)[:300],
                      config_ids=['matched38'] * len(payload['trajectories']), valid_trajectory_ids=[],
                      row_errors={r['trajectory_id']: 'request_or_contract_failure' for r in payload['trajectories']})
        return {key: record[key] for key in ('status', 'config_ids', 'valid_trajectory_ids', 'row_errors')}
    finally:
        record['elapsed_seconds'] = time.monotonic() - started
        atomic_json(target, record)
