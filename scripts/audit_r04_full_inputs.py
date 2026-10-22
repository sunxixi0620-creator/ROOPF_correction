import csv
import gzip
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / 'experiments/r04_full_llm_benchmark'


def audit():
    with (EXP / 'API_REQUESTS.csv').open() as stream:
        rows = [row for row in csv.DictReader(stream) if row['scope'] == 'formal']
    batches, verified, appended = {}, 0, 0
    for row in rows:
        record = json.loads((EXP / 'api_journal' / f"{row['request_sha256']}.json").read_text())
        content = record['request']['messages'][-1]['content']
        payload = json.loads(content.split('INPUT:\n', 1)[1].split('\nFINAL OUTPUT CONTRACT:', 1)[0])
        for location in row['executed_locations'].split(';'):
            task, batch, step = location.split(':')
            key = task, batch
            if key not in batches:
                with gzip.open(EXP / 'results' / task / batch / 'llm_live_DECISIONS.json.gz', 'rt') as stream:
                    batches[key] = json.load(stream)
            decision = batches[key][int(step[1:])]
            for trajectory in payload['trajectories']:
                feedback = decision['feedback'][int(trajectory['trajectory_id'])]
                expected = trajectory['observed_feedback']
                if feedback == expected:
                    verified += 1
                    continue
                recent = feedback['competing_slot_recent']
                causal = dict(feedback, competing_slot_recent=recent[:-1])
                if len(recent) != min(decision['step'], 32) + 1 or causal != expected:
                    raise RuntimeError('Actual API inputs cannot be reconciled to pre-decision feedback')
                appended += 1
                verified += 1
    result = dict(status='actual_api_predecision_feedback_verified', actual_api_queries=len(rows),
        trajectory_inputs_verified=verified, actual_api_inputs_contain_current_unobserved_outcome=False,
        raw_log_snapshots_with_one_later_history_append=appended,
        raw_log_history_correction='For the archived raw step logs, remove the final competing_slot_recent entry before treating feedback as pre-decision features; actual API request journals need no correction.',
        cause='Logger retained a history-list reference; observe appends one row after the API request was already serialized.',
        numerical_evaluation_results_affected=False,
        prospective_training_warning='Use actual request journals or the corrected history prefix, never raw recent-history snapshots as pre-action training features.')
    (EXP / 'API_INPUT_CAUSALITY.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    return result


if __name__ == '__main__':
    print(json.dumps(audit()))
