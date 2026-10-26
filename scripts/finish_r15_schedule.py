"""Finish reporting after durable R15 execution; never change solver, cases, or API decisions.

CSV tables use the union of observed status columns, so an absent partial/error
category in the first function cannot truncate other functions' evidence.
This is a separately bound reporting supplement to the frozen execution code.
"""
from pathlib import Path
import argparse,collections,csv,json,os,time,traceback
from decimal import Decimal
import r15_common as common

def rectangular_csv(path,rows):
    if not rows:return
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)

def finish():
    import report_r15_schedule as reporting
    reporting.csvwrite=rectangular_csv
    # Numerical summaries, case audit, paired points and API mapping use the frozen verifier unchanged.
    result=reporting.report(common.EXP,verify=True)
    assert result['completed_arm_batches']==8400,'Full execution is incomplete; never replace with a pilot'
    accounting={}
    for p in (common.EXP/'api_journal').glob('*.json'):
        r=common.read(p);ctx=r['execution_context'];key=(ctx['arm'],ctx['group'],ctx['fid'])
        a=accounting.setdefault(key,dict(attempts=0,ok=0,partial=0,failed=0,pending=0,valid_rows=0,fallback_rows=0,prompt_tokens=0,completion_tokens=0,credit=Decimal(0),responses_with_credit=0))
        a['attempts']+=1;a[r['status']]+=1;a['valid_rows']+=len(r.get('valid_trajectory_ids',[]));a['fallback_rows']+=len(r.get('row_errors',{}))
        usage=r.get('response',{}).get('usage',{})
        a['prompt_tokens']+=usage.get('prompt_tokens',0);a['completion_tokens']+=usage.get('completion_tokens',0)
        if isinstance(usage.get('credit'),(int,float)):
            a['credit']+=Decimal(str(usage['credit']));a['responses_with_credit']+=1
    rows=[]
    for (cid,g,f),a in sorted(accounting.items()):
        a=dict(a);a['provider_credit_units']=str(a.pop('credit'))
        rows.append(dict(config_id=cid,group=g,fid=f,**a,currency_inferred=False))
    rectangular_csv(common.EXP/'API_BY_CONFIGURATION_AND_FUNCTION.csv',rows)
    common.write(common.EXP/'REPORTING_SUPPLEMENT_BINDING.json',dict(status='verified',
        source='scripts/finish_r15_schedule.py',sha256=common.sha(__file__),execution_binding=common.read(common.EXP/'CODE_BINDING.json'),
        changes='Rectangular CSV schema uses every observed status column; add per-function provider credit accounting. No changes to observations, solver, support rules, API requests, or numerical analysis.',
        execution_failures_preserved=True))
    from publish_r15_records import publish
    publish(final=True)
    common.write(common.EXP/'EXECUTION_STATUS.json',dict(status=result['status'],completed_measurement_batches=8400,
        finished=time.strftime('%Y-%m-%dT%H:%M:%S%z'),reporting_supplement_verified=True))
    print(json.dumps(dict(status=result['status'],completed_trajectories=84000,actual_api_records=result['matched_actual_api_records'])),flush=True)
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--watch-pid',type=int);args=p.parse_args()
    if args.watch_pid:
        while True:
            try:os.kill(args.watch_pid,0)
            except ProcessLookupError:break
            time.sleep(30)
    try:finish()
    except BaseException:
        common.write(common.EXP/f'REPORTING_FAILURE_{time.time_ns()}.json',dict(status='partial_or_reporting_failed',traceback=traceback.format_exc()))
        raise

if __name__=='__main__':main()
