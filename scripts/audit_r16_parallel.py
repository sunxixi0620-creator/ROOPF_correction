"""Independent parallel audit of completed R15 evidence. No solver/API execution."""
from __future__ import annotations
import os
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[name]='1'
import argparse,collections,csv,gzip,hashlib,json,multiprocessing as mp,pickle,time,traceback
from pathlib import Path
from decimal import Decimal
import numpy as np
import r16_common as common
EXP=common.EXP
CONFIGS=common.CONFIGS
FUNCTIONS=[(g,f) for g,n in [('BBOB',24),('CEC-style',6),('UAV',5)] for f in range(1,n+1)]
MEMORIES={}

def csvwrite(path,rows):
    if not rows:return
    with Path(path).open('w',encoding='utf-8-sig',newline='') as stream:
        fields=list(dict.fromkeys(k for row in rows for k in row));w=csv.DictWriter(stream,fieldnames=fields);w.writeheader();w.writerows(rows)

def inspect_batch(task):
    g,f,b,cid=task
    from roopf.scheduled_evidence_agent import digest,request_context_digest
    from roopf.policy_api_r10 import validate_response,wire_bytes
    from experiment_r16_schedule import check_decisions
    folder=EXP/'results'/f'{g}_f{f:02d}'/f'b{b:02d}'
    receipt=common.read(folder/(cid+'.json'))
    assert receipt['binding']==common.read(EXP/'CODE_BINDING.json')
    assert receipt['config']==CONFIGS[cid] and receipt['n']==10 and receipt['nfe']==300
    assert receipt['objective_rng_boundaries_matched']==201 and receipt['final_rng_matches_frozen_original']
    assert receipt['native_fe']+receipt['cached_fe']==3000
    paired=common.ROOT/'experiments/r10_current_candidate_fusion/results'/f'{g}_f{f:02d}'/f'b{b:02d}/PAIRED_START.pkl.gz'
    with gzip.open(paired,'rb') as stream:start=pickle.load(stream)
    assert common.sha_bytes(start['population'].tobytes())==receipt['paired_initial_sha256']
    assert common.sha_bytes(start['rng'].tobytes())==receipt['paired_solver_rng_sha256']
    with np.load(folder/(cid+'.npz')) as data:
        values=data['values'];points=data['executed_points'];refs=data['reference_points']
        assert values.shape==(10,300) and np.isfinite(values).all()
        final=values.min(1).astype(float)
    with np.load(folder/'REFERENCES.npz') as data:
        old=data['frozen_final_values'].min(1).astype(float);base=data['original_values'].min(1).astype(float)
        for name,k in [('frozen_final','coordinate_changes_vs_frozen_final'),('original','coordinate_changes_vs_original')]:
            assert int((points!=data[name+'_points']).any(-1).sum())==receipt[k]
    native=folder/(cid+'_NATIVE_CALLS.pkl.gz');native_count=0
    if native.exists():
        with gzip.open(native,'rb') as stream:calls=pickle.load(stream)
        for call in calls:
            assert len(call['x'])==int(np.prod(call['shape']))*4
            assert len(call['y'])==int(np.prod(call['shape'][:2]))*4
            assert np.isfinite(np.frombuffer(call['y'],dtype=np.float32)).all()
            assert len(call['rng_before'])==len(start['rng'].tobytes())==len(call['rng_after'])
            native_count+=int(np.prod(call['shape']))//10
    assert native_count==receipt['native_fe']
    with gzip.open(folder/(cid+'_DECISIONS.json.gz'),'rt') as stream:decisions=json.load(stream)
    check_decisions(decisions,points,refs,CONFIGS[cid],True)
    assert receipt['agent_stats']['requests']+receipt['agent_stats']['api_reused']==2
    if cid not in MEMORIES:
        memory=common.read(EXP/CONFIGS[cid]['memory_path'])
        assert memory['schedule']==common.schedule(cid)
        assert memory['schedule_signature']==digest(common.schedule(cid))
        MEMORIES[cid]=(digest(memory),memory['schedule_signature'])
    memory_hash,signature=MEMORIES[cid]
    requests=[];covered=default_only=nondefault=queries=0
    control_rounds=sum(sum(n not in ('matched38','release') for n in d['configs']) for d in decisions)
    gen_rounds=sum(sum(n not in ('matched38','release') for n in d['applied_generation']) for d in decisions)
    for d in decisions:
        assert d['config_id']==cid and d['schedule_signature']==signature
        if not d['request_event']:continue
        identity=d['api_request_sha256'];r=common.read(EXP/'api_journal'/(identity+'.json'))
        assert r['status'] in ('ok','partial','failed')
        assert r['execution_context']==d['execution_context'] and r['config_ids']==d['configs']
        assert r['execution_context']['arm']==cid and r['execution_context']['group']==g and r['execution_context']['fid']==f and r['execution_context']['batch']==b
        assert hashlib.sha256(wire_bytes(r['request'])).hexdigest()==identity
        payload=json.loads(r['request']['messages'][-1]['content'].split('INPUT:\n',1)[1])
        assert payload['memory_sha256']==r['memory_sha256']==memory_hash
        assert payload['schedule_signature']==signature
        assert payload['current_fusion_rounds']==CONFIGS[cid]['fusion_rounds'] and payload['future_generation_rounds']==CONFIGS[cid]['generation_rounds']
        assert payload['execution_context_sha256']==digest(r['execution_context'])
        assert payload['decision_context_sha256']==r['decision_context_sha256']==request_context_digest(d['api_trajectories'],d['execution_context'])
        if r['status'] in ('ok','partial'):
            decoded=validate_response(r['response'],payload)
            assert all(decoded[k]==r[k] for k in decoded)
        assert d['api_row_validation']['valid_trajectory_ids']==r['valid_trajectory_ids']
        for t in d['api_trajectories']:
            queries+=1;covered+=int(t['evidence']['supported']);default_only+=int(t['eligible_configs']==['matched38'])
        nondefault+=sum(n!='matched38' for n in d['configs'])
        usage=r.get('response',{}).get('usage',{})
        requests.append(dict(identity=identity,status=r['status'],valid_rows=len(r['valid_trajectory_ids']),fallback_rows=len(r['row_errors']),
            prompt_tokens=usage.get('prompt_tokens',0),completion_tokens=usage.get('completion_tokens',0),
            credit=usage.get('credit'),error=r.get('error')))
    assert len(requests)==2
    return dict(group=g,fid=f,batch=b,config_id=cid,final=final.tolist(),frozen_final=old.tolist(),original=base.tolist(),
        coordinate_changes_vs_frozen_final=receipt['coordinate_changes_vs_frozen_final'],coordinate_changes_vs_original=receipt['coordinate_changes_vs_original'],
        native_fe=receipt['native_fe'],cached_fe=receipt['cached_fe'],requests=requests,queries=queries,covered=covered,default_only=default_only,
        nondefault_decisions=nondefault,nondefault_fusion_rounds=control_rounds,nondefault_generation_rounds=gen_rounds)

def summarize(rows,started):
    from report_r16_schedule import audit_cases
    case_audit=audit_cases(EXP)
    assert case_audit['configurations']==24
    grouped=collections.defaultdict(list);identities=set();api=collections.Counter();coverage={};counters={};request_rows=[];failure_reasons=collections.Counter()
    credit=Decimal(0);credit_n=0
    for r in rows:
        key=(r['group'],r['fid'],r['config_id']);grouped[key].append(r)
        coverage[key]=coverage.get(key,0)+sum(q['valid_rows'] for q in r['requests'])
        for q in r['requests']:
            assert q['identity'] not in identities;identities.add(q['identity'])
            api['attempts']+=1;api[q['status']]+=1;api['valid_rows']+=q['valid_rows'];api['fallback_rows']+=q['fallback_rows']
            api['prompt_tokens']+=q['prompt_tokens'];api['completion_tokens']+=q['completion_tokens']
            if q['credit'] is not None:credit+=Decimal(str(q['credit']));credit_n+=1
            if q['status']=='failed':failure_reasons[q['error'] or 'row_validation_failure']+=1
            request_rows.append(dict(config_id=r['config_id'],group=r['group'],fid=r['fid'],batch=r['batch'],**q))
    assert len(rows)==8400 and len(identities)==16800
    assert identities=={p.stem for p in (EXP/'api_journal').glob('*.json')}
    assert len(coverage)==840 and all(v>0 for v in coverage.values())
    values={};refs={};stats=[];comparisons=[];detail=[];measurements=[]
    for key,rr in grouped.items():
        assert sorted(r['batch'] for r in rr)==list(range(10))
        rr=sorted(rr,key=lambda r:r['batch']);g,f,cid=key
        x=np.concatenate([r['final'] for r in rr]);assert x.shape==(100,)
        values[key]=x;ref={name:np.concatenate([r[name] for r in rr]) for name in ('frozen_final','original')}
        for name,v in ref.items():
            if (g,f,name) in refs:assert np.array_equal(refs[g,f,name],v)
            refs[g,f,name]=v
        c={name:sum(r[name] for r in rr) for name in ['native_fe','cached_fe','coordinate_changes_vs_frozen_final','coordinate_changes_vs_original','queries','covered','default_only','nondefault_decisions','nondefault_fusion_rounds','nondefault_generation_rounds']}
        counters[key]=c
        stats.append(dict(group=g,fid=f,config_id=cid,n=100,mean=float(x.mean()),population_std=float(x.std(ddof=0)),variance=float(x.var(ddof=0)),**c))
        for i,v in enumerate(x):detail.append(dict(config_id=cid,group=g,fid=f,trajectory=i,final_value=float(v)))
    for name in ('frozen_final','original'):
        for g,f in FUNCTIONS:
            x=refs[g,f,name]
            measurements.append(dict(group=g,fid=f,config_id=name,n=100,mean=float(x.mean()),population_std=float(x.std(ddof=0)),variance=float(x.var(ddof=0))))
    groups={'primary_34_functions':[(g,f) for g,f in FUNCTIONS if (g,f)!=('BBOB',20)],'BBOB23':[(g,f) for g,f in FUNCTIONS if g=='BBOB' and f!=20],
        'BBOB24':[(g,f) for g,f in FUNCTIONS if g=='BBOB'],'CEC6':[(g,f) for g,f in FUNCTIONS if g=='CEC-style'],'UAV5':[(g,f) for g,f in FUNCTIONS if g=='UAV'],'BBOB20':[('BBOB',20)]}
    rankings=[]
    for name,fs in groups.items():
        for cid,cfg in CONFIGS.items():
            rank=[]
            for g,f in fs:
                m=values[g,f,cid].mean();others=np.array([values[g,f,c].mean() for c in CONFIGS]);rank.append(float((others<m).sum()+(1+(others==m).sum())/2))
            rankings.append(dict(group=name,config_id=cid,call_fe=f"{cfg['first_call_fe']}/{cfg['second_call_fe']}",fusion_rounds=cfg['fusion_rounds'],generation_rounds=cfg['generation_rounds'],functions=len(fs),average_rank=float(np.mean(rank))))
            for reference in ('frozen_final','original','matched_default'):
                fw=ft=fl=pw=pt=pl=0
                for g,f in fs:
                    x=values[g,f,cid];y=values[g,f,'S02_L05'] if reference=='matched_default' else refs[g,f,reference]
                    fw+=int(x.mean()<y.mean());ft+=int(x.mean()==y.mean());fl+=int(x.mean()>y.mean());pw+=int((x<y).sum());pt+=int((x==y).sum());pl+=int((x>y).sum())
                comparisons.append(dict(group=name,config_id=cid,reference=reference,functions=len(fs),function_wins=fw,function_ties=ft,function_losses=fl,paired_wins=pw,paired_ties=pt,paired_losses=pl))
    csvwrite(EXP/'INDEPENDENT_FUNCTION_STATISTICS.csv',stats+measurements)
    csvwrite(EXP/'INDEPENDENT_PARAMETER_RANKINGS.csv',rankings);csvwrite(EXP/'INDEPENDENT_COMPARISONS.csv',comparisons)
    csvwrite(EXP/'INDEPENDENT_API_REQUESTS.csv',request_rows);csvwrite(EXP/'INDEPENDENT_PAIRED_FINAL_VALUES.csv',detail)
    result=dict(status='complete_live_schedule_measurement_independently_verified',trajectories=84000,condition_batches=8400,
        api_records=16800,configuration_function_valid_API_coverage=840,all100_trajectories_per_function=True,
        initial_population_and_solver_rng_hashes_matched=True,all_actual_evaluation_points_and_API_decisions_matched=True,
        native_call_integrity_verified=True,api=dict(api),provider_credit_units=str(credit),responses_with_credit=credit_n,currency_inferred=False,
        native_test_fe=sum(r['native_fe'] for r in rows),cached_test_fe=sum(r['cached_fe'] for r in rows),
        actual_changed_points_vs_frozen_full=sum(r['coordinate_changes_vs_frozen_final'] for r in rows),actual_changed_points_vs_original=sum(r['coordinate_changes_vs_original'] for r in rows),
        case_audit=case_audit,rankings=rankings,comparisons=comparisons,failure_reasons=dict(failure_reasons),
        source_sha256=common.sha(__file__),execution_binding=common.read(EXP/'CODE_BINDING.json'),
        elapsed_audit_seconds=time.monotonic()-started,not_significance_test=True,bbob20_reported_separately=True)
    export_excel(result,stats+measurements)
    common.write(EXP/'INDEPENDENT_FINAL_AUDIT.json',result)
    return result

def export_excel(result,stats):
    from openpyxl import Workbook
    from openpyxl.styles import Font,PatternFill,Alignment,Border,Side
    from openpyxl.utils import get_column_letter
    wb=Workbook();wb.remove(wb.active)
    def sheet(name,title,note,headers,data,widths):
        ws=wb.create_sheet(name);n=len(headers)
        for row,text in [(1,title),(2,note)]:
            ws.merge_cells(start_row=row,start_column=1,end_row=row,end_column=n);c=ws.cell(row,1,text)
            c.font=Font(name='微软雅黑',size=17 if row==1 else 10,bold=row==1,color='193B60');c.alignment=Alignment(wrap_text=True,vertical='center')
            ws.row_dimensions[row].height=34 if row==1 else 42
        for j,h in enumerate(headers,1):
            c=ws.cell(4,j,h);c.font=Font(name='微软雅黑',bold=True,color='FFFFFF');c.fill=PatternFill('solid',fgColor='193B60');c.alignment=Alignment(wrap_text=True,vertical='center',horizontal='center')
        ws.row_dimensions[4].height=34
        for i,row in enumerate(data,5):
            for j,v in enumerate(row,1):
                c=ws.cell(i,j,v);c.font=Font(name='微软雅黑',size=11,color='193B60');c.alignment=Alignment(vertical='center',horizontal='left' if j==1 else 'center')
                if i%2:c.fill=PatternFill('solid',fgColor='EDF3F9')
                if isinstance(v,float):c.number_format='0.0000' if abs(v)<100 and name!='逐函数均值方差' else '0.0000000000E+00'
            ws.row_dimensions[i].height=23
        for j,w in enumerate(widths,1):ws.column_dimensions[get_column_letter(j)].width=w
        ws.freeze_panes='B5';ws.auto_filter.ref=f'A4:{get_column_letter(n)}{ws.max_row}';ws.sheet_view.showGridLines=False
        ws.sheet_properties.pageSetUpPr.fitToPage=True;ws.page_setup.orientation='landscape';ws.page_setup.paperSize=ws.PAPERSIZE_A4;ws.page_setup.fitToWidth=1;ws.page_setup.fitToHeight=0;ws.print_title_rows='1:4'
        return ws
    group_names={'primary_34_functions':'总体主排名（34函数）','BBOB23':'BBOB主排名（23函数）','BBOB24':'BBOB完整排名（24函数）','CEC6':'CEC分类排名','UAV5':'UAV分类排名','BBOB20':'BBOB20单列'}
    for g,title in group_names.items():
        rr=sorted([r for r in result['rankings'] if r['group']==g],key=lambda r:r['average_rank']);data=[]
        for r in rr:
            less=sum(x['average_rank']<r['average_rank'] for x in rr);eq=sum(x['average_rank']==r['average_rank'] for x in rr)
            cf=next(c for c in result['comparisons'] if c['group']==g and c['config_id']==r['config_id'] and c['reference']=='matched_default')
            data.append([f'并列{less+1}' if eq>1 else less+1,r['config_id'],r['call_fe'],r['fusion_rounds'],r['generation_rounds'],r['average_rank'],cf['function_wins'],cf['function_ties'],cf['function_losses']])
        sheet(g[:28],title,'各函数100条配对轨迹；平均秩在24组配置内按各函数最终均值计算，越小越好；胜平负均从该配置相对本轮默认S02_L05的角度统计。',
              ['排名','配置','调用FE','融合轮数','生成轮数','平均秩','函数胜','函数平','函数负'],data,[12,16,18,12,12,14,12,12,12])
    for ref,label in [('matched_default','对照本轮默认'),('frozen_final','对照当前确定Full'),('original','对照冻结基线')]:
        data=[]
        for c in sorted([r for r in result['comparisons'] if r['reference']==ref],key=lambda r:(r['config_id'],r['group'])):
            data.append([c['config_id'],c['group'],c['functions'],c['function_wins'],c['function_ties'],c['function_losses'],c['paired_wins'],c['paired_ties'],c['paired_losses']])
        sheet(label,label+'：分类配对比较','每函数100条轨迹；函数胜负按终值均值比较，配对胜负按同一初始种群及随机流的单条终值比较；BBOB24为完整补充口径。',
              ['配置','问题类别','函数数','函数胜','函数平','函数负','配对胜','配对平','配对负'],data,[16,28,12,12,12,12,14,14,14])
    data=[]
    for r in sorted(stats,key=lambda r:(r['config_id'],r['group'],r['fid'])):
        data.append([r['config_id'],f"{r['group']}_f{r['fid']:02d}",r['mean'],r['variance']])
    sheet('逐函数均值方差','具体测量数值：按配置分类','每函数100条终值；方差使用总体方差ddof=0。frozen_final为已确定Full，original为独立冻结基线。总体标准差及全部执行统计另见CSV记录。',
          ['配置／算法','测试函数','最终均值','总体方差'],data,[24,26,27,27])
    sheet('API与执行核验','真实API与执行核验','失败调用按照预定matched38回退；credit为供应商返回的单位，未推断币种；完整请求、部分有效及失败均保留。',
          ['指标','数值'],[[k,v] for k,v in result['api'].items()]+[['provider_credit_units',result['provider_credit_units']],['responses_with_credit',result['responses_with_credit']],['native_test_fe',result['native_test_fe']],['cached_test_fe',result['cached_test_fe']],['verified_trajectories',84000],['verified_API_records',16800]],[45,40])
    wb.save(EXP/'LLM_Agent_参数实验_独立核验结果.xlsx')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--wait',action='store_true');args=parser.parse_args()
    while True:
        completed=sum(1 for _ in EXP.glob('results/*/b??/S??_L??.json'))
        if completed==8400:break
        if not args.wait:raise RuntimeError(f'Incomplete: {completed}/8400')
        print(json.dumps(dict(event='waiting_for_full_measurement',completed_batches=completed,total=8400)),flush=True);time.sleep(30)
    started=time.monotonic();common.freeze();rows=[]
    tasks=[(g,f,b,c) for g,f in FUNCTIONS for b in range(10) for c in CONFIGS]
    with mp.get_context('spawn').Pool(24) as pool:
        for r in pool.imap_unordered(inspect_batch,tasks,chunksize=2):
            rows.append(r)
            if len(rows)%200==0:print(json.dumps(dict(event='independent_audit_progress',done=len(rows),total=8400,elapsed_seconds=time.monotonic()-started)),flush=True)
    result=summarize(rows,started);common.validate_frozen()
    print(json.dumps(dict(event='independent_full_audit_pass',status=result['status'],trajectories=84000,api_records=16800,elapsed_seconds=time.monotonic()-started)),flush=True)

if __name__=='__main__':
    try:main()
    except BaseException:
        common.write(EXP/f'INDEPENDENT_AUDIT_FAILURE_{time.time_ns()}.json',dict(status='audit_failed',traceback=traceback.format_exc()));raise
