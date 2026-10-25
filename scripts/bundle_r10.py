"""Read-only post-run R10 publication; not imported by the frozen solver."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from decimal import Decimal
import collections
import csv
import gzip
import hashlib
import json
from pathlib import Path
import sys
import tarfile
from zoneinfo import ZoneInfo

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def main():
    import numpy as np
    from run_r10_hardware import configure
    import report_r10
    p=ROOT/'experiments/r10_current_candidate_fusion'
    run,_=configure(str(p),json.loads((p/'HARDWARE_RUNTIME.json').read_text()))
    verification=report_r10.verify(run)
    assert verification['status']=='full_records_verified'
    publication=p/'publication';publication.mkdir(exist_ok=False)
    key=(Path.home()/'.config/roopf_agent/worldcodes.env').read_text().strip().split('=',1)[1].encode()
    journals=sorted((p/'api_journal').glob('*.json'))
    for path in journals:assert key not in path.read_bytes(),'Credential value present in journal'
    with tarfile.open(publication/'API_JOURNAL.tar.xz','w:xz',preset=3) as archive:
        for path in journals:archive.add(path,arcname=str(path.relative_to(p)))
    print('Packaged all real request/response journals without credential values.',flush=True)
    with tarfile.open(publication/'EVALUATION_EVIDENCE.tar.xz','w:xz',preset=3) as archive:
        for group,fid in run.legacy.TASKS:
            folder=p/'results'/f'{group}_f{fid:02d}'
            for path in sorted([folder/'COMPLETE.json',*folder.glob('*_FULL100.npz'),*folder.glob('b??/*.json'),*folder.glob('b??/PAIRED_START.pkl.gz')]):
                archive.add(path,arcname=str(path.relative_to(p)))
            frozen=ROOT/'experiments/r07_hardware_accelerated_v1/results'/f'{group}_f{fid:02d}/matched38_FULL100.npz'
            archive.add(frozen,arcname=f'frozen_reference/r07/{group}_f{fid:02d}/matched38_FULL100.npz')
    def shard(task):
        group,fid,arm=task
        path=publication/f'DECISIONS_{group}_f{fid:02d}_{arm}.jsonl.gz'
        rows=0
        with gzip.open(path,'wt') as stream:
            for batch in range(10):
                source=p/'results'/f'{group}_f{fid:02d}/b{batch:02d}/{arm}_DECISIONS.json.gz'
                with gzip.open(source,'rt') as reader:records=json.load(reader)
                for r in records:
                    if r['request_event']:
                        selected=r
                    elif r['step'] in (*range(6,10),*range(16,20)):
                        # Full requests retain current candidates and predictions.
                        # Subsequent executed rounds retain actual chosen coordinates
                        # and configs without replicating every unused candidate.
                        selected={k:r[k] for k in ('step','configs','applied_generation','planned_generation','selected','original','policy_id','execution_context','temporal_semantics')}
                        selected['executed_points']=[c['candidates'][c['selected_index']] for c in r['current']]
                        selected['raw_record_sha256']=hashlib.sha256(json.dumps(r,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
                    else:continue
                    stream.write(json.dumps(dict(group=group,fid=fid,batch=batch,arm=arm,decision=selected),separators=(',',':'),allow_nan=False)+'\n');rows+=1
        assert key not in path.read_bytes()
        return dict(path=path.name,rows=rows,sha256=sha(path))
    tasks=[(g,f,a) for g,f in run.legacy.TASKS if g!='CEC-style' for a in run.settings()['arms'] if a.startswith('llm_')]
    shards=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for future in as_completed([pool.submit(shard,t) for t in tasks]):
            shards.append(future.result())
            if len(shards)%10==0:print(f'Executed-decision shards: {len(shards)}/{len(tasks)}',flush=True)
    files={f.name:sha(f) for f in publication.iterdir() if f.is_file()}
    assert all(f.stat().st_size<95*1024**2 for f in publication.iterdir() if f.is_file()),'Split large artifact before publishing'
    manifest=dict(status='complete_scientific_evidence_bundle',sha256=files,decision_shards=sorted(shards,key=lambda r:r['path']),
        raw35_archives_retained_locally=True,raw_archives_manifest_sha256=sha(p/'ARCHIVE_MANIFEST.json'),
        partial_setup_attempt_preserved='setup_attempt_01_scalar_gpu',
        source_script_sha256=sha(Path(__file__)),api_credentials_included=False)
    (publication/'MANIFEST.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    summary=json.loads((p/'FINAL_RESULT.json').read_text());audit=verification['api']
    paired_rows=[];direct={}
    for alternative in ('llm_no_offline','llm_no_online'):
        totals=collections.Counter()
        for group,fid in run.legacy.TASKS:
            folder=p/'results'/f'{group}_f{fid:02d}'
            full=np.load(folder/'llm_live_FULL100.npz')['values'].min(1).astype(float)
            other=np.load(folder/f'{alternative}_FULL100.npz')['values'].min(1).astype(float)
            primary=not(group=='BBOB' and fid==20)
            row=dict(group=group,fid=fid,primary=primary,comparison='Full_vs_'+alternative,
                full_mean=float(full.mean()),full_std=float(full.std(ddof=0)),
                alternative_mean=float(other.mean()),alternative_std=float(other.std(ddof=0)),
                wins=int((full<other).sum()),ties=int((full==other).sum()),losses=int((full>other).sum()))
            paired_rows.append(row)
            if primary:
                totals.update(function_wins=int(full.mean()<other.mean()),function_ties=int(full.mean()==other.mean()),function_losses=int(full.mean()>other.mean()),
                    trajectory_wins=row['wins'],trajectory_ties=row['ties'],trajectory_losses=row['losses'])
        direct['Full_vs_'+alternative]=dict(totals)
    with (p/'AGENT_INPUT_ABLATIONS.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(paired_rows[0]));writer.writeheader();writer.writerows(paired_rows)
    api_detail={a:dict(counts=collections.Counter(),credit=Decimal(0),credit_responses=0) for a in ('llm_live','llm_no_offline','llm_no_online')}
    for path in journals:
        record=json.loads(path.read_text());detail=api_detail[record['execution_context']['arm']];counts=detail['counts']
        response=record.get('response',{});usage=response.get('usage',{});choice=response.get('choices',[{}])[0]
        counts[record['status']]+=1;counts['prompt_tokens']+=usage.get('prompt_tokens',0);counts['completion_tokens']+=usage.get('completion_tokens',0)
        counts['reasoning_responses']+=int(bool(choice.get('message',{}).get('reasoning_content')))
        counts['failed_completion_tokens']+=usage.get('completion_tokens',0) if record['status']=='failed' else 0
        if record['status']=='failed':counts['error:'+record.get('error','all_rows_invalid')]+=1
        credit=usage.get('credit')
        if isinstance(credit,(int,float)):detail['credit']+=Decimal(str(credit));detail['credit_responses']+=1
    api_detail={a:dict(counts=dict(d['counts']),provider_credit_sum=str(d['credit']),responses_with_credit=d['credit_responses'],currency='not inferred') for a,d in api_detail.items()}
    comparisons=[]
    for arm in ('llm_live','llm_no_offline','llm_no_online','retrieval'):
        a=summary['comparisons_to_matched38'][arm];b=summary['comparisons'][arm]
        comparisons.append(f"{arm}: vs matched38 函数{a['function_wins']}/{a['function_ties']}/{a['function_losses']}；轨迹{a['trajectory_wins']}/{a['trajectory_ties']}/{a['trajectory_losses']}；vs原版函数{b['function_wins']}/{b['function_ties']}/{b['function_losses']}，轨迹{b['trajectory_wins']}/{b['trajectory_ties']}/{b['trajectory_losses']}")
    actual_changes={}
    for arm in ('llm_live','llm_no_offline','llm_no_online'):
        action_counts={};support=0;selected_adjusted=0;variant_rounds=0;variant_selected=0
        for group,fid in run.legacy.TASKS:
            for batch in range(10):
                folder=p/'results'/f'{group}_f{fid:02d}/b{batch:02d}'
                receipt=json.loads((folder/f'{arm}.json').read_text())
                support+=receipt['agent_stats'].get('unsupported_queries',0)
                selected_adjusted+=receipt['agent_stats'].get('selected_adjusted_candidates',0)
                with gzip.open(folder/f'{arm}_DECISIONS.json.gz','rt') as stream:records=json.load(stream)
                for r in records:
                    if r['request_event']:
                        for name in r['configs']:action_counts[name]=action_counts.get(name,0)+1
                    for current in r['current']:
                        if current['extra_origin']=='anchor_variant':
                            variant_rounds+=1;variant_selected+=int(current['selected_index']==37)
        different=0
        for group,fid in run.legacy.TASKS:
            if group=='BBOB' and fid==20:continue
            folder=p/'results'/f'{group}_f{fid:02d}'
            different+=int((np.load(folder/f'{arm}_FULL100.npz')['values'].min(1)!=np.load(folder/'matched38_FULL100.npz')['values'].min(1)).sum())
        actual_changes[arm]=dict(executed_plan_counts_including_fallbacks=action_counts,
            offline_coverage_unsupported_queries=None if arm=='llm_no_offline' else support,
            offline_input_disabled_queries=support if arm=='llm_no_offline' else 0,
            selected_adjusted_candidates=selected_adjusted,anchor_variant_rounds=variant_rounds,anchor_variant_selected=variant_selected,
            primary_terminal_values_different_from_matched38=different)
    finish=[json.loads(line) for line in (p/'PROGRESS.jsonl').read_text().splitlines() if 'requested_phases_complete' in line]
    assert finish,'Pipeline did not complete requested phases'
    compatibility=json.loads((p/'compatibility/RESULT.json').read_text())
    analysis=dict(status='full_live_r10_completed_no_automatic_improvement_claim',reviewed_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        completion_event=finish[-1],comparisons_to_matched38=summary['comparisons_to_matched38'],comparisons_to_original=summary['comparisons'],
        api=audit,api_detail_by_arm=api_detail,direct_full_vs_ablations=direct,compatibility_api_separate=compatibility,actual_changes=actual_changes,
        caveats=['Development after R09, not blind confirmatory evidence','Ablations remove Agent input layers, not frozen Residual or current online proxy','No post-result changes to gates or test settings','BBOB20 excluded from primary efficacy','CEC six functions retain original no-intervention locks','Partial scalar-GPU setup objective cost is not reconstructed; raw partial records retained'])
    (p/'POSTRUN_ANALYSIS.json').write_text(json.dumps(analysis,ensure_ascii=False,indent=2)+'\n')
    text=['R10全量实测复核（胜/平/负；函数按均值，轨迹按配对最终值；不代表显著性检验）',
        '35函数x100轨迹，D10，FE300含初始100，六臂全部完成；冻结原版与matched38逐评价一致。',
        '主评估34函数/3400轨迹，BBOB20单列；CEC六函数结构锁保留。',*comparisons,
        f"真实正式API请求{audit['attempts']}；全有效{audit['fully_valid']}；部分有效{audit['partial']}；失败{audit['failures']}。有效动作行{audit['valid_rows']}；回退行{audit['fallback_rows']}。全部请求匹配执行，未消费有效响应0。",
        f"正式输入token{audit['prompt_tokens']}，输出token{audit['completion_tokens']}；供应商credit合计{audit['provider_credit_sum']}，报告该字段的响应{audit['responses_with_credit']}，不推断币种。",
        '兼容性检查另计1次真实API，不作为正式实验；单轨迹GPU低吞吐启动及部分采集记录完整保留。',
        '去离线：不提供案例、不使用案例支持门；去在线：不提供已选择点的历史反馈，检索移除对应特征。在线代理与数值核心保持，不能称为删除代理网络的消融。',
        'Full直接对照两项消融：'+json.dumps(direct,ensure_ascii=False),
        'changed_points相对于当前候选池原选择器；fusion_changed_points相对于同一池默认融合；终值差异相对于冻结matched38。三种口径不可混用。',
        '实际作用统计：'+json.dumps(actual_changes,ensure_ascii=False),
        '每臂评价/改点：'+json.dumps(summary['totals'],ensure_ascii=False),
        '逐函数均值、总体标准差ddof0、对两套基线配对胜平负见FUNCTION_COMPARISON.csv；完整API与覆盖见HARDWARE_API_EXECUTION_AUDIT.json。',
        '实验完成不自动意味着优化改进。']
    (p/'REVIEW.zh-CN.txt').write_text('\n\n'.join(text)+'\n')
    print(json.dumps(dict(status='bundled',api=audit['attempts'],comparisons=summary['comparisons_to_matched38']),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
