from pathlib import Path
import json,hashlib,shutil,zipfile,gzip,pickle,sys
import numpy as np
R=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
NAME='ROOPF_Full_CEC解锁_在线210FE_LLM210-230FE_融合8轮_生成7轮_复现包'
P=OUT/NAME
assert not P.exists();P.mkdir()
old=R/'exports/ROOPF_CEC解锁_最终运行代码_含权重_20261026.zip'
with zipfile.ZipFile(old) as z:
 for item in z.infolist():
  path=Path(item.filename);parts=path.parts[1:]
  if not parts or item.is_dir():continue
  dest=P.joinpath(*parts);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(z.read(item))
for path in ['RUNTIME_MANIFEST.json','VERSION.json','README_运行说明.txt']:(P/path).unlink()
for name in ['scheduled_evidence_agent_r18.py','policy_api_r18.py']:
 shutil.copy2(R/'roopf'/name,P/'roopf'/name)
mem=R/'experiments/r15_llm_schedule_matched/cases/S01_L08/offline_memory.json';shutil.copy2(mem,P/'assets/offline_memory.json')
case=json.loads(mem.read_text());schedule=case['schedule'];(P/'config/schedule.json').write_text(json.dumps(schedule,ensure_ascii=False,indent=2)+'\n')
setting=json.loads((P/'config/final.json').read_text());setting['release']='final-s01l08-online210-20261030';setting['source_run']='R19 S01_L08 live Full with repaired LLM output contract';setting['llm_call_fe']=[210,230];setting['fusion_rounds']=8;setting['generation_rounds']=7;(P/'config/final.json').write_text(json.dumps(setting,ensure_ascii=False,indent=2)+'\n')
run=P/'run_final.py';s=run.read_text()
s=s.replace('from roopf.current_evidence_agent import CurrentEvidenceAgent','from roopf.scheduled_evidence_agent_r18 import ScheduledEvidenceAgent')
s=s.replace("    model=builder.build_optimizer(SimpleNamespace(dimension=10,population_size=100,budget=300,agentic_mode='off'))","    builder.set_seed(20260630)  # Match R19 model construction before loading paired RNG.\n    model=builder.build_optimizer(SimpleNamespace(dimension=10,population_size=100,budget=300,agentic_mode='off'))")
s=s.replace("agent=CurrentEvidenceAgent(ROOT/'assets/offline_memory.json',OUTPUT/'api_journal',mode=mode,arm='joint',", "agent=ScheduledEvidenceAgent(ROOT/'assets/offline_memory.json',OUTPUT/'api_journal',schedule=read(ROOT/'config/schedule.json'),mode=mode,arm='joint',")
oldinit='''    np.random.seed(seed);torch.manual_seed(seed)
    population=p.genRandomPop((10,100,10))'''
newinit='''    np.random.seed(seed);torch.manual_seed(seed)
    fixture_path=ROOT/'paired_initials'/f'{group}_f{fid:02d}'/f'b{batch:02d}.npz'
    with np.load(fixture_path,allow_pickle=False) as fixture:
        population=torch.from_numpy(fixture['population'].copy())
        torch.set_rng_state(torch.from_numpy(fixture['rng'].copy()))
        original_after=torch.from_numpy(fixture['original_after'].copy())'''
assert oldinit in s;s=s.replace(oldinit,newinit)
s=s.replace("==([5,15] if mode=='llm' else [])", "==([0,10] if mode=='llm' else [])")
s=s.replace("assert len(decisions)==45", "assert torch.equal(torch.get_rng_state(),original_after),'Final RNG state differs from paired original'\n    assert len(decisions)==45")
s=s.replace('fresh_initials=True,','fresh_initials=False,paired_initials=True,')
s=s.replace('Fresh deterministic starts: never read prior results or cached API responses.','Paired initial populations and RNG states: no prior objective results or API responses.')
needle="        assert audit['attempts']==2*len(tasks)\n        result=dict(status='complete',functions=stats,api=audit,native_fe=len(tasks)*3000,cached_fe=0,"
replacement="""        assert audit['attempts']==2*len(tasks)
        coverage={(g,f):0 for g in selected for f in range(1,count[g]+1)}
        for entry in api_rows:
            context=entry['execution_context'];coverage[(context['group'],context['fid'])]+=len(entry.get('valid_trajectory_ids',[]))
        complete=all(v>0 for v in coverage.values()) and audit['statuses']['pending']==0
        result=dict(status='complete_live_api_coverage_verified' if complete else 'complete_trajectories_api_coverage_incomplete',functions=stats,api=audit,valid_decision_rows_by_function={f'{g}:{f}':v for (g,f),v in coverage.items()},native_fe=len(tasks)*3000,cached_fe=0,"""
assert needle in s;s=s.replace(needle,replacement)
s=s.replace("write(output/'FINAL_RESULT.json',result);emit('complete',output=str(output),**audit)","write(output/'FINAL_RESULT.json',result);emit('complete',output=str(output),validation_status=result['status'],**audit)")
s=s.replace("        if agent.resource_exhausted or agent.uncertain_api_outcome:raise RuntimeError('API incomplete or uncertain; inspect journal')\n    np.savez_compressed","        if agent.resource_exhausted or agent.uncertain_api_outcome:raise RuntimeError('API incomplete or uncertain; inspect journal')\n        from roopf.current_evidence_agent import request_context_digest\n        for decision in decisions:\n            if decision['request_event']:\n                journal=read(OUTPUT/'api_journal'/(decision['api_request_sha256']+'.json'))\n                assert journal['config_ids']==decision['configs']\n                assert journal['decision_context_sha256']==request_context_digest(decision['api_trajectories'],decision['execution_context'])\n                assert decision['api_row_validation']['valid_trajectory_ids']==journal['valid_trajectory_ids']\n    np.savez_compressed")
s=s.replace("                    write(output/'PROGRESS.json',progress);emit('progress',**progress);last=now","                    journal_rows=[read(p) for p in (output/'api_journal').glob('*.json')]\n                    status_counts={state:sum(row['status']==state for row in journal_rows) for state in ('ok','partial','failed','pending')}\n                    progress['api_status']=status_counts\n                    write(output/'PROGRESS.json',progress);emit('progress',**progress);last=now\n                    finished=status_counts['ok']+status_counts['partial']+status_counts['failed']\n                    if finished>=100 and (status_counts['ok']+status_counts['partial'])/finished<0.2:\n                        raise RuntimeError('API valid response rate below 20% over at least 100 completed requests; stopping with records preserved')")
s=s.replace("    agent.execution_context=dict(release=SETTINGS['release'],group=group,fid=fid,batch=batch,arm='llm_full',\n        execution_policy='same_as_bbob_no_cec_structural_exclusion',fresh_run=True)","    agent.execution_context=dict(round='r19_s01l08_component_ablations',group=group,fid=fid,batch=batch,arm='llm_full',\n        online_admission_start_fe=210,llm_decision_fe=[210,230],schedule_signature=read(ROOT/'config/schedule.json'),memory_sha256=agent.memory_hash)")
s=s.replace('    agent.execution_context.update(initial_population_sha256=hashlib.sha256(population.numpy().tobytes()).hexdigest(),\n        solver_rng_sha256=hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),seed=seed)','    agent.execution_context.update(paired_initial_sha256=hashlib.sha256(population.numpy().tobytes()).hexdigest(),\n        paired_solver_rng_sha256=hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest())')
run.write_text(s)
for group,n in [('BBOB',24),('CEC-style',6),('UAV',5)]:
 for fid in range(1,n+1):
  for b in range(10):
   oldf=R/'experiments/r10_current_candidate_fusion/results'/f'{group}_f{fid:02d}'/f'b{b:02d}'/'PAIRED_START.pkl.gz'
   with gzip.open(oldf,'rb') as f:data=pickle.load(f)
   assert set(data)=={'population','rng','original_after'} and data['population'].shape==(10,100,10)
   to=P/'paired_initials'/f'{group}_f{fid:02d}'/f'b{b:02d}.npz';to.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(to,**data)
readme='''ROOPF Full 可复现运行包：CEC 解锁、在线 210 FE、LLM 210/230 FE、融合8轮、下一轮起生成7轮

一、内容与适用范围
本包运行已核验的 R19 Full 分支：D10、初始种群100、总预算300 FE（含初始100）、每函数100轨迹（10批×10条），35函数（BBOB24、CEC-style6、UAV5）。包含冻结 Anchor 与 Residual 权重、S01_L08 464条检索案例及其覆盖半径、CEC解锁策略、修复后的API输出契约、BBOB偏移、UAV目标定义和350批配对初始种群及随机状态。无历史实验结果、目标函数缓存、API响应日志或密钥；不包含三项消融、论文对照算法及其结果。无需训练即可运行 Full。训练环境、离线案例重建所用完整训练日志和训练任务不在此包中。

二、运行
Linux / WSL2 的 ooneurgo 环境：
  conda activate ooneurgo
  python -u run_final.py --check
  python -u run_final.py --benchmark all --workers 24 --device cuda --progress-seconds 10
可选 --benchmark bbob、cec、uav；--device cpu 使用CPU代理，但计算数值可能不同。实时显示JSON进度并写入 outputs/时间/；Full 全35函数预计700次真实API请求。--check 不调用 API，不替代全量验证。

三、密钥与网络
按单行格式将密钥放入 ~/.config/roopf_agent/worldcodes.env：
  WORLDCODES_API_KEY=自己的密钥
密钥不包含在包里。默认使用 https://worldcodes.online/v1/chat/completions、deepseek-v4-flash 和 http://127.0.0.1:7890；代理可通过 --api-proxy 修改，空字符串表示不用代理。修改模型、端点或后端属于新的运行条件。API调用可能收费。失败时使用 matched38 回退，不自动重试。完成状态需全部函数有有效执行返回。

四、数据与数值说明
350份配对初始种群及 RNG 状态是运行输入，不含任何历史目标值、旧决策或 API 响应；BBOB20 保持10条轨迹的批次语义。实时LLM响应和计算后端可能产生不同动作与最终数值，不能保证历史结果逐位一致。案例库在离线训练函数上为完整架构采集；不会在线训练或重建。历史对照、消融和配对轨迹结果不包含在包内。

五、新环境
Python 3.10；torch 2.11.0+cu128、numpy 2.2.6、sympy 1.14.0；参考 requirements.txt。CUDA运行需要兼容的NVIDIA驱动。当前代码依赖 fcntl 和 Unix socket，使用 Linux/WSL2。\n'''
(P/'README_运行说明.txt').write_text(readme)
(P/'VERSION.json').write_text(json.dumps({'release':setting['release'],'full_configuration':'online210_call210-230_fusion8_generation7','source_experiment':'R19 S01_L08 fresh live Full','independent_audit_status':json.loads((R/'experiments/r19_s01l08_component_ablations/INDEPENDENT_AUDIT.json').read_text())['status'],'paired_initial_count':350,'api_replay':False,'historical_results_included':False,'credentials_included':False,'cuda_or_cpu':'CPU solver with shared CUDA ridge proxy by default'},ensure_ascii=False,indent=2)+'\n')
files={str(p.relative_to(P)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(P.rglob('*')) if p.is_file()}
(P/'RUNTIME_MANIFEST.json').write_text(json.dumps({'release':setting['release'],'files':files},ensure_ascii=False,indent=2)+'\n')
print('PACKAGE_DIR',P)
