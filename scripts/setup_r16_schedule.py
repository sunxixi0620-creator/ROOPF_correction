"""Create an isolated FE200 factorial experiment from the frozen R15 protocol."""
import csv, hashlib, json
from pathlib import Path
R=Path(__file__).resolve().parents[1]
E=R/'experiments/r16_fe200_llm_schedule'
E.mkdir(parents=True,exist_ok=False)
old=json.loads((R/'experiments/r15_llm_schedule_matched/PROTOCOL.json').read_text())
old.update(experiment_id='r16_fe200_llm_schedule',date='2026-10-28',online_admission_start_fe=200)
old['execution']['remote_baseline_required_before_execution']='baseline/r16-fe200-llm-schedule-20261028'
old['execution']['experiment_branch_required_before_execution']='experiment/r16-fe200-llm-schedule-20261028'
old['implementation_requirements']=[s for s in old['implementation_requirements'] if not s.startswith('Default S02_L05 must reproduce')]
old['implementation_requirements'].append('Agent active/history remains FE210; FE200-208 online selection uses the same pre-active numerical rule as R14.')
old['offline']['case_signature_fields'].append('online_admission_start_fe')
old['status']='plan_prespecified_not_executed'
(E/'PROTOCOL.json').write_text(json.dumps(old,indent=2,ensure_ascii=False)+'\n')
(E/'BASELINE_SNAPSHOT.json').write_bytes((R/'experiments/r15_llm_schedule_matched/BASELINE_SNAPSHOT.json').read_bytes())
with (E/'CONFIGURATIONS.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=old['configs'][0]);w.writeheader();w.writerows(old['configs'])
(E/'.gitignore').write_text('*\n!.gitignore\n')
files=['PROTOCOL.json','BASELINE_SNAPSHOT.json','CONFIGURATIONS.csv','.gitignore']
(E/'PLAN_BINDING.json').write_text(json.dumps({'status':'prespecified_plan','sha256':{s:hashlib.sha256((E/s).read_bytes()).hexdigest() for s in files}},indent=2)+'\n')
replacements={'r15_common':'r16_common','collect_r15':'collect_r16','experiment_r15_schedule':'experiment_r16_schedule','check_r15_schedule':'check_r16_schedule','report_r15_schedule':'report_r16_schedule'}
for src,dst in [('r15_common.py','r16_common.py'),('collect_r15.py','collect_r16.py'),('experiment_r15_schedule.py','experiment_r16_schedule.py'),('check_r15_schedule.py','check_r16_schedule.py'),('report_r15_schedule.py','report_r16_schedule.py')]:
 s=(R/'scripts'/src).read_text()
 for a,b in replacements.items():s=s.replace(a,b)
 if src=='r15_common.py':
  s=s.replace("experiments/r15_llm_schedule_matched'","experiments/r16_fe200_llm_schedule'")
  s=s.replace("'scripts/run_r15_schedule.sh','scripts/publish_r15_records.py'","'experiments/r14_online_entry_timing/variant_model.py'")
  s=s.replace("return dict(CONFIGS[cid],policy_version=POLICY_ID", "return dict(CONFIGS[cid],online_admission_start_fe=200,policy_version=POLICY_ID")
 if src=='collect_r15.py':
  s=s.replace('model, problem, population, rng, seed, function = generated.episode(function_index, seed_index)', 'model, problem, population, rng, seed, function = generated.episode(function_index, seed_index)\n    import experiment_r16_schedule as variant_run\n    variant_run.apply_variant(model)')
 if src=='experiment_r15_schedule.py':
  s=s.replace('def build(arm,mode=', '''def apply_variant(model):
    import roopf.model as frozen
    spec=importlib.util.spec_from_file_location('roopf.r16_variant',ROOT/'experiments/r14_online_entry_timing/variant_model.py')
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    module._is_structured_system_problem=frozen._is_structured_system_problem
    model.__class__=module.ROOPFOptimizer;model.online_entry_fe=200
    return model

def build(arm,mode=''' )
  s=s.replace('model=RUN.legacy.build()\n    journal=', 'model=apply_variant(RUN.legacy.build())\n    journal=')
  s=s.replace("points[:,:55],refs[:,:55]),'Online admitted before frozen210'", "points[:,:50],refs[:,:50]),'Online admitted before FE200'")
  s=s.replace('online_admission_start_fe=210,llm_decision_fe=', 'online_admission_start_fe=200,llm_decision_fe=')
  s=s.replace("                if result['completed_arm_batches']==8400:\n                    from publish_r15_records import publish\n                    publish(final=True)\n",'')
 if src=='report_r15_schedule.py':
  start=s.index('    if len(stats)==24:\n        # Rebuilt default')
  end=s.index('    return dict(configurations=',start)
  s=s[:start]+s[end:]
  s=s.replace('keys=list(data[0])','keys=list(dict.fromkeys(k for row in data for k in row))')
 (R/'scripts'/dst).write_text(s)
print(E)
