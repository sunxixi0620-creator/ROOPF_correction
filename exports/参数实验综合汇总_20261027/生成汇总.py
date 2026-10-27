"""Recompute and collate R14/R15 parameter studies; no solver or API execution."""
from pathlib import Path
import csv,json,hashlib,math,collections
import numpy as np
from openpyxl import Workbook,load_workbook
from openpyxl.styles import Font,PatternFill,Alignment,Border,Side
from openpyxl.formatting.rule import ColorScaleRule
from openpyxl.utils import get_column_letter
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
R14=ROOT/'experiments/r14_online_entry_timing';R15=ROOT/'experiments/r15_llm_schedule_matched'
SOURCES=['experiments/r14_online_entry_timing/PROTOCOL.json','experiments/r14_online_entry_timing/FINAL_RESULT.json',
 'experiments/r14_online_entry_timing/INDEPENDENT_RESULT_ANALYSIS.json','experiments/r14_online_entry_timing/FUNCTION_STATISTICS_AND_COMPARISONS.csv',
 'experiments/r14_online_entry_timing/PARAMETER_RANKINGS.csv','experiments/r15_llm_schedule_matched/PROTOCOL.json',
 'experiments/r15_llm_schedule_matched/FINAL_RESULT.json','experiments/r15_llm_schedule_matched/INDEPENDENT_FINAL_AUDIT.json',
 'experiments/r15_llm_schedule_matched/INDEPENDENT_FUNCTION_STATISTICS.csv','experiments/r15_llm_schedule_matched/INDEPENDENT_PARAMETER_RANKINGS.csv',
 'experiments/r15_llm_schedule_matched/INDEPENDENT_COMPARISONS.csv','experiments/r15_llm_schedule_matched/FACTORIAL_AND_ONE_FACTOR_VIEWS.csv']
def read(p):return json.loads(Path(p).read_text())
def csvread(p):
 with Path(p).open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def csvwrite(p,rows):
 with Path(p).open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
FUNCTIONS=[(g,f) for g,n in [('BBOB',24),('CEC-style',6),('UAV',5)] for f in range(1,n+1)]
COHORTS={'primary34':[x for x in FUNCTIONS if x!=('BBOB',20)],'all35':FUNCTIONS,
 'BBOB24':[x for x in FUNCTIONS if x[0]=='BBOB'],'BBOB23':[x for x in FUNCTIONS if x[0]=='BBOB' and x[1]!=20],
 'CEC6':[x for x in FUNCTIONS if x[0]=='CEC-style'],'UAV5':[x for x in FUNCTIONS if x[0]=='UAV'],'BBOB20':[('BBOB',20)]}
CN={'primary34':'主分析34函数','all35':'全35函数（补充）','BBOB24':'BBOB24（完整）','BBOB23':'BBOB23（主口径）','CEC6':'CEC6','UAV5':'UAV5','BBOB20':'BBOB20（单列）'}
def fid(g,f):return {'BBOB':'BBOB','CEC-style':'CEC','UAV':'UAV'}[g]+f'{f:02d}'
A14=[f'fe_{n}' for n in (100,130,150,180,210,230,260)];A15=list(read(R15/'PROTOCOL.json')['configs']);M15={r['config_id']:r for r in A15};A15=list(M15)
final14=read(R14/'FINAL_RESULT.json');final15=read(R15/'FINAL_RESULT.json');audit15=read(R15/'INDEPENDENT_FINAL_AUDIT.json')
assert final14['status']=='complete_live_timing_measurement_verified'
assert final15['status']=='complete_live_schedule_measurement_verified'
assert audit15['status']=='complete_live_schedule_measurement_independently_verified'
raw14=csvread(R14/'FUNCTION_STATISTICS_AND_COMPARISONS.csv');raw15=csvread(R15/'INDEPENDENT_FUNCTION_STATISTICS.csv')
D14={(r['group'],int(r['fid']),r['arm']):r for r in raw14 if r['reference']=='frozen_final'}
D15={(r['group'],int(r['fid']),r['config_id']):r for r in raw15 if r['config_id'] in M15}
assert len(D14)==245 and len(D15)==840
for lookup in (D14,D15):
 for r in lookup.values():
  assert int(r['n'])==100
  assert math.isclose(float(r['population_std'])**2,float(r['variance']),rel_tol=1e-12,abs_tol=1e-30)
# The two reports of R15 must agree on all 840 distributions and class ranks.
standard15=csvread(R15/'FUNCTION_STATISTICS_AND_COMPARISONS.csv')
for r in standard15:
 x=D15[r['group'],int(r['fid']),r['arm']]
 for k in ('mean','population_std','variance'):assert float(r[k])==float(x[k])
# The shared Full and independent baseline distributions must be identical across both rounds.
reference_checks=[]
for g,f in FUNCTIONS:
 for b in range(10):
  with np.load(R14/'results'/f'{g}_f{f:02d}'/f'b{b:02d}/REFERENCES.npz') as x,np.load(R15/'results'/f'{g}_f{f:02d}'/f'b{b:02d}/REFERENCES.npz') as y:
   for k in ('frozen_final_values','original_values'):assert np.array_equal(x[k],y[k])
 reference_checks.append(dict(group=g,fid=f,batches=10,identical=True))

def compute(experiment,arms,lookup,default):
 ranks={};function_rows=[];details=[]
 for g,f in FUNCTIONS:
  means={a:float(lookup[g,f,a]['mean']) for a in arms};best=min(means.values());best_ids=[a for a in arms if means[a]==best]
  rr={a:sum(v<means[a] for v in means.values())+(sum(v==means[a] for v in means.values())+1)/2 for a in arms}
  ranks[g,f]=rr
  function_rows.append(dict(experiment=experiment,group=g,function=fid(g,f),function_id=f,**rr,
    best_parameters=';'.join(best_ids),all_parameters_equal_mean=len(set(means.values()))==1,default_parameter=default,default_rank=rr[default]))
  for a in arms:
   r=lookup[g,f,a];details.append(dict(experiment=experiment,parameter=a,group=g,function=fid(g,f),n=100,
     mean=float(r['mean']),population_std=float(r['population_std']),variance=float(r['variance']),function_rank=rr[a],is_best=means[a]==best))
 class_rows=[]
 for cohort,fs in COHORTS.items():
  avg={a:float(np.mean([ranks[g,f][a] for g,f in fs])) for a in arms}
  for a in arms:
   num_equal=sum(v==avg[a] for v in avg.values());ordinal=1+sum(v<avg[a] for v in avg.values())
   class_rows.append(dict(experiment=experiment,cohort=cohort,parameter=a,functions=len(fs),average_rank=avg[a],
     ordinal_rank=ordinal,tied_ordinal=num_equal>1,best_or_tied_best_functions=sum(float(lookup[g,f,a]['mean'])==min(float(lookup[g,f,c]['mean']) for c in arms) for g,f in fs)))
 return class_rows,function_rows,details
C14,F14,V14=compute('R14',A14,D14,'fe_210');C15,F15,V15=compute('R15',A15,D15,'S02_L05')
# Cross-check pre-existing rankings rather than changing ranking definitions during summary.
cohort14={'BBOB':'BBOB23','CEC-style':'CEC6','UAV':'UAV5','primary_34_functions':'primary34','BBOB20_separate':'BBOB20'}
for old in csvread(R14/'PARAMETER_RANKINGS.csv'):
 new=next(r for r in C14 if r['cohort']==cohort14[old['group']] and r['parameter']==old['arm']);assert new['average_rank']==float(old['average_rank'])
cohort15={'BBOB23':'BBOB23','BBOB24':'BBOB24','CEC6':'CEC6','UAV5':'UAV5','primary_34_functions':'primary34','BBOB20':'BBOB20'}
for old in csvread(R15/'INDEPENDENT_PARAMETER_RANKINGS.csv'):
 new=next(r for r in C15 if r['cohort']==cohort15[old['group']] and r['parameter']==old['config_id']);assert new['average_rank']==float(old['average_rank'])

comp=[]
for cohort,fs in COHORTS.items():
 for a in A14:
  for ref in ('frozen_final','contemporary_210','frozen_original'):
   rr=[r for r in raw14 if r['arm']==a and r['reference']==ref and (r['group'],int(r['fid'])) in fs]
   comp.append(dict(experiment='R14',cohort=cohort,parameter=a,reference=ref,functions=len(fs),
    function_wins=sum(float(r['mean_difference'])<0 for r in rr),function_ties=sum(float(r['mean_difference'])==0 for r in rr),function_losses=sum(float(r['mean_difference'])>0 for r in rr),
    paired_wins=sum(int(r['wins']) for r in rr),paired_ties=sum(int(r['ties']) for r in rr),paired_losses=sum(int(r['losses']) for r in rr)))
 for a in A15:
  for ref in ('frozen_final','matched_default','frozen_original'):
   source_ref='original' if ref=='frozen_original' else ref
   rr=[r for r in standard15 if r['arm']==a and r['reference']==source_ref and (r['group'],int(r['fid'])) in fs]
   # Standard report calls the independent reference frozen_original.
   if not rr and ref=='frozen_original':rr=[r for r in standard15 if r['arm']==a and r['reference']=='frozen_original' and (r['group'],int(r['fid'])) in fs]
   assert len(rr)==len(fs),(a,ref,cohort,len(rr))
   comp.append(dict(experiment='R15',cohort=cohort,parameter=a,reference=ref,functions=len(fs),
    function_wins=sum(float(r['mean_difference'])<0 for r in rr),function_ties=sum(float(r['mean_difference'])==0 for r in rr),function_losses=sum(float(r['mean_difference'])>0 for r in rr),
    paired_wins=sum(int(r['wins']) for r in rr),paired_ties=sum(int(r['ties']) for r in rr),paired_losses=sum(int(r['losses']) for r in rr)))
for r in comp:
 assert r['function_wins']+r['function_ties']+r['function_losses']==r['functions']
 assert r['paired_wins']+r['paired_ties']+r['paired_losses']==100*r['functions']
# Full factorial, and the two fixed-factor views (ranks recomputed within each subset).
views=[]
for view,arms in [('固定融合5轮',[a for a in A15 if M15[a]['fusion_rounds']==5]),('固定调用220_240',[a for a in A15 if M15[a]['timing_group']=='S02'])]:
 for cohort,fs in COHORTS.items():
  for a in arms:
   rank=[]
   for g,f in fs:
    means=[float(D15[g,f,c]['mean']) for c in arms];m=float(D15[g,f,a]['mean']);rank.append(sum(v<m for v in means)+(sum(v==m for v in means)+1)/2)
   views.append(dict(view=view,cohort=cohort,parameter=a,functions=len(fs),average_rank=float(np.mean(rank)),comparison_parameters=len(arms)))
# Compare subset views to the completed official report, excluding newly added all35.
for old in csvread(R15/'FACTORIAL_AND_ONE_FACTOR_VIEWS.csv'):
 if old['view']=='factorial_6x4':continue
 co={'BBOB':'BBOB23','CEC-style':'CEC6','UAV':'UAV5','primary_34_functions':'primary34','BBOB20_separate':'BBOB20','BBOB24_supplementary':'BBOB24'}[old['group']]
 vv='固定融合5轮' if old['view']=='timing_at_L05' else '固定调用220_240'
 new=next(r for r in views if r['view']==vv and r['cohort']==co and r['parameter']==old['arm']);assert new['average_rank']==float(old['average_rank'])
for name,rows in [('R14_分类参数排名.csv',C14),('R15_分类参数排名.csv',C15),('R14_逐函数排名.csv',F14),('R15_逐函数排名.csv',F15),
 ('逐函数测量值及排名.csv',V14+V15),('分类对照胜平负.csv',comp),('R15_固定因素比较.csv',views)]:csvwrite(OUT/name,rows)

def best(rows,c):
 rr=[r for r in rows if r['cohort']==c];m=min(r['average_rank'] for r in rr);return [r for r in rr if r['average_rank']==m]
summary=dict(status='both_completed_experiments_reconciled',protocol=dict(functions=35,trajectories_per_function=100,dimension=10,population=100,budget_FE=300,initial_FE=100,cec_unlocked=True,primary_functions=34,bbob20_separate=True,all35_supplement=True),
 R14=dict(conditions=7,trajectories=24500,API_requests=4900,variable='online candidate admission FE',fixed_LLM_call_FE=[220,240],fixed_fusion_rounds=5,fixed_generation_rounds=4,fixed_case_library=True,best_by_cohort={c:best(C14,c) for c in COHORTS}),
 R15=dict(conditions=24,trajectories=84000,API_requests=16800,variable='LLM call FE and action duration',fixed_online_admission_FE=210,independent_matched_memories=24,best_by_cohort={c:best(C15,c) for c in COHORTS}),
 common_reference_batches_compared=350,common_references_identical=True,
 no_joint_class_specific_combinations_tested=['online100+LLM210/230_L10 for UAV','online150+LLM210/230_L08 for CEC'],
 R15_all_equal_mean_functions=[r['function'] for r in F15 if r['all_parameters_equal_mean']],
 fixed_factor_views=views,source_sha256={s:hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in SOURCES})
write(OUT/'汇总及核对记录.json',summary)

wb=Workbook();wb.remove(wb.active)
NAVY='193B60';BLUE='EDF3F9';GREEN='E1F0E7'
def sheet(name,title,note,headers,rows,widths,freeze='B5',scientific=()):
 ws=wb.create_sheet(name);n=len(headers)
 for i,text in [(1,title),(2,note)]:
  ws.merge_cells(start_row=i,start_column=1,end_row=i,end_column=n);c=ws.cell(i,1,text);c.font=Font(name='微软雅黑',size=17 if i==1 else 10,bold=i==1,color=NAVY);c.alignment=Alignment(wrap_text=True,vertical='center');ws.row_dimensions[i].height=34 if i==1 else 45
 for j,h in enumerate(headers,1):
  c=ws.cell(4,j,h);c.font=Font(name='微软雅黑',bold=True,color='FFFFFF',size=11);c.fill=PatternFill('solid',fgColor=NAVY);c.alignment=Alignment(wrap_text=True,vertical='center',horizontal='center')
 ws.row_dimensions[4].height=34
 for i,row in enumerate(rows,5):
  for j,v in enumerate(row,1):
   c=ws.cell(i,j,v);c.font=Font(name='微软雅黑',size=11,color=NAVY);c.alignment=Alignment(horizontal='left' if j==1 else 'center',vertical='center',wrap_text=True)
   if i%2:c.fill=PatternFill('solid',fgColor=BLUE)
   if isinstance(v,float):c.number_format='0.0000000000E+00' if j in scientific else '0.0000'
  ws.row_dimensions[i].height=26
 for j,w in enumerate(widths,1):ws.column_dimensions[get_column_letter(j)].width=w
 ws.freeze_panes=freeze;ws.auto_filter.ref=f'A4:{get_column_letter(n)}{ws.max_row}';ws.sheet_view.showGridLines=False
 ws.sheet_properties.pageSetUpPr.fitToPage=True;ws.page_setup.orientation='landscape';ws.page_setup.paperSize=ws.PAPERSIZE_A4;ws.page_setup.fitToWidth=1;ws.page_setup.fitToHeight=0;ws.print_title_rows='1:4'
 return ws

def label(a):
 if a.startswith('fe_'):return a[3:]+' FE'+('（默认）' if a=='fe_210' else '')
 c=M15[a];return f"{a}｜{c['first_call_fe']}/{c['second_call_fe']} FE｜{c['fusion_rounds']}/{c['generation_rounds']}轮"
def wtl(r,prefix='function'):return '/'.join(str(r[prefix+'_'+k]) for k in ['wins','ties','losses'])
def comparison(e,a,c,ref):return next(r for r in comp if r['experiment']==e and r['parameter']==a and r['cohort']==c and r['reference']==ref)
protocol_rows=[['共同设置','35函数＝BBOB24＋CEC6＋UAV5；D10；种群100；每函数100轨迹；总预算300FE含初始100FE'],
 ['R14：在线候选准入','100、130、150、180、210（默认）、230、260 FE；固定LLM220/240FE、融合5轮、生成4轮和案例库'],
 ['R15：LLM时序与窗口','6组调用时机×L=3/5/8/10；每组独立匹配案例库；固定在线候选210FE准入；生成下一轮起作用L−1轮'],
 ['主分析与补充口径','预先约定34函数主分析排除BBOB20；保留完整BBOB24及全35函数补充，不在结果产生后更改主指标'],
 ['排名','先按每函数100条终值均值排序；数值相等取平均秩，再在问题集合内平均；R14只在7组内排，R15只在24组内排'],
 ['胜平负','全部从当前参数配置角度统计；函数胜负按均值，轨迹胜负使用配对终值；等号为数值精确相等'],
 ['均值与方差','逐函数测量值按参数配置分类；方差为总体方差ddof=0；总体标准差及执行统计见CSV'],
 ['并列与冠军数','逐函数冠军包含并列，冠军函数数累计可超过函数总数；R15有15个函数各配置均值完全相同'],
 ['结论范围','两轮均为现有测试套件上的描述性调参；分类最优值不能未经联合测量就拼成已验证的新配置']]
sheet('口径与结论','两轮参数实验综合汇总','两轮实验均已完成；逐批核对350批冻结Full与独立基线，参照结果完全一致；本汇总不修改实验代码、参数或观测。',['项目','说明'],protocol_rows,[28,115])
ws=wb['口径与结论']
for i in range(5,ws.max_row+1):ws.row_dimensions[i].height=44
for exp,rows,arms,lookup in [('R14',C14,A14,D14),('R15',C15,A15,D15)]:
 data=[]
 for r in sorted(rows,key=lambda r:(list(COHORTS).index(r['cohort']),r['average_rank'])):
  ref='contemporary_210' if exp=='R14' else 'matched_default';cf=comparison(exp,r['parameter'],r['cohort'],ref)
  data.append([CN[r['cohort']],f"并列{r['ordinal_rank']}" if r['tied_ordinal'] else r['ordinal_rank'],label(r['parameter']),r['average_rank'],r['best_or_tied_best_functions'],wtl(cf),wtl(cf,'paired')])
 sheet(exp+'分类参数排名',exp+'：各类测试问题的完整参数排名','冠军函数数包含并列；不同参数轮次的平均秩数值仅在本轮组内比较。胜平负相对本轮默认组合。',
  ['问题集合','排名','参数配置','平均秩','冠军函数数','函数胜/平/负','配对胜/平/负'],data,[25,12,48,15,16,23,25])
# Main factorial matrix, all35 supplement immediately below in another sheet.
for cohort,name in [('primary34','R15主分析6乘4'),('all35','R15全35函数6乘4')]:
 data=[]
 for s in range(1,7):
  ids=[f'S{s:02d}_L{n:02d}' for n in (3,5,8,10)];c=M15[ids[0]]
  data.append([f"{c['first_call_fe']}/{c['second_call_fe']} FE"]+[next(r['average_rank'] for r in C15 if r['cohort']==cohort and r['parameter']==a) for a in ids])
 w=sheet(name,'LLM调用时机×融合窗口：'+CN[cohort],'数值为在24组内计算的函数内平均秩，越低越好。生成从下一轮起生效L−1轮；在线候选准入固定210FE。',
 ['两次调用时机','融合3轮','融合5轮','融合8轮','融合10轮'],data,[25,18,18,18,18])
 w.conditional_formatting.add('B5:E10',ColorScaleRule(start_type='min',start_color='C6E6D3',end_type='max',end_color='F5C9C5'))
for exp,arms,fr,default in [('R14',A14,F14,'fe_210'),('R15',A15,F15,'S02_L05')]:
 data=[]
 for r in fr:
  winners=r['best_parameters'].split(';');winner='全部'+str(len(arms))+'组并列' if len(winners)==len(arms) else '、'.join(winners)
  data.append([r['group'],r['function']]+[r[a] for a in arms]+[winner,r['default_rank']])
 w=sheet(exp+'逐函数排名矩阵',exp+'：35个函数上的参数排名','每一行独立排名，相同均值采用平均秩。最优列按均值最小选取，包含并列；每函数100条配对轨迹。',
 ['问题类别','函数']+arms+['最优参数（含并列）','默认组合排名'],data,[17,14]+[12]*len(arms)+[60,17],freeze='C5')
 for i,row in enumerate(data,5):
  if len(str(row[-2]))>55:w.row_dimensions[i].height=min(150,18*math.ceil(len(str(row[-2]))/42))
  rr=fr[i-5];bestids=rr['best_parameters'].split(';')
  for a in bestids:
   c=w.cell(i,3+arms.index(a));c.fill=PatternFill('solid',fgColor=GREEN);c.font=Font(name='微软雅黑',bold=True,color='17633D')
 # Long table is easier to filter, while matrices preserve per-function comparison.
 vv=V14 if exp=='R14' else V15
 sheet(exp+'逐函数名次明细',exp+'：逐函数参数名次与统计','每函数100条轨迹；该表保留总体标准差，便于核对。均值方差专表只保留均值与方差。',
 ['参数配置','问题类别','函数','函数内名次','均值','总体标准差','总体方差'],
 [[v['parameter'],v['group'],v['function'],v['function_rank'],v['mean'],v['population_std'],v['variance']] for v in sorted(vv,key=lambda r:(r['parameter'],r['group'],r['function']))],
 [18,17,14,16,25,25,25],scientific=(5,6,7))
 refs=[r for r in raw15 if r['config_id'] in ('frozen_final','original')]
 mv=[[v['parameter'],v['function'],v['mean'],v['variance']] for v in sorted(vv,key=lambda r:(r['parameter'],r['group'],r['function']))]
 mv += [[r['config_id'],fid(r['group'],int(r['fid'])),float(r['mean']),float(r['variance'])] for r in sorted(refs,key=lambda r:(r['config_id'],r['group'],int(r['fid'])))]
 sheet(exp+'均值方差',exp+'：按参数配置分类的具体数值','每函数100条终值；总体方差ddof=0；固定参照frozen_final（确定Full）和original（独立冻结基线）不参与参数组内排名。',
 ['参数配置／参照','函数','最终均值','总体方差'],mv,[26,20,28,28],scientific=(3,4))

sheet('分类对照胜平负','两轮实验：相对确定Full、本轮默认和独立基线的配对比较','胜负均从参数配置角度统计；frozen_final为当前确定Full，frozen_original为独立冻结基线；默认分别为R14 fe_210与R15 S02_L05。',
 ['实验','问题集合','参数配置','对照','函数数','函数胜','函数平','函数负','配对胜','配对平','配对负'],
 [[r['experiment'],CN[r['cohort']],r['parameter'],r['reference'],r['functions'],r['function_wins'],r['function_ties'],r['function_losses'],r['paired_wins'],r['paired_ties'],r['paired_losses']] for r in comp],
 [12,25,18,24,12,12,12,12,14,14,14])
sheet('R15固定因素比较','R15：固定5轮比较调用时机；固定220/240比较窗口','该表在6种调用时机或4种窗口内重新排名，因此不能与24组完整矩阵的平均秩数值直接横比。',
 ['固定条件','问题集合','参数配置','平均秩','组内比较数量'],[[r['view'],CN[r['cohort']],label(r['parameter']),r['average_rank'],r['comparison_parameters']] for r in views],[25,25,48,18,20])
analysis14=read(R14/'INDEPENDENT_RESULT_ANALYSIS.json')
uav=analysis14['uav_fe100_improvement']
sheet('UAV提前准入具体改善','UAV：100FE准入对照当前确定Full的均值变化','固定LLM220/240FE及5轮融合；相对降幅=(Full均值−100FE均值)/Full均值。各函数均值越低越好。',
 ['函数','当前确定Full均值','100FE准入均值','均值降低百分比'],[[f"UAV{r['fid']:02d}",r['final_mean'],r['fe100_mean'],r['relative_mean_reduction_percent']] for r in uav],[18,28,28,25])
metrics=[]
for e,api,credit,creditn,native,cached in [('R14',analysis14['api_totals'],analysis14['provider_credit_units'],analysis14['responses_with_credit'],sum(v['native_fe'] for v in analysis14['execution_metrics'].values()),sum(v['cached_fe'] for v in analysis14['execution_metrics'].values())),
 ('R15',audit15['api'],audit15['provider_credit_units'],audit15['responses_with_credit'],audit15['native_test_fe'],audit15['cached_test_fe'])]:
 metrics.append([e,api['attempts'],api['ok'],api['partial'],api['failed'],api['valid_rows'],api['fallback_rows'],api['prompt_tokens'],api['completion_tokens'],credit,creditn,native,cached])
sheet('API与评价统计','真实API和目标评价执行统计','credit为供应商实际返回的单位，仅在部分响应中提供，不推断币种；原生/缓存FE为测量阶段计数；案例采集阶段不调用LLM。',
 ['实验','API请求','完整有效','部分有效','失败','有效动作行','回退动作行','输入tokens','输出tokens','provider credit','含credit响应数','原生评价FE','缓存评价FE'],metrics,[12,16,16,16,16,18,18,20,20,20,20,20,20])
wb.save(OUT/'参数实验综合汇总.xlsx')
# Validate the generated workbook counts and representative matrix cells.
w=load_workbook(OUT/'参数实验综合汇总.xlsx',read_only=True,data_only=True)
assert w['R14逐函数排名矩阵'].max_row==39 and w['R15逐函数排名矩阵'].max_row==39
assert w['R14均值方差'].max_row-4==315 and w['R15均值方差'].max_row-4==910
assert math.isclose(w['R15主分析6乘4']['D5'].value,float(next(r['average_rank'] for r in C15 if r['cohort']=='primary34' and r['parameter']=='S01_L08')),rel_tol=1e-14)
assert math.isclose(w['R15全35函数6乘4']['C6'].value,10.057142857142857,rel_tol=1e-14)
summary['exporter_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
summary['workbook_validation']=dict(sheets=len(w.sheetnames),R14_function_rows=35,R15_function_rows=35,R14_mean_variance_rows=315,R15_mean_variance_rows=910,rankings_match_original_reports=True,common_frozen_references_identical=True)
write(OUT/'汇总及核对记录.json',summary)
conclusion='''两轮参数实验均已完成，并通过原实验及独立核验。共同设置为35个函数（BBOB24、CEC6、UAV5）、每函数100条配对轨迹、D10、种群100、总预算300FE含初始100FE；神经权重冻结，CEC采用解锁流程。

R14测量在线候选获准竞争真实评价位置的时机，阈值为100、130、150、180、210、230、260FE；LLM固定220/240FE，融合5轮、下一轮起生成4轮，案例库固定。共24500条轨迹、4900次API请求。源码基准为210FE。整体34函数平均秩最优是210FE（3.3824）；全35函数补充口径仍为210FE（3.4286）。分类最优为BBOB24/23：210FE，CEC6：150FE，UAV5：100FE。UAV100FE对确定Full的5个函数全部获胜，配对轨迹398胜、0平、102负，逐函数均值降低7.96%—15.79%。CEC150FE为3胜2平1负、配对97胜415平88负，但并不适合作为全部任务的统一阈值。

R15固定在线准入210FE，在6组LLM调用时机与L=3/5/8/10的24组合中测量，每组配套独立的时序匹配案例库；融合当轮起作用L轮，生成从下一轮起作用L−1轮。共84000条轨迹、16800次API请求。预先约定的34函数主指标最优为210/230FE调用、融合8轮、生成7轮（S01_L08），平均秩10.1765；默认220/240FE、5/4轮（S02_L05）为10.2941、排名第二。最优候选相对确定Full为7胜17平10负、配对443胜2510平447负，相对本轮默认为8胜17平9负；因此它不是稳定支配默认Full的配置。

统计口径会影响R15总排名：包含BBOB20的全35函数补充结果中，默认S02_L05以10.0571排名第一，S01_L08为10.2857、排名第二；该补充口径并未替换预先确定的34函数主分析。BBOB完整24函数上仍是默认S02_L05最优。CEC6最优为S01_L08，但对确定Full只有2胜597平1负的轨迹差异。UAV5最优为210/230FE、融合10轮、生成9轮（S01_L10），对本轮默认5个函数全胜，对确定Full为4胜0平1负。24组在15个函数上的最终均值完全相同，逐函数排名表明确保留并列。

在固定融合5轮的6组调用时机比较中，默认220/240FE最好；在固定220/240FE的4种窗口比较中，默认5轮最好。完整6×4的S01_L08仅有很小的主指标优势，说明应联合考虑时机和窗口，不能将“越早越好”或“越长越好”作为统一结论。

统一默认可继续保留在线210FE准入、LLM220/240FE调用、融合5轮和下一轮生成4轮。S01_L08保留为34函数主指标最优候选。UAV早准入100FE与LLM210/230FE、10/9轮分别在两轮中有优势，但它们拼成的联合方案没有被测量；CEC150FE准入与S01_L08的联合方案也没有被测量。这些是现有测试套件上的描述性调参结果，并非独立确认性显著性结论。

两轮真实API均存在失败并按固定matched38回退：R14完整有效3072、部分有效32、失败1796；R15完整有效10494、部分有效97、失败6209。各参数每函数均有有效API执行返回，但有效返回、默认动作和失败回退的混合均被保留。R15约87.49%的决策上下文只允许默认动作，因此不能把全部参数差异单独归因于LLM非默认选择。

Excel包含：实验口径、完整分类排名、34/35函数6×4矩阵、两轮35函数逐项排名矩阵和名次明细、按参数分类的均值方差、对确定Full/本轮默认/独立基线的胜平负、固定因素比较、UAV逐函数改善、API与评价统计。总体标准差和数值明细也保存在CSV中。两轮平均秩的比较组数不同（7组与24组），其数值不可直接横比。汇总逐批核对了350批共同参照，结果完全一致，没有拼接不同基线。
'''
(OUT/'实验结果与结论.txt').write_text(conclusion,encoding='utf-8')
print(json.dumps(dict(status=summary['status'],workbook=str(OUT/'参数实验综合汇总.xlsx'),validation=summary['workbook_validation'],best_R14=summary['R14']['best_by_cohort']['primary34'],best_R15=summary['R15']['best_by_cohort']['primary34'],best_R15_all35=summary['R15']['best_by_cohort']['all35']),ensure_ascii=False))
