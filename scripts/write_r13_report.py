"""Render a Chinese measurement report from verified R13 records only."""
from pathlib import Path
import csv
import json
from datetime import datetime
from zoneinfo import ZoneInfo
from html import escape
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,PageBreak
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors

ROOT=Path(__file__).resolve().parents[1]
EXP=ROOT/'experiments/r13_cec_uniform'
LABELS={'frozen_baseline':'冻结基线','no_agent':'移除Agent','llm_full':'Full','llm_no_anchor':'移除Anchor','llm_no_online_generators':'移除在线生成算子'}
ORDER=['llm_full','frozen_baseline','llm_no_anchor','llm_no_online_generators','no_agent']


def main():
    result=json.loads((EXP/'FINAL_RESULT.json').read_text());verification=json.loads((EXP/'INDEPENDENT_VERIFICATION.json').read_text())
    assert result['status']==verification['status']=='complete_live_cec_uniform_measurement_verified'
    with (EXP/'FUNCTION_STATISTICS.csv').open(encoding='utf-8-sig',newline='') as f:statistics=list(csv.DictReader(f))
    with (EXP/'FUNCTION_COMPARISONS.csv').open(encoding='utf-8-sig',newline='') as f:comparisons=list(csv.DictReader(f))
    lookup={(int(r['fid']),r['arm']):r for r in statistics}
    pdfmetrics.registerFont(TTFont('ZH','/mnt/c/Windows/Fonts/simhei.ttf'))
    body=ParagraphStyle('body',fontName='ZH',fontSize=10.2,leading=16.5,spaceAfter=9,wordWrap='CJK')
    title=ParagraphStyle('title',parent=body,fontSize=21,leading=29,textColor=colors.HexColor('#193B60'),spaceAfter=14)
    heading=ParagraphStyle('heading',parent=body,fontSize=14,leading=21,textColor=colors.HexColor('#193B60'),spaceBefore=12,spaceAfter=9,keepWithNext=True)
    cell=ParagraphStyle('cell',parent=body,fontSize=8.6,leading=12.8,spaceAfter=0)
    white=ParagraphStyle('white',parent=cell,textColor=colors.white)
    story=[]
    def p(text):story.append(Paragraph(escape(text),body))
    def h(text):story.append(Paragraph(escape(text),heading))
    def table(headers,rows,widths):
        data=[[Paragraph(escape(str(s)),white) for s in headers]]+[[Paragraph(escape(str(s)),cell) for s in row] for row in rows]
        t=Table(data,colWidths=widths,repeatRows=1,hAlign='LEFT');t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#193B60')),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.HexColor('#F2F6FB'),colors.white]),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7)]));story.extend([t,Spacer(1,10)])
    stamp=datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y年%m月%d日')
    story.append(Paragraph('CEC统一流程：完整测量结果',title));p('R13｜'+stamp+'｜真实LLM、冻结基线与三项整模块消融')
    p('本轮按用户指定只重测6个CEC-style函数。Full采用与BBOB一致的主流程：解除CEC专属结构排除，启用后期生成控制、Residual、候选扩展、案例准入、真实LLM及保护选择。三个消融在相同执行策略下分别移除对应模块；目标函数实现、函数名、边界、权重、案例与门限保留。冻结基线继续使用独立CPU代码和既定结构锁。')
    table(['项目','本轮记录'],[['规模','6函数；每条件600条轨迹；5条件共3000条轨迹'],['维度与预算','D10；每轨迹300 FE，含初始100；每轮2点'],['LLM时刻','FE=220／240评价前；融合当轮起5轮，生成下一轮起4轮'],['配对','同一初始种群及求解器随机流；300个条件批次、60,300个目标评价随机流边界逐次核验'],['核验','冻结基线全部目标值与冻结记录一致；新条件实际选点与决策记录对应；360个请求事件逐一映射']], [88,427])
    h('Full对照与整模块消融')
    ordered=sorted(result['comparisons'],key=lambda r:ORDER.index(r['reference']))
    table(['比较','函数胜／平／负','配对轨迹胜／平／负'],[[r['comparison'],f"{r['function_wins']}／{r['function_ties']}／{r['function_losses']}",f"{r['trajectory_wins']}／{r['trajectory_ties']}／{r['trajectory_losses']}"] for r in ordered],[205,135,175])
    p('胜平负均从Full角度统计，目标越小越好。函数胜负按100条轨迹终值均值比较，轨迹胜负逐条配对；这些计数属于描述性比较，不等同于显著性检验。')
    analysis_path=EXP/'POSTRUN_ANALYSIS.json'
    if analysis_path.exists():
        analysis=json.loads(analysis_path.read_text());early=sum(r['full_final_reached_by_FE210'] for r in analysis['timing'])
        p(f"Full执行{analysis['full_nondefault_actions']}个非默认动作；600条轨迹中{early}条在FE210前已达到其本轮最终已测最优值。相对基线唯一改善为CEC3轨迹99，其两次动作均为matched38（一次有效返回、一次失败回退），因此不能将这点微小改善归因于LLM主动调参，也不能据此宣称明显整体增益。")
    story.append(PageBreak());h('逐函数终值：均值、总体标准差与总体方差')
    p('每条轨迹先取300个已评价值中的最小值，再在100条终值之间计算均值、总体标准差及总体方差（ddof=0）。Excel数值明细继续按算法分类，只列均值和方差。')
    for fid in range(1,7):
        if fid in (3,5):story.append(PageBreak())
        h(f'CEC-style {fid}')
        table(['条件','均值','总体标准差','总体方差'],[[LABELS[a],f"{float(lookup[fid,a]['mean']):.10e}",f"{float(lookup[fid,a]['population_std']):.10e}",f"{float(lookup[fid,a]['variance']):.10e}"] for a in ORDER],[122,131,131,131])
        rows=[r for r in comparisons if int(r['fid'])==fid]
        p('Full配对胜／平／负：'+'；'.join(LABELS[r['reference']]+f" {r['wins']}／{r['ties']}／{r['losses']}" for r in rows)+'。')
    story.append(PageBreak());h('真实API执行与回退')
    api=result['api']
    table(['条件','请求','全有效／部分有效／失败','有效行／回退行'],[[LABELS[a],v['attempts'],f"{v.get('ok',0)}／{v.get('partial',0)}／{v.get('failed',0)}",f"{v['valid_rows']}／{v['fallback_rows']}"] for a,v in api['per_arm'].items()],[140,55,170,150])
    p(f"总请求{api['attempts']}次，有效响应{api['valid_responses']}次，失败{api['failures']}次；有效动作行{api['valid_rows']}，回退行{api['fallback_rows']}。所有实际请求已映射到执行记录，每个函数的每个LLM条件均有真实有效返回。失败不自动重试。")
    p(f"记录输入{api['prompt_tokens']:,} tokens、输出{api['completion_tokens']:,} tokens。供应商usage.credit字段合计{api['provider_credit_sum']}，仅覆盖实际返回该字段的{api['responses_with_credit']}条响应；不推断币种或完整账单金额。")
    h('实际评价与模块路径证据')
    table(['条件','原生／缓存FE','相对基线改变的坐标数','Residual前向次数'],[[LABELS[a],f"{result['evaluation_counts'][a]['native_fe']:,}／{result['evaluation_counts'][a]['cached_fe']:,}",result['actual_coordinate_changes'][a]['coordinates_changed_vs_baseline'],result['route_counts'][a]['residual_forward']] for a in ORDER],[120,145,140,110])
    p('坐标变化按每轨迹100轮×2个评价位置与冻结基线逐项比较，不包含相同的初始100点；它与Agent在相同候选池上的改点指标不同。精确目标缓存要求形状、坐标和评价前随机状态均匹配，并恢复评价后随机状态；缓存不替代LLM请求。')
    p('物理移除检查：移除Anchor条件的Anchor前向为0；移除在线生成算子条件的六算子调用为0；移除Agent条件没有Agent决策或API请求。有效默认动作、仅默认allowed和实际不改点均按实记录，不强制产生干预。')
    h('范围与数据文件')
    p('本地6个CEC-style函数不代表完整官方CEC测试集。Frozen基线使用CPU，新条件使用相同CUDA代理，数值后端差异包含在对照内。消融固定Full案例记忆，未针对移除模块重新标注收益。论文其他算法仅沿用归档数值，偏移一致性仍未完全确认，参考排名不等同于本轮复测。')
    p('FUNCTION_STATISTICS.csv：全部均值、总体标准差、方差与坐标变化；FUNCTION_COMPARISONS.csv及PAIRED_TRAJECTORIES.csv：逐函数和逐轨迹对照；API_EXECUTION_AUDIT.json及API_REQUESTS.csv：实际请求、回退、tokens与credit；CODE_BINDING.json及FROZEN_PARENT.json：新轮源码身份与冻结证据。')
    def footer(canvas,doc):
        canvas.setFont('ZH',8);canvas.setFillColor(colors.HexColor('#66758A'));canvas.drawString(40,20,'R13｜CEC统一流程完整测量');canvas.drawRightString(555,20,str(doc.page))
    target=ROOT/'exports/CEC统一流程重测_R13_完整结果报告.pdf'
    SimpleDocTemplate(str(target),pagesize=(595.28,841.89),leftMargin=40,rightMargin=40,topMargin=35,bottomMargin=35,title='CEC统一流程重测：完整结果报告',author='ROOPF').build(story,onFirstPage=footer,onLaterPages=footer)
    print(target)


if __name__=='__main__':main()
