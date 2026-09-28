"""Write the final report only after every declared remaining stage is complete."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from archive_remaining import NAMES
ROOT=Path(__file__).resolve().parents[1];DOC=ROOT/'docs/experiments';TABLE=DOC/'remaining_tables'

def wtl(values):
    v=np.asarray(values);return f'{(v<0).sum()}/{(v==0).sum()}/{(v>0).sum()}'

def main():
    for name in NAMES:assert (ROOT/'results'/name/'COMPLETE').exists(),name
    external={}
    for m in ['gp_ei','surr_rlde']:
        d=pd.read_csv(TABLE/(m+'_paired.csv'))
        external[m]={s:wtl(d[d.suite==s].difference) for s in ['coco','cec2017']}
    native=pd.read_csv(TABLE/'native20_eval_20260928_paired.csv')
    nt=['| 20-D完整重训方法相对 | 300 NFE，48条件 | 600 NFE，48条件 |','|---|---|---|']
    for m in native.comparator.unique():nt.append(f"| {m} | {wtl(native[(native.budget==300)&(native.comparator==m)].difference)} | {wtl(native[(native.budget==600)&(native.comparator==m)].difference)} |")
    order=pd.read_csv(TABLE/'training_order_paired.csv')
    ot='；'.join(f'{m}：{wtl(order[order.method==m].difference)}' for m in ['anchor_only','full'])
    timing=pd.read_csv(ROOT/'results/serial_external_timing_20260928/raw_results.csv')
    tt=['| 方法 | 在线耗时中位数（秒） |','|---|---:|']
    for m,s in timing.groupby('method').seconds.median().items():tt.append(f'| {m} | {s:.4f} |')
    training=[]
    for n in ['retrain_d10_original_20260928','retrain_d10_curated_20260928','retrain_d20_canonical_20260928']:
        d=json.loads((ROOT/'results'/n/'COMPLETE').read_text());h=json.loads((ROOT/'results'/n/'history.json').read_text())
        training.append({'name':n,**d,'best_training_epoch':min(h,key=lambda x:x['loss'])['epoch']+1})
    resid=json.loads((ROOT/'results/retrain_residual_d20_20260928/metrics.json').read_text())
    counts={}
    for name in NAMES:
        root=ROOT/'results'/name
        if name.startswith(('remaining_operators','remaining_replay')):
            rows=json.loads((root/'all_rows.json').read_text());counts[name]={'trajectories':len(rows),'main_points':sum(r['actual_nfe'] for r in rows),'diagnostic_points':sum(r['diagnostic_nfe'] for r in rows)}
        elif (root/'raw_results.csv').exists():
            rows=pd.read_csv(root/'raw_results.csv');counts[name]={'trajectories':len(rows),'main_points':int(rows.actual_nfe.sum()),'diagnostic_points':0}
    trajectories=sum(x['trajectories'] for x in counts.values());mainpoints=sum(x['main_points'] for x in counts.values())
    diagnostics=sum(x['diagnostic_points'] for x in counts.values())
    summary={'remaining_online':counts,'new_trajectories':trajectories,'new_main_points':mainpoints,'new_pool_diagnostic_points':diagnostics,
        'including_previous_trajectories':trajectories+9430,'including_previous_main_points':mainpoints+2829000,
        'including_previous_diagnostic_points':diagnostics+7920,'UAV_posthoc_component_checks':1050,
        'controlled_anchor_training':training,'native_residual_training':resid,
        'native_label_generation':json.loads((ROOT/'results/native_residual_logs_20260928/COMPLETE').read_text()),
        'scope_limits':['original historical training command/seed not recovered','new training is one initialization per treatment','benchmark family development history remains','no guarantee of superiority or flight safety']}
    (DOC/'FINAL_COMPLETION_COUNTS_20260928.json').write_text(json.dumps(summary,indent=2))
    text=f'''# ROOPF 全部本轮补充工作：结果与论文结论

本报告将最终10-D ROOPF作为固定主方法，不以历史版本差异作为贡献。原anchor/residual checkpoint保持不变。新增维度和训练对照均独立保存，明确标记为受控重训。

## 完成范围

本轮新增{trajectories:,}条正式/计时执行轨迹（跨实验有重复对照，并非全部独立样本），主目标评估{mainpoints:,}次；加上上一轮，共{trajectories+9430:,}条、{mainpoints+2829000:,}次。新增同池诊断{diagnostics:,}次，前轮诊断7,920次，均不参与在线优化。UAV的1,050次事后组成/几何复核另列。

此外完成三个80-epoch anchor受控训练（各36函数、batch64、每轨迹300主NFE），以及20-D residual的全池标注和训练。训练目标计算包含父代/候选重新求值，不能与在线300NFE直接混算。完整计数、训练时间、checkpoint哈希在[机器可读清单](FINAL_COMPLETION_COUNTS_20260928.json)。本报告生成前逐一检查了全部15个剩余阶段的完成标记。

## 1. 扩展主比较不支持总体领先

以下是完整方法相对对照的条件均值胜/平/负；不是显著性计数。

| 对照 | COCO10-D新实例，48条件 | CEC2017混合/组合，20条件 |
|---|---|---|
| 同一anchor | 46/2/0 | 20/0/0 |
| CMA-ES | 5/0/43 | 1/0/19 |
| GP-EI | {external['gp_ei']['coco']} | {external['gp_ei']['cec2017']} |
| Surr-RLDE发布策略 | {external['surr_rlde']['coco']} | {external['surr_rlde']['cec2017']} |
| DE | 40/0/8 | 14/0/6 |

ROOPF对自身anchor有稳定增益，但在多数条件上落后于CMA-ES和GP-EI；与学习型方法的比较也有明显任务差异。不能保留“普遍超过强基线”的摘要或结论。Surr-RLDE按上游停止逻辑将内部预算设为200，真实计数严格为300，相关预算状态适配已在协议披露。GP-EI采用20点LHS初始化、ARD Matern5/2核与固定候选搜索规则；它是明确配置的GP-EI实现，不冒称所有BO方法的代表。

图：[七方法排名](remaining_figures/external_seven_methods.pdf)。跨函数不直接平均原始目标值；排名也是描述性汇总。

## 2. 六算子与proxy：有价值，但不能声称所有组件都必要

固定36候选槽位的六种单家族移除与六种单家族组合已完成，每个条件30初始化。完整组合并未稳定优于每一个替代组合，因此证据不支持“六个算子各自都不可缺少”或“当前组合普遍最优”。固定槽位也不保证修复后有36个不同坐标。

同状态池内真实值诊断中，proxy均值与真实值的Spearman中位数：BBOB f9/f11/f15约0.097/0.034/0.157，移位CEC f1/f3/f6约0.999/0.970/0.855。仅随机化第一次pool短名单后，完整方法在三个移位CEC上均值更好，在BBOB f9更好、f11/f15更差。两组实验共同说明排序信息的效用依赖任务，而不是所有后期proxy都准确。

[算子配对表](remaining_tables/operators_paired.csv)、[候选来源](remaining_tables/operator_origins.csv)、[随机短名单对照](remaining_tables/random_shortlist_paired.csv)、[同池诊断](remaining_tables/same_pool_proxy.csv)。

## 3. 保护机制：直接诊断与长期分支都需要收紧主张

原生默认门控对score-only的优势没有得到一致支持。默认实现中`portfolio_seen`没有被相应开关更新，因此依赖它的residual veto不生效；residual仍会通过分数与rescue影响决策。完整公式、有效分支和状态更新见[精确算法说明](ALGORITHM_EXACT.zh-CN.md)，不能把未激活的分支写成已验证贡献。

单次分支回放使用NFE210/240/270，干预前的所有评估点逐点一致。强制换回anchor的490次实际改变中，最终改善129次、持平171次、变差190次；强制采用门控前提案的48次实际改变中，改善14次、持平21次、变差13次。这里“改善”指强制分支优于原始完整轨迹。说明单次干预的长期效应有好有坏，不能由即时目标值直接推导最终伤害，也不能把多个单次效应相加成为完整因果分解。

原核心实验中，有正向portfolio事件下降量的轨迹上，前三大事件的贡献占比中位数为BBOB77.8%、移位CEC50.9%。这是按槽位顺序避免重复计数的描述性下降量，不是相对anchor-only最终收益的因果百分比。

[分支回放表](remaining_tables/single_intervention_replay.csv)、[贡献集中度](remaining_tables/event_gain_concentration.csv)。

## 4. UAV：优化软惩罚目标不等于安全路径

五场景×七方法×30初始化全部完成。完整方法在五个场景上的目标均值均优于同一anchor，但五个场景都落后于CMA-ES；对score-only为0胜、1平、4负。

按明确的事后几何标准（线段对障碍核心/禁飞圆的精确距离与边界），完整方法返回的最低目标点中31/150满足标准，anchor为35/150，CMA-ES为70/150。该标准是补充诊断；原目标使用采样软惩罚，并未强制硬可行性。此数据不能支持真实飞行验证、硬约束满足或部署安全性。

[UAV方法表](remaining_tables/uav_methods.csv)、[目标与几何图](remaining_figures/uav_objective_and_geometry.pdf)。

## 5. 原生20-D与训练顺序对照

20-D从同一anchor结构重训，并在36生成函数上重新收集全池标签、训练native residual；原10-D residual迁移作为单独对照。全部24个COCO函数×实例101/102×10初始化，两种预算。300NFE保持绝对预算，600NFE保持原10-D的30×维数预算比例；没有10-D嵌入替代。

{chr(10).join(nt)}

这是一个训练初始化下的原生20-D扩展。搜索种子的配对区间不包含训练种子的不确定性，不能据此宣称任意高维或所有重训都稳定。全流程训练成本单列，未按20-D测试成绩选checkpoint。

原始36函数与审计后36函数成员集合、实现和参数分布相同，排序不同；10/20-D共72个按函数ID对齐的值/梯度检查完全一致。匹配随机实例、初始化与80epochs后，审计顺序相对原始顺序的48条件均值胜/平/负为：{ot}。这是训练顺序效应，不是“筛选了更优函数成员”的证据。不能再用不存在的成员差异来论证ELA筛选效果。

## 6. 可复现性与成本

补全了完整评分、两次排名、门控布尔式、archive裁剪、任务分支及有效计数；CPU/GPU原方法一致性、teacher日志不改变主轨迹、同池诊断180轨迹逐点一致检查均有记录。历史一个形状兼容但计算不同的模型已被排除；最终重训直接使用论文项目的anchor实现。

外部基线串行复测使用四条件×三新种子，在线耗时如下（包括搜索/拟合和解析目标求值；不含离线训练与checkpoint加载）：

{chr(10).join(tt)}

这些解析目标上的延迟不是昂贵仿真的端到端耗时。GPU训练时间记录了当时共享资源下的实际墙钟时间，不能称为独占GPU性能。原始历史训练的准确命令/种子/耗时仍未恢复；新增训练有完整配方，且不会被冒充为原始历史运行。

## 7. 对三项贡献与审稿意见的最终判断

| 原论文贡献 | 当前可以支持 | 必须收紧或删除 |
|---|---|---|
| 36函数生成与审计 | 可复现函数接口、梯度与训练来源；受控重训 | 全部函数通过阈值、ELA挑选了不同的更优36成员、全流程完全未见测试族 |
| 多算子与在线proxy | 某些任务上排序及融合有用，有算子来源和固定宽度证据 | 每个算子均必要、当前六算子普遍最优、后期proxy总是准确 |
| anchor/proxy/residual保护融合 | 明确的评估分配机制，相对同anchor的收益及完整诊断 | 逐次/长期不退化保证、门控普遍优于score-only、校准成功概率、未生效veto已验证 |

两位评审指出的消融、决策诊断、预算一致性、复杂函数、原生中维、运行时间和可复现描述，本轮均已执行相应补充工作。完成工作不等于证实原有所有主张：结果中仍有负面证据。历史benchmark参与开发/ELA参考这一事实不能被新实例实验抹去；新结果应称为冻结后的实例、复杂景观和维度扩展，不应写成整个研究从未接触测试族。

**结论：最终ROOPF的“融合可改善既有anchor”得到支持，但当前证据不足以支撑“可靠性保证或总体强基线领先”。论文需要按实测结果重写贡献强度与结论，单纯文字润色不能补足这些主张。** 不将不同训练checkpoint或历史版本混入最终模型主表以制造改进。

## 可复现资料

- [剩余实验协议](REMAINING_PROTOCOL_20260928.md)，[前轮完整结果](RESULTS_20260928.zh-CN.md)。
- [逐条审稿对应](REVIEW_RESPONSE_COMPLETION.zh-CN.md)，[英文修订材料](MANUSCRIPT_POSITIONING.en.md)，[复现步骤](REPRODUCE_REMAINING.md)。
- [数据归档](../../artifacts/remaining_20260928/README.md)：全部阶段、原轨迹、checkpoint、训练日志、ZIP与逐文件SHA256。
- Surr-RLDE的机制背景和发布代码：[论文](https://arxiv.org/abs/2503.18060)、[作者仓库](https://github.com/MetaEvo/Surr-RLDE)。
- GP实现依据：[scikit-learn Gaussian processes](https://scikit-learn.org/stable/modules/gaussian_process.html)。
'''
    (DOC/'FINAL_RESULTS_20260928.zh-CN.md').write_text(text)
    print(json.dumps({k:v for k,v in summary.items() if k.startswith(('new_','including_'))},indent=2))

if __name__=='__main__':main()
