# 最终 ROOPF 的可执行算法说明

本说明对应冻结的最终10-D方法，不把解锁行为列为贡献。实现依据为`roopf/model.py`、`roopf/anchor_backbone.py`和补充实验的固定构造配置。完整源代码是算子坐标变换与27维residual特征的精确定义，不能只用下述高层伪代码替代源码。

## 输入和已训练模块

默认维度10、初始种群100、总预算300（包含初始化）、每轮2点、六算子各6候选、五个ridge成员、ridge系数0.002、RFF数量40、residual权重0.008。主实验固定构造随机种子20260630，然后单独设置每条轨迹的搜索种子。anchor和residual从两个归档checkpoint加载。

**复现需要披露：发布路径只加载anchor和residual权重。** state encoder、portfolio的neural generator、operator gate没有额外训练checkpoint；其构造时随机初始化及标量初始值由固定种子确定。不能将这几个模块写成已经在36函数上训练的另一组模型。它们与已训练anchor是不同对象。

## 在线循环

```text
评估100个初始点；修复并按真实目标值升序排序，初始化archive。
while 已用NFE < B:
    rest = min(2, B - NFE)
    remaining = (B - NFE) / B
    stagnation = clip(连续未改善轮数 / max(1,(B-100)/2), 0,1)
    从当前种群/fitness/预算/停滞构造11维状态并编码。
    六算子生成36点，进行边界修复，计算operator prior。
    在archive上拟合五成员ridge proxy；为候选计算分数，取前rest点。
    冻结anchor产生两个提案。
    若remaining > 0.30：评估anchor的rest个提案。
    否则：保留第一anchor槽位；第二anchor与上述候选短名单重新打分。
          取最低分者，执行下述完整保护测试，决定第二槽位。
    仅评估最终选择的rest个点；更新NFE。
    记录候选是否比旧incumbent/旧种群最差点更好。
    更新算子成功记忆；将真实评估点追加到archive。
    archive长度>256时，拼接其中最好的100点和最近128点（不去重）。
    合并种群并保留目标值最好的100点；更新连续未改善轮数。
```

默认B=300时，NFE=210首次允许竞争，共45次竞争轮。warm-up期间仍计算pool/proxy，但它们不改变真实评估选择。第一anchor槽位的保留是一个评估分配规则，不保证最终轨迹不劣于独立anchor-only：第二槽位会改变未来种群和anchor提案。

## proxy、评分和二次比较

坐标按边界映射到[-1,1]。每个ridge成员的特征为常数、坐标、坐标平方及随机Fourier特征；archive目标按其总体标准差标准化。求解`(Phi^T Phi + .002 I) beta = Phi^T y_z`，求解失败时使用伪逆；预测还原目标单位。mu为五成员预测均值，sigma为成员总体标准差，加`.03 * std(archive_y) * 到archive的最近归一化距离`，并以epsilon为下界。

设r=remaining，s=stagnation，d_best为到`archive[:,0]`的归一化距离除sqrt(dim)，d_archive为到archive的最近距离。注意`archive[:,0]`并非每一轮都保证是当前最优点：archive追加后仅在裁剪时重新排序。

- trust=`(1-r)^2 * (1-.7s) * d_best`。
- novelty=`(.15+.35r+.35s) * d_archive`。
- prior bonus=`.03 * log(prior)`。
- proxy score=`mu - 1.25*.25*sigma + trust - novelty - prior bonus`。

UAV/HPO分支将trust乘.55、novelty乘.65、prior系数改为.06；36点池额外给trust/coordinate/current-to-pbest算子`.030/.020/.010 * std(archive_y)`的分数优惠。第二次短名单比较的prior统一为1，且不触发36点池专有优惠；因此必须重新计算分数，不可直接沿用第一次排名。

普通BBOB分支（非UAV/HPO、非cecf、非嵌入高维）增加结构冲突修正：将proxy score和`trust-novelty-prior bonus`分别减其池最小值、除池总体标准差，得到p和h；增加`.020 * archive_std * clip(h-p,0,3) * clip(p/.75,0,1)`。

启用residual时，对每个候选计算sigmoid输出q，再作本次候选集合内的z标准化。分数减去`.008 * archive_std * clip(z_q,-2.5,2.5)`。q的训练标签是候选是否改善当前incumbent，并非候选是否优于竞争anchor；不能混同两种目标。最终cecf与HPO路径跳过residual，UAV路径启用。原生20-D没有`high_dim > low_dim`标志，不走嵌入高维分支。

## 完整默认保护测试

第二次比较中，索引0为第二anchor，j为最低分候选，a为分数，S为archive目标总体标准差（下限1e-8）。令P=(j>=1)。

```text
safe = P and a[j] < a[0] - .10*S
if residual输出存在:
    z = (q - mean(q)) / max(std(q),1e-6)
    veto = P and z[j]+.35 < z[0] and z[j]<-.25
           and portfolio_seen>=8 and portfolio_best_count<=0
    support = (not P) or z[j]>z[0]+.55 or z[j]>1.0
    rescue = P and support and a[j]<a[0]-.04*S
    safe = (safe and not veto) or rescue
selected = j if safe else 0
```

**状态更新的实情：**默认最终配置没有开启更新`portfolio_seen/portfolio_best_count`所需的三个附加开关，因此两个计数保持0，veto的`portfolio_seen>=8`条件不会成立。residual仍能通过分数和rescue改变决策，但不能宣称默认版本已经执行基于失败历史的veto。本轮记录并说明此事实，没有改变最终方法来使论文叙述成立。

实际生效的成功记忆另有更新：每轮先乘.86；每个portfolio候选相对旧incumbent改善加1、否则进入旧种群加.25、否则减.05，裁剪到[-2,3]。operator权重为gate logits加`.35*成功记忆`后softmax，再与均匀分布以.92/.08混合。它不等价于上述未更新的保护计数。

## 成本与边界

诊断教师评估、同状态反事实及UAV事后几何检查均单独记录，不进入在线archive或主预算。CPU并行耗时不能作为独占延迟；已单独保存串行复测。受控重训的离线成本包括真实训练函数调用与loss重新计算，和300NFE的在线测试口径分开。

任何新训练的20-D模型属于维度专用重训扩展；它既不是10-D权重直接泛化，也不是最终10-Dcheckpoint的替换。源码还包含未启用的历史分支，不应把它们列为最终算法的有效组件。
