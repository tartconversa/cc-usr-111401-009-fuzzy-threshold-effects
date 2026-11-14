# 资料与现有计算约定

## 修订快照

schema_version为整数1，metadata为业务方案字典，collections为业务集合字典。每条记录有非空字符串id和正整数revision。
同集合相同id取最高revision；同id同revision内容不同即冲突，即便存在更高修订也拒绝。deleted=true为撤回，之后更高修订可恢复。
撤回记录不要求携带业务字段。输入顺序与相同记录重试不改变语义；非有限数字拒绝。
Snapshot.rows返回独立副本；input_digest绑定有效修订和全部metadata。输出data_version沿用metadata，不由程序自造资料版本。
既有摘要入口为app.service.inspect(payload)，输出schema_version、input_digest、data_version和summary。

## 业务资料

applications 含person、score、score_day、decision_day、received；score_day不得晚于decision_day。资格由score>=metadata.cutoff定义，不从received反推。outcomes 含person、day、value，结果窗口为相对decision_day的[left,right)，同主体窗口内取均值，只计一次。baseline.analyze 计算门槛两侧人数、领取数及可用主体结果均值，不作局部因果估计。

各集合必要字段：{"applications": ["person", "score", "score_day", "decision_day", "received"], "outcomes": ["person", "day", "value"]}。

## 当前能力

app.baseline.analyze(snapshot)产生已有摘要，app.service.inspect负责绑定输入身份；新增分析可以增设独立入口，已有接口及上述数据语义保持兼容。

## 模糊断点分析（app.frd）

app.frd 在不改动 baseline/service 的前提下增设独立入口 app.frd.analyze_frd(snapshot) 与 app.frd.inspect_frd(payload)，CLI 为 `python -m app.frd <资料JSON>`。估计目标与识别约定：

- 运行变量只取干预前 score，门槛只取 metadata.cutoff 单一门槛；不引入后补评分或其他门槛。资格 Z=1{score>=cutoff} 与实际领取 received 分开建模，LATE=结果跳跃 tau_y / 领取跳跃 tau_d，对象为门槛处依从者。
- 同主体多期结果沿用窗口 [left,right) 内取均值、每主体计一次的语义；结果方程用完整案例，领取方程用全部申请，联合估计在两方程主体交集上对齐。
- 结果跳跃与领取跳跃用同一主体级加权矩体系联合估计，面包分块对角、肉矩阵保留跨方程得分乘积。由于评分离散，同评分点的多个主体不是独立观测，协方差按评分点聚类计算联合三明治并加 G/(G-1) 小样本校正（Lee & Card 2008），不做“两数相除套正态区间”；簇（评分点）数偏少时区间天然偏宽或集合无界，属于诚实的低精度表现，主估计单元回报 n_score_clusters。
- 弱识别判据为领取跳跃 F<10；主推断为 Anderson–Rubin 二次型接受集合（单工具恰好识别下与同水平 Fieller 集合代数等价，故只报告一次），集合可为有界、双侧无界、整实线或空集（空集表示联合模型在该数据下被拒绝，而非效应为零）；Wald 比值与 delta 法标准误仅作参照。另附门槛两侧分层、按评分点重抽的聚类 bootstrap 比值分位与复现弱识别（F<10）份额。
- 带宽按 Imbens–Kalyanaraman(2012) 离散适配分别为两个方程选取；局部多项式支持一阶/二阶、三角/均匀核。
- 诊断单独成块：离散评分 McCrary 密度跳跃、评分点堆积（heaping）与粒度、门槛两侧前定变量构成跳跃、结果缺失跳跃、带宽内两侧唯一评分点稀疏标志。这些是操纵/堆积/选择性流失的诊断，不作为政策随机化的证明。
- 敏感性覆盖 IK 带宽 0.5–1.5 倍网格、按评分点剔除门槛两侧最邻近 1–2 个点的甜甜圈、以及结果窗口半宽/基准/加倍；每个单元给出实际使用主体数 n_used 与估计目标说明（带宽改变局部权重但目标不变，甜甜圈改变局部依从者人群，窗口改变随访 horizon）。无法估计的单元记录 error 而不中断整体输出。

fixtures/demo_frd.json 为人工合成的可完整运行示例（不对应真实主体）；fixtures/sample.json 的两人数据下主估计按设计返回 not_identified 而非编造区间。
