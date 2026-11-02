# Hybrid V1 实现、训练与系统消融计划

2026-11-02。先由root审定 architecture.md 与 paper_module_map.md，再由未参与开发的Auditor检查设计/源对应，之后Executor整合。常规技术修复与范围内资源决策由root负责，不逐模块向用户确认。本计划不是将所有配置一次性提交训练的许可。

## 1. 工程基线和目录

基于独立hybrid_ad包的本地dae6124；新工作使用codex/hybrid-v1-20261102。旧candidate_*、LEFT、TimeRadar、Autoformer、所有快照和results只作基线，不覆盖或删去。不再一实验一模型driver。

```text
src/hybrid_tsad/hybrid_v1/
  config.py                  # 完整配置与开关
  provenance.py              # 三组外置官方来源及版本
  temporal.py                # 轻量TimesFM因果patch块
  fractional.py              # 原生FRFT encode/decode适配
  lag.py                     # Self/Cross有限lag与索引
  multiview.py               # LEFT多尺度与选链交互
  model.py                   # 统一两pass接口、预测/重建heads
  objectives.py              # mask/未来预测/一致性及MSE-Huber
  scoring.py                 # 分项证据、校准、签发/可用索引
  data.py                    # 保留完整C的多源/entity/run采样
  training.py                # Stage1及Z/U/F参数策略
  cli.py                     # smoke/pretrain/adapt/infer/evaluate
configs/hybrid_v1/           # base、smoke、主要消融、Z/U/F
tests/hybrid_v1/             # 各模块与完整端到端测试
docs/hybrid_v1/              # 当前三份canonical文档及变更
```

布局可按现有共享engine减少重复文件，但不能以legacy外壳冒称已实现训练。构建前记录显式假设，不能在结果后补写成预登记。

## 2. 构建顺序与验收

|阶段|真实实现|验收原件|状态|
|---|---|---|---|
|D0 设计冻结|三份文档、来源/时间/lag/损失/适配定义、职责独立复核|root DESIGN_APPROVAL与review|当前阶段|
|I1 完整模型|三个真实视图＋独立Self/Cross＋交互＋重建/预测；无假模块进入生产路径|小B/多C forward、两loss backward、各块非零更新；原模块数值对照|待设计批准|
|I2 完整训练接口|多变量多源、Stage1、Z/U/F、鲁棒开关、输入权限和索引|纯合成多run端到端，Z零变化/U不读标签/F确用K事件|待I1|
|I3 CPU与A40 L0|已有环境，无大下载；先CPU回归再单卡短smoke|实际日志/退出码/峰内存/成本；测试替身明示，不冒充真模块|待I2|
|E1 完整V1开发|既有公开开发域，完整模型先跑，固定正常/无标签合法验证选模|训练/推理/原分数/覆盖/多指标及错误分析|待L0|
|E2 系统消融|围绕完整V1关闭主要组件，不恢复只做四mode的局部主线|同初始化/曝光/输入权限的配对结果|待E1合法运行|
|E3 Stage1正式预训|多源公开FIT、独立源SELECT/CAL，不是C1采样|来源采样trace、训练目标、checkpoint、seed/资源记录|分波次|
|E4 Stage2 Z/U/F|同一预训练checkpoint分三条权限路径|Z零目标更新、U混合无标签、F事件监督实际更新|分波次|
|E5 污染与泛化|固定污染谱、真实/合成分列、多seed及未参与训练的域/实体|负面案例、最差seed/域、配对差与不确定性|核心验证|

I1/I2必须覆盖完整链和真实外置模块。小型fixture只验证接口，不得输出看似正式检测的数字。I3失败优先修具体实现，不据此否定研究方向；E1单seed没有涨点只限制该开发实例。

## 3. 关键断言

1. `forecast(history,target=None)`接口不接future真值；改未来target、重建pass当前点或另一原batch样本，不得改变受保护预测。
2. 归一化只用允许可见点；TimesFM3原预处理与CPM先后差异已知，不能先放入真实future再mask。
3. 原始窗口足迹、T patch端点、FRFT原生轴、多尺度footprint、每一lag源索引都可追踪。
4. 正/负符号输入的FRFT逆变换，author_abs与signed-real分开；copy不提前池化掉方差。
5. Self只用i=i/正lag，Cross只用i≠j；有界lag无wrap，零lag控制可执行；C1跨变量退化显式。
6. all-missing、padding、短run、不同C和通道置换；source/target不使用固定实体ID参数泄漏。
7. 每项loss与参数组实际梯度/更新，decoder不能仍冻结而声称联合重建；鲁棒开关需真正改变逐元素loss。
8. Z无目标fit/更新；U的标签文件不可达；F的K个独立异常事件真实出现在support，query完整事件不参与训练。
9. 索引覆盖/原连续段、固定阈值、miss删失保留；AP/PR积分分列，不用test-best或PA伪装部署指标。

## 4. 最小系统消融矩阵

完整模型先实现并完成E1，再冻结下列主要比较，不按query结果删掉不利项：

|组|改变因素|主要回答|
|---|---|---|
|A0|原TimeRadar、原LEFT已有结果/合法缓存|原参考，协议不同则明确非同表；不重跑原八基准|
|A1|完整V1|统一方法是否可训练、有何真实优劣|
|A2|去FRFT；固定order=0/1另作控制|自适应时频是否有额外证据|
|A3|long temporal→同容量短历史/轻量时序控制|长上下文和因果编码是否必要|
|A4|Self off、Cross off、全部lag off|单变量历史与跨变量滞后作用分离|
|A5|Cross仅零lag、旧循环lag控制|收益是否真的来自有界异步关系|
|A6|去M多尺度|LEFT结构视图是否必要|
|A7|feature mean、concat-linear、late-score equal、LEFT interaction|交互是否超过简单拼接/等权|
|A8|reconstruction-only、prediction-only、固定双证据；h1与固定多horizon控制|不是靠只改评分凑增益|
|A9|无源预训/单源/多源，同曝光|两阶段与异构来源增益|
|A10|MSE/Huber×污染谱，Z/U/F分列|是否具有污染鲁棒和目标适配价值|

先在一个开发域完成A1–A8接口与必要控制，再选预登记关键消融跨域复核；不将10组全笛卡尔积扩到全部域/seed。原模块和自适配模块的参数量、上下文、优化步数及训练曝光单独列，不声称容量完全匹配，匹配控制按实际差异补齐。

## 5. 数据、污染、少样本与指标

优先复用SMD/MSL/PSM/SWaT/TEP现有缓存与审计。PSM明确为已多轮开发域。正式多源首选SMD源机器、MSL源实体、PSM源训练区间；SMD留出机器及SWaT/TEP目标run依manifest固定。若官方缓存缺实体边界，先使用已可证连续记录，不为凑数据跨拼接边界取窗。

所有这些数据历史上已有接触；“未参与本次源训练”不等于整个研究过程完全盲的独立新数据。LODO/实体划分、训练/适配/校准/query足迹在新协议中冻结，不把旧已见域改名盲确认。无需为此重新做原TimeRadar复现或重新下载数据。

真实公共异常污染来自与query隔离的源事件池；合成污染另列并保留生成操作与边界。0/1/5/10/20%按事件/片段账本构造，学习不读取这些标签；source可靠性与U适配污染是两个轴，先分别再做必要联合。F沿用修复后的事件级K预算、完整覆盖/隔离/正常匹配方案，保留旧M3负结果作为不同表示/协议的历史记录。

评价同时报告AP、梯形PR、ROC、合法固定阈值P/R/F1、官方Aff、原事件召回/一对一Precision、延迟/删失、正常误报/暴露、跨域及成本；Range/VUS在绑定原实现与固定buffer后增加，缺项明确NA，不自造同名指标。正式主比较默认3个训练种子，按独立事件/run/机器配对，不把重叠点当独立样本。主报告包含最差seed/事件和失败案例。

## 6. 分工与资源

- root：唯一总体设计、源与目标权限、波次预算、原始证据验收及结论；常规实现选择不再逐模块问用户。
- Executor：共享包实现、测试、训练、诊断、原件归档；完成/阻断主动事件交接，不只抄指标。
- Independent Auditor：不参与模型开发，审输入足迹、真实模块使用、loss梯度和原分数/指标；同家族共享工作区限制公开，不称外部盲审。
- 文献角色按具体问题接续，不长期无界查新。不开定时器，不为占满4卡增加无科学问题作业。

4×A40按独立seed/config并行，本机3080Ti做轻量测试。训练每卡20GiB/评价30GiB。第一整合CPU不计GPU；A40 L0先给≤0.25GPUh；完整开发首波≤4GPUh，必须依据实测估计再放全量。正式后续新增计划暂分配≤120GPUh（预训45、核心消融35、Z/U/F25、污染/复核15），这是上限规划而非启动许可或实际消耗。开始正式波次前核去重历史账本，始终满足历史累计＋新作业≤300GPUh；不足时缩可选矩阵与重复次数，不删除六项核心目标或虚称预算重置。

## 7. 一个月执行顺序

第1–3天：完整V1设计、原始模块接线、双pass与梯度测试、真实多C smoke。第4–7天：多源数据/Stage1/ZUF最小可执行闭环、完整V1开发及核心消融。第8–14天：三seed正式源预训和关键组件对照。第15–21天：跨域Z/U/F、污染谱、机制正负控制与最差案例。第22–30天：补最必要证据、主表/图、创新与局限核对、初稿。

“微弱”分层和lag变化的受控检验必须事前定义、真实/合成分列，不能用测试模型分数挑弱事件。若某组件当前实现失败，修实现并保留失败，不任意更换主线；若充分有效的系统消融不支持某贡献，缩对应主张，并向用户报告实证原因，不擅自宣布整个研究目标失败。

## 8. I1/I2 唯一接线合同

模型核心与训练工具分别开发时，以以下字段共同实现，不采用旧建议中的“M流替换为cross-lag”或“补旧循环算子全H出口”作为额外架构。

`HybridV1.forecast(history, observed_mask, mask_seed)`只接 `[B,H,C]`，返回 `prediction[B,P,C]`、`valid[B,P,C]`、T/F端点features及轴/统计记录；不能有future或label参数。表示只使用该prefix，不把Rpass缓存传入。

`HybridV1.reconstruct(context, observed_mask, hide_mask, mask_seed)`接当前已到达context，入口统一遮蔽，返回 `tf_reconstruction[B,G,K,W,C]`、`ms_reconstruction[B,G,K,H,C]`、各自可用mask、`hidden_mask[B,G,H,C]`、每组统计和T/F特征。G为固定互补端点组（至多2），K为FRFT副本；目标原值从调用方batch单独送loss/score。每通道末点只取该通道被遮蔽组，不能把另一可见组混进该通道重建平均。

MS另外返回 `ms_reconstruction_by_scale[B,G,K,S,H,C]`；loss和主r_ms对各尺度误差等权，不能以先平均预测掩盖尺度间相反误差。公共ms_reconstruction是聚合预测，供consistency/可视化，不替代by-scale主损失。当前接口只命名稳定语义，具体代码dataclass或dict由Executor统一。

共享batch字典包含history/observed、reconstruction_context/observed、future_targets/future_observed、原current_target/observed及原entity/run/index；F另有support_event_id/labels，Stage1/U对象不得携带异常标签字段。损失返回total及逐窗/逐分项/valid-count，评分返回原索引/available-index/分项及valid，不只有一个浮点数。实现可加dataclass封装，但不改变上述语义。

I1交完整core和原模块等价/遮蔽/变C/梯度测试；I2交可执行多源抽样、MSE-Huber目标、固定分项校准、Z/U/F策略/监督loss及统一CLI。I2要真正从小合成多run数据走通pretrain→分叉Z/U/F→分数/配置/权重保存，不能以NotImplemented或占位head冒完成。真实公开数据大训练仍需后续成本与manifest明确后分波次运行。
