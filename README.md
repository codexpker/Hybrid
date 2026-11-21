# Hybrid TSAD 独立研究工程

## 2026-11-21 受控互补性协议：CPU预检完成，未启动推理

仅设计H1短脉冲、H2持续振荡、H3平滑相对延时及逐点能量匹配对照，不训练、不融合、不改旧CAL或模型。两个公开正常源前缀各固定4块，CPU构造184条记录并逐一重建SHA；H3仅11/16配对通过边际／自相关资格，5项保留NA、不换块。PSM名义正常训练背景不冒充逐点认证健康。所有模型前向和GPU使用均为0，等待主研究决策批准后才可推理。

[简短执行协议与实际资格](docs/experiments/controlled_complementarity_20261121/PROTOCOL.md) · [全部CPU案例](docs/experiments/controlled_complementarity_20261121/ALL_CPU_CASES.csv) · [H3失败项](docs/experiments/controlled_complementarity_20261121/H3_QUALIFICATION.csv) · [成本估算](docs/experiments/controlled_complementarity_20261121/COST_ESTIMATE.csv)。合格子集174条流、每模型311808点；输入身份复用后预计两模型合计约2.95 GPU小时，加浮动与启动约3.8小时，并非已发生或已批准的计算。旧分数与所有负面结论保留。


## 2026-11-21 已完成：冻结缓存事件／九分位数／逐变量误差归因

仅用既有SWaT／PSM缓存，保持模型、原生V、中位数MSE、阈值、事件及负结果不变。发现部分持续段的低预测误差与较高V共存；额外预测触发多为短促偏离，且与正常误报存在相似统计形态。九分位数位置也没有建立正常／异常的清楚区分。它支持继续研究有边界的阶段互补假设，不能证明长历史或变量注意力贡献，更不是已成立的融合增量。完整TimesFM残差基准不等于对TimesFM架构思想的验证或否定。

[完整归因与研究判断](docs/experiments/timesfm3_v_cached_attribution_20261121/RESULTS.md) · [所有事件阶段](docs/experiments/timesfm3_v_cached_attribution_20261121/EVENT_PHASES.csv) · [误差贡献](docs/experiments/timesfm3_v_cached_attribution_20261121/CHANNEL_ERROR_CONTRIBUTIONS.csv) · [原缓存身份与只读记录](docs/experiments/timesfm3_v_cached_attribution_20261121/ANALYSIS.json)。6项CPU单元测试通过、431项同次自洽核验无差异；主统计18.826308秒，表格后处理1.165681秒，模型前向／训练／GPU均为0。全部14个长事件、共有／独有／未命中组和正常误报片段均保留，无新融合或训练任务。


## 2026-11-21 完整官方预训练 TimesFM3 冻结基准完成

使用指定官方源码918df02及HF权重43046b，完整330,710,976参数冻结。每个起点联合全部变量、严格过去512点独立预测一步，唯一评分为源FIT标准化空间的中位数预测MSE；不是官方异常头或Hybrid贡献。SWaT/PSM完整共同QUERY轴AP分别0.093973/0.344512、ROC0.288375/0.566754，均低于复用的原生TimeRadar V。低误报区间及长稳态覆盖也没有建立优势。

保留额外短事件证据：三连后相对V各有2/8个独有any-hit事件；SWaT其中长事件仅1/601点、稳态0，PSM八个均为短事件，不能宣称持续检出或协同创新。严格点与三连、any-hit与一对一匹配、各自CAL误报代价完整分列。CAL仍含SWaT34攻击点、PSM512全攻击，不是正常误报预算认证。

[完整方法/操作点/事件互补与技术决定](docs/experiments/timesfm3_frozen_20261121/RESULTS.md) · [逐方法原精度表](docs/experiments/timesfm3_frozen_20261121/methods.csv) · [局部事件证据](docs/experiments/timesfm3_frozen_20261121/LOCALIZED_EVENT_EVIDENCE.csv) · [模型与缓存SHA](docs/experiments/timesfm3_frozen_20261121/MODEL_AND_CACHE_IDENTITIES.json) · [固定配置](configs/timesfm3-frozen-20261121.json)。402,073个CAL/QUERY点、23个实际GPU作业OS0且清理，实际5.487239 A40 GPU小时；无训练、适配或旧方法重推。完整输出/九分位数/逐通道误差留存，公开仅汇总和哈希。MeanT扩展正式结束，历史产物不改；本轮后无自动融合、适配或后继候选。


## 2026-11-20 固定SMD子集验证完成

只在预定machine-3-2／3-3上新增seed2025的R和MeanT四次适配，noU仅补兼容当前norm0／独立B1协议的必要推理。两实体MeanT相对R的AP分别−0.214／−0.484pp，ROC和两项VUS亦较低；3-3虽优于noU，但R的排序收益更强。保留3-3原CAL下三连误报75→60、匹配事件9→10/16的有限正项，同时披露严格召回下降、低误报三连区间代价及最长漏检未改善。

两个固定QUERY均无>512点完整事件，长稳态宏／合并覆盖记NA，不能补零或宣称已完成长异常推广验证。当前证据不足将MeanT提升为跨任务正式方法主线，继续保留PSM可重复的局部学习结果及所有SWaT/SMD负项；不继续换机器、加种子或调配方。这是已曝光SMD两个实体的单种子验证，不是盲测试或全28台结论。

[完整方法表、MeanT−R／noU及低误报补表](docs/experiments/timeradar_target_mean_smd_20261120/RESULTS.md)、[原始精度CSV](docs/experiments/timeradar_target_mean_smd_20261120/methods.csv)、[模型/缓存身份](docs/experiments/timeradar_target_mean_smd_20261120/MODEL_AND_CACHE_IDENTITIES.json)。18个实际GPU作业成功并清理，新增3.228192A40GPU小时；首轮部署缺已有依赖的CPU导入失败原件保留，无成功计算重跑。[配置](configs/timeradar-target-mean-smd-20261120.json)／[复核记录](docs/experiments/timeradar_target_mean_smd_20261120/KEY_RESULT_REVIEW.json)。

## 2026-11-20 R／R-MeanT三种子固定重复已完成

八次新增适配及16份新银行已完成，seed2025、noU、TT/SS与兼容外部结果全部复用。PSM的R-MeanT−R AP三seed均正，均值+0.891±0.505pp；相对单个冻结noU为+1.029±0.122pp。SWaT AP为−0.427±0.241pp，未建立对应持续检出收益。PSM三连原CAL下召回+5.393±2.816pp伴随正常误报+2.256±2.678pp；两项VUS相对R的代价、事件匹配及noU覆盖差距完整保留。SD均为样本SD(ddof=1)，n=3，是共享官方初态下的适配随机性重复，不是三个预训练模型或独立域。

[完整方法与配对结果、技术决定](docs/experiments/timeradar_target_mean_repeat_20261120/RESULTS.md)、[逐方法均值/SD](docs/experiments/timeradar_target_mean_repeat_20261120/method_summary.csv)、[唯一低误报补表](docs/experiments/timeradar_target_mean_repeat_20261120/LOW_FPR_SUPPLEMENT.md)。strict与C3分别计算0–1%/0–5%经验ROC折线面积及固定1/3/5%下离散可达召回；PSM收益主要在较宽预算，1%与事件匹配反例保留。QUERY曲线只作描述，不更改CAL、选权或训练。SWaT CAL34攻击点、PSM512点全攻击，q99不是正常1%误报保证。

本轮新增实际10.828306A40GPU小时。52实际作业成功并清理；另有1项启动前资源guard拒绝，GPU零消耗，原30读出成功及10项未启动差集恢复均保留。[实际成本](docs/experiments/timeradar_target_mean_repeat_20261120/COMPUTE.csv)、[模型/缓存身份](docs/experiments/timeradar_target_mean_repeat_20261120/MODEL_AND_CACHE_IDENTITIES.json)、[固定配置](configs/timeradar-target-mean-repeat-20261120.json)。候选保留为可重复的有限PSM正项，不扩写为通用保持或污染鲁棒贡献；本组计算完成，无新配方或后继GPU任务。

## 2026-11-20 已完成：目标均值重构／源逐副本重构单因素比较

只改变TimeRadar R目标半批损失，SWaT/PSM seed2025两个模型、4新银行及14GPU作业均完成，旧noU/TT/SS/R和外部缓存全部复用。PSM相对R的AP/ROC为+0.864/+0.774pp，三连长事件宏/合并覆盖42.016/54.195%→53.380/72.179%，但正常误报3.589%→8.924%，两VUS下降；SWaT误报略降、覆盖略降，AP下降而ROC略升，未建立相同增量。保留有限PSM正项及所有代价，不宣称稳定跨任务贡献或污染鲁棒性。

[完整方法表、单因素对照及结论](docs/experiments/timeradar_target_mean_20261120/RESULTS.md)、[固定公式和实现](docs/experiments/timeradar_target_mean_20261120/IMPLEMENTATION.md)、[实际模型/缓存SHA和成本](docs/experiments/timeradar_target_mean_20261120/PROVENANCE.json)。22项相关CPU测试通过，关键独立复算45048项吻合；实际2.747929552 A40 GPU小时，无失败或重跑。所有原始结果保留，不自动扩系数、TT-mean、教师、骨干、种子、数据域或其他候选。

## 2026-11-20 原生TimeRadar头部适配：TT／SS／R固定比较完成

SWaT31/PSM25、seed2025六次适配及12份新银行已完成，旧A/noU完整复用。保留官方权重、SOURCE_FIT scaler与原生末点V，仅更新原重构头；每个原生forward严格B=1，以64窗梯度累积实现匹配批量。PSM R−TT的AP/ROC为+5.426/+2.151pp，持续覆盖部分恢复但误报上升；SWaT未建立相同的有用增量。PSM R相对SS保留排序正项，但覆盖代价仍在；不解释为污染因果或Hybrid原创贡献。38个GPU作业全部成功并清理，实际8.080724358 GPU小时；一次关键独审通过。

[完整方法表、学习方式对照与技术判断](docs/experiments/timeradar_native_v_replay_20261120/RESULTS.md)、[精确方法CSV](docs/experiments/timeradar_native_v_replay_20261120/methods.csv)、[配对CSV](docs/experiments/timeradar_native_v_replay_20261120/learning_pairs.csv)、[模型／缓存SHA与实际成本](docs/experiments/timeradar_native_v_replay_20261120/PROVENANCE.json)。SWaT CAL34攻击点、PSM全攻击CAL限制继续保留；旧38文字已按原CSV更正。停止的B/C及其R保持停止，本轮不自动增加种子、比例、损失、评分或其他后继任务。

## 2026-11-19 原生V条件动态：固定验证完成

公开SWaT31/PSM25、seed2025的A原生V、B历史MLP分位数、C官方TimesFM3轻量模块分位数和D原值预测参照已完成。六次正常源训练、16评分银行及34实际GPU作业全成功，累计2.290003054 GPU小时；13项增量测试通过。B/C未建立超出A的检测增量，C不自动进入R；保留A在SWaT的具体能力、长宏/合并覆盖失衡，以及D在PSM的有限正项。PSM全攻击CAL、外部上下文和统计权限差异完整保留。[完整方法/配对/逐事件结果与曲线](docs/experiments/timeradar_native_v_conditioning_20261119/RESULTS.md)、[实现](docs/experiments/timeradar_native_v_conditioning_20261119/IMPLEMENTATION.md)、[六个项目模型与SHA](docs/experiments/timeradar_native_v_conditioning_20261119/models/MODEL_MANIFEST.json)、[实际运行及缓存身份](docs/experiments/timeradar_native_v_conditioning_20261119/ARTIFACT_MANIFEST.json)。官方TimeRadar权重仅绑定身份，不重新分发；TimesFM只复用官方代码，没有官方预训练权重。旧Energy及其他已停止分支保持停止，未开启适配或新候选。

## 2026-11-19 已完成：冻结TimeRadar响应几何的A/B匹配实验

严格512点历史条件生成器及SWaT/PSM seed2025的A/B、noU/R固定实验已完成，TimesFM3复用官方轻量模块而非官方权重。冻结TimeRadar响应未建立可保持到混合适配后的独立排序增量：A−B AP在SWaT为−1.0594pp/noU、−7.8658pp/R，PSM为+0.0165pp、−0.5707pp。保留SWaT noU显著减少误报但牺牲覆盖的局部取舍，不将其升级为普遍优势。8次训练、16银行、48方法/stream指标及一次职责独立复算完成；正式作业加点检21.396963已完整计量GPU小时，早期profile进退计时缺口另列。本配方不自动扩展，不恢复旧分支。[完整方法表与A/B配对表](docs/experiments/tf_conditional_energy_20261119/RESULTS.md)，[实现与公式](docs/experiments/tf_conditional_energy_20261119/IMPLEMENTATION.md)，[验证与成本](docs/experiments/tf_conditional_energy_20261119/VERIFICATION.md)。仅公开数据；LEFT+R、MLP、CANDI、完整TimeRadar保持比较身份，旧结果不覆盖。

## 2026-11-19 跨骨干TT／R比较完成

只新增SWaT/PSM seed2025的plain MLP TT/R四次适配，源初态、LEFT/CANDI及旧分数全部复用。两个骨干均有R−TT排序正项，但MLP在SWaT仍约96%正常误报，PSM排序更强却持续覆盖显著降低。LEFT保留具体操作与覆盖价值，同时参数／成本更高；这些结果不构成时频机制因果或Hybrid原创贡献。[完整七方法表、骨干内部配对差、曲线与成本](docs/experiments/mlp_replay_20261119/RESULTS.md)，[当前正文](manuscript/next_generation_20261115/draft.md)。本轮基础骨干／普通重放比较已收口，不自动追加MLP种子、污染比例、损失或阈值搜索。既有LEFT＋R两域三种子结论保持锁定。

## 2026-11-18 Fixed exposure replication completed and locked

LEFT+source replay R and matched TT now have three fixed seeds on SWaT/PSM for the same controlled B/C exposure, reusing all A/noU/SS and seed2025 artifacts. Exactly16 new models completed. SWaT protection repeats in direction with lower absolute R-B false alarms than TT-B, but no added events and limited long-event macro coverage. PSM retains different ranking/coverage/false-alarm responses, not a general robustness win. [Complete main and paired tables](docs/experiments/left_replay_exposure_repeat_20261118/RESULTS.md), [per-seed results](docs/experiments/left_replay_exposure_repeat_20261118/PER_SEED_TABLE.md), [fixed curves](docs/experiments/left_replay_exposure_repeat_20261118/curves/CURVE_FINDINGS.md), [updated manuscript](manuscript/next_generation_20261115/draft.md). This fixed contamination experiment is locked; no further rate/seed/loss/module search is launched. Native LEFT and mature replay retain their attribution. P/V and older stopped branches remain historical.

## 2026-11-18 Fixed additional-anomaly exposure with LEFT source replay

Eight seed2025 native LEFT TT/R x B/C adaptations are complete, reusing A/noU/SS. Only the shared target half receives the fixed25,600 donor replacements,5% of all window exposure; source and target-prime plans remain unchanged. SWaT supports a bounded replay retention effect under this pressure condition with false-alarm costs. PSM has no corresponding AP-specific contamination harm and does not establish a cross-task robustness claim. [Full paired results and costs](docs/experiments/left_replay_exposure_20261118/RESULTS.md), [current draft](manuscript/next_generation_20261115/draft.md). P/V remains stopped. Native LEFT and source replay retain their attribution; these results are not relabeled as an original Hybrid mechanism. No successor variant is launched.

## 2026-11-17 当前适配实现分支已完成评估并结束

“冻结TimeRadar编码器＋目标域重构训练＋浅层跨变量适配”已完成评估，未建立广泛稳定增量，不再作为接近定型的论文主方法。停止追加聚合器、冻结策略、正则、蒸馏、评分组合和训练预算搜索。完整TimeRadar、LEFT、原三台SMD重复性正项、全部负项及原始代码/权重/分数/统计保持，不能把成熟基线或历史最好值记为Hybrid贡献。第二篇仍定位方法论文；后续方法决策与本分支明确分开，本次没有提出新候选或启动计算。

[既有草稿中的结果与边界](manuscript/next_generation_20261115/draft.md#completed-adaptation-branch-evidence-17-november-2026)已同步。以下段落为历史阶段状态，不代表当前任务仍在运行或旧方案仍待扩展。

## 2026-11-16 当前：固定选择性通道上下文适配实验

完整[SMD28扩展与同权限LEFT比较](docs/experiments/timeradar_smd_full_left_matched_20261116/RESULTS.md)已经完成并独立复算：原3台可重复AP正项保留，但新增25台Cross相对Self为10胜15负，未建立广泛稳定增量。均值Cross停止扩种、扩域和污染矩阵；完整TimeRadar基础保留。

当前只在适配层将其他变量等权平均替换为接收变量相关的软上下文聚合，保留原生自身残差、骨干、解码器和L1评分。PSM及原3台SMD固定seed2024；旧结果兼容复用，新增参数与等参非跨变量控制、供体信息干预及因果权限边界见[本轮固定定义](docs/experiments/timeradar_selective_context_20261116/README.md)。这是正在验证的结构假设，尚无新的检测收益或无关通道根因结论。

以下阶段记录保留其历史状态，不代表当前自动扩展旧任务。

## 2026-11-16 当前配方三种子重复性：已完成并验收

固定 `9deb6c5` 的完整TimeRadar跨变量适配配方，复用2024全部原件，仅新增2025/2026的24次匹配训练及24次评分。[逐实体逐种子、配对差和PR/ROC曲线](docs/experiments/timeradar_cross_variable_repeat_20261116/RESULTS.md)已完成关键独立复算：三台SMD各三个种子相对等参Self及Head均有AP增量；PSM负项和误报代价保留。建议保持配方进入更广泛固定验证，不判整体NO-GO，也不宣称已取得跨域低误报持续优势。SMD三台机器属于一个数据域，本轮未自动追加新方法或更广矩阵。

## 2026-11-16 当前固定实验：完整 TimeRadar 跨变量残差适配

从 `ab65ff6` 派生 `codex/timeradar-cross-variable-20261116`，复用原官方源码、预训练权重、完整前向和原生方差评分，比较零更新、重构头适配、等参数非跨变量适配和跨变量适配。固定 PSM 与预选三台 SMD 开发实体，不重跑八基准复现。代码与冻结定义见[本轮实验入口](docs/experiments/timeradar_cross_variable_20261116/README.md)。所有原生 TimeRadar 结果属于整窗及批次共享 FRFT 阶数的离线评价；新增适配不是已验证的时滞或污染鲁棒机制。

已完成12次匹配训练、32评分银行及关键独立复算，实际合计2.2982 GPU小时。[完整内外比较与结论](docs/experiments/timeradar_cross_variable_20261116/RESULTS.md)：三台SMD相对等参数Self的AP增加1.281/0.465/0.209个百分点，但PSM排序下降、持续覆盖增加伴正常误报代价，尚未建立跨域误报受控优势。保留正负结果，不自动追加新配方。

最新可用代码持续推送到 `codex/latest`；main、审阅快照及历史正负结果保留。下文均为历史阶段记录，不代表当前任务仍在等待旧阶段授权。

## 2026-11-15 新一代完整方法与论文初稿：实际执行终态

独立研发分支`codex/hybrid-joint-normality-20261115`已完成交替时域/Haar域的可逆联合正常性密度模型、九组真实训练、三域固定评价和关键结果独审。训练与异常评分同为归一化联合NLL，没有恢复已停止预测/遮蔽候选。实际模型见[src/hybrid_tsad/normality_density.py](src/hybrid_tsad/normality_density.py)，配置见[configs/normality-density-20261115.json](configs/normality-density-20261115.json)。

[主验收与完整比较](docs/experiments/normality_density_20261115/RESULTS.md)：PSM排序有真实正项，但持续检测未改善；SWaT接近全部正常点报警；TEP存在覆盖/误报取舍且尚未建立超出简单密度的学习增量。因此该完整候选不进入U/F或继续局部调参，不能作为已验证的论文贡献。44项有界独审通过不等于科研收益成立。四节可编辑稿见[manuscript/next_generation_20261115](manuscript/next_generation_20261115/README.md)。main、原审阅快照及全部历史正负实验保持。

以下日期段落是历史阶段状态，不代表本研发分支当前仍在等待旧审计、复现或任务授权。

## 2026-11-14 有界审计工程修复

`fix/audit-c600125` 从下述固定审阅提交独立派生；复核 F01—F06、必要工程修复及真实测试见[复核结果](docs/audit_c600125/REVIEW_AND_REPAIRS.md)和[测试分组](docs/testing.md)。旧审阅分支、main、历史评分/选权/实验结论不变；本分支没有训练或恢复候选。需要精确历史身份时必须使用原 commit/runtime，不能将修复后的接口冒充旧快照。

## 2026-11-14 全局技术审阅快照

当前分支 `review/hybrid-20261114` 用于独立代码审阅，**不代表已确立有效的最终方法**。Hybrid 是项目所提方法；LEFT、TimeRadar、CATCH 等为成熟参考或外部比较。V1、LEFT 基座适配及已停止候选均保留，不能把历史入口、已有梯度或单个有利指标当作科研收益。

审阅从 [REVIEW_GUIDE.md](REVIEW_GUIDE.md) 开始：其中连接共享代码、外置实验入口、公开结果摘要、版本关系、测试与依赖，以及不能公开的资产边界。`src/hybrid_tsad/` 是当前工程实现；`review_artifacts/` 保存有来源映射的公开实验快照和摘要，未将它们伪装为已完成统一重构的 CLI。

最新两个通道盲点候选均已限定 NO-GO 并停止：包含盲点推理的完整方案未在可接受正常误报下形成稳定收益；仅辅助训练、原生读出的固定 SMD 三种子确认亦未复现稳定学习增量。2026 种子和 PSM/SWaT 的正项原样保留，不能追认为原候选成功。当前暂停自动生成新候选和 GPU 训练，论文类型和总体任务不自动改变。

下面的日期段落是原始工程阶段记录，保留其当时范围；当前审阅状态以上述说明和公开实验索引为准。

这是公开时间序列异常检测研究的独立工程，保留既有 Hybrid 核心、参考方法适配和可复用实验入口。当前按2026-11-02正式纠偏，构建微弱异常与时滞关联异常的完整 Hybrid V1；工程可运行不代表检测收益或创新已成立。

## 当前入口：完整 Hybrid V1

**2026-11-04 状态：**完整 V1、系统消融、F 公平比较、已批准的 H5 污染实验、MSL 外域评价及有限受控弱幅值/Self-lag/Cross-lag 验证均已完成相应独审与主验收。工程更新及已验证边界见[本次进度与结果](docs/hybrid_v1/progress_20261104.md)。这表示这些固定实验完成，不表示污染鲁棒或稳定检测优势已经成立；整体研究仍为 **HOLD**。当前没有因测试结果修改模型架构。

先读[完整架构](docs/hybrid_v1/architecture.md)、[正式论文与实际代码映射](docs/hybrid_v1/paper_module_map.md)、[实现和系统消融计划](docs/hybrid_v1/implementation_plan.md)。共享实现位于 `src/hybrid_tsad/hybrid_v1/`，使用真实外置 TimeRadar FRFT、TimesFM 3.0 小型因果块、LEFT 多尺度与交互子组件，并明确区分项目适配的有限 Self-lag 和 Cross-lag。TimesFM 3.0 是官方技术发布，不冒称已核实的正式顶会论文。旧实现、权重和全部正负结果保留。

统一运行入口见[训练与评分用法](docs/hybrid_v1/tools_usage.md)：

首轮公开多源的角色、三seed、8000次抽样、源验证/校准和权重存档见[Stage1说明](docs/hybrid_v1/stage1_source.md)；原始数据与运行产物保持外置。

2026-11-02 实际进展：完整 V1 约 346 万参数；SMD 完整 38 通道与原生单通道 Sunspot 的三种子预训练各完成 8000 次有效更新，源权重及 3072 条源校准记录已独立审查并由主 Agent 接受。本波含小作业及复核实耗约 1.798 GPU 小时。流式原始银行已通过新旧推理逐输出等价验收；目标域 Z/U/F、系统消融与污染收益尚未完成，源损失下降不能作为检测有效性的结论。

2026-11-03 更新：三种子完整 PSM 零样本读出与两个历史基线已完成原始分数封存和独立复算。V1 AP 为 0.316671±0.001040，Persistence 为 0.295886，History mean512 为 0.322624；V1 strict 事件召回较 History mean512 高，事件 F1 与正常误报则较差，官方 Aff-F1 相对两个控制一升一降。当前是已开发 PSM 域上的混合结果，保持 HOLD，不据此否定完整路线或宣称创新。原始模型读出实耗 6.97844 GPU 小时；固定评价只使用 CPU。详见上一级 `notes/hybrid_v1_20261102/INDEPENDENT_PSM_FIXED_EVALUATION_AUDIT.md` 及 `PRIMARY_PSM_Z1_EVALUATION_ACCEPTANCE.json`。

后续固定阶段已完成：融合控制、四组组件消融、Z/U 目标检测对比、含匹配正常适配及单纯校准的 F 比较、H5 六方法污染比较、MSL 和受控 wave1。U 与 F 结果存在排序、持续检出和正常误报的取舍；H5 的有限 Stage1→Z 结果不等于完整 Stage2-U 污染鲁棒性。目标校准会改变分项尺度和排序，U 必须与同目标校准的零更新模型比较。协议见[Stage2 适配说明](docs/hybrid_v1/stage2_adaptation.md)，验收范围见[本次进度与结果](docs/hybrid_v1/progress_20261104.md)。历史 OS、原分数和完整日志保存在外部研究工作区，本仓库不包含原数据或权重。

```text
python -m hybrid_tsad.hybrid_v1.cli --config configs/hybrid-v1-synthetic.example.json --phase synthetic_smoke --output-dir /new/output
```

命令默认只预览；需绑定本机已有官方源路径及相应运行配置。`--execute` 才实际执行，已有配置示例不代表正式大规模训练已获放行。CPU合成闭环、A40兼容、公开数据科学实验分别验收；最新真实进展在上一级 `CURRENT_RESEARCH_STATE.md`。模型采用源自监督与独立Z/U/F接口，尚不能据代码存在声称污染鲁棒或检测收益成立。

## 保留基线：LEFT官方实现

参见[LEFT基线使用说明](docs/left_baseline.md)，统一入口为 `python -m hybrid_tsad.left_baseline --config ...`。作者代码、公开数据和检查点放在工程之外，本包只提供自有适配与配置；原模型保持固定版本。PSM seed2025完整训练、固定读出和点级/VUS评价均已有限验收（AP 0.576644、VUS-PR 0.572602），没有新方法收益结论。原TimeRadar八基准复评保留，不重新启动。

独立normal-run适配使用 `python -m hybrid_tsad.left_normal_runs --config ...`，默认只预览；正式运行绑定独立协议、来源及批准。当前C2只作为原LEFT的机制能力诊断：LIN/WEAK各seed2025完整正常训练、80固定分数银行及评价均已独审/root接受，没有弱关系稳定收益结论；不改原模型、不接续PSM25权重。透明weak-lag CPU控制保留全部负结果，不能当新的真实公开数据集或已证明的创新。结果导航见上一级 `results/START_HERE.md`，真实终态以 `CURRENT_RESEARCH_STATE.md` 为准。

下方Hybrid工程阶段记录为历史资产，不触发旧训练/适配作业。最新状态在上一级 `CURRENT_RESEARCH_STATE.md`；第二篇继续只使用公开数据。

已有频域强对照使用[CATCH外置官方实现](docs/catch_baseline.md)：共享normal-run训练、固定原生分项捕获与独立CPU评价入口。C2正常训练及80raw已完成并有限复核，完整固定评价也已实际运行；当前没有稳定超过强简单lag对照的证据，不新增模型。两方法训练预算、原生随机mask和回溯时间权限的差异保留，不能把本工程代码入口当论文创新。

先读[研究范围与尚缺部分](docs/research_scope.md)和[官方源码对应表](docs/provenance/references.md)。FRFT本身已含时域信息，新增时域分支的必要性仍须检验；当前多源正常数据预训练不等于已经解决异常污染的无标签学习。

## 历史工程阶段（2026-11-01，旧路线已冻结）

已增加[已知正常数据的预测头适配入口](docs/stage2_u_normal_head.md)：固定现有645参数头，主干、原核心和评分数学不变。三种操作共用`stage2 --config`：1000步正式训练、源/目标阈值校准，以及独立五步丢弃L0；默认都只预览。

[有限独立工程审查](docs/provenance/stage2_independent_review/REVIEW.md)已通过；包括真实三模式CPU步进对照、两初始化合成L0和三个已修复反例。职责独立但同模型家族，不称外部审稿。目标分数[校准修订](docs/provenance/stage2_canonical_revision/README.md)要求从绑定原始预测银行逐点复算，两个阈值轨共用同一canonical分数对象。

该阶段初始记录为“实际GPU L0、正式目标适配与新query尚未执行、服务器SSH超时”；这不是当前运行状态，后续完成与失败记录以上一级状态和原run为准。该控制使用已知正常数据，不是混合异常污染鲁棒或完整U/F成果。旧作业保持冻结，原始TimeRadar复评及已接受的负结果保留。

## 安装与入口

Python ≥3.10。管理 CLI 仅使用标准库，可使用已有 Python：

```powershell
cd research_workspace/hybrid_ad
python -m pip install --no-deps --no-build-isolation -e .
python -m hybrid_tsad --help
hybrid-tsad check-env
python -m hybrid_tsad verify-core --source-root ..
python -m hybrid_tsad config validate configs/core.example.json
python -m hybrid_tsad inspect-run ../results/hybrid_10k_wave3_20261031_v1
python -m unittest discover -s tests -v
```

安装命令只安装本包，不下载模型依赖；需要已有 setuptools≥68。也可临时设置 `PYTHONPATH=src` 使用模块入口。数据接口需要已有 NumPy/Pandas；模型需要已有 PyTorch、torch-frft和外部 TimeRadar 文件自身的依赖。model extra已声明torch-frft（远端已有版本0.8.1）；extra不是完整官方环境锁文件。`check-env --require-model` 在缺少核心依赖时返回1，不导入torch、不探测CUDA。检查到模块存在不等于其二进制/上游依赖可运行。本轮未为现有环境安装任何新依赖。

## 配置与外部资产

`configs/core.example.json` 是可验证的工程库存配置，不是训练合同或新实验计划。复制为 `configs/core.local.json` 后按实际位置填写；路径相对配置文件解析，支持 `${ENV_NAME}` 和用户目录，不硬编码个人服务器目录。默认示例的外部 TimeRadar 文件尚未供应；`config validate ... --check-paths` 会要求两个manifest、合同、官方文件存在，并核验官方SHA。常规validate仅校验结构与模型参数/目标排除，不读取数据。示例种子只为API配置，不代替原wave冻结配置；实际历史运行仍以原run内config、inputs、plan及合同为准。

原数据、manifest、TimeRadar源码、权重、连续分数与运行目录均外置，不能放入本仓库。上游代码/权重/数据许可分别待确认，仓库未附带上游源码、权重或再分发许可；本项目也尚未指定对外开源许可证，不应将可见性视为授权。SHA只验证文件身份，不证明许可或科研正确性。历史manifest内部路径保持原样；迁移磁盘不能仅改库存配置就宣称历史manifest可移植。

## 真正的预测 API

```python
import torch
from hybrid_tsad.api import predictor_from_config, predict

torch.manual_seed(41)
model = predictor_from_config("configs/core.local.json", checkpoint="/external/step10000.pt", device="cpu")
# past_values为已有传感器历史，形状[batch,400,channels]；不传入未来target。
torch.manual_seed(53)
output = predict(model, past_values)
forecast = output["prediction"]  # [batch,5,channels]
```

也可调用 `create_predictor(official_file, mode=..., history=..., copies=..., temporal_depth=..., checkpoint=...)`。API是原 `HybridPredictor` 的薄封装；checkpoint必须是可由 `torch.load(weights_only=True)` 读取的原生state_dict，严格匹配全部键/张量形状。未传checkpoint时为随机初始化，不能当训练后模型。返回原模型输出dict，不增加分数、平滑或阈值策略。TF掩码随机性由调用者管理；固定模型仍需固定RNG才能精确重放。原数据、协议模块可分别通过 `hybrid_tsad.data`、`hybrid_tsad.protocol` 直接导入；保留原接口和限制。

## 已审工程范围与尚缺部分

已审A已推送至`7931ffe`。以下为限定工程状态，不代表真实目标评价完成、检测收益或创新成立：

|范围|状态与入口|说明|
|---|---|---|
|原生源训练|已通过限定独审；`python -m hybrid_tsad train --config configs/train.example.json`|schema3调用包内固定预算训练/源验证循环，100/400历史与plan、源cal、checkpoint metadata绑定；见[原生训练](docs/native_training.md)|
|源L0诊断|已通过限定独审；`python -m hybrid_tsad.engine.source_gate --config /external/source_l0.local.json`|使用外部已批准诊断配置，5个固定batch及mask案例后丢弃模型；入口见[source_gate](src/hybrid_tsad/engine/source_gate.py)|
|多实体固定Z读出A|已通过限定独审；`python -m hybrid_tsad readout --config configs/readout_infer.example.json`|schema4共享source controls、按白名单infer、全组seal和all-bank标签门禁；见[多实体读出](docs/multi_readout.md)|
|多实体评价B|有限合成CPU独审通过；`python -m hybrid_tsad readout --config configs/readout_evaluate.example.json`|逐实体官方Aff外置适配、AP/ROC与固定run聚类统计；见[B评价说明](docs/multi_entity_evaluation.md)及[限定验收](docs/provenance/multi_evaluation_b_acceptance.md)|
|U正常头适配|已有独立Stage2入口；`python -m hybrid_tsad stage2-train --config configs/stage2-train.example.json`|仅固定645参数正常头适配；见[Stage2范围](docs/stage2_u_normal_head.md)，不代表混合污染鲁棒或F完成|
|U读出schema6|v9有限CPU独审通过；`python -m hybrid_tsad stage2-readout --config configs/stage2-readout.infer.example.json`|18U独立身份、9Z正常阈值、单canonical双阈值及全组门禁；见[读出说明](docs/stage2_readout.md)|
|U评价schema7|v2及资源v3有限CPU独审通过；`python -m hybrid_tsad stage2-evaluate --config configs/stage2-evaluation.example.json`|原rank/事件/Aff核与固定run统计，合成300秒/未来TEP900秒上限；见[评价说明](docs/stage2_evaluation.md)|
|旧正常头分支的F接口|未实现于该历史分支|当前完整V1另有事件适配与已验收F公平比较，见上方V1入口；两条实现不混称|

上述命令默认只预览。示例路径/SHA是占位，不是可执行科学合同；`source_l0.local.json`须由外部已批准资产提供。真实执行需要具体合同、规范operation manifest绑定的主批准、固定输入SHA和`--execute`。旧评价入口的`--allow-labels`要求以其命令为准；schema7的`stage2-evaluate`通过独立主label release授权，不接受该旧参数。入口存在不构成执行授权。

A的其余配置为[源controls](configs/readout_controls.example.json)和[全组seal](configs/readout_seal.example.json)，共用`readout --config ...`入口。TEP100完整组要求9权重×400实体的3600个物理预测/RNG银行、6方法×3seed×400实体的7200个固定评分，逐键核验并绑定源校准/权重、原始数组/目标、索引/足迹和分数单位。L0、部分分片、重复或损坏银行均不能获得标签评价资格；门禁已实现，不再列为未来缺项。B有限合成CPU独审已通过；真实TEP评价仍须完整封存和主label release，工程验收不代替真实query放行。

核心model/data/protocol保持原字节。较早共享引擎的单组`infer/calibrate/evaluate`仍保留，见[共享引擎](docs/shared_engine.md)；Z多实体合同使用schema4的`readout`入口，U使用显式独立stage路由。尚未迁入完整prepare/实体准入、作业级精确恢复、通用数据重定位器和F适配，不能据此宣称两阶段科研流程全部完成。

schema6原读出32文件与schema7资源v3评价34文件分别以原字节外置冻结，后者独立目录名为`runtime_evaluation_31141c00_v1`；新增评价模块不回填旧读出运行时。仓库主源码与两个执行快照的身份不同，不能拿当前动态identity替代旧gate。交付身份及候选清单见[交付索引](docs/provenance/stage2_delivery/README.md)，[原字节审查回执](docs/provenance/stage2_review_receipts/COPY_MANIFEST.json)注明同家族、共享工作区、职责独立且非外部双盲。CPU验收不授予真实标签许可：真实读出、完整seal、原冻结runtime实际gate的独审/主接受和独立label release仍按各自实际原件推进，本文不充当实时实验状态或性能报告。

## 历史runner兼容桥

legacy显式桥用于追溯外部冻结runner，不是当前原生train或多实体A的主入口，默认仅输出命令：

```powershell
python -m hybrid_tsad legacy source-wave --script ../scripts/hybrid_10k_wave.py -- --help
python -m hybrid_tsad legacy readout --script ../scripts/run_hybrid_10k_psm_inference.py -- --help
```

readout桥锁定已恢复的implementation_v2版本（ce37f1…），拒绝已知job键冲突的旧v1（3cd868…）。桥接核外部脚本固定SHA，不复制/修改原脚本、run或snapshot；`--`后参数逐字传给原runner，用同一Python启动。执行需显式`--execute`及既有科学合同/预算/主批准，桥本身不替代这些门禁。readout仍依赖原相邻`hybrid_readout_guard.py`，source-wave使用run内implementation快照，不会自动使用本包新代码。旧结果和失败证据保持原位。

## 目录与分阶段验证

`src/hybrid_tsad`为安装包；`configs`为库存/操作配置；`tests`为基础检查与有条件的合成CPU回归；`docs/provenance`为来源及迁移记录。[历史索引](docs/provenance/history_index.md)保留外置资产索引，三核心来源SHA见[core_provenance.json](src/hybrid_tsad/core_provenance.json)。

基础管理CLI不导入torch。可选模型测试需要已有torch和外部官方文件；旧实现/API对照还需`HYBRID_OFFICIAL_FILE`、`HYBRID_LEGACY_MODEL`（同工作区可自动识别旧实现）。条件不足明确SKIP，不算数值验证通过。完整测试集合包含可选合成优化/源验证回归，不是全部无torch或零优化步；工程fixture不计科学训练证据。`verify-core`只证明三核心字节身份；`inspect-run`仅读限定元数据，不加载数组/权重/标签，也不把status当独审验收。

历史测试数字对应各自冻结阶段，不能相加或当作当前全套测试数：

- 初始包阶段：本机11项通过、1项条件跳过；另有三模式旧实现/API、strict state_dict及checkpoint加载CPU逐元素一致，wheel/独立安装检查通过。见[初始迁移](docs/provenance/migration.md)与[初始验证](docs/provenance/validation_results.json)。
- 共享引擎阶段：独立CPU25项通过；对应当时主本机37项中29通过、8跳过。多块Aff整体NA、coverage/连续事件边界和源cal/score-space批准绑定的修复见[共享修订](docs/provenance/shared_engine_revision.md)及[事件边界修订](docs/provenance/adjacent_event_revision.md)。
- 原生train/source_gate/A：均已通过各自限定独审，范围见上表。原循环来源、合成等价/seed报告修复及A原始反例与修订保留在[原生来源记录](docs/provenance/native_training_validation.json)、[seed报告修订](docs/provenance/native_seed_report_revision.md)、[A绑定修订](docs/provenance/multi_readout_a_revision.md)和[A冷启动修订](docs/provenance/readout_cold_cuda_revision.md)；其中早期文件的待审状态是历史记录，不回写成当时已验收。
- B：有限合成CPU独审通过，四份独立复核报告的范围/SHA及旧dist边界见[B限定验收](docs/provenance/multi_evaluation_b_acceptance.md)。[开发验证记录](docs/provenance/multi_evaluation_b_validation.json)保留当时待审状态，不回写历史；真实TEP评价仍须完整封存和主label release，不据合成指标声称检测增益。

物理尺度核验已改为[显式生产后端与执行证据绑定](docs/physical_validation_backend.md)，见[有限主验收](docs/provenance/physical_backend_acceptance.md)。真实TEP seal/evaluate需配置固定CUDA参考；原模型/银行/分数不变，全部400实体仍须具体运行验收，不能用CPU跨后端重算冒作CUDA身份基准。
