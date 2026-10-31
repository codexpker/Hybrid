# 多实体固定Z评价：B工程交付

本阶段实现配置到多实体指标、逐实体官方Aff、20个run聚类统计及封存CSV的实际路径。仅做合成CPU验证；不读取真实TEP数值或标签，不执行训练/GPU。B已通过有限合成CPU独审，验收范围和四份独审报告SHA见[限定验收记录](provenance/multi_evaluation_b_acceptance.md)；不据此声称检测收益。真实TEP评价仍须完整封存和主label release。A/core/train/source_gate保持原字节。

## 操作入口与标签门禁

使用`configs/readout_evaluate.example.json`，将外置路径和全部SHA替换为实际冻结资产。示例中的零SHA不能执行。统一入口已有evaluate分支，本阶段未改CLI或A模块：

```text
python -m hybrid_tsad readout --config configs/readout_evaluate.local.json
python -m hybrid_tsad readout --config configs/readout_evaluate.local.json --execute --allow-labels
```

第一条默认预览。执行需要schema4合同/审批绑定规范operation manifest，且必须明确`--allow-labels`。仍先调用已审A的`verify_label_unlock`，核3600个原始预测银行和7200个固定评分及其实际索引、物理目标、源校准/权重身份。缺bank、损坏、L0、部分分片均在读取标签前拒绝。标签只用于冻结指标；不选模型/阈值/数值版本。固定profile为20fault×run31..50、history100/P5、原960点、故障[160,960)、评分[100,960)，AP背景800/860。

每个实体再次核原始label SHA，要求二值数组与冻结原事件区间逐值一致；不修补/重标标签。新输出目录不可覆盖。处理失败写`evaluation_failure.json`，不写最终`evaluation_manifest.json`；消费者需要最终manifest且不得有失败marker。

## 官方Aff外置适配

`configs/official_affiliation_identity.example.json`固定已有项目独审的6个官方Aff源码SHA，路径可配置。`official_affiliation.load_official_affiliation`校验完整文件集合与每个SHA，加载外部`pr_from_events`，不复制源码到包。缺文件、版本不符、依赖缺失或没有许可来源记录均明确BLOCKED，不回退自写Aff。许可沿既有`UNVERIFIED_EXISTING_LOCAL_DEPENDENCY`披露；不据此声称可再分发/商用。

复用已审`evaluate_fixed_scores`：每实体按真实连续块分别调用官方实现；无GT/官方未定义保留NA及原因。多块没有整体平均；跨400实体整体Aff始终NA（`NO_OFFICIAL_GLOBAL_DEFINITION_ACROSS_ENTITIES`）。不拼接实体制造一个长事件。

## 固定统计估计量

先在同fault/run实体计算每seed的AP和ROC，再平均3seed，最后对20fault×20run等权平均。保存全部方法、seed、实体、fault和run，不选最佳seed。报告3个seed的macro及样本标准差、最差实体/fault/run/seed和完整表。AP不是PR曲线梯形积分。

主差分别为hybrid减temporal、late、history_mean：在同fault/run配对，先均值3seed，再每run跨20fault平均。按20个run簇生成2000次共同有放回抽样，`default_rng(99174)`，各比较使用同一draw矩阵；保存原draw、runID、逐run差、有效性和所有2000结果。报告绝对macro AP差和未校正95%百分位区间，只称探索性run聚类敏感性。未调用PSM时间块/weighted pooled AP重采样。

缺少/重复任何method×seed×fault×run单元拒绝。非有限AP/ROC保留失败表并使相关run共同无效；包含它的draw全比较NA，不重抽、不删坏seed。仅有一个数据集，不能称20个独立数据集或跨域确认。

## 真实产物与原始坐标

`entity_metrics.csv`含AP/ROC、点混淆计数/Precision/Recall、固定阈值、事件匹配、故障前报警/跨故障持续报警、正常观察小时/报警片段率。时间来自冻结3分钟采样合同；无正常评分段/无预测阳性/缺工况元数据都有原因，不猜工况。`event_results.csv`保存完整事件召回、原始边界和延迟点数/分钟；零延迟须结合故障前/跨边界报警标志解释。匹配数量另外与项目已有augmenting-path函数交叉核对。

输出还包括`alarm_events.csv`、`entity_affiliation.csv`、`macro_summary.csv`、`entity_seed_mean.csv`、`seed_summary.csv`、`fault_summary.csv`、`run_summary.csv`、`worst_cases.csv`、`paired_run_intervals.csv`和`failed_entity_metrics.csv`。`paired_run_bootstrap.npz`保留完整共同抽样；每method/seed/entity的`coordinates/*.npz`保存原time_index、逐点评分label、完整事件mask、forecast origin和history/target足迹。每实体1个原事件在18组方法/seed中重复，不能把重复行当独立事件样本。

`evaluation.json`记录固定定义、外部Aff来源、代码SHA和全部指标；最终`evaluation_manifest.json`逐文件记录SHA。结尾再次核消费过的定义、评分、原label和官方源码。上层仍负责原始全标签/元数据真实性及评分域外边界证据，此包装不证明训练史或未知工况。

## 工程验证与未迁移项

`tests/test_run_statistics.py`包含正式20×20×3×6合成指标独立算术核对，以及2fault×2run、36银行/72评分完整配置正路径（外部真实官方Aff）。另测缺格/重复格/非有限共同NA、依赖失败、未封存开锁拒绝和label身份破坏。可选`HYBRID_SYNTHETIC_EVALUATION_ARCHIVE`须指定不存在的新目录，保留纯合成输入/配置/银行/指标及拒绝证据；该fixture绝不代替正式release。

尚未执行真实TEP评价、真实GPU成本验证、U/F适配或论文收益判断；尚无工况分层输入时只标缺失原因。不提供未经官方定义的全局Aff，也不恢复历史driver。本次限定工程独审已完成；真实TEP评价仍须完整封存和主label release，U/F与科研收益判断不在本次验收范围。
