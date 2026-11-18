# SWaT seed2025 TT/R × A/B/C：缓存曲线与九长事件

CPU缓存解释性评价；没有导入模型、推理、训练、选择阈值或生成训练/门控输入。只覆盖seed2025，三种子重复结果在上级主表独立报告，不由此图替代。A缓存来自原source-replay，B/C来自current-U exposure。所有输入SHA见`CHECKS.json`，原始归档产物SHA见`ARCHIVED_OUTPUT_SHA256.json`（包括留在外部的完整曲线）；公开文件SHA见`PUBLISHED_SHA256.json`；可复现脚本为`build_curves.py`。

## 一致性与权限

- 六模型共同QUERY轴严格为`[608535,944856)`，336321点（正常288544，攻击47777）。完整CAL轴为`[585624,607320)`，按原规则等间距选512点。每模型独立`q99/higher`，严格`score > threshold`，各自CAL超阈值均5点；阈值和QUERY严格TP/FP、C3计数、AP/ROC-AUC均与已接受METRICS完全一致。
- 原float32通道均值分数、有效性、绝对时间轴及`available_index=495000+floor((index−495000)/192)*192+191`全部核对。全窗分数属于离线回顾，窗口末点才可用；图上横轴为观测点对应的正常FPR，不据此声称在线检测或物理时间延迟。
- CAL操作点为六条曲线上的空心标记：圆TT、三角R；颜色区分A/B/C，实线TT、虚线R。曲线仅解释原工作点，不把图上新阈值送回模型或主表。
- 原定义length>512、steady从原事件start+512的九事件全部保留，全部在此共同轴完整可观测，没有边界截断。其他不完整/不在轴内事件没有被当作九事件的漏检；没有把未评分点补为正常。C3只从第三个连续超阈值点开始，不回填前两点；完整事件与steady覆盖分别保存。不重新计算或混淆any-hit和一对一事件匹配。

## 曲线说明的事实

六模型全ROC-AUC范围约0.880514–0.886953；这个全阈值面积不能确定某个固定CAL工作点的召回。尤其TT_B与TT_C的ROC-AUC为0.885519/0.885894，非常接近，但原CAL工作点正常FPR为1.895%/1.906%，严格召回为10.829%/49.358%，C3召回为8.835%/48.186%。低FPR图显示该区间存在位置不同的陡升段，整体面积接近仍可伴随局部工作点大幅分离。TT_B的AP也降至0.577183，低于TT_A的0.610342和TT_C的0.599753，所以不能把差别全部说成单纯阈值数值漂移。各模型分数尺度不同，不能仅按阈值大小比较松紧。

R_A/B/C的原工作点严格召回分别58.327%/56.914%/58.614%，正常FPR约1.782%/1.814%/1.766%。这些是当前固定缓存的局部表现，不证明重放对任何污染都鲁棒。图表揭示全局排序、低FPR区域和CAL到QUERY转移之间的区别；未识别训练变化、分数分布和校准转移的唯一物理或算法原因。

## 为什么macro与pooled差异很大

九事件steady分母依次为209、185、100、933、35388、659、89、28、1178点，总计38769。E000022独占35388/38769≈91.28%，而macro给每个事件1/9权重。它不是事后挑选的个案：全九事件热图和分母同时展示。

| 模型 | C3 steady macro | C3 steady pooled |
|---|---:|---:|
| TT_A | 10.102% | 65.333% |
| TT_B | 3.450% | 9.216% |
| TT_C | 9.174% | 57.716% |
| R_A | 10.637% | 69.731% |
| R_B | 10.448% | 68.181% |
| R_C | 10.680% | 70.082% |

E000022的C3 steady覆盖约为TT_A/B/C 71.4%/9.9%/63.1%，R_A/B/C 76.2%/74.5%/76.6%。大量其他事件steady覆盖低或为零；高pooled不代表九事件普遍持续检测良好。六模型在E000016、E000026、E000030、E000035的steady覆盖均为零。完整事件覆盖与严格steady覆盖另见全CSV，未用整体指标替代逐事件事实。

## 交付

- `PR_lowFPR_ROC.{png,svg,pdf}`：完整PR和正常FPR0–5% ROC双panel，保留原CAL标记。图例AUC为完整ROC面积，不是该局部区间面积。
- `long_event_coverage_denominators.{png,svg,pdf}`：九事件C3 steady热图及每事件分母/pooled权重。
- `macro_vs_pooled.{png,svg,pdf}`：同九事件的宏/合并覆盖对比。
- `operating_points.csv`、`long_events_all_streams.csv`、`macro_pooled_summary.csv`：完整六模型阈值及点工作点、108条模型×stream×长事件记录及分母、宏/合并统计。
- `CURVE_ARRAYS.npz`及每模型`*_PR.csv`/`*_ROC.csv`：无抽样完整曲线与曲线阈值，仅作解释性存档。ROC使用`drop_intermediate=False`；PR终点无对应阈值，CSV记NaN。曲线阈值不是新操作点。

图已视觉检查，数学与聚合从缓存分数重算后逐项对上原记录；事件匹配F1/延迟没有另行重算。本文只报告单seed发展数据的解释事实，不构成独立因果或泛化确认。

## Public artifact storage

The complete12 PR/ROC CSVs and34MB compressed curve arrays remain immutable in the external result directory `results/left_replay_exposure_repeat_curves_20261118_v1`; they are not discarded or downsampled. The repository contains the figures, full nine-event/operating-point summaries, input identities and a reproduction script. The original builder and its output hashes are retained externally. The public counterpart only adds explicit workspace/output arguments; curve mathematics and plots are unchanged.
