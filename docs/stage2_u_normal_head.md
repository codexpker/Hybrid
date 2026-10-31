# Stage2：已知正常数据的现有预测头适配

本入口实现 `U_NORMAL_HEAD` 合同，不代表完整 U、混合污染鲁棒性或 F。
既有 Z/source 路径及其排除 TEP 的守卫保持原样；U 使用独立 schema5 和身份。
本轮只验收合成 CPU 工程行为，真实 GPU L0、目标训练及校准均未执行。

## 配置与入口

```text
python -m hybrid_tsad stage2 --runtime-identity
python -m hybrid_tsad stage2 --config configs/stage2-l0.example.json
python -m hybrid_tsad stage2 --config configs/stage2-train.example.json
python -m hybrid_tsad stage2 --config configs/stage2-calibrate.example.json
```

默认只输出操作清单，不打开数据或权重。示例中的零 SHA 是待装配占位符；
`--execute` 仍需逐文件 SHA、合同、配方、具名 job、运行源码身份与精确操作审批。
审批清单包含操作、输入、参数、输出和分数空间，不包含 approval 自指。
`--runtime-identity` 输出包内 Python 文件原字节 SHA，不代替运行环境/硬件回执。
静态 18 job 矩阵及 PENDING L0 需求均不构成执行授权。

正式训练固定 1000×64、warmup100/cosine、lr3e-4、wd.01、clip1，无任意步数开关。
只允许现有 `head.weight/head.bias` 共645参数更新，其他参数及 buffer 冻结，
backbone 保持 eval、head 保持 train。直接调用 `model(history)` 保留 head 梯度；
未来目标只进入损失。TF 辅助损失的实际无 head 梯度路径单列记录，不伪造连接。
预训练路加载整套配对 parent（含原 head）；随机路完整随机初始化且不加载 parent。
主 checkpoint 固定 step1000，正常 select 的 best 仅为补充产物。

## 数据与随机边界

只允许冻结正常 Training 缓存：fit run1..20/51..60、select61..80、cal81..100。
manifest、原始缓存身份、来源链和三 seed trace 均核对；不提供 Testing/query/label 访问器。
输入历史 H100、预测 P5；fit/select 为 C1，target-cal 为 B1C52。
每次 select/source-cal/target-cal 调用开始仅设置84551一次，严格原 plan 顺序，
结束恢复训练 RNG 及全部 module 模式。target-cal 是校准池，不能解释为连续事件；
seed0 前两项 run95/origin465、100 不得排序。

适配后重新用原 source-cal trace 计算 source 尺度和 q99。
每个输入池/模型只保存一套 source-canonical float32 分数；target 仅在同单位拟合 q99，
不另拟合目标全局尺度。S/T 阈值轨引用同一对象，当前入口不生成 query 分数或 AP/ROC。
U metadata 区分预训练排除目标、适配目标、是否实际加载 parent，绑定拟合 checkpoint、
manifest、trace、配方、运行源码和操作清单。U 权重不能作为旧 Z parent，target 阈值不能伪装为 source-cal。
后续校准只接受完整正式训练归档及其主 checkpoint，拒绝失败、缺件或身份变动的归档。

## 独立 L0 丢弃入口

`stage2-l0` 仅接受原需求中的 hybrid/seed0 预训练与随机 pair。
固定5×64更新，仍使用1000/W100调度步1..5；正常 select64 C1、source-cal64 C1、target-cal2 C52。
source 前64项全为 PSM，只验证该前缀，不冒称覆盖 SMD。
使用正式路径共享的 head 一步、数据权限、评估与随机核；模型/优化器状态丢弃，
不写 checkpoint、可复用 calibration 或正式阈值，仅写 `L0_ONLY` 诊断数组与回执。
记录梯度、参数更新、冻结哈希、模式、未来扰动及 RNG 恢复；120秒操作体限制，
CUDA峰值20GiB拒绝检查。进程启动/退出和完整外部资源计量仍由执行壳提供。

真实 L0 需 `U_NORMAL_HEAD_L0_DISCARD_ONLY` 精确审批；合成审批只允许 CPU。
正式 train 还要求主接受的独立 L0 gate，绑定 pair 两份报告、独审、输入与运行身份。
报告自述 `GPU_executed` 本身不能授权。首 pair 以外16 job 还需正式 pair 主验收与新的具名审批。

## 验证范围与未迁移项

本轮 CPU 回执与逐文件 SHA 见 [交付索引](provenance/stage2_u_normal_head_validation.json)。
合成完整1000步使用645头的轻量 fixture；另直接以外置官方源构造现有 temporal/tf/hybrid
核心，各做5×64 CPU更新及源/目标形状通路检查。未读取真实目标值或标签，未运行 GPU。
测试中的固定数据 SHA 替换只存在于 fixture，不能成为真实审批资产。

尚缺真实 GPU L0 的耗时/峰值/设备/退出回执和独审，以及另签的正式 pair。
其余16 job、U query 读出、F、全参数适配、adapter 和恢复中断训练不属于本入口本轮交付。
不能把本轮 head-only 工程有效性写成检测收益、混合污染鲁棒性或完整两阶段方法结论。
