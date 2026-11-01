# Stage2 固定双阈值评价（CPU 工程候选）

本模块只装配既定 U 正常头适配合同的评价，不训练、不推理、不拟合尺度或阈值。当前验收仅合成 CPU；不存在由本候选产生的真实 TEP 性能结论或标签释放许可。

资源 v3 窄修订依据预 label 补充合同（SHA `4f57994daa517278ea6456c00a447643b61ca74c79025488604e5702b66b020b`），仅由固定 profile 决定限额：SYNTHETIC 维持 300 秒，未来 TEP100 预先固定 900 秒；2 GiB 不变，未来外层 timeout 及终止宽限仍需主签。900 秒只是资源上限，不是全 400 实体实测吞吐，也不提供真实评价许可。旧 v2 冻结和回执保留，v3 记录见 `provenance/stage2_evaluation_resource_v3`。

## 版本与入口

读出 v9 的 **32** 个 Python 文件先逐字节复制至工作区 `preparations/runtime_stage2_readout_a712825a_v1`，本地/服务器副本均已校验且未执行；原合同中的 31 由独立计数勘误更正。`COPY_MANIFEST.json` 的 SHA 为 `531c33f68b8df54fc8313f188616510a944158b14e634cdc3f65b26f18c38004`。新增评价运行时有独立完整身份，不能插入该读出快照。

```text
python -m hybrid_tsad stage2-evaluate --runtime-identity
python -m hybrid_tsad stage2-evaluate --config configs/stage2-evaluation.example.json
python -m hybrid_tsad stage2-evaluate --config /external/approved-evaluation.json --execute
```

默认只计算规范操作清单及其摘要，不打开外置资产或数组。示例审批为 PENDING/零 SHA，不能执行。schema7 / `U_NORMAL_HEAD_EVALUATION` 的操作清单绑定合同、全部输入、参数、输出及分数单位，排除 approval 自指；审批须明确 `labels/normal_prefix_values/evaluate=true`，`inference/training/calibration/F=false`。生产 scope 必须为 `U_NORMAL_HEAD_EVALUATION_EXPLICIT_LABEL_RELEASE`，CPU fixture scope 不能代替它。

## 原读出门禁的必要输入

11 个输入各为 `{path, sha256}`：`seal`、`gate_acceptance`、`gate_summary`、`gate_config`、`gate_os_wait`、`gate_resource`、`readout_runtime`、`readout_copy_manifest`、`query_manifest`、`affiliation`、独立评价 `runtime`。`parameters` 仅固定 `profile` 与 `readout_runtime_root`，无模型、设备、阈值、指标或随机种子可选项。

生产 `gate_acceptance` 须为 `stage=U_NORMAL_HEAD_READOUT`、`status=PRIMARY_ACCEPTED_U_READOUT_GATE`，逐项引用上述 seal/gate_summary/gate_config/gate_os_wait/gate_resource/readout_runtime/readout_copy_manifest 及 query_manifest 原记录，并引用 `independent_review`。外部有限独审须 PASS；resource 的 `synthetic_fixture` 须为 false。这里核验实际原 gate config 的规范审批摘要、完整 seal 与 summary 身份、原运行时源码、OS wait exit0/正时长和 resource summary 绑定；**不调用当前新包的动态 schema6 gate**。真实原冻结运行时的 gate 实际执行、独审和主接受须另行完成，本候选不能生成或自签这些证据。

合成 fixture 使用独立的 `SYNTHETIC_GATE_ACCEPTANCE_ONLY` / `SYNTHETIC_REVIEW_ONLY` 标记，只验证消费接口。它不证明真实 gate 已运行、模型预测正确或生产 OS 链成立。

## 固定指标与产物

- native 27 权重 × 400 实体 = 10,800 canonical 对象，每对象只调用一次原 `evaluate_fixed_scores` 计算 AP/ROC；S/T 共用同一 file/field/index/SHA 和 rank 结果，得到 21,600 行。T 仅调用从原函数抽取的 threshold-only 数学。旧 Z late/history_mean/persistence 另有 3,600 个 source-only 行；target 单列 NA 原因。
- 逐实体混淆、事件、报警、3 分钟延迟及真实块官方 Aff 沿原核。外部 Aff 六文件按已冻结 SHA 加载，缺失即阻断，不下载、不 vendor、不造整体 Aff；跨实体 overall Aff 为 NA。漏检延迟保持 None/NOT_DETECTED，保留 eligible/detected/missed，延迟中位数只含 detected。
- 同实体先均 3 seed，再对 20 fault × 20 run 等权 macro。六个固定 AP 对比先作同 run 跨 fault 平均，再使用唯一共同 `default_rng(99174)` 的 2,000 × 20 run 抽样矩阵。保存原矩阵、逐 run 差、所有 seed/fault/run/最差项与失败；不采用 pooled AP 重采样，不将 seed 当独立数据集。区间仅探索敏感性，无多重检验确证。
- 正常前缀 feature/score/binary 分开核重复性。逐实体原计数保留；唯一 run 统计仅在 feature 与 binary 都相同且计数一致时成立，计数与时间同分母。否则唯一统计 NA，不选代表、不 union、不平均补救。

输出包含 `input_registry.json`、`coordinates.json`、规范操作清单、全部逐实体/事件/报警/Aff/seed均值/故障/run/最差/失败 CSV、正常前缀与延迟表、source-only 控制的 target NA 表、`paired_run_bootstrap.npz`、summary 和全输出 SHA 清单。输出目录必须全新，异常留下 `evaluation_failure.json`。标签和数值数组第一次读取发生在批准、原冻结门禁、完整对象注册及外部 Aff 可用性校验之后。

## CPU 证据与边界

7 项合成回归覆盖完整配置→CSV/NPZ 正路、一次 rank 调用、旧阈值核差分（相等阈值/无报警/预报警/多块/不完整事件/官方 undefined）、审批/门禁/OS/运行时/对象缺失及 stage/init/S-T 错配、手算共同 bootstrap、正常分母和漏检保留。使用现有外置官方 Aff。实际运行 3.057 秒、峰值 RSS 150,601,728 bytes，2 GiB 地址空间上限，未导入 Torch、CUDA 为空。

来源与冻结记录见 [provenance/stage2_evaluation_cpu](provenance/stage2_evaluation_cpu)。v1 测试的 ndarray JSON 序列化错误及原 ZIP 保留，v2 修复仅测试序列化并补齐拒绝测试；不抹除失败。旧 31 个非 CLI Python 文件保持原字节；CLI 仅新增独立评价路由。未执行真实查询、标签/分数读取、GPU、模型前向、训练或校准。当前合成规模为 2 fault × 2 run，不冒称全 400 实体的实耗或真实链验证；实际运行仍须独审、主 release 与资源限制。
