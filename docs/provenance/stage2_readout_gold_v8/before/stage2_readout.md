# Schema6：U 原生读出与 Z 正常阈值适配候选

合同 `HYBRID_TEP100_U_READOUT_20261101`，stage=`U_NORMAL_HEAD_READOUT`。
本轮仅合成 CPU 工程；没有真实 catalog、Z 正常校准、query-L0、查询推理、标签或 GPU 执行。
18项训练/cal全部接受之前不能冻结生产 catalog。完整新维度评价属于后续子交付，当前没有 label loader。

## 入口

```text
python -m hybrid_tsad stage2-readout --runtime-identity
python -m hybrid_tsad stage2-readout --config configs/stage2-readout.catalog.example.json
python -m hybrid_tsad stage2-readout --config configs/stage2-readout.z-normal-cal.example.json
python -m hybrid_tsad stage2-readout --config configs/stage2-readout.infer.example.json
python -m hybrid_tsad stage2-readout --config configs/stage2-readout.seal.example.json
python -m hybrid_tsad stage2-readout --config configs/stage2-readout.gate.example.json
```

默认 preview 不打开输入资产、数据或模型。示例为真实TEP形状、外置路径与零 SHA 占位符，不是批准。
`--execute` 要求固定合同 SHA、精确规范 operation manifest、独立主审批与当前全源码 runtime 身份。
SYNTHETIC 必须 `SYNTHETIC_CPU_ENGINEERING_ONLY` 且CPU；真实TEP必须另签
`U_NORMAL_HEAD_READOUT_EXPLICIT_OPERATION`，推理/正常cal只允许固定A40 CUDA float32，无fallback。
权限字段分别声明 operation、training=false、source_fit=false、query_infer、target_normal_cal、labels=false、F=false。
`gate` 仅返回可申请主 label release 的资格，`labels_permission`仍为false；不执行评价。

## 源码与身份边界

原 core、bank/score、Z schema4、U schema5和运行中的9579090训练快照不改。
两个新增模块只组装既有数值核；单CLI路由。新的读出 runtime 全Python集合包含新模块/CLI，
与拟合U权重的旧training runtime是两份不同身份，不能把新代码倒填成旧训练执行来源。
每操作复制包内原字节源码一次并在结束复核，沿用既有 `_snapshot/_finish`，不复制官方源码或权重。

`catalog` 输入 definition 的字段：

|字段|约束|
|---|---|
|schema_version/stage/profile|1 / U_NORMAL_HEAD_READOUT / 固定TEP100，或仅CPU合成profile|
|official|外置固定官方单文件 path/SHA，不vendor|
|query_manifest|原400实体manifest path/SHA，H100/P5、52通道、960点、172 origins、原860 indices|
|source_references/z_catalog/z_seal|九个原Z parent、source单位、旧已封存3600银行/7200六方法score，只读引用|
|normal_manifest/normal_plans/source_traces|已固定正常角色、三seed正常plan、原source-cal trace，不重抽|
|training_runtime/training_runtime_root|拟合U所用旧runtime身份文件及原字节源码根，不是当前读出runtime|
|u_models|精确18个具名模型，字段见下文|
|all18_acceptance|主实际汇总接受文件，不能预生成PASS|
|physical_bridge|主核旧Z物理来源后签的真实reference桥，缺失则HOLD|

每U row为 job_id、stage=U_NORMAL_HEAD、seed/mode/initialization、step=1000，及以下 path/SHA：
checkpoint、checkpoint_metadata、training_summary、train_archive、cal_archive、source_calibration、target_threshold、primary_acceptance。
共14字段占位例为 `configs/stage2-readout.u-model-row.PENDING.example.json`；
汇总占位例为 `configs/stage2-readout.all18-acceptance.PENDING.example.json`，均非真实接受。
主 `PRIMARY_ACCEPTED_ALL18_U_TRAIN_CAL` 文件须 stage=U_NORMAL_HEAD，
`models[job_id]` 等于该完整row的规范JSON SHA（含逐波接受、两归档引用）。
归档所有列出的文件逐SHA复核；只允许 `step1000.pt`，拒绝best替代、部分模型、错误parent/role/seed/plan。
primary_acceptance只接受实际 `PRIMARY_ACCEPTED_FORMAL_PAIR` 或 `PRIMARY_ACCEPTED_NAMED_WAVE`；
后者无需补造allowed_jobs，按其accepted_train解引用训练接受：allowed_jobs含该job、summaries[job]精确匹配training_summary。
同时核independent_cal_review的实际 `PASS_LIMITED_FORMAL_CAL_PAIR` 或对应 `PASS_LIMITED_WAVE{wave}_CAL`。
单独训练接受不够，不能为适配新loader改旧接受原件。
calibration_tracks原核逐点闭合U正常cal物理bank与单canonical，继承既有×2反例拒绝。
random路径不声称加载parent：source域/exclusion为空、initial_checkpoint=null、parent仅为配对参考。

## 实际数据通路

- `infer` 使用精确 `job_id/entity/origins` 白名单；key为 `U_NORMAL_HEAD_READOUT/job_id/entity`，两初始化不撞键。
  全shard包含完整entity origins；L0恰为seed0六模型、首entity前2origin，不能当全组。
  创建/严格加载eval模型后原 `collect_forecasts` 每entity84551一次，不外套第二层seed。
  原前2块device RNG重放、逐block RNG、预测/target/scale/residual/raw/canonical/index/valid完全保留。
  读出全state（含head和buffers）前后hash，计时含原定义的forward样本与whole entity写出成本。
- `z-normal-cal` 只读九个原Z权重与原source scalar。用原 `evaluate_target_cal` / `TargetNormal`，
  同seed原256 B1C52正常cal顺序、84551每call一次并恢复RNG/模式；不走要求origin有序的query collector。
  无optimizer/source拟合/query forward。全参数requires_grad=false，全部state不变。
  输出独立 `Z_TARGET_NORMAL_THRESHOLD_ONLY`，不伪造U step1000或全流程未接触目标的Z。
  原source scalar和source q99不改，只拟合同单位target q99；保存物理cal bank、单canonical、run/origin和backend/runtime/operation身份。
  消费者还核实际正常缓存按plan的target值、原score_block逐点分数及q99，拒绝×2、错plan、错角色/源单位。
  阈值引用原operation config的path/SHA，消费者重核规范digest、精确主审批、catalog及readout runtime；不只信operation SHA自述。
- `seal` 必须18×400=7200 U bank；合成测试用完整18×2。要求全部归档summary匹配，
  每bank/hash/RNG/物理source/index/scale/原raw与source-canonical逐点闭合，缺/多/重复/L0不可封存。
  S/T两个threshold descriptor引用同一bank的 `standardized_score` 与 `time_index`，不生成第二份T分数。
  旧Z3600银行/7200分数逐SHA引用，不推理、不重写；9个Z正常阈值缺任何一个都拒绝。
  旧Z simple/late仅保留原source证据，本入口不增加其target轨、late-U或新baseline。

## 旧Z物理reference桥和执行证据

桥文件 `status=PRIMARY_ACCEPTED_Z_PHYSICAL_REFERENCE_BRIDGE`，绑定profile、z_seal、query_manifest。
`entities[entity]` 必须原seal中固定seed0/temporal的bank/rng/metadata descriptor，不能凭空声称另有缓存。
TEP还必须绑定原 `execution_evidence`、原固定CUDA backend、`independent_physical_audit` path/SHA。
消费者核完整旧Z集合，实际query文件SHA、goldbank的真实target/index/origin/scale，以及其checkpoint/来源metadata；
新U对该原preprocess参考使用原2e-6/2e-6，不换CPU公式或增容差。缺桥/资产不齐为HOLD，不执行补reference模型。
本轮未打开真实old Z资产；生产桥是否齐全必须由主独立核实后另签，测试fixture不算证明。

生产 `executions` 为每个原shard文件SHA到launch/os_wait/resource三个path/SHA的映射。
要求OS exit0、真实process_seconds、launch精确operation digest、resource完成状态及同内容summary SHA，
以及实际A40/Torch2.8.0+cu128/CUDA12.8环境。SYNTHETIC只能使用明确CPU工程标记。
resource.current_runtime.shared_sha256必须是本冻结stage2_readout模块SHA；shard和每bank另绑定完整runtime及operation digest。
真实新资源壳/启动、GPU L0、profiling和总账本仍须另签并独审，本候选不会借旧train壳授权新query。

## 验证与未迁移

冻结diff、出处/原字节SHA、CPU回执、初期失败留档见 `provenance/stage2_readout_cpu/FREEZE.json`。
完整配置正路用合成source/normal/query值和元数据fixture，未进行真实训练，也不把fixture汇总接受称主真实批准。
另以外置官方文件构造既有真实三mode，用合成B1H100C52、两origin，直接比较旧collector与新wrapper的全部数组/逐块RNG。
这仅证明工程传输等价，非检测增益、GPU等价或新科研结果。

未实现：新stage评价/标签加载、新维度配对统计适配、真实production catalog与bridge、真实Z正常cal、
query GPU L0/profiling/7200全推、实际readout资源账本。6GiB/8GPUh只是合同候选预算，不是自动执行许可。
下一步必须独审/主接受本CPU候选；全部18 train/cal接受后，再依合同逐项另签。
